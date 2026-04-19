# System Design — Client-Server 아키텍처

[ALGORITHMS](./ALGORITHMS.md) 에서 선택한 Client-Server 변형을 어떻게 구현했는지, 컴포넌트 / 메시지 / 상태 관점에서 정리한다. 각 설계 결정은 [SCENARIOS](./SCENARIOS.md) 의 특정 시나리오/요구에서 도출됐다.

---

## 전체 구성

```
┌──────────────────────────────── Client Process ───────────────────────────────┐
│                                                                               │
│  ┌──────────────────┐   propose()   ┌────────────────────┐                    │
│  │  DraftProposer   │  ───────────▶ │    DraftClient     │                    │
│  │  • N-gram        │               │  (ZMQ DEALER)      │                    │
│  │  • Suffix        │               │                    │                    │
│  │  • EAGLE         │               │  + Adaptive K      │                    │
│  └──────────────────┘               │  + Fault FSM       │                    │
│                                     └─────────┬──────────┘                    │
│                                               │                               │
└───────────────────────────────────────────────┼───────────────────────────────┘
                                                │  DraftRequest  (msgpack)
                                                │  ◀ VerifyResponse
                                                │
┌───────────────────────────────────────────────┼───────────────────────────────┐
│                                  Server Process                               │
│                                               │                               │
│                                     ┌─────────▼──────────┐                    │
│                                     │    TargetServer    │                    │
│                                     │  (ZMQ ROUTER)      │                    │
│                                     │                    │                    │
│                                     │  active_requests[] │                    │
│                                     └─────────┬──────────┘                    │
│                                     verify()  │                               │
│                                     ┌─────────▼──────────┐                    │
│                                     │  TargetVerifier    │                    │
│                                     │  + RejectionSampler│                    │
│                                     │  + KV truncate     │                    │
│                                     └────────────────────┘                    │
└───────────────────────────────────────────────────────────────────────────────┘
```

- **Client** = 경량 연산 (draft 생성) + 네트워크 I/O + 상태 기계.
- **Server** = 무거운 target forward + rejection sampling + per-request 상태 캐시.
- 둘 사이에는 ZMQ DEALER ↔ ROUTER 소켓만 놓여 있고, 메시지는 모두 msgpack 으로 직렬화.

---

## 메시지 프로토콜

### 정의 위치

모든 메시지는 `prototype/common/protocol.py` 의 `@dataclass` 로 정의되어 있고, `MsgpackEncoder/Decoder` 로 양방향 변환된다.

| 메시지 | 방향 | 주요 필드 |
|---|---|---|
| `DraftRequest` | C → S | `request_id`, `prompt_tokens` (첫 요청만), `draft_tokens`, `draft_probs` (EAGLE), `sampling_params`, `kv_cache_info` |
| `VerifyResponse` | S → C | `request_id`, `accepted_tokens`, `num_accepted`, `bonus_token`, `hidden_states`, `finished` |
| `HealthCheck` / `HealthResponse` | C ↔ S | server 가용성 확인 |

### 직렬화

- **msgspec.msgpack** 우선, 없으면 JSON fallback.
- Tensor/ndarray 는 `{_tensor: True, data: bytes, dtype, shape}` 로 내려가고 수신측에서 `np.frombuffer` → `torch.from_numpy` 복원.
- 제네릭 dataclass 는 `_type: ClassName` 태그로 올라가 `MsgpackDecoder.TYPE_MAP` 에서 재구성.

### 한 스텝 시퀀스

```mermaid
sequenceDiagram
    participant P as DraftProposer
    participant C as DraftClient
    participant S as TargetServer
    participant V as TargetVerifier

    Note over C: adaptive K 결정
    C->>P: propose(context, K)
    P-->>C: DraftOutput (draft_tokens, draft_probs?)
    C->>S: DraftRequest (msgpack)
    S->>S: active_requests[state_key] 갱신
    S->>V: verify(draft, context, sampling_params)
    V->>V: target forward (context+draft)
    V->>V: RejectionSampler
    V->>V: _truncate_kv_cache(거절된 토큰분)
    V-->>S: VerifyOutput
    S-->>C: VerifyResponse
    C->>C: adaptive.record_result(rtt, α)
    C-->>C: yield decode(accepted + bonus)
```

---

## ZMQ 토폴로지

### 왜 ZMQ 인가

- **비동기 메시지 큐 내장**: `asyncio` 와 통합 (`zmq.asyncio`).
- **Multi-client 자동 라우팅**: ROUTER 소켓이 각 peer 의 identity 를 유지해 응답을 정확히 되돌려준다. 직접 구현하면 상당량의 상태 관리가 필요.
- **Frame 기반**: `recv_multipart()` 로 `[identity, empty, payload]` 구조를 자연스럽게 표현.

### DEALER ↔ ROUTER 패턴

| 역할 | 소켓 타입 | 특성 |
|---|---|---|
| Client | `DEALER` | identity 를 자동 부여받음 (또는 명시), 라운드 로빈으로 메시지 송신 |
| Server | `ROUTER` | 수신한 peer identity 를 보존해 응답 시 사용 |

- **Scenario 2 (다중 클라이언트) 해결**: 서버는 `frames[0]` 을 identity 로 써서 `active_requests[f"{identity}:{request_id}"]` 키로 상태를 격리 유지. 별도의 세션 테이블 관리가 불필요.

### 타임아웃 / 재시도

- `RCVTIMEO` / `SNDTIMEO` 를 `config.timeout * 1000` (ms) 로 설정.
- 타임아웃 시 `zmq.error.Again` 예외 → `DraftClient.generate()` 에서 재시도, 연속 실패는 [ADAPTIVE_CONTROL](./ADAPTIVE_CONTROL.md) 의 FSM 으로 전파.

---

## 상태 관리

### Client 측 상태

```python
DraftClient:
    draft_proposer        # 상태 있음 (Suffix tree) 또는 없음 (N-gram)
    adaptive_controller   # RTT / α 히스토리 (deque)
    _request_count        # request_id 생성용
```

Proposer 와 controller 는 별개로 리셋 가능 (`reset()`). 요청이 끝나도 controller 히스토리는 유지되어 다음 요청에 이어서 사용된다 — **RTT/α 특성은 네트워크 환경에 묶여 있지 한 요청에 묶이지 않는다**.

### Server 측 상태

```python
TargetServer:
    active_requests: Dict[str, RequestState]  # key = f"{client_id}:{request_id}"
    verifier: TargetVerifier                  # 모델 + KV cache
    metrics: ServerMetrics
```

**`RequestState`**: 각 요청의 `prompt_tokens + generated_tokens` 를 누적. 클라이언트는 첫 요청에만 prompt 를 싣고, 이후 요청은 `draft_tokens` 만 전송 → 네트워크 트래픽 절약.

**KV cache**: `TargetVerifier` 가 하나의 `past_key_values` 를 유지하며, rejection 발생 시 `_truncate_kv_cache(keep_len)` 로 거절 토큰 위치의 K/V 를 잘라낸다.

### 암묵적 전제

- **Sticky routing**: 같은 요청은 같은 서버 인스턴스에 붙어야 한다. Phase 1 은 단일 서버 가정.
- **단일 유저 대화 단위**: `request_id` 는 스트림 ID. 여러 턴 대화를 동일 `request_id` 로 묶으면 KV cache 누적 혜택을 얻는다.

이 두 전제는 Phase 2 (vLLM 통합) 에서 vLLM 의 request scheduler 와 block manager 로 자연스럽게 대체된다.

---

## 디렉토리 / 파일 매핑

| 디렉토리 | 책임 | 주요 파일 |
|---|---|---|
| `prototype/common/` | 데이터 계약 (프로토콜 + 설정 + 공통 로직) | `protocol.py`, `config.py`, `confidence.py` |
| `prototype/client/` | Draft 생성 + 통신 + FSM | `draft_proposer.py`, `draft_client.py`, `fault_tolerant_client.py`, `confidence_client.py` |
| `prototype/server/` | Target 모델 로딩 + 검증 + 서빙 | `target_verifier.py`, `target_server.py` |

**핵심 엔트리 포인트**
- Server 실행: `python -m prototype.server.target_server --model ... --listen-address ...` (`target_server.py:404-441`)
- Client 사용: `FaultTolerantClient(config)` 를 async context manager 로 → `generate(prompt)` 이 async generator 로 토큰 스트리밍

---

## 설계 트레이드오프 요약

| 결정 | 이유 | 트레이드오프 |
|---|---|---|
| ZMQ DEALER/ROUTER | multi-client + async 친화, multi-node 확장 가능 | gRPC 대비 스키마 검증 약함 (msgspec 으로 보완) |
| msgpack over JSON | 이진 포맷, 수치 배열 효율적 | 가독성 낮음 (디버그 시 `jq` 류 툴 안 먹힘) |
| 서버에 context 누적 | 네트워크 절약 + KV cache 재사용 | sticky routing 필요 (Phase 1 한계) |
| Per-request KV cache | 단순하고 정확성 확실 | 동시 요청 시 GPU 메모리 증가 — batched KV 는 Phase B |
| Proposer 추상화 | N-gram/Suffix/EAGLE 런타임 교체 | 공통 인터페이스에 맞추느라 각 기법의 고유 최적화 포기 |

---

**다음 섹션들**:
- Draft 를 어떻게 만드는지: [DRAFT_METHODS](./DRAFT_METHODS.md)
- Target 검증과 분포 보존: [VERIFICATION](./VERIFICATION.md)
- Adaptive K 와 Fault-Tolerant FSM: [ADAPTIVE_CONTROL](./ADAPTIVE_CONTROL.md)
