# User Scenarios — 어떤 환경에서 쓰는가

[PROBLEM](./PROBLEM.md) 에서 정의한 문제(RTT 누적 + 원격 target 의존 + 장애 가능성)가 실제로 발생하는 세 가지 대표 시나리오를 정리한다. 이 세 시나리오가 [DESIGN](./DESIGN.md) 의 모든 설계 결정(ZMQ DEALER/ROUTER, Adaptive K, Fault-Tolerant FSM)의 출발점이 된다.

---

## Scenario 1 — 엣지 클라이언트 + 클라우드 Target

| 필드 | 내용 |
|---|---|
| **Actor** | 모바일/IoT/엣지 디바이스. CPU 혹은 저사양 NPU만 보유, 대형 target 모델(70B+)을 로컬에 띄울 수 없음 |
| **Precondition** | Draft proposer 는 로컬 실행 가능. N-gram 은 무모델, EAGLE 계열은 1B 이하 모델을 로컬에서 float16 로 띄울 수 있어야 함 |
| **Trigger** | 사용자가 긴 응답(예: 요약, 코드 생성)을 요청 |
| **Flow** | (1) 프롬프트 수신 → (2) Draft Proposer 가 K 토큰 제안 → (3) Target Server 에 `DraftRequest` 전송 → (4) Server 가 Rejection Sampling → (5) Client 는 수락 토큰 + bonus 를 streaming 으로 yield → (6) 종료 조건(EOS/max_tokens)까지 반복 |
| **Postcondition** | 출력 분포가 target 모델 단독 실행과 수학적으로 동일 (Rejection Sampling 보장) |
| **Exception** | Server 타임아웃 → Scenario 3 으로 전이 |

**핵심 기능 요구**
- SD 의 분포 보존성(lossless) 을 유지해야 한다. 품질 저하 없이 오직 latency 만 개선.
- 하나의 proposer 인터페이스에서 N-gram/Suffix/EAGLE 을 런타임 교체 가능해야 한다 (디바이스 성능에 따라 선택).

**비기능 요구**
- **P99 첫 토큰 latency < 500ms** (RTT 100ms 환경). 이는 SD 한 스텝에 draft 생성(~50ms) + 왕복(100ms) + verify(~50ms) + 버퍼 여유를 포함한 값.
- **RTT 상각 효과**: 동일 조건에서 naive 대비 **3× 이상** end-to-end throughput 향상 (α=0.7, K=5 기준, [PROBLEM](./PROBLEM.md) § 정량적 Impact 참조).

→ 이 요구를 만족시키기 위해 [ADAPTIVE_CONTROL](./ADAPTIVE_CONTROL.md) 의 Adaptive K 가 필요하다.

---

## Scenario 2 — 다중 클라이언트 공유 Target

| 필드 | 내용 |
|---|---|
| **Actor** | 한 팀/조직의 여러 개발자/앱이 하나의 고성능 target GPU 를 공유 |
| **Precondition** | Target Server 는 단일 고성능 GPU(예: A100/H100) 에서 구동, 다수 클라이언트가 동시 접속 |
| **Trigger** | 여러 클라이언트가 동시에 생성 요청 |
| **Flow** | 각 클라이언트는 서로 다른 ZMQ identity 로 접속. Server 는 `active_requests[client_id:request_id]` 로 요청별 상태(`RequestState`)를 격리 유지. 클라이언트는 첫 요청에만 prompt_tokens 를 싣고 이후에는 draft 만 전송 → 네트워크 트래픽 절약. |
| **Postcondition** | 각 클라이언트의 요청이 서로 간섭 없이 병렬 진행 |
| **Exception** | Server 메모리/KV 캐시 초과 → 요청 거절 또는 지연 |

**핵심 기능 요구**
- **요청 상태 격리**: 한 클라이언트의 context 가 다른 클라이언트에 유출되지 않는다.
- **Context 누적 서버 보관**: 클라이언트가 매 요청마다 전체 context 를 재전송하지 않아도 서버가 이어서 검증할 수 있어야 한다.

**비기능 요구**
- **동시 100 요청에서 target GPU 활용률 > 80%**. 단일 요청 처리 중에도 다른 요청이 큐잉되어 GPU idle time 을 최소화해야 한다.
- **Verify batch 처리**: 여러 클라이언트의 draft 검증을 한 번의 target forward 로 묶어 throughput 을 확보.
  - Phase 1 `HfVerifier`: `verify_batch()` 가 직접 패딩 + attention_mask 로 배치.
  - Phase 2 `VllmVerifier`: vLLM 스케줄러에 여러 요청을 `llm.generate(prompts=[...])` 로 한꺼번에 넘기면 continuous batching + PagedAttention 이 자동 적용됨. GPU 활용률 요구를 backend 가 흡수.

→ 이 요구를 만족시키기 위해 [DESIGN](./DESIGN.md) 의 ZMQ ROUTER 패턴 + `RequestState` 관리, 그리고 [VERIFICATION § Backend 별 구현 차이](./VERIFICATION.md#backend-별-구현-차이) 의 backend 선택이 필요하다.

---

## Scenario 3 — 네트워크 불안정 환경

| 필드 | 내용 |
|---|---|
| **Actor** | 모바일 네트워크 / 공용 Wi-Fi / VPN / 테더링 환경의 클라이언트 |
| **Precondition** | 서버는 정상 운영 중이지만 네트워크 경로가 불안정 (간헐적 패킷 손실, 긴 꼬리 RTT, 일시적 단절) |
| **Trigger** | Draft 요청에 대한 응답 timeout 이 발생하거나 수락률(α)이 급락 |
| **Flow** | (1) SPECULATIVE 모드: 정상 SD 수행 → (2) 최근 10회 평균 α < 0.2 → DEGRADED (K=min 으로 강제) → (3) 연속 실패 ≥ `max_retries` → FALLBACK (Draft 모델로만 로컬 생성, 품질 저하하지만 가용성 유지) → (4) 주기적 `HealthCheck` 로 서버 회복 확인 → SPECULATIVE 복귀 |
| **Postcondition** | 네트워크 이상과 무관하게 생성은 중단되지 않는다 |
| **Exception** | Fallback 중에도 draft 모델이 실패하면 최종 에러 반환 |

**핵심 기능 요구**
- 네트워크 장애가 사용자에게 생성 실패로 직접 노출되지 않아야 한다 (graceful degradation).
- Server 가 복구되면 자동으로 SD 모드로 복귀.

**비기능 요구**
- **3회 연속 타임아웃 내 fallback 전환** (사용자 대기 시간 한계).
- **복구 탐지는 30초 주기**로 충분 (서버 장애는 보통 수 분 단위).

→ 이 요구를 만족시키기 위해 [ADAPTIVE_CONTROL](./ADAPTIVE_CONTROL.md) 의 3-state FSM(SPECULATIVE / DEGRADED / FALLBACK) 이 필요하다.

---

## 시나리오 → 설계 요구 정리

| 시나리오 | 핵심 요구사항 | 해결 메커니즘 | 관련 문서 |
|---|---|---|---|
| 1. 엣지-클라우드 | 분포 보존 + RTT 상각 | Rejection Sampling + Adaptive K | [VERIFICATION](./VERIFICATION.md), [07](./ADAPTIVE_CONTROL.md) |
| 2. 다중 클라이언트 | 요청 격리 + 서버 상태 누적 | ROUTER identity + `RequestState` | [DESIGN](./DESIGN.md) |
| 3. 네트워크 불안정 | 장애 내성 + 자동 복구 | 3-state FSM + HealthCheck | [ADAPTIVE_CONTROL](./ADAPTIVE_CONTROL.md) |

---

**다음 섹션**: 이 시나리오들을 풀기 위해 어떤 SD 알고리즘과 분산 아키텍처를 선택해야 하는지 비교한다 → [ALGORITHMS](./ALGORITHMS.md)
