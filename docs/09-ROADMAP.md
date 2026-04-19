# 09. Roadmap — Phase 1 완료 조건과 Phase 2 (vLLM 통합)

본 프로젝트는 두 단계로 나뉜다. **Phase 1** 은 순수 PyTorch + HF 기반의 참조 구현으로 알고리즘과 시스템 설계의 올바름을 증명하는 것이고, **Phase 2** 는 이를 vLLM 엔진 위에 얹어 production 급 성능을 확보하는 것이다.

---

## 1. Phase 1 — 참조 구현 (현재)

### 1.1 목표

- Client-Server 분산 SD 의 올바름 증명 (분포 보존 + 기능 완성)
- 세 가지 draft 전략의 교체 가능한 인터페이스 확정
- Adaptive K 와 Fault-Tolerant FSM 의 동작 검증
- 향후 vLLM 통합의 참조 사양 역할

### 1.2 완료 항목

- [x] `common/protocol.py` — 메시지 / 직렬화
- [x] `common/config.py` — 클라이언트/서버 설정
- [x] `common/confidence.py` — confidence 메트릭
- [x] `client/draft_proposer.py` — N-gram / Suffix / EAGLE 경량
- [x] `client/draft_client.py` — ZMQ DEALER + Adaptive K
- [x] `client/fault_tolerant_client.py` — 3-state FSM
- [x] `client/confidence_client.py` — BiLD / RouteLLM / SVIP 변형
- [x] `server/target_verifier.py` — Rejection Sampling (greedy + random)
- [x] `server/target_server.py` — ZMQ ROUTER + per-request state

### 1.3 Phase B 마무리 (완료)

- [x] **Batched Target Verify**: `HfVerifier.verify_batch` 를 padded forward + per-request slicing 으로 재작성. [`tests/test_batched_verify.py`](../tests/test_batched_verify.py) 에서 per-request 와 정합성 검증.
- [x] **Rejection Sampling 정확성 테스트**: χ² goodness-of-fit 로 3개 $(p, q)$ 쌍에서 target 분포 보존 경험적 증명. [`tests/test_rejection_sampling.py`](../tests/test_rejection_sampling.py).
- [x] **End-to-end 스모크 테스트**: `sshleifer/tiny-gpt2` 쌍으로 greedy SD vs target-only 토큰 동일성. Slow marker 로 CI 분리. [`tests/test_e2e.py`](../tests/test_e2e.py).
- [x] **Adaptive Controller 단위 테스트**: 9 케이스 전부 통과, α/RTT 단조성 확인. [`tests/test_adaptive.py`](../tests/test_adaptive.py).
- [x] **EAGLE hidden state 경량 concat**: `EagleDraftProposer(use_hidden_states=True)` 에서 target hidden 을 draft 입력의 마지막 position embedding 에 additive fuse. 차원 불일치 시 id-only 로 fallback.

이로써 Phase 1 마감 — v0.1 태그 준비.

### 1.4 Phase 1 의 알려진 제한

| # | 제한 | 본문 참조 |
|---|---|---|
| L1 | KV cache 를 HF `past_key_values` 튜플로 직접 슬라이싱 → 메모리 비효율, 프레임워크 의존적 | [06-VERIFICATION § 6](./06-VERIFICATION.md#6-kv-cache-관리) |
| L2 | `serialize_kv_cache` 의 delta/full 모드가 프로토콜만 정의되어 있고 client/server 파이프라인에서 "none" 만 실제 사용 | `common/protocol.py:372-425` |
| L3 | 다중 서버 수평 확장 미지원 (`active_requests` 가 in-memory dict, sticky routing 전제) | [04-DESIGN § 4.3](./04-DESIGN.md#43-암묵적-전제) |
| L4 | EAGLE 의 hidden state 주입이 draft head 학습 없이 경량 concat 수준 | [05-DRAFT_METHODS § 4.2](./05-DRAFT_METHODS.md#42-논문-eagle-과의-차이) |
| L5 | 단일 GPU target 만 지원 (TP/PP 없음) | — |
| L6 | `adaptive_controller` 의 $T_{\text{decode}}$ 가 설정값 고정 (실측 미사용) | [07 § 1.6](./07-ADAPTIVE_CONTROL.md#16-한계) |

L1/L2/L3/L5 는 Phase 2 에서 vLLM 으로 **자연스럽게 해결**되므로 Phase 1 에서 고치지 않는다. L4 는 별도 학습 프로젝트, L6 은 Phase 2 에서도 개선 필요.

### 1.5 Non-Goals (Phase 1 범위 밖)

- Production-급 throughput 최적화 (continuous batching, PagedAttention)
- Multi-GPU target (Tensor/Pipeline/Expert parallelism)
- Draft head 재학습 기반 full EAGLE / Medusa
- Tree attention / multi-candidate SD
- 실시간 multi-tenant 스케줄링

---

## 2. Phase 2 — vLLM 통합

### 2.1 목표

Phase 1 의 시스템 설계(Client-Server, FSM, Adaptive K) 는 **유지**하고, server 내부의 model execution 과 KV 관리를 vLLM 엔진으로 교체한다. 그 결과:

- Target throughput 이 production 수준으로 증가 (continuous batching)
- KV cache 메모리 효율 대폭 개선 (PagedAttention)
- Multi-GPU target 지원 (TP/PP)
- Draft 전략도 vLLM V1 의 내장 spec_decode 구현과 선택적으로 호환

### 2.2 컴포넌트 매핑

| Phase 1 | Phase 2 (vLLM 통합) | 교체 방식 |
|---|---|---|
| `TargetVerifier.model` (HF `AutoModelForCausalLM`) | `vllm.LLMEngine` / V1 `EngineCore` | `verifier` 내부 구현 교체, 외부 인터페이스 유지 |
| HF `past_key_values` tuple | vLLM `KVCacheManager` + block table | `_truncate_kv_cache` 를 block free 호출로 |
| `TargetVerifier.verify()` (blocking) | `engine.add_request()` + `engine.step()` loop | 비동기 step-driven 패턴 |
| `RejectionSampler` (자체 구현) | `vllm/v1/sample/rejection_sampler.py` | 동등 수학 구현 재사용 |
| `BatchVerifier` | vLLM scheduler (continuous batching 자동) | 커스텀 배처 폐기 |
| `client/draft_proposer.py:NgramDraftProposer` | `vllm/v1/spec_decode/ngram_proposer.py` | 선택적 — client 쪽은 경량 유지 가능 |
| `client/draft_proposer.py:EagleDraftProposer` | `vllm/v1/spec_decode/eagle.py` | draft model 을 vLLM 엔진으로 실행 |

### 2.3 유지되는 것

- `common/protocol.py` — 메시지 정의는 그대로
- `common/config.py` — `ServerConfig` 에 vLLM 파라미터 추가만
- `client/draft_client.py` — 통신 계층 유지
- `client/fault_tolerant_client.py` — FSM 유지
- `client/draft_proposer.py` — 클라이언트 측 proposer 는 그대로 쓸 수 있음

### 2.4 예상 작업 단계

1. **S1. Server wrapping** ✅ **(완료)**: `VllmVerifier` 를 `vllm.LLM` 기반으로 구현. `prompt_logprobs` 로 한 번의 forward 에서 draft 위치별 target argmax 추출, greedy rejection 수행. Batch 는 vLLM scheduler 에 위임 (`verify_batch` 가 단일 `llm.generate()` 호출). `ServerConfig.backend="vllm"` 으로 선택 가능. `tests/test_vllm_verifier_unit.py` 의 8 mock 테스트가 rejection 알고리즘 정합성 증명.
2. **S2. RejectionSampler 대체 (random 모드)**: 현재 S1 은 greedy only. Random 모드는 vLLM 의 `vllm.v1.sample.rejection_sampler` 를 호출하거나, vLLM 이 내뱉는 full-vocab logprobs + draft_probs 로 자체 sampler 를 돌리는 경로 필요.
3. **S3. Batch scheduling 위임** ✅ **(S1 과 함께 완료)**: vLLM 스케줄러가 `llm.generate(prompt_token_ids=[...])` 의 list 입력을 continuous batching 으로 처리.
4. **S4. Multi-GPU 옵션** ✅ **(인터페이스 완료)**: `ServerConfig.tensor_parallel_size` 가 `VllmVerifier(tensor_parallel_size=N)` 으로 전달됨. 성능 검증은 실 모델로 추가 필요.
5. **S5. KV sharing 검토**: draft 와 target 이 같은 서버에 있을 경우 KV 공유 가능성 탐색 (optional).
6. **S6. Multi-client 진짜 스케일링**: 여러 client 의 요청을 vLLM scheduler 가 동시에 처리하게 integration 테스트.

**Phase 2 S1/S3/S4 상태**: 인터페이스/알고리즘 구현 완료. 실 vLLM engine 을 띄우는 `tests/test_vllm_verifier.py` 는 `@pytest.mark.vllm` 로 분리되어 GPU 환경에서만 실행.

### 2.5 Phase 2 Non-Goals

- vLLM 자체에 업스트림 기여 (이 프로젝트는 **vLLM 사용자**로서 통합)
- 커스텀 attention kernel 작성
- TPU / ROCm 등 비-CUDA 백엔드 최적화

---

## 3. Phase 3 — 아이디어 단계 (미확정)

우선순위 없이 나열만 한다:

- **SpecInfer 스타일 tree-based SD**: 여러 candidate 를 트리로 묶어 한 번의 verify 에 여러 경로 검증
- **Multi-draft 합의**: 여러 proposer 의 draft 를 합쳐 투표
- **Proactive pre-drafting**: 이전 토큰 yield 중에 다음 draft 를 미리 생성하여 순차 대기 제거
- **Cross-device KV migration**: draft ↔ target 간 KV 를 실제 전송 (Phase 1 L2 완성)

---

## 4. 릴리스 전략

| 태그 | 조건 | 상태 |
|---|---|---|
| v0.1.0.dev0 | Phase 1 참조 구현 | ✅ 릴리스 (Phase B 완료 포함) |
| v0.2.0.dev0 | Phase 2 S1/S3/S4 greedy MVP | 🚧 진행 중 (이 커밋 반영) |
| v0.2 | Phase 2 S2 random mode | 예정 |
| v0.3 | Phase 2 S4~S6 완전 검증 (multi-GPU, multi-client) | 예정 |
| v1.0 | Phase 2 전체 + 문서 안정 | 예정 |

---

## 5. 기여 방향 (예시)

기여가 가능하다면 다음 영역이 유용하다:

- Phase 1 L4 (EAGLE draft head 학습) — 별도 학습 레포로 분리 가능
- Phase 2 S5 (KV sharing 실험)
- [08-EVALUATION](./08-EVALUATION.md) 의 벤치마크 스크립트 (HW 다양화)
- 다국어 프롬프트 / 코드 생성 프롬프트 수집

---

**이전**: [08-EVALUATION](./08-EVALUATION.md) · **처음**: [01-PROBLEM](./01-PROBLEM.md)
