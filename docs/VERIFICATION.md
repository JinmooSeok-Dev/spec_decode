# Verification — Rejection Sampling 과 Confidence 기반 최적화

[DRAFT_METHODS](./DRAFT_METHODS.md) 에서 생성된 draft 토큰을 target 모델에서 어떻게 검증하는지, 그리고 일부 토큰에 대해 검증 자체를 **건너뛸** 수 있는 confidence 기반 최적화를 정리한다.

---

## 기본 흐름

```
Client: DraftRequest(draft_tokens, draft_probs?, context_info)
   ↓
Server.verify():
    1. input_ids = context_tokens + draft_tokens
    2. target forward → logits [L+K+1, vocab]
    3. RejectionSampler.forward(target_logits, draft_tokens, draft_probs, params)
    4. KV cache 정리 (거절된 토큰 K/V 제거)
    5. VerifyOutput(accepted, bonus, hidden, logprobs)
```

구현 위치: `prototype/server/target_verifier.py:362-432`

---

## Rejection Sampling — Greedy 모드

$\text{temperature}=0$ 일 때는 argmax 비교만으로 충분하다.

```python
for i, draft_token in enumerate(draft_tokens):
    target_token = target_logits[i].argmax()
    if target_token == draft_token:
        accepted.append(draft_token)
    else:
        bonus = target_token  # 불일치 위치의 target argmax 를 bonus 로
        break

# 모두 수락되면 다음 위치의 argmax 를 bonus 로
if len(accepted) == len(draft_tokens):
    bonus = target_logits[len(draft_tokens)].argmax()
```

- **수락률**: draft 와 target 이 같은 argmax 를 내는 비율. N-gram 패턴이 정확히 맞으면 매우 높고, 일반 텍스트에서는 모델 품질차에 좌우.
- **분포 보존**: greedy 는 결정론적이므로 자명하게 target 와 동일 결과.

구현: `target_verifier.py:114-145`

---

## Rejection Sampling — Random 모드 (수학적 핵심)

$\text{temperature} > 0$ 일 때 draft 토큰 $\tilde{x}$ 에 대해:

$$
A = \min\left(1, \frac{p(\tilde{x})}{q(\tilde{x})}\right)
$$

$u \sim U(0,1)$ 이 $u < A$ 이면 수락, 아니면 거절 후 recovered 분포에서 bonus 샘플링:

$$
p_{\text{rec}}(x) = \frac{\max(0, p(x) - q(x))}{\sum_{y} \max(0, p(y) - q(y))}
$$

### 왜 이 공식인가 — 분포 보존 증명

draft 분포 $q$, target 분포 $p$, 전체 수락 확률 $A_{\text{total}} = \sum_x \min(q(x), p(x))$. 한 토큰 위치에서 최종 출력 토큰의 분포는:

$$
\begin{aligned}
\Pr[X = x] &= \underbrace{q(x) \cdot \min(1, p(x)/q(x))}_{\text{draft 에서 수락}} + \underbrace{(1 - A_{\text{total}}) \cdot p_{\text{rec}}(x)}_{\text{거절 후 recovered}} \\
&= \min(q(x), p(x)) + \max(0, p(x) - q(x)) \\
&= p(x)
\end{aligned}
$$

즉 **수락된 토큰과 recovered 토큰을 합친 출력 분포는 정확히 target 분포와 같다**. 이것이 SD 가 quality-preserving 한 이유다 ([Chen et al., 2023](https://arxiv.org/abs/2302.01318) Theorem 1).

### Draft 확률이 없을 때 (N-gram)

N-gram 은 확률 분포 없이 토큰 ID 만 내므로, verifier 는 $q(x) = 1/V$ (uniform) 을 가정한다 (`target_verifier.py:172-175`).

이 경우 수락 확률은 $\min(1, V \cdot p(\tilde{x}))$ 가 되어 **$p(\tilde{x})$ 가 uniform 보다 충분히 크면 거의 항상 수락**. 실제로 N-gram 이 고르게 맞추는 "반복 패턴" 환경에서는 $p(\tilde{x})$ 가 매우 높아 이 근사가 잘 작동한다.

### Top-k / Top-p 필터링

Sampling param 의 `top_k`, `top_p` 는 **target logits 에만** 적용 (`_apply_sampling_filters`, `target_verifier.py:238-266`). 이렇게 해야 정의 그대로의 target 분포에서 샘플하는 것과 동치.

### 구현

`target_verifier.py:147-213` (`_random_verify`), `:215-236` (`_sample_from_recovered`).

---

## Confidence 기반 최적화

Rejection sampling 을 **항상** 수행하는 대신, draft 쪽의 확신도가 매우 높은 토큰은 서버 검증을 건너뛸 수 있다. 이는 품질 보존과 속도 향상의 트레이드오프다.

### 세 가지 접근법

| 방식 | 스코프 | 전형적 논문 | 효과 |
|---|---|---|---|
| Token-level Skip | 각 토큰 | **BiLD** ([Kim et al., 2023](https://arxiv.org/abs/2302.07863)) | confidence 높은 prefix 는 바로 yield, 나머지만 서버로 |
| Query-level Routing | 요청 단위 | **RouteLLM** ([Ong et al., 2024](https://arxiv.org/abs/2406.18665)) | 쉬운 질의는 draft 단독, 어려운 질의만 target |
| Adaptive Window | draft 윈도우 크기 | **SVIP** ([Liu et al., 2024](https://arxiv.org/abs/2407.06677)) | draft 가 불확실해지면 더 이상 뽑지 않고 조기 종료 |

### Confidence 메트릭

`common/confidence.py` 에서 3 종 지원:

| 메트릭 | 정의 | 특징 |
|---|---|---|
| `entropy` | $H(P) = -\sum p_i \log p_i$ | 분포 전체를 반영, 계산 비쌈 |
| `max_prob` | $\max(P)$ | 단순, 빠름 |
| `logit_margin` | $\text{logit}_1 - \text{logit}_2$ | 1, 2위 격차로 직관적 |

### 트레이드오프

- 높은 `skip_threshold` → 속도 ↑ 품질 ↓ (잘못된 draft 를 통과시킬 위험).
- 낮은 threshold → 속도 ↓ 품질 ↑ (모든 토큰을 검증).
- Phase 1 기본값: `skip_threshold = 0.8`, `warmup_steps = 5` (초반 몇 스텝은 무조건 검증).

### 구현

- `common/confidence.py` — 공통 계산기와 분류기
- `client/confidence_client.py` — `ConfidenceSkipClient`, `QueryRoutingClient`, `AdaptiveWindowClient`

**주의**: Confidence skip 은 엄밀한 분포 보존을 **깬다** (confidence 가 잘못 예측할 수 있으므로). 연구용/선택적 기능이며, 기본 `DraftClient` / `FaultTolerantClient` 는 항상 full rejection sampling 을 수행한다.

---

## Backend 별 구현 차이

`verify()` / `verify_batch()` 의 **시그니처는 backend 간 동일**하지만 내부 경로는 다릅니다.

### HfVerifier (Phase 1)

- `transformers.AutoModelForCausalLM.from_pretrained(...)` 로 target 로드.
- 매 verify 호출마다 `context + draft` 전체를 `use_cache=False` 로 forward. 단순/정확성 우선.
- `verify_batch()` 는 요청들을 `pad_token_id` 로 패딩해 한 번의 forward 로 처리 (`tests/test_batched_verify.py` 로 회귀 방지).
- 구현: `src/distspec/server/hf_verifier.py`.

### VllmVerifier (Phase 2)

- `vllm.LLM(model=..., tensor_parallel_size=...)` 로 target 로드. PagedAttention, continuous batching, prefix cache 를 모두 vLLM 이 관리.
- verify 한 번당 `llm.generate(prompts=context+draft, sampling_params=SamplingParams(max_tokens=1, prompt_logprobs=max(2,K)))`.
- draft 위치별 target 분포는 **`prompt_logprobs[ctx_len + k]` 의 rank-1 토큰**으로 추출.
- `verify_batch()` 는 여러 요청의 prompt list 를 한 번에 넘겨 vLLM 스케줄러에 배치 맡김 — 별도 패딩 로직 불필요.
- **현재 Greedy only**. `temperature > 0` 은 `NotImplementedError` (Phase 2 S2 로 예정).
- 구현: `src/distspec/server/vllm_verifier.py`. 사용법/튜닝: [VLLM_BACKEND](./VLLM_BACKEND.md).

### 한눈 비교

| 관점 | HfVerifier | VllmVerifier |
|---|---|---|
| Target 실행 | transformers forward | vllm.LLM.generate |
| KV 관리 | use_cache=False (simplicity) | PagedAttention block table |
| 배치 | pad+attention_mask로 직접 | 스케줄러에 위임 |
| Multi-GPU | 미지원 | `tensor_parallel_size=N` |
| 샘플링 모드 | greedy + random | greedy 전용 (S1) |
| 의존성 | `[torch]` extra | `[vllm]` extra + CUDA |
| 전형적 latency (gpt2, 1토큰 verify) | ~200ms | 첫 요청 ~200ms, 이후 5–10ms |

### 선택 기준

- **CPU 로 돌리거나 random 모드가 필요** → `--backend hf`
- **GPU + 처리량/지연 최적화가 필요** → `--backend vllm`
- **Client / Protocol / FSM / Adaptive K 는 backend 와 무관** — 서버 옵션만 바꾸면 동일 클라이언트로 동작합니다.

---

## References

### 1차 자료
- Chen, C., et al. (2023). *Accelerating LLM Decoding with Speculative Sampling.* DeepMind. [[arXiv:2302.01318]](https://arxiv.org/abs/2302.01318)
- Leviathan, Y., et al. (2023). *Fast Inference from Transformers via Speculative Decoding.* [[arXiv:2211.17192]](https://arxiv.org/abs/2211.17192)
- Kim, S., et al. (2023). *Speculative Decoding with Big Little Decoder.* (BiLD) [[arXiv:2302.07863]](https://arxiv.org/abs/2302.07863)
- Ong, I., et al. (2024). *RouteLLM: Learning to Route LLMs with Preference Data.* [[arXiv:2406.18665]](https://arxiv.org/abs/2406.18665)
- Liu, Z., et al. (2024). *SVIP: Self-Verification Length Policy for Speculative Decoding.* [[arXiv:2411.18462]](https://arxiv.org/abs/2411.18462)

### 2차 자료
- vLLM RejectionSampler (참고 구현): [vllm/v1/sample/rejection_sampler.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/rejection_sampler.py)

---

**다음 섹션**: verify 결과의 수락률(α)과 RTT 관측을 어떻게 제어 루프에 피드백하는지 → [ADAPTIVE_CONTROL](./ADAPTIVE_CONTROL.md)
