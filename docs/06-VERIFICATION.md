# 06. Verification — Rejection Sampling 과 Confidence 기반 최적화

[05-DRAFT_METHODS](./05-DRAFT_METHODS.md) 에서 생성된 draft 토큰을 target 모델에서 어떻게 검증하는지, 그리고 일부 토큰에 대해 검증 자체를 **건너뛸** 수 있는 confidence 기반 최적화를 정리한다.

---

## 1. 기본 흐름

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

## 2. Rejection Sampling — Greedy 모드

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

## 3. Rejection Sampling — Random 모드 (수학적 핵심)

$\text{temperature} > 0$ 일 때 draft 토큰 $\tilde{x}$ 에 대해:

$$
A = \min\left(1, \frac{p(\tilde{x})}{q(\tilde{x})}\right)
$$

$u \sim U(0,1)$ 이 $u < A$ 이면 수락, 아니면 거절 후 recovered 분포에서 bonus 샘플링:

$$
p_{\text{rec}}(x) = \frac{\max(0, p(x) - q(x))}{\sum_{y} \max(0, p(y) - q(y))}
$$

### 3.1 왜 이 공식인가 — 분포 보존 증명

draft 분포 $q$, target 분포 $p$, 전체 수락 확률 $A_{\text{total}} = \sum_x \min(q(x), p(x))$. 한 토큰 위치에서 최종 출력 토큰의 분포는:

$$
\begin{aligned}
\Pr[X = x] &= \underbrace{q(x) \cdot \min(1, p(x)/q(x))}_{\text{draft 에서 수락}} + \underbrace{(1 - A_{\text{total}}) \cdot p_{\text{rec}}(x)}_{\text{거절 후 recovered}} \\
&= \min(q(x), p(x)) + \max(0, p(x) - q(x)) \\
&= p(x)
\end{aligned}
$$

즉 **수락된 토큰과 recovered 토큰을 합친 출력 분포는 정확히 target 분포와 같다**. 이것이 SD 가 quality-preserving 한 이유다 ([Chen et al., 2023](https://arxiv.org/abs/2302.01318) Theorem 1).

### 3.2 Draft 확률이 없을 때 (N-gram)

N-gram 은 확률 분포 없이 토큰 ID 만 내므로, verifier 는 $q(x) = 1/V$ (uniform) 을 가정한다 (`target_verifier.py:172-175`).

이 경우 수락 확률은 $\min(1, V \cdot p(\tilde{x}))$ 가 되어 **$p(\tilde{x})$ 가 uniform 보다 충분히 크면 거의 항상 수락**. 실제로 N-gram 이 고르게 맞추는 "반복 패턴" 환경에서는 $p(\tilde{x})$ 가 매우 높아 이 근사가 잘 작동한다.

### 3.3 Top-k / Top-p 필터링

Sampling param 의 `top_k`, `top_p` 는 **target logits 에만** 적용 (`_apply_sampling_filters`, `target_verifier.py:238-266`). 이렇게 해야 정의 그대로의 target 분포에서 샘플하는 것과 동치.

### 3.4 구현

`target_verifier.py:147-213` (`_random_verify`), `:215-236` (`_sample_from_recovered`).

---

## 4. Confidence 기반 최적화

Rejection sampling 을 **항상** 수행하는 대신, draft 쪽의 확신도가 매우 높은 토큰은 서버 검증을 건너뛸 수 있다. 이는 품질 보존과 속도 향상의 트레이드오프다.

### 4.1 세 가지 접근법

| 방식 | 스코프 | 전형적 논문 | 효과 |
|---|---|---|---|
| Token-level Skip | 각 토큰 | **BiLD** ([Kim et al., 2023](https://arxiv.org/abs/2302.07863)) | confidence 높은 prefix 는 바로 yield, 나머지만 서버로 |
| Query-level Routing | 요청 단위 | **RouteLLM** ([Ong et al., 2024](https://arxiv.org/abs/2406.18665)) | 쉬운 질의는 draft 단독, 어려운 질의만 target |
| Adaptive Window | draft 윈도우 크기 | **SVIP** ([Liu et al., 2024](https://arxiv.org/abs/2407.06677)) | draft 가 불확실해지면 더 이상 뽑지 않고 조기 종료 |

### 4.2 Confidence 메트릭

`common/confidence.py` 에서 3 종 지원:

| 메트릭 | 정의 | 특징 |
|---|---|---|
| `entropy` | $H(P) = -\sum p_i \log p_i$ | 분포 전체를 반영, 계산 비쌈 |
| `max_prob` | $\max(P)$ | 단순, 빠름 |
| `logit_margin` | $\text{logit}_1 - \text{logit}_2$ | 1, 2위 격차로 직관적 |

### 4.3 트레이드오프

- 높은 `skip_threshold` → 속도 ↑ 품질 ↓ (잘못된 draft 를 통과시킬 위험).
- 낮은 threshold → 속도 ↓ 품질 ↑ (모든 토큰을 검증).
- Phase 1 기본값: `skip_threshold = 0.8`, `warmup_steps = 5` (초반 몇 스텝은 무조건 검증).

### 4.4 구현

- `common/confidence.py` — 공통 계산기와 분류기
- `client/confidence_client.py` — `ConfidenceSkipClient`, `QueryRoutingClient`, `AdaptiveWindowClient`

**주의**: Confidence skip 은 엄밀한 분포 보존을 **깬다** (confidence 가 잘못 예측할 수 있으므로). 연구용/선택적 기능이며, 기본 `DraftClient` / `FaultTolerantClient` 는 항상 full rejection sampling 을 수행한다.

---

## 5. Batched Verification (Phase B 에서 완성)

현재 `BatchVerifier` (`target_verifier.py:461-554`) 는 골격만 구현되어 있고, `_process_batch` 는 요청별 순차 `run_in_executor` 로 돌아간다. 진짜 batched target forward 는 다음을 요구한다:

1. 각 요청의 `context + draft` 를 길이가 다른 시퀀스로 패딩
2. `attention_mask` 로 유효 토큰 표시
3. 한 번의 target forward 에서 각 요청의 draft 위치 logits 를 추출
4. 요청별로 별도 `RejectionSampler.forward` 호출

이 작업이 Phase B 의 첫 번째 항목이다 ([09-ROADMAP](./09-ROADMAP.md) 참조).

---

## 6. KV Cache 관리

Rejection 이 발생하면 거절된 draft 토큰의 K/V 는 무효하므로 즉시 잘라내야 한다 (`_truncate_kv_cache`, `target_verifier.py:434-450`):

```python
keep_len = len(context_tokens) + num_accepted
kv_cache = tuple(
    (k[:, :, :keep_len, :], v[:, :, :keep_len, :])
    for (k, v) in kv_cache
)
```

**한계 (Phase 1)**: HF `past_key_values` 튜플을 직접 슬라이싱하므로 **자료구조 변경에 깨지기 쉽고**, **메모리 재사용이 비효율적** (슬라이스 복사 비용). Phase 2 에서 vLLM 의 block table 기반 할당/해제로 교체된다 ([09-ROADMAP](./09-ROADMAP.md)).

---

## 7. References

### 1차 자료
- Chen, C., et al. (2023). *Accelerating LLM Decoding with Speculative Sampling.* DeepMind. [[arXiv:2302.01318]](https://arxiv.org/abs/2302.01318)
- Leviathan, Y., et al. (2023). *Fast Inference from Transformers via Speculative Decoding.* [[arXiv:2211.17192]](https://arxiv.org/abs/2211.17192)
- Kim, S., et al. (2023). *Speculative Decoding with Big Little Decoder.* (BiLD) [[arXiv:2302.07863]](https://arxiv.org/abs/2302.07863)
- Ong, I., et al. (2024). *RouteLLM: Learning to Route LLMs with Preference Data.* [[arXiv:2406.18665]](https://arxiv.org/abs/2406.18665)
- Liu, Z., et al. (2024). *SVIP: Self-Verification Length Policy for Speculative Decoding.* [[arXiv:2411.18462]](https://arxiv.org/abs/2411.18462)

### 2차 자료
- vLLM RejectionSampler (참고 구현): [vllm/v1/sample/rejection_sampler.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/rejection_sampler.py)

---

**다음 섹션**: verify 결과의 수락률(α)과 RTT 관측을 어떻게 제어 루프에 피드백하는지 → [07-ADAPTIVE_CONTROL](./07-ADAPTIVE_CONTROL.md)
