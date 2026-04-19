# Draft Methods — N-gram, Suffix, EAGLE

[DESIGN](./DESIGN.md) 의 `DraftProposer` 추상화를 구현하는 세 가지 전략을 정리한다. 세 방식은 **"모델 필요성 ↔ 수락률"** 의 트레이드오프 축에 놓여 있고, 런타임에 교체 가능하다.

---

## 공통 인터페이스

```python
class BaseDraftProposer(ABC):
    def propose(
        self,
        context_tokens: list[int],
        hidden_states: Optional[torch.Tensor] = None,  # EAGLE 전용
        sampling_params: Optional[SamplingParams] = None,
    ) -> DraftOutput:
        """K 토큰을 제안. confidence_scores 포함 가능."""
```

정의 위치: `prototype/client/draft_proposer.py:43-78`

`DraftOutput` 의 필드:
- `draft_tokens: list[int]`
- `draft_probs: Optional[Tensor]` — EAGLE 은 분포, N-gram/Suffix 는 `None` (서버에서 uniform 가정)
- `hidden_states: Optional[Tensor]` — EAGLE 이 다음 propose 에 사용
- `confidence_scores: Optional[list[float]]` — [VERIFICATION](./VERIFICATION.md) 의 BiLD/SVIP 최적화에 사용

---

## N-gram Draft Proposer

**아이디어**: context 의 마지막 $n$ 개 토큰을 패턴으로 삼아, **같은 context 의 과거 구간**에서 동일 패턴이 나타난 위치를 찾고 그 직후 토큰들을 draft 로 제안.

### 알고리즘

```
for match_len in [ngram_window, ngram_window-1, ..., min_match_length]:
    pattern = tokens[-match_len:]
    for i in 0 .. n - match_len:
        if tokens[i:i+match_len] == pattern:
            return tokens[i+match_len : i+match_len+K]
```

- 최장 일치(longest first) 로 탐색하여 짧은 가짜 매칭을 피함.
- Numba 가용 시 `_kmp_search_numba` 로 JIT 가속 (큰 context 에서 O(n) 보장).
- **무상태**. 매 호출마다 context 내부만 본다.

### 언제 효과적인가

- 반복 구조가 있는 텍스트 (코드, JSON, 표, 로그, 채팅에서 이전 답변 인용 등).
- 토큰 단위 수락률이 매우 높을 수 있음 (α > 0.9 관측되기도 함 — vLLM ngram proposer 벤치).

### 언제 부적절한가

- 자유 텍스트 생성 (소설, 장문 추론) 에서 α 가 급락.
- 프롬프트가 매우 짧으면(< `min_match_length`) 아예 작동 안 함.

### 구현 레퍼런스

`prototype/client/draft_proposer.py:85-275`

---

## Suffix Decoding Proposer

**아이디어**: N-gram 과 유사하지만, **이전 요청들의 패턴까지 suffix tree 에 축적** 하여 활용. 현재 context 에 없는 패턴도 과거 대화에서 배운 것이라면 제안 가능.

### 알고리즘

```
_tree: dict[suffix_tuple, dict[next_token, count]]

# 제안 시
for suf_len in [max_suffix_len .. min_suffix_len]:
    suffix = current[-suf_len:]
    if suffix in tree:
        # 가장 빈도 높은 next_token 선택 (min_token_prob 이상)
        ...

# 생성 완료 후
update_tree(generated_sequence)  # 모든 suffix 길이로 count 갱신
```

- 각 suffix → {token: count} 매핑으로 확률 추정.
- `min_token_prob` 이하의 저확률 후보는 제외.
- `max_tree_size` 로 메모리 상한.

### N-gram 과의 차이

| 속성 | N-gram | Suffix |
|---|---|---|
| 패턴 탐색 범위 | 현재 context 내부만 | 모든 과거 생성 |
| 상태 | 없음 | suffix tree |
| confidence 근거 | 매칭 길이 | 빈도 기반 확률 |
| 부적절한 상황 | 자유 텍스트 | 도메인이 급변하는 환경 (학습된 패턴이 오히려 방해) |

### 구현 레퍼런스

`prototype/client/draft_proposer.py:282-452`

---

## EAGLE Draft Proposer

**아이디어**: target 모델의 **hidden state 정보를 draft 에 전달**하여 draft 품질을 높이는 기법 ([Li et al., 2024](https://arxiv.org/abs/2401.15077)). 작은 draft 모델 (예: 1B) 을 `autoregressive` 하게 돌리되, target 의 최근 hidden state 를 입력으로 함께 넣어 그 분포를 더 잘 흉내내도록 한다.

### 알고리즘 (본 프로토타입의 경량 버전)

```python
for step in range(K):
    outputs = draft_model(
        input_ids=current_ids,
        past_key_values=kv_cache,
        use_cache=True,
    )
    logits = outputs.logits[:, -1, :]
    next_token = sample(logits, sampling_params)  # greedy / top-k / top-p
    draft_tokens.append(next_token)
    draft_probs.append(softmax(logits))
    current_ids = next_token

return DraftOutput(draft_tokens, draft_probs=stack(draft_probs),
                   hidden_states=last_layer_hidden)
```

### 논문 EAGLE 과의 차이

| 항목 | 논문 EAGLE | 본 프로토타입 |
|---|---|---|
| Draft 모델 | 전용 head 를 target 위에 학습 | 기성 HF 작은 모델 그대로 |
| Hidden state 사용 | target 의 마지막 레이어 hidden 을 draft 입력에 주입 | **Phase B 에서 경량 concat 만 구현 예정** — 현재는 `pass` |
| Tree attention | 지원 | 미지원 |
| 수락률 | ~0.8+ | 0.3~0.6 수준 예상 (draft 모델 품질에 좌우) |

즉 **본 구현은 "EAGLE 스타일 구조 + 기성 draft 모델" 의 경량 버전**이다. 진짜 논문 EAGLE 재현에는 draft head 재학습이 필요하므로 Phase 1 범위 밖이다.

### 구현 레퍼런스

`prototype/client/draft_proposer.py:459-673`

---

## 선택 가이드

| 환경 | 권장 |
|---|---|
| CPU-only 엣지 디바이스 | **N-gram** (무모델, 반복 텍스트에서 매우 빠름) |
| 긴 대화 / 다중 턴 환경 (과거 요청 재활용) | **Suffix** |
| GPU 있는 클라이언트 + 일반 텍스트 생성 | **EAGLE 경량** |
| 최고 수락률 필요 + draft head 학습 자원 있음 | 논문 EAGLE (Phase 2 이후) |

**팩토리**: `create_draft_proposer(method="ngram" | "suffix" | "eagle", ...)` — `draft_proposer.py:680-718`

---

## Confidence 통합

세 proposer 모두 `DraftOutput.confidence_scores` 를 선택적으로 채울 수 있다. 이 점수는 [06-VERIFICATION §. Confidence 기반 최적화](./VERIFICATION.md#confidence-기반-최적화) 에서 Token-level skip / Query-level routing / Adaptive window 에 사용된다.

- **N-gram**: 매칭 길이 기반 (`base_conf * 0.95^i`)
- **Suffix**: 빈도 기반 (`count / total`)
- **EAGLE**: 분포의 max prob

---

## References

- Li, Y., et al. (2024). *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.* [[arXiv:2401.15077]](https://arxiv.org/abs/2401.15077)
- Cai, T., et al. (2024). *Medusa: Simple LLM Inference Acceleration Framework.* [[arXiv:2401.10774]](https://arxiv.org/abs/2401.10774) (참고: 본 프로토타입 미구현)
- Fu, Y., et al. (2024). *Break the Sequential Dependency of LLM Inference Using Lookahead Decoding.* [[arXiv:2402.02057]](https://arxiv.org/abs/2402.02057) (N-gram 의 학술적 상위 기법)
- vLLM ngram proposer 소스: [vllm/v1/spec_decode/ngram_proposer.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/ngram_proposer.py)

---

**다음 섹션**: 이렇게 생성된 draft 가 target 에서 어떻게 검증되는지 → [VERIFICATION](./VERIFICATION.md)
