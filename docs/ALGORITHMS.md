# Algorithms — 분산 SD 알고리즘 비교와 선택 근거

[SCENARIOS](./SCENARIOS.md) 에서 도출된 요구사항(분포 보존, RTT 상각, 다중 클라이언트 격리, 장애 내성)을 만족시키는 분산 Speculative Decoding 알고리즘을 비교하고, 본 프로토타입이 Client-Server 변형을 선택한 근거를 정리한다.

---

## Speculative Decoding 기본 복기

두 모델을 조합: **Draft Model $M_q$** (빠르고 작음, 분포 $q$) + **Target Model $M_p$** (정확하고 큼, 분포 $p$).

한 스텝에서:
1. $M_q$ 로 $K$ 개 draft 토큰 $\tilde{x}_1, \ldots, \tilde{x}_K$ 를 **autoregressive** 하게 생성
2. $M_p$ 에 $\text{context} + \tilde{x}_{1:K}$ 를 **한 번의 forward** 로 넣어 각 위치의 $p(x_i | \cdot)$ 를 얻음
3. Rejection Sampling 으로 각 $\tilde{x}_i$ 수락/거절:
   - $u \sim U(0,1)$, 수락 확률 $\min(1, p(\tilde{x}_i)/q(\tilde{x}_i))$
   - 거절 시 recovered 분포 $\max(0, p - q)/Z$ 에서 bonus 토큰 샘플링 후 중단
4. 모두 수락된 경우 $M_p$ 의 다음 위치에서 bonus 토큰 샘플링

**수학적 보장**: 이 과정의 전체 출력 분포는 $M_p$ 단독 샘플링과 **정확히 동일**하다 ([Chen et al., 2023](https://arxiv.org/abs/2302.01318) Theorem 1, [Leviathan et al., 2023](https://arxiv.org/abs/2211.17192) §). 즉 품질은 보존하고 latency 만 개선한다.

**속도 이득의 출처**: target forward 가 $K$ 토큰을 **병렬**로 처리하는 한편, draft 는 $K$ 토큰을 순차로 만들지만 모델 크기가 작아 훨씬 싸다. 한 스텝당 기대 수락 토큰:

$$
\mathbb{E}[\#\text{accepted}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}, \quad \alpha = \text{수락률}
$$

---

## 분산 SD 변형 비교

SD 를 "어디서 draft 가 만들어지고 / 어디서 verify 가 일어나며 / 상태가 어디 유지되는가" 로 구분하면 네 가지 대표 패턴이 나온다.

| 변형 | Draft 위치 | Verify 위치 | 네트워크 왕복 | 대표 예시 |
|---|---|---|---|---|
| **Monolithic** | Target GPU | Target GPU | 0 | vLLM in-process SD |
| **Client-Server** | Client | Target Server | per-step 1회 | 본 프로토타입 |
| **Peer-to-Peer** | 대등 노드들이 교대 | 대등 노드들이 교대 | per-step 2회+ | 학술 연구 레벨 |
| **Hierarchical** | 여러 단계 (tiny → small) | Target | 단계 수 × 1회 | Staged SD, SpecInfer |

### Monolithic (vLLM in-process)

Draft 와 target 이 같은 프로세스(같은 GPU 또는 같은 노드) 안에서 동작.

**장점**
- 네트워크 왕복 0. 모든 오버헤드가 PCIe / NVLink 내부.
- KV cache 를 공유 메모리로 직접 참조 (PagedAttention).

**단점**
- Client 가 target 모델을 로컬에 로드해야 함 → [02 Scenario 1](./SCENARIOS.md#scenario-1--엣지-클라이언트--클라우드-target) 성립 불가.
- 다중 클라이언트 서빙 시 draft 연산이 target GPU 를 점유 (자원 경합).

**참고**: [vLLM V1 spec_decode](https://github.com/vllm-project/vllm/tree/main/vllm/v1/spec_decode), [vLLM blog V1 alpha](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)

### Client-Server (본 프로토타입)

Draft 는 client 에서, target verify 는 server 에서. 둘은 ZMQ 메시지로 통신.

**장점**
- [02 Scenario 1, 2](./SCENARIOS.md) 를 자연스럽게 만족.
- Draft 전략을 client 별로 다르게 선택 가능 (N-gram vs EAGLE).
- Target server 는 verify 에 전념 → GPU 자원 분리.

**단점**
- Per-step 네트워크 왕복 1회. $\text{RTT}$ 가 한 스텝의 하한.
- 서버가 per-client state 를 들고 있어야 함 (sticky routing 가정).

**왕복 오버헤드 완화**: 한 왕복에 $K$ 토큰을 담아 상각 ([PROBLEM](./PROBLEM.md) § 정량적 Impact).

### Peer-to-Peer

노드들이 역할을 돌아가며 교대. 한 노드가 이번 스텝은 draft, 다음 스텝은 target 역할.

**장점**
- 역할 고정 없음 → 자원 균형.

**단점**
- 복잡도 급증 (합의/역할 조정/상태 동기화).
- 실전 환경에서 검증된 구현이 드묾.

→ **본 프로젝트 범위 외** (Non-goal).

### Hierarchical (Staged SD)

여러 크기의 draft 를 체인으로 연결. 가장 작은 모델이 가장 작은 draft 를 만들고, 중간 모델이 검증/재생성, 최종적으로 target 이 검증.

**장점**
- 수락률이 단계별로 상승 → 최종 토큰당 비용 더 절감.

**단점**
- 단계 수만큼 왕복 증가 → 분산 환경에서는 오히려 손해.
- 구현/튜닝 복잡도 큼.

대표 예: [SpecInfer (Miao et al., 2023)](https://arxiv.org/abs/2305.09781).

→ **Phase 1 범위 외**. Phase 2 에서 vLLM 통합 후 고려 가능.

---

## 선택: Client-Server

### 결정 매트릭스

| 기준 (가중치) | Monolithic | **Client-Server** | P2P | Hierarchical |
|---|---|---|---|---|
| Scenario 1 (엣지) | ❌ 0 | ✅ 3 | △ 2 | △ 2 |
| Scenario 2 (공유) | △ 1 | ✅ 3 | △ 2 | △ 2 |
| Scenario 3 (장애 내성) | N/A | ✅ 3 | △ 1 | △ 1 |
| 구현 복잡도 (낮을수록 ↑) | ✅ 3 | △ 2 | ❌ 0 | ❌ 1 |
| Phase 2 vLLM 통합 친화성 | ✅ 3 | ✅ 3 | ❌ 0 | △ 2 |
| **총점** | 7 | **14** | 5 | 8 |

**결정: Client-Server**. 세 시나리오를 모두 만족하면서 구현 복잡도가 통제 가능하고, Phase 2 에서 target server 를 vLLM `LLMEngine` 으로 교체하는 경로가 자연스럽다.

### 채택하지 않은 대안 — 기록 (ADR)

- **Monolithic**: Scenario 1(엣지) 이 성립 불가 → 기각.
- **Peer-to-Peer**: 프로토타입 단계에서 합의 알고리즘 구현 비용이 과도 → Non-goal.
- **Hierarchical**: 분산 환경에서 왕복 증가로 이득 상쇄 → Phase 2 이후로 보류.

---

## Rejection Sampling 의 핵심 수식 (구현에 직결)

[VERIFICATION](./VERIFICATION.md) 에서 자세히 다루지만, 본 문서에서는 "왜 이 알고리즘이 분포를 보존하는가" 만 요약한다.

토큰 $\tilde{x}$ 가 draft 분포 $q$ 에서 나왔다고 하자. 수락 확률을 $r = \min(1, p(\tilde{x})/q(\tilde{x}))$ 로 두면, 수락된 토큰의 분포는:

$$
\Pr[\text{수락} \wedge X = x] = q(x) \cdot \min(1, p(x)/q(x)) = \min(q(x), p(x))
$$

거절 시 recovered 분포 $p_{\text{rec}}(x) = \max(0, p(x) - q(x)) / Z$ 에서 샘플하면:

$$
\Pr[\text{거절}] \cdot p_{\text{rec}}(x) = (1 - A) \cdot \frac{\max(0, p(x) - q(x))}{Z} = \max(0, p(x) - q(x))
$$

($A = \sum_x \min(q(x), p(x))$, $Z = 1 - A$)

합치면:

$$
\min(q(x), p(x)) + \max(0, p(x) - q(x)) = p(x)
$$

즉 전체 출력은 정확히 target 분포 $p$ 와 같다. **이 등식이 SD 의 lossless 성질을 보장한다.**

실제 코드 구현: `prototype/server/target_verifier.py:147-213` (`_random_verify`), `:215-236` (`_sample_from_recovered`).

---

## References

### 1차 자료 (논문)
- Leviathan, Y., Kalman, M., & Matias, Y. (2023). *Fast Inference from Transformers via Speculative Decoding.* ICML 2023. [[arXiv:2211.17192]](https://arxiv.org/abs/2211.17192)
- Chen, C., Borgeaud, S., Irving, G., et al. (2023). *Accelerating Large Language Model Decoding with Speculative Sampling.* DeepMind. [[arXiv:2302.01318]](https://arxiv.org/abs/2302.01318)
- Li, Y., Wei, F., Zhang, C., Zhang, H. (2024). *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.* [[arXiv:2401.15077]](https://arxiv.org/abs/2401.15077)
- Miao, X., et al. (2023). *SpecInfer: Accelerating LLM Serving with Tree-based Speculative Inference.* [[arXiv:2305.09781]](https://arxiv.org/abs/2305.09781)

### 2차 자료 (블로그/구현)
- vLLM V1 Alpha Release. (2025). [[blog.vllm.ai]](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html) (접근: 2026-04)
- vLLM SD source: [github.com/vllm-project/vllm/tree/main/vllm/v1/spec_decode](https://github.com/vllm-project/vllm/tree/main/vllm/v1/spec_decode)

---

**다음 섹션**: 선택한 Client-Server 아키텍처를 어떻게 구현했는지, 프로토콜과 컴포넌트 관점에서 기술한다 → [DESIGN](./DESIGN.md)
