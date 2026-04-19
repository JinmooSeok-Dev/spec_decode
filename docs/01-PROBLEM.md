# 01. Problem Definition — 왜 분산 Speculative Decoding 인가

## Executive Summary

원격 GPU 에서 LLM 을 서빙받는 환경에서, 일반 autoregressive 디코딩은 **토큰 하나마다 네트워크 왕복(RTT)을 누적**시켜 end-to-end latency 를 크게 악화시킨다. 예를 들어 256 토큰 응답에 RTT 가 50ms 면 그 자체로 **12.8s 의 네트워크 오버헤드**가 붙는다. 이 프로젝트는 draft/target 을 **서로 다른 프로세스(네트워크 건너편)** 에 두는 Client-Server 분산 구조로 Speculative Decoding(SD)을 재구성하여, 한 번의 왕복당 여러 토큰을 수락해 RTT 상각(amortize) 하는 것을 목표로 한다.

## Who / What / When

| 항목 | 내용 |
|---|---|
| **Who** | 원격/클라우드 GPU 를 사용하는 LLM 소비자 (엣지 디바이스, 모바일, 리소스 제약 클라이언트, 다중 클라이언트 공용 서빙) |
| **What** | Autoregressive 디코딩이 토큰마다 RTT 를 직렬 누적시키는 구조적 문제. 또한 클라이언트가 target 모델을 로컬에 로드할 수 없는 환경적 제약. |
| **When/Where** | (a) 엣지 클라이언트가 클라우드 target 모델을 호출할 때, (b) 한 target 서버를 다수 클라이언트가 공유할 때, (c) 네트워크 품질이 불안정한 환경 |

## 정량적 Impact

### 1) RTT 가 latency 에 직접 누적된다

단일 target 모델을 원격에서 호출하는 naive 구조에서 생성 한 토큰당 최소 한 번의 왕복이 발생한다.

$$
T_{\text{total}} = N \cdot (T_{\text{decode}} + \text{RTT})
$$

- $N=256$, $T_{\text{decode}} = 20\text{ms}$, $\text{RTT} = 50\text{ms}$
- $T_{\text{total}} = 256 \cdot 70\text{ms} = 17.92\text{s}$ (이 중 RTT 만 $12.8\text{s}$, 71%)

LAN 환경($\text{RTT}=1\text{ms}$) 이라면 RTT 비중은 5% 미만으로 무시할 수 있지만, WAN/모바일/엣지 환경에서는 **RTT 가 전체 latency 의 절반 이상을 차지**하게 된다.

### 2) Speculative Decoding 의 효과는 수락률(α) 과 K 에 비례

SD 한 스텝에서 기대 수락 토큰 수 ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192)):

$$
\mathbb{E}[\text{accepted}] = \frac{1 - \alpha^K}{1 - \alpha}
$$

- $\alpha = 0.7$, $K=5$ → 기대 수락 $\approx 2.7$ 토큰/스텝 (1토큰 bonus 포함 시 3.7)
- 즉 왕복 1회로 평균 3~4 토큰 확정 → 위 예에서 $T_{\text{total}}$ 이 약 $17.9\text{s} \rightarrow 5.4\text{s}$ 로 단축 가능 (3.3× 속도 향상)

### 3) 분산 환경에서 특히 큰 효과

SD 는 원래 target GPU 내부의 순차 디코딩 병목을 풀려고 고안됐지만 ([Chen et al., 2023](https://arxiv.org/abs/2302.01318)), **RTT 가 지배적인 분산 환경에서 체감 효과가 훨씬 크다**. 1ms LAN 대비 50ms WAN 에서 SD 의 상대적 이득이 10×+ 로 커진다.

## Urgency — 왜 지금 해결해야 하는가

- **LLM-as-a-Service 보편화**: 사용자가 로컬 GPU 를 두는 대신 원격 API 에 의존하는 비율이 급격히 증가. OpenAI/Anthropic/Gemini API 호출은 전부 RTT 누적 구조.
- **엣지 AI 확산**: 모바일/IoT 에서 LLM 을 "거드는" 형태로 쓰려면 local draft + remote target 패턴이 자연스럽다.
- **Multi-tenant 공용 GPU**: 하나의 고성능 target 서버를 다수 클라이언트가 공유하는 인프라가 표준이 되면서, 각 클라이언트가 draft 를 자기 쪽에서 만들고 서버는 검증만 하는 역할 분리가 필요.

## 이 프로젝트에서 해결하는 것과 하지 않는 것

### 해결 (Goals)
- Client-Server 구조로 SD 를 재구성하여 RTT 상각
- 여러 draft 생성 전략(N-gram, Suffix, EAGLE) 을 동일 인터페이스로 교체 가능
- 수락률/RTT 에 따라 K 를 동적으로 조정 (Adaptive Speculation)
- 네트워크 장애 시 degraded/fallback 모드로 가용성 유지

### 하지 않음 (Non-Goals — 현 Phase 1)
- Production 수준의 scheduler/throughput 최적화 (PagedAttention, continuous batching 등은 Phase 2 에서 vLLM 통합으로)
- Multi-GPU / Tensor parallel target 서빙
- Draft-head 재학습 기반 full EAGLE (hidden state 주입은 토이 수준)
- 실시간 multi-tenant 라우팅 / 영속적 상태 이관

---

**다음 섹션**: 위에서 정의한 문제가 실제로 어떤 상황에서 발생하는지, 시나리오 관점에서 정리한다 → [02-SCENARIOS](./02-SCENARIOS.md)
