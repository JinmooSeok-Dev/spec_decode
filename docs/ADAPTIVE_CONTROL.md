# Adaptive Control — K 튜닝과 Fault-Tolerant FSM

[SCENARIOS](./SCENARIOS.md) 의 Scenario 1 (RTT 상각) 과 Scenario 3 (네트워크 불안정) 은 같은 뿌리를 공유한다: **"네트워크 조건과 수락률이 실시간으로 변하면, 고정된 K 와 고정된 운영 모드는 최적이 아니다"**. 본 문서는 이 두 시나리오에 대응하는 두 제어 메커니즘을 정리한다.

1. **Adaptive Speculation Length Controller** — 매 요청의 RTT 와 수락률 α 를 관측해 K 를 실시간 조정.
2. **Fault-Tolerant Client FSM** — SPECULATIVE / DEGRADED / FALLBACK 3-state 로 네트워크/서버 장애에 대응.

---

## Adaptive Speculation Length (K) Controller

### 왜 K 를 바꿔야 하는가

한 SD 스텝의 비용은 대략:

$$
T_{\text{step}} = K \cdot T_{\text{draft}} + T_{\text{verify}} + \text{RTT}
$$

한 스텝의 기대 수락 토큰 수는 [03-ALGORITHMS § ](./ALGORITHMS.md) 에서 본 것처럼:

$$
\mathbb{E}[\text{accepted}] = \frac{1 - \alpha^K}{1 - \alpha}
$$

토큰 하나당 **amortized latency** 는:

$$
L(K) = \frac{T_{\text{step}}}{\mathbb{E}[\text{accepted}]}
$$

- $\alpha$ 가 크면 → 큰 $K$ 일수록 기대 수락이 더 많이 늘어 amortization 효과 ↑
- $\alpha$ 가 작으면 → 큰 $K$ 는 낭비 (대부분 거절). $K$ 를 줄여야 draft 비용과 verify forward 길이를 절약
- $\text{RTT}$ 가 크면 → 한 왕복에 더 많이 담아야 하므로 $K$ ↑
- $\text{RTT}$ 가 작으면 → $K$ 가 커봐야 한계효용 ↓ (왕복 비용이 무의미)

즉 **최적 $K^*$ 는 $(\alpha, \text{RTT})$ 의 함수**이고, 이 둘은 런타임에 관측 가능하다.

### Controller 공식

`AdaptiveSpeculationController` (`prototype/client/draft_client.py:60-162`) 의 결정 규칙:

$$
K^* = \arg\max_{K \in [K_{\min}, K_{\max}]} \frac{\mathbb{E}[\text{accepted}(K)]}{K + \text{RTT} / T_{\text{decode}}}
$$

- 분자: 기대 수락 토큰 수
- 분모: amortized cost — draft 토큰 K 개 생성 비용 + RTT 의 "토큰 환산"

**$T_{\text{decode}}$** 는 설정값 (기본 10ms, `AdaptiveSpecConfig.decode_time_estimate`). 이 값이 정확할수록 $K^*$ 가 정확.

### 동작 조건

```python
if len(history) < 5:                    return max_spec_tokens   # 워밍업
if avg_acceptance < 0.1:                return min_spec_tokens   # α 붕괴
else:                                    return argmax_K of score
```

- 히스토리가 5회 미만이면 관측이 부족해 공식 대신 상한값 사용.
- α 가 10% 미만이면 공식이 비정상 값을 낼 수 있어 하한값으로 clamp.

### 합성 시나리오 동작 예시

고정 RTT = 100ms, $T_{\text{decode}} = 10\text{ms}$ 일 때:

| α (수락률) | $K^*$ (합성) | 해석 |
|---|---|---|
| 0.3 | 2~3 | 수락이 낮음 → 큰 K 는 낭비 |
| 0.5 | 4~5 | 중간 |
| 0.7 | 6~8 | 수락률 충분 → K 확대로 RTT 상각 |
| 0.9 | 10 (상한) | 거의 다 수락 → 최대 K |

고정 α = 0.7 일 때 RTT 변화:

| RTT | $K^*$ (합성) | 해석 |
|---|---|---|
| 1ms | 2~3 | 왕복이 싸므로 K 키워도 이득 적음 |
| 50ms | 5~6 | 균형 |
| 200ms | 8~10 | 왕복이 비쌈 → 한 번에 최대한 담기 |
| 1000ms | 10 (상한) | 왕복 지배 — 최대 K |

### 측정값 기록 흐름

```
DraftClient.generate() 루프:
    start = time.time()
    draft = proposer.propose(..., K=controller.current_k)
    resp = await server.send_recv(draft)
    rtt = time.time() - start

    controller.record_result(rtt, num_draft=K, num_accepted=resp.num_accepted)
    # → _compute_optimal_k() 호출되어 controller._current_k 갱신
```

히스토리는 `deque(maxlen=20)` 로 최근 20 요청만 유지. 이는 네트워크 특성 변화(예: Wi-Fi ↔ 4G 전환) 에 적응할 수 있게 한다.

### 한계

- $T_{\text{decode}}$ 가 설정값 고정 — 실제 client 의 draft 속도가 다르면 부정확.
- Verify latency 를 별도 추정 안 함 (RTT 에 포함된 것으로 취급).
- 비정상적으로 긴 꼬리 RTT 가 평균을 흔들 수 있음 — 중앙값 기반으로 바꾸는 게 미래 개선 방향.

---

## Fault-Tolerant Client FSM

### 3-State 정의

| 상태 | 의미 | K | 서버 의존 |
|---|---|---|---|
| `SPECULATIVE` | 정상 SD 동작 | controller 가 결정 | 있음 (모든 토큰 검증) |
| `DEGRADED` | 낮은 수락률 감지 → K 강제 최소화 | `min_spec_tokens` | 있음 |
| `FALLBACK` | 서버 접근 불가 → draft 모델 단독 | n/a (1토큰씩) | 없음 |

정의 위치: `common/config.py:27-31` (`ClientMode` enum).

### 전이 조건

```
SPECULATIVE ──(최근 10회 평균 α < 0.2)──▶ DEGRADED
    │                                        │
    │                                        │ (consecutive_failures ≥ max_retries)
    │                                        ▼
    │                                     FALLBACK
    │                                        │
    │         (HealthCheck 성공)              │
    └─────◀──────────────────────────────────┘
              (30초 주기 _recovery_loop)
```

- **SPECULATIVE → DEGRADED**: 품질 저하가 아니라 **낭비 방지**. α 가 낮으면 큰 K 로 여러 번 왕복해도 수락이 별로 없으므로, 한 왕복에 1 토큰씩만 올려 서버 부담을 줄인다.
- **DEGRADED/SPECULATIVE → FALLBACK**: 네트워크/서버 자체가 문제. 연속 실패 카운트가 `max_retries` 초과 시 전환.
- **FALLBACK → SPECULATIVE**: `_recovery_loop` 이 30초마다 `HealthCheck` 를 보내 서버 회복 시 복귀. 이때 `consecutive_failures` 와 acceptance 이력도 리셋.

구현: `client/fault_tolerant_client.py:186-246`.

### DEGRADED 가 FALLBACK 과 다른 이유

- **FALLBACK 은 품질 희생**: draft 모델만 쓰므로 출력 분포가 더 이상 target 과 같지 않다.
- **DEGRADED 는 품질 유지**: 여전히 서버가 모든 토큰을 검증하므로 분포 보존성은 그대로. 단지 **처리량이 낮을 뿐**.

즉 "네트워크는 괜찮은데 수락률이 낮다" (예: 도메인 불일치) 와 "네트워크/서버가 죽었다" 를 분리 처리한다.

### 한계

- Recovery 주기가 고정 30초 — 서버가 자주 깜빡거리면 오버헤드.
- Fallback 중 생성된 내용은 품질 표시가 없어 사용자가 품질 저하를 인지하기 어려움. 향후 `generation_meta` 필드로 모드를 streaming 에 실을 필요.

---

## 두 메커니즘의 상호작용

두 제어기는 독립적으로 동작하지만 신호를 공유한다:

| 신호 | 출처 | 사용처 |
|---|---|---|
| `rtt` | `DraftClient.generate()` 의 타이밍 | Adaptive controller |
| `num_accepted / num_draft` | `VerifyResponse` | Adaptive controller + FSM (DEGRADED 판정) |
| `zmq.error.Again` / 예외 | 네트워크 레이어 | FSM (FALLBACK 판정) |
| `HealthResponse.is_healthy` | `HealthCheck` | FSM (SPECULATIVE 복귀) |

결과: **"처리량 최적화" (Adaptive K) 와 "가용성 보장" (FSM) 이 분리된 책임**을 갖고 한 client 안에 공존한다.

---

## 튜닝 가이드

| 환경 | `min_spec_tokens` | `max_spec_tokens` | `low_acceptance_threshold` | 비고 |
|---|---|---|---|---|
| LAN (RTT<5ms) | 1 | 5 | 0.3 | K 늘려도 이득 적음 |
| WAN (RTT~50ms) | 2 | 10 | 0.3 | 기본 설정 |
| 모바일/4G (RTT>200ms) | 3 | 15 | 0.2 | K 공격적으로 키움 |
| 코드/JSON (반복 패턴) | 3 | 15 | 0.4 | α 가 높을 때 K 상한 올리기 |

실제 적용은 `AdaptiveSpecConfig`(`common/config.py:171-190`) 를 통해.

---

**다음 섹션**: 위 메커니즘들이 실제로 분포 보존하고 목표 성능을 달성하는지 검증 결과 → [EVALUATION](./EVALUATION.md)
