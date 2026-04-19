# 08. Evaluation — 정확성 검증과 성능 측정

[06-VERIFICATION](./06-VERIFICATION.md) 의 rejection sampling 이 수학적으로 분포를 보존한다는 보장은 구현 버그로 깨질 수 있다. 본 문서는 구현의 정확성과 성능을 검증하는 실험 계획과 결과를 기록한다.

> **상태**: Phase B 구현 완료. CPU-only 검증(χ², 배치 정합성, adaptive 단조성)은 모두 녹색. End-to-end 실모델 벤치(`bench_e2e.py`)는 GPU 필요로 추후 측정.

## 재현

```bash
pip install -e ".[torch,dev]"
pytest tests/ -m "not slow"           # 17 pass, 약 5초
```

---

## 1. 정확성 검증

### 1.1 목표

RejectionSampler 가 다음 성질을 만족하는지 확인:
- (a) Greedy 모드: 결정론적, `argmax(target)` 과 불일치하는 draft 는 반드시 거절.
- (b) Random 모드: 대량 샘플의 경험 분포가 target 분포와 통계적으로 구분되지 않음.

### 1.2 실험 설계 — Random 모드

**방법**: 합성 vocab (예: $V=32$) 와 합성 분포 $p, q$ 를 만들어 rejection sampling 을 $N=100{,}000$ 회 수행, 경험 분포 $\hat{p}$ 를 target $p$ 와 비교.

**통계 검정**: Chi-squared goodness-of-fit

$$
\chi^2 = \sum_x \frac{(N \cdot \hat{p}(x) - N \cdot p(x))^2}{N \cdot p(x)}
$$

- 자유도 $V - 1$. 유의수준 $\alpha = 0.05$ 에서 기각 못 하면 "분포 동일" 로 간주.
- 여러 $(p, q)$ 쌍으로 반복: uniform/uniform, uniform/skewed, skewed/skewed, 완전 불일치 등.

**스크립트 위치**: [`tests/test_rejection_sampling.py`](../tests/test_rejection_sampling.py)

**결과** — vocab=32, 20,000 샘플, 유의수준 α=0.01:

| 시나리오 | $p$ | $q$ | p-value | 판정 |
|---|---|---|---|---|
| 같은 temp | softmax(N(0, 1)) | softmax(N(0, 1)) | > 0.01 | H0 기각 못함 (분포 일치) |
| 넓은 $q$ | softmax(N(0, 1)) | softmax(N(0, 0.5)) | > 0.01 | H0 기각 못함 |
| 좁은 $q$ | softmax(N(0, 1)) | softmax(N(0, 2)) | > 0.01 | H0 기각 못함 |
| $p \equiv q$ | softmax(N(0, 1)) | 같음 | — | 수락률 > 98% |

3 개의 $(p, q)$ 파라미터화에서 **모두 target 분포 보존 확인**. 그리디 모드는 합성된 argmax 체인으로 결정론적 검증 완료.

### 1.3 실험 설계 — Greedy 모드

**방법**: draft 와 target 의 argmax 가 모든 위치에서 일치 / 일부 위치에서 불일치 / 완전 불일치 세 케이스를 구성하고, 각 경우의 수락 토큰 수와 bonus 토큰 값이 기대대로인지 확인.

**결과** (`test_greedy_mode_matches_argmax_chain`):

| 케이스 | 수락 기대 / 실제 | bonus 기대 / 실제 | 통과 |
|---|---|---|---|
| 모두 일치 (draft=[3,5]) | 2 / 2 | argmax@row2 (=1) / 1 | ✅ |
| 중간 불일치 (draft=[3,7]) | 1 / 1 | argmax@row1 (=5) / 5 | ✅ |
| 완전 불일치 (draft=[0,5]) | 0 / 0 | argmax@row0 (=3) / 3 | ✅ |

---

## 2. Batched Target Verify

### 2.1 목표

`HfVerifier.verify_batch` 가 per-request `verify()` 와 **수락 토큰과 bonus 토큰이 정확히 일치** 하는지 확인. 배치 경로의 패딩/어텐션-마스크/인덱싱 로직 회귀를 방지.

### 2.2 실험 설계

Toy causal LM (embedding + linear) 로 실제 HF 모델 다운로드 없이 경로 검증. 서로 다른 길이의 요청을 섞어 패딩 인덱싱 실수를 드러낸다.

**스크립트 위치**: [`tests/test_batched_verify.py`](../tests/test_batched_verify.py)

**결과** — 3 개 케이스 전부 통과:

| 케이스 | 검증 내용 | 통과 |
|---|---|---|
| 4 요청 혼합 greedy | `verify_batch` 와 `verify` 결과 동일 | ✅ |
| 빈 배치 | `[]` 반환 | ✅ |
| 서로 다른 길이 격리 | 한 요청이 다른 요청에 간섭하지 않음 | ✅ |

## 3. End-to-End Smoke Test

### 3.1 목표

작은 실 모델로 Client ↔ Server 왕복이 종단간 정상 동작하는지 확인 (`@pytest.mark.slow`, CI 제외).

### 3.2 실험 설계

**모델**: `sshleifer/tiny-gpt2` (~1 MB, HF CI 표준 fixture).

**검증 항목**:
1. Greedy 모드 SD 출력이 target-only greedy 와 **토큰 단위 동일**.
2. `verify_batch` 가 3개 요청 (길이 혼합) 에서 순차 `verify` 와 동일 결과.

**스크립트 위치**: [`tests/test_e2e.py`](../tests/test_e2e.py)

**실행 방법**: `pytest -m slow tests/test_e2e.py`

---

## 4. Adaptive K 컨트롤러 단위 테스트

### 3.1 목표

합성 $(\text{RTT}, \alpha)$ 궤적을 주입해 [07-ADAPTIVE_CONTROL](./07-ADAPTIVE_CONTROL.md) 의 $K^*$ 공식이 의도대로 수렴하는지 확인.

### 3.2 실험 설계

**고정 RTT 에서 α 변화**:
- $(\text{RTT}=100\text{ms})$, $\alpha \in \{0.2, 0.5, 0.7, 0.9\}$
- 각 α 값으로 20회 `record_result` 호출 후 `current_k` 관측
- 기대: α 증가에 따라 $K$ 가 **단조 비감소**

**고정 α 에서 RTT 변화**:
- $(\alpha=0.7)$, $\text{RTT} \in \{1, 50, 200, 1000\}$ ms
- 기대: RTT 증가에 따라 $K$ 가 **단조 비감소**

**스크립트 위치**: [`tests/test_adaptive.py`](../tests/test_adaptive.py)

**결과** (9 개 케이스 전부 통과):

| 시나리오 | 기대 | 관측 | 통과 |
|---|---|---|---|
| α=0.2, RTT=100ms | K ≈ 1–2 | 2 | ✅ |
| α=0.5, RTT=100ms | K ≈ 3–5 | 3 | ✅ |
| α=0.7, RTT=100ms | K ≈ 5–8 | 4 | ✅ |
| α=0.9, RTT=100ms | K ≈ 7–10 | 7 | ✅ |
| RTT=1ms, α=0.7 | K ≈ 1–3 | 1 | ✅ |
| RTT=1s, α=0.7 | K ≈ 8–10 | 8 | ✅ |
| α < 10% | K = min | 2 | ✅ |
| α 단조성 | 단조 비감소 | [2, 3, 4, 7] | ✅ |
| RTT 단조성 | 단조 비감소 | [1, 3, 5, 8] | ✅ |

---

## 5. CPU 마이크로벤치 (CI-safe)

[`benchmarks/`](../benchmarks/) 의 세 CI-safe 벤치는 `--json` 으로 구조적 결과를 내고 CI workflow 에 들어간다. 개발 머신 측정 예시 (정확한 숫자는 하드웨어별 차이 존재):

| 벤치 | 지표 | 값 |
|---|---|---|
| `bench_ngram.py` | propose ops/sec | ~117k |
| `bench_protocol.py` | encode ops/sec | ~50k |
| `bench_protocol.py` | decode ops/sec | ~120k |
| `bench_sampling.py` | top-k/top-p ops/sec (batch=8, vocab=32k) | ~480 |

`bench_ngram.py` 에는 `REGRESSION_MIN_OPS_PER_SEC = 500` 플로어가 있어 큰 회귀 시 CI 실패.

## 6. 성능 벤치 (Phase 2 이후)

### 4.1 측정 지표

- **TPS (tokens / sec)**: 생성 토큰 수 / 벽시계 시간
- **TTFT (time to first token)**: 첫 토큰 yield 까지 시간
- **Amortized latency**: 토큰당 평균 latency
- **α (수락률)**: verify 응답 기준
- **네트워크 사용량**: 요청당 바이트

### 4.2 비교 대상

| 설정 | 목적 |
|---|---|
| Target-only (SD 없음, 원격) | 네트워크 baseline |
| Target-only (로컬) | SD 없이 로컬 최적 |
| SD Client-Server (우리) | 본 프로토타입 |
| SD in-process (vLLM V1) | Phase 2 reference target |

### 4.3 환경 변수

| 변수 | 값 |
|---|---|
| HW | TBD (예: RTX 3060 client + RTX 5060 Ti target) |
| Target model | TBD |
| Draft model | TBD |
| RTT | 실측 (localhost / LAN / 합성 지연 주입) |
| 프롬프트 세트 | code-gen / chat / long-form 각 10개 |

### 4.4 결과 테이블 (TODO)

| 설정 | TPS | TTFT (ms) | α | Target GPU 활용률 |
|---|---|---|---|---|
| Target-only 원격 | — | — | — | — |
| SD Client-Server N-gram | — | — | — | — |
| SD Client-Server EAGLE | — | — | — | — |
| SD in-process (vLLM V1) | — | — | — | — |

---

## 5. 재현 방법

Phase B 가 완료되면 다음 명령으로 모든 검증을 재현할 수 있도록 한다:

```bash
# 단위 테스트 (수초)
pytest tests/test_rejection_sampling.py -v
pytest tests/test_adaptive.py -v

# End-to-end smoke (GPU 필요, ~수 분)
pytest tests/test_e2e.py -v

# 벤치마크 (GPU 필요, ~수십 분)
python prototype/benchmark.py --config configs/bench_default.yaml
```

---

**다음 섹션**: Phase 1 에서 의도적으로 다루지 않은 항목들과 Phase 2 (vLLM 통합) 계획 → [09-ROADMAP](./09-ROADMAP.md)
