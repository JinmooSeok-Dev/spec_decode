# distspec — Distributed Speculative Decoding

[![ci](https://img.shields.io/badge/ci-github%20actions-blue)](./.github/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache%202.0-green)](./LICENSE)
[![status](https://img.shields.io/badge/status-alpha-orange)](#status)

Client-Server reference implementation of **Speculative Decoding** for LLM
serving. The draft proposer runs on the client; the target model verifies
draft tokens on the server. Communication is over ZMQ with a msgpack-based
protocol, and the system is designed to keep output distribution identical
to the target model (Rejection Sampling is lossless — see
[docs/VERIFICATION.md](./docs/VERIFICATION.md)).

## Why distspec?

Naive remote inference pays **one network round-trip per generated token**.
With RTT = 50 ms and 256 output tokens, that is 12.8 s of pure network
overhead. Speculative Decoding amortizes this by confirming several tokens
per round-trip. See [docs/PROBLEM.md](./docs/PROBLEM.md) for the
quantitative motivation.

## Status

- **Phase 1 (current):** Pure PyTorch + HuggingFace reference. All message
  flow, proposers (N-gram / Suffix / EAGLE-lite), Rejection Sampling,
  Adaptive K, and Fault-Tolerant FSM are implemented.
- **Phase 2 (in progress):** `VllmVerifier` is implemented and wired up —
  select it via `--backend vllm` or `ServerConfig.backend="vllm"`. Greedy
  rejection works end-to-end; random-mode rejection sampling is tracked
  in [docs/ROADMAP.md](./docs/ROADMAP.md). The client, protocol, and
  serving loop are unchanged.

## Quick Start

### Install

Create a virtual environment and install the package. Pick the extras you
need — they are additive and can be combined (e.g. `".[torch,dev]"`).

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Core only (offline examples + CPU benchmarks work, no model runtime):
pip install -e .

# HuggingFace backend (Phase 1 reference — CPU or GPU):
pip install -e ".[torch]"

# vLLM backend (Phase 2 — requires a CUDA GPU):
pip install -e ".[vllm]"

# Development + test tooling (pytest, ruff, scipy for χ² tests):
pip install -e ".[dev]"
```

After installation, the module is importable as `distspec` and a
`distspec-server` console script is available — no `PYTHONPATH=src`
required.

### Run without a server (N-gram proposer only)

```bash
python examples/offline_ngram.py
# context          : [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3]
# draft tokens     : [4, 5, 1, 2, 3]
```

### Run the adaptive-K controller demo

```bash
python examples/adaptive_k.py
# Vary acceptance rate at RTT=100ms:
#   alpha=0.2 -> K*=2
#   alpha=0.9 -> K*=7
```

### Run a real client/server session

Two backends are available; pick one with `--backend`:

```bash
# Terminal 1 — HuggingFace backend (Phase 1 reference):
distspec-server --backend hf \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --listen-address 0.0.0.0:8000

# Terminal 1 alternative — vLLM backend (Phase 2, requires GPU + [vllm] extra):
distspec-server --backend vllm \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --listen-address 0.0.0.0:8000

# For the full vLLM-backend walkthrough (GPU selection, one-shot script,
# tuning knobs, troubleshooting), see docs/VLLM_BACKEND.md.

# Terminal 2 — client (same either way):
distspec-client --prompt "Hello, world." --server localhost:8000
```

Both are regular Python entry points, so the module form works too:

```bash
python -m distspec.server.target_server --backend vllm --model gpt2
python -m distspec.client.cli --prompt "Hello" --server 127.0.0.1:8000
```

### Run the test suite

```bash
# CI-safe tests (unit + stat, no GPU, no model downloads):
pytest -m "not slow and not vllm"

# End-to-end with a tiny HuggingFace model (downloads sshleifer/tiny-gpt2):
pytest -m slow

# vLLM-backed tests (requires a CUDA GPU and the [vllm] extra):
pytest -m vllm
```

## Architecture

```
┌─── Client ────────────────────┐     ZMQ      ┌──── Server ──────────┐
│ DraftProposer (N-gram/EAGLE)  │              │ TargetServer (ROUTER)│
│        │                      │              │        │             │
│        ▼                      │  msgpack     │        ▼             │
│ DraftClient (DEALER) ─────────┼──────────────▶ BaseVerifier         │
│   + Adaptive K controller     │   ◀─────────  │   + RejectionSampler │
│   + Fault-tolerant FSM        │              │   (HfVerifier /      │
│                               │              │    VllmVerifier)     │
└───────────────────────────────┘              └──────────────────────┘
```

Component details: [docs/DESIGN.md](./docs/DESIGN.md).

## Documentation

Read top-down — each document links to the next.

1. [PROBLEM](./docs/PROBLEM.md) — why distributed SD, quantitatively.
2. [SCENARIOS](./docs/SCENARIOS.md) — edge / shared / unstable-network.
3. [ALGORITHMS](./docs/ALGORITHMS.md) — variants compared; client-server chosen.
4. [DESIGN](./docs/DESIGN.md) — components, protocol, ZMQ topology.
5. [DRAFT_METHODS](./docs/DRAFT_METHODS.md) — N-gram / Suffix / EAGLE.
6. [VERIFICATION](./docs/VERIFICATION.md) — Rejection Sampling + confidence.
7. [ADAPTIVE_CONTROL](./docs/ADAPTIVE_CONTROL.md) — adaptive K and the FSM.
8. [EVALUATION](./docs/EVALUATION.md) — accuracy + performance tests.
9. [ROADMAP](./docs/ROADMAP.md) — Phase 1 close-out and Phase 2 plan.

How-to:
- [VLLM_BACKEND](./docs/VLLM_BACKEND.md) — install, select a GPU, run a
  one-shot script or a server/client session against the vLLM verifier,
  tuning knobs, and a troubleshooting table.

## Layout

```
distspec/
├── docs/          # Public documentation (stories 01-09).
├── src/distspec/  # Package source.
│   ├── common/    # Protocol, config, sampling utilities.
│   ├── client/    # Draft proposers and streaming clients.
│   └── server/    # Verifier backends + ZMQ serving loop.
├── examples/      # Runnable demos; each one is stand-alone.
├── benchmarks/    # Micro-benchmarks, CI-safe by default.
├── tests/         # Pytest suite (see pyproject markers).
└── study/         # Personal study notes — gitignored, not part of the repo.
```

## Running benchmarks

Benchmarks emit human-readable output by default; pass `--json` to get
machine-readable output for CI:

```bash
python benchmarks/bench_ngram.py --json
python benchmarks/bench_protocol.py --json
python benchmarks/bench_sampling.py --json   # torch-optional; skips if absent
```

A regression floor is enforced on `bench_ngram.py` (see `--regression-floor`).
See [docs/EVALUATION.md](./docs/EVALUATION.md) for the planned full
end-to-end measurement matrix.

## Contributing

Contributions welcome — particularly on the Phase 2 roadmap items and the
EAGLE draft-head retraining track. Open an issue or a PR against `main`.
By contributing you agree to license your work under Apache 2.0.

## License

Apache 2.0. See [LICENSE](./LICENSE).

## Acknowledgements

The architecture borrows from vLLM's speculative decoding implementation
(the `NgramProposer` / `EagleProposer` / `RejectionSampler` patterns). See
[docs/ALGORITHMS.md § References](./docs/ALGORITHMS.md#references)
for the full bibliography.
