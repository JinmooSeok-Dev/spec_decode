# distspec — Distributed Speculative Decoding

[![ci](https://img.shields.io/badge/ci-github%20actions-blue)](./.github/workflows/ci.yml)
[![license](https://img.shields.io/badge/license-Apache%202.0-green)](./LICENSE)
[![status](https://img.shields.io/badge/status-alpha-orange)](#status)

Client-Server reference implementation of **Speculative Decoding** for LLM
serving. The draft proposer runs on the client; the target model verifies
draft tokens on the server. Communication is over ZMQ with a msgpack-based
protocol, and the system is designed to keep output distribution identical
to the target model (Rejection Sampling is lossless — see
[docs/06-VERIFICATION.md](./docs/06-VERIFICATION.md)).

## Why distspec?

Naive remote inference pays **one network round-trip per generated token**.
With RTT = 50 ms and 256 output tokens, that is 12.8 s of pure network
overhead. Speculative Decoding amortizes this by confirming several tokens
per round-trip. See [docs/01-PROBLEM.md](./docs/01-PROBLEM.md) for the
quantitative motivation.

## Status

- **Phase 1 (current):** Pure PyTorch + HuggingFace reference. All message
  flow, proposers (N-gram / Suffix / EAGLE-lite), Rejection Sampling,
  Adaptive K, and Fault-Tolerant FSM are implemented.
- **Phase 2 (planned):** Replace the HF-based `HfVerifier` with a vLLM
  `LLMEngine` backend (see [docs/09-ROADMAP.md](./docs/09-ROADMAP.md)).
  The client, protocol, and serving loop stay the same.

## Quick Start

### Install

```bash
# Core dependencies only (offline examples + CPU benchmarks work):
pip install -e .

# HuggingFace backend (needed to run a real target server):
pip install -e ".[torch]"

# Dev tooling (pytest, ruff, scipy for χ² tests):
pip install -e ".[dev]"
```

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

```bash
# Terminal 1 — target server (GPU recommended):
python -m distspec.server.target_server \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --listen-address 0.0.0.0:8000

# Terminal 2 — client:
python examples/client_server.py --prompt "Hello, world." --server localhost:8000
```

After `pip install -e .` the server is also available as a console script:

```bash
distspec-server --model meta-llama/Llama-3.2-3B-Instruct
```

## Architecture

```
┌─── Client ────────────────────┐     ZMQ      ┌──── Server ──────────┐
│ DraftProposer (N-gram/EAGLE)  │              │ TargetServer (ROUTER)│
│        │                      │              │        │             │
│        ▼                      │  msgpack     │        ▼             │
│ DraftClient (DEALER) ─────────┼──────────────▶ BaseVerifier         │
│   + Adaptive K controller     │   ◀─────────  │   + RejectionSampler │
│   + Fault-tolerant FSM        │              │   (HfVerifier now;   │
│                               │              │    VllmVerifier later)│
└───────────────────────────────┘              └──────────────────────┘
```

Component details: [docs/04-DESIGN.md](./docs/04-DESIGN.md).

## Documentation

Read top-down — each document links to the next.

1. [01-PROBLEM](./docs/01-PROBLEM.md) — why distributed SD, quantitatively.
2. [02-SCENARIOS](./docs/02-SCENARIOS.md) — edge / shared / unstable-network.
3. [03-ALGORITHMS](./docs/03-ALGORITHMS.md) — variants compared; client-server chosen.
4. [04-DESIGN](./docs/04-DESIGN.md) — components, protocol, ZMQ topology.
5. [05-DRAFT_METHODS](./docs/05-DRAFT_METHODS.md) — N-gram / Suffix / EAGLE.
6. [06-VERIFICATION](./docs/06-VERIFICATION.md) — Rejection Sampling + confidence.
7. [07-ADAPTIVE_CONTROL](./docs/07-ADAPTIVE_CONTROL.md) — adaptive K and the FSM.
8. [08-EVALUATION](./docs/08-EVALUATION.md) — accuracy + performance tests.
9. [09-ROADMAP](./docs/09-ROADMAP.md) — Phase 1 close-out and Phase 2 plan.

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
See [docs/08-EVALUATION.md](./docs/08-EVALUATION.md) for the planned full
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
[docs/03-ALGORITHMS.md § References](./docs/03-ALGORITHMS.md#5-references)
for the full bibliography.
