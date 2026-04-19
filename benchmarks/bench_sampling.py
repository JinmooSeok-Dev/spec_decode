# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark for top-k / top-p filters — CPU-only, CI-safe.

Measures throughput of :func:`distspec.common.sampling.apply_sampling_filters`
on a synthetic logits tensor. Skipped (with a clear message) if torch is not
available.

Run::

    PYTHONPATH=src python benchmarks/bench_sampling.py [--full] [--json]
"""

from __future__ import annotations

from _common import build_parser, emit, stopwatch


def run(quick: bool) -> dict:
    try:
        import torch
    except ImportError:
        return {"name": "bench_sampling", "skipped": "torch not available"}

    from distspec.common.sampling import apply_sampling_filters

    vocab = 32_000
    batch = 8 if quick else 64
    iterations = 500 if quick else 5000

    logits = torch.randn(batch, vocab)

    with stopwatch() as t:
        for _ in range(iterations):
            apply_sampling_filters(logits.clone(), top_k=50, top_p=0.9)

    return {
        "name": "bench_sampling",
        "batch": batch,
        "vocab": vocab,
        "iterations": iterations,
        "seconds": t["seconds"],
        "ops_per_sec": iterations / t["seconds"],
    }


def main() -> int:
    parser = build_parser(__doc__ or "")
    args = parser.parse_args()
    emit(run(quick=args.quick), as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
