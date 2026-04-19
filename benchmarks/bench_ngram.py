# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark for the N-gram proposer — CPU only, CI-safe.

Measures ``propose()`` throughput on a synthetic repeating-pattern context.

Run::

    PYTHONPATH=src python benchmarks/bench_ngram.py [--full] [--json]
"""

from __future__ import annotations

import random

from _common import build_parser, emit, stopwatch
from distspec.client import NgramDraftProposer

REGRESSION_MIN_OPS_PER_SEC = 500.0  # floor for CI; raise if hardware improves.


def run(quick: bool) -> dict:
    proposer = NgramDraftProposer(num_speculative_tokens=5, ngram_window=4)

    # Build a repeating-pattern context.
    pattern = [random.randint(0, 32000) for _ in range(8)]
    n_repeats = 8 if quick else 64
    context = pattern * n_repeats

    iterations = 200 if quick else 2000

    with stopwatch() as t:
        for _ in range(iterations):
            proposer.propose(context)

    ops_per_sec = iterations / t["seconds"]
    return {
        "name": "bench_ngram",
        "context_len": len(context),
        "iterations": iterations,
        "seconds": t["seconds"],
        "ops_per_sec": ops_per_sec,
    }


def main() -> int:
    parser = build_parser(__doc__ or "")
    parser.add_argument(
        "--regression-floor",
        type=float,
        default=REGRESSION_MIN_OPS_PER_SEC,
        help="Fail if ops/sec falls below this (for CI).",
    )
    args = parser.parse_args()

    result = run(quick=args.quick)
    emit(dict(result), as_json=args.json)

    if result["ops_per_sec"] < args.regression_floor:
        print(
            f"REGRESSION: ops_per_sec={result['ops_per_sec']:.1f} "
            f"< floor={args.regression_floor}",
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
