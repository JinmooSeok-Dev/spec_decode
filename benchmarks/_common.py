# SPDX-License-Identifier: Apache-2.0
"""Benchmark helpers shared across scripts.

Each benchmark script should:

  * Accept ``--json`` for machine-readable output (for CI parsing).
  * Accept ``--quick`` (default) vs ``--full`` to control workload size.
  * Exit with code 0 on success, non-zero if a regression threshold is
    exceeded (thresholds are defined per script where applicable).
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON to stdout.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=True,
        help="Small workload suitable for CI (default).",
    )
    parser.add_argument(
        "--full",
        dest="quick",
        action="store_false",
        help="Larger workload for local measurements.",
    )
    return parser


@contextmanager
def stopwatch() -> Iterator[dict[str, float]]:
    """Measure wall-clock time in seconds."""
    timing: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield timing
    finally:
        timing["seconds"] = time.perf_counter() - start


def emit(result: dict[str, Any], as_json: bool) -> None:
    """Emit a result dict either as JSON or a human-readable block."""
    if as_json:
        print(json.dumps(result, sort_keys=True))
        return

    name = result.pop("name", "benchmark")
    print(f"== {name} ==")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"  {key:<24} {value:.6f}")
        else:
            print(f"  {key:<24} {value}")
