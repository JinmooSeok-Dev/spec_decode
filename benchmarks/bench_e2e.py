# SPDX-License-Identifier: Apache-2.0
"""End-to-end SD benchmark — requires GPU + a running target server.

Measures tokens/sec, TTFT and acceptance rate for a real model pair. Excluded
from CI; intended for manual performance tracking. See
``docs/08-EVALUATION.md`` for the planned measurement matrix.

Not yet implemented — the Phase B smoke tests must land first to ensure the
client-server loop runs on real models.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "bench_e2e: not yet implemented — pending Phase B smoke tests.\n"
        "See docs/08-EVALUATION.md § 4 for the planned design.",
        file=sys.stderr,
    )
    return 77  # conventional "skip" exit code


if __name__ == "__main__":
    raise SystemExit(main())
