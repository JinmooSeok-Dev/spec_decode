# SPDX-License-Identifier: Apache-2.0
"""Offline N-gram proposer demo — no server required.

Shows that :class:`NgramDraftProposer` produces sensible draft tokens on a
repeating token sequence without any model or network traffic.

Run::

    PYTHONPATH=src python examples/offline_ngram.py
"""

from __future__ import annotations

from distspec.client import NgramDraftProposer


def main() -> int:
    proposer = NgramDraftProposer(num_speculative_tokens=5, ngram_window=4)

    # A repeating pattern [1, 2, 3, 4, 5] — the proposer should detect it.
    context = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3]
    out = proposer.propose(context)

    print(f"context          : {context}")
    print(f"draft tokens     : {out.draft_tokens}")
    print(f"confidence       : {out.confidence_scores}")

    # Sanity: the proposer should find tokens after the last [1, 2, 3].
    assert out.draft_tokens, "expected non-empty draft on a repeating pattern"
    assert out.draft_tokens[:2] == [4, 5], (
        f"expected draft to start with [4, 5], got {out.draft_tokens}"
    )
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
