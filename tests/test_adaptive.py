# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :class:`AdaptiveSpeculationController`.

Verifies monotonicity of the chosen K* with respect to:
  * the observed acceptance rate α (higher → larger K), and
  * the observed RTT (larger → larger K).

These are the invariants documented in ``docs/ADAPTIVE_CONTROL.md § 1.4``.
"""

from __future__ import annotations

import pytest

from distspec.client import AdaptiveSpeculationController


def _drive(
    alpha: float,
    rtt_s: float,
    n_rounds: int = 20,
    min_k: int = 1,
    max_k: int = 10,
) -> int:
    """Run ``n_rounds`` synthetic observations and return the chosen K*."""
    ctrl = AdaptiveSpeculationController(
        min_spec_tokens=min_k,
        max_spec_tokens=max_k,
        decode_time_estimate=0.01,
    )
    k_used = 5
    accepted = int(round(alpha * k_used))
    for _ in range(n_rounds):
        ctrl.record_result(rtt=rtt_s, num_draft=k_used, num_accepted=accepted)
    return ctrl.current_k


def test_warmup_returns_max_before_enough_history():
    ctrl = AdaptiveSpeculationController(
        min_spec_tokens=1,
        max_spec_tokens=8,
        decode_time_estimate=0.01,
    )
    # History length < 5 → should stay at the max.
    for _ in range(3):
        ctrl.record_result(rtt=0.1, num_draft=5, num_accepted=4)
    assert ctrl.current_k == 8


def test_collapses_to_min_when_acceptance_is_below_floor():
    # α well below 10% should clamp to min.
    k = _drive(alpha=0.05, rtt_s=0.1, min_k=2, max_k=10)
    assert k == 2


@pytest.mark.parametrize("alpha", [0.2, 0.5, 0.7, 0.9])
def test_k_increases_with_acceptance(alpha: float):
    """Higher α should not yield a smaller K than the lowest-α run."""
    low = _drive(alpha=0.2, rtt_s=0.1)
    this = _drive(alpha=alpha, rtt_s=0.1)
    assert this >= low, (
        f"alpha={alpha}: K*={this} < K*(alpha=0.2)={low}"
    )


def test_k_is_monotonic_in_alpha():
    ks = [_drive(alpha=a, rtt_s=0.1) for a in (0.2, 0.5, 0.7, 0.9)]
    for prev, nxt in zip(ks, ks[1:]):
        assert nxt >= prev, f"non-monotonic in α: {ks}"


def test_k_is_monotonic_in_rtt():
    ks = [_drive(alpha=0.7, rtt_s=rtt) for rtt in (0.001, 0.050, 0.200, 1.000)]
    for prev, nxt in zip(ks, ks[1:]):
        assert nxt >= prev, f"non-monotonic in RTT: {ks}"


def test_reset_clears_history():
    ctrl = AdaptiveSpeculationController(
        min_spec_tokens=1,
        max_spec_tokens=8,
        decode_time_estimate=0.01,
    )
    for _ in range(10):
        ctrl.record_result(rtt=0.1, num_draft=5, num_accepted=1)
    ctrl.reset()
    assert ctrl.current_k == 8  # back to max (warmup regime)
