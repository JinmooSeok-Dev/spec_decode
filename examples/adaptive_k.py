# SPDX-License-Identifier: Apache-2.0
"""Adaptive speculation-length controller demo.

Feeds synthetic ``(rtt, num_draft, num_accepted)`` observations into
:class:`AdaptiveSpeculationController` and prints the chosen K. Demonstrates
monotonicity described in ``docs/ADAPTIVE_CONTROL.md``.

Run::

    PYTHONPATH=src python examples/adaptive_k.py
"""

from __future__ import annotations

from distspec.client import AdaptiveSpeculationController


def simulate(alpha: float, rtt: float, n_rounds: int = 20) -> int:
    ctrl = AdaptiveSpeculationController(
        min_spec_tokens=1,
        max_spec_tokens=10,
        decode_time_estimate=0.01,
    )
    k = 5
    accepted = int(alpha * k)
    for _ in range(n_rounds):
        ctrl.record_result(rtt=rtt, num_draft=k, num_accepted=accepted)
    return ctrl.current_k


def main() -> int:
    print("Vary acceptance rate at RTT=100ms:")
    for alpha in (0.2, 0.5, 0.7, 0.9):
        k = simulate(alpha=alpha, rtt=0.1)
        print(f"  alpha={alpha:.1f} -> K*={k}")

    print("\nVary RTT at alpha=0.7:")
    for rtt_ms in (1, 50, 200, 1000):
        k = simulate(alpha=0.7, rtt=rtt_ms / 1000.0)
        print(f"  RTT={rtt_ms:>4}ms -> K*={k}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
