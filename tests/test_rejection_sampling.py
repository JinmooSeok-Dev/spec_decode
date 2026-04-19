# SPDX-License-Identifier: Apache-2.0
"""Statistical tests for :class:`RejectionSampler`.

Empirically verifies the distribution-preserving property proved in
``docs/06-VERIFICATION.md § 3.1``: the aggregate distribution of
(accepted ∪ recovered) tokens must equal the target distribution ``p``.

Runs on CPU, requires torch + scipy.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
scipy_stats = pytest.importorskip("scipy.stats")

from distspec.common.protocol import SamplingParams  # noqa: E402
from distspec.server.hf_verifier import RejectionSampler  # noqa: E402


def _random_dist(rng: torch.Generator, vocab: int, temperature: float = 1.0) -> torch.Tensor:
    logits = torch.randn(vocab, generator=rng) / temperature
    return torch.softmax(logits, dim=-1)


def _chi_square_same(
    empirical_counts: torch.Tensor,
    expected_probs: torch.Tensor,
    min_expected: float = 5.0,
) -> tuple[float, float]:
    """Return (chi², p-value) for ``empirical_counts`` vs ``expected_probs``.

    Pools bins with expected count below ``min_expected`` into a single bin to
    keep the chi-squared approximation valid.
    """
    total = int(empirical_counts.sum().item())
    expected_counts = expected_probs * total

    keep = expected_counts >= min_expected
    obs_kept = empirical_counts[keep]
    exp_kept = expected_counts[keep]

    pooled_obs = empirical_counts[~keep].sum()
    pooled_exp = expected_counts[~keep].sum()

    if pooled_exp >= min_expected:
        obs_kept = torch.cat([obs_kept, pooled_obs.unsqueeze(0)])
        exp_kept = torch.cat([exp_kept, pooled_exp.unsqueeze(0)])

    chi2, p_value = scipy_stats.chisquare(
        f_obs=obs_kept.numpy(),
        f_exp=exp_kept.numpy(),
    )
    return float(chi2), float(p_value)


def _sample_one_token(
    sampler: RejectionSampler,
    p: torch.Tensor,
    q: torch.Tensor,
    rng: torch.Generator,
) -> int:
    """Draw one token from a 1-draft-token rejection sample.

    The sampler returns either the accepted draft token, or a bonus token
    sampled from the recovered distribution (or from target, if draft fully
    covers target at the unaccepted mass). In either case the returned token
    is distributed according to ``p``.
    """
    # Sample a single draft token from q.
    draft_token = int(torch.multinomial(q, 1, generator=rng).item())

    # Verifier consumes target "logits". We'll pass log-probs as logits (softmax
    # recovers p exactly) and a second row so len(target_logits) > len(draft).
    target_logits = torch.stack([torch.log(p + 1e-30), torch.log(p + 1e-30)])

    out = sampler.forward(
        target_logits=target_logits,
        draft_tokens=[draft_token],
        draft_probs=q.unsqueeze(0),
        sampling_params=SamplingParams(temperature=1.0, top_p=1.0, top_k=-1),
    )
    if out.accepted_tokens:
        return int(out.accepted_tokens[0])
    assert out.bonus_token is not None
    return int(out.bonus_token)


@pytest.mark.parametrize(
    "vocab, temp_q",
    [
        (32, 1.0),   # moderately mixed q close to p
        (32, 2.0),   # softer q (broader)
        (32, 0.5),   # sharper q (can diverge from p)
    ],
)
def test_random_mode_preserves_target_distribution(vocab: int, temp_q: float):
    """χ² goodness-of-fit must not reject H0 (empirical == target)."""
    rng = torch.Generator().manual_seed(0xC0FFEE)

    p = _random_dist(rng, vocab, temperature=1.0)
    q = _random_dist(rng, vocab, temperature=temp_q)

    sampler = RejectionSampler(vocab_size=vocab, device="cpu")
    n_samples = 20_000

    counts = torch.zeros(vocab)
    for _ in range(n_samples):
        tok = _sample_one_token(sampler, p, q, rng)
        counts[tok] += 1

    chi2, p_value = _chi_square_same(counts, p)
    # At n=20000 we require p > 0.01 — i.e. cannot reject H0 at 1% significance.
    assert p_value > 0.01, (
        f"χ²={chi2:.2f}, p-value={p_value:.4f} — empirical distribution "
        f"diverges from target"
    )


def test_random_mode_identical_distributions():
    """When p == q every draft should be accepted (up to numeric noise)."""
    vocab = 16
    rng = torch.Generator().manual_seed(7)
    p = _random_dist(rng, vocab)
    q = p.clone()

    sampler = RejectionSampler(vocab_size=vocab, device="cpu")
    n = 2000
    accepted = 0
    for _ in range(n):
        draft_token = int(torch.multinomial(q, 1, generator=rng).item())
        target_logits = torch.stack([torch.log(p + 1e-30), torch.log(p + 1e-30)])
        out = sampler.forward(
            target_logits=target_logits,
            draft_tokens=[draft_token],
            draft_probs=q.unsqueeze(0),
            sampling_params=SamplingParams(temperature=1.0),
        )
        if out.accepted_tokens:
            accepted += 1
    # Accept rate should be ~100%.
    assert accepted / n > 0.98


def test_greedy_mode_matches_argmax_chain():
    """Greedy: accept while draft matches target argmax; bonus = first mismatch argmax."""
    vocab = 8
    # Construct target logits s.t. argmax at each row is position [3, 5, 1].
    target_logits = torch.full((3, vocab), -10.0)
    for row, col in enumerate((3, 5, 1)):
        target_logits[row, col] = 1.0

    sampler = RejectionSampler(vocab_size=vocab, device="cpu")
    params = SamplingParams(temperature=0.0)  # greedy

    # Case 1: all match → accept 2, bonus = target argmax at row 2 (= 1).
    out = sampler.forward(
        target_logits=target_logits,
        draft_tokens=[3, 5],
        draft_probs=None,
        sampling_params=params,
    )
    assert out.accepted_tokens == [3, 5]
    assert out.bonus_token == 1

    # Case 2: mismatch at position 1 → accept 1 (=3), bonus = target argmax at row 1 (=5).
    out = sampler.forward(
        target_logits=target_logits,
        draft_tokens=[3, 7],
        draft_probs=None,
        sampling_params=params,
    )
    assert out.accepted_tokens == [3]
    assert out.bonus_token == 5

    # Case 3: mismatch at position 0 → accept 0, bonus = target argmax at row 0 (=3).
    out = sampler.forward(
        target_logits=target_logits,
        draft_tokens=[0, 5],
        draft_probs=None,
        sampling_params=params,
    )
    assert out.accepted_tokens == []
    assert out.bonus_token == 3
