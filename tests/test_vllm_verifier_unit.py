# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :class:`VllmVerifier` rejection logic.

Covers the pure-Python verification path (`_verify_one` + `_argmax_from_logprobs`)
with synthetic vLLM-shaped outputs so the algorithm can be validated without
a real LLMEngine. End-to-end tests against a live vLLM backend live in
``test_vllm_verifier.py`` (``@pytest.mark.vllm``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from distspec.common.protocol import SamplingParams
from distspec.server.base import BatchRequest
from distspec.server.vllm_verifier import VllmVerifier


def _logprob(rank: int, logprob: float = -1.0) -> SimpleNamespace:
    """Build a minimal object that mimics ``vllm.sequence.Logprob``."""
    return SimpleNamespace(rank=rank, logprob=logprob, decoded_token=None)


def _make_prompt_logprobs(argmax_chain: list[int | None]) -> list[dict | None]:
    """Construct prompt_logprobs where position i has rank-1 = argmax_chain[i].

    ``None`` positions emit ``None`` (as vLLM does for position 0). Each entry
    is a dict {token_id -> Logprob}; we use two entries to match a realistic
    ``prompt_logprobs=2`` output.
    """
    out: list[dict | None] = []
    for argmax in argmax_chain:
        if argmax is None:
            out.append(None)
            continue
        out.append(
            {
                argmax: _logprob(rank=1, logprob=-0.1),
                # An arbitrary rank-2 entry so dict isn't size-1.
                (argmax + 1) % 50000: _logprob(rank=2, logprob=-2.0),
            }
        )
    return out


class _FakeOut:
    """Stand-in for a vLLM RequestOutput."""

    def __init__(self, prompt_logprobs, generated_token: int):
        self.prompt_logprobs = prompt_logprobs
        self.outputs = [SimpleNamespace(token_ids=(generated_token,))]


def _make_verifier() -> VllmVerifier:
    """Build a VllmVerifier without invoking ``__init__`` (no vLLM needed)."""
    return VllmVerifier.__new__(VllmVerifier)


def test_argmax_from_logprobs_rank1():
    v = _make_verifier()
    d = {7: _logprob(rank=1), 42: _logprob(rank=2)}
    assert v._argmax_from_logprobs(d) == 7


def test_argmax_from_logprobs_none():
    v = _make_verifier()
    assert v._argmax_from_logprobs(None) is None


def test_verify_one_all_accepted_takes_generated_as_bonus():
    """Whole draft matches target argmax → all accepted, bonus = generated."""
    v = _make_verifier()
    # context = [c0, c1, c2], draft = [d0, d1]
    # prompt_logprobs[0] = None (BOS), [1] = argmax@pos1, [2] = argmax@pos2,
    # [3] = argmax@pos3 (predicts d0), [4] = argmax@pos4 (predicts d1).
    prompt_logprobs = _make_prompt_logprobs([None, 99, 88, 100, 200])
    out = _FakeOut(prompt_logprobs=prompt_logprobs, generated_token=333)

    req = BatchRequest(
        client_id="c",
        request_id="r",
        draft_tokens=[100, 200],
        draft_probs=None,
        context_tokens=[1, 2, 3],
        sampling_params=SamplingParams(temperature=0.0),
    )
    result = v._verify_one(req, out)
    assert result.accepted_tokens == [100, 200]
    assert result.bonus_token == 333  # from generated, not prompt_logprobs


def test_verify_one_mid_mismatch_bonus_is_target_argmax():
    """First mismatch → accept prefix, bonus = target argmax at that position."""
    v = _make_verifier()
    prompt_logprobs = _make_prompt_logprobs([None, 0, 0, 100, 999])  # draft[1]=200 vs argmax=999
    out = _FakeOut(prompt_logprobs=prompt_logprobs, generated_token=-1)

    req = BatchRequest(
        client_id="c",
        request_id="r",
        draft_tokens=[100, 200],
        draft_probs=None,
        context_tokens=[1, 2, 3],
        sampling_params=SamplingParams(temperature=0.0),
    )
    result = v._verify_one(req, out)
    assert result.accepted_tokens == [100]
    assert result.bonus_token == 999


def test_verify_one_full_rejection():
    """Draft fails at position 0 → empty accept, bonus = target argmax[0]."""
    v = _make_verifier()
    prompt_logprobs = _make_prompt_logprobs([None, 0, 0, 55, 66])
    out = _FakeOut(prompt_logprobs=prompt_logprobs, generated_token=-1)

    req = BatchRequest(
        client_id="c",
        request_id="r",
        draft_tokens=[100, 200],
        draft_probs=None,
        context_tokens=[1, 2, 3],
        sampling_params=SamplingParams(temperature=0.0),
    )
    result = v._verify_one(req, out)
    assert result.accepted_tokens == []
    assert result.bonus_token == 55


def test_verify_one_missing_prompt_logprobs_returns_empty():
    v = _make_verifier()
    out = _FakeOut(prompt_logprobs=None, generated_token=-1)
    req = BatchRequest(
        client_id="c",
        request_id="r",
        draft_tokens=[100, 200],
        draft_probs=None,
        context_tokens=[1, 2, 3],
        sampling_params=SamplingParams(temperature=0.0),
    )
    result = v._verify_one(req, out)
    assert result.accepted_tokens == []
    assert result.bonus_token is None


def test_verify_batch_rejects_non_greedy():
    """Random mode is Phase 2 S2; must raise NotImplementedError."""
    v = _make_verifier()
    # Calling verify_batch on a non-empty requests list with temperature > 0:
    req = BatchRequest(
        client_id="c",
        request_id="r",
        draft_tokens=[1],
        draft_probs=None,
        context_tokens=[2, 3],
        sampling_params=SamplingParams(temperature=0.7),
    )
    with pytest.raises(NotImplementedError, match="greedy"):
        v.verify_batch([req])


def test_verify_batch_empty_returns_empty():
    v = _make_verifier()
    assert v.verify_batch([]) == []


def test_verify_one_handles_empty_draft():
    """Empty draft (N-gram on a non-repeating prompt) → bonus from generated."""
    v = _make_verifier()
    # prompt_logprobs has an entry per prompt position; with no drafts we only
    # need those covering the context (so len == len(context)).
    prompt_logprobs = _make_prompt_logprobs([None, 99, 88])  # ctx_len=3
    out = _FakeOut(prompt_logprobs=prompt_logprobs, generated_token=444)

    req = BatchRequest(
        client_id="c",
        request_id="r",
        draft_tokens=[],  # <-- empty
        draft_probs=None,
        context_tokens=[1, 2, 3],
        sampling_params=SamplingParams(temperature=0.0),
    )
    result = v._verify_one(req, out)
    assert result.accepted_tokens == []
    assert result.bonus_token == 444
