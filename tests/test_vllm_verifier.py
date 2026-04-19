# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for :class:`VllmVerifier`.

Requires vLLM + a CUDA GPU. Downloads a small model on first run. Excluded
from the default CI job via the ``vllm`` marker; run manually with::

    pytest -m vllm tests/test_vllm_verifier.py
"""

from __future__ import annotations

import pytest

pytest.importorskip("vllm")
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("vLLM tests require a CUDA device", allow_module_level=True)

from distspec.common.protocol import SamplingParams  # noqa: E402
from distspec.server.base import BatchRequest  # noqa: E402
from distspec.server.vllm_verifier import VllmVerifier  # noqa: E402

SMALL_MODEL = "facebook/opt-125m"


@pytest.fixture(scope="module")
def verifier() -> VllmVerifier:
    v = VllmVerifier(
        model_name=SMALL_MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
        max_model_len=512,
        enforce_eager=True,
        dtype="float16",
    )
    yield v


@pytest.mark.vllm
def test_greedy_all_correct_draft_accepts_all(verifier: VllmVerifier):
    """If draft equals the target's own greedy continuation, all tokens accept."""
    from vllm import SamplingParams as VllmSP

    prompt_ids = verifier.tokenizer.encode("The quick brown")
    # Ask vLLM to greedily produce 3 tokens — these are exactly what the target
    # would want on its own.
    params = VllmSP(temperature=0.0, max_tokens=3, detokenize=False)
    out = verifier.llm.generate(
        prompts=[prompt_ids],
        sampling_params=params,
        use_tqdm=False,
    )
    target_tokens = list(out[0].outputs[0].token_ids)
    assert len(target_tokens) == 3

    # Feed exactly those tokens as draft; verifier should accept all three
    # and the bonus should be the target's 4th argmax.
    result = verifier.verify(
        draft_tokens=target_tokens,
        draft_probs=None,
        context_tokens=prompt_ids,
        sampling_params=SamplingParams(temperature=0.0),
    )

    assert result.accepted_tokens == target_tokens
    assert result.bonus_token is not None


@pytest.mark.vllm
def test_greedy_wrong_draft_rejects(verifier: VllmVerifier):
    """Deliberately wrong draft tokens should be rejected immediately."""
    prompt_ids = verifier.tokenizer.encode("Hello world")
    # Pick an implausibly specific token sequence.
    bogus_tokens = [100, 200, 300]  # unlikely to match greedy continuation

    result = verifier.verify(
        draft_tokens=bogus_tokens,
        draft_probs=None,
        context_tokens=prompt_ids,
        sampling_params=SamplingParams(temperature=0.0),
    )

    # Not every case guarantees zero accepts (target could by chance match),
    # but at least one should fail and the verifier must return a bonus.
    assert len(result.accepted_tokens) < len(bogus_tokens)
    assert result.bonus_token is not None


@pytest.mark.vllm
def test_batched_verify_matches_sequential(verifier: VllmVerifier):
    """verify_batch must agree with sequential verify() on each request."""
    from vllm import SamplingParams as VllmSP

    prompts = [
        verifier.tokenizer.encode("Hello world"),
        verifier.tokenizer.encode("The quick brown"),
    ]

    # Build consistent drafts: target's own greedy produces K tokens.
    params = VllmSP(temperature=0.0, max_tokens=2, detokenize=False)
    gens = verifier.llm.generate(
        prompts=prompts,
        sampling_params=params,
        use_tqdm=False,
    )
    drafts = [list(g.outputs[0].token_ids) for g in gens]

    reqs = [
        BatchRequest(
            client_id="c",
            request_id=f"r{i}",
            draft_tokens=d,
            draft_probs=None,
            context_tokens=p,
            sampling_params=SamplingParams(temperature=0.0),
        )
        for i, (p, d) in enumerate(zip(prompts, drafts))
    ]

    per_req = [verifier.verify(r.draft_tokens, None, r.context_tokens, r.sampling_params) for r in reqs]
    batched = verifier.verify_batch(reqs)

    for s, b in zip(per_req, batched):
        assert s.accepted_tokens == b.accepted_tokens
        assert s.bonus_token == b.bonus_token


@pytest.mark.vllm
def test_random_mode_not_implemented(verifier: VllmVerifier):
    """Random-mode is scheduled for Phase 2 S2, not S1."""
    with pytest.raises(NotImplementedError, match="greedy"):
        verifier.verify(
            draft_tokens=[1],
            draft_probs=None,
            context_tokens=[2, 3],
            sampling_params=SamplingParams(temperature=0.7),
        )
