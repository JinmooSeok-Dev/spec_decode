# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke tests against a real HuggingFace model pair.

Slow: downloads a small model the first time it runs. Excluded from the
default CI job via the ``slow`` marker. Invoke manually with::

    pytest -m slow tests/test_e2e.py
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from distspec.client import NgramDraftProposer  # noqa: E402
from distspec.common.protocol import SamplingParams  # noqa: E402
from distspec.server.hf_verifier import HfVerifier  # noqa: E402

SMALL_MODEL = "sshleifer/tiny-gpt2"  # ~1MB test fixture widely used in HF CI


@pytest.mark.slow
def test_hf_verifier_greedy_matches_target_only():
    """SD with temperature=0 must be token-identical to plain greedy decoding."""
    verifier = HfVerifier(model_name=SMALL_MODEL, device="cpu")
    _ = verifier.tokenizer  # trigger load

    prompt = verifier.tokenizer.encode("Hello")
    params = SamplingParams(temperature=0.0, max_tokens=10)

    # Reference: generate 10 tokens greedily from the target alone.
    inputs = torch.tensor([prompt], dtype=torch.long)
    with torch.no_grad():
        ref = verifier.model.generate(inputs, max_new_tokens=10, do_sample=False)
    ref_ids = ref[0, len(prompt):].tolist()

    # SD path: use an N-gram proposer (may propose nothing on a short prompt,
    # in which case we fall back to letting the verifier pick bonus = target
    # argmax). We loop manually emulating what the client does.
    verifier.reset()
    proposer = NgramDraftProposer(num_speculative_tokens=3, min_match_length=2)

    context = list(prompt)
    sd_ids: list[int] = []
    while len(sd_ids) < 10:
        draft_out = proposer.propose(context, sampling_params=params)
        drafts = draft_out.draft_tokens or [context[-1]]  # placeholder to force verify

        out = verifier.verify(
            draft_tokens=drafts[:1] if not draft_out.draft_tokens else drafts,
            draft_probs=None,
            context_tokens=context,
            sampling_params=params,
        )
        # accepted + bonus (bonus may be None if we hit max)
        new_tokens = list(out.accepted_tokens)
        if out.bonus_token is not None:
            new_tokens.append(out.bonus_token)
        if not new_tokens:
            break

        take = new_tokens[: max(1, 10 - len(sd_ids))]
        sd_ids.extend(take)
        context.extend(take)

    assert sd_ids[: len(ref_ids)] == ref_ids, (
        f"SD greedy diverges from target-only greedy:\n"
        f"  ref: {ref_ids}\n"
        f"  sd : {sd_ids}"
    )


@pytest.mark.slow
def test_hf_verifier_batch_matches_sequential_greedy():
    """Batched verify_batch must match sequential verify for the same inputs."""
    from distspec.server.base import BatchRequest

    verifier = HfVerifier(model_name=SMALL_MODEL, device="cpu")
    _ = verifier.tokenizer

    params = SamplingParams(temperature=0.0)
    prompts = [[1, 2, 3], [1, 2, 3, 4], [5, 6]]
    drafts = [[7, 8], [9], [10, 11, 12]]

    reqs = [
        BatchRequest(
            client_id="c",
            request_id=f"r{i}",
            draft_tokens=d,
            draft_probs=None,
            context_tokens=p,
            sampling_params=params,
        )
        for i, (p, d) in enumerate(zip(prompts, drafts))
    ]

    # Per-request
    per_req = []
    for r in reqs:
        verifier.reset()
        per_req.append(
            verifier.verify(
                r.draft_tokens, r.draft_probs, r.context_tokens, r.sampling_params
            )
        )

    # Batched
    verifier.reset()
    batched = verifier.verify_batch(reqs)

    for single, bat in zip(per_req, batched):
        assert single.accepted_tokens == bat.accepted_tokens
        assert single.bonus_token == bat.bonus_token
