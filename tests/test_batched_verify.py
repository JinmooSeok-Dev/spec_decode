# SPDX-License-Identifier: Apache-2.0
"""Tests for :meth:`HfVerifier.verify_batch`.

Uses a hand-rolled toy causal LM (no real HF model) to exercise the padding,
attention-mask, and per-request slicing logic without paying the cost of
downloading a real checkpoint. The model is a deterministic linear layer so
that both per-request ``verify()`` and batched ``verify_batch()`` are
comparable.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from distspec.common.protocol import SamplingParams  # noqa: E402
from distspec.server.base import BatchRequest  # noqa: E402
from distspec.server.hf_verifier import HfVerifier  # noqa: E402

VOCAB = 16
HIDDEN = 8


class ToyCausalLM(torch.nn.Module):
    """A tiny deterministic causal LM: embed → single linear projection → logits.

    The 'logits' at each position are a function of **only** the token at that
    position (no real causal attention), which is enough for the verify_batch
    test — we only care that the batched path extracts the same per-position
    logits as the per-request path, modulo padding semantics.
    """

    def __init__(self, vocab: int = VOCAB, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab, bias=False)
        self.config = SimpleNamespace(vocab_size=vocab)

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        return_dict=True,
        output_hidden_states=False,
    ):
        h = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        logits = self.head(h)
        return SimpleNamespace(
            logits=logits,
            past_key_values=None,
            hidden_states=(h,) if output_hidden_states else None,
        )

    def get_input_embeddings(self):
        return self.embed


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text):
        return [2, 3, 4]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def _build_verifier() -> HfVerifier:
    v = HfVerifier.__new__(HfVerifier)
    v.model_name = "toy"
    v.device = "cpu"
    v.tensor_parallel_size = 1
    v.gpu_memory_utilization = 0.9
    v._model = ToyCausalLM()
    v._model.eval()
    v._tokenizer = ToyTokenizer()
    v._rejection_sampler = None
    v._kv_cache = None
    return v


def test_verify_batch_matches_per_request_greedy():
    v = _build_verifier()
    params = SamplingParams(temperature=0.0, max_tokens=32)

    reqs = [
        BatchRequest(
            client_id="c",
            request_id=f"r{i}",
            draft_tokens=[5, 6, 7][: 1 + i % 3],
            draft_probs=None,
            context_tokens=[2, 3, 4, 5][: 2 + i % 3],
            sampling_params=params,
        )
        for i in range(4)
    ]

    # Per-request baseline.
    per_req = []
    for r in reqs:
        v.reset()
        per_req.append(
            v.verify(r.draft_tokens, r.draft_probs, r.context_tokens, r.sampling_params)
        )

    # Batched path.
    v.reset()
    batched = v.verify_batch(reqs)

    assert len(batched) == len(per_req)
    for single, bat in zip(per_req, batched):
        # For the toy LM with no attention, padding shouldn't affect the
        # attended positions — accepted tokens and bonus should match.
        assert single.accepted_tokens == bat.accepted_tokens
        assert single.bonus_token == bat.bonus_token


def test_verify_batch_empty_returns_empty():
    v = _build_verifier()
    assert v.verify_batch([]) == []


def test_verify_batch_respects_request_independence():
    """Different requests in the same batch must not interfere."""
    v = _build_verifier()
    params = SamplingParams(temperature=0.0)

    # Two intentionally different-length requests.
    r1 = BatchRequest(
        client_id="c",
        request_id="r1",
        draft_tokens=[5],
        draft_probs=None,
        context_tokens=[2, 3],
        sampling_params=params,
    )
    r2 = BatchRequest(
        client_id="c",
        request_id="r2",
        draft_tokens=[6, 7, 8],
        draft_probs=None,
        context_tokens=[2, 3, 4, 5, 6],
        sampling_params=params,
    )

    out_batch = v.verify_batch([r1, r2])
    out_single_r1 = v.verify(r1.draft_tokens, None, r1.context_tokens, params)
    v.reset()
    out_single_r2 = v.verify(r2.draft_tokens, None, r2.context_tokens, params)

    assert out_batch[0].accepted_tokens == out_single_r1.accepted_tokens
    assert out_batch[0].bonus_token == out_single_r1.bonus_token
    assert out_batch[1].accepted_tokens == out_single_r2.accepted_tokens
    assert out_batch[1].bonus_token == out_single_r2.bonus_token
