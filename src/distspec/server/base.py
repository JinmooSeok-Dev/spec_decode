# SPDX-License-Identifier: Apache-2.0
"""Abstract verifier interface.

``BaseVerifier`` is the contract that :class:`distspec.server.TargetServer`
depends on. Concrete implementations plug in a specific model backend:

  * :class:`distspec.server.hf_verifier.HfVerifier` — HuggingFace ``transformers``
    (Phase 1 reference implementation).
  * :class:`distspec.server.vllm_verifier.VllmVerifier` — vLLM ``LLMEngine``
    (Phase 2; stub).

Separating the interface lets Phase 2 swap the model execution layer without
changing the ZMQ server, protocol, or client code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..common.protocol import SamplingParams, VerifyOutput

if TYPE_CHECKING:
    import torch


@dataclass
class BatchRequest:
    """A single entry in a batched verification call.

    Moved here from :mod:`hf_verifier` so every ``BaseVerifier`` backend shares
    the shape.
    """

    client_id: str
    request_id: str
    draft_tokens: list[int]
    draft_probs: torch.Tensor | None
    context_tokens: list[int]
    sampling_params: SamplingParams


class BaseVerifier(ABC):
    """Abstract backend for draft-token verification.

    A verifier holds the target model (or an engine managing it), performs a
    forward pass over ``context + draft`` tokens, applies rejection sampling,
    and returns the set of accepted tokens plus a bonus token. It also owns
    whatever KV-cache state the backend requires.
    """

    @property
    @abstractmethod
    def tokenizer(self):
        """Return the backend's tokenizer. Must be attribute-compatible with a
        HuggingFace tokenizer (``eos_token_id``, ``encode``, ``decode``)."""

    @abstractmethod
    def verify(
        self,
        draft_tokens: list[int],
        draft_probs: torch.Tensor | None,
        context_tokens: list[int],
        sampling_params: SamplingParams,
    ) -> VerifyOutput:
        """Verify a draft sequence against the target distribution.

        Args:
            draft_tokens: The draft tokens to verify.
            draft_probs: Per-position draft distributions, or ``None`` if the
                proposer does not produce them (e.g. N-gram). In the latter
                case the verifier should assume a uniform draft distribution.
            context_tokens: Prompt + previously generated tokens. The verifier
                may use its own KV cache to avoid re-running forward on the
                full prefix each call.
            sampling_params: Sampling configuration.

        Returns:
            A :class:`VerifyOutput` with accepted tokens, an optional bonus
            token, and optional hidden states for EAGLE-style follow-up.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear any per-request cached state (e.g. KV cache)."""

    def verify_batch(self, requests: list[BatchRequest]) -> list[VerifyOutput]:
        """Verify a batch of independent requests.

        Default implementation delegates to :meth:`verify` one request at a
        time, which is correct but does not exploit GPU batching. Concrete
        backends may override this with a single padded forward pass; see
        :meth:`HfVerifier.verify_batch`.
        """
        return [
            self.verify(
                r.draft_tokens,
                r.draft_probs,
                r.context_tokens,
                r.sampling_params,
            )
            for r in requests
        ]
