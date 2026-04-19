# SPDX-License-Identifier: Apache-2.0
"""vLLM-backed verifier (Phase 2 stub).

Placeholder for the Phase 2 integration where target model execution is
delegated to vLLM's ``LLMEngine``. The mapping is documented in
``docs/09-ROADMAP.md`` § 2.2:

    HF past_key_values    →  vLLM KVCacheManager / block table
    HF forward(...)       →  LLMEngine.add_request + step loop
    self-written sampler  →  vllm.v1.sample.rejection_sampler

The class shape below is intentionally similar to :class:`HfVerifier` so the
``TargetServer`` can swap backends without code changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..common.protocol import SamplingParams, VerifyOutput
from .base import BaseVerifier

if TYPE_CHECKING:
    import torch


class VllmVerifier(BaseVerifier):
    """Verifier backed by vLLM's ``LLMEngine`` (not yet implemented)."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        **engine_kwargs,
    ) -> None:
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.engine_kwargs = engine_kwargs
        raise NotImplementedError(
            "VllmVerifier is a Phase 2 stub. See docs/09-ROADMAP.md § 2."
        )

    @property
    def tokenizer(self):
        raise NotImplementedError

    def verify(
        self,
        draft_tokens: list[int],
        draft_probs: torch.Tensor | None,
        context_tokens: list[int],
        sampling_params: SamplingParams,
    ) -> VerifyOutput:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
