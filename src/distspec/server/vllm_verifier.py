# SPDX-License-Identifier: Apache-2.0
"""vLLM-backed verifier (Phase 2).

Delegates target execution to :class:`vllm.LLM`, so the server inherits
PagedAttention, continuous batching, and multi-GPU support from the vLLM
runtime. Only the verification logic lives here.

Approach
--------
For a verify request with context ``C`` (length ``L``) and drafts ``D``
(length ``K``), we build a single prompt ``C + D`` and ask vLLM to:

  * generate one extra token (``max_tokens=1``), and
  * return the top-k prompt logprobs at each input position
    (``prompt_logprobs=K``).

From ``prompt_logprobs[L + k]`` we extract the rank-1 (argmax) token that the
target would have generated at position ``L + k`` — this is the target's
prediction for ``D[k]``. Greedy rejection then compares each draft token
against this argmax and accepts the longest matching prefix. The bonus token
is the argmax of the first mismatch, or the single generated token when the
entire draft is accepted.

Scope
-----
* **Greedy only** in this version. ``sampling_params.temperature > 0`` raises
  :class:`NotImplementedError` — random-mode rejection sampling against vLLM
  distributions is tracked as Phase 2 S2 in ``docs/09-ROADMAP.md``.
* Batched verify is delegated to vLLM's scheduler: we hand it a list of
  prompts in one :meth:`vllm.LLM.generate` call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..common.protocol import SamplingParams, VerifyOutput
from .base import BaseVerifier, BatchRequest

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class VllmVerifier(BaseVerifier):
    """Verifier backed by vLLM's ``LLM`` engine (Phase 2)."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        enforce_eager: bool = True,
        dtype: str = "auto",
        **engine_kwargs,
    ) -> None:
        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError(
                "VllmVerifier requires the 'vllm' package. "
                "Install with: pip install -e '.[vllm]'"
            ) from exc

        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len

        logger.info("Loading vLLM target model: %s", model_name)
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            dtype=dtype,
            **engine_kwargs,
        )
        self._tokenizer = self.llm.get_tokenizer()

    @property
    def tokenizer(self):
        return self._tokenizer

    def verify(
        self,
        draft_tokens: list[int],
        draft_probs: torch.Tensor | None,
        context_tokens: list[int],
        sampling_params: SamplingParams,
    ) -> VerifyOutput:
        request = BatchRequest(
            client_id="_single",
            request_id="_single",
            draft_tokens=draft_tokens,
            draft_probs=draft_probs,
            context_tokens=context_tokens,
            sampling_params=sampling_params,
        )
        return self.verify_batch([request])[0]

    def verify_batch(self, requests: list[BatchRequest]) -> list[VerifyOutput]:
        """Run a batched verify. Greedy-only for the Phase 2 MVP."""
        if not requests:
            return []

        # Validate homogeneous greedy mode before touching vLLM — this keeps
        # the "random mode not implemented" contract observable even on hosts
        # that don't have vLLM installed.
        for r in requests:
            if not r.sampling_params.is_greedy:
                raise NotImplementedError(
                    "VllmVerifier currently supports greedy (temperature=0) only. "
                    "Random-mode rejection sampling is tracked as Phase 2 S2 — "
                    "see docs/09-ROADMAP.md."
                )

        from vllm import SamplingParams as VllmSP

        # Per-request prompt = context + draft. vLLM's scheduler will batch
        # these internally.
        prompt_token_ids = [r.context_tokens + r.draft_tokens for r in requests]
        max_draft = max(len(r.draft_tokens) for r in requests)

        # prompt_logprobs returns a dict at each prompt position containing the
        # top-N (token_id -> Logprob); we need rank-1 for argmax. Keep N small:
        # 2 is enough to distinguish draft from argmax (top-1 is either the
        # draft itself or the target argmax).
        vllm_params = VllmSP(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=max(2, max_draft),
            detokenize=False,
        )

        outputs = self.llm.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=vllm_params,
            use_tqdm=False,
        )

        results: list[VerifyOutput] = []
        for req, out in zip(requests, outputs):
            results.append(self._verify_one(req, out))
        return results

    def reset(self) -> None:
        # vLLM manages per-request KV state through its own scheduler; there's
        # nothing to clear at the verifier level.
        pass

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _argmax_from_logprobs(logprob_dict) -> int | None:
        """Extract the rank-1 token id from a vLLM prompt-logprobs entry."""
        if logprob_dict is None:
            return None
        best_id: int | None = None
        best_rank: int | None = None
        for tok_id, lp in logprob_dict.items():
            rank = getattr(lp, "rank", None)
            if rank == 1:
                return int(tok_id)
            if best_rank is None or (rank is not None and rank < best_rank):
                best_rank = rank
                best_id = int(tok_id)
        return best_id

    def _verify_one(self, req: BatchRequest, out) -> VerifyOutput:
        """Greedy rejection for a single request using prompt_logprobs."""
        ctx_len = len(req.context_tokens)
        n_draft = len(req.draft_tokens)

        prompt_logprobs = out.prompt_logprobs  # type: ignore[attr-defined]
        # prompt_logprobs[i] is the distribution for prompt[i] given prompt[:i];
        # position 0 is None (no prediction for the first token).
        if prompt_logprobs is None:
            # Should not happen when prompt_logprobs is requested, but be defensive.
            return VerifyOutput(accepted_tokens=[], bonus_token=None)

        accepted: list[int] = []
        bonus: int | None = None

        for k in range(n_draft):
            pos = ctx_len + k
            if pos >= len(prompt_logprobs):
                break
            target_argmax = self._argmax_from_logprobs(prompt_logprobs[pos])
            if target_argmax is None:
                break
            if target_argmax == req.draft_tokens[k]:
                accepted.append(req.draft_tokens[k])
            else:
                bonus = target_argmax
                break

        if len(accepted) == n_draft:
            # Whole draft accepted; the extra generated token is the bonus.
            bonus = int(out.outputs[0].token_ids[0])

        return VerifyOutput(accepted_tokens=accepted, bonus_token=bonus)
