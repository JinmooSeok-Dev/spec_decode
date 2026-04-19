# SPDX-License-Identifier: Apache-2.0
"""Sampling utilities shared between draft proposers and target verifier.

Provides top-k and top-p (nucleus) logit filtering used by both the client-side
draft generation (``prototype.client.draft_proposer.EagleDraftProposer``) and
the server-side ``RejectionSampler``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

if TYPE_CHECKING:
    import torch


NEG_INF = float("-inf")


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Mask out all but the top-k logits in the last dimension.

    Args:
        logits: Tensor of shape ``[..., vocab_size]``.
        top_k: Number of top entries to keep. Values ``<= 0`` disable filtering.

    Returns:
        A tensor of the same shape as ``logits`` where non-top-k positions are
        set to ``-inf``.
    """
    if top_k <= 0:
        return logits
    k = min(top_k, logits.size(-1))
    threshold = torch.topk(logits, k, dim=-1)[0][..., -1, None]
    return logits.masked_fill(logits < threshold, NEG_INF)


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Mask out the long tail such that the cumulative probability exceeds ``top_p``.

    Implements nucleus sampling: sort logits descending, take the smallest
    prefix whose softmax-cumulative probability reaches ``top_p``, and mask the
    rest with ``-inf``.

    Args:
        logits: Tensor of shape ``[..., vocab_size]``.
        top_p: Cumulative probability threshold in ``(0, 1]``. Values ``>= 1.0``
            disable filtering.

    Returns:
        Filtered logits with the same shape.
    """
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Shift so that we always keep at least the top-1 token.
    to_remove = cum_probs > top_p
    to_remove[..., 1:] = to_remove[..., :-1].clone()
    to_remove[..., 0] = False

    mask = to_remove.scatter(-1, sorted_idx, to_remove)
    return logits.masked_fill(mask, NEG_INF)


def apply_sampling_filters(
    logits: torch.Tensor,
    top_k: int = -1,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Apply top-k then top-p filtering.

    The order follows the convention used by HuggingFace and vLLM: top-k first
    to bound the candidate set, then top-p to trim the tail within it.
    """
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)
    return logits
