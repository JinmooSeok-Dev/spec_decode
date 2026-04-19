# SPDX-License-Identifier: Apache-2.0
"""Client module: draft proposers and streaming clients."""

from .confidence_client import (
    AdaptiveWindowClient,
    ConfidenceSkipClient,
    QueryRoutingClient,
    create_confidence_client,
)
from .draft_client import AdaptiveSpeculationController, DraftClient
from .draft_proposer import (
    BaseDraftProposer,
    EagleDraftProposer,
    NgramDraftProposer,
    SuffixDraftProposer,
    create_draft_proposer,
)
from .fault_tolerant_client import FaultTolerantClient

__all__ = [
    # Proposers
    "BaseDraftProposer",
    "NgramDraftProposer",
    "SuffixDraftProposer",
    "EagleDraftProposer",
    "create_draft_proposer",
    # Clients
    "DraftClient",
    "AdaptiveSpeculationController",
    "FaultTolerantClient",
    # Confidence clients
    "ConfidenceSkipClient",
    "QueryRoutingClient",
    "AdaptiveWindowClient",
    "create_confidence_client",
]
