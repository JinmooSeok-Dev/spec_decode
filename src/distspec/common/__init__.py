# SPDX-License-Identifier: Apache-2.0
"""Common module: protocol, config, confidence, sampling utilities."""

from .confidence import (
    ConfidenceCalculator,
    ConfidenceConfig,
    QueryClassifier,
    TokenConfidenceResult,
)
from .config import (
    ClientConfig,
    ConfidenceMetric,
    ServerConfig,
    VerificationMode,
)
from .protocol import (
    DraftOutput,
    DraftRequest,
    HealthCheck,
    HealthResponse,
    KVCacheInfo,
    MsgpackDecoder,
    MsgpackEncoder,
    SamplingParams,
    VerifyOutput,
    VerifyResponse,
)
from .sampling import apply_sampling_filters, apply_top_k, apply_top_p

__all__ = [
    # Protocol
    "SamplingParams",
    "DraftRequest",
    "DraftOutput",
    "VerifyResponse",
    "VerifyOutput",
    "KVCacheInfo",
    "HealthCheck",
    "HealthResponse",
    "MsgpackEncoder",
    "MsgpackDecoder",
    # Config
    "ClientConfig",
    "ServerConfig",
    "ConfidenceMetric",
    "VerificationMode",
    # Confidence
    "ConfidenceConfig",
    "ConfidenceCalculator",
    "TokenConfidenceResult",
    "QueryClassifier",
    # Sampling
    "apply_sampling_filters",
    "apply_top_k",
    "apply_top_p",
]
