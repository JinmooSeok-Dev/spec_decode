# SPDX-License-Identifier: Apache-2.0
"""distspec — Distributed Speculative Decoding.

A reference implementation of client-server speculative decoding that
separates draft-token generation (client) from target verification (server)
over a ZMQ transport.

Public subpackages:
  - :mod:`distspec.common` — message protocol, configuration, sampling utils.
  - :mod:`distspec.client` — draft proposers and streaming clients.
  - :mod:`distspec.server` — verifier backends (HF now, vLLM planned) and
    the ZMQ serving loop.
"""

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
