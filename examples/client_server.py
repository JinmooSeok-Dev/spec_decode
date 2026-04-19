# SPDX-License-Identifier: Apache-2.0
"""End-to-end client/server demo.

Requires a running target server::

    python -m distspec.server.target_server \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --listen-address 0.0.0.0:8000

Then run this script::

    PYTHONPATH=src python examples/client_server.py --prompt "Hello, world."
"""

from __future__ import annotations

import argparse
import asyncio

from distspec.client import FaultTolerantClient
from distspec.common import ClientConfig, SamplingParams


async def run(prompt: str, server: str, max_tokens: int) -> None:
    config = ClientConfig(
        server_address=server,
        draft_method="ngram",
        num_speculative_tokens=5,
        timeout=10.0,
    )

    async with FaultTolerantClient(config) as client:
        print(f"prompt : {prompt}")
        print("output : ", end="", flush=True)

        params = SamplingParams(temperature=0.7, max_tokens=max_tokens)
        async for token in client.generate(prompt, params):
            print(token, end="", flush=True)
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Hello, how are you?")
    parser.add_argument("--server", default="localhost:8000")
    parser.add_argument("--max-tokens", type=int, default=50)
    args = parser.parse_args()

    asyncio.run(run(args.prompt, args.server, args.max_tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
