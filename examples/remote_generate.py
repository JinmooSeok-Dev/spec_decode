# SPDX-License-Identifier: Apache-2.0
"""Generate text against a remote target server over ZMQ.

A streaming client demo: start a ``distspec-server`` in another terminal
(HuggingFace or vLLM backend) and this script connects to it, sends draft
tokens produced by the local N-gram proposer, and prints the accepted
tokens as they come back.

Example ::

    # Terminal 1 — target server
    distspec-server --backend hf   --model meta-llama/Llama-3.2-3B-Instruct
    #   or
    distspec-server --backend vllm --model gpt2 --max-model-len 1024

    # Terminal 2 — this client
    python examples/remote_generate.py --prompt "Hello, world."
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
