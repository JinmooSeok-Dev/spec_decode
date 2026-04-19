# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for the streaming client.

Installed as the ``distspec-client`` console script (see ``pyproject.toml``).
Connects to a running ``distspec-server`` and streams tokens back to stdout.

Example::

    # Terminal 1 — target server
    distspec-server --backend vllm --model gpt2 --max-model-len 1024

    # Terminal 2 — client
    distspec-client --prompt "Hello, world." --server 127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from ..common.config import ClientConfig
from ..common.protocol import SamplingParams
from .fault_tolerant_client import FaultTolerantClient


async def run(
    *,
    prompt: str,
    server: str,
    max_tokens: int,
    temperature: float,
    draft_method: str,
    num_speculative_tokens: int,
    timeout: float,
    tokenizer_name: str | None,
) -> None:
    config = ClientConfig(
        server_address=server,
        draft_method=draft_method,
        num_speculative_tokens=num_speculative_tokens,
        timeout=timeout,
        tokenizer_name=tokenizer_name,
    )

    async with FaultTolerantClient(config) as client:
        print(f"prompt : {prompt}")
        print("output : ", end="", flush=True)

        params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        async for token in client.generate(prompt, params):
            print(token, end="", flush=True)
        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="distspec — streaming client for the target server.",
    )
    parser.add_argument("--prompt", default="Hello, how are you?")
    parser.add_argument(
        "--server",
        default="localhost:8000",
        help="Target server address, host:port.",
    )
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. Use 0.0 for greedy (required by the vLLM backend).",
    )
    parser.add_argument(
        "--draft-method",
        choices=["ngram", "suffix", "eagle"],
        default="ngram",
        help="Client-side draft proposer.",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=5,
        help="Initial number of draft tokens per round (adaptive K may adjust).",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "HuggingFace tokenizer name. Defaults to the draft model, which is "
            "usually what you want when server and client share a tokenizer."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        asyncio.run(
            run(
                prompt=args.prompt,
                server=args.server,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                draft_method=args.draft_method,
                num_speculative_tokens=args.num_speculative_tokens,
                timeout=args.timeout,
                tokenizer_name=args.tokenizer,
            )
        )
    except KeyboardInterrupt:
        print()  # newline after partial output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
