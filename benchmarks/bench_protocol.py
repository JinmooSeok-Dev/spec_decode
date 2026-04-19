# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark for the msgpack protocol codec — CPU only, CI-safe.

Measures encode and decode throughput for ``DraftRequest`` messages.

Run::

    PYTHONPATH=src python benchmarks/bench_protocol.py [--full] [--json]
"""

from __future__ import annotations

from _common import build_parser, emit, stopwatch
from distspec.common import (
    DraftRequest,
    KVCacheInfo,
    MsgpackDecoder,
    MsgpackEncoder,
    SamplingParams,
)


def build_sample(draft_len: int) -> DraftRequest:
    return DraftRequest(
        request_id="bench",
        prompt_tokens=list(range(64)),
        draft_tokens=list(range(draft_len)),
        sampling_params=SamplingParams(temperature=0.7, max_tokens=256),
        kv_cache_info=KVCacheInfo(seq_len=64 + draft_len, prev_seq_len=64),
    )


def run(quick: bool) -> dict:
    draft_len = 8
    iterations = 2000 if quick else 20_000

    sample = build_sample(draft_len)
    enc = MsgpackEncoder()
    dec = MsgpackDecoder(DraftRequest)

    with stopwatch() as t_enc:
        for _ in range(iterations):
            blob = enc.encode(sample)

    # Use the last-encoded blob for decode.
    with stopwatch() as t_dec:
        for _ in range(iterations):
            dec.decode(blob)

    return {
        "name": "bench_protocol",
        "iterations": iterations,
        "encoded_bytes": len(blob),
        "encode_seconds": t_enc["seconds"],
        "decode_seconds": t_dec["seconds"],
        "encode_ops_per_sec": iterations / t_enc["seconds"],
        "decode_ops_per_sec": iterations / t_dec["seconds"],
    }


def main() -> int:
    parser = build_parser(__doc__ or "")
    args = parser.parse_args()
    emit(run(quick=args.quick), as_json=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
