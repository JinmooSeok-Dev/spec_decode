# SPDX-License-Identifier: Apache-2.0
"""Protocol round-trip demo.

Serializes a :class:`DraftRequest` with msgpack, deserializes it, and verifies
that all fields are preserved. Exercises the codec used by the ZMQ transport.

Run::

    PYTHONPATH=src python examples/protocol_roundtrip.py
"""

from __future__ import annotations

from distspec.common import (
    DraftRequest,
    KVCacheInfo,
    MsgpackDecoder,
    MsgpackEncoder,
    SamplingParams,
)


def main() -> int:
    request = DraftRequest(
        request_id="demo_001",
        prompt_tokens=[1, 2, 3, 4, 5],
        draft_tokens=[6, 7, 8],
        sampling_params=SamplingParams(temperature=0.7, max_tokens=64),
        kv_cache_info=KVCacheInfo(seq_len=8, prev_seq_len=5),
    )

    encoded = MsgpackEncoder().encode(request)
    decoded = MsgpackDecoder(DraftRequest).decode(encoded)

    print(f"encoded size     : {len(encoded)} bytes")
    print(f"request_id       : {decoded.request_id}")
    print(f"prompt_tokens    : {decoded.prompt_tokens}")
    print(f"draft_tokens     : {decoded.draft_tokens}")
    print(f"temperature      : {decoded.sampling_params.temperature}")
    print(f"kv_cache.seq_len : {decoded.kv_cache_info.seq_len}")

    assert decoded.request_id == request.request_id
    assert decoded.prompt_tokens == request.prompt_tokens
    assert decoded.draft_tokens == request.draft_tokens
    assert decoded.sampling_params.temperature == request.sampling_params.temperature
    assert decoded.kv_cache_info.seq_len == request.kv_cache_info.seq_len
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
