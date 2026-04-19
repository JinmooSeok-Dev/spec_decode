# SPDX-License-Identifier: Apache-2.0
"""Wire protocol: message dataclasses + msgpack codec.

Defines the Client ↔ Server messages (``DraftRequest``, ``VerifyResponse``,
``HealthCheck`` …) and the :class:`MsgpackEncoder` / :class:`MsgpackDecoder`
pair used to serialize them over ZMQ. Deliberately transport-agnostic so the
same messages can be reused with the Phase 2 vLLM backend.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Optional imports - msgspec for high-performance serialization
try:
    import msgspec
    HAS_MSGSPEC = True
except ImportError:
    HAS_MSGSPEC = False
    import json

# Optional imports - torch for tensor handling
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================================
# Sampling Parameters
# ============================================================================

@dataclass
class SamplingParams:
    """샘플링 파라미터

    vLLM의 SamplingParams와 호환되는 최소 구현
    """
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 256
    seed: int | None = None

    # Greedy decoding flag (temperature=0)
    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0


# ============================================================================
# KV Cache Metadata
# ============================================================================

@dataclass
class KVCacheInfo:
    """KV Cache 메타데이터

    Attributes:
        seq_len: 현재 시퀀스 길이
        prev_seq_len: 이전 동기화 시점의 시퀀스 길이
        transfer_mode: KV Cache 전송 모드 ("none" | "delta" | "full")
        block_ids: 블록 ID 목록 (PagedAttention용)
    """
    seq_len: int
    prev_seq_len: int = 0
    transfer_mode: str = "none"  # "none" | "delta" | "full"
    block_ids: list | None = None


# ============================================================================
# Draft Request/Output (Client -> Server)
# ============================================================================

@dataclass
class DraftRequest:
    """Client -> Server: Draft 토큰 검증 요청

    Attributes:
        request_id: 요청 고유 ID
        prompt_tokens: 초기 프롬프트 토큰 (첫 요청만)
        draft_tokens: Draft 토큰들
        draft_probs: Draft 확률 분포 (EAGLE), None (N-gram)
        sampling_params: 샘플링 파라미터
        kv_cache_info: KV Cache 메타데이터
        kv_cache_data: KV Cache 데이터 (delta/full 모드)
    """
    request_id: str
    prompt_tokens: list = field(default_factory=list)
    draft_tokens: list = field(default_factory=list)
    draft_probs: Any | None = None  # torch.Tensor or numpy array
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    kv_cache_info: KVCacheInfo | None = None
    kv_cache_data: bytes | None = None


@dataclass
class DraftOutput:
    """Draft Proposer 출력

    Client 내부에서 DraftProposer가 생성하는 출력

    Attributes:
        draft_tokens: 생성된 Draft 토큰 리스트
        draft_probs: Draft 토큰의 확률 분포 (EAGLE만)
        hidden_states: Hidden States (다음 EAGLE 입력용)
        kv_cache_info: KV Cache 메타데이터
    """
    draft_tokens: list = field(default_factory=list)
    draft_probs: Any | None = None  # torch.Tensor or numpy array
    hidden_states: Any | None = None  # torch.Tensor
    kv_cache_info: KVCacheInfo | None = None
    confidence_scores: list | None = None  # per-token confidence [float, ...]


# ============================================================================
# Verify Response/Output (Server -> Client)
# ============================================================================

@dataclass
class VerifyResponse:
    """Server -> Client: 검증 결과

    Attributes:
        request_id: 요청 고유 ID
        accepted_tokens: 수락된 토큰 리스트
        num_accepted: 수락된 토큰 수
        bonus_token: Bonus 토큰 (모두 수락 시)
        hidden_states: 다음 Draft용 Hidden States (EAGLE)
        finished: 생성 완료 여부 (EOS 또는 max_tokens)
        logprobs: 토큰 로그 확률 (선택)
    """
    request_id: str
    accepted_tokens: list = field(default_factory=list)
    num_accepted: int = 0
    bonus_token: int | None = None
    hidden_states: Any | None = None  # torch.Tensor
    finished: bool = False
    logprobs: list | None = None


@dataclass
class VerifyOutput:
    """Target Verifier 내부 출력

    Server 내부에서 TargetVerifier가 생성하는 출력

    Attributes:
        accepted_tokens: 수락된 토큰 리스트
        bonus_token: Bonus 토큰
        hidden_states: Hidden States
        target_logits: Target 모델 Logits
        logprobs: 로그 확률
    """
    accepted_tokens: list = field(default_factory=list)
    bonus_token: int | None = None
    hidden_states: Any | None = None
    target_logits: Any | None = None
    logprobs: list | None = None


# ============================================================================
# Health Check Messages
# ============================================================================

@dataclass
class HealthCheck:
    """서버 상태 확인 요청"""
    client_id: str
    timestamp: float


@dataclass
class HealthResponse:
    """서버 상태 응답"""
    server_id: str
    is_healthy: bool = True
    load: float = 0.0  # 0.0 ~ 1.0
    queue_length: int = 0


# ============================================================================
# Serialization / Deserialization
# ============================================================================

class MsgpackEncoder:
    """메시지 인코더

    msgspec 기반 고속 직렬화 (fallback: json)
    """

    def __init__(self):
        if HAS_MSGSPEC:
            self.encoder = msgspec.msgpack.Encoder()
        else:
            self.encoder = None

    def encode(self, obj: Any) -> bytes:
        """객체를 bytes로 직렬화"""
        # dataclass -> dict 변환
        if hasattr(obj, '__dataclass_fields__'):
            data = self._dataclass_to_dict(obj)
        else:
            data = obj

        if HAS_MSGSPEC:
            return self.encoder.encode(data)
        else:
            return json.dumps(data).encode('utf-8')

    def _dataclass_to_dict(self, obj: Any) -> dict:
        """Dataclass를 dict로 변환 (재귀)"""
        result = {'_type': type(obj).__name__}

        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            result[field_name] = self._serialize_value(value)

        return result

    def _serialize_value(self, value: Any) -> Any:
        """값 직렬화"""
        if value is None:
            return None

        # Tensor -> bytes with metadata
        if HAS_TORCH and isinstance(value, torch.Tensor):
            return self._tensor_to_dict(value)

        # numpy array -> bytes with metadata
        if isinstance(value, np.ndarray):
            return self._ndarray_to_dict(value)

        # Nested dataclass
        if hasattr(value, '__dataclass_fields__'):
            return self._dataclass_to_dict(value)

        # List/tuple
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]

        # Dict
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}

        # Primitive types (int, float, str, bool)
        return value

    def _tensor_to_dict(self, tensor: torch.Tensor) -> dict:
        """Torch Tensor를 직렬화 가능한 dict로 변환"""
        np_array = tensor.detach().cpu().numpy()
        return {
            '_tensor': True,
            'data': np_array.tobytes(),
            'dtype': str(np_array.dtype),
            'shape': list(np_array.shape),
        }

    def _ndarray_to_dict(self, array: np.ndarray) -> dict:
        """Numpy array를 직렬화 가능한 dict로 변환"""
        return {
            '_ndarray': True,
            'data': array.tobytes(),
            'dtype': str(array.dtype),
            'shape': list(array.shape),
        }


class MsgpackDecoder:
    """메시지 디코더

    msgspec 기반 고속 역직렬화 (fallback: json)
    """

    # 지원하는 타입 매핑
    TYPE_MAP = {
        'SamplingParams': SamplingParams,
        'KVCacheInfo': KVCacheInfo,
        'DraftRequest': DraftRequest,
        'DraftOutput': DraftOutput,
        'VerifyResponse': VerifyResponse,
        'VerifyOutput': VerifyOutput,
        'HealthCheck': HealthCheck,
        'HealthResponse': HealthResponse,
    }

    def __init__(self, target_class: type = None):
        if HAS_MSGSPEC:
            self.decoder = msgspec.msgpack.Decoder()
        else:
            self.decoder = None
        self.target_class = target_class

    def decode(self, data: bytes) -> Any:
        """bytes를 객체로 역직렬화"""
        if HAS_MSGSPEC:
            obj_dict = self.decoder.decode(data)
        else:
            obj_dict = json.loads(data.decode('utf-8'))

        return self._deserialize_value(obj_dict)

    def _deserialize_value(self, value: Any) -> Any:
        """값 역직렬화"""
        if value is None:
            return None

        # Dict with special markers
        if isinstance(value, dict):
            # Tensor
            if value.get('_tensor'):
                return self._dict_to_tensor(value)

            # Numpy array
            if value.get('_ndarray'):
                return self._dict_to_ndarray(value)

            # Dataclass
            if '_type' in value:
                return self._dict_to_dataclass(value)

            # Regular dict
            return {k: self._deserialize_value(v) for k, v in value.items()}

        # List
        if isinstance(value, list):
            return [self._deserialize_value(v) for v in value]

        # Primitive types
        return value

    def _dict_to_tensor(self, d: dict) -> torch.Tensor:
        """dict를 Torch Tensor로 변환"""
        if not HAS_TORCH:
            # torch 없으면 numpy array로 반환
            return self._dict_to_ndarray(d)

        arr = np.frombuffer(d['data'], dtype=d['dtype'])
        arr = arr.reshape(d['shape'])
        return torch.from_numpy(arr.copy())

    def _dict_to_ndarray(self, d: dict) -> np.ndarray:
        """dict를 Numpy array로 변환"""
        arr = np.frombuffer(d['data'], dtype=d['dtype'])
        return arr.reshape(d['shape']).copy()

    def _dict_to_dataclass(self, d: dict) -> Any:
        """dict를 Dataclass로 변환"""
        type_name = d.get('_type')
        cls = self.TYPE_MAP.get(type_name, self.target_class)

        if cls is None:
            return d

        # 필드별로 역직렬화
        kwargs = {}
        for field_name, field_info in cls.__dataclass_fields__.items():
            if field_name not in d:
                continue

            value = d[field_name]
            kwargs[field_name] = self._deserialize_value(value)

        return cls(**kwargs)


# ============================================================================
# Utility Functions
# ============================================================================

def serialize_kv_cache(
    k_cache: Any,  # torch.Tensor or np.ndarray
    v_cache: Any,
    prev_seq_len: int,
    curr_seq_len: int,
    mode: str = "delta",
) -> tuple:
    """KV Cache 직렬화

    Args:
        k_cache: Key cache tensor [num_layers, num_heads, seq_len, head_dim]
        v_cache: Value cache tensor
        prev_seq_len: 이전 시퀀스 길이
        curr_seq_len: 현재 시퀀스 길이
        mode: 전송 모드 ("none", "delta", "full")

    Returns:
        (bytes, KVCacheInfo)
    """
    if mode == "none":
        return None, KVCacheInfo(
            seq_len=curr_seq_len,
            prev_seq_len=prev_seq_len,
            transfer_mode="none",
        )

    # Convert to numpy if tensor
    if HAS_TORCH and isinstance(k_cache, torch.Tensor):
        k_np = k_cache.cpu().numpy()
        v_np = v_cache.cpu().numpy()
    else:
        k_np = k_cache
        v_np = v_cache

    # Extract relevant portion
    if mode == "delta":
        k_data = k_np[..., prev_seq_len:curr_seq_len, :]
        v_data = v_np[..., prev_seq_len:curr_seq_len, :]
    else:  # full
        k_data = k_np[..., :curr_seq_len, :]
        v_data = v_np[..., :curr_seq_len, :]

    # Serialize: [k_size(8bytes), k_data, v_data]
    k_bytes = k_data.tobytes()
    v_bytes = v_data.tobytes()
    data = struct.pack('Q', len(k_bytes)) + k_bytes + v_bytes

    info = KVCacheInfo(
        seq_len=curr_seq_len,
        prev_seq_len=prev_seq_len,
        transfer_mode=mode,
    )

    return data, info


def deserialize_kv_cache(
    data: bytes,
    info: KVCacheInfo,
    dtype: str = 'float32',
) -> tuple:
    """KV Cache 역직렬화

    Args:
        data: 직렬화된 KV Cache 데이터
        info: KV Cache 메타데이터
        dtype: 데이터 타입

    Returns:
        (k_cache, v_cache) as numpy arrays
    """
    if info.transfer_mode == "none" or data is None:
        return None, None

    k_size = struct.unpack('Q', data[:8])[0]
    k_bytes = data[8:8+k_size]
    v_bytes = data[8+k_size:]

    k_arr = np.frombuffer(k_bytes, dtype=dtype)
    v_arr = np.frombuffer(v_bytes, dtype=dtype)

    return k_arr, v_arr
