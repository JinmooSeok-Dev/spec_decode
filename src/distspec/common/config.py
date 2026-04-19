# SPDX-License-Identifier: Apache-2.0
"""Configuration dataclasses and scenario presets.

Holds :class:`ClientConfig`, :class:`ServerConfig`, :class:`AdaptiveSpecConfig`
plus preset builders (``get_low_latency_config`` etc.). ``ServerConfig`` will
be extended in Phase 2 with a ``backend`` switch to pick between HF and vLLM
verifier backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DraftMethod(Enum):
    """Draft 방법"""
    EAGLE = "eagle"      # EAGLE Proposer (Hidden State 기반)
    NGRAM = "ngram"      # N-gram Proposer (패턴 매칭)
    MEDUSA = "medusa"    # Medusa (다중 헤드)


class KVTransferMode(Enum):
    """KV Cache 전송 모드"""
    NONE = "none"        # 전송 없음
    DELTA = "delta"      # 증분 전송
    FULL = "full"        # 전체 전송


class ClientMode(Enum):
    """Client 동작 모드"""
    SPECULATIVE = "speculative"  # 정상: Speculative Decoding
    DEGRADED = "degraded"        # 저하: 낮은 Speculation Length
    FALLBACK = "fallback"        # Fallback: Draft 모델만 사용


class ConfidenceMetric(Enum):
    """Confidence 측정 방식"""
    ENTROPY = "entropy"            # H(P) = -Σ pᵢ·log(pᵢ)
    MAX_PROB = "max_prob"          # max(P)
    LOGIT_MARGIN = "logit_margin"  # logit_1st - logit_2nd


class VerificationMode(Enum):
    """검증 모드"""
    ALWAYS = "always"                    # 항상 검증 (기존 동작)
    CONFIDENCE_SKIP = "confidence_skip"  # Token-level skip (BiLD)
    QUERY_ROUTING = "query_routing"      # Query-level routing (RouteLLM)
    ADAPTIVE_WINDOW = "adaptive_window"  # Adaptive draft length (SVIP)


# ============================================================================
# Client Configuration
# ============================================================================

@dataclass
class ClientConfig:
    """Client 설정

    Attributes:
        draft_model: Draft 모델 이름/경로
        draft_method: Draft 방법 (EAGLE, N-gram, Medusa)
        server_address: Target Server 주소
        num_speculative_tokens: 기본 Speculation 토큰 수
        adaptive_speculation: 적응형 Speculation Length 활성화
        min_spec_tokens: 최소 Speculation 토큰 수
        max_spec_tokens: 최대 Speculation 토큰 수
        kv_transfer_mode: KV Cache 전송 모드
        kv_compression: KV Cache 압축 방식
        timeout: 요청 타임아웃 (초)
        max_retries: 최대 재시도 횟수
        enable_pre_drafting: Pre-drafting 활성화
    """
    # Draft 모델 설정
    draft_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    draft_method: str = "ngram"  # "eagle" | "ngram" | "medusa"

    # Speculation 설정
    num_speculative_tokens: int = 5
    adaptive_speculation: bool = True
    min_spec_tokens: int = 1
    max_spec_tokens: int = 10

    # 통신 설정
    server_address: str = "localhost:8000"
    timeout: float = 5.0
    max_retries: int = 3

    # KV Cache 설정
    kv_transfer_mode: str = "none"  # "none" | "delta" | "full"
    kv_compression: str | None = None  # "lz4" | None

    # 최적화 설정
    enable_pre_drafting: bool = True

    # Tokenizer 설정
    tokenizer_name: str | None = None  # None이면 draft_model 사용

    def validate(self) -> None:
        """설정 유효성 검증"""
        if self.num_speculative_tokens < 1:
            raise ValueError("num_speculative_tokens must be >= 1")

        if self.min_spec_tokens > self.max_spec_tokens:
            raise ValueError("min_spec_tokens must be <= max_spec_tokens")

        if self.timeout <= 0:
            raise ValueError("timeout must be > 0")

        if self.kv_transfer_mode not in ("none", "delta", "full"):
            raise ValueError(f"Invalid kv_transfer_mode: {self.kv_transfer_mode}")

        if self.draft_method not in ("eagle", "ngram", "medusa"):
            raise ValueError(f"Invalid draft_method: {self.draft_method}")


# ============================================================================
# Server Configuration
# ============================================================================

@dataclass
class ServerConfig:
    """Server 설정

    Attributes:
        target_model: Target 모델 이름/경로
        listen_address: 리슨 주소
        tensor_parallel_size: Tensor Parallel 크기
        pipeline_parallel_size: Pipeline Parallel 크기
        max_batch_size: 최대 배치 크기
        max_wait_time: 배치 대기 최대 시간 (초)
        gpu_memory_utilization: GPU 메모리 사용률
        max_model_len: 최대 모델 길이
    """
    # Backend 선택: "hf" (Phase 1) | "vllm" (Phase 2, 스텁)
    backend: str = "hf"

    # Target 모델 설정
    target_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    # 서빙 설정
    listen_address: str = "0.0.0.0:8000"
    max_batch_size: int = 32
    max_wait_time: float = 0.005  # 5ms

    # GPU 설정
    gpu_memory_utilization: float = 0.9
    # ``None`` lets the backend pick the model's native ``max_position_embeddings``.
    # Set explicitly to cap sequence length or to override what the model exposes.
    max_model_len: int | None = None

    # Worker 설정
    num_workers: int = 1

    # Tokenizer 설정
    tokenizer_name: str | None = None

    def validate(self) -> None:
        """설정 유효성 검증"""
        if self.backend not in ("hf", "vllm"):
            raise ValueError(f"Invalid backend: {self.backend}")

        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")

        if self.pipeline_parallel_size < 1:
            raise ValueError("pipeline_parallel_size must be >= 1")

        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")

        if not (0.0 < self.gpu_memory_utilization <= 1.0):
            raise ValueError("gpu_memory_utilization must be in (0, 1]")


# ============================================================================
# Adaptive Speculation Configuration
# ============================================================================

@dataclass
class AdaptiveSpecConfig:
    """적응형 Speculation 설정

    Attributes:
        min_spec_tokens: 최소 Speculation 토큰 수
        max_spec_tokens: 최대 Speculation 토큰 수
        history_size: 히스토리 크기
        decode_time_estimate: 토큰당 디코드 시간 추정 (초)
        low_acceptance_threshold: 낮은 수락률 임계값
        high_acceptance_threshold: 높은 수락률 임계값
    """
    min_spec_tokens: int = 1
    max_spec_tokens: int = 10
    history_size: int = 20
    decode_time_estimate: float = 0.01  # 10ms per token

    # 임계값
    low_acceptance_threshold: float = 0.3  # 30% 미만이면 K 감소
    high_acceptance_threshold: float = 0.8  # 80% 이상이면 K 증가


# ============================================================================
# Metrics Configuration
# ============================================================================

@dataclass
class MetricsConfig:
    """메트릭 수집 설정

    Attributes:
        enable_metrics: 메트릭 수집 활성화
        log_interval: 로그 출력 간격 (요청 수)
        export_prometheus: Prometheus 메트릭 내보내기
        prometheus_port: Prometheus 포트
    """
    enable_metrics: bool = True
    log_interval: int = 100
    export_prometheus: bool = False
    prometheus_port: int = 9090


# ============================================================================
# Full Configuration
# ============================================================================

@dataclass
class DistributedSpecDecodeConfig:
    """전체 분산 Speculative Decoding 설정"""
    client: ClientConfig = field(default_factory=ClientConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    adaptive: AdaptiveSpecConfig = field(default_factory=AdaptiveSpecConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    def validate(self) -> None:
        """전체 설정 유효성 검증"""
        self.client.validate()
        self.server.validate()

    @classmethod
    def from_dict(cls, d: dict) -> DistributedSpecDecodeConfig:
        """딕셔너리에서 설정 생성"""
        return cls(
            client=ClientConfig(**d.get('client', {})),
            server=ServerConfig(**d.get('server', {})),
            adaptive=AdaptiveSpecConfig(**d.get('adaptive', {})),
            metrics=MetricsConfig(**d.get('metrics', {})),
        )

    @classmethod
    def from_yaml(cls, path: str) -> DistributedSpecDecodeConfig:
        """YAML 파일에서 설정 로드"""
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    def to_dict(self) -> dict:
        """설정을 딕셔너리로 변환"""
        from dataclasses import asdict
        return {
            'client': asdict(self.client),
            'server': asdict(self.server),
            'adaptive': asdict(self.adaptive),
            'metrics': asdict(self.metrics),
        }


# ============================================================================
# Default Configurations for Different Scenarios
# ============================================================================

def get_low_latency_config() -> DistributedSpecDecodeConfig:
    """저지연 시나리오 설정

    짧은 응답, 빠른 응답이 중요한 경우
    """
    return DistributedSpecDecodeConfig(
        client=ClientConfig(
            draft_method="ngram",  # 빠른 N-gram
            num_speculative_tokens=3,  # 적은 토큰
            adaptive_speculation=False,
            kv_transfer_mode="none",
            enable_pre_drafting=False,
        ),
        server=ServerConfig(
            max_batch_size=8,
            max_wait_time=0.001,  # 1ms
        ),
    )


def get_high_throughput_config() -> DistributedSpecDecodeConfig:
    """고처리량 시나리오 설정

    대량 요청, 처리량이 중요한 경우
    """
    return DistributedSpecDecodeConfig(
        client=ClientConfig(
            draft_method="eagle",  # 높은 수락률
            num_speculative_tokens=8,
            adaptive_speculation=True,
            max_spec_tokens=12,
            kv_transfer_mode="delta",
            enable_pre_drafting=True,
        ),
        server=ServerConfig(
            max_batch_size=64,
            max_wait_time=0.01,  # 10ms
        ),
    )


def get_resource_constrained_config() -> DistributedSpecDecodeConfig:
    """리소스 제한 시나리오 설정

    클라이언트 리소스가 제한된 경우
    """
    return DistributedSpecDecodeConfig(
        client=ClientConfig(
            draft_method="ngram",  # 모델 없이 동작
            num_speculative_tokens=4,
            adaptive_speculation=True,
            kv_transfer_mode="none",  # 네트워크 최소화
            enable_pre_drafting=False,  # 추가 계산 없음
        ),
        server=ServerConfig(
            max_batch_size=16,
        ),
    )
