# SPDX-License-Identifier: Apache-2.0
"""ZMQ ROUTER-based target server.

Holds one :class:`BaseVerifier` instance and multiplexes draft-verify requests
from many clients. Backend selection (HF now, vLLM planned) is abstracted so
the server code is backend-agnostic.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

try:
    import zmq
    import zmq.asyncio
    _HAS_ZMQ = True
except ImportError:
    _HAS_ZMQ = False

from ..common.config import ServerConfig
from ..common.protocol import (
    DraftRequest,
    HealthCheck,
    HealthResponse,
    MsgpackDecoder,
    MsgpackEncoder,
    SamplingParams,
    VerifyResponse,
)
from .base import BaseVerifier
from .hf_verifier import HfVerifier

logger = logging.getLogger(__name__)


def _build_verifier(config: ServerConfig) -> BaseVerifier:
    """Instantiate the verifier backend selected by ``config.backend``."""
    if config.backend == "hf":
        return HfVerifier(
            model_name=config.target_model,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
        )
    if config.backend == "vllm":
        # Phase 2: delegate to vLLM LLMEngine. Currently a stub.
        from .vllm_verifier import VllmVerifier
        return VllmVerifier(
            model_name=config.target_model,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
        )
    raise ValueError(f"Unknown backend: {config.backend!r}")


# ============================================================================
# Request State
# ============================================================================

@dataclass
class RequestState:
    """요청 상태

    각 요청의 상태를 추적
    """
    prompt_tokens: list[int] = field(default_factory=list)
    generated_tokens: list[int] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def all_tokens(self) -> list[int]:
        """전체 토큰 (프롬프트 + 생성)"""
        return self.prompt_tokens + self.generated_tokens


# ============================================================================
# Metrics
# ============================================================================

@dataclass
class ServerMetrics:
    """서버 메트릭"""
    total_requests: int = 0
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    total_bonus_tokens: int = 0
    total_latency: float = 0.0

    @property
    def avg_acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.total_latency / self.total_requests) * 1000

    def record(
        self,
        num_draft: int,
        num_accepted: int,
        has_bonus: bool,
        latency: float,
    ):
        """메트릭 기록"""
        self.total_requests += 1
        self.total_draft_tokens += num_draft
        self.total_accepted_tokens += num_accepted
        if has_bonus:
            self.total_bonus_tokens += 1
        self.total_latency += latency

    def report(self) -> str:
        """메트릭 리포트"""
        return f"""
Server Metrics:
--------------
Total Requests: {self.total_requests}
Total Draft Tokens: {self.total_draft_tokens}
Total Accepted Tokens: {self.total_accepted_tokens}
Total Bonus Tokens: {self.total_bonus_tokens}
Avg Acceptance Rate: {self.avg_acceptance_rate:.2%}
Avg Latency: {self.avg_latency_ms:.2f}ms
"""


# ============================================================================
# Target Server
# ============================================================================

class TargetServer:
    """Target 서버

    ZMQ ROUTER 소켓으로 다중 Client의 요청 처리

    동작:
    1. Client에서 DraftRequest 수신
    2. BaseVerifier 로 검증 (backend: HfVerifier / VllmVerifier)
    3. VerifyResponse 반환
    """

    def __init__(
        self,
        config: ServerConfig,
        verifier: BaseVerifier | None = None,
        server_id: str | None = None,
    ):
        """
        Args:
            config: 서버 설정
            verifier: Concrete :class:`BaseVerifier` instance. If ``None``,
                an :class:`HfVerifier` is constructed from ``config``.
            server_id: 서버 ID
        """
        if not _HAS_ZMQ:
            raise RuntimeError("TargetServer requires pyzmq. Install with: pip install pyzmq")

        self.config = config
        self.server_id = server_id or f"server_{id(self)}"

        # Verifier (backend-specific)
        self.verifier: BaseVerifier = (
            verifier if verifier is not None else _build_verifier(config)
        )

        # ZMQ
        self.context: zmq.asyncio.Context | None = None
        self.socket: zmq.asyncio.Socket | None = None

        # 메시지 인코더/디코더
        self.encoder = MsgpackEncoder()
        self.request_decoder = MsgpackDecoder(DraftRequest)
        self.health_decoder = MsgpackDecoder(HealthCheck)

        # 상태
        self.active_requests: dict[str, RequestState] = {}
        self.running = False

        # 메트릭
        self.metrics = ServerMetrics()

        # EOS 토큰 ID
        self._eos_token_id = None

    @property
    def eos_token_id(self) -> int:
        """EOS 토큰 ID"""
        if self._eos_token_id is None:
            self._eos_token_id = self.verifier.tokenizer.eos_token_id
        return self._eos_token_id

    async def start(self) -> None:
        """서버 시작"""
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.ROUTER)

        listen_addr = f"tcp://{self.config.listen_address}"
        self.socket.bind(listen_addr)

        logger.info("Target server listening on %s", listen_addr)
        logger.info("Model: %s", self.config.target_model)

        self.running = True

    async def serve(self) -> None:
        """메인 서버 루프"""
        if not self.running:
            await self.start()

        logger.info("Server ready, waiting for requests...")

        while self.running:
            try:
                # Receive multipart. DEALER peers may or may not prepend an
                # empty delimiter, so we accept 2 or 3 frames: ``frames[0]``
                # is the peer identity (prepended by ROUTER), the remaining
                # non-empty frame carries the payload.
                frames = await self.socket.recv_multipart()

                if len(frames) < 2:
                    continue

                identity = frames[0]
                request_bytes = frames[-1]

                # 요청 처리
                response = await self._process_message(identity, request_bytes)

                if response is not None:
                    # 응답 전송
                    await self.socket.send_multipart([
                        identity,
                        b"",
                        self.encoder.encode(response),
                    ])

            except zmq.error.ContextTerminated:
                break
            except Exception:
                logger.exception("Error processing request")
                continue

    async def _process_message(
        self,
        identity: bytes,
        data: bytes,
    ) -> VerifyResponse | None:
        """메시지 처리"""
        client_id = identity.decode()

        # 메시지 타입 판별 및 디코딩
        try:
            # Health Check 시도
            message = self.health_decoder.decode(data)
            if isinstance(message, HealthCheck):
                return self._handle_health_check(message)
        except Exception:
            pass

        try:
            # DraftRequest 시도
            request = self.request_decoder.decode(data)
            if isinstance(request, DraftRequest):
                return await self._handle_draft_request(client_id, request)
        except Exception as e:
            logger.warning("Failed to decode message: %s", e)
            return None

        return None

    def _handle_health_check(self, check: HealthCheck) -> HealthResponse:
        """Health Check 처리"""
        return HealthResponse(
            server_id=self.server_id,
            is_healthy=True,
            load=len(self.active_requests) / self.config.max_batch_size,
            queue_length=len(self.active_requests),
        )

    async def _handle_draft_request(
        self,
        client_id: str,
        request: DraftRequest,
    ) -> VerifyResponse:
        """Draft 요청 처리"""
        start_time = time.time()

        # 상태 키
        state_key = f"{client_id}:{request.request_id}"

        # 요청 상태 관리
        if state_key not in self.active_requests:
            # 새 요청
            self.active_requests[state_key] = RequestState(
                prompt_tokens=request.prompt_tokens,
            )

        state = self.active_requests[state_key]

        # 검증 수행
        verify_output = await asyncio.get_event_loop().run_in_executor(
            None,
            self.verifier.verify,
            request.draft_tokens,
            request.draft_probs,
            state.all_tokens,
            request.sampling_params,
        )

        # 상태 업데이트
        state.generated_tokens.extend(verify_output.accepted_tokens)
        if verify_output.bonus_token is not None:
            state.generated_tokens.append(verify_output.bonus_token)

        # 종료 확인
        finished = self._check_finished(state, request.sampling_params)

        if finished:
            del self.active_requests[state_key]

        # 메트릭 기록
        latency = time.time() - start_time
        self.metrics.record(
            num_draft=len(request.draft_tokens),
            num_accepted=len(verify_output.accepted_tokens),
            has_bonus=verify_output.bonus_token is not None,
            latency=latency,
        )

        return VerifyResponse(
            request_id=request.request_id,
            accepted_tokens=verify_output.accepted_tokens,
            num_accepted=len(verify_output.accepted_tokens),
            bonus_token=verify_output.bonus_token,
            hidden_states=verify_output.hidden_states,
            finished=finished,
            logprobs=verify_output.logprobs,
        )

    def _check_finished(
        self,
        state: RequestState,
        params: SamplingParams,
    ) -> bool:
        """생성 완료 확인"""
        # EOS 토큰 확인
        if state.generated_tokens and state.generated_tokens[-1] == self.eos_token_id:
            return True

        # Max tokens 확인
        if len(state.generated_tokens) >= params.max_tokens:
            return True

        return False

    async def stop(self) -> None:
        """서버 종료"""
        self.running = False

        if self.socket is not None:
            self.socket.close()
            self.socket = None

        if self.context is not None:
            self.context.term()
            self.context = None

        # 메트릭 출력
        logger.info(self.metrics.report())

    async def __aenter__(self):
        """Context manager 진입"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        await self.stop()


# ============================================================================
# Main Entry Point
# ============================================================================

async def run_server(
    model: str = "meta-llama/Llama-3.2-3B-Instruct",
    listen_address: str = "0.0.0.0:8000",
    **kwargs,
):
    """서버 실행

    Args:
        model: Target 모델
        listen_address: 리슨 주소
        **kwargs: 추가 설정
    """
    config = ServerConfig(
        target_model=model,
        listen_address=listen_address,
        **kwargs,
    )

    async with TargetServer(config) as server:
        await server.serve()


def main():
    """CLI 엔트리포인트"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Distributed Speculative Decoding - Target Server"
    )
    parser.add_argument(
        "--model", "-m",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Target model name"
    )
    parser.add_argument(
        "--listen-address", "-l",
        default="0.0.0.0:8000",
        help="Listen address (host:port)"
    )
    parser.add_argument(
        "--tensor-parallel-size", "-tp",
        type=int,
        default=1,
        help="Tensor parallel size"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0-1)"
    )
    parser.add_argument(
        "--backend",
        choices=["hf", "vllm"],
        default="hf",
        help="Verifier backend (Phase 1: hf, Phase 2: vllm)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help=(
            "Maximum sequence length. When omitted, the backend uses the model's "
            "native max_position_embeddings."
        ),
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    asyncio.run(run_server(
        model=args.model,
        listen_address=args.listen_address,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        backend=args.backend,
        max_model_len=args.max_model_len,
    ))


if __name__ == "__main__":
    main()
