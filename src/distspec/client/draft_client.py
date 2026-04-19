# SPDX-License-Identifier: Apache-2.0
"""Draft client: streaming generation over a ZMQ DEALER socket.

Pairs a ``BaseDraftProposer`` with network I/O and an adaptive controller that
tunes the speculation length K based on observed RTT and acceptance rate.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from collections.abc import AsyncGenerator

try:
    import zmq
    import zmq.asyncio
    _HAS_ZMQ = True
except ImportError:
    _HAS_ZMQ = False

try:
    from transformers import AutoTokenizer
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

from ..common.config import ClientConfig
from ..common.protocol import (
    DraftRequest,
    HealthCheck,
    HealthResponse,
    MsgpackDecoder,
    MsgpackEncoder,
    SamplingParams,
    VerifyResponse,
)
from .draft_proposer import BaseDraftProposer, create_draft_proposer

logger = logging.getLogger(__name__)


# ============================================================================
# Adaptive Speculation Controller
# ============================================================================

class AdaptiveSpeculationController:
    """적응형 Speculation Length 컨트롤러

    네트워크 지연(RTT)과 수락률에 따라 최적의 K(Speculation Length)를 동적으로 조정

    최적 K 공식:
        K* = argmax_k { E[accepted] / amortized_cost }

    여기서:
        E[accepted] = (1 - α^k) / (1 - α)  (α = 수락률)
        amortized_cost = k + RTT / decode_time
    """

    def __init__(
        self,
        min_spec_tokens: int = 1,
        max_spec_tokens: int = 10,
        history_size: int = 20,
        decode_time_estimate: float = 0.01,
    ):
        """
        Args:
            min_spec_tokens: 최소 Speculation 토큰 수
            max_spec_tokens: 최대 Speculation 토큰 수
            history_size: 히스토리 크기
            decode_time_estimate: 토큰당 디코드 시간 추정 (초)
        """
        self.min_spec_tokens = min_spec_tokens
        self.max_spec_tokens = max_spec_tokens
        self.decode_time_estimate = decode_time_estimate

        # 히스토리
        self.latency_history = deque(maxlen=history_size)
        self.acceptance_history = deque(maxlen=history_size)

        # 현재 K
        self._current_k = max_spec_tokens

    @property
    def current_k(self) -> int:
        """현재 Speculation Length"""
        return self._current_k

    def record_result(
        self,
        rtt: float,
        num_draft: int,
        num_accepted: int,
    ) -> None:
        """결과 기록

        Args:
            rtt: Round-Trip Time (초)
            num_draft: Draft 토큰 수
            num_accepted: 수락된 토큰 수
        """
        self.latency_history.append(rtt)
        acceptance_rate = num_accepted / num_draft if num_draft > 0 else 0
        self.acceptance_history.append(acceptance_rate)

        # K 업데이트
        self._current_k = self._compute_optimal_k()

    def _compute_optimal_k(self) -> int:
        """최적 Speculation Length 계산"""
        if len(self.latency_history) < 5:
            return self.max_spec_tokens

        import numpy as np

        avg_rtt = np.mean(self.latency_history)
        avg_acceptance = np.mean(self.acceptance_history)

        if avg_acceptance < 0.1:
            return self.min_spec_tokens

        best_k = self.min_spec_tokens
        best_score = 0

        for k in range(self.min_spec_tokens, self.max_spec_tokens + 1):
            # 기대 수락 토큰 수
            if avg_acceptance >= 1.0:
                expected_accepted = k
            else:
                expected_accepted = (1 - avg_acceptance ** k) / (1 - avg_acceptance)

            # Amortized cost
            amortized_cost = k + avg_rtt / self.decode_time_estimate

            # Score
            score = expected_accepted / amortized_cost

            if score > best_score:
                best_score = score
                best_k = k

        return best_k

    def reset(self) -> None:
        """상태 초기화"""
        self.latency_history.clear()
        self.acceptance_history.clear()
        self._current_k = self.max_spec_tokens


# ============================================================================
# Draft Client
# ============================================================================

class DraftClient:
    """Draft 클라이언트

    Server와 ZMQ로 통신하여 Speculative Decoding 수행

    동작 흐름:
    1. Draft Proposer로 Draft 토큰 생성
    2. Server에 검증 요청 전송
    3. 검증 결과 수신 (수락된 토큰 + Bonus 토큰)
    4. 상태 업데이트 및 반복
    """

    def __init__(
        self,
        config: ClientConfig,
        draft_proposer: BaseDraftProposer | None = None,
        client_id: str | None = None,
    ):
        """
        Args:
            config: 클라이언트 설정
            draft_proposer: Draft Proposer (None이면 자동 생성)
            client_id: 클라이언트 ID (None이면 자동 생성)
        """
        if not _HAS_ZMQ:
            raise RuntimeError(
                "DraftClient requires pyzmq. Install with: pip install pyzmq"
            )

        self.config = config
        self.client_id = client_id or f"client_{uuid.uuid4().hex[:8]}"

        # Draft Proposer
        if draft_proposer is not None:
            self.draft_proposer = draft_proposer
        else:
            self.draft_proposer = create_draft_proposer(
                method=config.draft_method,
                num_speculative_tokens=config.num_speculative_tokens,
            )

        # Tokenizer
        self._tokenizer = None

        # Adaptive Controller
        if config.adaptive_speculation:
            self.adaptive_controller = AdaptiveSpeculationController(
                min_spec_tokens=config.min_spec_tokens,
                max_spec_tokens=config.max_spec_tokens,
            )
        else:
            self.adaptive_controller = None

        # ZMQ 설정
        self.context: zmq.asyncio.Context | None = None
        self.socket: zmq.asyncio.Socket | None = None
        self._connected = False

        # 메시지 인코더/디코더
        self.encoder = MsgpackEncoder()
        self.decoder = MsgpackDecoder(VerifyResponse)

        # 상태
        self._request_count = 0

    @property
    def tokenizer(self):
        """토크나이저 (lazy loading)"""
        if self._tokenizer is None:
            if not _HAS_TRANSFORMERS:
                raise RuntimeError(
                    "Tokenizer requires transformers. "
                    "Install with: pip install transformers"
                )

            tokenizer_name = self.config.tokenizer_name or self.config.draft_model
            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=True,
            )
        return self._tokenizer

    async def connect(self) -> None:
        """Server에 연결"""
        if self._connected:
            return

        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.setsockopt_string(zmq.IDENTITY, self.client_id)
        self.socket.setsockopt(zmq.RCVTIMEO, int(self.config.timeout * 1000))
        self.socket.setsockopt(zmq.SNDTIMEO, int(self.config.timeout * 1000))

        server_addr = f"tcp://{self.config.server_address}"
        self.socket.connect(server_addr)
        self._connected = True

        logger.info("Connected to server at %s", server_addr)

    async def disconnect(self) -> None:
        """Server 연결 해제"""
        if not self._connected:
            return

        if self.socket is not None:
            self.socket.close()
            self.socket = None

        if self.context is not None:
            self.context.term()
            self.context = None

        self._connected = False

    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
    ) -> AsyncGenerator[str, None]:
        """텍스트 생성 (스트리밍)

        Args:
            prompt: 입력 프롬프트
            sampling_params: 샘플링 파라미터

        Yields:
            생성된 토큰 (디코딩됨)
        """
        # 연결 확인
        if not self._connected:
            await self.connect()

        # 기본 샘플링 파라미터
        if sampling_params is None:
            sampling_params = SamplingParams()

        # 요청 ID
        request_id = f"{self.client_id}_{self._request_count}"
        self._request_count += 1

        # 토큰화
        prompt_tokens = self.tokenizer.encode(prompt)
        context_tokens = prompt_tokens.copy()
        hidden_states = None

        # 생성 루프
        generated_count = 0

        while generated_count < sampling_params.max_tokens:
            # 1. Speculation Length 결정
            if self.adaptive_controller is not None:
                spec_length = self.adaptive_controller.current_k
                self.draft_proposer.num_speculative_tokens = spec_length
            else:
                spec_length = self.config.num_speculative_tokens

            # 2. Draft 토큰 생성
            start_time = time.time()

            draft_output = self.draft_proposer.propose(
                context_tokens=context_tokens,
                hidden_states=hidden_states,
                sampling_params=sampling_params,
            )

            if not draft_output.draft_tokens:
                # Draft 실패 시 종료
                break

            # 3. Server에 검증 요청
            request = DraftRequest(
                request_id=request_id,
                prompt_tokens=prompt_tokens if generated_count == 0 else [],
                draft_tokens=draft_output.draft_tokens,
                draft_probs=draft_output.draft_probs,
                sampling_params=sampling_params,
                kv_cache_info=draft_output.kv_cache_info,
            )

            try:
                # DEALER: prepend empty delimiter to match ROUTER's 3-frame
                # reply and stay REQ-REP-compatible.
                await self.socket.send_multipart(
                    [b"", self.encoder.encode(request)]
                )
                reply_frames = await self.socket.recv_multipart()
                response_bytes = reply_frames[-1]
                response = self.decoder.decode(response_bytes)
            except zmq.error.Again:
                # 타임아웃
                logger.warning("Request timeout, retrying...")
                continue

            rtt = time.time() - start_time

            # 4. 적응형 컨트롤러 업데이트
            if self.adaptive_controller is not None:
                self.adaptive_controller.record_result(
                    rtt=rtt,
                    num_draft=len(draft_output.draft_tokens),
                    num_accepted=response.num_accepted,
                )

            # 5. 수락된 토큰 yield
            for token_id in response.accepted_tokens:
                yield self.tokenizer.decode([token_id])
                context_tokens.append(token_id)
                generated_count += 1

            # 6. Bonus 토큰 yield
            if response.bonus_token is not None:
                yield self.tokenizer.decode([response.bonus_token])
                context_tokens.append(response.bonus_token)
                generated_count += 1

            # 7. Hidden States 업데이트
            hidden_states = response.hidden_states

            # 8. 종료 확인
            if response.finished:
                break

            # EOS 확인
            if context_tokens and context_tokens[-1] == self.tokenizer.eos_token_id:
                break

    async def generate_batch(
        self,
        prompts: list[str],
        sampling_params: SamplingParams | None = None,
    ) -> list[str]:
        """배치 텍스트 생성

        Args:
            prompts: 입력 프롬프트 리스트
            sampling_params: 샘플링 파라미터

        Returns:
            생성된 텍스트 리스트
        """
        results = []

        for prompt in prompts:
            tokens = []
            async for token in self.generate(prompt, sampling_params):
                tokens.append(token)
            results.append(''.join(tokens))

        return results

    async def health_check(self) -> bool:
        """Server 상태 확인

        Returns:
            Server가 정상이면 True
        """
        if not self._connected:
            await self.connect()

        health_check = HealthCheck(
            client_id=self.client_id,
            timestamp=time.time(),
        )

        try:
            await self.socket.send_multipart(
                [b"", self.encoder.encode(health_check)]
            )
            reply_frames = await self.socket.recv_multipart()
            response_bytes = reply_frames[-1]
            response = MsgpackDecoder(HealthResponse).decode(response_bytes)
            return response.is_healthy
        except Exception:
            return False

    def reset(self) -> None:
        """상태 초기화"""
        self.draft_proposer.reset()
        if self.adaptive_controller is not None:
            self.adaptive_controller.reset()
        self._request_count = 0

    async def __aenter__(self):
        """Context manager 진입"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        await self.disconnect()
