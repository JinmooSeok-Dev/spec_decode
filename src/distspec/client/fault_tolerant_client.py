# SPDX-License-Identifier: Apache-2.0
"""Fault-tolerant client with a three-state availability FSM.

See ``docs/ADAPTIVE_CONTROL.md`` for the state transitions. Falls back to
local draft-only generation when the target server is unreachable.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator

from ..common.config import ClientConfig, ClientMode
from ..common.protocol import SamplingParams
from .draft_client import DraftClient
from .draft_proposer import BaseDraftProposer, create_draft_proposer

logger = logging.getLogger(__name__)


class FaultTolerantClient:
    """장애 내성 클라이언트

    동작 모드:
    1. SPECULATIVE: 정상 Speculative Decoding (Client-Server)
    2. DEGRADED: 낮은 Speculation Length로 동작
    3. FALLBACK: Draft 모델만으로 생성 (Server 없이)

    모드 전환 조건:
    - SPECULATIVE → DEGRADED: 낮은 수락률 (<20%)
    - DEGRADED → FALLBACK: 연속 장애 (max_retries 초과)
    - FALLBACK → SPECULATIVE: Server 복구 확인
    """

    def __init__(
        self,
        config: ClientConfig,
        draft_proposer: BaseDraftProposer | None = None,
    ):
        """
        Args:
            config: 클라이언트 설정
            draft_proposer: Draft Proposer
        """
        self.config = config

        # Draft Client (Server 통신용)
        self.draft_client = DraftClient(config, draft_proposer)

        # Draft Proposer (Fallback용)
        if draft_proposer is not None:
            self.draft_proposer = draft_proposer
        else:
            self.draft_proposer = create_draft_proposer(
                method=config.draft_method,
                num_speculative_tokens=config.num_speculative_tokens,
            )

        # Tokenizer
        self._tokenizer = None

        # 상태
        self.mode = ClientMode.SPECULATIVE
        self.consecutive_failures = 0
        self.recent_acceptance_rates: list[float] = []

        # 설정
        self.max_retries = config.max_retries
        self.min_acceptance_rate = 0.2
        self.recovery_check_interval = 30.0  # 30초마다 복구 확인

        # 복구 태스크
        self._recovery_task: asyncio.Task | None = None

    @property
    def tokenizer(self):
        """토크나이저"""
        if self._tokenizer is None:
            self._tokenizer = self.draft_client.tokenizer
        return self._tokenizer

    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
    ) -> AsyncGenerator[str, None]:
        """장애 내성 텍스트 생성

        Args:
            prompt: 입력 프롬프트
            sampling_params: 샘플링 파라미터

        Yields:
            생성된 토큰
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        if self.mode == ClientMode.FALLBACK:
            # Fallback 모드: Draft 모델만 사용
            async for token in self._generate_fallback(prompt, sampling_params):
                yield token
        else:
            # Speculative/Degraded 모드
            try:
                async for token in self._generate_speculative(prompt, sampling_params):
                    yield token
                # 성공 시 실패 카운터 리셋
                self.consecutive_failures = 0
            except Exception as e:
                logger.warning("Speculative generation failed: %s", e)
                self._handle_failure()

                # Fallback 모드로 전환되었으면 Fallback으로 재시도
                if self.mode == ClientMode.FALLBACK:
                    async for token in self._generate_fallback(prompt, sampling_params):
                        yield token

    async def _generate_speculative(
        self,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> AsyncGenerator[str, None]:
        """Speculative Decoding 생성"""
        async for token in self.draft_client.generate(prompt, sampling_params):
            yield token

            # 수락률 기록 (Degraded 모드 전환 판단용)
            if self.draft_client.adaptive_controller is not None:
                history = self.draft_client.adaptive_controller.acceptance_history
                if history:
                    self.recent_acceptance_rates.append(history[-1])

                    # 최근 10회 평균 수락률 확인
                    if len(self.recent_acceptance_rates) >= 10:
                        avg_rate = sum(self.recent_acceptance_rates[-10:]) / 10
                        if avg_rate < self.min_acceptance_rate:
                            self._switch_to_degraded()

    async def _generate_fallback(
        self,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> AsyncGenerator[str, None]:
        """Fallback 생성 (Draft 모델만)

        Server 없이 Draft 모델만으로 토큰 생성
        품질은 낮지만 가용성 보장
        """
        context_tokens = self.tokenizer.encode(prompt)

        for _ in range(sampling_params.max_tokens):
            # Draft Proposer로 1개 토큰 생성
            draft_output = self.draft_proposer.propose(
                context_tokens=context_tokens,
                hidden_states=None,
                sampling_params=sampling_params,
            )

            # 첫 번째 Draft 토큰만 사용
            if draft_output.draft_tokens:
                token_id = draft_output.draft_tokens[0]
                context_tokens.append(token_id)
                yield self.tokenizer.decode([token_id])

                # EOS 확인
                if token_id == self.tokenizer.eos_token_id:
                    break
            else:
                # Draft 실패
                break

    def _handle_failure(self) -> None:
        """장애 처리"""
        self.consecutive_failures += 1

        if self.consecutive_failures >= self.max_retries:
            if self.mode == ClientMode.SPECULATIVE:
                self._switch_to_degraded()
            elif self.mode == ClientMode.DEGRADED:
                self._switch_to_fallback()

    def _switch_to_degraded(self) -> None:
        """Degraded 모드로 전환"""
        if self.mode != ClientMode.DEGRADED:
            logger.info("Switching to DEGRADED mode (low speculation length)")
            self.mode = ClientMode.DEGRADED

            # Speculation Length 감소
            if self.draft_client.adaptive_controller is not None:
                self.draft_client.adaptive_controller._current_k = \
                    self.config.min_spec_tokens

    def _switch_to_fallback(self) -> None:
        """Fallback 모드로 전환"""
        if self.mode != ClientMode.FALLBACK:
            logger.warning("Switching to FALLBACK mode (draft-only)")
            self.mode = ClientMode.FALLBACK

            # 복구 태스크 시작
            self._start_recovery_task()

    def _switch_to_speculative(self) -> None:
        """Speculative 모드로 전환"""
        logger.info("Switching back to SPECULATIVE mode")
        self.mode = ClientMode.SPECULATIVE
        self.consecutive_failures = 0
        self.recent_acceptance_rates.clear()

        # 상태 초기화
        self.draft_client.reset()

    def _start_recovery_task(self) -> None:
        """복구 태스크 시작"""
        if self._recovery_task is None or self._recovery_task.done():
            self._recovery_task = asyncio.create_task(self._recovery_loop())

    async def _recovery_loop(self) -> None:
        """복구 루프

        주기적으로 Server 상태를 확인하고 복구되면 Speculative 모드로 전환
        """
        while self.mode == ClientMode.FALLBACK:
            await asyncio.sleep(self.recovery_check_interval)

            try:
                is_healthy = await self.draft_client.health_check()
                if is_healthy:
                    self._switch_to_speculative()
                    break
            except Exception:
                # 복구 실패, 계속 대기
                pass

    async def close(self) -> None:
        """클라이언트 종료"""
        if self._recovery_task is not None:
            self._recovery_task.cancel()
            try:
                await self._recovery_task
            except asyncio.CancelledError:
                pass

        await self.draft_client.disconnect()

    def reset(self) -> None:
        """상태 초기화"""
        self.mode = ClientMode.SPECULATIVE
        self.consecutive_failures = 0
        self.recent_acceptance_rates.clear()
        self.draft_client.reset()
        self.draft_proposer.reset()

    async def __aenter__(self):
        """Context manager 진입"""
        await self.draft_client.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        await self.close()


# ============================================================================
# Convenience Functions
# ============================================================================

async def create_client(
    server_address: str = "localhost:8000",
    draft_method: str = "ngram",
    fault_tolerant: bool = True,
    **kwargs,
) -> FaultTolerantClient:
    """클라이언트 생성 편의 함수

    Args:
        server_address: Server 주소
        draft_method: Draft 방법
        fault_tolerant: 장애 내성 모드 사용
        **kwargs: 추가 설정

    Returns:
        FaultTolerantClient 또는 DraftClient
    """
    config = ClientConfig(
        server_address=server_address,
        draft_method=draft_method,
        **kwargs,
    )

    if fault_tolerant:
        client = FaultTolerantClient(config)
    else:
        client = DraftClient(config)

    await client.draft_client.connect() if hasattr(client, 'draft_client') else await client.connect()

    return client
