# SPDX-License-Identifier: Apache-2.0
"""Confidence-based verification optimization clients.

Three approaches, each wrapping a base ``DraftClient`` by composition:

  * :class:`ConfidenceSkipClient` — token-level skip (BiLD-style).
  * :class:`QueryRoutingClient` — query-level routing (RouteLLM-style).
  * :class:`AdaptiveWindowClient` — adaptive draft window (SVIP-style).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import AsyncGenerator

from ..common.confidence import (
    ConfidenceConfig,
    QueryClassifier,
)
from ..common.config import ClientConfig, VerificationMode
from ..common.protocol import (
    DraftOutput,
    DraftRequest,
    SamplingParams,
)
from .draft_client import DraftClient
from .draft_proposer import BaseDraftProposer

logger = logging.getLogger(__name__)


# ============================================================================
# Approach A: Token-level Confidence Skip (BiLD-style)
# ============================================================================

class ConfidenceSkipClient:
    """Token-level Confidence Skip Client (BiLD-style)

    Draft 토큰의 confidence가 threshold 이상이면 서버 검증을 건너뛴다.
    높은 confidence의 연속 prefix → 바로 yield (서버 전송 없음)
    나머지 토큰 → 서버로 전송하여 검증

    BiLD (Big Little Decoder) 논문의 핵심 아이디어:
    "작은 모델이 확신하는 토큰은 큰 모델이 동의할 가능성이 높다"

    동작:
      1. draft_proposer.propose() → draft_tokens + confidence_scores
      2. confident prefix 분리:
         - confidence_scores[i] >= threshold인 연속 prefix → skip_tokens
         - 나머지 → verify_tokens
      3. skip_tokens → 바로 yield (서버 전송 없음)
      4. verify_tokens → 서버로 전송 → VerifyResponse → yield
    """

    def __init__(
        self,
        config: ClientConfig,
        confidence_config: ConfidenceConfig | None = None,
        draft_proposer: BaseDraftProposer | None = None,
    ):
        self.config = config
        self.confidence_config = confidence_config or ConfidenceConfig()
        self.draft_client = DraftClient(config, draft_proposer)
        self.draft_proposer = self.draft_client.draft_proposer

        # 통계 추적
        self.total_tokens = 0
        self.skipped_tokens = 0
        self.verified_tokens = 0
        self.server_calls = 0
        self._step = 0

    @property
    def tokenizer(self):
        return self.draft_client.tokenizer

    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
    ) -> AsyncGenerator[str, None]:
        """Confidence Skip 방식 텍스트 생성"""
        if not self.draft_client._connected:
            await self.draft_client.connect()

        if sampling_params is None:
            sampling_params = SamplingParams()

        prompt_tokens = self.tokenizer.encode(prompt)
        context_tokens = prompt_tokens.copy()
        hidden_states = None
        generated_count = 0
        self._step = 0

        while generated_count < sampling_params.max_tokens:
            self._step += 1

            # Adaptive speculation length
            if self.draft_client.adaptive_controller is not None:
                spec_length = self.draft_client.adaptive_controller.current_k
                self.draft_proposer.num_speculative_tokens = spec_length

            # 1. Draft 토큰 생성
            draft_output = self.draft_proposer.propose(
                context_tokens=context_tokens,
                hidden_states=hidden_states,
                sampling_params=sampling_params,
            )

            if not draft_output.draft_tokens:
                break

            # 2. Confidence 기반 split
            skip_tokens, verify_tokens = self._split_by_confidence(
                draft_output
            )

            # 3. Skip tokens → 바로 yield (warmup 기간에는 skip 안 함)
            if skip_tokens and self._step > self.confidence_config.warmup_steps:
                for token_id in skip_tokens:
                    yield self.tokenizer.decode([token_id])
                    context_tokens.append(token_id)
                    generated_count += 1
                    self.skipped_tokens += 1
                    self.total_tokens += 1

                    if token_id == self.tokenizer.eos_token_id:
                        return

            # 4. Verify tokens → 서버 전송
            if verify_tokens:
                start_time = time.time()

                request = DraftRequest(
                    request_id=f"{self.draft_client.client_id}_{self.draft_client._request_count}",
                    prompt_tokens=prompt_tokens if generated_count == 0 and not skip_tokens else [],
                    draft_tokens=verify_tokens,
                    draft_probs=draft_output.draft_probs,
                    sampling_params=sampling_params,
                    kv_cache_info=draft_output.kv_cache_info,
                )
                self.draft_client._request_count += 1

                try:
                    await self.draft_client.socket.send_multipart(
                        [b"", self.draft_client.encoder.encode(request)]
                )
                    _reply = await self.draft_client.socket.recv_multipart()
                    response_bytes = _reply[-1]
                    response = self.draft_client.decoder.decode(response_bytes)
                except Exception:
                    break

                rtt = time.time() - start_time
                self.server_calls += 1

                # Adaptive controller 업데이트
                if self.draft_client.adaptive_controller is not None:
                    self.draft_client.adaptive_controller.record_result(
                        rtt=rtt,
                        num_draft=len(verify_tokens),
                        num_accepted=response.num_accepted,
                    )

                # 수락된 토큰 yield
                for token_id in response.accepted_tokens:
                    yield self.tokenizer.decode([token_id])
                    context_tokens.append(token_id)
                    generated_count += 1
                    self.verified_tokens += 1
                    self.total_tokens += 1

                # Bonus 토큰
                if response.bonus_token is not None:
                    yield self.tokenizer.decode([response.bonus_token])
                    context_tokens.append(response.bonus_token)
                    generated_count += 1
                    self.verified_tokens += 1
                    self.total_tokens += 1

                hidden_states = response.hidden_states

                if response.finished:
                    break

            # EOS 확인
            if context_tokens and context_tokens[-1] == self.tokenizer.eos_token_id:
                break

    def _split_by_confidence(
        self, draft_output: DraftOutput
    ) -> tuple:
        """Confidence 기반으로 skip/verify 토큰 분리

        Returns:
            (skip_tokens, verify_tokens)
        """
        scores = draft_output.confidence_scores
        tokens = draft_output.draft_tokens
        threshold = self.confidence_config.skip_threshold

        if scores is None or len(scores) == 0:
            return [], tokens

        # 연속 confident prefix 찾기
        prefix_len = 0
        for s in scores:
            if s >= threshold:
                prefix_len += 1
            else:
                break

        skip_tokens = tokens[:prefix_len]
        verify_tokens = tokens[prefix_len:]

        return skip_tokens, verify_tokens

    def get_stats(self) -> dict:
        """통계 반환"""
        return {
            "total_tokens": self.total_tokens,
            "skipped_tokens": self.skipped_tokens,
            "verified_tokens": self.verified_tokens,
            "server_calls": self.server_calls,
            "skip_rate": self.skipped_tokens / max(self.total_tokens, 1),
        }

    def reset(self):
        self.draft_client.reset()
        self.total_tokens = 0
        self.skipped_tokens = 0
        self.verified_tokens = 0
        self.server_calls = 0
        self._step = 0

    async def __aenter__(self):
        await self.draft_client.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.draft_client.disconnect()


# ============================================================================
# Approach B: Query-level Routing (RouteLLM-style)
# ============================================================================

class QueryRoutingClient:
    """Query-level Routing Client (RouteLLM-style)

    Prompt의 난이도를 사전 분류하여 검증 전략을 결정한다.

    "easy" → Draft Proposer만으로 생성 (서버 호출 0회)
    "medium" → 매 N step마다만 서버 검증 (간헐적)
    "hard" → 기존 speculative decoding (매번 검증)

    FrugalGPT, RouteLLM, AutoMix 논문의 아이디어:
    "모든 쿼리에 큰 모델이 필요한 것은 아니다"
    """

    def __init__(
        self,
        config: ClientConfig,
        confidence_config: ConfidenceConfig | None = None,
        draft_proposer: BaseDraftProposer | None = None,
        classifier: QueryClassifier | None = None,
        medium_verify_interval: int = 3,
    ):
        """
        Args:
            medium_verify_interval: "medium" 난이도에서 N step마다 검증
        """
        self.config = config
        self.confidence_config = confidence_config or ConfidenceConfig()
        self.draft_client = DraftClient(config, draft_proposer)
        self.draft_proposer = self.draft_client.draft_proposer
        self.classifier = classifier or QueryClassifier(
            easy_threshold=self.confidence_config.query_routing_threshold,
        )
        self.medium_verify_interval = medium_verify_interval

        # 통계
        self.total_tokens = 0
        self.server_calls = 0
        self.queries_by_difficulty = {"easy": 0, "medium": 0, "hard": 0}

    @property
    def tokenizer(self):
        return self.draft_client.tokenizer

    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
    ) -> AsyncGenerator[str, None]:
        """Query Routing 방식 텍스트 생성"""
        if sampling_params is None:
            sampling_params = SamplingParams()

        prompt_tokens = self.tokenizer.encode(prompt)

        # 1. Query 난이도 분류
        difficulty = self.classifier.classify(
            prompt_tokens=prompt_tokens,
            prompt_text=prompt,
        )
        self.queries_by_difficulty[difficulty] += 1

        # 2. 난이도에 따른 생성 전략
        if difficulty == "easy":
            async for token in self._generate_draft_only(
                prompt_tokens, sampling_params
            ):
                yield token
        elif difficulty == "medium":
            async for token in self._generate_intermittent(
                prompt, prompt_tokens, sampling_params
            ):
                yield token
        else:  # hard
            async for token in self._generate_full_verify(
                prompt, sampling_params
            ):
                yield token

    async def _generate_draft_only(
        self,
        prompt_tokens: list[int],
        sampling_params: SamplingParams,
    ) -> AsyncGenerator[str, None]:
        """Easy: Draft Proposer만으로 생성 (서버 호출 없음)"""
        context_tokens = prompt_tokens.copy()

        for _ in range(sampling_params.max_tokens):
            draft_output = self.draft_proposer.propose(
                context_tokens=context_tokens,
                sampling_params=sampling_params,
            )

            if not draft_output.draft_tokens:
                break

            # 첫 번째 토큰만 사용 (검증 없이)
            token_id = draft_output.draft_tokens[0]
            context_tokens.append(token_id)
            self.total_tokens += 1
            yield self.tokenizer.decode([token_id])

            if token_id == self.tokenizer.eos_token_id:
                break

    async def _generate_intermittent(
        self,
        prompt: str,
        prompt_tokens: list[int],
        sampling_params: SamplingParams,
    ) -> AsyncGenerator[str, None]:
        """Medium: 매 N step마다만 서버 검증"""
        if not self.draft_client._connected:
            await self.draft_client.connect()

        context_tokens = prompt_tokens.copy()
        hidden_states = None
        generated_count = 0
        step = 0

        while generated_count < sampling_params.max_tokens:
            step += 1

            draft_output = self.draft_proposer.propose(
                context_tokens=context_tokens,
                hidden_states=hidden_states,
                sampling_params=sampling_params,
            )

            if not draft_output.draft_tokens:
                break

            # N step마다만 서버 검증
            if step % self.medium_verify_interval == 0:
                # 서버에 검증 요청
                request = DraftRequest(
                    request_id=f"{self.draft_client.client_id}_{self.draft_client._request_count}",
                    prompt_tokens=prompt_tokens if generated_count == 0 else [],
                    draft_tokens=draft_output.draft_tokens,
                    draft_probs=draft_output.draft_probs,
                    sampling_params=sampling_params,
                    kv_cache_info=draft_output.kv_cache_info,
                )
                self.draft_client._request_count += 1

                try:
                    await self.draft_client.socket.send_multipart(
                        [b"", self.draft_client.encoder.encode(request)]
                )
                    _reply = await self.draft_client.socket.recv_multipart()
                    response_bytes = _reply[-1]
                    response = self.draft_client.decoder.decode(response_bytes)
                    self.server_calls += 1
                except Exception:
                    # 서버 실패 시 draft만 사용
                    token_id = draft_output.draft_tokens[0]
                    context_tokens.append(token_id)
                    generated_count += 1
                    self.total_tokens += 1
                    yield self.tokenizer.decode([token_id])
                    continue

                for token_id in response.accepted_tokens:
                    yield self.tokenizer.decode([token_id])
                    context_tokens.append(token_id)
                    generated_count += 1
                    self.total_tokens += 1

                if response.bonus_token is not None:
                    yield self.tokenizer.decode([response.bonus_token])
                    context_tokens.append(response.bonus_token)
                    generated_count += 1
                    self.total_tokens += 1

                hidden_states = response.hidden_states

                if response.finished:
                    break
            else:
                # 서버 없이 draft 토큰 직접 사용
                token_id = draft_output.draft_tokens[0]
                context_tokens.append(token_id)
                generated_count += 1
                self.total_tokens += 1
                yield self.tokenizer.decode([token_id])

            if context_tokens and context_tokens[-1] == self.tokenizer.eos_token_id:
                break

    async def _generate_full_verify(
        self,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> AsyncGenerator[str, None]:
        """Hard: 기존 speculative decoding (매번 검증)"""
        async for token in self.draft_client.generate(prompt, sampling_params):
            self.total_tokens += 1
            self.server_calls += 1
            yield token

    def get_stats(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "server_calls": self.server_calls,
            "queries_by_difficulty": dict(self.queries_by_difficulty),
        }

    def reset(self):
        self.draft_client.reset()
        self.total_tokens = 0
        self.server_calls = 0
        self.queries_by_difficulty = {"easy": 0, "medium": 0, "hard": 0}

    async def __aenter__(self):
        await self.draft_client.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.draft_client.disconnect()


# ============================================================================
# Approach C: Adaptive Draft Window (SVIP-style)
# ============================================================================

class AdaptiveWindowClient:
    """Adaptive Draft Window Client (SVIP-style)

    Confidence + acceptance history를 기반으로 speculation length K를 동적 조절한다.

    SVIP, SpecDec++ 논문의 핵심 아이디어:
    "고정 K가 아닌, 현재 생성 상황에 맞는 최적 K를 선택한다"

    동작:
      1. confidence + acceptance history로 current K를 결정
      2. K개 draft 토큰 생성 → 전체를 서버에 전송
      3. 서버 검증 결과로 히스토리 업데이트 → 다음 K 조절
         - confidence 높고 acceptance rate 높으면 → K 증가 (최대 2*base_K)
         - confidence 낮고 acceptance rate 낮으면 → K 감소 (최소 1)

    기존 trim 방식의 문제:
      K=5 생성 후 confident prefix만 전송 (종종 1개) → 서버호출 폭증 → RTT 병목
    Adaptive K 방식의 장점:
      K 자체를 조절 → 서버호출 수는 baseline과 유사, 서버 compute 최적화
    """

    HISTORY_WINDOW = 5

    def __init__(
        self,
        config: ClientConfig,
        confidence_config: ConfidenceConfig | None = None,
        draft_proposer: BaseDraftProposer | None = None,
    ):
        self.config = config
        self.confidence_config = confidence_config or ConfidenceConfig()
        self.draft_client = DraftClient(config, draft_proposer)
        self.draft_proposer = self.draft_client.draft_proposer

        # Adaptive K 상태
        self._base_k = config.num_speculative_tokens
        self._current_k = self._base_k
        self._min_k = 1
        self._max_k = self._base_k * 2
        self._confidence_history = deque(maxlen=self.HISTORY_WINDOW)
        self._acceptance_history = deque(maxlen=self.HISTORY_WINDOW)

        # 통계
        self.total_tokens = 0
        self.server_calls = 0

    @property
    def tokenizer(self):
        return self.draft_client.tokenizer

    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
    ) -> AsyncGenerator[str, None]:
        """Adaptive Window 방식 텍스트 생성"""
        if not self.draft_client._connected:
            await self.draft_client.connect()

        if sampling_params is None:
            sampling_params = SamplingParams()

        prompt_tokens = self.tokenizer.encode(prompt)
        context_tokens = prompt_tokens.copy()
        hidden_states = None
        generated_count = 0

        while generated_count < sampling_params.max_tokens:
            # Adaptive K 적용
            self.draft_proposer.num_speculative_tokens = self._current_k

            # 1. Draft 토큰 생성
            draft_output = self.draft_proposer.propose(
                context_tokens=context_tokens,
                hidden_states=hidden_states,
                sampling_params=sampling_params,
            )

            if not draft_output.draft_tokens:
                self._current_k = max(self._min_k, self._current_k - 1)
                break

            # 2. 전체 draft를 서버에 전송 (trim 하지 않음)
            start_time = time.time()

            request = DraftRequest(
                request_id=f"{self.draft_client.client_id}_{self.draft_client._request_count}",
                prompt_tokens=prompt_tokens if generated_count == 0 else [],
                draft_tokens=draft_output.draft_tokens,
                draft_probs=draft_output.draft_probs,
                sampling_params=sampling_params,
                kv_cache_info=draft_output.kv_cache_info,
            )
            self.draft_client._request_count += 1

            try:
                await self.draft_client.socket.send_multipart(
                    [b"", self.draft_client.encoder.encode(request)]
            )
                _reply = await self.draft_client.socket.recv_multipart()
                response_bytes = _reply[-1]
                response = self.draft_client.decoder.decode(response_bytes)
            except Exception:
                break

            rtt = time.time() - start_time
            self.server_calls += 1

            # 3. 히스토리 업데이트 → K 조절
            if draft_output.confidence_scores:
                avg_conf = sum(draft_output.confidence_scores) / len(draft_output.confidence_scores)
                self._confidence_history.append(avg_conf)

            acc_rate = response.num_accepted / len(draft_output.draft_tokens) if draft_output.draft_tokens else 0
            self._acceptance_history.append(acc_rate)

            self._update_k()

            # 4. 수락된 토큰 yield
            for token_id in response.accepted_tokens:
                yield self.tokenizer.decode([token_id])
                context_tokens.append(token_id)
                generated_count += 1
                self.total_tokens += 1

            if response.bonus_token is not None:
                yield self.tokenizer.decode([response.bonus_token])
                context_tokens.append(response.bonus_token)
                generated_count += 1
                self.total_tokens += 1

            hidden_states = response.hidden_states

            if response.finished:
                break

            if context_tokens and context_tokens[-1] == self.tokenizer.eos_token_id:
                break

    def _update_k(self):
        """Confidence + acceptance history 기반 K 업데이트"""
        if len(self._acceptance_history) < 2:
            return

        recent_acc = sum(self._acceptance_history) / len(self._acceptance_history)
        recent_conf = sum(self._confidence_history) / len(self._confidence_history) if self._confidence_history else 0.5

        combined = 0.6 * recent_acc + 0.4 * recent_conf

        if combined > 0.7:
            self._current_k = min(self._max_k, self._current_k + 1)
        elif combined < 0.3:
            self._current_k = max(self._min_k, self._current_k - 1)

    def get_stats(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "server_calls": self.server_calls,
            "current_k": self._current_k,
            "k_range": f"{self._min_k}~{self._max_k}",
        }

    def reset(self):
        self.draft_client.reset()
        self._current_k = self._base_k
        self._confidence_history.clear()
        self._acceptance_history.clear()
        self.total_tokens = 0
        self.server_calls = 0

    async def __aenter__(self):
        await self.draft_client.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.draft_client.disconnect()


# ============================================================================
# Factory Function
# ============================================================================

def create_confidence_client(
    mode: str,
    config: ClientConfig,
    confidence_config: ConfidenceConfig | None = None,
    draft_proposer: BaseDraftProposer | None = None,
    **kwargs,
):
    """Confidence Client 생성 팩토리 함수

    Args:
        mode: 검증 모드 ("confidence_skip" | "query_routing" | "adaptive_window")
        config: Client 설정
        confidence_config: Confidence 설정
        draft_proposer: Draft Proposer

    Returns:
        해당 모드의 Client 인스턴스
    """
    if mode == "confidence_skip" or mode == VerificationMode.CONFIDENCE_SKIP.value:
        return ConfidenceSkipClient(
            config=config,
            confidence_config=confidence_config,
            draft_proposer=draft_proposer,
        )
    elif mode == "query_routing" or mode == VerificationMode.QUERY_ROUTING.value:
        return QueryRoutingClient(
            config=config,
            confidence_config=confidence_config,
            draft_proposer=draft_proposer,
            medium_verify_interval=kwargs.get("medium_verify_interval", 3),
        )
    elif mode == "adaptive_window" or mode == VerificationMode.ADAPTIVE_WINDOW.value:
        return AdaptiveWindowClient(
            config=config,
            confidence_config=confidence_config,
            draft_proposer=draft_proposer,
        )
    else:
        raise ValueError(f"Unknown verification mode: {mode}")
