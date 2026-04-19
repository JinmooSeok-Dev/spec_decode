# SPDX-License-Identifier: Apache-2.0
"""HuggingFace-backed target verifier.

Phase 1 reference implementation of :class:`BaseVerifier`. Runs the target
model via ``transformers.AutoModelForCausalLM`` and performs rejection
sampling in pure PyTorch. Intended to be swapped for
:class:`distspec.server.vllm_verifier.VllmVerifier` in Phase 2.
"""

from __future__ import annotations

import logging

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

from ..common.protocol import SamplingParams, VerifyOutput
from ..common.sampling import apply_sampling_filters
from .base import BaseVerifier, BatchRequest

logger = logging.getLogger(__name__)


# ============================================================================
# Rejection Sampler
# ============================================================================

class RejectionSampler:
    """Rejection Sampler

    Speculative Decoding의 핵심 컴포넌트로,
    Draft 토큰을 검증하고 Target 분포를 정확히 보존

    알고리즘:
    1. Greedy 모드:
       - argmax(target) == draft → Accept
       - 불일치 시 → argmax(target) 반환

    2. Random 모드:
       - U ~ Uniform(0, 1)
       - U < target_prob / draft_prob → Accept
       - Reject 시 → Recovered 분포에서 샘플링

    수학적 보장:
    - 출력 분포 = Target 분포 (증명됨)
    - Recovered 분포: max(0, π_t - π_d) / Z
    """

    def __init__(
        self,
        vocab_size: int,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
    ):
        """
        Args:
            vocab_size: Vocabulary size.
            device: Torch device string.
            dtype: Torch dtype. Defaults to ``torch.float32`` when torch is
                available; ignored otherwise (sampler requires torch).
        """
        self.vocab_size = vocab_size
        self.device = device
        self.dtype = dtype if dtype is not None else (
            torch.float32 if _HAS_TORCH else None
        )

    def forward(
        self,
        target_logits: torch.Tensor,
        draft_tokens: list[int],
        draft_probs: torch.Tensor | None,
        sampling_params: SamplingParams,
    ) -> VerifyOutput:
        """Rejection Sampling 수행

        Args:
            target_logits: Target 모델 logits [num_draft, vocab_size]
            draft_tokens: Draft 토큰 리스트
            draft_probs: Draft 확률 분포 [num_draft, vocab_size] (EAGLE)
                         None이면 uniform 가정 (N-gram)
            sampling_params: 샘플링 파라미터

        Returns:
            VerifyOutput: 검증 결과
        """
        if not _HAS_TORCH:
            raise RuntimeError("RejectionSampler requires PyTorch")

        num_draft = len(draft_tokens)

        if sampling_params.is_greedy:
            return self._greedy_verify(target_logits, draft_tokens)
        else:
            return self._random_verify(
                target_logits,
                draft_tokens,
                draft_probs,
                sampling_params,
            )

    def _greedy_verify(
        self,
        target_logits: torch.Tensor,
        draft_tokens: list[int],
    ) -> VerifyOutput:
        """Greedy 모드 검증

        argmax(target) == draft 비교
        """
        accepted_tokens = []
        bonus_token = None

        for i, draft_token in enumerate(draft_tokens):
            target_token = target_logits[i].argmax().item()

            if target_token == draft_token:
                accepted_tokens.append(draft_token)
            else:
                # 거절: Target 토큰 반환
                bonus_token = target_token
                break

        # 모두 수락된 경우 bonus token 계산
        if len(accepted_tokens) == len(draft_tokens):
            # 마지막 다음 토큰
            if len(target_logits) > len(draft_tokens):
                bonus_token = target_logits[len(draft_tokens)].argmax().item()

        return VerifyOutput(
            accepted_tokens=accepted_tokens,
            bonus_token=bonus_token,
        )

    def _random_verify(
        self,
        target_logits: torch.Tensor,
        draft_tokens: list[int],
        draft_probs: torch.Tensor | None,
        sampling_params: SamplingParams,
    ) -> VerifyOutput:
        """Random 모드 검증 (Rejection Sampling)

        수락 기준: U < target_prob / draft_prob
        거절 시: Recovered 분포에서 샘플링
        """
        accepted_tokens = []
        bonus_token = None

        # Temperature 적용
        target_logits = target_logits / sampling_params.temperature

        # Top-k, Top-p 필터링
        target_logits = apply_sampling_filters(
            target_logits,
            top_k=sampling_params.top_k,
            top_p=sampling_params.top_p,
        )

        # Target 확률 계산
        target_probs = F.softmax(target_logits, dim=-1)

        # Draft 확률 (없으면 uniform)
        if draft_probs is None:
            # N-gram: uniform 분포 가정
            draft_probs = torch.ones_like(target_probs) / self.vocab_size
        else:
            draft_probs = draft_probs.to(self.device)

        for i, draft_token in enumerate(draft_tokens):
            # 확률 추출
            p_target = target_probs[i, draft_token].item()
            p_draft = draft_probs[i, draft_token].item()

            # 수락 확률: min(1, p_target / p_draft)
            if p_draft > 0:
                accept_prob = min(1.0, p_target / p_draft)
            else:
                accept_prob = 1.0 if p_target > 0 else 0.0

            # 랜덤 수락/거절
            u = torch.rand(1).item()

            if u < accept_prob:
                accepted_tokens.append(draft_token)
            else:
                # Rejection: Recovered 분포에서 샘플링
                recovered_token = self._sample_from_recovered(
                    target_probs[i],
                    draft_probs[i],
                )
                bonus_token = recovered_token
                break

        # 모두 수락된 경우 bonus token 계산
        if len(accepted_tokens) == len(draft_tokens):
            if len(target_logits) > len(draft_tokens):
                # 다음 위치에서 샘플링
                next_probs = F.softmax(target_logits[len(draft_tokens)], dim=-1)
                bonus_token = torch.multinomial(next_probs, 1).item()

        return VerifyOutput(
            accepted_tokens=accepted_tokens,
            bonus_token=bonus_token,
        )

    def _sample_from_recovered(
        self,
        target_probs: torch.Tensor,
        draft_probs: torch.Tensor,
    ) -> int:
        """Recovered 분포에서 샘플링

        Recovered 분포: max(0, π_t - π_d) / Z

        이 분포에서 샘플링하면 전체 출력이 Target 분포와 동일해짐
        """
        # Recovered 분포 계산
        recovered = torch.clamp(target_probs - draft_probs, min=0)

        # 정규화
        z = recovered.sum()
        if z > 0:
            recovered = recovered / z
            return torch.multinomial(recovered, 1).item()
        else:
            # Fallback: Target 분포에서 샘플링
            return torch.multinomial(target_probs, 1).item()


# ============================================================================
# Target Verifier
# ============================================================================

class HfVerifier(BaseVerifier):
    """HuggingFace-backed target verifier (Phase 1 reference).

    Loads the target model via ``transformers`` and verifies draft tokens

    동작:
    1. Draft 토큰들을 입력으로 Target 모델 Forward
    2. 생성된 Logits로 Rejection Sampling
    3. 수락된 토큰 + Bonus 토큰 반환
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        device: str = "cuda",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ):
        """
        Args:
            model_name: Target 모델 이름
            device: 디바이스
            tensor_parallel_size: Tensor Parallel 크기
            gpu_memory_utilization: GPU 메모리 사용률
        """
        self.model_name = model_name
        self.device = device
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization

        # 모델 (lazy loading)
        self._model = None
        self._tokenizer = None

        # Rejection Sampler (vocab_size는 모델 로드 후 설정)
        self._rejection_sampler = None

        # KV Cache
        self._kv_cache = None

    @property
    def model(self):
        """Target 모델 (lazy loading)"""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        """토크나이저 (lazy loading)"""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    @property
    def rejection_sampler(self) -> RejectionSampler:
        """Rejection Sampler (lazy loading)"""
        if self._rejection_sampler is None:
            vocab_size = self.model.config.vocab_size
            self._rejection_sampler = RejectionSampler(
                vocab_size=vocab_size,
                device=self.device,
            )
        return self._rejection_sampler

    def _load_model(self):
        """Target 모델 로드"""
        if not _HAS_TORCH:
            raise RuntimeError("HfVerifier requires PyTorch")

        if not _HAS_TRANSFORMERS:
            raise RuntimeError("HfVerifier requires transformers")

        logger.info("Loading target model: %s", self.model_name)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        # Load to host first, then move to the target device. ``device_map``
        # is reserved for multi-GPU sharding (requires the ``accelerate``
        # package); single-device loads are simpler and dependency-free.
        dtype = torch.float16 if str(self.device).startswith("cuda") else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self._model.to(self.device)
        self._model.eval()

    def verify(
        self,
        draft_tokens: list[int],
        draft_probs: torch.Tensor | None,
        context_tokens: list[int],
        sampling_params: SamplingParams,
    ) -> VerifyOutput:
        """Draft 토큰 검증

        Args:
            draft_tokens: Draft 토큰 리스트
            draft_probs: Draft 확률 분포
            context_tokens: Context 토큰 (프롬프트 + 이전 생성)
            sampling_params: 샘플링 파라미터

        Returns:
            VerifyOutput: 검증 결과
        """
        if not draft_tokens:
            return VerifyOutput()

        # 입력 준비: context + draft
        all_tokens = context_tokens + draft_tokens
        input_ids = torch.tensor([all_tokens], device=self.device)

        with torch.no_grad():
            # Full forward over context + draft. ``use_cache=False`` keeps the
            # Phase 1 code HF-cache-format-agnostic; KV reuse is the job of the
            # Phase 2 vLLM backend (PagedAttention via ``KVCacheManager``).
            outputs = self.model(
                input_ids=input_ids,
                use_cache=False,
                return_dict=True,
                output_hidden_states=True,
            )

            # logits[0, pos, :] predicts the token at pos + 1. For |context|=L
            # and |draft|=K we need positions [L-1 .. L+K-1] inclusive
            # (K verify positions + 1 bonus).
            start_idx = len(context_tokens) - 1
            end_idx = start_idx + len(draft_tokens) + 1
            target_logits = outputs.logits[0, start_idx:end_idx, :]

            # Last-layer hidden state at the final attended position.
            hidden_states = None
            if outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1][0, -1, :]

        verify_output = self.rejection_sampler.forward(
            target_logits=target_logits,
            draft_tokens=draft_tokens,
            draft_probs=draft_probs,
            sampling_params=sampling_params,
        )
        verify_output.hidden_states = hidden_states
        return verify_output

    def _truncate_kv_cache(
        self,
        kv_cache: tuple,
        keep_len: int,
    ) -> tuple:
        """KV Cache 자르기

        거절된 토큰의 KV Cache 제거
        """
        truncated = []
        for layer_kv in kv_cache:
            k, v = layer_kv
            truncated.append((
                k[:, :, :keep_len, :],
                v[:, :, :keep_len, :],
            ))
        return tuple(truncated)

    def reset(self) -> None:
        """상태 초기화"""
        self._kv_cache = None

    def verify_batch(self, requests: list[BatchRequest]) -> list[VerifyOutput]:
        """True batched target forward across many client requests.

        Pads each ``context + draft`` sequence to the batch max, runs a single
        forward with an attention mask, then feeds each request's slice of
        logits into the (per-request) rejection sampler.

        Notes
        -----
        * KV cache reuse is disabled for the batched path — mixing sequences
          with different prefixes in one ``past_key_values`` is not supported
          by vanilla HF. The per-request path (``verify()``) still reuses
          KV. The vLLM backend (Phase 2) removes this trade-off via
          PagedAttention.
        * Padding uses ``tokenizer.pad_token_id`` if set, else ``eos_token_id``
          if set, else 0. Only attended positions influence the output.
        """
        if not requests:
            return []

        seqs = [r.context_tokens + r.draft_tokens for r in requests]
        max_len = max(len(s) for s in seqs)
        batch_size = len(seqs)

        pad_id = (
            getattr(self.tokenizer, "pad_token_id", None)
            or getattr(self.tokenizer, "eos_token_id", None)
            or 0
        )

        input_ids = torch.full(
            (batch_size, max_len), pad_id, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros(
            (batch_size, max_len), dtype=torch.long, device=self.device
        )
        for i, seq in enumerate(seqs):
            input_ids[i, : len(seq)] = torch.tensor(
                seq, dtype=torch.long, device=self.device
            )
            attention_mask[i, : len(seq)] = 1

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
                output_hidden_states=True,
            )

        # logits[i, pos, :] is the distribution for position pos + 1.
        # For request i with |context|=L and |draft|=K, we need positions
        # [L-1 .. L+K-1] inclusive — that's K verify positions + 1 bonus.
        all_logits = outputs.logits
        last_hidden = (
            outputs.hidden_states[-1] if outputs.hidden_states is not None else None
        )

        results: list[VerifyOutput] = []
        for i, req in enumerate(requests):
            ctx_len = len(req.context_tokens)
            n_draft = len(req.draft_tokens)
            target_logits = all_logits[i, ctx_len - 1 : ctx_len + n_draft, :]

            verify_out = self.rejection_sampler.forward(
                target_logits=target_logits,
                draft_tokens=req.draft_tokens,
                draft_probs=req.draft_probs,
                sampling_params=req.sampling_params,
            )
            if last_hidden is not None:
                # Hidden state at the last non-padded position of this row.
                verify_out.hidden_states = last_hidden[i, ctx_len + n_draft - 1, :]
            results.append(verify_out)

        return results


class BatchVerifier:
    """배치 검증기

    여러 요청을 배치로 처리하여 GPU 활용 최대화
    """

    def __init__(
        self,
        verifier: BaseVerifier,
        max_batch_size: int = 32,
        max_wait_time: float = 0.005,
    ):
        """
        Args:
            verifier: Underlying :class:`BaseVerifier` instance.
            max_batch_size: 최대 배치 크기
            max_wait_time: 배치 대기 최대 시간 (초)
        """
        self.verifier = verifier
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time

        self.pending_requests: list[BatchRequest] = []
        self._lock = None  # asyncio.Lock (lazy init)

    async def add_request(self, request: BatchRequest) -> VerifyOutput:
        """요청 추가 및 배치 처리

        Args:
            request: 배치 요청

        Returns:
            검증 결과
        """
        import asyncio

        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            self.pending_requests.append(request)

            if len(self.pending_requests) >= self.max_batch_size:
                results = await self._process_batch()
                return results[-1]  # 마지막 추가된 요청의 결과

        # 타임아웃 대기
        await asyncio.sleep(self.max_wait_time)

        async with self._lock:
            if self.pending_requests:
                results = await self._process_batch()
                # 요청 ID로 결과 찾기
                for i, req in enumerate(self.pending_requests):
                    if req.request_id == request.request_id:
                        return results[i]

        return VerifyOutput()

    async def _process_batch(self) -> list[VerifyOutput]:
        """Run a single batched ``verify_batch`` call off the event loop."""
        import asyncio

        if not self.pending_requests:
            return []

        requests = self.pending_requests
        self.pending_requests = []

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            self.verifier.verify_batch,
            requests,
        )
        return results
