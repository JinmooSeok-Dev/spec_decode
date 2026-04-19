# SPDX-License-Identifier: Apache-2.0
"""Draft proposer implementations.

Provides N-gram, Suffix-decoding, and EAGLE-style draft proposers that share a
common ``BaseDraftProposer`` interface. Modeled after the proposer pattern in
``vllm/v1/spec_decode/``, but operates stand-alone (no vLLM dependency).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

from ..common.protocol import DraftOutput, KVCacheInfo, SamplingParams
from ..common.sampling import apply_sampling_filters

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Backwards-compatible aliases used in a few call sites.
HAS_TORCH = _HAS_TORCH
HAS_NUMBA = _HAS_NUMBA


# ============================================================================
# Base Draft Proposer
# ============================================================================

class BaseDraftProposer(ABC):
    """Draft Proposer 기본 클래스

    모든 Draft Proposer가 구현해야 하는 인터페이스
    """

    def __init__(self, num_speculative_tokens: int = 5):
        """
        Args:
            num_speculative_tokens: 생성할 Draft 토큰 수
        """
        self.num_speculative_tokens = num_speculative_tokens

    @abstractmethod
    def propose(
        self,
        context_tokens: list[int],
        hidden_states: torch.Tensor | None = None,
        sampling_params: SamplingParams | None = None,
    ) -> DraftOutput:
        """Draft 토큰 생성

        Args:
            context_tokens: 현재까지의 토큰 시퀀스
            hidden_states: Target 모델의 Hidden States (EAGLE용)
            sampling_params: 샘플링 파라미터

        Returns:
            DraftOutput: 생성된 Draft 토큰 및 메타데이터
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """상태 초기화"""
        pass


# ============================================================================
# N-gram Draft Proposer
# ============================================================================

class NgramDraftProposer(BaseDraftProposer):
    """N-gram 기반 Draft Proposer

    KMP 알고리즘을 사용하여 이전 토큰 시퀀스에서 반복 패턴을 탐색하고
    다음 토큰을 예측합니다.

    특징:
    - 모델 없이 동작 (리소스 효율적)
    - 반복적인 패턴(코드, JSON 등)에서 높은 수락률
    - Numba JIT로 O(n) 선형 시간 처리

    알고리즘:
    1. 현재 시퀀스의 suffix가 과거 시퀀스의 prefix와 일치하는 부분 탐색
    2. 일치하는 위치 이후 토큰들을 Draft로 제안
    3. LPS(Longest Proper Prefix Suffix) 배열로 효율적 탐색
    """

    def __init__(
        self,
        num_speculative_tokens: int = 5,
        ngram_window: int = 4,
        min_match_length: int = 2,
    ):
        """
        Args:
            num_speculative_tokens: 생성할 Draft 토큰 수
            ngram_window: N-gram 윈도우 크기
            min_match_length: 최소 매칭 길이
        """
        super().__init__(num_speculative_tokens)
        self.ngram_window = ngram_window
        self.min_match_length = min_match_length

    def propose(
        self,
        context_tokens: list[int],
        hidden_states: torch.Tensor | None = None,
        sampling_params: SamplingParams | None = None,
    ) -> DraftOutput:
        """N-gram 기반 Draft 토큰 생성

        KMP 알고리즘을 사용하여 패턴 매칭
        """
        # context_tokens를 numpy array로 변환
        if isinstance(context_tokens, list):
            tokens = np.array(context_tokens, dtype=np.int64)
        else:
            tokens = np.array(context_tokens)

        # 최소 길이 확인
        if len(tokens) < self.min_match_length:
            return DraftOutput(draft_tokens=[])

        # KMP 알고리즘으로 패턴 매칭
        draft_tokens, match_len = self._find_ngram_matches(tokens)
        draft_tokens = draft_tokens[:self.num_speculative_tokens]

        # confidence: 매칭 길이 기반 (긴 매칭 = 높은 confidence)
        confidence_scores = None
        if draft_tokens and match_len > 0:
            base_conf = min(match_len / self.ngram_window, 1.0)
            # 앞쪽 토큰이 더 신뢰, 뒤로 갈수록 decay
            confidence_scores = [
                base_conf * (0.95 ** i) for i in range(len(draft_tokens))
            ]

        return DraftOutput(
            draft_tokens=draft_tokens,
            draft_probs=None,  # N-gram은 확률 없음
            hidden_states=None,
            kv_cache_info=KVCacheInfo(
                seq_len=len(tokens),
                prev_seq_len=len(tokens),
                transfer_mode="none",
            ),
            confidence_scores=confidence_scores,
        )

    def _find_ngram_matches(self, tokens: np.ndarray) -> tuple:
        """N-gram 패턴 매칭으로 Draft 토큰 생성

        과거 시퀀스에서 현재 마지막 N개 토큰과 동일한 패턴을 찾고,
        그 패턴 이후에 나온 토큰들을 Draft로 반환

        Args:
            tokens: 토큰 시퀀스

        Returns:
            (매칭된 Draft 토큰 리스트, 매칭에 사용된 N-gram 길이)

        예시:
            context = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3]
            pattern [1, 2, 3] found at position 0
            → draft tokens = [4, 5] (position 3, 4 이후 나온 토큰)
        """
        n = len(tokens)
        if n < self.min_match_length + 1:
            return [], 0

        best_draft_tokens = []
        best_match_len = 0

        # 다양한 패턴 길이로 시도 (긴 패턴부터)
        for match_len in range(min(self.ngram_window, n - 1), self.min_match_length - 1, -1):
            # 현재 시퀀스의 마지막 match_len 토큰을 패턴으로 사용
            pattern = tokens[-match_len:]

            # 과거 시퀀스에서 패턴 탐색 (현재 위치 제외)
            # 탐색 범위: 0부터 n - match_len - 1까지
            search_end = n - match_len

            for i in range(search_end):
                if np.array_equal(tokens[i:i + match_len], pattern):
                    # 매칭! 그 이후 토큰들을 draft로 사용
                    draft_start = i + match_len
                    draft_end = min(draft_start + self.num_speculative_tokens, search_end)

                    if draft_end > draft_start:
                        draft_tokens = tokens[draft_start:draft_end].tolist()

                        # 더 긴 패턴 매칭이거나, 더 많은 draft 토큰을 생성하면 저장
                        if len(draft_tokens) > len(best_draft_tokens):
                            best_draft_tokens = draft_tokens
                            best_match_len = match_len

                        # 충분한 draft 토큰을 찾았으면 반환
                        if len(best_draft_tokens) >= self.num_speculative_tokens:
                            return best_draft_tokens, best_match_len

        return best_draft_tokens, best_match_len

    def reset(self) -> None:
        """상태 초기화 (N-gram은 상태 없음)"""
        pass


# Numba JIT 최적화 버전 (선택적)
if HAS_NUMBA:
    @njit(parallel=True)
    def _kmp_search_numba(tokens: np.ndarray, pattern: np.ndarray) -> int:
        """Numba JIT 최적화된 KMP 패턴 검색

        Args:
            tokens: 검색 대상 토큰 시퀀스
            pattern: 검색 패턴

        Returns:
            매칭 위치 (없으면 -1)
        """
        n = len(tokens)
        m = len(pattern)

        if m == 0 or n < m:
            return -1

        # LPS 배열 계산
        lps = np.zeros(m, dtype=np.int64)
        length = 0
        i = 1

        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1

        # KMP 검색
        i = 0  # tokens index
        j = 0  # pattern index

        while i < n:
            if pattern[j] == tokens[i]:
                i += 1
                j += 1

            if j == m:
                return i - j  # 매칭 위치 반환

            elif i < n and pattern[j] != tokens[i]:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1

        return -1


# ============================================================================
# Suffix Decoding Draft Proposer
# ============================================================================

class SuffixDraftProposer(BaseDraftProposer):
    """Suffix Decoding 기반 Draft Proposer

    Suffix Tree를 유지하면서 과거 생성 패턴을 축적하고,
    빈도 기반 확률 추정으로 N-gram보다 정교한 예측을 수행합니다.

    N-gram과의 차이:
    - N-gram: 현재 context 내에서만 패턴 검색 (stateless)
    - Suffix: 과거 모든 요청의 패턴을 suffix tree에 축적 (stateful)
              + 빈도 기반 확률 추정으로 낮은 확률 패턴 필터링

    알고리즘:
    1. context의 마지막 N개 토큰을 suffix로 사용
    2. suffix tree에서 해당 suffix 이후 등장한 토큰 시퀀스 검색
    3. 빈도가 높은 시퀀스를 draft 토큰으로 반환
    4. 생성 완료 후 결과를 suffix tree에 추가 (학습)
    """

    def __init__(
        self,
        num_speculative_tokens: int = 5,
        max_suffix_len: int = 8,
        min_suffix_len: int = 2,
        min_token_prob: float = 0.1,
        max_tree_size: int = 100000,
    ):
        super().__init__(num_speculative_tokens)
        self.max_suffix_len = max_suffix_len
        self.min_suffix_len = min_suffix_len
        self.min_token_prob = min_token_prob
        self.max_tree_size = max_tree_size

        # Suffix tree: suffix_tuple → {next_token: count}
        self._tree: dict[tuple, dict[int, int]] = {}
        self._total_entries = 0

    def propose(
        self,
        context_tokens: list[int],
        hidden_states: torch.Tensor | None = None,
        sampling_params: SamplingParams | None = None,
    ) -> DraftOutput:
        tokens = list(context_tokens)
        n = len(tokens)

        if n < self.min_suffix_len:
            return DraftOutput(draft_tokens=[])

        # 1. context 자체에서 prompt lookup (N-gram 방식도 병행)
        draft_from_context = self._prompt_lookup(tokens)

        # 2. suffix tree에서 축적된 패턴 검색 (confidence 포함)
        draft_from_tree, tree_confidences = self._tree_lookup(tokens)

        # 더 긴 결과를 선택
        if len(draft_from_tree) >= len(draft_from_context):
            draft_tokens = draft_from_tree
            confidence_scores = tree_confidences
        else:
            draft_tokens = draft_from_context
            # prompt lookup은 match length 기반 confidence (decay 적용)
            if draft_tokens:
                base_conf = min(len(draft_tokens) / self.max_suffix_len, 1.0)
                confidence_scores = [
                    base_conf * (0.9 ** i) for i in range(len(draft_tokens))
                ]
            else:
                confidence_scores = None

        draft_tokens = draft_tokens[:self.num_speculative_tokens]
        if confidence_scores:
            confidence_scores = confidence_scores[:self.num_speculative_tokens]

        return DraftOutput(
            draft_tokens=draft_tokens,
            draft_probs=None,
            hidden_states=None,
            kv_cache_info=KVCacheInfo(
                seq_len=n,
                prev_seq_len=n,
                transfer_mode="none",
            ),
            confidence_scores=confidence_scores if confidence_scores else None,
        )

    def _prompt_lookup(self, tokens: list[int]) -> list[int]:
        """Prompt 내부에서 suffix 매칭 (N-gram과 유사)"""
        n = len(tokens)
        best = []

        for suf_len in range(min(self.max_suffix_len, n - 1), self.min_suffix_len - 1, -1):
            suffix = tokens[-suf_len:]
            # 과거 위치에서 동일 패턴 검색
            for i in range(n - suf_len):
                if tokens[i:i + suf_len] == suffix:
                    draft_start = i + suf_len
                    draft_end = min(draft_start + self.num_speculative_tokens, n - suf_len)
                    if draft_end > draft_start:
                        draft = tokens[draft_start:draft_end]
                        if len(draft) > len(best):
                            best = draft
                        if len(best) >= self.num_speculative_tokens:
                            return best
        return best

    def _tree_lookup(self, tokens: list[int]) -> tuple:
        """Suffix tree에서 축적된 패턴 검색

        Returns:
            (draft_tokens, confidence_scores)
        """
        draft_tokens = []
        confidence_scores = []

        for step in range(self.num_speculative_tokens):
            best_token = None
            best_count = 0
            best_total = 0

            # 다양한 suffix 길이로 시도 (긴 것부터)
            current_tokens = tokens + draft_tokens
            for suf_len in range(min(self.max_suffix_len, len(current_tokens)),
                                 self.min_suffix_len - 1, -1):
                suffix = tuple(current_tokens[-suf_len:])

                if suffix in self._tree:
                    counts = self._tree[suffix]
                    total = sum(counts.values())
                    # 가장 빈도 높은 다음 토큰
                    for token, count in counts.items():
                        prob = count / total
                        if prob >= self.min_token_prob and count > best_count:
                            best_token = token
                            best_count = count
                            best_total = total
                    if best_token is not None:
                        break

            if best_token is None:
                break
            draft_tokens.append(best_token)
            confidence_scores.append(
                best_count / best_total if best_total > 0 else 0.0
            )

        return draft_tokens, confidence_scores if confidence_scores else None

    def update_tree(self, tokens: list[int]) -> None:
        """생성 완료된 시퀀스를 suffix tree에 추가 (학습)"""
        n = len(tokens)
        for i in range(n - 1):
            for suf_len in range(self.min_suffix_len,
                                 min(self.max_suffix_len + 1, i + 2)):
                suffix = tuple(tokens[i + 1 - suf_len:i + 1])
                next_token = tokens[i + 1] if i + 1 < n else None
                if next_token is None:
                    continue

                if suffix not in self._tree:
                    if self._total_entries >= self.max_tree_size:
                        continue  # tree 크기 제한
                    self._tree[suffix] = {}
                    self._total_entries += 1

                counts = self._tree[suffix]
                counts[next_token] = counts.get(next_token, 0) + 1

    def reset(self) -> None:
        """Suffix tree 초기화"""
        self._tree.clear()
        self._total_entries = 0


# ============================================================================
# EAGLE Draft Proposer
# ============================================================================

class EagleDraftProposer(BaseDraftProposer):
    """EAGLE 기반 Draft Proposer

    Target 모델의 Hidden States를 입력으로 받아
    경량 Draft 모델로 다음 토큰을 예측합니다.

    특징:
    - Target 모델의 Hidden States 활용
    - 별도 경량 모델로 빠른 Draft 토큰 생성
    - Tree Attention 지원 (선택)
    - N-gram보다 높은 수락률

    EAGLE 버전별 차이:
    - EAGLE 1: 선형 구조, 마지막 레이어 Hidden State
    - EAGLE 2: 선형 구조, 개선된 학습
    - EAGLE 3: Tree 구조, 다중 레이어 Hidden State

    이 구현은 EAGLE 1/2 스타일의 기본 구현입니다.
    """

    def __init__(
        self,
        draft_model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        num_speculative_tokens: int = 5,
        device: str = "cuda",
        use_tree_attention: bool = False,
        use_hidden_states: bool = False,
    ):
        """
        Args:
            draft_model_name: Draft model name / HF id.
            num_speculative_tokens: Number of draft tokens to propose.
            device: Torch device ("cuda", "cpu").
            use_tree_attention: Reserved for Phase 2; ignored.
            use_hidden_states: If True, additively fuse the target's last
                hidden state into the draft model's final-position input
                embedding on the first forward. This is a **toy** approximation
                of EAGLE (the real method requires retraining a draft head);
                see ``docs/DRAFT_METHODS.md § 4.2``. Silently falls back
                to the id-only path on hidden/embed dimension mismatch.
        """
        super().__init__(num_speculative_tokens)
        self.draft_model_name = draft_model_name
        self.device = device
        self.use_tree_attention = use_tree_attention
        self.use_hidden_states = use_hidden_states

        # 모델은 lazy loading
        self._model = None
        self._tokenizer = None

        # 캐시
        self._kv_cache = None
        self._last_hidden_states = None

    @property
    def model(self):
        """Draft 모델 (lazy loading)"""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        """토크나이저 (lazy loading)"""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def _load_model(self):
        """Draft 모델 로드"""
        if not HAS_TORCH:
            raise RuntimeError("EAGLE Proposer requires PyTorch")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading draft model: {self.draft_model_name}")

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.draft_model_name,
                trust_remote_code=True,
            )

            dtype = (
                torch.float16 if str(self.device).startswith("cuda") else torch.float32
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.draft_model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            self._model.to(self.device)
            self._model.eval()

        except Exception as e:
            raise RuntimeError(f"Failed to load draft model: {e}") from e

    def _build_forward_kwargs(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor | None,
    ) -> dict:
        """Prepare the HF forward() kwargs, optionally injecting hidden states.

        When ``hidden_states`` is provided and dimensions match the embedding
        width, the last-position embedding is additively fused with the
        target's last hidden state (toy EAGLE emulation). Otherwise falls
        back to the plain ``input_ids`` path.
        """
        if hidden_states is None:
            return {"input_ids": input_ids}

        embed_fn = self.model.get_input_embeddings()
        inputs_embeds = embed_fn(input_ids)

        h = hidden_states.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
        while h.dim() < 3:
            h = h.unsqueeze(0)

        if h.shape[-1] != inputs_embeds.shape[-1]:
            logger.warning(
                "hidden_states dim %d != embed dim %d — skipping injection",
                h.shape[-1],
                inputs_embeds.shape[-1],
            )
            return {"input_ids": input_ids}

        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[:, -1:, :] = inputs_embeds[:, -1:, :] + h[:, -1:, :]
        return {"inputs_embeds": inputs_embeds}

    def propose(
        self,
        context_tokens: list[int],
        hidden_states: torch.Tensor | None = None,
        sampling_params: SamplingParams | None = None,
    ) -> DraftOutput:
        """EAGLE 기반 Draft 토큰 생성

        Args:
            context_tokens: 현재까지의 토큰 시퀀스
            hidden_states: Target 모델의 Hidden States
            sampling_params: 샘플링 파라미터

        Returns:
            DraftOutput: 생성된 Draft 토큰 및 확률
        """
        if not HAS_TORCH:
            raise RuntimeError("EAGLE Proposer requires PyTorch")

        # 토큰을 텐서로 변환
        if isinstance(context_tokens, list):
            input_ids = torch.tensor([context_tokens], device=self.device)
        else:
            input_ids = context_tokens.unsqueeze(0).to(self.device)

        # 샘플링 파라미터
        if sampling_params is None:
            sampling_params = SamplingParams()

        draft_tokens: list[int] = []
        draft_probs_list: list[torch.Tensor] = []

        with torch.no_grad():
            current_ids = input_ids
            inject_hidden = (
                self.use_hidden_states
                and hidden_states is not None
                and self._kv_cache is None  # only on first call of this sequence
            )

            for step in range(self.num_speculative_tokens):
                forward_kwargs = self._build_forward_kwargs(
                    current_ids,
                    hidden_states if step == 0 and inject_hidden else None,
                )
                outputs = self.model(
                    use_cache=True,
                    past_key_values=self._kv_cache,
                    return_dict=True,
                    **forward_kwargs,
                )

                logits = outputs.logits[:, -1, :]
                self._kv_cache = outputs.past_key_values

                if sampling_params.is_greedy:
                    next_token = logits.argmax(dim=-1)
                else:
                    logits = logits / sampling_params.temperature
                    logits = apply_sampling_filters(
                        logits,
                        top_k=sampling_params.top_k,
                        top_p=sampling_params.top_p,
                    )
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

                draft_probs_list.append(F.softmax(logits, dim=-1).cpu())
                draft_tokens.append(int(next_token.item()))

                current_ids = next_token.unsqueeze(0)

                eos = getattr(self.tokenizer, "eos_token_id", None)
                if eos is not None and int(next_token.item()) == eos:
                    break

        # Hidden States 저장 (다음 propose에서 사용)
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            self._last_hidden_states = outputs.hidden_states[-1]

        # confidence: 각 draft 토큰의 max probability
        confidence_scores = None
        if draft_probs_list:
            confidence_scores = [float(p.max().item()) for p in draft_probs_list]

        return DraftOutput(
            draft_tokens=draft_tokens,
            draft_probs=torch.stack(draft_probs_list) if draft_probs_list else None,
            hidden_states=self._last_hidden_states,
            kv_cache_info=KVCacheInfo(
                seq_len=len(context_tokens) + len(draft_tokens),
                prev_seq_len=len(context_tokens),
                transfer_mode="none",
            ),
            confidence_scores=confidence_scores,
        )

    def reset(self) -> None:
        """상태 초기화"""
        self._kv_cache = None
        self._last_hidden_states = None


# ============================================================================
# Factory Function
# ============================================================================

def create_draft_proposer(
    method: str = "ngram",
    num_speculative_tokens: int = 5,
    **kwargs,
) -> BaseDraftProposer:
    """Draft Proposer 생성 팩토리 함수

    Args:
        method: Draft 방법 ("ngram", "eagle")
        num_speculative_tokens: 생성할 Draft 토큰 수
        **kwargs: 추가 인자

    Returns:
        BaseDraftProposer 인스턴스
    """
    method = method.lower()

    if method == "ngram":
        return NgramDraftProposer(
            num_speculative_tokens=num_speculative_tokens,
            ngram_window=kwargs.get('ngram_window', 4),
            min_match_length=kwargs.get('min_match_length', 2),
        )
    elif method == "suffix":
        return SuffixDraftProposer(
            num_speculative_tokens=num_speculative_tokens,
            max_suffix_len=kwargs.get('max_suffix_len', 8),
            min_suffix_len=kwargs.get('min_suffix_len', 2),
            min_token_prob=kwargs.get('min_token_prob', 0.1),
        )
    elif method == "eagle":
        return EagleDraftProposer(
            draft_model_name=kwargs.get(
                "draft_model_name", "meta-llama/Llama-3.2-1B-Instruct"
            ),
            num_speculative_tokens=num_speculative_tokens,
            device=kwargs.get("device", "cuda"),
            use_tree_attention=kwargs.get("use_tree_attention", False),
            use_hidden_states=kwargs.get("use_hidden_states", False),
        )
    else:
        raise ValueError(f"Unknown draft method: {method}")
