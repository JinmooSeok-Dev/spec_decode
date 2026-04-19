# SPDX-License-Identifier: Apache-2.0
"""Confidence metrics and classifiers for verification-skip clients.

Provides the calculators and query classifiers that feed into the
BiLD / SVIP / RouteLLM-style clients in
:mod:`distspec.client.confidence_client`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

# ============================================================================
# Confidence Configuration
# ============================================================================

@dataclass
class ConfidenceConfig:
    """Confidence 기반 검증 설정

    Attributes:
        metric: confidence 측정 방식 ("entropy" | "max_prob" | "logit_margin")
        skip_threshold: 이 이상이면 verification skip (Token-level)
        query_routing_threshold: query 난이도 분류 경계 (Query-level)
        adaptive_entropy_threshold: 이 이상이면 drafting 중단 (Adaptive)
        warmup_steps: 초기 N step은 항상 verify
        fallback_confidence: confidence 계산 불가 시 기본값
    """
    metric: str = "entropy"
    skip_threshold: float = 0.8
    query_routing_threshold: float = 0.7
    adaptive_entropy_threshold: float = 1.0
    warmup_steps: int = 5
    fallback_confidence: float = 0.0

    def validate(self) -> None:
        if self.metric not in ("entropy", "max_prob", "logit_margin"):
            raise ValueError(f"Invalid metric: {self.metric}")
        if not (0.0 <= self.skip_threshold <= 1.0):
            raise ValueError("skip_threshold must be in [0, 1]")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")


# ============================================================================
# Token Confidence Result
# ============================================================================

@dataclass
class TokenConfidenceResult:
    """Per-token confidence 결과

    Attributes:
        scores: 각 토큰의 confidence 점수 (0~1, 높을수록 확신)
        metric: 사용된 metric 이름
        confident_prefix_len: threshold 이상인 연속 prefix 길이
    """
    scores: list[float] = field(default_factory=list)
    metric: str = "entropy"
    confident_prefix_len: int = 0

    @classmethod
    def from_scores(
        cls,
        scores: list[float],
        metric: str,
        threshold: float,
    ) -> TokenConfidenceResult:
        prefix_len = 0
        for s in scores:
            if s >= threshold:
                prefix_len += 1
            else:
                break
        return cls(scores=scores, metric=metric, confident_prefix_len=prefix_len)


# ============================================================================
# Confidence Calculator
# ============================================================================

class ConfidenceCalculator:
    """Confidence 점수 계산기

    다양한 metric으로 토큰 생성의 확신도를 측정한다.
    모든 메서드는 0~1 범위의 confidence를 반환한다 (높을수록 확신).
    """

    @staticmethod
    def entropy(probs: np.ndarray) -> float:
        """Entropy 기반 confidence: 1 - H(P)/log(V)

        낮은 entropy = 높은 confidence (분포가 집중됨)

        Args:
            probs: 확률 분포 [vocab_size] (합=1)

        Returns:
            confidence (0~1)
        """
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs[probs > 0]
        if len(probs) == 0:
            return 0.0

        h = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(probs))
        if max_entropy == 0:
            return 1.0

        return float(max(0.0, 1.0 - h / max_entropy))

    @staticmethod
    def max_prob(probs: np.ndarray) -> float:
        """최대 확률 기반 confidence: max(P)

        가장 직관적인 metric.

        Args:
            probs: 확률 분포 [vocab_size]

        Returns:
            confidence (0~1)
        """
        probs = np.asarray(probs, dtype=np.float64)
        if len(probs) == 0:
            return 0.0
        return float(np.max(probs))

    @staticmethod
    def logit_margin(logits: np.ndarray) -> float:
        """Logit margin 기반 confidence: sigmoid(logit_1st - logit_2nd)

        1위와 2위 logit 차이가 클수록 확신.
        sigmoid로 0~1 범위로 정규화한다.

        Args:
            logits: raw logits [vocab_size]

        Returns:
            confidence (0~1)
        """
        logits = np.asarray(logits, dtype=np.float64)
        if len(logits) < 2:
            return 1.0 if len(logits) == 1 else 0.0

        top2_idx = np.argpartition(logits, -2)[-2:]
        top2 = logits[top2_idx]
        margin = abs(float(top2[1] - top2[0]))

        return float(1.0 / (1.0 + math.exp(-margin)))

    @staticmethod
    def from_frequency(count: int, total: int) -> float:
        """빈도 기반 confidence (N-gram/Suffix용)

        Args:
            count: 해당 토큰이 등장한 횟수
            total: 전체 후보 토큰 수

        Returns:
            confidence (0~1)
        """
        if total <= 0:
            return 0.0
        return float(count / total)

    @staticmethod
    def from_match_length(match_len: int, max_window: int) -> float:
        """매칭 길이 기반 confidence (N-gram용)

        긴 N-gram 매칭 = 더 신뢰할 수 있는 draft

        Args:
            match_len: 실제 매칭된 N-gram 길이
            max_window: 최대 N-gram window 크기

        Returns:
            confidence (0~1)
        """
        if max_window <= 0:
            return 0.0
        return float(min(match_len / max_window, 1.0))

    @classmethod
    def compute_token_confidence(
        cls,
        metric: str,
        probs: np.ndarray | None = None,
        logits: np.ndarray | None = None,
        count: int = 0,
        total: int = 0,
        match_len: int = 0,
        max_window: int = 1,
    ) -> float:
        """metric 이름으로 confidence 계산 디스패치"""
        if metric == "entropy" and probs is not None:
            return cls.entropy(probs)
        elif metric == "max_prob" and probs is not None:
            return cls.max_prob(probs)
        elif metric == "logit_margin" and logits is not None:
            return cls.logit_margin(logits)
        elif metric == "frequency":
            return cls.from_frequency(count, total)
        elif metric == "match_length":
            return cls.from_match_length(match_len, max_window)
        else:
            return 0.0


# ============================================================================
# Query Classifier
# ============================================================================

class QueryClassifier:
    """Query 난이도 분류기 (RouteLLM-style)

    Prompt 특성을 기반으로 생성 난이도를 추정한다.
    "easy" 쿼리는 draft 모델만으로 충분, "hard"는 반드시 target 검증.

    분류 기준:
      - 반복 패턴 밀도: 높으면 easy
      - prompt 길이: context가 길수록 예측 쉬움
      - 특수 도메인 키워드: 코드/수학 → hard
    """

    # 난이도를 높이는 도메인 패턴 (코드, 수학, 논리)
    HARD_PATTERNS = [
        r'\bdef\s+\w+',        # Python function
        r'\bclass\s+\w+',      # class definition
        r'\bimport\s+\w+',     # import
        r'\b(if|for|while)\s*\(', # control flow (C-like)
        r'[∑∫∂√∞]',           # 수학 기호
        r'\b(prove|theorem|lemma)\b',
        r'\b(SELECT|FROM|WHERE)\b',  # SQL
    ]

    # 난이도를 낮추는 패턴 (일상 대화, 반복)
    EASY_PATTERNS = [
        r'^(hello|hi|hey|안녕)',
        r'\b(thanks|thank you|감사)\b',
        r'\b(yes|no|ok|okay)\b',
    ]

    def __init__(
        self,
        easy_threshold: float = 0.7,
        hard_threshold: float = 0.3,
    ):
        self.easy_threshold = easy_threshold
        self.hard_threshold = hard_threshold
        self._hard_re = [re.compile(p, re.IGNORECASE) for p in self.HARD_PATTERNS]
        self._easy_re = [re.compile(p, re.IGNORECASE) for p in self.EASY_PATTERNS]

    def classify(
        self,
        prompt_tokens: Sequence[int] | None = None,
        prompt_text: str | None = None,
    ) -> str:
        """쿼리 난이도 분류

        Returns:
            "easy" | "medium" | "hard"
        """
        score = 0.5  # 기본: medium

        # 텍스트 기반 분석
        if prompt_text is not None:
            score = self._score_from_text(prompt_text)

        # 토큰 기반 분석 (텍스트 없을 때 보조)
        if prompt_tokens is not None:
            token_score = self._score_from_tokens(prompt_tokens)
            if prompt_text is not None:
                score = 0.7 * score + 0.3 * token_score
            else:
                score = token_score

        if score >= self.easy_threshold:
            return "easy"
        elif score <= self.hard_threshold:
            return "hard"
        else:
            return "medium"

    def _score_from_text(self, text: str) -> float:
        """텍스트 기반 easiness 점수 (0=hard, 1=easy)"""
        score = 0.5

        # hard 패턴 카운트
        hard_count = sum(1 for r in self._hard_re if r.search(text))
        if hard_count > 0:
            score -= 0.15 * min(hard_count, 3)

        # easy 패턴 카운트
        easy_count = sum(1 for r in self._easy_re if r.search(text))
        if easy_count > 0:
            score += 0.15 * min(easy_count, 3)

        # 짧은 텍스트 = 보통 쉬움
        if len(text) < 50:
            score += 0.1
        elif len(text) > 500:
            score -= 0.1

        # 반복 패턴 밀도
        repetition = self._repetition_ratio(text)
        score += 0.2 * repetition

        return max(0.0, min(1.0, score))

    def _score_from_tokens(self, tokens: Sequence[int]) -> float:
        """토큰 기반 easiness 점수"""
        if len(tokens) == 0:
            return 0.5

        score = 0.5
        n = len(tokens)

        # 긴 context = 예측 쉬움
        if n > 100:
            score += 0.1
        elif n < 10:
            score += 0.05  # 아주 짧은 것도 쉬울 수 있음

        # 토큰 반복성 (unique ratio)
        unique = len(set(tokens))
        unique_ratio = unique / n
        if unique_ratio < 0.5:
            score += 0.15  # 반복 많음 → easy

        return max(0.0, min(1.0, score))

    @staticmethod
    def _repetition_ratio(text: str) -> float:
        """텍스트 내 반복 패턴 비율 (0~1)"""
        if len(text) < 10:
            return 0.0

        # 3-gram character 반복 측정
        ngrams = [text[i:i+3] for i in range(len(text) - 2)]
        if not ngrams:
            return 0.0

        unique = len(set(ngrams))
        return 1.0 - (unique / len(ngrams))
