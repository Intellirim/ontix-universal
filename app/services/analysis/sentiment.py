"""
Sentiment Analysis Service - Production Grade v1.0
감정 분석 서비스

Features:
- 한국어/영어 감정 분석
- 다중 레이블 지원 (positive, negative, neutral, mixed)
- 감정 강도 점수화
- 키워드 기반 빠른 분석 + LLM 기반 정밀 분석
- 배치 처리 지원

Author: ONTIX Universal Team
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
import re
import logging

logger = logging.getLogger(__name__)


# ============================================================
# Enums and Data Classes
# ============================================================

class SentimentLabel(str, Enum):
    """감정 레이블"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class AnalysisMode(str, Enum):
    """분석 모드"""
    FAST = "fast"          # 키워드 기반 빠른 분석
    ACCURATE = "accurate"  # LLM 기반 정밀 분석
    AUTO = "auto"          # 자동 선택


@dataclass
class SentimentScore:
    """개별 감정 점수"""
    positive: float = 0.0
    negative: float = 0.0
    neutral: float = 0.0

    def dominant(self) -> SentimentLabel:
        """지배적인 감정 반환"""
        scores = {
            SentimentLabel.POSITIVE: self.positive,
            SentimentLabel.NEGATIVE: self.negative,
            SentimentLabel.NEUTRAL: self.neutral,
        }

        # 혼합 감정 감지 (positive와 negative가 모두 높음)
        if self.positive > 0.3 and self.negative > 0.3:
            return SentimentLabel.MIXED

        return max(scores, key=scores.get)

    def confidence(self) -> float:
        """확신도 (가장 높은 점수)"""
        return max(self.positive, self.negative, self.neutral)


@dataclass
class SentimentResult:
    """감정 분석 결과"""
    label: SentimentLabel
    score: SentimentScore
    confidence: float
    keywords_found: List[str] = field(default_factory=list)
    analysis_mode: AnalysisMode = AnalysisMode.FAST

    # 상세 분석 결과 (LLM 분석 시)
    explanation: Optional[str] = None
    aspects: Dict[str, SentimentLabel] = field(default_factory=dict)  # 측면별 감정

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "label": self.label.value,
            "score": {
                "positive": round(self.score.positive, 3),
                "negative": round(self.score.negative, 3),
                "neutral": round(self.score.neutral, 3),
            },
            "confidence": round(self.confidence, 3),
            "keywords_found": self.keywords_found,
            "analysis_mode": self.analysis_mode.value,
            "explanation": self.explanation,
            "aspects": {k: v.value for k, v in self.aspects.items()} if self.aspects else {},
        }


@dataclass
class SentimentConfig:
    """감정 분석 설정"""
    # 분석 모드
    mode: AnalysisMode = AnalysisMode.AUTO

    # 언어 설정
    language: str = "ko"  # ko, en, auto

    # 임계값
    positive_threshold: float = 0.6
    negative_threshold: float = 0.6
    mixed_threshold: float = 0.3

    # LLM 설정 (accurate 모드)
    use_llm_for_ambiguous: bool = True
    llm_model: str = "gpt-5-mini"

    # 배치 설정
    batch_size: int = 10

    # 키워드 가중치
    keyword_weight: float = 0.4
    pattern_weight: float = 0.3
    length_weight: float = 0.1
    context_weight: float = 0.2


# ============================================================
# Sentiment Analyzer
# ============================================================

class SentimentAnalyzer:
    """
    감정 분석기

    한국어와 영어 텍스트의 감정을 분석합니다.
    빠른 키워드 기반 분석과 정밀한 LLM 기반 분석을 지원합니다.
    """

    # 한국어 감정 키워드
    KOREAN_POSITIVE_KEYWORDS = [
        # 기쁨/만족
        "좋아", "좋다", "좋네", "좋습니다", "좋음", "최고", "대박", "짱",
        "완벽", "훌륭", "멋지", "멋진", "멋있", "예쁘", "이쁘", "귀엽",
        "사랑", "감사", "고마", "축하", "행복", "기뻐", "즐거", "신나",
        "추천", "강추", "인정", "굿", "베스트", "만족", "감동", "힐링",
        # 긍정 표현
        "ㅎㅎ", "ㅋㅋ", "ㅠㅠ", "♥", "❤", "👍", "😊", "🥰", "😍",
    ]

    KOREAN_NEGATIVE_KEYWORDS = [
        # 불만/부정
        "싫어", "싫다", "싫음", "별로", "최악", "쓰레기", "망", "노답",
        "실망", "후회", "짜증", "화나", "열받", "빡치", "어이없", "황당",
        "불만", "불편", "아쉬", "안좋", "안 좋", "비추", "노노", "별점",
        "환불", "취소", "반품", "문제", "고장", "하자", "불량",
        # 부정 표현
        "ㅡㅡ", "ㅠㅠ", "😢", "😭", "😡", "😤", "👎",
    ]

    KOREAN_NEUTRAL_PATTERNS = [
        r'^(질문|문의|궁금|어떻게|뭐|언제|어디|왜)',
        r'(입니다|합니다|됩니다|있습니다)$',
        r'^(안녕|반갑)',
    ]

    # 영어 감정 키워드
    ENGLISH_POSITIVE_KEYWORDS = [
        "love", "great", "amazing", "awesome", "excellent", "perfect",
        "best", "good", "nice", "wonderful", "fantastic", "beautiful",
        "recommend", "happy", "enjoy", "thanks", "thank", "appreciate",
    ]

    ENGLISH_NEGATIVE_KEYWORDS = [
        "hate", "bad", "terrible", "awful", "horrible", "worst",
        "disappointed", "disappointed", "angry", "upset", "sad",
        "refund", "return", "cancel", "broken", "defect", "problem",
        "poor", "waste", "scam", "fraud", "fake",
    ]

    def __init__(self, config: Optional[SentimentConfig] = None):
        """
        감정 분석기 초기화

        Args:
            config: 분석 설정
        """
        self.config = config or SentimentConfig()
        self._llm_client = None

        logger.info(
            f"[SentimentAnalyzer] Initialized "
            f"(mode={self.config.mode.value}, language={self.config.language})"
        )

    @property
    def llm_client(self):
        """LLM 클라이언트 지연 로딩"""
        if self._llm_client is None:
            try:
                from app.services.shared.llm import get_llm_client
                self._llm_client = get_llm_client()
            except Exception as e:
                logger.warning(f"[SentimentAnalyzer] LLM client unavailable: {e}")
        return self._llm_client

    def analyze(self, text: str, context: Optional[Dict[str, Any]] = None) -> SentimentResult:
        """
        단일 텍스트 감정 분석

        Args:
            text: 분석할 텍스트
            context: 추가 컨텍스트 (optional)

        Returns:
            SentimentResult: 감정 분석 결과
        """
        if not text or not text.strip():
            return SentimentResult(
                label=SentimentLabel.NEUTRAL,
                score=SentimentScore(neutral=1.0),
                confidence=1.0,
                analysis_mode=AnalysisMode.FAST,
            )

        # 언어 감지
        language = self._detect_language(text) if self.config.language == "auto" else self.config.language

        # 분석 모드 결정
        mode = self.config.mode
        if mode == AnalysisMode.AUTO:
            mode = self._determine_analysis_mode(text)

        # 분석 실행
        if mode == AnalysisMode.FAST:
            return self._analyze_fast(text, language)
        else:
            return self._analyze_accurate(text, language, context)

    def analyze_batch(self, texts: List[str], context: Optional[Dict[str, Any]] = None) -> List[SentimentResult]:
        """
        배치 감정 분석

        Args:
            texts: 분석할 텍스트 리스트
            context: 추가 컨텍스트

        Returns:
            List[SentimentResult]: 감정 분석 결과 리스트
        """
        results = []

        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batch_results = [self.analyze(text, context) for text in batch]
            results.extend(batch_results)

        return results

    def analyze_with_aspects(
        self,
        text: str,
        aspects: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> SentimentResult:
        """
        측면별 감정 분석 (Aspect-based Sentiment Analysis)

        Args:
            text: 분석할 텍스트
            aspects: 분석할 측면들 (예: ['품질', '가격', '배송'])
            context: 추가 컨텍스트

        Returns:
            SentimentResult: 측면별 감정이 포함된 분석 결과
        """
        # 기본 분석
        base_result = self.analyze(text, context)

        # 측면별 분석
        aspect_sentiments = {}
        for aspect in aspects:
            aspect_score = self._analyze_aspect(text, aspect)
            aspect_sentiments[aspect] = aspect_score.dominant()

        base_result.aspects = aspect_sentiments
        return base_result

    def get_summary(self, results: List[SentimentResult]) -> Dict[str, Any]:
        """
        분석 결과 요약

        Args:
            results: 감정 분석 결과 리스트

        Returns:
            요약 통계
        """
        if not results:
            return {
                "total": 0,
                "distribution": {},
                "average_confidence": 0,
                "average_scores": {
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0,
                }
            }

        # 레이블별 분포
        distribution = {label: 0 for label in SentimentLabel}
        total_positive = 0
        total_negative = 0
        total_neutral = 0
        total_confidence = 0

        for result in results:
            distribution[result.label] += 1
            total_positive += result.score.positive
            total_negative += result.score.negative
            total_neutral += result.score.neutral
            total_confidence += result.confidence

        count = len(results)

        return {
            "total": count,
            "distribution": {label.value: count for label, count in distribution.items()},
            "distribution_percent": {
                label.value: round(count / len(results) * 100, 1)
                for label, count in distribution.items()
            },
            "average_confidence": round(total_confidence / count, 3),
            "average_scores": {
                "positive": round(total_positive / count, 3),
                "negative": round(total_negative / count, 3),
                "neutral": round(total_neutral / count, 3),
            },
            "dominant_sentiment": max(distribution, key=distribution.get).value,
        }

    # ============================================================
    # Private Methods
    # ============================================================

    def _detect_language(self, text: str) -> str:
        """언어 감지"""
        # 간단한 한국어/영어 감지
        korean_chars = len(re.findall(r'[가-힣]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))

        if korean_chars > english_chars:
            return "ko"
        elif english_chars > korean_chars:
            return "en"
        else:
            return self.config.language  # 기본값 사용

    def _determine_analysis_mode(self, text: str) -> AnalysisMode:
        """자동 분석 모드 결정"""
        # 짧은 텍스트는 빠른 분석
        if len(text) < 50:
            return AnalysisMode.FAST

        # 긴 텍스트나 복잡한 문장은 정밀 분석
        if len(text) > 200 or text.count('.') > 3:
            return AnalysisMode.ACCURATE

        return AnalysisMode.FAST

    def _analyze_fast(self, text: str, language: str) -> SentimentResult:
        """키워드 기반 빠른 분석"""
        text_lower = text.lower()

        # 언어별 키워드 선택
        if language == "ko":
            positive_keywords = self.KOREAN_POSITIVE_KEYWORDS
            negative_keywords = self.KOREAN_NEGATIVE_KEYWORDS
            neutral_patterns = self.KOREAN_NEUTRAL_PATTERNS
        else:
            positive_keywords = self.ENGLISH_POSITIVE_KEYWORDS
            negative_keywords = self.ENGLISH_NEGATIVE_KEYWORDS
            neutral_patterns = []

        # 키워드 매칭
        found_positive = [kw for kw in positive_keywords if kw in text_lower]
        found_negative = [kw for kw in negative_keywords if kw in text_lower]

        # 중립 패턴 확인
        is_neutral_pattern = any(
            re.search(pattern, text_lower) for pattern in neutral_patterns
        )

        # 점수 계산
        pos_count = len(found_positive)
        neg_count = len(found_negative)
        total = pos_count + neg_count + 1  # +1 for smoothing

        if is_neutral_pattern and pos_count == 0 and neg_count == 0:
            score = SentimentScore(neutral=1.0)
        else:
            positive_score = pos_count / total
            negative_score = neg_count / total
            neutral_score = 1.0 - positive_score - negative_score

            # 정규화
            total_score = positive_score + negative_score + neutral_score
            score = SentimentScore(
                positive=positive_score / total_score,
                negative=negative_score / total_score,
                neutral=max(0, neutral_score / total_score),
            )

        # 확신도 계산
        confidence = score.confidence()
        if pos_count == 0 and neg_count == 0:
            confidence = 0.5  # 키워드 없으면 낮은 확신도

        return SentimentResult(
            label=score.dominant(),
            score=score,
            confidence=confidence,
            keywords_found=found_positive + found_negative,
            analysis_mode=AnalysisMode.FAST,
        )

    def _analyze_accurate(
        self,
        text: str,
        language: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SentimentResult:
        """LLM 기반 정밀 분석"""
        # LLM이 없으면 빠른 분석으로 대체
        if not self.llm_client:
            logger.warning("[SentimentAnalyzer] LLM unavailable, falling back to fast mode")
            return self._analyze_fast(text, language)

        try:
            # 프롬프트 구성
            prompt = self._build_sentiment_prompt(text, language, context)

            # LLM 호출
            # GPT-5 모델은 temperature/top_p 파라미터를 지원하지 않음
            llm_params = {
                "model": self.config.llm_model,
                "messages": [
                    {"role": "system", "content": "You are a sentiment analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 200,
            }

            # GPT-5 계열이 아닌 경우에만 temperature 추가
            if not self.config.llm_model.startswith("gpt-5"):
                llm_params["temperature"] = 0.1

            response = self.llm_client.chat.completions.create(**llm_params)

            # 응답 파싱
            result_text = response.choices[0].message.content
            return self._parse_llm_response(result_text)

        except Exception as e:
            logger.error(f"[SentimentAnalyzer] LLM analysis failed: {e}")
            return self._analyze_fast(text, language)

    def _build_sentiment_prompt(
        self,
        text: str,
        language: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """LLM 분석용 프롬프트 생성"""
        lang_instruction = "한국어로" if language == "ko" else "in English"

        prompt = f"""Analyze the sentiment of the following text {lang_instruction}.

Text: "{text}"

Respond in JSON format:
{{
    "label": "positive" | "negative" | "neutral" | "mixed",
    "confidence": 0.0-1.0,
    "positive_score": 0.0-1.0,
    "negative_score": 0.0-1.0,
    "neutral_score": 0.0-1.0,
    "explanation": "brief explanation"
}}

Only output the JSON, nothing else."""

        return prompt

    def _parse_llm_response(self, response: str) -> SentimentResult:
        """LLM 응답 파싱"""
        import json

        try:
            # JSON 추출
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                label_map = {
                    "positive": SentimentLabel.POSITIVE,
                    "negative": SentimentLabel.NEGATIVE,
                    "neutral": SentimentLabel.NEUTRAL,
                    "mixed": SentimentLabel.MIXED,
                }

                return SentimentResult(
                    label=label_map.get(data.get("label", "neutral"), SentimentLabel.NEUTRAL),
                    score=SentimentScore(
                        positive=data.get("positive_score", 0),
                        negative=data.get("negative_score", 0),
                        neutral=data.get("neutral_score", 0),
                    ),
                    confidence=data.get("confidence", 0.5),
                    analysis_mode=AnalysisMode.ACCURATE,
                    explanation=data.get("explanation"),
                )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"[SentimentAnalyzer] Failed to parse LLM response: {e}")

        # 파싱 실패 시 기본값
        return SentimentResult(
            label=SentimentLabel.NEUTRAL,
            score=SentimentScore(neutral=1.0),
            confidence=0.3,
            analysis_mode=AnalysisMode.ACCURATE,
        )

    def _analyze_aspect(self, text: str, aspect: str) -> SentimentScore:
        """특정 측면에 대한 감정 분석"""
        # 측면 관련 문장 추출
        sentences = re.split(r'[.!?]', text)
        relevant_sentences = [s for s in sentences if aspect.lower() in s.lower()]

        if not relevant_sentences:
            return SentimentScore(neutral=1.0)

        # 관련 문장들의 감정 분석
        combined_text = ' '.join(relevant_sentences)
        result = self._analyze_fast(combined_text, self._detect_language(text))

        return result.score
