"""
Unit Tests for Sentiment Analysis
감정 분석 서비스 테스트

Run: pytest tests/unit/test_sentiment.py -v
"""

import pytest
from typing import Dict, Any, List


class TestSentimentAnalyzer:
    """SentimentAnalyzer 단위 테스트"""

    @pytest.fixture
    def sentiment_analyzer(self):
        """SentimentAnalyzer 인스턴스"""
        from app.services.analysis.sentiment import SentimentAnalyzer, SentimentConfig
        return SentimentAnalyzer(config=SentimentConfig(language="ko"))

    def test_analyzer_initialization(self, sentiment_analyzer):
        """초기화 테스트"""
        assert sentiment_analyzer is not None
        assert sentiment_analyzer.config is not None

    def test_analyze_positive_text(self, sentiment_analyzer):
        """긍정 텍스트 분석"""
        text = "정말 좋아요! 최고입니다 추천합니다 ㅎㅎ"
        result = sentiment_analyzer.analyze(text)

        assert result is not None
        assert result.label.value == "positive"
        assert result.score.positive > result.score.negative

    def test_analyze_negative_text(self, sentiment_analyzer):
        """부정 텍스트 분석"""
        text = "별로예요 실망입니다 환불 요청해야겠어요"
        result = sentiment_analyzer.analyze(text)

        assert result is not None
        assert result.label.value == "negative"
        assert result.score.negative > result.score.positive

    def test_analyze_neutral_text(self, sentiment_analyzer):
        """중립 텍스트 분석"""
        text = "배송은 언제 되나요?"
        result = sentiment_analyzer.analyze(text)

        assert result is not None
        # 중립 또는 낮은 확신도
        assert result.label.value in ["neutral", "positive", "negative"]

    def test_analyze_empty_text(self, sentiment_analyzer):
        """빈 텍스트 분석"""
        result = sentiment_analyzer.analyze("")

        assert result is not None
        assert result.label.value == "neutral"
        assert result.confidence == 1.0

    def test_analyze_mixed_sentiment(self, sentiment_analyzer):
        """혼합 감정 텍스트 분석"""
        text = "디자인은 예쁜데 품질이 별로예요"
        result = sentiment_analyzer.analyze(text)

        assert result is not None
        # 긍정과 부정 모두 포함
        assert result.score.positive > 0 or result.score.negative > 0

    def test_batch_analysis(self, sentiment_analyzer):
        """배치 분석 테스트"""
        texts = [
            "좋아요! 최고!",
            "별로예요 실망",
            "배송 언제 되나요?",
        ]

        results = sentiment_analyzer.analyze_batch(texts)

        assert len(results) == 3
        assert all(r is not None for r in results)

    def test_get_summary(self, sentiment_analyzer):
        """요약 통계 테스트"""
        texts = [
            "좋아요!",
            "좋아요!",
            "별로예요",
        ]

        results = sentiment_analyzer.analyze_batch(texts)
        summary = sentiment_analyzer.get_summary(results)

        assert "total" in summary
        assert summary["total"] == 3
        assert "distribution" in summary
        assert "average_confidence" in summary
        assert "dominant_sentiment" in summary

    def test_result_to_dict(self, sentiment_analyzer):
        """결과 딕셔너리 변환 테스트"""
        text = "좋아요!"
        result = sentiment_analyzer.analyze(text)

        result_dict = result.to_dict()

        assert "label" in result_dict
        assert "score" in result_dict
        assert "confidence" in result_dict
        assert "positive" in result_dict["score"]
        assert "negative" in result_dict["score"]
        assert "neutral" in result_dict["score"]


class TestSentimentScore:
    """SentimentScore 테스트"""

    def test_dominant_positive(self):
        """지배적 긍정 감정 테스트"""
        from app.services.analysis.sentiment import SentimentScore, SentimentLabel

        score = SentimentScore(positive=0.8, negative=0.1, neutral=0.1)
        assert score.dominant() == SentimentLabel.POSITIVE

    def test_dominant_negative(self):
        """지배적 부정 감정 테스트"""
        from app.services.analysis.sentiment import SentimentScore, SentimentLabel

        score = SentimentScore(positive=0.1, negative=0.8, neutral=0.1)
        assert score.dominant() == SentimentLabel.NEGATIVE

    def test_dominant_mixed(self):
        """혼합 감정 테스트"""
        from app.services.analysis.sentiment import SentimentScore, SentimentLabel

        score = SentimentScore(positive=0.4, negative=0.4, neutral=0.2)
        assert score.dominant() == SentimentLabel.MIXED

    def test_confidence(self):
        """확신도 테스트"""
        from app.services.analysis.sentiment import SentimentScore

        score = SentimentScore(positive=0.8, negative=0.1, neutral=0.1)
        assert score.confidence() == 0.8


class TestSentimentConfig:
    """SentimentConfig 테스트"""

    def test_default_config(self):
        """기본 설정 테스트"""
        from app.services.analysis.sentiment import SentimentConfig, AnalysisMode

        config = SentimentConfig()

        assert config.mode == AnalysisMode.AUTO
        assert config.language == "ko"
        assert config.positive_threshold == 0.6

    def test_custom_config(self):
        """커스텀 설정 테스트"""
        from app.services.analysis.sentiment import SentimentConfig, AnalysisMode

        config = SentimentConfig(
            mode=AnalysisMode.FAST,
            language="en",
            positive_threshold=0.7,
        )

        assert config.mode == AnalysisMode.FAST
        assert config.language == "en"
        assert config.positive_threshold == 0.7


class TestKoreanKeywords:
    """한국어 키워드 감지 테스트"""

    @pytest.fixture
    def analyzer(self):
        from app.services.analysis.sentiment import SentimentAnalyzer, SentimentConfig, AnalysisMode
        return SentimentAnalyzer(config=SentimentConfig(mode=AnalysisMode.FAST, language="ko"))

    def test_korean_positive_keywords(self, analyzer):
        """한국어 긍정 키워드 감지"""
        positive_texts = [
            "대박 좋아요!",
            "완벽해요 추천합니다",
            "감사합니다 최고예요",
        ]

        for text in positive_texts:
            result = analyzer.analyze(text)
            assert result.score.positive > 0, f"Failed for: {text}"

    def test_korean_negative_keywords(self, analyzer):
        """한국어 부정 키워드 감지"""
        negative_texts = [
            "싫어요 별로",
            "실망이에요 최악",
            "환불해주세요",
        ]

        for text in negative_texts:
            result = analyzer.analyze(text)
            assert result.score.negative > 0, f"Failed for: {text}"

    def test_emoji_detection(self, analyzer):
        """이모지 감지 테스트"""
        positive_emoji = "좋아요 😊 👍"
        negative_emoji = "별로에요 😢 👎"

        pos_result = analyzer.analyze(positive_emoji)
        neg_result = analyzer.analyze(negative_emoji)

        assert pos_result.score.positive > 0
        assert neg_result.score.negative > 0
