"""
Unit Tests for Filters v2.0
QualityFilter, TrustFilter, RelevanceFilter, ValidationFilter 테스트

Run: pytest tests/unit/test_filters.py -v
"""

import pytest
from typing import Dict, Any


class TestQualityFilter:
    """QualityFilter 단위 테스트"""

    def test_quality_filter_initialization(self, quality_filter):
        """필터 초기화 테스트"""
        assert quality_filter is not None
        assert quality_filter.config is not None

    def test_quality_filter_valid_response(self, quality_filter, sample_response, sample_context):
        """정상 응답 품질 검증"""
        result = quality_filter.validate(sample_response, sample_context)

        assert result is not None
        assert hasattr(result, 'score')
        assert hasattr(result, 'valid')
        assert hasattr(result, 'level')
        assert 0.0 <= result.score <= 1.0

    def test_quality_filter_short_response(self, quality_filter, short_response, sample_context):
        """짧은 응답 품질 검증 - 낮은 점수 예상"""
        result = quality_filter.validate(short_response, sample_context)

        assert result is not None
        # 짧은 응답은 품질 점수가 낮아야 함
        assert result.score < 0.7

    def test_quality_filter_empty_response(self, quality_filter, sample_context):
        """빈 응답 처리"""
        result = quality_filter.validate("", sample_context)

        assert result is not None
        assert result.valid is False or result.score < 0.5

    def test_quality_filter_none_context(self, quality_filter, sample_response):
        """컨텍스트 없이 검증"""
        result = quality_filter.validate(sample_response, None)

        assert result is not None
        assert hasattr(result, 'score')

    def test_quality_filter_dimension_scores(self, quality_filter, sample_response, sample_context):
        """차원별 점수 확인"""
        result = quality_filter.validate(sample_response, sample_context)

        assert result is not None
        if hasattr(result, 'dimension_scores') and result.dimension_scores:
            for dim, score in result.dimension_scores.items():
                assert 0.0 <= score.score <= 1.0


class TestTrustFilter:
    """TrustFilter 단위 테스트"""

    def test_trust_filter_initialization(self, trust_filter):
        """필터 초기화 테스트"""
        assert trust_filter is not None
        assert trust_filter.config is not None

    def test_trust_filter_valid_response(self, trust_filter, sample_response, sample_context):
        """정상 응답 신뢰성 검증"""
        result = trust_filter.validate(sample_response, sample_context)

        assert result is not None
        assert hasattr(result, 'score')
        assert hasattr(result, 'valid')
        assert hasattr(result, 'hallucination_risk')
        assert 0.0 <= result.score <= 1.0

    def test_trust_filter_hallucination_detection(self, trust_filter, hallucination_response, sample_context):
        """환각 응답 감지"""
        result = trust_filter.validate(hallucination_response, sample_context)

        assert result is not None
        # 환각 응답은 환각 위험도가 높아야 함
        assert result.hallucination_risk > 0.0

    def test_trust_filter_with_question(self, trust_filter, sample_response):
        """질문 포함 컨텍스트로 검증"""
        context = {'question': '테스트 질문입니다'}
        result = trust_filter.validate(sample_response, context)

        assert result is not None
        assert hasattr(result, 'score')

    def test_trust_filter_verifications(self, trust_filter, sample_response, sample_context):
        """검증 결과 확인"""
        result = trust_filter.validate(sample_response, sample_context)

        assert result is not None
        if hasattr(result, 'verifications') and result.verifications:
            for vtype, vresult in result.verifications.items():
                assert hasattr(vresult, 'passed')
                assert hasattr(vresult, 'confidence')


class TestRelevanceFilter:
    """RelevanceFilter 단위 테스트"""

    def test_relevance_filter_initialization(self, relevance_filter):
        """필터 초기화 테스트"""
        assert relevance_filter is not None
        assert relevance_filter.config is not None

    def test_relevance_filter_valid_response(self, relevance_filter, sample_response, sample_context):
        """정상 응답 관련성 검증"""
        result = relevance_filter.validate(sample_response, sample_context)

        assert result is not None
        assert hasattr(result, 'score')
        assert hasattr(result, 'valid')
        assert hasattr(result, 'level')
        assert 0.0 <= result.score <= 1.0

    def test_relevance_filter_response_type(self, relevance_filter, sample_response, sample_context):
        """응답 유형 분석"""
        result = relevance_filter.validate(sample_response, sample_context)

        assert result is not None
        assert hasattr(result, 'response_type')

    def test_relevance_filter_irrelevant_response(self, relevance_filter, sample_context):
        """무관한 응답 검증"""
        irrelevant = "오늘 날씨가 좋습니다. 산책하기 좋은 날이네요."
        context = {**sample_context, 'question': '복싱 글러브 추천해주세요'}

        result = relevance_filter.validate(irrelevant, context)

        assert result is not None
        # 무관한 응답은 관련성 점수가 낮아야 함
        assert result.score < 0.7

    def test_relevance_filter_issues(self, relevance_filter, sample_response, sample_context):
        """이슈 목록 확인"""
        result = relevance_filter.validate(sample_response, sample_context)

        assert result is not None
        assert hasattr(result, 'issues')
        assert isinstance(result.issues, list)


class TestValidationFilter:
    """ValidationFilter 단위 테스트 (종합 검증)"""

    def test_validation_filter_initialization(self, validation_filter):
        """필터 초기화 테스트"""
        assert validation_filter is not None
        assert validation_filter.config is not None

    def test_validation_filter_valid_response(self, validation_filter, sample_response, sample_context):
        """정상 응답 종합 검증"""
        result = validation_filter.validate(sample_response, sample_context)

        assert result is not None
        assert hasattr(result, 'score')
        assert hasattr(result, 'valid')
        assert hasattr(result, 'grade')
        assert hasattr(result, 'status')
        assert 0.0 <= result.score <= 1.0

    def test_validation_filter_grade(self, validation_filter, sample_response, sample_context):
        """등급 판정 확인"""
        result = validation_filter.validate(sample_response, sample_context)

        assert result is not None
        # grade는 A, B, C, D, F 중 하나
        assert result.grade.value in ['A', 'B', 'C', 'D', 'F']

    def test_validation_filter_summary(self, validation_filter, sample_response, sample_context):
        """요약 정보 확인"""
        result = validation_filter.validate(sample_response, sample_context)

        assert result is not None
        assert hasattr(result, 'summary')
        if result.summary:
            assert hasattr(result.summary, 'total_issues')
            assert hasattr(result.summary, 'passed_filters')

    def test_validation_filter_suggestions(self, validation_filter, sample_response, sample_context):
        """개선 제안 확인"""
        result = validation_filter.validate(sample_response, sample_context)

        assert result is not None
        assert hasattr(result, 'suggestions')
        assert isinstance(result.suggestions, list)

    def test_validation_filter_filter_results(self, validation_filter, sample_response, sample_context):
        """개별 필터 결과 확인"""
        result = validation_filter.validate(sample_response, sample_context)

        assert result is not None
        assert hasattr(result, 'filter_results')
        if result.filter_results:
            # 최소 trust, quality 필터 결과가 있어야 함
            assert len(result.filter_results) >= 2


class TestFilterIntegration:
    """필터 통합 테스트"""

    def test_all_filters_on_same_response(
        self,
        quality_filter,
        trust_filter,
        relevance_filter,
        validation_filter,
        sample_response,
        sample_context
    ):
        """모든 필터를 동일 응답에 적용"""
        quality_result = quality_filter.validate(sample_response, sample_context)
        trust_result = trust_filter.validate(sample_response, sample_context)
        relevance_result = relevance_filter.validate(sample_response, sample_context)
        validation_result = validation_filter.validate(sample_response, sample_context)

        assert quality_result is not None
        assert trust_result is not None
        assert relevance_result is not None
        assert validation_result is not None

        # 모든 점수가 유효한 범위
        assert 0.0 <= quality_result.score <= 1.0
        assert 0.0 <= trust_result.score <= 1.0
        assert 0.0 <= relevance_result.score <= 1.0
        assert 0.0 <= validation_result.score <= 1.0

    def test_filters_consistency(
        self,
        quality_filter,
        trust_filter,
        sample_response,
        sample_context
    ):
        """동일 입력에 대한 일관성 테스트"""
        result1 = quality_filter.validate(sample_response, sample_context)
        result2 = quality_filter.validate(sample_response, sample_context)

        # 동일 입력에 동일 결과
        assert result1.score == result2.score
        assert result1.valid == result2.valid
