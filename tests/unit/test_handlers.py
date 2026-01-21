"""
Unit Tests for Feature Handlers
AdvisorHandler, AnalyticsHandler 등 핸들러 테스트

Run: pytest tests/unit/test_handlers.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any


class TestAdvisorHandler:
    """AdvisorHandler 단위 테스트"""

    @pytest.fixture
    def advisor_handler(self, sample_brand_config):
        """AdvisorHandler 인스턴스"""
        from app.features.advisor.handler import AdvisorHandler
        return AdvisorHandler('test_brand', sample_brand_config)

    def test_advisor_handler_initialization(self, advisor_handler):
        """초기화 테스트"""
        assert advisor_handler is not None
        assert advisor_handler.brand_id == 'test_brand'
        assert advisor_handler.advisor_config is not None

    def test_advisor_handler_has_all_filters(self, advisor_handler):
        """모든 필터 존재 확인"""
        assert hasattr(advisor_handler, 'quality_filter')
        assert hasattr(advisor_handler, 'trust_filter')
        assert hasattr(advisor_handler, 'relevance_filter')
        assert hasattr(advisor_handler, 'validation_filter')

    def test_advisor_handler_can_handle(self, advisor_handler):
        """can_handle 메서드 테스트"""
        # 조언 관련 질문
        assert advisor_handler.can_handle('어떻게 하면 좋을까요?', {})
        assert advisor_handler.can_handle('추천해주세요', {})
        assert advisor_handler.can_handle('조언 부탁드립니다', {})

        # 무관한 질문
        assert not advisor_handler.can_handle('안녕하세요', {})

    def test_advisor_handler_config_loading(self, sample_brand_config):
        """설정 로딩 테스트"""
        from app.features.advisor.handler import AdvisorHandler

        config = {
            **sample_brand_config,
            'advisor': {
                'quality_filter_enabled': False,
                'min_quality_score': 0.7,
            }
        }

        handler = AdvisorHandler('test_brand', config)

        assert handler.advisor_config.quality_filter_enabled is False
        assert handler.advisor_config.min_quality_score == 0.7

    def test_advisor_handler_filter_stats(self, advisor_handler):
        """get_filter_stats 메서드 테스트"""
        stats = advisor_handler.get_filter_stats()

        assert 'quality_filter_enabled' in stats
        assert 'trust_filter_enabled' in stats
        assert 'relevance_filter_enabled' in stats
        assert 'validation_filter_enabled' in stats
        assert 'min_quality_score' in stats
        assert 'min_trust_score' in stats


class TestProductRecommendationHandler:
    """ProductRecommendationHandler 단위 테스트"""

    @pytest.fixture
    def product_handler(self, sample_brand_config):
        """ProductRecommendationHandler 인스턴스"""
        from app.features.product_recommendation.handler import ProductRecommendationHandler
        return ProductRecommendationHandler('test_brand', sample_brand_config)

    def test_product_handler_initialization(self, product_handler):
        """초기화 테스트"""
        assert product_handler is not None
        assert product_handler.brand_id == 'test_brand'

    def test_product_handler_has_all_filters(self, product_handler):
        """모든 필터 존재 확인"""
        assert hasattr(product_handler, 'quality_filter')
        assert hasattr(product_handler, 'trust_filter')
        assert hasattr(product_handler, 'relevance_filter')
        assert hasattr(product_handler, 'validation_filter')

    def test_product_handler_can_handle(self, product_handler):
        """can_handle 메서드 테스트"""
        assert product_handler.can_handle('상품 추천해주세요', {})
        assert product_handler.can_handle('복싱 글러브 좋은 거 있어요?', {})


class TestAnalyticsHandler:
    """AnalyticsHandler 단위 테스트"""

    @pytest.fixture
    def analytics_handler(self, sample_brand_config):
        """AnalyticsHandler 인스턴스"""
        from app.features.analytics.handler import AnalyticsHandler
        return AnalyticsHandler('test_brand', sample_brand_config)

    def test_analytics_handler_initialization(self, analytics_handler):
        """초기화 테스트"""
        assert analytics_handler is not None
        assert analytics_handler.brand_id == 'test_brand'

    def test_analytics_handler_has_all_filters(self, analytics_handler):
        """모든 필터 존재 확인"""
        assert hasattr(analytics_handler, 'quality_filter')
        assert hasattr(analytics_handler, 'trust_filter')
        assert hasattr(analytics_handler, 'relevance_filter')
        assert hasattr(analytics_handler, 'validation_filter')

    def test_analytics_handler_can_handle(self, analytics_handler):
        """can_handle 메서드 테스트"""
        assert analytics_handler.can_handle('인기 콘텐츠 알려줘', {})
        assert analytics_handler.can_handle('통계 분석해줘', {})
        assert analytics_handler.can_handle('트렌드가 어때?', {})


class TestContentGenerationHandler:
    """ContentGenerationHandler 단위 테스트"""

    @pytest.fixture
    def content_handler(self, sample_brand_config):
        """ContentGenerationHandler 인스턴스"""
        from app.features.content_generation.handler import ContentGenerationHandler
        return ContentGenerationHandler('test_brand', sample_brand_config)

    def test_content_handler_initialization(self, content_handler):
        """초기화 테스트"""
        assert content_handler is not None
        assert content_handler.brand_id == 'test_brand'

    def test_content_handler_has_all_filters(self, content_handler):
        """모든 필터 존재 확인"""
        assert hasattr(content_handler, 'quality_filter')
        assert hasattr(content_handler, 'trust_filter')
        assert hasattr(content_handler, 'relevance_filter')
        assert hasattr(content_handler, 'validation_filter')

    def test_content_handler_can_handle(self, content_handler):
        """can_handle 메서드 테스트"""
        assert content_handler.can_handle('마케팅 콘텐츠 만들어줘', {})
        assert content_handler.can_handle('SNS 포스트 작성해줘', {})


class TestOnboardingHandler:
    """OnboardingHandler 단위 테스트"""

    @pytest.fixture
    def onboarding_handler(self, sample_brand_config):
        """OnboardingHandler 인스턴스"""
        from app.features.onboarding.handler import OnboardingHandler
        return OnboardingHandler('test_brand', sample_brand_config)

    def test_onboarding_handler_initialization(self, onboarding_handler):
        """초기화 테스트"""
        assert onboarding_handler is not None
        assert onboarding_handler.brand_id == 'test_brand'

    def test_onboarding_handler_has_all_filters(self, onboarding_handler):
        """모든 필터 존재 확인"""
        assert hasattr(onboarding_handler, 'quality_filter')
        assert hasattr(onboarding_handler, 'trust_filter')
        assert hasattr(onboarding_handler, 'relevance_filter')
        assert hasattr(onboarding_handler, 'validation_filter')


class TestSocialMonitoringHandler:
    """SocialMonitoringHandler 단위 테스트"""

    @pytest.fixture
    def social_handler(self, sample_brand_config):
        """SocialMonitoringHandler 인스턴스"""
        from app.features.social_monitoring.handler import SocialMonitoringHandler
        return SocialMonitoringHandler('test_brand', sample_brand_config)

    def test_social_handler_initialization(self, social_handler):
        """초기화 테스트"""
        assert social_handler is not None
        assert social_handler.brand_id == 'test_brand'

    def test_social_handler_has_all_filters(self, social_handler):
        """모든 필터 존재 확인"""
        assert hasattr(social_handler, 'quality_filter')
        assert hasattr(social_handler, 'trust_filter')
        assert hasattr(social_handler, 'relevance_filter')
        assert hasattr(social_handler, 'validation_filter')


class TestHandlerFilterIntegration:
    """Handler-Filter 통합 테스트"""

    def test_all_handlers_have_four_filters(self, sample_brand_config):
        """모든 Handler가 4개 필터를 가지는지 확인"""
        from app.features.advisor.handler import AdvisorHandler
        from app.features.analytics.handler import AnalyticsHandler
        from app.features.product_recommendation.handler import ProductRecommendationHandler
        from app.features.content_generation.handler import ContentGenerationHandler
        from app.features.onboarding.handler import OnboardingHandler
        from app.features.social_monitoring.handler import SocialMonitoringHandler

        handlers = [
            AdvisorHandler('test', sample_brand_config),
            AnalyticsHandler('test', sample_brand_config),
            ProductRecommendationHandler('test', sample_brand_config),
            ContentGenerationHandler('test', sample_brand_config),
            OnboardingHandler('test', sample_brand_config),
            SocialMonitoringHandler('test', sample_brand_config),
        ]

        for handler in handlers:
            assert hasattr(handler, 'quality_filter'), f"{handler.__class__.__name__} missing quality_filter"
            assert hasattr(handler, 'trust_filter'), f"{handler.__class__.__name__} missing trust_filter"
            assert hasattr(handler, 'relevance_filter'), f"{handler.__class__.__name__} missing relevance_filter"
            assert hasattr(handler, 'validation_filter'), f"{handler.__class__.__name__} missing validation_filter"

    def test_all_handlers_have_filter_stats(self, sample_brand_config):
        """모든 Handler가 get_filter_stats 메서드를 가지는지 확인"""
        from app.features.advisor.handler import AdvisorHandler
        from app.features.analytics.handler import AnalyticsHandler
        from app.features.product_recommendation.handler import ProductRecommendationHandler
        from app.features.content_generation.handler import ContentGenerationHandler
        from app.features.onboarding.handler import OnboardingHandler
        from app.features.social_monitoring.handler import SocialMonitoringHandler

        handlers = [
            AdvisorHandler('test', sample_brand_config),
            AnalyticsHandler('test', sample_brand_config),
            ProductRecommendationHandler('test', sample_brand_config),
            ContentGenerationHandler('test', sample_brand_config),
            OnboardingHandler('test', sample_brand_config),
            SocialMonitoringHandler('test', sample_brand_config),
        ]

        for handler in handlers:
            assert hasattr(handler, 'get_filter_stats'), f"{handler.__class__.__name__} missing get_filter_stats"
            stats = handler.get_filter_stats()
            assert 'relevance_filter_enabled' in stats, f"{handler.__class__.__name__} missing relevance in stats"
            assert 'validation_filter_enabled' in stats, f"{handler.__class__.__name__} missing validation in stats"
