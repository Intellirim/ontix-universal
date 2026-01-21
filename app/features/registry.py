
"""
Feature Registry
모든 기능 핸들러 등록 및 관리
"""

from typing import Dict, Type, Optional, List
from app.models.feature import FeatureHandler
import logging

logger = logging.getLogger(__name__)


class FeatureRegistry:
    """
    기능 레지스트리
    
    모든 기능 핸들러를 등록하고 관리
    """
    
    _handlers: Dict[str, Type[FeatureHandler]] = {}
    
    @classmethod
    def register(cls, feature_name: str, handler_class: Type[FeatureHandler]):
        """
        핸들러 등록
        
        Args:
            feature_name: 기능 이름
            handler_class: 핸들러 클래스
        """
        cls._handlers[feature_name] = handler_class
        logger.info(f"Registered feature handler: {feature_name}")
    
    @classmethod
    def get_handler(cls, feature_name: str, brand_config: Dict) -> Optional[FeatureHandler]:
        """
        핸들러 인스턴스 반환
        
        Args:
            feature_name: 기능 이름
            brand_config: 브랜드 설정
        
        Returns:
            핸들러 인스턴스 or None
        """
        handler_class = cls._handlers.get(feature_name)
        
        if not handler_class:
            logger.warning(f"Unknown feature: {feature_name}")
            return None
        
        try:
            return handler_class(brand_config)
        except Exception as e:
            logger.error(f"Failed to instantiate handler {feature_name}: {e}")
            return None
    
    @classmethod
    def list_features(cls) -> List[str]:
        """등록된 기능 목록"""
        return list(cls._handlers.keys())
    
    @classmethod
    def has_feature(cls, feature_name: str) -> bool:
        """기능 존재 여부"""
        return feature_name in cls._handlers


# 핸들러 자동 등록
def _auto_register_handlers():
    """핸들러 자동 등록"""
    try:
        from app.features.product_recommendation.handler import ProductRecommendationHandler
        FeatureRegistry.register('product_recommendation', ProductRecommendationHandler)
    except ImportError:
        logger.warning("ProductRecommendationHandler not available")
    
    try:
        from app.features.analytics.handler import AnalyticsHandler
        FeatureRegistry.register('analytics', AnalyticsHandler)
    except ImportError:
        logger.warning("AnalyticsHandler not available")
    
    try:
        from app.features.advisor.handler import AdvisorHandler
        FeatureRegistry.register('advisor', AdvisorHandler)
    except ImportError:
        logger.warning("AdvisorHandler not available")
    
    try:
        from app.features.content_generation.handler import ContentGenerationHandler
        FeatureRegistry.register('content_generation', ContentGenerationHandler)
    except ImportError:
        logger.warning("ContentGenerationHandler not available")
    
    try:
        from app.features.social_monitoring.handler import SocialMonitoringHandler
        FeatureRegistry.register('social_monitoring', SocialMonitoringHandler)
    except ImportError:
        logger.warning("SocialMonitoringHandler not available")
    
    try:
        from app.features.onboarding.handler import OnboardingHandler
        FeatureRegistry.register('onboarding', OnboardingHandler)
    except ImportError:
        logger.warning("OnboardingHandler not available")


# 모듈 로드 시 자동 등록
_auto_register_handlers()
