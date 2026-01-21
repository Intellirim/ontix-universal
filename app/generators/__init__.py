"""
Generators Package - Production Grade v2.0
응답 생성기 모듈

Features:
    - 다양한 응답 유형 지원
    - 프로덕션 수준 메트릭 추적
    - 출력 검증 및 포맷팅
    - 확장 가능한 아키텍처
"""

# ============================================================
# Base Generator
# ============================================================
from app.generators.base import (
    # Enums
    GeneratorType,
    OutputFormat,
    ResponseTone,
    # Config & Metrics
    GeneratorConfig,
    GenerationMetrics,
    # Utility Classes
    TemplateManager,
    ResponseFormatter,
    OutputValidator,
    # Base Class
    BaseGenerator,
)

# ============================================================
# Factual Generator
# ============================================================
from app.generators.factual import (
    # Types
    SourceCitation,
    FactualResponse,
    # Generator
    FactualGenerator,
)

# ============================================================
# Insight Generator
# ============================================================
from app.generators.insight import (
    # Enums
    InsightType,
    TrendDirection,
    # Types
    InsightItem,
    InsightResponse,
    # Utilities
    StatisticsHelper,
    # Generator
    InsightGenerator,
)

# ============================================================
# Conversational Generator
# ============================================================
from app.generators.conversational import (
    # Enums
    EmotionType,
    ConversationIntent,
    # Types
    ConversationState,
    ConversationResponse,
    # Detectors
    EmotionDetector,
    IntentDetector,
    FollowUpGenerator,
    # Generator
    ConversationalGenerator,
)

# ============================================================
# Recommendation Generator
# ============================================================
from app.generators.recommendation import (
    # Enums
    RecommendationStrategy,
    RecommendationReason,
    SortOrder,
    # Types
    RecommendationItem,
    RecommendationResponse,
    # Utilities
    RecommendationScorer,
    DiversityOptimizer,
    # Generator
    RecommendationGenerator,
)


# ============================================================
# Factory Function
# ============================================================

def create_generator(
    generator_type: GeneratorType,
    brand_config: dict
) -> BaseGenerator:
    """
    Generator 팩토리 함수

    Args:
        generator_type: 생성기 유형
        brand_config: 브랜드 설정

    Returns:
        해당 타입의 Generator 인스턴스

    Raises:
        ValueError: 지원하지 않는 generator_type인 경우
    """
    generators = {
        GeneratorType.FACTUAL: FactualGenerator,
        GeneratorType.INSIGHT: InsightGenerator,
        GeneratorType.CONVERSATIONAL: ConversationalGenerator,
        GeneratorType.RECOMMENDATION: RecommendationGenerator,
    }

    generator_class = generators.get(generator_type)
    if not generator_class:
        raise ValueError(f"Unsupported generator type: {generator_type}")

    return generator_class(brand_config)


# ============================================================
# Exports
# ============================================================

__all__ = [
    # Base
    'GeneratorType',
    'OutputFormat',
    'ResponseTone',
    'GeneratorConfig',
    'GenerationMetrics',
    'TemplateManager',
    'ResponseFormatter',
    'OutputValidator',
    'BaseGenerator',

    # Factual
    'SourceCitation',
    'FactualResponse',
    'FactualGenerator',

    # Insight
    'InsightType',
    'TrendDirection',
    'InsightItem',
    'InsightResponse',
    'StatisticsHelper',
    'InsightGenerator',

    # Conversational
    'EmotionType',
    'ConversationIntent',
    'ConversationState',
    'ConversationResponse',
    'EmotionDetector',
    'IntentDetector',
    'FollowUpGenerator',
    'ConversationalGenerator',

    # Recommendation
    'RecommendationStrategy',
    'RecommendationReason',
    'SortOrder',
    'RecommendationItem',
    'RecommendationResponse',
    'RecommendationScorer',
    'DiversityOptimizer',
    'RecommendationGenerator',

    # Factory
    'create_generator',
]
