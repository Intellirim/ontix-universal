"""
ONTIX Universal - Core Package
코어 레이어: 엔진, 파이프라인, 라우팅, 컨텍스트 관리

Production Grade v2.0

Usage:
    from app.core import (
        UniversalEngine,
        Pipeline,
        QuestionRouter,
        QueryContext,
    )

    # 기본 사용
    engine = UniversalEngine.get_instance("brand_id")
    response = engine.ask("질문")
"""

# Settings
from app.core.config import settings

# Context Types
from app.core.context import (
    QueryContext,
    QuestionType,
    ProcessingStage,
    RetrievalResult,
    PerformanceMetrics,
    ConversationMessage,
)

# Routing Types
from app.core.routing import (
    QuestionRouter,
    IntentType,
    ClassificationResult,
    RouterConfig,
    PatternMatcher,
    EntityExtractor,
    IntentDetector,
)

# Pipeline Types
from app.core.pipeline import (
    Pipeline,
    PipelineStage,
    RetrievalMode,
    FallbackStrategy,
    PipelineConfig,
    StepMetrics,
    PipelineTrace,
    PromptManager,
    PipelineHook,
    ContentFilterHook,
    ContextTruncationHook,
)

# Engine Types
from app.core.engine import (
    UniversalEngine,
    EngineState,
    ErrorType,
    EngineConfig,
    RequestMetrics,
    EngineMetrics,
    RateLimiter,
    EngineError,
    Middleware,
    LoggingMiddleware,
    ValidationMiddleware,
)


__all__ = [
    # Settings
    "settings",

    # Context
    "QueryContext",
    "QuestionType",
    "ProcessingStage",
    "RetrievalResult",
    "PerformanceMetrics",
    "ConversationMessage",

    # Routing
    "QuestionRouter",
    "IntentType",
    "ClassificationResult",
    "RouterConfig",
    "PatternMatcher",
    "EntityExtractor",
    "IntentDetector",

    # Pipeline
    "Pipeline",
    "PipelineStage",
    "RetrievalMode",
    "FallbackStrategy",
    "PipelineConfig",
    "StepMetrics",
    "PipelineTrace",
    "PromptManager",
    "PipelineHook",
    "ContentFilterHook",
    "ContextTruncationHook",

    # Engine
    "UniversalEngine",
    "EngineState",
    "ErrorType",
    "EngineConfig",
    "RequestMetrics",
    "EngineMetrics",
    "RateLimiter",
    "EngineError",
    "Middleware",
    "LoggingMiddleware",
    "ValidationMiddleware",
]

__version__ = "2.0.0"
