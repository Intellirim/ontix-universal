"""
Interfaces Module - Production Grade v2.0
모든 컴포넌트의 인터페이스 정의

이 모듈은 시스템의 핵심 인터페이스를 제공합니다:
- BaseInterface: 모든 컴포넌트의 기본 인터페이스
- RetrieverInterface: 검색기 인터페이스
- GeneratorInterface: 응답 생성기 인터페이스
- ValidatorInterface: 응답 검증기 인터페이스

Features:
    - 라이프사이클 관리
    - 헬스체크 지원
    - 메트릭 추적
    - 설정 관리
    - 타입 안전성

Usage:
    from app.interfaces import (
        RetrieverInterface,
        GeneratorInterface,
        ValidatorInterface,
    )

    class MyRetriever(RetrieverInterface):
        def _do_retrieve(self, context):
            # 구현
            pass
"""

# Base Interface
from .base import (
    BaseInterface,
    ManagedComponent,
    ComponentStatus,
    ComponentType,
    ComponentMetrics,
    HealthCheckResult,
    BaseConfig,
)

# Retriever Interface
from .retriever import (
    RetrieverInterface,
    RetrievalStatus,
    RetrievalSource,
    RetrievalConfig,
    RetrievalItem,
    RetrievalResult,
    RetrievalMetrics,
    RetrievalCache,
)

# Generator Interface
from .generator import (
    GeneratorInterface,
    GenerationStatus,
    GeneratorType,
    OutputFormat,
    ResponseTone,
    GenerationConfig,
    GenerationResult,
    GenerationMetrics,
    PromptTemplate,
)

# Validator Interface
from .validator import (
    ValidatorInterface,
    ValidationStatus,
    ValidationSeverity,
    ValidationType,
    ValidationIssue,
    ValidationResult,
    ValidationConfig,
    ValidationMetrics,
    ValidationRule,
    BuiltInRules,
)

__all__ = [
    # Base
    'BaseInterface',
    'ManagedComponent',
    'ComponentStatus',
    'ComponentType',
    'ComponentMetrics',
    'HealthCheckResult',
    'BaseConfig',

    # Retriever
    'RetrieverInterface',
    'RetrievalStatus',
    'RetrievalSource',
    'RetrievalConfig',
    'RetrievalItem',
    'RetrievalResult',
    'RetrievalMetrics',
    'RetrievalCache',

    # Generator
    'GeneratorInterface',
    'GenerationStatus',
    'GeneratorType',
    'OutputFormat',
    'ResponseTone',
    'GenerationConfig',
    'GenerationResult',
    'GenerationMetrics',
    'PromptTemplate',

    # Validator
    'ValidatorInterface',
    'ValidationStatus',
    'ValidationSeverity',
    'ValidationType',
    'ValidationIssue',
    'ValidationResult',
    'ValidationConfig',
    'ValidationMetrics',
    'ValidationRule',
    'BuiltInRules',
]

__version__ = '2.0.0'
