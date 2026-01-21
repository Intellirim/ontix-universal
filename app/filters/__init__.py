"""
Filters - Production Grade v2.0
응답 품질 검증 시스템

모듈:
- trust: 신뢰성 검증 (환각 감지, 출처 확인)
- quality: 품질 검증 (길이, 구조, 가독성)
- relevance: 관련성 검증 (키워드, 의미, 토픽)
- validation: 종합 검증 (오케스트레이션)
"""

# Trust Filter
from app.filters.trust import (
    TrustFilter,
    TrustResult,
    TrustConfig,
    TrustIssue,
    TrustLevel,
    TrustIssueSeverity,
    HallucinationType,
    VerificationType,
    VerificationResult,
)

# Quality Filter
from app.filters.quality import (
    QualityFilter,
    QualityResult,
    QualityConfig,
    QualityIssue,
    QualityLevel,
    QualityDimension,
    IssueSeverity,
    DimensionScore,
)

# Relevance Filter
from app.filters.relevance import (
    RelevanceFilter,
    RelevanceResult,
    RelevanceConfig,
    RelevanceIssue,
    RelevanceLevel,
    RelevanceType,
    RelevanceScore,
    ResponseType,
)

# Validation Filter (종합)
from app.filters.validation import (
    ValidationFilter,
    ValidationResult,
    ValidationConfig,
    ValidationStatus,
    ValidationSummary,
    FilterType,
    FilterResult,
    FilterWeight,
    OverallGrade,
    validate_response,
)


__version__ = "2.0.0"

__all__ = [
    # Trust
    'TrustFilter',
    'TrustResult',
    'TrustConfig',
    'TrustIssue',
    'TrustLevel',
    'TrustIssueSeverity',
    'HallucinationType',
    'VerificationType',
    'VerificationResult',

    # Quality
    'QualityFilter',
    'QualityResult',
    'QualityConfig',
    'QualityIssue',
    'QualityLevel',
    'QualityDimension',
    'IssueSeverity',
    'DimensionScore',

    # Relevance
    'RelevanceFilter',
    'RelevanceResult',
    'RelevanceConfig',
    'RelevanceIssue',
    'RelevanceLevel',
    'RelevanceType',
    'RelevanceScore',
    'ResponseType',

    # Validation
    'ValidationFilter',
    'ValidationResult',
    'ValidationConfig',
    'ValidationStatus',
    'ValidationSummary',
    'FilterType',
    'FilterResult',
    'FilterWeight',
    'OverallGrade',
    'validate_response',
]
