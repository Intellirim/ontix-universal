"""
Validator Interface - Production Grade v2.0
응답 검증기 인터페이스 정의

Features:
    - 다양한 검증 규칙
    - 품질 점수화
    - 자동 수정 제안
    - 검증 메트릭
    - 규칙 체이닝
    - 커스텀 검증기 지원
"""

from abc import abstractmethod
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
import logging

from app.interfaces.base import (
    BaseInterface,
    ComponentType,
    ComponentStatus,
    HealthCheckResult,
)

logger = logging.getLogger(__name__)


# ============================================================
# Enums
# ============================================================

class ValidationStatus(str, Enum):
    """검증 상태"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class ValidationSeverity(str, Enum):
    """검증 심각도"""
    ERROR = "error"  # 반드시 수정 필요
    WARNING = "warning"  # 권장 수정
    INFO = "info"  # 참고 사항


class ValidationType(str, Enum):
    """검증 유형"""
    LENGTH = "length"
    FORMAT = "format"
    CONTENT = "content"
    QUALITY = "quality"
    SAFETY = "safety"
    RELEVANCE = "relevance"
    CUSTOM = "custom"


# ============================================================
# Data Classes
# ============================================================

@dataclass
class ValidationIssue:
    """검증 이슈"""
    type: ValidationType
    severity: ValidationSeverity
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'type': self.type.value,
            'severity': self.severity.value,
            'message': self.message,
            'location': self.location,
            'suggestion': self.suggestion,
            'metadata': self.metadata,
        }


@dataclass
class ValidationResult:
    """검증 결과"""
    valid: bool
    status: ValidationStatus
    score: float = 1.0  # 0.0 ~ 1.0
    issues: List[ValidationIssue] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    corrected_content: Optional[str] = None
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        """에러 개수"""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        """경고 개수"""
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)

    @property
    def has_errors(self) -> bool:
        """에러 존재 여부"""
        return self.error_count > 0

    @property
    def has_warnings(self) -> bool:
        """경고 존재 여부"""
        return self.warning_count > 0

    def add_issue(
        self,
        type: ValidationType,
        severity: ValidationSeverity,
        message: str,
        **kwargs
    ):
        """이슈 추가"""
        self.issues.append(ValidationIssue(
            type=type,
            severity=severity,
            message=message,
            **kwargs
        ))

        # 에러가 있으면 invalid
        if severity == ValidationSeverity.ERROR:
            self.valid = False
            self.status = ValidationStatus.FAILED

    def add_suggestion(self, suggestion: str):
        """제안 추가"""
        self.suggestions.append(suggestion)

    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """다른 결과와 병합"""
        merged = ValidationResult(
            valid=self.valid and other.valid,
            status=ValidationStatus.PASSED if (self.valid and other.valid) else ValidationStatus.FAILED,
            score=min(self.score, other.score),
            issues=self.issues + other.issues,
            suggestions=self.suggestions + other.suggestions,
            latency_ms=self.latency_ms + other.latency_ms,
        )
        return merged

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'valid': self.valid,
            'status': self.status.value,
            'score': round(self.score, 4),
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'issues': [i.to_dict() for i in self.issues],
            'suggestions': self.suggestions,
            'has_correction': self.corrected_content is not None,
            'latency_ms': round(self.latency_ms, 2),
        }


@dataclass
class ValidationConfig:
    """검증 설정"""
    # 길이 검증
    min_length: int = 10
    max_length: int = 5000

    # 품질 검증
    min_quality_score: float = 0.5
    enable_quality_check: bool = True

    # 안전 검증
    enable_safety_check: bool = True
    forbidden_patterns: List[str] = field(default_factory=list)

    # 관련성 검증
    enable_relevance_check: bool = True
    min_relevance_score: float = 0.3

    # 자동 수정
    enable_auto_correction: bool = False

    # 검증 레벨
    strict_mode: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationConfig':
        """딕셔너리에서 생성"""
        return cls(
            min_length=data.get('min_length', 10),
            max_length=data.get('max_length', 5000),
            min_quality_score=data.get('min_quality_score', 0.5),
            enable_quality_check=data.get('enable_quality_check', True),
            enable_safety_check=data.get('enable_safety_check', True),
            forbidden_patterns=data.get('forbidden_patterns', []),
            enable_relevance_check=data.get('enable_relevance_check', True),
            min_relevance_score=data.get('min_relevance_score', 0.3),
            enable_auto_correction=data.get('enable_auto_correction', False),
            strict_mode=data.get('strict_mode', False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'min_length': self.min_length,
            'max_length': self.max_length,
            'min_quality_score': self.min_quality_score,
            'enable_quality_check': self.enable_quality_check,
            'enable_safety_check': self.enable_safety_check,
            'enable_relevance_check': self.enable_relevance_check,
            'min_relevance_score': self.min_relevance_score,
            'enable_auto_correction': self.enable_auto_correction,
            'strict_mode': self.strict_mode,
        }


@dataclass
class ValidationMetrics:
    """검증 메트릭"""
    total_validations: int = 0
    passed_count: int = 0
    failed_count: int = 0
    warning_count: int = 0
    total_issues: int = 0
    average_score: float = 0.0
    average_latency_ms: float = 0.0
    auto_corrections: int = 0

    @property
    def pass_rate(self) -> float:
        """통과율"""
        if self.total_validations == 0:
            return 0.0
        return self.passed_count / self.total_validations

    def record_validation(self, result: ValidationResult, auto_corrected: bool = False):
        """검증 기록"""
        self.total_validations += 1

        if result.valid:
            self.passed_count += 1
        else:
            self.failed_count += 1

        if result.has_warnings:
            self.warning_count += 1

        self.total_issues += len(result.issues)

        if auto_corrected:
            self.auto_corrections += 1

        # 이동 평균
        n = self.total_validations
        self.average_score = (
            (self.average_score * (n - 1) + result.score) / n
        )
        self.average_latency_ms = (
            (self.average_latency_ms * (n - 1) + result.latency_ms) / n
        )

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'total_validations': self.total_validations,
            'pass_rate': round(self.pass_rate, 4),
            'failed_count': self.failed_count,
            'warning_count': self.warning_count,
            'total_issues': self.total_issues,
            'average_score': round(self.average_score, 4),
            'average_latency_ms': round(self.average_latency_ms, 2),
            'auto_corrections': self.auto_corrections,
        }


# ============================================================
# Validation Rules
# ============================================================

@dataclass
class ValidationRule:
    """검증 규칙"""
    name: str
    type: ValidationType
    check_fn: Callable[[str, Dict[str, Any]], Tuple[bool, Optional[str]]]
    severity: ValidationSeverity = ValidationSeverity.ERROR
    enabled: bool = True
    description: str = ""

    def check(self, content: str, context: Dict[str, Any] = None) -> Tuple[bool, Optional[str]]:
        """규칙 검사"""
        if not self.enabled:
            return True, None
        return self.check_fn(content, context or {})


class BuiltInRules:
    """내장 검증 규칙"""

    @staticmethod
    def length_rule(min_len: int, max_len: int) -> ValidationRule:
        """길이 검증 규칙"""
        def check(content: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
            length = len(content)
            if length < min_len:
                return False, f"Content too short: {length} < {min_len}"
            if length > max_len:
                return False, f"Content too long: {length} > {max_len}"
            return True, None

        return ValidationRule(
            name="length_check",
            type=ValidationType.LENGTH,
            check_fn=check,
            description=f"Length must be between {min_len} and {max_len}",
        )

    @staticmethod
    def empty_rule() -> ValidationRule:
        """빈 콘텐츠 검증 규칙"""
        def check(content: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
            if not content or not content.strip():
                return False, "Content is empty"
            return True, None

        return ValidationRule(
            name="empty_check",
            type=ValidationType.CONTENT,
            check_fn=check,
            description="Content must not be empty",
        )

    @staticmethod
    def forbidden_pattern_rule(patterns: List[str]) -> ValidationRule:
        """금지 패턴 검증 규칙"""
        compiled_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

        def check(content: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
            for pattern in compiled_patterns:
                if pattern.search(content):
                    return False, f"Forbidden pattern detected: {pattern.pattern}"
            return True, None

        return ValidationRule(
            name="forbidden_pattern_check",
            type=ValidationType.SAFETY,
            check_fn=check,
            description="Content must not contain forbidden patterns",
        )

    @staticmethod
    def sensitive_info_rule() -> ValidationRule:
        """민감 정보 검증 규칙"""
        patterns = [
            r'(?i)(api.?key|password|secret|token)\s*[:=]\s*\S+',
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN 패턴
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # 이메일
        ]
        compiled = [re.compile(p) for p in patterns]

        def check(content: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
            for pattern in compiled:
                if pattern.search(content):
                    return False, "Sensitive information detected"
            return True, None

        return ValidationRule(
            name="sensitive_info_check",
            type=ValidationType.SAFETY,
            check_fn=check,
            severity=ValidationSeverity.WARNING,
            description="Check for sensitive information",
        )

    @staticmethod
    def format_rule(required_format: str) -> ValidationRule:
        """포맷 검증 규칙"""
        def check(content: str, context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
            if required_format == "markdown":
                # 기본 마크다운 체크 (헤딩, 리스트 등)
                has_formatting = bool(re.search(r'[#*\-\[\]]', content))
                if not has_formatting:
                    return False, "Content should use markdown formatting"
            elif required_format == "json":
                import json
                try:
                    json.loads(content)
                except json.JSONDecodeError:
                    return False, "Content is not valid JSON"
            return True, None

        return ValidationRule(
            name=f"format_check_{required_format}",
            type=ValidationType.FORMAT,
            check_fn=check,
            severity=ValidationSeverity.WARNING,
            description=f"Content should be in {required_format} format",
        )


# ============================================================
# Validator Interface
# ============================================================

class ValidatorInterface(BaseInterface):
    """
    프로덕션급 응답 검증기 인터페이스

    응답의 품질, 안전성, 관련성 등을 검증합니다.

    Features:
        - 다양한 검증 규칙
        - 품질 점수화
        - 자동 수정 제안
        - 검증 메트릭
        - 규칙 체이닝

    Usage:
        class MyValidator(ValidatorInterface):
            def _do_validate(self, content, context) -> ValidationResult:
                result = ValidationResult(valid=True, status=ValidationStatus.PASSED)
                # 커스텀 검증 로직
                return result

        validator = MyValidator(brand_config)
        result = validator.validate(response, context)

        if not result.valid:
            print(f"Validation failed: {result.issues}")
    """

    # 기본 금지 패턴
    DEFAULT_FORBIDDEN_PATTERNS = [
        r'(?i)(api.?key|password|secret|token)',
        r'(?i)(내부|confidential|private)',
    ]

    def __init__(
        self,
        brand_config: Dict[str, Any],
        validator_type: str = "default"
    ):
        """
        Args:
            brand_config: 브랜드 설정
            validator_type: 검증기 유형
        """
        super().__init__(brand_config, ComponentType.VALIDATOR)

        self.validator_type = validator_type

        # 설정 로드
        val_config = brand_config.get('validation', {})
        self._config = ValidationConfig.from_dict(val_config)

        # 금지 패턴 설정
        if not self._config.forbidden_patterns:
            self._config.forbidden_patterns = self.DEFAULT_FORBIDDEN_PATTERNS

        # 검증 메트릭
        self._validation_metrics = ValidationMetrics()

        # 검증 규칙
        self._rules: List[ValidationRule] = []
        self._init_default_rules()

        # 커스텀 검증기
        self._custom_validators: List[Callable[[str, Dict], ValidationResult]] = []

    def _init_default_rules(self):
        """기본 규칙 초기화"""
        # 빈 콘텐츠 체크
        self._rules.append(BuiltInRules.empty_rule())

        # 길이 체크
        self._rules.append(BuiltInRules.length_rule(
            self._config.min_length,
            self._config.max_length
        ))

        # 안전성 체크
        if self._config.enable_safety_check:
            self._rules.append(BuiltInRules.forbidden_pattern_rule(
                self._config.forbidden_patterns
            ))
            self._rules.append(BuiltInRules.sensitive_info_rule())

    # === Main Methods ===

    def validate(
        self,
        response: str,
        context: Dict[str, Any] = None
    ) -> ValidationResult:
        """
        응답 검증

        Args:
            response: 검증할 응답
            context: 컨텍스트 정보

        Returns:
            검증 결과
        """
        import time
        start_time = time.time()

        context = context or {}

        try:
            # 기본 검증 결과 초기화
            result = ValidationResult(
                valid=True,
                status=ValidationStatus.PASSED,
                score=1.0
            )

            # 규칙 기반 검증
            for rule in self._rules:
                passed, message = rule.check(response, context)
                if not passed:
                    result.add_issue(
                        type=rule.type,
                        severity=rule.severity,
                        message=message,
                        metadata={'rule': rule.name}
                    )

            # 커스텀 검증
            custom_result = self._do_validate(response, context)
            result = result.merge(custom_result)

            # 커스텀 검증기 실행
            for validator in self._custom_validators:
                try:
                    custom = validator(response, context)
                    result = result.merge(custom)
                except Exception as e:
                    logger.error(f"Custom validator error: {e}")

            # 점수 계산
            result.score = self._calculate_score(result)

            # 최종 상태 결정
            if result.has_errors:
                result.status = ValidationStatus.FAILED
                result.valid = False
            elif result.has_warnings:
                result.status = ValidationStatus.WARNING
            else:
                result.status = ValidationStatus.PASSED

            # 자동 수정
            auto_corrected = False
            if not result.valid and self._config.enable_auto_correction:
                corrected = self._auto_correct(response, result)
                if corrected:
                    result.corrected_content = corrected
                    auto_corrected = True

            result.latency_ms = (time.time() - start_time) * 1000

            # 메트릭 기록
            self._record_validation(result, auto_corrected)

            logger.debug(
                f"Validation: {result.status.value}, "
                f"score={result.score:.2f}, "
                f"issues={len(result.issues)}"
            )

            return result

        except Exception as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            return ValidationResult(
                valid=False,
                status=ValidationStatus.FAILED,
                score=0.0,
                issues=[ValidationIssue(
                    type=ValidationType.CUSTOM,
                    severity=ValidationSeverity.ERROR,
                    message=f"Validation error: {str(e)}"
                )],
                latency_ms=(time.time() - start_time) * 1000
            )

    def validate_with_correction(
        self,
        response: str,
        context: Dict[str, Any] = None
    ) -> Tuple[ValidationResult, Optional[str]]:
        """
        검증 및 수정된 콘텐츠 반환

        Args:
            response: 검증할 응답
            context: 컨텍스트 정보

        Returns:
            (검증 결과, 수정된 콘텐츠 또는 None)
        """
        result = self.validate(response, context)
        return result, result.corrected_content

    # === Abstract Method ===

    def _do_validate(
        self,
        response: str,
        context: Dict[str, Any]
    ) -> ValidationResult:
        """
        실제 검증 로직 (오버라이드 가능)

        Args:
            response: 검증할 응답
            context: 컨텍스트 정보

        Returns:
            검증 결과
        """
        return ValidationResult(valid=True, status=ValidationStatus.PASSED)

    # === Rule Management ===

    def add_rule(self, rule: ValidationRule) -> 'ValidatorInterface':
        """
        규칙 추가

        Args:
            rule: 검증 규칙

        Returns:
            self (체이닝용)
        """
        self._rules.append(rule)
        logger.debug(f"Rule added: {rule.name}")
        return self

    def remove_rule(self, name: str) -> bool:
        """
        규칙 제거

        Args:
            name: 규칙 이름

        Returns:
            제거 성공 여부
        """
        for i, rule in enumerate(self._rules):
            if rule.name == name:
                self._rules.pop(i)
                return True
        return False

    def clear_rules(self):
        """모든 규칙 제거"""
        self._rules.clear()

    def get_rules(self) -> List[str]:
        """규칙 목록"""
        return [rule.name for rule in self._rules]

    def add_custom_validator(
        self,
        validator: Callable[[str, Dict], ValidationResult]
    ) -> 'ValidatorInterface':
        """
        커스텀 검증기 추가

        Args:
            validator: 검증 함수

        Returns:
            self (체이닝용)
        """
        self._custom_validators.append(validator)
        return self

    # === Configuration ===

    def configure(self, **kwargs) -> 'ValidatorInterface':
        """
        검증 설정 변경 (체이닝 지원)

        Usage:
            validator.configure(strict_mode=True).validate(response)

        Args:
            **kwargs: 설정 키-값

        Returns:
            self (체이닝용)
        """
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
                logger.debug(f"Validator config: {key}={value}")
        return self

    def get_validation_config(self) -> ValidationConfig:
        """검증 설정 조회"""
        return self._config

    def _get_component_config(self) -> Dict[str, Any]:
        """컴포넌트 설정 (오버라이드)"""
        return self._config.to_dict()

    # === Metrics ===

    def get_validation_metrics(self) -> Dict[str, Any]:
        """검증 메트릭 조회"""
        return self._validation_metrics.to_dict()

    def reset_validation_metrics(self):
        """검증 메트릭 초기화"""
        self._validation_metrics = ValidationMetrics()

    # === Health Check ===

    def _do_health_check(self) -> HealthCheckResult:
        """헬스체크 (오버라이드)"""
        try:
            metrics = self._validation_metrics

            # 통과율이 너무 낮으면 경고
            if metrics.total_validations > 10 and metrics.pass_rate < 0.3:
                return HealthCheckResult(
                    healthy=True,
                    status=ComponentStatus.DEGRADED,
                    message=f"Low pass rate: {metrics.pass_rate:.1%}",
                    details={'pass_rate': metrics.pass_rate}
                )

            return HealthCheckResult(
                healthy=True,
                status=ComponentStatus.READY,
                message="OK",
                details={
                    'total_validations': metrics.total_validations,
                    'pass_rate': round(metrics.pass_rate, 4),
                    'rules_count': len(self._rules),
                }
            )

        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                status=ComponentStatus.ERROR,
                message=str(e)
            )

    # === Debug ===

    def get_debug_info(self) -> Dict[str, Any]:
        """디버그 정보 (확장)"""
        base_info = super().get_debug_info()
        base_info.update({
            'validator_type': self.validator_type,
            'validation_config': self._config.to_dict(),
            'validation_metrics': self.get_validation_metrics(),
            'rules': self.get_rules(),
            'custom_validators_count': len(self._custom_validators),
        })
        return base_info

    # === Private Helpers ===

    def _calculate_score(self, result: ValidationResult) -> float:
        """품질 점수 계산"""
        if not result.issues:
            return 1.0

        # 이슈 심각도별 감점
        penalties = {
            ValidationSeverity.ERROR: 0.3,
            ValidationSeverity.WARNING: 0.1,
            ValidationSeverity.INFO: 0.02,
        }

        score = 1.0
        for issue in result.issues:
            score -= penalties.get(issue.severity, 0.05)

        return max(0.0, score)

    def _auto_correct(
        self,
        response: str,
        result: ValidationResult
    ) -> Optional[str]:
        """
        자동 수정 시도

        Args:
            response: 원본 응답
            result: 검증 결과

        Returns:
            수정된 응답 또는 None
        """
        corrected = response

        for issue in result.issues:
            if issue.type == ValidationType.LENGTH:
                # 길이 수정
                if len(corrected) > self._config.max_length:
                    corrected = self._truncate_smart(
                        corrected,
                        self._config.max_length
                    )

            elif issue.type == ValidationType.CONTENT:
                # 공백 정리
                corrected = corrected.strip()
                corrected = re.sub(r'\n{3,}', '\n\n', corrected)
                corrected = re.sub(r' {2,}', ' ', corrected)

        # 변경사항이 있으면 반환
        if corrected != response:
            return corrected

        return None

    def _truncate_smart(self, content: str, max_length: int) -> str:
        """스마트 자르기"""
        if len(content) <= max_length:
            return content

        # 문장 단위로 자르기
        effective_max = max_length - 3
        truncated = content[:effective_max]

        last_period = max(
            truncated.rfind('.'),
            truncated.rfind('。'),
            truncated.rfind('!'),
            truncated.rfind('?')
        )

        if last_period > effective_max * 0.7:
            return truncated[:last_period + 1]

        return truncated + "..."

    def _record_validation(self, result: ValidationResult, auto_corrected: bool):
        """검증 기록"""
        self._validation_metrics.record_validation(result, auto_corrected)
        self._record_call(result.valid, result.latency_ms)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"brand={self.brand_id}, "
            f"type={self.validator_type})"
        )


# ============================================================
# Exports
# ============================================================

__all__ = [
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
