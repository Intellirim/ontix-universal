"""
Validation Filter - Production Grade v2.0
종합 검증 시스템

Features:
- 다중 필터 오케스트레이션
- 가중 점수 계산
- 검증 파이프라인
- 자동 수정 제안
- 상세 리포트 생성
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import logging

from app.filters.trust import (
    TrustFilter,
    TrustResult,
    TrustConfig,
    TrustLevel,
)
from app.filters.quality import (
    QualityFilter,
    QualityResult,
    QualityConfig,
    QualityLevel,
)
from app.filters.relevance import (
    RelevanceFilter,
    RelevanceResult,
    RelevanceConfig,
    RelevanceLevel,
    ResponseType,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class ValidationStatus(Enum):
    """검증 상태"""
    PASSED = "passed"           # 통과
    WARNING = "warning"         # 경고
    FAILED = "failed"           # 실패
    SKIPPED = "skipped"         # 스킵됨


class FilterType(Enum):
    """필터 유형"""
    TRUST = "trust"
    QUALITY = "quality"
    RELEVANCE = "relevance"


class OverallGrade(Enum):
    """종합 등급"""
    A = "A"     # 90-100: 우수
    B = "B"     # 80-89: 양호
    C = "C"     # 70-79: 보통
    D = "D"     # 60-69: 미흡
    F = "F"     # 0-59: 불합격


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FilterWeight:
    """필터 가중치"""
    trust: float = 1.5
    quality: float = 1.2
    relevance: float = 1.3


@dataclass
class ValidationConfig:
    """검증 설정"""
    # 필터별 설정
    trust_config: Optional[TrustConfig] = None
    quality_config: Optional[QualityConfig] = None
    relevance_config: Optional[RelevanceConfig] = None

    # 가중치
    weights: FilterWeight = field(default_factory=FilterWeight)

    # 임계값
    min_pass_score: float = 0.6
    warning_threshold: float = 0.75

    # 필수 필터 (하나라도 실패하면 전체 실패)
    required_filters: List[FilterType] = field(default_factory=lambda: [
        FilterType.TRUST,
        FilterType.QUALITY,
    ])

    # 활성화된 필터
    enabled_filters: List[FilterType] = field(default_factory=lambda: [
        FilterType.TRUST,
        FilterType.QUALITY,
        FilterType.RELEVANCE,
    ])


@dataclass
class FilterResult:
    """개별 필터 결과"""
    filter_type: FilterType
    status: ValidationStatus
    score: float
    weight: float
    valid: bool
    issues: List[str]
    warnings: List[str]
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationSummary:
    """검증 요약"""
    total_issues: int
    total_warnings: int
    critical_issues: int
    passed_filters: int
    failed_filters: int
    skipped_filters: int


@dataclass
class ValidationResult:
    """종합 검증 결과"""
    valid: bool
    status: ValidationStatus
    score: float
    grade: OverallGrade
    filter_results: Dict[FilterType, FilterResult]
    summary: ValidationSummary
    all_issues: List[str]
    all_warnings: List[str]
    suggestions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "status": self.status.value,
            "score": self.score,
            "grade": self.grade.value,
            "filter_results": {
                ft.value: {
                    "status": fr.status.value,
                    "score": fr.score,
                    "valid": fr.valid,
                    "issues": fr.issues,
                    "warnings": fr.warnings,
                    "details": fr.details,
                }
                for ft, fr in self.filter_results.items()
            },
            "summary": {
                "total_issues": self.summary.total_issues,
                "total_warnings": self.summary.total_warnings,
                "critical_issues": self.summary.critical_issues,
                "passed_filters": self.summary.passed_filters,
                "failed_filters": self.summary.failed_filters,
                "skipped_filters": self.summary.skipped_filters,
            },
            "all_issues": self.all_issues,
            "all_warnings": self.all_warnings,
            "suggestions": self.suggestions,
            "metadata": self.metadata,
        }


# ============================================================================
# Main Validation Filter
# ============================================================================

class ValidationFilter:
    """
    종합 검증 필터 - Production Grade v2.0

    통합 검증:
    - 신뢰성 (Trust)
    - 품질 (Quality)
    - 관련성 (Relevance)
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()

        # 개별 필터 초기화
        self._trust_filter = TrustFilter(self.config.trust_config)
        self._quality_filter = QualityFilter(self.config.quality_config)
        self._relevance_filter = RelevanceFilter(self.config.relevance_config)

        # 커스텀 검증기
        self._pre_validators: List[Callable[[str, Dict[str, Any]], Optional[str]]] = []
        self._post_validators: List[Callable[[ValidationResult], ValidationResult]] = []

    def register_pre_validator(
        self,
        validator: Callable[[str, Dict[str, Any]], Optional[str]]
    ) -> None:
        """
        사전 검증기 등록

        Args:
            validator: (response, context) -> error_message or None
        """
        self._pre_validators.append(validator)

    def register_post_validator(
        self,
        validator: Callable[[ValidationResult], ValidationResult]
    ) -> None:
        """
        사후 검증기 등록

        Args:
            validator: (result) -> modified_result
        """
        self._post_validators.append(validator)

    def validate(
        self,
        response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        종합 검증 수행

        Args:
            response: 검증할 응답 텍스트
            context: 컨텍스트 (question, retrieval_results 등)

        Returns:
            ValidationResult: 종합 검증 결과
        """
        context = context or {}
        start_time = datetime.now()

        # 사전 검증
        pre_validation_errors = []
        for validator in self._pre_validators:
            try:
                error = validator(response, context)
                if error:
                    pre_validation_errors.append(error)
            except Exception as e:
                logger.error(f"사전 검증기 오류: {e}")

        if pre_validation_errors:
            return self._create_failed_result(
                pre_validation_errors,
                "사전 검증 실패"
            )

        # 개별 필터 실행
        filter_results: Dict[FilterType, FilterResult] = {}

        # Trust Filter
        if FilterType.TRUST in self.config.enabled_filters:
            trust_result = self._run_trust_filter(response, context)
            filter_results[FilterType.TRUST] = trust_result
        else:
            filter_results[FilterType.TRUST] = self._create_skipped_result(FilterType.TRUST)

        # Quality Filter
        if FilterType.QUALITY in self.config.enabled_filters:
            quality_result = self._run_quality_filter(response, context)
            filter_results[FilterType.QUALITY] = quality_result
        else:
            filter_results[FilterType.QUALITY] = self._create_skipped_result(FilterType.QUALITY)

        # Relevance Filter
        if FilterType.RELEVANCE in self.config.enabled_filters:
            relevance_result = self._run_relevance_filter(response, context)
            filter_results[FilterType.RELEVANCE] = relevance_result
        else:
            filter_results[FilterType.RELEVANCE] = self._create_skipped_result(FilterType.RELEVANCE)

        # 종합 점수 계산
        final_score = self._calculate_weighted_score(filter_results)

        # 필수 필터 체크
        required_failed = any(
            not filter_results[ft].valid
            for ft in self.config.required_filters
            if ft in filter_results and filter_results[ft].status != ValidationStatus.SKIPPED
        )

        # 상태 결정
        if required_failed or final_score < self.config.min_pass_score:
            status = ValidationStatus.FAILED
            valid = False
        elif final_score < self.config.warning_threshold:
            status = ValidationStatus.WARNING
            valid = True
        else:
            status = ValidationStatus.PASSED
            valid = True

        # 등급 결정
        grade = self._determine_grade(final_score)

        # 요약 생성
        summary = self._create_summary(filter_results)

        # 이슈 및 경고 수집
        all_issues = []
        all_warnings = []
        for fr in filter_results.values():
            all_issues.extend(fr.issues)
            all_warnings.extend(fr.warnings)

        # 제안 생성
        suggestions = self._generate_suggestions(filter_results, final_score)

        # 결과 생성
        result = ValidationResult(
            valid=valid,
            status=status,
            score=round(final_score, 3),
            grade=grade,
            filter_results=filter_results,
            summary=summary,
            all_issues=all_issues,
            all_warnings=all_warnings,
            suggestions=suggestions,
            metadata={
                "validated_at": start_time.isoformat(),
                "duration_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "response_length": len(response),
                "enabled_filters": [f.value for f in self.config.enabled_filters],
                "required_filters": [f.value for f in self.config.required_filters],
            }
        )

        # 사후 검증
        for validator in self._post_validators:
            try:
                result = validator(result)
            except Exception as e:
                logger.error(f"사후 검증기 오류: {e}")

        return result

    def _run_trust_filter(
        self,
        response: str,
        context: Dict[str, Any]
    ) -> FilterResult:
        """Trust 필터 실행"""
        try:
            result: TrustResult = self._trust_filter.validate(response, context)

            status = (
                ValidationStatus.PASSED if result.valid
                else ValidationStatus.FAILED
            )
            if result.valid and result.hallucination_risk > 0.1:
                status = ValidationStatus.WARNING

            issues = [i.message for i in result.issues]
            warnings = []

            if result.level in (TrustLevel.LOW, TrustLevel.UNVERIFIED):
                warnings.append(f"신뢰 수준이 낮습니다: {result.level.value}")

            return FilterResult(
                filter_type=FilterType.TRUST,
                status=status,
                score=result.score,
                weight=self.config.weights.trust,
                valid=result.valid,
                issues=issues,
                warnings=warnings,
                details={
                    "level": result.level.value,
                    "hallucination_risk": result.hallucination_risk,
                    "verifications": {
                        k.value: v.passed for k, v in result.verifications.items()
                    },
                }
            )
        except Exception as e:
            logger.error(f"Trust 필터 오류: {e}")
            return self._create_error_result(FilterType.TRUST, str(e))

    def _run_quality_filter(
        self,
        response: str,
        context: Dict[str, Any]
    ) -> FilterResult:
        """Quality 필터 실행"""
        try:
            result: QualityResult = self._quality_filter.validate(response, context)

            status = (
                ValidationStatus.PASSED if result.valid
                else ValidationStatus.FAILED
            )
            if result.valid and result.level == QualityLevel.ACCEPTABLE:
                status = ValidationStatus.WARNING

            issues = [i.message for i in result.issues]
            warnings = result.warnings

            return FilterResult(
                filter_type=FilterType.QUALITY,
                status=status,
                score=result.score,
                weight=self.config.weights.quality,
                valid=result.valid,
                issues=issues,
                warnings=warnings,
                details={
                    "level": result.level.value,
                    "dimension_scores": {
                        k.value: v.score for k, v in result.dimension_scores.items()
                    },
                    "suggestions": result.suggestions,
                }
            )
        except Exception as e:
            logger.error(f"Quality 필터 오류: {e}")
            return self._create_error_result(FilterType.QUALITY, str(e))

    def _run_relevance_filter(
        self,
        response: str,
        context: Dict[str, Any]
    ) -> FilterResult:
        """Relevance 필터 실행"""
        try:
            result: RelevanceResult = self._relevance_filter.validate(response, context)

            status = (
                ValidationStatus.PASSED if result.valid
                else ValidationStatus.FAILED
            )
            if result.valid and result.level == RelevanceLevel.PARTIALLY_RELEVANT:
                status = ValidationStatus.WARNING

            issues = [i.message for i in result.issues]
            warnings = result.warnings

            # 회피성 응답 경고
            if result.response_type == ResponseType.EVASIVE:
                issues.append("회피성 응답이 감지되었습니다")
            elif result.response_type == ResponseType.OFF_TOPIC:
                issues.append("주제 이탈이 감지되었습니다")

            return FilterResult(
                filter_type=FilterType.RELEVANCE,
                status=status,
                score=result.score,
                weight=self.config.weights.relevance,
                valid=result.valid,
                issues=issues,
                warnings=warnings,
                details={
                    "level": result.level.value,
                    "response_type": result.response_type.value,
                    "relevance_scores": {
                        k.value: v.score for k, v in result.scores.items()
                    },
                }
            )
        except Exception as e:
            logger.error(f"Relevance 필터 오류: {e}")
            return self._create_error_result(FilterType.RELEVANCE, str(e))

    def _calculate_weighted_score(
        self,
        filter_results: Dict[FilterType, FilterResult]
    ) -> float:
        """가중 평균 점수 계산"""
        active_results = [
            fr for fr in filter_results.values()
            if fr.status != ValidationStatus.SKIPPED
        ]

        if not active_results:
            return 0.0

        total_weight = sum(fr.weight for fr in active_results)
        weighted_sum = sum(fr.score * fr.weight for fr in active_results)

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _determine_grade(self, score: float) -> OverallGrade:
        """등급 결정"""
        percentage = score * 100
        if percentage >= 90:
            return OverallGrade.A
        elif percentage >= 80:
            return OverallGrade.B
        elif percentage >= 70:
            return OverallGrade.C
        elif percentage >= 60:
            return OverallGrade.D
        else:
            return OverallGrade.F

    def _create_summary(
        self,
        filter_results: Dict[FilterType, FilterResult]
    ) -> ValidationSummary:
        """검증 요약 생성"""
        total_issues = sum(len(fr.issues) for fr in filter_results.values())
        total_warnings = sum(len(fr.warnings) for fr in filter_results.values())

        # 심각한 이슈 카운트 (간단히 "critical", "치명" 등 포함 여부로 판단)
        critical_keywords = ['critical', 'critical', '치명', '심각', '에러']
        critical_issues = sum(
            1 for fr in filter_results.values()
            for issue in fr.issues
            if any(kw in issue.lower() for kw in critical_keywords)
        )

        passed = sum(1 for fr in filter_results.values() if fr.status == ValidationStatus.PASSED)
        failed = sum(1 for fr in filter_results.values() if fr.status == ValidationStatus.FAILED)
        skipped = sum(1 for fr in filter_results.values() if fr.status == ValidationStatus.SKIPPED)

        return ValidationSummary(
            total_issues=total_issues,
            total_warnings=total_warnings,
            critical_issues=critical_issues,
            passed_filters=passed,
            failed_filters=failed,
            skipped_filters=skipped,
        )

    def _generate_suggestions(
        self,
        filter_results: Dict[FilterType, FilterResult],
        final_score: float
    ) -> List[str]:
        """개선 제안 생성"""
        suggestions = []

        # 점수가 낮은 필터에 대한 제안
        for ft, fr in filter_results.items():
            if fr.status == ValidationStatus.SKIPPED:
                continue

            if fr.score < 0.5:
                if ft == FilterType.TRUST:
                    suggestions.append("검색 데이터를 기반으로 응답하고, 불확실한 정보에는 출처를 명시하세요")
                elif ft == FilterType.QUALITY:
                    suggestions.append("응답의 구조와 완결성을 개선하세요")
                elif ft == FilterType.RELEVANCE:
                    suggestions.append("질문에 더 직접적으로 답변하세요")

        # 전체 점수 기반 제안
        if final_score < 0.6:
            suggestions.append("전반적인 응답 품질 개선이 필요합니다")
        elif final_score < 0.75:
            suggestions.append("응답이 양호하지만 개선의 여지가 있습니다")

        # 상세 제안 (필터별 details에서 수집)
        for ft, fr in filter_results.items():
            if 'suggestions' in fr.details:
                suggestions.extend(fr.details['suggestions'][:2])  # 최대 2개

        return list(set(suggestions))[:5]  # 중복 제거 후 최대 5개

    def _create_skipped_result(self, filter_type: FilterType) -> FilterResult:
        """스킵된 필터 결과 생성"""
        return FilterResult(
            filter_type=filter_type,
            status=ValidationStatus.SKIPPED,
            score=0.0,
            weight=0.0,
            valid=True,
            issues=[],
            warnings=[],
            details={"reason": "filter_disabled"},
        )

    def _create_error_result(self, filter_type: FilterType, error: str) -> FilterResult:
        """에러 발생 시 필터 결과 생성"""
        return FilterResult(
            filter_type=filter_type,
            status=ValidationStatus.FAILED,
            score=0.0,
            weight=getattr(self.config.weights, filter_type.value, 1.0),
            valid=False,
            issues=[f"필터 실행 오류: {error}"],
            warnings=[],
            details={"error": error},
        )

    def _create_failed_result(
        self,
        errors: List[str],
        reason: str
    ) -> ValidationResult:
        """실패 결과 생성"""
        return ValidationResult(
            valid=False,
            status=ValidationStatus.FAILED,
            score=0.0,
            grade=OverallGrade.F,
            filter_results={},
            summary=ValidationSummary(
                total_issues=len(errors),
                total_warnings=0,
                critical_issues=len(errors),
                passed_filters=0,
                failed_filters=0,
                skipped_filters=0,
            ),
            all_issues=errors,
            all_warnings=[],
            suggestions=["사전 검증 오류를 해결하세요"],
            metadata={"failure_reason": reason},
        )

    @staticmethod
    def quick_validate(response: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        빠른 검증 (정적 메서드, 하위 호환성)

        Args:
            response: 검증할 응답
            context: 컨텍스트

        Returns:
            검증 결과 딕셔너리 (이전 버전 호환)
        """
        filter_instance = ValidationFilter()
        result = filter_instance.validate(response, context or {})

        # 하위 호환성을 위한 간단한 형식
        return {
            'valid': result.valid,
            'score': result.score,
            'issues': result.all_issues,
            'warnings': result.all_warnings,
            'filter_results': {
                ft.value: {
                    'valid': fr.valid,
                    'issues': fr.issues,
                    'score': fr.score,
                }
                for ft, fr in result.filter_results.items()
            },
        }


# ============================================================================
# Convenience Functions (하위 호환성)
# ============================================================================

def validate_response(response: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    응답 검증 편의 함수

    Args:
        response: 검증할 응답
        context: 컨텍스트

    Returns:
        검증 결과 딕셔너리
    """
    return ValidationFilter.quick_validate(response, context)
