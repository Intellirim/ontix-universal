"""
Advisor Handler - Production Grade v2.0
조언 기능

Features:
- Filters v2.0 통합 (Quality + Trust)
- 6차원 품질 분석
- 환각 감지 및 경고
- 자동 개선 제안
- 상세 메트릭 로깅
"""

from app.models.feature import FeatureHandler
from app.filters import (
    QualityFilter,
    TrustFilter,
    QualityConfig,
    TrustConfig,
    QualityResult,
    TrustResult,
    QualityLevel,
    TrustLevel,
)
from app.filters.relevance import (
    RelevanceFilter,
    RelevanceConfig,
    RelevanceResult,
    RelevanceLevel,
)
from app.filters.validation import (
    ValidationFilter,
    ValidationConfig,
    ValidationResult,
    ValidationStatus,
    OverallGrade,
    FilterWeight,
)
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

@dataclass
class AdvisorConfig:
    """Advisor 설정"""
    # 필터 활성화
    quality_filter_enabled: bool = True
    trust_filter_enabled: bool = True
    relevance_filter_enabled: bool = True
    validation_filter_enabled: bool = True

    # 품질 임계값
    min_quality_score: float = 0.5
    min_trust_score: float = 0.5
    min_relevance_score: float = 0.5

    # 환각 임계값
    max_hallucination_risk: float = 0.3

    # 종합 검증 임계값
    validation_pass_threshold: float = 0.7
    validation_warning_threshold: float = 0.5

    # 로깅
    log_filter_results: bool = True
    log_suggestions: bool = True

    # 메타데이터에 상세 정보 포함
    include_quality_details: bool = True
    include_trust_details: bool = True
    include_relevance_details: bool = True
    include_validation_details: bool = True


# ============================================================
# Advisor Handler
# ============================================================

class AdvisorHandler(FeatureHandler):
    """
    조언 핸들러 - Production Grade v2.0

    Features:
    - 조언 생성 전/후 품질 검증
    - 환각(Hallucination) 감지
    - 자동 개선 제안
    - 상세 메트릭 로깅
    """

    def __init__(self, brand_config: Dict[str, Any]):
        super().__init__(brand_config)

        # Advisor 설정 로드
        advisor_config_dict = brand_config.get('advisor', {})
        self.advisor_config = AdvisorConfig(
            quality_filter_enabled=advisor_config_dict.get('quality_filter_enabled', True),
            trust_filter_enabled=advisor_config_dict.get('trust_filter_enabled', True),
            relevance_filter_enabled=advisor_config_dict.get('relevance_filter_enabled', True),
            validation_filter_enabled=advisor_config_dict.get('validation_filter_enabled', True),
            min_quality_score=advisor_config_dict.get('min_quality_score', 0.5),
            min_trust_score=advisor_config_dict.get('min_trust_score', 0.5),
            min_relevance_score=advisor_config_dict.get('min_relevance_score', 0.5),
            max_hallucination_risk=advisor_config_dict.get('max_hallucination_risk', 0.3),
            validation_pass_threshold=advisor_config_dict.get('validation_pass_threshold', 0.7),
            validation_warning_threshold=advisor_config_dict.get('validation_warning_threshold', 0.5),
            log_filter_results=advisor_config_dict.get('log_filter_results', True),
            log_suggestions=advisor_config_dict.get('log_suggestions', True),
            include_quality_details=advisor_config_dict.get('include_quality_details', True),
            include_trust_details=advisor_config_dict.get('include_trust_details', True),
            include_relevance_details=advisor_config_dict.get('include_relevance_details', True),
            include_validation_details=advisor_config_dict.get('include_validation_details', True),
        )

        # Filters v2.0 초기화
        self.quality_filter = QualityFilter(
            config=QualityConfig(
                language="ko",
                min_length=20,
                optimal_min_length=50,
            )
        )
        self.trust_filter = TrustFilter(
            config=TrustConfig(
                min_trust_score=self.advisor_config.min_trust_score,
                hallucination_threshold=self.advisor_config.max_hallucination_risk,
            )
        )
        self.relevance_filter = RelevanceFilter(
            config=RelevanceConfig(
                min_relevance_score=self.advisor_config.min_relevance_score,
            )
        )
        self.validation_filter = ValidationFilter(
            config=ValidationConfig(
                trust_config=TrustConfig(
                    min_trust_score=self.advisor_config.min_trust_score,
                    hallucination_threshold=self.advisor_config.max_hallucination_risk,
                ),
                quality_config=QualityConfig(
                    language="ko",
                    min_length=20,
                ),
                relevance_config=RelevanceConfig(
                    min_relevance_score=self.advisor_config.min_relevance_score,
                ),
                min_pass_score=self.advisor_config.validation_pass_threshold,
                warning_threshold=self.advisor_config.validation_warning_threshold,
            )
        )

        logger.info(
            f"[AdvisorHandler] Initialized for {self.brand_id} "
            f"(Quality={self.advisor_config.quality_filter_enabled}, "
            f"Trust={self.advisor_config.trust_filter_enabled}, "
            f"Relevance={self.advisor_config.relevance_filter_enabled}, "
            f"Validation={self.advisor_config.validation_filter_enabled})"
        )

    def _extract_feature_config(self) -> Dict[str, Any]:
        return self.brand_config.get('advisor', {})

    def can_handle(self, question: str, context: Dict[str, Any]) -> bool:
        """조언 관련 질문인지 판단"""
        keywords = [
            '어떻게', '방법', '조언', '해야', '하면', '좋을까', '팁',
            '추천', '권장', '제안', '도움', '알려', '가르쳐',
            '어떤', '무슨', '뭘', '왜', '언제', '어디서',
        ]
        question_lower = question.lower()

        return any(kw in question_lower for kw in keywords)

    def process(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        조언 처리 및 품질 검증

        Args:
            question: 사용자 질문
            context: 컨텍스트

        Returns:
            처리 결과 (response, metadata, filter_results)
        """
        start_time = datetime.now()

        # 메타데이터 초기화
        metadata = {
            'handled_by': 'advisor',
            'advisor_version': '2.0',
            'processed_at': start_time.isoformat(),
        }

        # 검증 결과 저장소
        filter_results = {
            'quality': None,
            'trust': None,
            'relevance': None,
            'validation': None,
        }
        all_suggestions: List[str] = []

        # 기본 파이프라인에서 응답 생성 (None 반환시 파이프라인이 처리)
        response = self._generate_advice(question, context)

        # 응답이 있으면 필터 적용
        if response:
            # === Quality Filter 적용 ===
            if self.advisor_config.quality_filter_enabled:
                quality_result = self._apply_quality_filter(response, context)
                filter_results['quality'] = quality_result

                if quality_result:
                    all_suggestions.extend(quality_result.get('suggestions', []))

                    # 품질 상세 정보 추가
                    if self.advisor_config.include_quality_details:
                        metadata['quality'] = {
                            'score': quality_result.get('score'),
                            'level': quality_result.get('level'),
                            'valid': quality_result.get('valid'),
                        }

            # === Trust Filter 적용 ===
            if self.advisor_config.trust_filter_enabled:
                trust_result = self._apply_trust_filter(response, question, context)
                filter_results['trust'] = trust_result

                if trust_result:
                    # Trust 상세 정보 추가
                    if self.advisor_config.include_trust_details:
                        metadata['trust'] = {
                            'score': trust_result.get('score'),
                            'level': trust_result.get('level'),
                            'hallucination_risk': trust_result.get('hallucination_risk'),
                            'valid': trust_result.get('valid'),
                        }

                    # 환각 위험 경고
                    if trust_result.get('hallucination_risk', 0) > self.advisor_config.max_hallucination_risk:
                        logger.warning(
                            f"[AdvisorHandler] High hallucination risk detected: "
                            f"{trust_result.get('hallucination_risk'):.2f}"
                        )
                        metadata['hallucination_warning'] = True

            # === Relevance Filter 적용 ===
            if self.advisor_config.relevance_filter_enabled:
                relevance_result = self._apply_relevance_filter(response, question, context)
                filter_results['relevance'] = relevance_result

                if relevance_result:
                    # Relevance 상세 정보 추가
                    if self.advisor_config.include_relevance_details:
                        metadata['relevance'] = {
                            'score': relevance_result.get('score'),
                            'level': relevance_result.get('level'),
                            'response_type': relevance_result.get('response_type'),
                            'valid': relevance_result.get('valid'),
                        }

            # === Validation Filter 적용 (종합 검증) ===
            if self.advisor_config.validation_filter_enabled:
                validation_result = self._apply_validation_filter(response, question, context)
                filter_results['validation'] = validation_result

                if validation_result:
                    all_suggestions.extend(validation_result.get('suggestions', []))

                    # Validation 상세 정보 추가
                    if self.advisor_config.include_validation_details:
                        metadata['validation'] = {
                            'score': validation_result.get('score'),
                            'grade': validation_result.get('grade'),
                            'status': validation_result.get('status'),
                            'valid': validation_result.get('valid'),
                        }

                    # 종합 검증 실패 경고
                    if validation_result.get('status') == 'failed':
                        logger.warning(
                            f"[AdvisorHandler] Validation failed: "
                            f"Grade={validation_result.get('grade')}, "
                            f"Score={validation_result.get('score'):.2f}"
                        )
                        metadata['validation_warning'] = True

            # 개선 제안 로깅
            if self.advisor_config.log_suggestions and all_suggestions:
                logger.info(
                    f"[AdvisorHandler] Improvement suggestions: "
                    f"{', '.join(all_suggestions[:3])}"
                )
                metadata['improvement_suggestions'] = all_suggestions

        # 처리 시간 기록
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        metadata['processing_time_ms'] = round(processing_time, 2)

        return {
            'response': response,
            'metadata': metadata,
            'filter_results': filter_results,
        }

    def _generate_advice(
        self,
        question: str,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """
        조언 생성

        Neo4j 데이터를 검색하고 LLM을 사용하여 데이터 기반 조언을 생성합니다.
        """
        # 파이프라인에 위임하여 Neo4j 검색 + LLM 생성 수행
        # None을 반환하면 engine이 파이프라인을 실행함
        return None

    def _apply_quality_filter(
        self,
        response: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        품질 필터 적용

        Args:
            response: 생성된 응답
            context: 컨텍스트

        Returns:
            품질 검증 결과
        """
        try:
            result: QualityResult = self.quality_filter.validate(response, context)

            # 로깅
            if self.advisor_config.log_filter_results:
                logger.info(
                    f"[AdvisorHandler:Quality] "
                    f"Score={result.score:.2f}, "
                    f"Level={result.level.value}, "
                    f"Valid={result.valid}"
                )

                # 이슈가 있으면 경고
                if result.issues:
                    for issue in result.issues[:3]:
                        logger.warning(
                            f"[AdvisorHandler:Quality] Issue: "
                            f"[{issue.dimension.value}] {issue.message}"
                        )

            return {
                'score': result.score,
                'level': result.level.value,
                'valid': result.valid,
                'issues': [
                    {
                        'dimension': issue.dimension.value,
                        'severity': issue.severity.value,
                        'message': issue.message,
                    }
                    for issue in result.issues
                ],
                'suggestions': result.suggestions,
                'dimension_scores': {
                    dim.value: ds.score
                    for dim, ds in result.dimension_scores.items()
                },
            }

        except Exception as e:
            logger.error(f"[AdvisorHandler:Quality] Filter error: {e}")
            return None

    def _apply_trust_filter(
        self,
        response: str,
        question: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        신뢰성 필터 적용

        Args:
            response: 생성된 응답
            question: 원본 질문
            context: 컨텍스트

        Returns:
            신뢰성 검증 결과
        """
        try:
            # 질문을 컨텍스트에 추가
            trust_context = {
                **context,
                'question': question,
            }

            result: TrustResult = self.trust_filter.validate(response, trust_context)

            # 로깅
            if self.advisor_config.log_filter_results:
                logger.info(
                    f"[AdvisorHandler:Trust] "
                    f"Score={result.score:.2f}, "
                    f"Level={result.level.value}, "
                    f"HallucinationRisk={result.hallucination_risk:.2f}, "
                    f"Valid={result.valid}"
                )

                # 이슈가 있으면 경고
                if result.issues:
                    for issue in result.issues[:3]:
                        logger.warning(
                            f"[AdvisorHandler:Trust] Issue: "
                            f"[{issue.issue_type.value}] {issue.message}"
                        )

            return {
                'score': result.score,
                'level': result.level.value,
                'hallucination_risk': result.hallucination_risk,
                'valid': result.valid,
                'issues': [
                    {
                        'type': issue.issue_type.value,
                        'severity': issue.severity.value,
                        'message': issue.message,
                        'evidence': issue.evidence,
                    }
                    for issue in result.issues
                ],
                'verifications': {
                    vtype.value: {
                        'passed': vr.passed,
                        'confidence': vr.confidence,
                    }
                    for vtype, vr in result.verifications.items()
                },
            }

        except Exception as e:
            logger.error(f"[AdvisorHandler:Trust] Filter error: {e}")
            return None

    def _apply_relevance_filter(
        self,
        response: str,
        question: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        관련성 필터 적용

        Args:
            response: 생성된 응답
            question: 원본 질문
            context: 컨텍스트

        Returns:
            관련성 검증 결과
        """
        try:
            # 질문을 컨텍스트에 추가
            relevance_context = {
                **context,
                'question': question,
            }

            result: RelevanceResult = self.relevance_filter.validate(response, relevance_context)

            # 로깅
            if self.advisor_config.log_filter_results:
                logger.info(
                    f"[AdvisorHandler:Relevance] "
                    f"Score={result.score:.2f}, "
                    f"Level={result.level.value}, "
                    f"ResponseType={result.response_type.value}, "
                    f"Valid={result.valid}"
                )

                # 이슈가 있으면 경고
                if result.issues:
                    for issue in result.issues[:3]:
                        logger.warning(
                            f"[AdvisorHandler:Relevance] Issue: "
                            f"[{issue.relevance_type.value}] {issue.message}"
                        )

            return {
                'score': result.score,
                'level': result.level.value,
                'response_type': result.response_type.value,
                'valid': result.valid,
                'issues': [
                    {
                        'type': issue.relevance_type.value,
                        'severity': issue.severity.value,
                        'message': issue.message,
                    }
                    for issue in result.issues
                ],
                'scores': {
                    rtype.value: rs.score
                    for rtype, rs in result.scores.items()
                } if hasattr(result, 'scores') and result.scores else {},
            }

        except Exception as e:
            logger.error(f"[AdvisorHandler:Relevance] Filter error: {e}")
            return None

    def _apply_validation_filter(
        self,
        response: str,
        question: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        종합 검증 필터 적용

        Args:
            response: 생성된 응답
            question: 원본 질문
            context: 컨텍스트

        Returns:
            종합 검증 결과
        """
        try:
            # 질문을 컨텍스트에 추가
            validation_context = {
                **context,
                'question': question,
            }

            result: ValidationResult = self.validation_filter.validate(response, validation_context)

            # 로깅
            if self.advisor_config.log_filter_results:
                logger.info(
                    f"[AdvisorHandler:Validation] "
                    f"Score={result.score:.2f}, "
                    f"Grade={result.grade.value}, "
                    f"Status={result.status.value}, "
                    f"Valid={result.valid}"
                )

                # 이슈가 있으면 경고
                if result.all_issues:
                    for issue in result.all_issues[:3]:
                        logger.warning(f"[AdvisorHandler:Validation] Issue: {issue}")

            return {
                'score': result.score,
                'grade': result.grade.value,
                'status': result.status.value,
                'valid': result.valid,
                'summary': {
                    'total_issues': result.summary.total_issues,
                    'total_warnings': result.summary.total_warnings,
                    'passed_filters': result.summary.passed_filters,
                    'failed_filters': result.summary.failed_filters,
                },
                'issues': result.all_issues,
                'warnings': result.all_warnings,
                'suggestions': result.suggestions,
            }

        except Exception as e:
            logger.error(f"[AdvisorHandler:Validation] Filter error: {e}")
            return None

    def get_filter_stats(self) -> Dict[str, Any]:
        """필터 통계 반환 (디버깅용)"""
        return {
            'quality_filter_enabled': self.advisor_config.quality_filter_enabled,
            'trust_filter_enabled': self.advisor_config.trust_filter_enabled,
            'relevance_filter_enabled': self.advisor_config.relevance_filter_enabled,
            'validation_filter_enabled': self.advisor_config.validation_filter_enabled,
            'min_quality_score': self.advisor_config.min_quality_score,
            'min_trust_score': self.advisor_config.min_trust_score,
            'min_relevance_score': self.advisor_config.min_relevance_score,
            'validation_pass_threshold': self.advisor_config.validation_pass_threshold,
            'max_hallucination_risk': self.advisor_config.max_hallucination_risk,
        }
