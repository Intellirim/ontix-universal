"""
Onboarding Handler - Production Grade v2.0
온보딩 기능

Features:
- Filters v2.0 통합 (Quality + Trust)
- 단계별 온보딩 안내
- 사용자 맞춤 가이드
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
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

@dataclass
class OnboardingConfig:
    """온보딩 설정"""
    # 필터 활성화
    quality_filter_enabled: bool = True
    trust_filter_enabled: bool = True
    relevance_filter_enabled: bool = True
    validation_filter_enabled: bool = True

    # 품질 임계값
    min_quality_score: float = 0.5
    min_trust_score: float = 0.5
    min_relevance_score: float = 0.5

    # 종합 검증 임계값
    validation_pass_threshold: float = 0.7
    validation_warning_threshold: float = 0.5

    # 온보딩 설정
    steps_enabled: bool = True
    personalized: bool = True
    include_tips: bool = True

    # 로깅
    log_filter_results: bool = True
    log_onboarding: bool = True

    # 메타데이터
    include_quality_details: bool = True
    include_trust_details: bool = True
    include_relevance_details: bool = True
    include_validation_details: bool = True


# ============================================================
# Onboarding Handler
# ============================================================

class OnboardingHandler(FeatureHandler):
    """
    온보딩 핸들러 - Production Grade v2.0

    Features:
    - 신규 사용자 안내
    - 단계별 가이드 제공
    - 온보딩 응답 품질 검증
    - 맞춤형 추천
    """

    def __init__(self, brand_config: Dict[str, Any]):
        super().__init__(brand_config)

        # 설정 로드
        config_dict = brand_config.get('onboarding', {})
        self.handler_config = OnboardingConfig(
            quality_filter_enabled=config_dict.get('quality_filter_enabled', True),
            trust_filter_enabled=config_dict.get('trust_filter_enabled', True),
            relevance_filter_enabled=config_dict.get('relevance_filter_enabled', True),
            validation_filter_enabled=config_dict.get('validation_filter_enabled', True),
            min_quality_score=config_dict.get('min_quality_score', 0.5),
            min_trust_score=config_dict.get('min_trust_score', 0.5),
            min_relevance_score=config_dict.get('min_relevance_score', 0.5),
            validation_pass_threshold=config_dict.get('validation_pass_threshold', 0.7),
            validation_warning_threshold=config_dict.get('validation_warning_threshold', 0.5),
            steps_enabled=config_dict.get('steps_enabled', True),
            personalized=config_dict.get('personalized', True),
            include_tips=config_dict.get('include_tips', True),
            log_filter_results=config_dict.get('log_filter_results', True),
            log_onboarding=config_dict.get('log_onboarding', True),
            include_quality_details=config_dict.get('include_quality_details', True),
            include_trust_details=config_dict.get('include_trust_details', True),
            include_relevance_details=config_dict.get('include_relevance_details', True),
            include_validation_details=config_dict.get('include_validation_details', True),
        )

        # Filters v2.0 초기화
        self.quality_filter = QualityFilter(
            config=QualityConfig(
                language="ko",
                min_length=50,
                optimal_min_length=150,
            )
        )
        self.trust_filter = TrustFilter(
            config=TrustConfig(
                min_trust_score=self.handler_config.min_trust_score,
            )
        )
        self.relevance_filter = RelevanceFilter(
            config=RelevanceConfig(
                min_relevance_score=self.handler_config.min_relevance_score,
            )
        )
        self.validation_filter = ValidationFilter(
            config=ValidationConfig(
                trust_config=TrustConfig(
                    min_trust_score=self.handler_config.min_trust_score,
                ),
                quality_config=QualityConfig(
                    language="ko",
                    min_length=50,
                ),
                relevance_config=RelevanceConfig(
                    min_relevance_score=self.handler_config.min_relevance_score,
                ),
                min_pass_score=self.handler_config.validation_pass_threshold,
                warning_threshold=self.handler_config.validation_warning_threshold,
            )
        )

        logger.info(
            f"[OnboardingHandler] Initialized for {self.brand_id} "
            f"(Quality={self.handler_config.quality_filter_enabled}, "
            f"Trust={self.handler_config.trust_filter_enabled}, "
            f"Relevance={self.handler_config.relevance_filter_enabled}, "
            f"Validation={self.handler_config.validation_filter_enabled})"
        )

    def _extract_feature_config(self) -> Dict[str, Any]:
        return self.brand_config.get('onboarding', {})

    def can_handle(self, question: str, context: Dict[str, Any]) -> bool:
        """온보딩 관련 질문인지 판단"""
        keywords = [
            '시작', '처음', '온보딩', '가입', '등록',
            '사용법', '어떻게 쓰', '뭐부터', '안내', '소개',
            '첫', '입문', '초보', '신규', '새로',
        ]
        question_lower = question.lower()

        return any(kw in question_lower for kw in keywords)

    def process(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        온보딩 처리 및 품질 검증

        Args:
            question: 사용자 질문
            context: 컨텍스트

        Returns:
            처리 결과 (response, metadata, filter_results)
        """
        start_time = datetime.now()

        # 메타데이터 초기화
        metadata = {
            'handled_by': 'onboarding',
            'handler_version': '2.0',
            'processed_at': start_time.isoformat(),
            'onboarding_type': self._detect_onboarding_type(question),
        }

        # 검증 결과 저장소
        filter_results = {
            'quality': None,
            'trust': None,
            'relevance': None,
            'validation': None,
        }
        all_suggestions: List[str] = []

        # 온보딩 응답 생성
        response = self._generate_onboarding_response(question, context)

        if response:
            # === Quality Filter 적용 ===
            if self.handler_config.quality_filter_enabled:
                quality_result = self._apply_quality_filter(response, context)
                filter_results['quality'] = quality_result

                if quality_result:
                    all_suggestions.extend(quality_result.get('suggestions', []))

                    if self.handler_config.include_quality_details:
                        metadata['quality'] = {
                            'score': quality_result.get('score'),
                            'level': quality_result.get('level'),
                            'valid': quality_result.get('valid'),
                        }

            # === Trust Filter 적용 ===
            if self.handler_config.trust_filter_enabled:
                trust_result = self._apply_trust_filter(response, question, context)
                filter_results['trust'] = trust_result

                if trust_result:
                    if self.handler_config.include_trust_details:
                        metadata['trust'] = {
                            'score': trust_result.get('score'),
                            'level': trust_result.get('level'),
                            'hallucination_risk': trust_result.get('hallucination_risk'),
                            'valid': trust_result.get('valid'),
                        }

            # === Relevance Filter 적용 ===
            if self.handler_config.relevance_filter_enabled:
                relevance_result = self._apply_relevance_filter(response, question, context)
                filter_results['relevance'] = relevance_result

                if relevance_result:
                    if self.handler_config.include_relevance_details:
                        metadata['relevance'] = {
                            'score': relevance_result.get('score'),
                            'level': relevance_result.get('level'),
                            'response_type': relevance_result.get('response_type'),
                            'valid': relevance_result.get('valid'),
                        }

            # === Validation Filter 적용 (종합 검증) ===
            if self.handler_config.validation_filter_enabled:
                validation_result = self._apply_validation_filter(response, question, context)
                filter_results['validation'] = validation_result

                if validation_result:
                    all_suggestions.extend(validation_result.get('suggestions', []))

                    if self.handler_config.include_validation_details:
                        metadata['validation'] = {
                            'score': validation_result.get('score'),
                            'grade': validation_result.get('grade'),
                            'status': validation_result.get('status'),
                            'valid': validation_result.get('valid'),
                        }

                    # 종합 검증 실패 경고
                    if validation_result.get('status') == 'failed':
                        logger.warning(
                            f"[OnboardingHandler] Validation failed: "
                            f"Grade={validation_result.get('grade')}, "
                            f"Score={validation_result.get('score'):.2f}"
                        )
                        metadata['validation_warning'] = True

            # 개선 제안 추가
            if self.handler_config.log_filter_results and all_suggestions:
                logger.info(
                    f"[OnboardingHandler] Suggestions: {', '.join(all_suggestions[:3])}"
                )
                metadata['improvement_suggestions'] = all_suggestions

            if self.handler_config.log_onboarding:
                logger.info(
                    f"[OnboardingHandler] Processed {metadata['onboarding_type']} "
                    f"for: {question[:50]}..."
                )

        # 처리 시간 기록
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        metadata['processing_time_ms'] = round(processing_time, 2)

        return {
            'response': response,
            'metadata': metadata,
            'filter_results': filter_results,
        }

    def _detect_onboarding_type(self, question: str) -> str:
        """온보딩 타입 감지"""
        question_lower = question.lower()

        if any(kw in question_lower for kw in ['가입', '등록', '계정']):
            return 'registration'
        elif any(kw in question_lower for kw in ['사용법', '어떻게', '방법']):
            return 'tutorial'
        elif any(kw in question_lower for kw in ['소개', '뭐', '어떤']):
            return 'introduction'
        else:
            return 'general'

    def _generate_onboarding_response(
        self,
        question: str,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """
        온보딩 응답 생성

        기본적으로 None을 반환하여 파이프라인이 처리하도록 위임.
        """
        # 기본적으로 파이프라인에 위임
        return None

    def _apply_quality_filter(
        self,
        response: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """품질 필터 적용"""
        try:
            result: QualityResult = self.quality_filter.validate(response, context)

            if self.handler_config.log_filter_results:
                logger.info(
                    f"[OnboardingHandler:Quality] "
                    f"Score={result.score:.2f}, Level={result.level.value}"
                )

            return {
                'score': result.score,
                'level': result.level.value,
                'valid': result.valid,
                'suggestions': result.suggestions,
            }

        except Exception as e:
            logger.error(f"[OnboardingHandler:Quality] Error: {e}")
            return None

    def _apply_trust_filter(
        self,
        response: str,
        question: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """신뢰성 필터 적용"""
        try:
            trust_context = {
                **context,
                'question': question,
            }

            result: TrustResult = self.trust_filter.validate(response, trust_context)

            if self.handler_config.log_filter_results:
                logger.info(
                    f"[OnboardingHandler:Trust] "
                    f"Score={result.score:.2f}, "
                    f"HallucinationRisk={result.hallucination_risk:.2f}"
                )

            return {
                'score': result.score,
                'level': result.level.value,
                'hallucination_risk': result.hallucination_risk,
                'valid': result.valid,
            }

        except Exception as e:
            logger.error(f"[OnboardingHandler:Trust] Error: {e}")
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
            if self.handler_config.log_filter_results:
                logger.info(
                    f"[OnboardingHandler:Relevance] "
                    f"Score={result.score:.2f}, "
                    f"Level={result.level.value}, "
                    f"ResponseType={result.response_type.value}, "
                    f"Valid={result.valid}"
                )

                # 이슈가 있으면 경고
                if result.issues:
                    for issue in result.issues[:3]:
                        logger.warning(
                            f"[OnboardingHandler:Relevance] Issue: "
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
            logger.error(f"[OnboardingHandler:Relevance] Filter error: {e}")
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
            if self.handler_config.log_filter_results:
                logger.info(
                    f"[OnboardingHandler:Validation] "
                    f"Score={result.score:.2f}, "
                    f"Grade={result.grade.value}, "
                    f"Status={result.status.value}, "
                    f"Valid={result.valid}"
                )

                # 이슈가 있으면 경고
                if result.all_issues:
                    for issue in result.all_issues[:3]:
                        logger.warning(f"[OnboardingHandler:Validation] Issue: {issue}")

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
            logger.error(f"[OnboardingHandler:Validation] Filter error: {e}")
            return None

    def get_filter_stats(self) -> Dict[str, Any]:
        """필터 통계 반환"""
        return {
            'quality_filter_enabled': self.handler_config.quality_filter_enabled,
            'trust_filter_enabled': self.handler_config.trust_filter_enabled,
            'relevance_filter_enabled': self.handler_config.relevance_filter_enabled,
            'validation_filter_enabled': self.handler_config.validation_filter_enabled,
            'min_quality_score': self.handler_config.min_quality_score,
            'min_trust_score': self.handler_config.min_trust_score,
            'min_relevance_score': self.handler_config.min_relevance_score,
            'validation_pass_threshold': self.handler_config.validation_pass_threshold,
            'steps_enabled': self.handler_config.steps_enabled,
        }
