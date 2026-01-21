"""
Analytics Handler - Production Grade v2.0
분석 기능

Features:
- Filters v2.0 통합 (Quality + Trust)
- 데이터 기반 신뢰성 검증
- 분석 결과 품질 관리
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
class AnalyticsConfig:
    """분석 설정"""
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

    # 분석 설정
    max_results: int = 10
    include_trends: bool = True

    # 로깅
    log_filter_results: bool = True
    log_analytics: bool = True

    # 메타데이터
    include_quality_details: bool = True
    include_trust_details: bool = True
    include_relevance_details: bool = True
    include_validation_details: bool = True


# ============================================================
# Analytics Handler
# ============================================================

class AnalyticsHandler(FeatureHandler):
    """
    분석 핸들러 - Production Grade v2.0

    Features:
    - 인기 콘텐츠 분석
    - 트렌드 분석
    - 분석 결과 품질 검증
    - 데이터 신뢰성 검증
    """

    def __init__(self, brand_config: Dict[str, Any]):
        super().__init__(brand_config)

        # 설정 로드
        config_dict = brand_config.get('analytics', {})
        self.handler_config = AnalyticsConfig(
            quality_filter_enabled=config_dict.get('quality_filter_enabled', True),
            trust_filter_enabled=config_dict.get('trust_filter_enabled', True),
            relevance_filter_enabled=config_dict.get('relevance_filter_enabled', True),
            validation_filter_enabled=config_dict.get('validation_filter_enabled', True),
            min_quality_score=config_dict.get('min_quality_score', 0.5),
            min_trust_score=config_dict.get('min_trust_score', 0.5),
            min_relevance_score=config_dict.get('min_relevance_score', 0.5),
            validation_pass_threshold=config_dict.get('validation_pass_threshold', 0.7),
            validation_warning_threshold=config_dict.get('validation_warning_threshold', 0.5),
            max_results=config_dict.get('max_results', 10),
            include_trends=config_dict.get('include_trends', True),
            log_filter_results=config_dict.get('log_filter_results', True),
            log_analytics=config_dict.get('log_analytics', True),
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
            f"[AnalyticsHandler] Initialized for {self.brand_id} "
            f"(Quality={self.handler_config.quality_filter_enabled}, "
            f"Trust={self.handler_config.trust_filter_enabled}, "
            f"Relevance={self.handler_config.relevance_filter_enabled}, "
            f"Validation={self.handler_config.validation_filter_enabled})"
        )

    def _extract_feature_config(self) -> Dict[str, Any]:
        return self.brand_config.get('analytics', {})

    def can_handle(self, question: str, context: Dict[str, Any]) -> bool:
        """분석 관련 질문인지 판단"""
        keywords = [
            '인기', '많은', 'top', '순위', '베스트', '통계', '분석',
            '트렌드', '추이', '성과', '성장', '비교', '평균',
            '총', '전체', '합계', '몇 개', '몇 명', '얼마나',
        ]
        question_lower = question.lower()

        return any(kw in question_lower for kw in keywords)

    def process(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        분석 처리 및 품질 검증

        Args:
            question: 사용자 질문
            context: 컨텍스트

        Returns:
            처리 결과 (response, metadata, filter_results)
        """
        start_time = datetime.now()

        # 메타데이터 초기화
        metadata = {
            'handled_by': 'analytics',
            'handler_version': '2.0',
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

        try:
            # 분석 데이터 조회
            analytics_data = self._fetch_analytics_data()
            metadata['data_count'] = len(analytics_data.get('posts', []))

            # 응답 생성
            if analytics_data.get('posts'):
                response = self._format_analytics(analytics_data)

                if self.handler_config.log_analytics:
                    logger.info(
                        f"[AnalyticsHandler] Analyzed {len(analytics_data['posts'])} items "
                        f"for question: {question[:50]}..."
                    )
            else:
                response = self._generate_no_data_response()

            # === Quality Filter 적용 ===
            if self.handler_config.quality_filter_enabled and response:
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
            if self.handler_config.trust_filter_enabled and response:
                trust_context = {
                    **context,
                    'retrieval_results': analytics_data,
                }
                trust_result = self._apply_trust_filter(response, question, trust_context)
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
            if self.handler_config.relevance_filter_enabled and response:
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
            if self.handler_config.validation_filter_enabled and response:
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
                            f"[AnalyticsHandler] Validation failed: "
                            f"Grade={validation_result.get('grade')}, "
                            f"Score={validation_result.get('score'):.2f}"
                        )
                        metadata['validation_warning'] = True

            # 개선 제안 추가
            if self.handler_config.log_filter_results and all_suggestions:
                logger.info(
                    f"[AnalyticsHandler] Suggestions: {', '.join(all_suggestions[:3])}"
                )
                metadata['improvement_suggestions'] = all_suggestions

            # 분석 메트릭 추가
            if analytics_data.get('posts'):
                metadata['analytics_summary'] = {
                    'total_posts': len(analytics_data['posts']),
                    'total_likes': analytics_data.get('total_likes', 0),
                    'avg_likes': analytics_data.get('avg_likes', 0),
                }

        except Exception as e:
            logger.error(f"[AnalyticsHandler] Error: {e}")
            response = "분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
            metadata['error'] = str(e)

        # 처리 시간 기록
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        metadata['processing_time_ms'] = round(processing_time, 2)

        return {
            'response': response,
            'metadata': metadata,
            'filter_results': filter_results,
        }

    def _fetch_analytics_data(self) -> Dict[str, Any]:
        """분석 데이터 조회"""
        try:
            from app.services.shared.neo4j import get_neo4j_client

            neo4j = get_neo4j_client()

            query = """
            MATCH (p:Post)
            WHERE p.brand_id = $brand_id
              AND p.likes IS NOT NULL
            RETURN p.id as id, p.content as content, p.likes as likes,
                   p.comments as comments, p.shares as shares
            ORDER BY p.likes DESC
            LIMIT $limit
            """

            posts = neo4j.query(query, {
                'brand_id': self.brand_id,
                'limit': self.handler_config.max_results,
            }) or []

            total_likes = sum(p.get('likes', 0) for p in posts)
            avg_likes = total_likes / len(posts) if posts else 0

            return {
                'posts': posts,
                'total_likes': total_likes,
                'avg_likes': round(avg_likes, 1),
            }

        except Exception as e:
            logger.error(f"[AnalyticsHandler] Fetch error: {e}")
            return {'posts': [], 'total_likes': 0, 'avg_likes': 0}

    def _format_analytics(self, data: Dict[str, Any]) -> str:
        """분석 결과 포맷팅"""
        posts = data.get('posts', [])
        total_likes = data.get('total_likes', 0)
        avg_likes = data.get('avg_likes', 0)

        lines = ["📊 **분석 결과 리포트**\n"]

        lines.append("**📈 요약 통계**")
        lines.append(f"- 분석 게시물: {len(posts)}개")
        lines.append(f"- 총 좋아요: {total_likes:,}개")
        lines.append(f"- 평균 좋아요: {avg_likes:,.1f}개")
        lines.append("")

        lines.append("**🏆 TOP 5 인기 게시물**")
        for i, post in enumerate(posts[:5], 1):
            content = post.get('content', '')[:40]
            likes = post.get('likes', 0)
            comments = post.get('comments', 0) or 0

            lines.append(f"{i}. {content}...")
            lines.append(f"   ❤️ {likes:,} | 💬 {comments:,}")

        lines.append("")
        lines.append("더 자세한 분석이 필요하시면 말씀해주세요!")

        return "\n".join(lines)

    def _generate_no_data_response(self) -> str:
        """데이터 없을 때 응답"""
        return (
            "분석할 데이터가 아직 충분하지 않습니다.\n\n"
            "데이터가 수집되면 다음 분석을 제공해드릴 수 있습니다:\n"
            "- 인기 게시물 순위\n"
            "- 좋아요/댓글 트렌드\n"
            "- 성과 비교 분석"
        )

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
                    f"[AnalyticsHandler:Quality] "
                    f"Score={result.score:.2f}, Level={result.level.value}"
                )

            return {
                'score': result.score,
                'level': result.level.value,
                'valid': result.valid,
                'suggestions': result.suggestions,
            }

        except Exception as e:
            logger.error(f"[AnalyticsHandler:Quality] Error: {e}")
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
                    f"[AnalyticsHandler:Trust] "
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
            logger.error(f"[AnalyticsHandler:Trust] Error: {e}")
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
                    f"[AnalyticsHandler:Relevance] "
                    f"Score={result.score:.2f}, "
                    f"Level={result.level.value}, "
                    f"ResponseType={result.response_type.value}, "
                    f"Valid={result.valid}"
                )

                # 이슈가 있으면 경고
                if result.issues:
                    for issue in result.issues[:3]:
                        logger.warning(
                            f"[AnalyticsHandler:Relevance] Issue: "
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
            logger.error(f"[AnalyticsHandler:Relevance] Filter error: {e}")
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
                    f"[AnalyticsHandler:Validation] "
                    f"Score={result.score:.2f}, "
                    f"Grade={result.grade.value}, "
                    f"Status={result.status.value}, "
                    f"Valid={result.valid}"
                )

                # 이슈가 있으면 경고
                if result.all_issues:
                    for issue in result.all_issues[:3]:
                        logger.warning(f"[AnalyticsHandler:Validation] Issue: {issue}")

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
            logger.error(f"[AnalyticsHandler:Validation] Filter error: {e}")
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
            'max_results': self.handler_config.max_results,
        }
