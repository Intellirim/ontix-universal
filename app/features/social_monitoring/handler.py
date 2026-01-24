"""
Social Monitoring Handler - Production Grade v2.0
소셜 모니터링 기능

Features:
- Filters v2.0 통합 (Quality + Trust)
- 실시간 소셜 미디어 모니터링
- 멘션/해시태그 추적
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
from app.services.analysis.sentiment import (
    SentimentAnalyzer,
    SentimentConfig,
    SentimentResult,
    SentimentLabel,
    AnalysisMode,
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
class SocialMonitoringConfig:
    """소셜 모니터링 설정"""
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

    # 모니터링 설정
    platforms: List[str] = None
    track_mentions: bool = True
    track_hashtags: bool = True
    max_results: int = 20

    # 감정 분석 설정
    sentiment_analysis_enabled: bool = True
    sentiment_analysis_mode: str = "auto"  # fast, accurate, auto

    # 로깅
    log_filter_results: bool = True
    log_monitoring: bool = True

    # 메타데이터
    include_quality_details: bool = True
    include_trust_details: bool = True
    include_relevance_details: bool = True
    include_validation_details: bool = True
    include_sentiment_details: bool = True

    def __post_init__(self):
        if self.platforms is None:
            self.platforms = ['instagram', 'twitter', 'youtube', 'tiktok']


# ============================================================
# Social Monitoring Handler
# ============================================================

class SocialMonitoringHandler(FeatureHandler):
    """
    소셜 모니터링 핸들러 - Production Grade v2.0

    Features:
    - 멀티 플랫폼 모니터링
    - 멘션/해시태그 추적
    - 모니터링 결과 품질 검증
    - 실시간 알림
    """

    def __init__(self, brand_config: Dict[str, Any]):
        super().__init__(brand_config)

        # 설정 로드
        config_dict = brand_config.get('social_monitoring', {})
        self.handler_config = SocialMonitoringConfig(
            quality_filter_enabled=config_dict.get('quality_filter_enabled', True),
            trust_filter_enabled=config_dict.get('trust_filter_enabled', True),
            relevance_filter_enabled=config_dict.get('relevance_filter_enabled', True),
            validation_filter_enabled=config_dict.get('validation_filter_enabled', True),
            min_quality_score=config_dict.get('min_quality_score', 0.5),
            min_trust_score=config_dict.get('min_trust_score', 0.5),
            min_relevance_score=config_dict.get('min_relevance_score', 0.5),
            validation_pass_threshold=config_dict.get('validation_pass_threshold', 0.7),
            validation_warning_threshold=config_dict.get('validation_warning_threshold', 0.5),
            platforms=config_dict.get('platforms', ['instagram', 'twitter', 'youtube', 'tiktok']),
            track_mentions=config_dict.get('track_mentions', True),
            track_hashtags=config_dict.get('track_hashtags', True),
            max_results=config_dict.get('max_results', 20),
            sentiment_analysis_enabled=config_dict.get('sentiment_analysis_enabled', True),
            sentiment_analysis_mode=config_dict.get('sentiment_analysis_mode', 'auto'),
            log_filter_results=config_dict.get('log_filter_results', True),
            log_monitoring=config_dict.get('log_monitoring', True),
            include_quality_details=config_dict.get('include_quality_details', True),
            include_trust_details=config_dict.get('include_trust_details', True),
            include_relevance_details=config_dict.get('include_relevance_details', True),
            include_validation_details=config_dict.get('include_validation_details', True),
            include_sentiment_details=config_dict.get('include_sentiment_details', True),
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

        # 감정 분석기 초기화
        mode_map = {
            'fast': AnalysisMode.FAST,
            'accurate': AnalysisMode.ACCURATE,
            'auto': AnalysisMode.AUTO,
        }
        self.sentiment_analyzer = SentimentAnalyzer(
            config=SentimentConfig(
                mode=mode_map.get(self.handler_config.sentiment_analysis_mode, AnalysisMode.AUTO),
                language="ko",
            )
        )

        logger.info(
            f"[SocialMonitoringHandler] Initialized for {self.brand_id} "
            f"(Quality={self.handler_config.quality_filter_enabled}, "
            f"Trust={self.handler_config.trust_filter_enabled}, "
            f"Relevance={self.handler_config.relevance_filter_enabled}, "
            f"Validation={self.handler_config.validation_filter_enabled}, "
            f"Sentiment={self.handler_config.sentiment_analysis_enabled})"
        )

    def _extract_feature_config(self) -> Dict[str, Any]:
        return self.brand_config.get('social_monitoring', {})

    def can_handle(self, question: str, context: Dict[str, Any]) -> bool:
        """소셜 모니터링 관련 질문인지 판단"""
        keywords = [
            '모니터링', '추적', '감시', '소셜', 'sns',
            '멘션', '해시태그', '언급', '반응', '댓글',
            '인스타', '트위터', '유튜브', '틱톡',
        ]
        question_lower = question.lower()

        return any(kw in question_lower for kw in keywords)

    def process(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        소셜 모니터링 처리 및 품질 검증

        Args:
            question: 사용자 질문
            context: 컨텍스트

        Returns:
            처리 결과 (response, metadata, filter_results)
        """
        start_time = datetime.now()

        # 메타데이터 초기화
        metadata = {
            'handled_by': 'social_monitoring',
            'handler_version': '2.0',
            'processed_at': start_time.isoformat(),
            'monitoring_type': self._detect_monitoring_type(question),
            'platforms': self.handler_config.platforms,
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
            # 모니터링 데이터 조회
            monitoring_data = self._fetch_monitoring_data(question)
            metadata['data_count'] = len(monitoring_data.get('items', []))

            # 응답 생성
            if monitoring_data.get('items'):
                response = self._format_monitoring_results(monitoring_data)

                if self.handler_config.log_monitoring:
                    logger.info(
                        f"[SocialMonitoringHandler] Found {len(monitoring_data['items'])} items "
                        f"for: {question[:50]}..."
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
                    'retrieval_results': monitoring_data,
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
                            f"[SocialMonitoringHandler] Validation failed: "
                            f"Grade={validation_result.get('grade')}, "
                            f"Score={validation_result.get('score'):.2f}"
                        )
                        metadata['validation_warning'] = True

            # 개선 제안 추가
            if self.handler_config.log_filter_results and all_suggestions:
                logger.info(
                    f"[SocialMonitoringHandler] Suggestions: {', '.join(all_suggestions[:3])}"
                )
                metadata['improvement_suggestions'] = all_suggestions

            # === Sentiment Analysis 적용 ===
            if self.handler_config.sentiment_analysis_enabled and monitoring_data.get('items'):
                sentiment_result = self._apply_sentiment_analysis(monitoring_data['items'])
                filter_results['sentiment'] = sentiment_result

                if sentiment_result and self.handler_config.include_sentiment_details:
                    metadata['sentiment'] = sentiment_result

                    if self.handler_config.log_filter_results:
                        logger.info(
                            f"[SocialMonitoringHandler:Sentiment] "
                            f"Dominant={sentiment_result.get('dominant')}, "
                            f"Positive={sentiment_result.get('distribution_percent', {}).get('positive', 0)}%, "
                            f"Negative={sentiment_result.get('distribution_percent', {}).get('negative', 0)}%"
                        )

            # 모니터링 요약 추가
            if monitoring_data.get('items'):
                metadata['monitoring_summary'] = {
                    'total_items': len(monitoring_data['items']),
                    'total_engagement': monitoring_data.get('total_engagement', 0),
                    'platforms_covered': monitoring_data.get('platforms_covered', []),
                    'sentiment_summary': metadata.get('sentiment', {}),
                }

        except Exception as e:
            logger.error(f"[SocialMonitoringHandler] Error: {e}")
            response = "모니터링 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
            metadata['error'] = str(e)

        # 처리 시간 기록
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        metadata['processing_time_ms'] = round(processing_time, 2)

        return {
            'response': response,
            'metadata': metadata,
            'filter_results': filter_results,
        }

    def _detect_monitoring_type(self, question: str) -> str:
        """모니터링 타입 감지"""
        question_lower = question.lower()

        if any(kw in question_lower for kw in ['멘션', '언급', '@']):
            return 'mentions'
        elif any(kw in question_lower for kw in ['해시태그', '#', '태그']):
            return 'hashtags'
        elif any(kw in question_lower for kw in ['댓글', '반응', '피드백']):
            return 'engagement'
        else:
            return 'general'

    def _fetch_monitoring_data(self, question: str) -> Dict[str, Any]:
        """모니터링 데이터 조회 - Content와 Interaction 노드 사용"""
        try:
            from app.services.shared.neo4j import get_neo4j_client

            neo4j = get_neo4j_client()

            # Content와 Interaction 데이터 조회
            query = """
            MATCH (c:Content)
            WHERE c.brand_id = $brand_id
            OPTIONAL MATCH (c)<-[:BELONGS_TO]-(i:Interaction)
            WITH c,
                 count(CASE WHEN i.type = 'like' THEN 1 END) as likes,
                 count(CASE WHEN i.type = 'comment' THEN 1 END) as comments,
                 collect(CASE WHEN i.type = 'comment' THEN {text: i.text, sentiment: i.sentiment} END)[0..5] as recent_comments
            RETURN c.id as id,
                   c.platform as platform,
                   c.caption as content,
                   c.content_type as content_type,
                   likes,
                   comments,
                   c.hashtags as hashtags,
                   recent_comments,
                   c.posted_at as posted_at
            ORDER BY c.posted_at DESC
            LIMIT $limit
            """

            items = neo4j.query(query, {
                'brand_id': self.brand_id,
                'limit': self.handler_config.max_results,
            }) or []

            # Interaction 감정 분석 요약
            sentiment_query = """
            MATCH (c:Content {brand_id: $brand_id})<-[:BELONGS_TO]-(i:Interaction)
            WHERE i.type = 'comment' AND i.sentiment IS NOT NULL
            RETURN i.sentiment as sentiment, count(*) as count
            """
            sentiment_data = neo4j.query(sentiment_query, {'brand_id': self.brand_id}) or []
            sentiment_summary = {row['sentiment']: row['count'] for row in sentiment_data}

            total_engagement = sum(
                (item.get('likes', 0) or 0) + (item.get('comments', 0) or 0)
                for item in items
            )
            platforms_covered = list(set(
                item.get('platform', 'unknown') for item in items if item.get('platform')
            ))

            return {
                'items': items,
                'total_engagement': total_engagement,
                'platforms_covered': platforms_covered,
                'sentiment_summary': sentiment_summary,
            }

        except Exception as e:
            logger.error(f"[SocialMonitoringHandler] Fetch error: {e}")
            return {'items': [], 'total_engagement': 0, 'platforms_covered': [], 'sentiment_summary': {}}

    def _format_monitoring_results(self, data: Dict[str, Any]) -> str:
        """모니터링 결과 포맷팅"""
        items = data.get('items', [])
        total_engagement = data.get('total_engagement', 0)
        platforms = data.get('platforms_covered', [])
        sentiment_summary = data.get('sentiment_summary', {})

        lines = ["📡 **소셜 모니터링 리포트**\n"]

        lines.append("**📊 요약**")
        lines.append(f"- 콘텐츠 수: {len(items)}개")
        lines.append(f"- 총 인게이지먼트: {total_engagement:,}")
        lines.append(f"- 플랫폼: {', '.join(platforms) if platforms else 'N/A'}")

        # 감정 분석 요약
        if sentiment_summary:
            positive = sentiment_summary.get('positive', 0)
            neutral = sentiment_summary.get('neutral', 0)
            negative = sentiment_summary.get('negative', 0)
            total_sentiment = positive + neutral + negative
            if total_sentiment > 0:
                lines.append(f"- 댓글 감정: 😊 {positive}개 ({positive*100//total_sentiment}%) | 😐 {neutral}개 | 😟 {negative}개")
        lines.append("")

        lines.append("**📝 최근 콘텐츠**")
        for i, item in enumerate(items[:5], 1):
            platform = item.get('platform', 'unknown')
            content = (item.get('content', '') or '')[:50]
            likes = item.get('likes', 0) or 0
            comments = item.get('comments', 0) or 0
            content_type = item.get('content_type', '')

            platform_emoji = {
                'instagram': '📸',
                'twitter': '🐦',
                'youtube': '📺',
                'tiktok': '🎵',
            }.get(platform, '📱')

            type_tag = f"[{content_type}] " if content_type else ""
            lines.append(f"{i}. {platform_emoji} {type_tag}{content}...")
            lines.append(f"   ❤️ {likes:,} | 💬 {comments:,}")

            # 최근 댓글 표시
            recent_comments = item.get('recent_comments', [])
            if recent_comments:
                for comment in recent_comments[:2]:
                    if comment and comment.get('text'):
                        sentiment_emoji = {'positive': '😊', 'neutral': '😐', 'negative': '😟'}.get(comment.get('sentiment', ''), '💬')
                        comment_text = (comment.get('text', '') or '')[:30]
                        lines.append(f"      {sentiment_emoji} \"{comment_text}...\"")

        lines.append("")
        lines.append("더 자세한 분석이 필요하시면 말씀해주세요!")

        return "\n".join(lines)

    def _generate_no_data_response(self) -> str:
        """데이터 없을 때 응답"""
        return (
            "현재 모니터링할 소셜 데이터가 없습니다.\n\n"
            "다음 기능을 설정하시면 모니터링을 시작할 수 있습니다:\n"
            "- SNS 계정 연동\n"
            "- 키워드/해시태그 설정\n"
            "- 알림 설정"
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
                    f"[SocialMonitoringHandler:Quality] "
                    f"Score={result.score:.2f}, Level={result.level.value}"
                )

            return {
                'score': result.score,
                'level': result.level.value,
                'valid': result.valid,
                'suggestions': result.suggestions,
            }

        except Exception as e:
            logger.error(f"[SocialMonitoringHandler:Quality] Error: {e}")
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
                    f"[SocialMonitoringHandler:Trust] "
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
            logger.error(f"[SocialMonitoringHandler:Trust] Error: {e}")
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
            relevance_context = {
                **context,
                'question': question,
            }

            result: RelevanceResult = self.relevance_filter.validate(response, relevance_context)

            if self.handler_config.log_filter_results:
                logger.info(
                    f"[SocialMonitoringHandler:Relevance] "
                    f"Score={result.score:.2f}, "
                    f"Level={result.level.value}, "
                    f"ResponseType={result.response_type.value}, "
                    f"Valid={result.valid}"
                )

                if result.issues:
                    for issue in result.issues[:3]:
                        logger.warning(
                            f"[SocialMonitoringHandler:Relevance] Issue: "
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
            logger.error(f"[SocialMonitoringHandler:Relevance] Error: {e}")
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
            validation_context = {
                **context,
                'question': question,
            }

            result: ValidationResult = self.validation_filter.validate(response, validation_context)

            if self.handler_config.log_filter_results:
                logger.info(
                    f"[SocialMonitoringHandler:Validation] "
                    f"Score={result.score:.2f}, "
                    f"Grade={result.grade.value}, "
                    f"Status={result.status.value}, "
                    f"Valid={result.valid}"
                )

                if result.all_issues:
                    for issue in result.all_issues[:3]:
                        logger.warning(f"[SocialMonitoringHandler:Validation] Issue: {issue}")

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
            logger.error(f"[SocialMonitoringHandler:Validation] Error: {e}")
            return None

    def _apply_sentiment_analysis(
        self,
        items: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        감정 분석 적용

        Args:
            items: 모니터링 항목 리스트

        Returns:
            감정 분석 결과 요약
        """
        try:
            # 텍스트 추출
            texts = [
                item.get('content', '') or item.get('text', '')
                for item in items
                if item.get('content') or item.get('text')
            ]

            if not texts:
                return None

            # 배치 분석
            results: List[SentimentResult] = self.sentiment_analyzer.analyze_batch(texts)

            # 요약 생성
            summary = self.sentiment_analyzer.get_summary(results)

            # 개별 결과도 포함
            individual_results = [
                {
                    'label': r.label.value,
                    'confidence': round(r.confidence, 3),
                    'keywords': r.keywords_found[:5],  # 상위 5개 키워드
                }
                for r in results[:10]  # 상위 10개만
            ]

            return {
                **summary,
                'individual_results': individual_results,
                'dominant': summary.get('dominant_sentiment'),
            }

        except Exception as e:
            logger.error(f"[SocialMonitoringHandler:Sentiment] Error: {e}")
            return None

    def analyze_content_sentiment(self, content: str) -> Dict[str, Any]:
        """
        단일 콘텐츠 감정 분석 (외부 호출용)

        Args:
            content: 분석할 텍스트

        Returns:
            감정 분석 결과
        """
        try:
            result = self.sentiment_analyzer.analyze(content)
            return result.to_dict()
        except Exception as e:
            logger.error(f"[SocialMonitoringHandler:Sentiment] Single analysis error: {e}")
            return {'label': 'neutral', 'confidence': 0, 'error': str(e)}

    def get_filter_stats(self) -> Dict[str, Any]:
        """필터 통계 반환"""
        return {
            'quality_filter_enabled': self.handler_config.quality_filter_enabled,
            'trust_filter_enabled': self.handler_config.trust_filter_enabled,
            'relevance_filter_enabled': self.handler_config.relevance_filter_enabled,
            'validation_filter_enabled': self.handler_config.validation_filter_enabled,
            'sentiment_analysis_enabled': self.handler_config.sentiment_analysis_enabled,
            'min_quality_score': self.handler_config.min_quality_score,
            'min_trust_score': self.handler_config.min_trust_score,
            'min_relevance_score': self.handler_config.min_relevance_score,
            'validation_pass_threshold': self.handler_config.validation_pass_threshold,
            'sentiment_analysis_mode': self.handler_config.sentiment_analysis_mode,
            'platforms': self.handler_config.platforms,
            'max_results': self.handler_config.max_results,
        }
