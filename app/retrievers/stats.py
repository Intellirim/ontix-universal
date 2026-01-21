"""
Stats Retriever - Production Grade v2.0
통계 및 분석 데이터 검색

Features:
    - 다양한 통계 유형 (인기, 트렌딩, 성장률 등)
    - 시간 기반 분석 (일별, 주별, 월별)
    - 플랫폼별 비교 분석
    - 인게이지먼트 메트릭 계산
    - 토픽/컨셉 트렌드 분석
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging

from app.interfaces.retriever import RetrieverInterface, RetrievalResult, RetrievalItem, RetrievalSource
from app.core.context import QueryContext
from app.services.shared.neo4j import get_neo4j_client

logger = logging.getLogger(__name__)


class StatsType(str, Enum):
    """통계 유형"""
    POPULAR = "popular"  # 인기 콘텐츠
    TRENDING = "trending"  # 급상승 트렌드
    ENGAGEMENT = "engagement"  # 인게이지먼트 분석
    PLATFORM = "platform"  # 플랫폼별 분석
    TOPIC = "topic"  # 토픽/컨셉 분석
    ACTOR = "actor"  # 크리에이터/작성자 분석
    GROWTH = "growth"  # 성장률 분석
    OVERVIEW = "overview"  # 전체 개요


class TimePeriod(str, Enum):
    """시간 범위"""
    TODAY = "today"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    ALL_TIME = "all_time"


class SortMetric(str, Enum):
    """정렬 기준"""
    LIKES = "likes"
    VIEWS = "views"
    COMMENTS = "comments"
    ENGAGEMENT_RATE = "engagement_rate"
    SHARES = "shares"
    RECENT = "recent"


@dataclass
class StatsConfig:
    """통계 검색 설정"""
    stats_type: StatsType = StatsType.OVERVIEW
    period: TimePeriod = TimePeriod.MONTH
    sort_by: SortMetric = SortMetric.LIKES
    limit: int = 20
    platforms: Optional[List[str]] = None
    min_engagement: int = 0


@dataclass
class StatsResult:
    """통계 결과"""
    stats_type: str
    period: str
    data: Dict[str, Any]
    summary: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class MetricsCalculator:
    """메트릭 계산기"""

    @staticmethod
    def engagement_rate(likes: int, comments: int, views: int) -> float:
        """인게이지먼트 비율 계산"""
        if views <= 0:
            return 0.0
        engagement = (likes + comments * 2) / views * 100
        return round(min(engagement, 100.0), 2)

    @staticmethod
    def growth_rate(current: float, previous: float) -> float:
        """성장률 계산"""
        if previous <= 0:
            return 100.0 if current > 0 else 0.0
        return round((current - previous) / previous * 100, 2)

    @staticmethod
    def trend_score(recent_engagement: float, avg_engagement: float) -> float:
        """트렌드 점수 계산 (1.0 = 평균)"""
        if avg_engagement <= 0:
            return 1.0
        return round(recent_engagement / avg_engagement, 2)


class TimeHelper:
    """시간 관련 헬퍼"""

    PERIOD_DAYS = {
        TimePeriod.TODAY: 1,
        TimePeriod.WEEK: 7,
        TimePeriod.MONTH: 30,
        TimePeriod.QUARTER: 90,
        TimePeriod.YEAR: 365,
        TimePeriod.ALL_TIME: None,
    }

    @classmethod
    def get_days(cls, period: TimePeriod) -> Optional[int]:
        """기간을 일수로 변환"""
        return cls.PERIOD_DAYS.get(period)

    @classmethod
    def get_date_filter(cls, period: TimePeriod) -> str:
        """Cypher 날짜 필터 생성"""
        days = cls.get_days(period)
        if days is None:
            return ""
        return f"AND c.created_at >= datetime() - duration({{days: {days}}})"


class StatsRetriever(RetrieverInterface):
    """
    프로덕션급 통계 검색기

    SNS 데이터의 다양한 통계와 분석 정보를 제공합니다.
    인기 콘텐츠, 트렌드, 인게이지먼트 분석 등을 지원합니다.
    """

    def __init__(self, brand_config):
        super().__init__(brand_config, RetrievalSource.STATS)
        self.neo4j = get_neo4j_client()
        self.stats_config = StatsConfig()

    def _do_retrieve(self, context: QueryContext) -> RetrievalResult:
        """
        통계 검색 실행 (추상 메서드 구현)

        Args:
            context: 쿼리 컨텍스트

        Returns:
            검색 결과
        """
        try:
            # 쿼리에서 통계 유형 감지
            detected_type = self._detect_stats_type(context.question)
            if detected_type:
                self.stats_config.stats_type = detected_type

            # 통계 유형별 처리
            result = self._get_stats()

            if result:
                # StatsResult를 RetrievalItem으로 변환
                items = [
                    RetrievalItem(
                        id=f"stats_{result.stats_type}",
                        content=str(result.data),
                        score=1.0,
                        source='stats_analysis',
                        node_type='Stats',
                        metadata={
                            'stats_type': result.stats_type,
                            'period': result.period,
                            'summary': result.summary,
                            'generated_at': result.generated_at,
                            'raw_data': result.data,
                        }
                    )
                ]

                logger.info(
                    f"Stats retrieval: type={self.stats_config.stats_type.value}, "
                    f"period={self.stats_config.period.value}"
                )

                return RetrievalResult(
                    source='stats_analysis',
                    items=items,
                    metadata={
                        'stats_type': result.stats_type,
                        'period': result.period,
                        'summary': result.summary,
                    }
                )

            return RetrievalResult(source='stats_analysis', items=[])

        except Exception as e:
            logger.error(f"Stats retrieval error: {e}", exc_info=True)
            return RetrievalResult(source='stats_analysis', items=[], error=str(e))

    def _detect_stats_type(self, question: str) -> Optional[StatsType]:
        """쿼리에서 통계 유형 감지"""
        question_lower = question.lower()

        patterns = {
            StatsType.POPULAR: ['인기', 'popular', '많이', '좋아요', 'top', 'best'],
            StatsType.TRENDING: ['트렌딩', 'trending', '급상승', '떠오르는', 'hot'],
            StatsType.ENGAGEMENT: ['인게이지먼트', 'engagement', '참여', '반응'],
            StatsType.PLATFORM: ['플랫폼', 'platform', '인스타', '유튜브', '틱톡'],
            StatsType.TOPIC: ['토픽', 'topic', '주제', '키워드', '해시태그'],
            StatsType.ACTOR: ['크리에이터', 'creator', '인플루언서', '작성자'],
            StatsType.GROWTH: ['성장', 'growth', '증가', '변화'],
        }

        for stats_type, keywords in patterns.items():
            if any(kw in question_lower for kw in keywords):
                return stats_type

        return None

    def _get_stats(self) -> Optional[StatsResult]:
        """통계 유형별 데이터 조회"""
        if self.stats_config.stats_type == StatsType.POPULAR:
            return self._get_popular_content()
        elif self.stats_config.stats_type == StatsType.TRENDING:
            return self._get_trending_content()
        elif self.stats_config.stats_type == StatsType.ENGAGEMENT:
            return self._get_engagement_stats()
        elif self.stats_config.stats_type == StatsType.PLATFORM:
            return self._get_platform_stats()
        elif self.stats_config.stats_type == StatsType.TOPIC:
            return self._get_topic_stats()
        elif self.stats_config.stats_type == StatsType.ACTOR:
            return self._get_actor_stats()
        elif self.stats_config.stats_type == StatsType.GROWTH:
            return self._get_growth_stats()
        else:
            return self._get_overview_stats()

    def _get_popular_content(self) -> StatsResult:
        """인기 콘텐츠 조회"""
        date_filter = TimeHelper.get_date_filter(self.stats_config.period)
        platform_filter = ""
        if self.stats_config.platforms:
            platform_filter = "AND c.platform IN $platforms"

        # 정렬 필드 결정
        sort_field = {
            SortMetric.LIKES: "c.like_count",
            SortMetric.VIEWS: "c.view_count",
            SortMetric.COMMENTS: "c.comment_count",
            SortMetric.RECENT: "c.created_at",
        }.get(self.stats_config.sort_by, "c.like_count")

        query = f"""
        MATCH (c:Content)
        WHERE c.brand_id = $brand_id
          {date_filter}
          {platform_filter}
        RETURN c.id as id,
               c.text as text,
               c.url as url,
               c.platform as platform,
               c.like_count as likes,
               c.view_count as views,
               c.comment_count as comments,
               c.share_count as shares,
               c.created_at as created_at
        ORDER BY {sort_field} DESC
        LIMIT $limit
        """

        params = {
            'brand_id': self.brand_id,
            'limit': self.stats_config.limit,
        }
        if self.stats_config.platforms:
            params['platforms'] = self.stats_config.platforms

        results = self.neo4j.query(query, params)

        # 요약 통계 계산
        total_likes = sum(r.get('likes', 0) or 0 for r in results)
        total_views = sum(r.get('views', 0) or 0 for r in results)
        total_comments = sum(r.get('comments', 0) or 0 for r in results)

        return StatsResult(
            stats_type='popular',
            period=self.stats_config.period.value,
            data={
                'contents': [
                    {
                        'id': r['id'],
                        'text': (r.get('text') or '')[:200],
                        'url': r.get('url'),
                        'platform': r.get('platform'),
                        'likes': r.get('likes', 0),
                        'views': r.get('views', 0),
                        'comments': r.get('comments', 0),
                        'engagement_rate': MetricsCalculator.engagement_rate(
                            r.get('likes', 0) or 0,
                            r.get('comments', 0) or 0,
                            r.get('views', 0) or 1
                        ),
                    }
                    for r in results
                ],
            },
            summary={
                'total_contents': len(results),
                'total_likes': total_likes,
                'total_views': total_views,
                'total_comments': total_comments,
                'avg_likes': round(total_likes / max(len(results), 1), 1),
                'avg_engagement_rate': MetricsCalculator.engagement_rate(
                    total_likes, total_comments, total_views
                ) if total_views > 0 else 0,
            }
        )

    def _get_trending_content(self) -> StatsResult:
        """트렌딩 콘텐츠 조회 (최근 vs 평균 비교)"""
        # 최근 7일 콘텐츠
        recent_query = """
        MATCH (c:Content)
        WHERE c.brand_id = $brand_id
          AND c.created_at >= datetime() - duration({days: 7})
        RETURN c.id as id,
               c.text as text,
               c.platform as platform,
               c.like_count as likes,
               c.view_count as views,
               c.comment_count as comments,
               c.created_at as created_at
        ORDER BY c.like_count DESC
        LIMIT $limit
        """

        # 전체 평균
        avg_query = """
        MATCH (c:Content)
        WHERE c.brand_id = $brand_id
        RETURN avg(c.like_count) as avg_likes,
               avg(c.view_count) as avg_views,
               avg(c.comment_count) as avg_comments
        """

        recent_results = self.neo4j.query(recent_query, {
            'brand_id': self.brand_id,
            'limit': self.stats_config.limit,
        })

        avg_results = self.neo4j.query(avg_query, {'brand_id': self.brand_id})
        avg_stats = avg_results[0] if avg_results else {}

        avg_likes = avg_stats.get('avg_likes', 0) or 0
        avg_views = avg_stats.get('avg_views', 0) or 0

        trending_contents = []
        for r in recent_results:
            likes = r.get('likes', 0) or 0
            views = r.get('views', 0) or 0
            trend_score = MetricsCalculator.trend_score(likes, avg_likes)

            if trend_score >= 1.2:  # 평균보다 20% 이상
                trending_contents.append({
                    'id': r['id'],
                    'text': (r.get('text') or '')[:200],
                    'platform': r.get('platform'),
                    'likes': likes,
                    'views': views,
                    'trend_score': trend_score,
                    'status': 'hot' if trend_score >= 2.0 else 'rising',
                })

        return StatsResult(
            stats_type='trending',
            period='week',
            data={'trending_contents': trending_contents},
            summary={
                'total_trending': len(trending_contents),
                'hot_count': len([c for c in trending_contents if c['status'] == 'hot']),
                'avg_baseline_likes': round(avg_likes, 1),
            }
        )

    def _get_engagement_stats(self) -> StatsResult:
        """인게이지먼트 분석"""
        date_filter = TimeHelper.get_date_filter(self.stats_config.period)

        query = f"""
        MATCH (c:Content)
        WHERE c.brand_id = $brand_id
          {date_filter}
        RETURN c.platform as platform,
               count(c) as content_count,
               sum(c.like_count) as total_likes,
               sum(c.view_count) as total_views,
               sum(c.comment_count) as total_comments,
               avg(c.like_count) as avg_likes,
               avg(c.view_count) as avg_views,
               avg(c.comment_count) as avg_comments
        """

        results = self.neo4j.query(query, {'brand_id': self.brand_id})

        platform_stats = []
        for r in results:
            platform = r.get('platform', 'unknown')
            total_likes = r.get('total_likes', 0) or 0
            total_views = r.get('total_views', 0) or 1
            total_comments = r.get('total_comments', 0) or 0

            platform_stats.append({
                'platform': platform,
                'content_count': r.get('content_count', 0),
                'total_likes': total_likes,
                'total_views': total_views,
                'total_comments': total_comments,
                'avg_likes': round(r.get('avg_likes', 0) or 0, 1),
                'avg_views': round(r.get('avg_views', 0) or 0, 1),
                'engagement_rate': MetricsCalculator.engagement_rate(
                    total_likes, total_comments, total_views
                ),
            })

        # 전체 요약
        total_likes = sum(p['total_likes'] for p in platform_stats)
        total_views = sum(p['total_views'] for p in platform_stats)
        total_comments = sum(p['total_comments'] for p in platform_stats)

        return StatsResult(
            stats_type='engagement',
            period=self.stats_config.period.value,
            data={'platform_engagement': platform_stats},
            summary={
                'total_likes': total_likes,
                'total_views': total_views,
                'total_comments': total_comments,
                'overall_engagement_rate': MetricsCalculator.engagement_rate(
                    total_likes, total_comments, total_views
                ),
                'best_platform': max(platform_stats, key=lambda x: x['engagement_rate'])['platform'] if platform_stats else None,
            }
        )

    def _get_platform_stats(self) -> StatsResult:
        """플랫폼별 분석"""
        query = """
        MATCH (c:Content)
        WHERE c.brand_id = $brand_id
        RETURN c.platform as platform,
               count(c) as content_count,
               sum(c.like_count) as total_likes,
               sum(c.view_count) as total_views,
               sum(c.comment_count) as total_comments,
               max(c.like_count) as max_likes,
               min(c.created_at) as first_content,
               max(c.created_at) as last_content
        ORDER BY total_likes DESC
        """

        results = self.neo4j.query(query, {'brand_id': self.brand_id})

        total_content = sum(r.get('content_count', 0) for r in results)

        platform_data = []
        for r in results:
            content_count = r.get('content_count', 0)
            platform_data.append({
                'platform': r.get('platform'),
                'content_count': content_count,
                'content_share': round(content_count / max(total_content, 1) * 100, 1),
                'total_likes': r.get('total_likes', 0),
                'total_views': r.get('total_views', 0),
                'total_comments': r.get('total_comments', 0),
                'max_likes': r.get('max_likes', 0),
                'first_content': str(r.get('first_content'))[:10] if r.get('first_content') else None,
                'last_content': str(r.get('last_content'))[:10] if r.get('last_content') else None,
            })

        return StatsResult(
            stats_type='platform',
            period='all_time',
            data={'platforms': platform_data},
            summary={
                'total_platforms': len(platform_data),
                'total_content': total_content,
                'dominant_platform': platform_data[0]['platform'] if platform_data else None,
            }
        )

    def _get_topic_stats(self) -> StatsResult:
        """토픽/컨셉 분석"""
        # Concept 노드 기반 분석
        query = """
        MATCH (c:Concept)<-[:MENTIONS_CONCEPT]-(content:Content)
        WHERE c.brand_id = $brand_id
        RETURN c.id as concept,
               c.type as type,
               count(content) as mention_count,
               sum(content.like_count) as total_likes,
               avg(content.like_count) as avg_likes
        ORDER BY mention_count DESC
        LIMIT $limit
        """

        results = self.neo4j.query(query, {
            'brand_id': self.brand_id,
            'limit': self.stats_config.limit,
        })

        topic_data = [
            {
                'concept': r['concept'],
                'type': r.get('type'),
                'mention_count': r.get('mention_count', 0),
                'total_likes': r.get('total_likes', 0),
                'avg_likes': round(r.get('avg_likes', 0) or 0, 1),
            }
            for r in results
        ]

        return StatsResult(
            stats_type='topic',
            period='all_time',
            data={'topics': topic_data},
            summary={
                'total_topics': len(topic_data),
                'top_topic': topic_data[0]['concept'] if topic_data else None,
                'total_mentions': sum(t['mention_count'] for t in topic_data),
            }
        )

    def _get_actor_stats(self) -> StatsResult:
        """크리에이터/작성자 분석"""
        query = """
        MATCH (a:Actor)-[:CREATED]->(c:Content)
        WHERE c.brand_id = $brand_id
        RETURN a.id as actor_id,
               a.username as username,
               a.platform as platform,
               count(c) as content_count,
               sum(c.like_count) as total_likes,
               avg(c.like_count) as avg_likes,
               a.follower_count as followers
        ORDER BY total_likes DESC
        LIMIT $limit
        """

        results = self.neo4j.query(query, {
            'brand_id': self.brand_id,
            'limit': self.stats_config.limit,
        })

        actor_data = [
            {
                'actor_id': r['actor_id'],
                'username': r.get('username'),
                'platform': r.get('platform'),
                'content_count': r.get('content_count', 0),
                'total_likes': r.get('total_likes', 0),
                'avg_likes': round(r.get('avg_likes', 0) or 0, 1),
                'followers': r.get('followers', 0),
            }
            for r in results
        ]

        return StatsResult(
            stats_type='actor',
            period='all_time',
            data={'actors': actor_data},
            summary={
                'total_actors': len(actor_data),
                'top_actor': actor_data[0]['username'] if actor_data else None,
                'total_content': sum(a['content_count'] for a in actor_data),
            }
        )

    def _get_growth_stats(self) -> StatsResult:
        """성장률 분석 (이번 주 vs 지난 주)"""
        # 이번 주
        current_query = """
        MATCH (c:Content)
        WHERE c.brand_id = $brand_id
          AND c.created_at >= datetime() - duration({days: 7})
        RETURN count(c) as content_count,
               sum(c.like_count) as total_likes,
               sum(c.view_count) as total_views
        """

        # 지난 주
        previous_query = """
        MATCH (c:Content)
        WHERE c.brand_id = $brand_id
          AND c.created_at >= datetime() - duration({days: 14})
          AND c.created_at < datetime() - duration({days: 7})
        RETURN count(c) as content_count,
               sum(c.like_count) as total_likes,
               sum(c.view_count) as total_views
        """

        current = self.neo4j.query(current_query, {'brand_id': self.brand_id})
        previous = self.neo4j.query(previous_query, {'brand_id': self.brand_id})

        current_stats = current[0] if current else {}
        previous_stats = previous[0] if previous else {}

        current_content = current_stats.get('content_count', 0) or 0
        previous_content = previous_stats.get('content_count', 0) or 0
        current_likes = current_stats.get('total_likes', 0) or 0
        previous_likes = previous_stats.get('total_likes', 0) or 0
        current_views = current_stats.get('total_views', 0) or 0
        previous_views = previous_stats.get('total_views', 0) or 0

        return StatsResult(
            stats_type='growth',
            period='week',
            data={
                'current_week': {
                    'content_count': current_content,
                    'total_likes': current_likes,
                    'total_views': current_views,
                },
                'previous_week': {
                    'content_count': previous_content,
                    'total_likes': previous_likes,
                    'total_views': previous_views,
                },
                'growth_rates': {
                    'content_growth': MetricsCalculator.growth_rate(current_content, previous_content),
                    'likes_growth': MetricsCalculator.growth_rate(current_likes, previous_likes),
                    'views_growth': MetricsCalculator.growth_rate(current_views, previous_views),
                },
            },
            summary={
                'overall_growth': MetricsCalculator.growth_rate(
                    current_likes + current_views,
                    previous_likes + previous_views
                ),
                'trend': 'up' if current_likes > previous_likes else 'down',
            }
        )

    def _get_overview_stats(self) -> StatsResult:
        """전체 개요 통계"""
        query = """
        MATCH (c:Content)
        WHERE c.brand_id = $brand_id
        RETURN count(c) as total_content,
               sum(c.like_count) as total_likes,
               sum(c.view_count) as total_views,
               sum(c.comment_count) as total_comments,
               count(DISTINCT c.platform) as platform_count,
               min(c.created_at) as first_content,
               max(c.created_at) as last_content
        """

        results = self.neo4j.query(query, {'brand_id': self.brand_id})
        stats = results[0] if results else {}

        # Concept 수
        concept_query = """
        MATCH (c:Concept)
        WHERE c.brand_id = $brand_id
        RETURN count(c) as total_concepts
        """
        concept_results = self.neo4j.query(concept_query, {'brand_id': self.brand_id})
        concept_count = concept_results[0].get('total_concepts', 0) if concept_results else 0

        total_likes = stats.get('total_likes', 0) or 0
        total_views = stats.get('total_views', 0) or 1
        total_comments = stats.get('total_comments', 0) or 0

        return StatsResult(
            stats_type='overview',
            period='all_time',
            data={
                'total_content': stats.get('total_content', 0),
                'total_likes': total_likes,
                'total_views': total_views,
                'total_comments': total_comments,
                'total_concepts': concept_count,
                'platform_count': stats.get('platform_count', 0),
                'date_range': {
                    'first': str(stats.get('first_content'))[:10] if stats.get('first_content') else None,
                    'last': str(stats.get('last_content'))[:10] if stats.get('last_content') else None,
                },
            },
            summary={
                'overall_engagement_rate': MetricsCalculator.engagement_rate(
                    total_likes, total_comments, total_views
                ),
                'avg_likes_per_content': round(total_likes / max(stats.get('total_content', 1), 1), 1),
            }
        )

    def configure(
        self,
        stats_type: StatsType = None,
        period: TimePeriod = None,
        sort_by: SortMetric = None,
        limit: int = None,
        platforms: List[str] = None,
    ) -> 'StatsRetriever':
        """
        설정 변경 (체이닝 지원)

        Usage:
            retriever.configure(stats_type=StatsType.POPULAR, period=TimePeriod.WEEK).retrieve(context)
        """
        if stats_type is not None:
            self.stats_config.stats_type = stats_type
        if period is not None:
            self.stats_config.period = period
        if sort_by is not None:
            self.stats_config.sort_by = sort_by
        if limit is not None:
            self.stats_config.limit = limit
        if platforms is not None:
            self.stats_config.platforms = platforms
        return self
