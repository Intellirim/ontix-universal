"""
Social Monitoring API
소셜 모니터링 데이터 엔드포인트
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/social")


# ============================================================
# Response Models
# ============================================================

class SentimentDistribution(BaseModel):
    """감정 분포"""
    positive: int = Field(0, description="긍정 비율 (%)")
    neutral: int = Field(0, description="중립 비율 (%)")
    negative: int = Field(0, description="부정 비율 (%)")


class SocialMention(BaseModel):
    """소셜 멘션"""
    id: Optional[str] = None
    platform: str
    content: str
    sentiment: str = "neutral"
    timestamp: str
    likes: int = 0
    comments: int = 0
    author: Optional[str] = None


class TrendingTopic(BaseModel):
    """트렌딩 토픽"""
    topic: str
    count: int
    change: Optional[float] = None


class SocialMonitoringResponse(BaseModel):
    """소셜 모니터링 응답"""
    brand_id: str
    mentions: int
    sentiment: SentimentDistribution
    platforms: Dict[str, int]
    recent_mentions: List[SocialMention]
    trending_topics: List[TrendingTopic]
    total_engagement: int = 0
    period_days: int = 7


# ============================================================
# API Endpoints
# ============================================================

@router.get("/{brand_id}", response_model=SocialMonitoringResponse)
async def get_social_monitoring(
    brand_id: str,
    days: int = Query(7, ge=1, le=90, description="조회 기간 (일)")
):
    """
    소셜 모니터링 데이터 조회

    Neo4j에서 실제 수집된 소셜 미디어 데이터를 조회합니다.

    Args:
        brand_id: 브랜드 ID
        days: 조회 기간

    Returns:
        소셜 모니터링 데이터
    """
    try:
        from app.services.shared.neo4j import get_neo4j_client

        neo4j = get_neo4j_client()

        # 최근 포스트 조회 (Post 또는 Content 노드)
        posts_query = """
        MATCH (p)
        WHERE (p:Post OR p:Content)
          AND p.brand_id = $brand_id
        RETURN
            p.id as id,
            p.platform as platform,
            coalesce(p.content, p.text) as content,
            coalesce(p.sentiment, 'neutral') as sentiment,
            coalesce(p.like_count, p.likes, 0) as likes,
            coalesce(p.comment_count, p.comments, 0) as comments,
            coalesce(p.share_count, p.shares, 0) as shares,
            coalesce(p.view_count, 0) as views,
            p.author as author,
            p.created_at as created_at,
            p.metrics as metrics
        ORDER BY p.created_at DESC
        LIMIT 100
        """

        posts = neo4j.query(posts_query, {
            'brand_id': brand_id,
            'days': days
        }) or []

        # Interaction 노드에서 감성 데이터 조회 (댓글 반응)
        interactions_query = """
        MATCH (i:Interaction {brand_id: $brand_id})
        WHERE i.sentiment IS NOT NULL
        RETURN i.sentiment as sentiment, count(*) as count
        """
        interactions = neo4j.query(interactions_query, {'brand_id': brand_id}) or []
        interaction_sentiments = {row['sentiment']: row['count'] for row in interactions}

        # 데이터가 없으면 기본값 반환
        if not posts:
            return SocialMonitoringResponse(
                brand_id=brand_id,
                mentions=0,
                sentiment=SentimentDistribution(positive=0, neutral=100, negative=0),
                platforms={},
                recent_mentions=[],
                trending_topics=[],
                total_engagement=0,
                period_days=days
            )

        # 플랫폼별 분포 계산
        platform_counts: Dict[str, int] = {}
        for post in posts:
            platform = post.get('platform', 'unknown')
            platform_counts[platform] = platform_counts.get(platform, 0) + 1

        # 비율로 변환
        total_posts = len(posts)
        platform_percentages = {
            k: round(v / total_posts * 100)
            for k, v in platform_counts.items()
        }

        # 감정 분석 집계 (Content + Interaction 모두 포함)
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}

        # Content 감성 집계
        for post in posts:
            sentiment = post.get('sentiment', 'neutral')
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1

        # Interaction(댓글) 감성 추가
        for sentiment_type in ['positive', 'neutral', 'negative']:
            sentiment_counts[sentiment_type] += interaction_sentiments.get(sentiment_type, 0)

        # 총 분석 대상 (Content + Interaction)
        total_analyzed = sum(sentiment_counts.values())

        # 비율로 변환
        sentiment_distribution = SentimentDistribution(
            positive=round(sentiment_counts['positive'] / total_analyzed * 100) if total_analyzed > 0 else 0,
            neutral=round(sentiment_counts['neutral'] / total_analyzed * 100) if total_analyzed > 0 else 100,
            negative=round(sentiment_counts['negative'] / total_analyzed * 100) if total_analyzed > 0 else 0
        )

        # 총 인게이지먼트 (좋아요 + 댓글 + 공유 + 조회수)
        total_engagement = sum(
            (post.get('likes', 0) or 0) +
            (post.get('comments', 0) or 0) +
            (post.get('shares', 0) or 0) +
            (post.get('views', 0) or 0)
            for post in posts
        )

        # 최근 멘션 (상위 20개)
        recent_mentions = [
            SocialMention(
                id=str(post.get('id', '')),
                platform=post.get('platform', 'unknown'),
                content=(post.get('content', '') or '')[:200],
                sentiment=post.get('sentiment', 'neutral'),
                timestamp=_format_timestamp(post.get('created_at')),
                likes=post.get('likes', 0) or 0,
                comments=post.get('comments', 0) or 0,
                author=post.get('author')
            )
            for post in posts[:20]
        ]

        # 트렌딩 토픽 (해시태그 집계)
        trending_topics = _extract_trending_topics(posts)

        return SocialMonitoringResponse(
            brand_id=brand_id,
            mentions=total_posts,
            sentiment=sentiment_distribution,
            platforms=platform_percentages,
            recent_mentions=recent_mentions,
            trending_topics=trending_topics,
            total_engagement=total_engagement,
            period_days=days
        )

    except Exception as e:
        logger.error(f"Social monitoring error for {brand_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{brand_id}/mentions", response_model=List[SocialMention])
async def get_social_mentions(
    brand_id: str,
    platform: Optional[str] = Query(None, description="플랫폼 필터"),
    sentiment: Optional[str] = Query(None, description="감정 필터 (positive/neutral/negative)"),
    limit: int = Query(50, ge=1, le=200, description="최대 개수"),
    offset: int = Query(0, ge=0, description="시작 위치")
):
    """
    소셜 멘션 목록 조회

    Args:
        brand_id: 브랜드 ID
        platform: 플랫폼 필터
        sentiment: 감정 필터
        limit: 최대 개수
        offset: 시작 위치

    Returns:
        멘션 목록
    """
    try:
        from app.services.shared.neo4j import get_neo4j_client

        neo4j = get_neo4j_client()

        # 동적 쿼리 구성
        where_clauses = ["(p:Post OR p:Content)", "p.brand_id = $brand_id"]
        params = {'brand_id': brand_id, 'limit': limit, 'offset': offset}

        if platform:
            where_clauses.append("p.platform = $platform")
            params['platform'] = platform

        if sentiment:
            where_clauses.append("p.sentiment = $sentiment")
            params['sentiment'] = sentiment

        query = f"""
        MATCH (p)
        WHERE {' AND '.join(where_clauses)}
        RETURN
            p.id as id,
            p.platform as platform,
            coalesce(p.content, p.text) as content,
            coalesce(p.sentiment, 'neutral') as sentiment,
            coalesce(p.like_count, p.likes, 0) as likes,
            coalesce(p.comment_count, p.comments, 0) as comments,
            p.author as author,
            p.created_at as created_at
        ORDER BY p.created_at DESC
        SKIP $offset
        LIMIT $limit
        """

        posts = neo4j.query(query, params) or []

        return [
            SocialMention(
                id=str(post.get('id', '')),
                platform=post.get('platform', 'unknown'),
                content=(post.get('content', '') or '')[:500],
                sentiment=post.get('sentiment', 'neutral'),
                timestamp=_format_timestamp(post.get('created_at')),
                likes=post.get('likes', 0) or 0,
                comments=post.get('comments', 0) or 0,
                author=post.get('author')
            )
            for post in posts
        ]

    except Exception as e:
        logger.error(f"Get mentions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{brand_id}/sentiment")
async def get_sentiment_analysis(
    brand_id: str,
    days: int = Query(30, ge=1, le=90, description="분석 기간")
):
    """
    감정 분석 상세 데이터

    Args:
        brand_id: 브랜드 ID
        days: 분석 기간

    Returns:
        일별 감정 추이 및 통계
    """
    try:
        from app.services.shared.neo4j import get_neo4j_client

        neo4j = get_neo4j_client()

        # 일별 감정 분석 (Content + Interaction 포함)
        query = """
        MATCH (n)
        WHERE (n:Post OR n:Content OR n:Interaction)
          AND n.brand_id = $brand_id
        WITH date(n.created_at) as day, coalesce(n.sentiment, 'neutral') as sentiment, count(*) as count
        RETURN day, sentiment, count
        ORDER BY day ASC
        """

        results = neo4j.query(query, {'brand_id': brand_id, 'days': days}) or []

        # 일별 데이터 집계
        daily_data: Dict[str, Dict[str, int]] = {}
        for row in results:
            day = str(row.get('day', ''))
            sentiment = row.get('sentiment', 'neutral')
            count = row.get('count', 0)

            if day not in daily_data:
                daily_data[day] = {'positive': 0, 'neutral': 0, 'negative': 0}

            if sentiment in daily_data[day]:
                daily_data[day][sentiment] = count

        # 응답 구성
        daily_stats = [
            {
                'date': day,
                'positive': data['positive'],
                'neutral': data['neutral'],
                'negative': data['negative'],
                'total': data['positive'] + data['neutral'] + data['negative']
            }
            for day, data in sorted(daily_data.items())
        ]

        # 전체 통계
        total_positive = sum(d['positive'] for d in daily_stats)
        total_neutral = sum(d['neutral'] for d in daily_stats)
        total_negative = sum(d['negative'] for d in daily_stats)
        total = total_positive + total_neutral + total_negative

        return {
            'brand_id': brand_id,
            'period_days': days,
            'summary': {
                'total': total,
                'positive': total_positive,
                'neutral': total_neutral,
                'negative': total_negative,
                'positive_rate': round(total_positive / total * 100, 1) if total > 0 else 0,
                'negative_rate': round(total_negative / total * 100, 1) if total > 0 else 0,
            },
            'daily_stats': daily_stats
        }

    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{brand_id}/platforms")
async def get_platform_breakdown(
    brand_id: str,
    days: int = Query(30, ge=1, le=90, description="분석 기간")
):
    """
    플랫폼별 분석 데이터

    Args:
        brand_id: 브랜드 ID
        days: 분석 기간

    Returns:
        플랫폼별 통계
    """
    try:
        from app.services.shared.neo4j import get_neo4j_client

        neo4j = get_neo4j_client()

        query = """
        MATCH (p)
        WHERE (p:Post OR p:Content)
          AND p.brand_id = $brand_id
        WITH p.platform as platform,
             count(*) as post_count,
             sum(coalesce(p.like_count, p.likes, 0)) as total_likes,
             sum(coalesce(p.comment_count, p.comments, 0)) as total_comments,
             sum(coalesce(p.share_count, p.shares, 0)) as total_shares,
             sum(coalesce(p.view_count, 0)) as total_views
        RETURN platform, post_count, total_likes, total_comments, total_shares, total_views
        ORDER BY post_count DESC
        """

        results = neo4j.query(query, {'brand_id': brand_id, 'days': days}) or []

        platforms = [
            {
                'platform': row.get('platform', 'unknown'),
                'post_count': row.get('post_count', 0),
                'total_likes': row.get('total_likes', 0) or 0,
                'total_comments': row.get('total_comments', 0) or 0,
                'total_shares': row.get('total_shares', 0) or 0,
                'total_views': row.get('total_views', 0) or 0,
                'engagement': (
                    (row.get('total_likes', 0) or 0) +
                    (row.get('total_comments', 0) or 0) +
                    (row.get('total_shares', 0) or 0) +
                    (row.get('total_views', 0) or 0)
                )
            }
            for row in results
        ]

        return {
            'brand_id': brand_id,
            'period_days': days,
            'platforms': platforms,
            'total_posts': sum(p['post_count'] for p in platforms),
            'total_engagement': sum(p['engagement'] for p in platforms)
        }

    except Exception as e:
        logger.error(f"Platform breakdown error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Helper Functions
# ============================================================

def _format_timestamp(ts) -> str:
    """타임스탬프 포맷팅"""
    if ts is None:
        return ""

    if isinstance(ts, datetime):
        return ts.isoformat()

    if isinstance(ts, str):
        return ts

    # Neo4j DateTime 객체
    try:
        return ts.isoformat()
    except:
        return str(ts)


def _extract_trending_topics(posts: List[Dict[str, Any]]) -> List[TrendingTopic]:
    """포스트에서 트렌딩 토픽 추출"""
    import re

    hashtag_counts: Dict[str, int] = {}

    for post in posts:
        content = post.get('content', '') or ''
        # 해시태그 추출
        hashtags = re.findall(r'#(\w+)', content)

        for tag in hashtags:
            hashtag_counts[tag] = hashtag_counts.get(tag, 0) + 1

    # 상위 10개 토픽
    sorted_topics = sorted(hashtag_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    return [
        TrendingTopic(topic=f"#{tag}", count=count)
        for tag, count in sorted_topics
    ]
