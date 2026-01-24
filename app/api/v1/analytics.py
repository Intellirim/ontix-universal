"""
Analytics API
분석 데이터 엔드포인트
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics")


# ============================================================
# Response Models
# ============================================================

class DailyStat(BaseModel):
    """일별 통계"""
    date: str
    sessions: int = 0
    messages: int = 0
    engagement: int = 0


class GradeDistribution(BaseModel):
    """등급 분포"""
    A: int = 0
    B: int = 0
    C: int = 0
    D: int = 0
    F: int = 0


class TopQuestion(BaseModel):
    """자주 묻는 질문"""
    question: str
    count: int
    category: Optional[str] = None


class ContentPerformance(BaseModel):
    """콘텐츠 성과"""
    id: str
    content: str
    platform: str
    likes: int = 0
    comments: int = 0
    shares: int = 0
    engagement_rate: float = 0.0


class AnalyticsSummary(BaseModel):
    """분석 요약"""
    brand_id: str
    period_days: int
    total_sessions: int
    total_messages: int
    avg_messages_per_session: float
    grade_distribution: GradeDistribution
    daily_stats: List[DailyStat]
    top_questions: List[TopQuestion]


# ============================================================
# API Endpoints
# ============================================================

@router.get("/{brand_id}", response_model=AnalyticsSummary)
async def get_analytics_summary(
    brand_id: str,
    days: int = Query(30, ge=1, le=365, description="분석 기간 (일)")
):
    """
    분석 요약 데이터 조회

    PostgreSQL에서 채팅 세션, 메시지, 등급 분포 등의 데이터를 조회합니다.
    (Neo4j 지식그래프 오염 방지)

    Args:
        brand_id: 브랜드 ID
        days: 분석 기간

    Returns:
        분석 요약 데이터
    """
    try:
        from app.services.chat.chat_storage import get_chat_storage

        storage = get_chat_storage()

        # PostgreSQL에서 분석 데이터 조회
        analytics = storage.get_analytics(brand_id=brand_id, days=days)

        # 등급 분포 변환
        grade_counts = analytics.grade_distribution or {}
        total_graded = sum(grade_counts.values()) if grade_counts else 0

        grade_distribution = GradeDistribution(
            A=round(grade_counts.get('A', 0) / total_graded * 100) if total_graded > 0 else 0,
            B=round(grade_counts.get('B', 0) / total_graded * 100) if total_graded > 0 else 0,
            C=round(grade_counts.get('C', 0) / total_graded * 100) if total_graded > 0 else 0,
            D=round(grade_counts.get('D', 0) / total_graded * 100) if total_graded > 0 else 0,
            F=round(grade_counts.get('F', 0) / total_graded * 100) if total_graded > 0 else 0,
        )

        # 일별 통계 변환 - sessions 수와 engagement 계산
        daily_stats = []
        for row in analytics.daily_message_counts:
            msg_count = row.get('count', 0)
            # 대략적인 세션 수 추정 (평균 5개 메시지/세션 가정)
            session_count = max(1, msg_count // 5) if msg_count > 0 else 0
            # engagement = sessions * 10 + messages (프론트엔드 공식)
            engagement = session_count * 10 + msg_count
            daily_stats.append(DailyStat(
                date=row.get('date', ''),
                sessions=session_count,
                messages=msg_count,
                engagement=engagement
            ))

        # 질문 유형 분포를 top_questions로 변환
        top_questions = [
            TopQuestion(
                question=f"{qtype} 유형 질문",
                count=count,
                category=qtype
            )
            for qtype, count in (analytics.question_type_distribution or {}).items()
        ]

        return AnalyticsSummary(
            brand_id=brand_id,
            period_days=days,
            total_sessions=analytics.total_sessions,
            total_messages=analytics.total_messages,
            avg_messages_per_session=round(analytics.avg_messages_per_session, 1),
            grade_distribution=grade_distribution,
            daily_stats=daily_stats,
            top_questions=top_questions
        )

    except Exception as e:
        logger.error(f"Analytics summary error for {brand_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{brand_id}/content-performance")
async def get_content_performance(
    brand_id: str,
    days: int = Query(30, ge=1, le=90, description="분석 기간"),
    platform: Optional[str] = Query(None, description="플랫폼 필터"),
    limit: int = Query(20, ge=1, le=100, description="최대 개수")
):
    """
    콘텐츠 성과 분석

    Args:
        brand_id: 브랜드 ID
        days: 분석 기간
        platform: 플랫폼 필터
        limit: 최대 개수

    Returns:
        콘텐츠 성과 목록
    """
    try:
        from app.services.shared.neo4j import get_neo4j_client

        neo4j = get_neo4j_client()

        where_clause = "p.brand_id = $brand_id AND p.created_at >= datetime() - duration({days: $days})"
        params = {'brand_id': brand_id, 'days': days, 'limit': limit}

        if platform:
            where_clause += " AND p.platform = $platform"
            params['platform'] = platform

        query = f"""
        MATCH (p:Post)
        WHERE {where_clause}
        WITH p,
             coalesce(p.likes, 0) as likes,
             coalesce(p.comments, 0) as comments,
             coalesce(p.shares, 0) as shares,
             (coalesce(p.likes, 0) + coalesce(p.comments, 0) + coalesce(p.shares, 0)) as engagement
        RETURN
            p.id as id,
            p.content as content,
            p.platform as platform,
            likes,
            comments,
            shares,
            engagement
        ORDER BY engagement DESC
        LIMIT $limit
        """

        results = neo4j.query(query, params) or []

        # 최대 인게이지먼트로 비율 계산
        max_engagement = max((r.get('engagement', 0) or 0) for r in results) if results else 1

        content_list = [
            ContentPerformance(
                id=str(row.get('id', '')),
                content=(row.get('content', '') or '')[:200],
                platform=row.get('platform', 'unknown'),
                likes=row.get('likes', 0) or 0,
                comments=row.get('comments', 0) or 0,
                shares=row.get('shares', 0) or 0,
                engagement_rate=round((row.get('engagement', 0) or 0) / max_engagement * 100, 1) if max_engagement > 0 else 0
            )
            for row in results
        ]

        return {
            'brand_id': brand_id,
            'period_days': days,
            'total_content': len(content_list),
            'content': content_list
        }

    except Exception as e:
        logger.error(f"Content performance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{brand_id}/engagement-trends")
async def get_engagement_trends(
    brand_id: str,
    days: int = Query(30, ge=7, le=90, description="분석 기간")
):
    """
    인게이지먼트 추이 분석

    Args:
        brand_id: 브랜드 ID
        days: 분석 기간

    Returns:
        일별 인게이지먼트 추이
    """
    try:
        from app.services.shared.neo4j import get_neo4j_client

        neo4j = get_neo4j_client()

        query = """
        MATCH (p:Post)
        WHERE p.brand_id = $brand_id
          AND p.created_at >= datetime() - duration({days: $days})
        WITH date(p.created_at) as day,
             count(*) as post_count,
             sum(coalesce(p.likes, 0)) as total_likes,
             sum(coalesce(p.comments, 0)) as total_comments,
             sum(coalesce(p.shares, 0)) as total_shares
        RETURN
            toString(day) as date,
            post_count,
            total_likes,
            total_comments,
            total_shares,
            (total_likes + total_comments + total_shares) as total_engagement
        ORDER BY day ASC
        """

        results = neo4j.query(query, {
            'brand_id': brand_id,
            'days': days
        }) or []

        daily_trends = [
            {
                'date': row.get('date', ''),
                'posts': row.get('post_count', 0),
                'likes': row.get('total_likes', 0),
                'comments': row.get('total_comments', 0),
                'shares': row.get('total_shares', 0),
                'engagement': row.get('total_engagement', 0)
            }
            for row in results
        ]

        # 총계 계산
        total_posts = sum(d['posts'] for d in daily_trends)
        total_engagement = sum(d['engagement'] for d in daily_trends)
        avg_daily_engagement = total_engagement / len(daily_trends) if daily_trends else 0

        return {
            'brand_id': brand_id,
            'period_days': days,
            'summary': {
                'total_posts': total_posts,
                'total_engagement': total_engagement,
                'avg_daily_engagement': round(avg_daily_engagement, 1),
                'avg_engagement_per_post': round(total_engagement / total_posts, 1) if total_posts > 0 else 0
            },
            'daily_trends': daily_trends
        }

    except Exception as e:
        logger.error(f"Engagement trends error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{brand_id}/quality-metrics")
async def get_quality_metrics(
    brand_id: str,
    days: int = Query(30, ge=1, le=90, description="분석 기간")
):
    """
    응답 품질 메트릭 (PostgreSQL 기반)

    Args:
        brand_id: 브랜드 ID
        days: 분석 기간

    Returns:
        품질 메트릭 데이터
    """
    try:
        from app.services.chat.chat_storage import get_chat_storage

        storage = get_chat_storage()
        analytics = storage.get_analytics(brand_id=brand_id, days=days)

        # 등급 분포
        grade_counts = analytics.grade_distribution or {}
        grade_metrics = {
            grade: {'count': count, 'avg_score': 0.0}
            for grade, count in grade_counts.items()
        }

        # 전체 통계
        total_count = sum(grade_counts.values()) if grade_counts else 0

        # A, B, C 등급 통과율 계산
        pass_count = (
            grade_counts.get('A', 0) +
            grade_counts.get('B', 0) +
            grade_counts.get('C', 0)
        )

        return {
            'brand_id': brand_id,
            'period_days': days,
            'overall': {
                'total_responses': total_count,
                'avg_score': 0.0,  # PostgreSQL에서 별도 계산 필요
                'pass_rate': round(pass_count / total_count * 100, 1) if total_count > 0 else 0
            },
            'grade_breakdown': grade_metrics,
            'daily_quality': []  # PostgreSQL에서 별도 쿼리 필요
        }

    except Exception as e:
        logger.error(f"Quality metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{brand_id}/export")
async def export_chat_data(
    brand_id: str,
    days: int = Query(30, ge=1, le=365, description="내보내기 기간 (일)"),
    format: str = Query("json", description="내보내기 형식 (json, csv)")
):
    """
    채팅 데이터 내보내기 (PostgreSQL 기반)

    Args:
        brand_id: 브랜드 ID
        days: 내보내기 기간
        format: 내보내기 형식 (json 또는 csv)

    Returns:
        채팅 데이터 (JSON 또는 CSV)
    """
    from fastapi.responses import StreamingResponse
    import json
    import io

    try:
        from app.services.chat.chat_storage import get_chat_storage

        storage = get_chat_storage()

        # 세션 목록 조회
        sessions = storage.list_sessions(brand_id=brand_id, limit=1000)

        sessions_data = []
        for session in sessions:
            # 각 세션의 메시지 조회
            messages = storage.get_messages(session_id=session.id, limit=500)

            session_data = {
                'session_id': session.id,
                'user_id': session.user_id,
                'created_at': session.created_at.isoformat(),
                'updated_at': session.updated_at.isoformat(),
                'message_count': session.message_count,
                'messages': [
                    {
                        'message_id': msg.id,
                        'role': msg.role,
                        'content': msg.content,
                        'grade': msg.metadata.get('grade'),
                        'score': msg.metadata.get('score'),
                        'question_type': msg.metadata.get('question_type'),
                        'timestamp': msg.timestamp.isoformat()
                    }
                    for msg in messages
                ]
            }
            sessions_data.append(session_data)

        if format == "csv":
            output = io.StringIO()
            output.write("session_id,user_id,created_at,role,content,grade,score,question_type,message_timestamp\n")

            for session in sessions_data:
                for msg in session.get('messages', []):
                    content = (msg.get('content', '') or '').replace('"', '""').replace('\n', ' ')
                    row = f'"{session["session_id"]}","{session.get("user_id", "")}","{session["created_at"]}","{msg.get("role", "")}","{content}","{msg.get("grade", "")}","{msg.get("score", "")}","{msg.get("question_type", "")}","{msg.get("timestamp", "")}"\n'
                    output.write(row)

            output.seek(0)
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename={brand_id}_chat_export_{days}days.csv"
                }
            )
        else:
            export_data = {
                'brand_id': brand_id,
                'export_date': datetime.now().isoformat(),
                'period_days': days,
                'total_sessions': len(sessions_data),
                'total_messages': sum(len(s.get('messages', [])) for s in sessions_data),
                'sessions': sessions_data
            }

            json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
            output = io.BytesIO(json_str.encode('utf-8'))

            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename={brand_id}_chat_export_{days}days.json"
                }
            )

    except Exception as e:
        logger.error(f"Export chat data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{brand_id}/sessions")
async def get_chat_sessions(
    brand_id: str,
    days: int = Query(30, ge=1, le=365, description="조회 기간 (일)"),
    limit: int = Query(50, ge=1, le=200, description="최대 개수"),
    offset: int = Query(0, ge=0, description="시작 위치")
):
    """
    채팅 세션 목록 조회 (PostgreSQL 기반)

    Args:
        brand_id: 브랜드 ID
        days: 조회 기간
        limit: 최대 개수
        offset: 시작 위치

    Returns:
        세션 목록
    """
    try:
        from app.services.chat.chat_storage import get_chat_storage

        storage = get_chat_storage()

        # 세션 목록 조회
        sessions_list = storage.list_sessions(
            brand_id=brand_id,
            limit=limit,
            offset=offset
        )

        sessions = [
            {
                'session_id': s.id,
                'user_id': s.user_id,
                'created_at': s.created_at.isoformat(),
                'updated_at': s.updated_at.isoformat(),
                'message_count': s.message_count,
                'grades': []  # PostgreSQL에서 별도 조회 필요
            }
            for s in sessions_list
        ]

        # 총 개수는 list_sessions의 결과로 추정
        total = len(sessions_list) + offset

        return {
            'brand_id': brand_id,
            'period_days': days,
            'total': total,
            'offset': offset,
            'limit': limit,
            'sessions': sessions
        }

    except Exception as e:
        logger.error(f"Get chat sessions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{brand_id}/sessions/{session_id}/messages")
async def get_session_messages(
    brand_id: str,
    session_id: str,
    limit: int = Query(100, ge=1, le=500, description="최대 개수")
):
    """
    특정 세션의 메시지 조회 (PostgreSQL)

    Args:
        brand_id: 브랜드 ID
        session_id: 세션 ID
        limit: 최대 개수

    Returns:
        메시지 목록
    """
    try:
        from app.services.chat.chat_storage import get_chat_storage

        storage = get_chat_storage()

        # PostgreSQL에서 메시지 조회
        stored_messages = storage.get_messages(session_id=session_id, limit=limit)

        messages = [
            {
                'message_id': msg.id,
                'role': msg.role,
                'content': msg.content,
                'grade': msg.metadata.get('grade') if msg.metadata else None,
                'score': msg.metadata.get('score') if msg.metadata else None,
                'question_type': msg.metadata.get('question_type') if msg.metadata else None,
                'timestamp': msg.timestamp.isoformat() if msg.timestamp else ''
            }
            for msg in stored_messages
        ]

        return {
            'brand_id': brand_id,
            'session_id': session_id,
            'total_messages': len(messages),
            'messages': messages
        }

    except Exception as e:
        logger.error(f"Get session messages error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
