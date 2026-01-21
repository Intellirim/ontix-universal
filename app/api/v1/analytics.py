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

    Neo4j에서 채팅 세션, 메시지, 등급 분포 등의 데이터를 조회합니다.

    Args:
        brand_id: 브랜드 ID
        days: 분석 기간

    Returns:
        분석 요약 데이터
    """
    try:
        from app.services.shared.neo4j import get_neo4j_client

        neo4j = get_neo4j_client()

        # 세션 및 메시지 통계
        session_query = """
        MATCH (s:ChatSession)-[:HAS_MESSAGE]->(m:ChatMessage)
        WHERE s.brand_id = $brand_id
          AND s.created_at >= datetime() - duration({days: $days})
        WITH s, count(m) as message_count
        RETURN count(s) as total_sessions,
               sum(message_count) as total_messages,
               avg(message_count) as avg_messages
        """

        session_result = neo4j.query(session_query, {
            'brand_id': brand_id,
            'days': days
        })

        total_sessions = 0
        total_messages = 0
        avg_messages = 0.0

        if session_result and len(session_result) > 0:
            row = session_result[0]
            total_sessions = row.get('total_sessions', 0) or 0
            total_messages = row.get('total_messages', 0) or 0
            avg_messages = row.get('avg_messages', 0) or 0.0

        # 등급 분포 (검증 등급)
        grade_query = """
        MATCH (m:ChatMessage)
        WHERE m.brand_id = $brand_id
          AND m.created_at >= datetime() - duration({days: $days})
          AND m.grade IS NOT NULL
        RETURN m.grade as grade, count(*) as count
        """

        grade_results = neo4j.query(grade_query, {
            'brand_id': brand_id,
            'days': days
        }) or []

        grade_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
        total_graded = 0

        for row in grade_results:
            grade = row.get('grade', '')
            count = row.get('count', 0)
            if grade in grade_counts:
                grade_counts[grade] = count
                total_graded += count

        # 비율로 변환
        grade_distribution = GradeDistribution(
            A=round(grade_counts['A'] / total_graded * 100) if total_graded > 0 else 0,
            B=round(grade_counts['B'] / total_graded * 100) if total_graded > 0 else 0,
            C=round(grade_counts['C'] / total_graded * 100) if total_graded > 0 else 0,
            D=round(grade_counts['D'] / total_graded * 100) if total_graded > 0 else 0,
            F=round(grade_counts['F'] / total_graded * 100) if total_graded > 0 else 0,
        )

        # 일별 통계
        daily_query = """
        MATCH (s:ChatSession)
        WHERE s.brand_id = $brand_id
          AND s.created_at >= datetime() - duration({days: $days})
        WITH date(s.created_at) as day, count(s) as sessions
        OPTIONAL MATCH (m:ChatMessage)
        WHERE m.brand_id = $brand_id
          AND date(m.created_at) = day
        WITH day, sessions, count(m) as messages
        RETURN toString(day) as date, sessions, messages
        ORDER BY day ASC
        """

        daily_results = neo4j.query(daily_query, {
            'brand_id': brand_id,
            'days': days
        }) or []

        daily_stats = [
            DailyStat(
                date=row.get('date', ''),
                sessions=row.get('sessions', 0) or 0,
                messages=row.get('messages', 0) or 0
            )
            for row in daily_results
        ]

        # 자주 묻는 질문 (질문 유형별 집계)
        questions_query = """
        MATCH (m:ChatMessage)
        WHERE m.brand_id = $brand_id
          AND m.role = 'user'
          AND m.created_at >= datetime() - duration({days: $days})
        WITH m.content as question, m.question_type as category, count(*) as count
        RETURN question, category, count
        ORDER BY count DESC
        LIMIT 10
        """

        questions_results = neo4j.query(questions_query, {
            'brand_id': brand_id,
            'days': days
        }) or []

        top_questions = [
            TopQuestion(
                question=(row.get('question', '') or '')[:100],
                count=row.get('count', 0),
                category=row.get('category')
            )
            for row in questions_results
        ]

        return AnalyticsSummary(
            brand_id=brand_id,
            period_days=days,
            total_sessions=total_sessions,
            total_messages=total_messages,
            avg_messages_per_session=round(avg_messages, 1),
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
    응답 품질 메트릭

    Args:
        brand_id: 브랜드 ID
        days: 분석 기간

    Returns:
        품질 메트릭 데이터
    """
    try:
        from app.services.shared.neo4j import get_neo4j_client

        neo4j = get_neo4j_client()

        # 등급별 통계
        grade_query = """
        MATCH (m:ChatMessage)
        WHERE m.brand_id = $brand_id
          AND m.role = 'assistant'
          AND m.created_at >= datetime() - duration({days: $days})
          AND m.grade IS NOT NULL
        WITH m.grade as grade,
             count(*) as count,
             avg(coalesce(m.score, 0)) as avg_score
        RETURN grade, count, avg_score
        ORDER BY grade ASC
        """

        grade_results = neo4j.query(grade_query, {
            'brand_id': brand_id,
            'days': days
        }) or []

        grade_metrics = {
            row.get('grade', ''): {
                'count': row.get('count', 0),
                'avg_score': round(row.get('avg_score', 0) or 0, 3)
            }
            for row in grade_results
        }

        # 일별 평균 점수
        daily_query = """
        MATCH (m:ChatMessage)
        WHERE m.brand_id = $brand_id
          AND m.role = 'assistant'
          AND m.created_at >= datetime() - duration({days: $days})
          AND m.score IS NOT NULL
        WITH date(m.created_at) as day,
             avg(m.score) as avg_score,
             count(*) as count
        RETURN toString(day) as date, avg_score, count
        ORDER BY day ASC
        """

        daily_results = neo4j.query(daily_query, {
            'brand_id': brand_id,
            'days': days
        }) or []

        daily_quality = [
            {
                'date': row.get('date', ''),
                'avg_score': round(row.get('avg_score', 0) or 0, 3),
                'count': row.get('count', 0)
            }
            for row in daily_results
        ]

        # 전체 통계
        total_count = sum(gm['count'] for gm in grade_metrics.values())
        weighted_score = sum(
            gm['count'] * gm['avg_score']
            for gm in grade_metrics.values()
        )
        overall_avg = weighted_score / total_count if total_count > 0 else 0

        return {
            'brand_id': brand_id,
            'period_days': days,
            'overall': {
                'total_responses': total_count,
                'avg_score': round(overall_avg, 3),
                'pass_rate': round(
                    (grade_metrics.get('A', {}).get('count', 0) +
                     grade_metrics.get('B', {}).get('count', 0) +
                     grade_metrics.get('C', {}).get('count', 0)) / total_count * 100, 1
                ) if total_count > 0 else 0
            },
            'grade_breakdown': grade_metrics,
            'daily_quality': daily_quality
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
    채팅 데이터 내보내기

    Neo4j에서 채팅 세션 및 메시지 데이터를 내보냅니다.

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
        from app.services.shared.neo4j import get_neo4j_client

        neo4j = get_neo4j_client()

        # 세션 및 메시지 조회
        query = """
        MATCH (s:ChatSession)
        WHERE s.brand_id = $brand_id
          AND s.created_at >= datetime() - duration({days: $days})
        OPTIONAL MATCH (s)-[:HAS_MESSAGE]->(m:ChatMessage)
        WITH s, m
        ORDER BY s.created_at, m.created_at
        WITH s, collect({
            message_id: m.id,
            role: m.role,
            content: m.content,
            grade: m.grade,
            score: m.score,
            question_type: m.question_type,
            timestamp: toString(m.created_at)
        }) as messages
        RETURN
            s.id as session_id,
            s.user_id as user_id,
            toString(s.created_at) as created_at,
            toString(s.updated_at) as updated_at,
            s.message_count as message_count,
            messages
        ORDER BY s.created_at DESC
        """

        results = neo4j.query(query, {
            'brand_id': brand_id,
            'days': days
        }) or []

        sessions_data = []
        for row in results:
            session = {
                'session_id': row.get('session_id', ''),
                'user_id': row.get('user_id'),
                'created_at': row.get('created_at', ''),
                'updated_at': row.get('updated_at', ''),
                'message_count': row.get('message_count', 0),
                'messages': [m for m in row.get('messages', []) if m.get('message_id')]
            }
            sessions_data.append(session)

        if format == "csv":
            # CSV 형식으로 변환
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
            # JSON 형식
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
    채팅 세션 목록 조회

    Args:
        brand_id: 브랜드 ID
        days: 조회 기간
        limit: 최대 개수
        offset: 시작 위치

    Returns:
        세션 목록
    """
    try:
        from app.services.shared.neo4j import get_neo4j_client

        neo4j = get_neo4j_client()

        query = """
        MATCH (s:ChatSession)
        WHERE s.brand_id = $brand_id
          AND s.created_at >= datetime() - duration({days: $days})
        OPTIONAL MATCH (s)-[:HAS_MESSAGE]->(m:ChatMessage)
        WITH s, count(m) as msg_count,
             collect(DISTINCT m.grade) as grades
        RETURN
            s.id as session_id,
            s.user_id as user_id,
            toString(s.created_at) as created_at,
            toString(s.updated_at) as updated_at,
            msg_count as message_count,
            grades
        ORDER BY s.created_at DESC
        SKIP $offset
        LIMIT $limit
        """

        results = neo4j.query(query, {
            'brand_id': brand_id,
            'days': days,
            'limit': limit,
            'offset': offset
        }) or []

        sessions = [
            {
                'session_id': row.get('session_id', ''),
                'user_id': row.get('user_id'),
                'created_at': row.get('created_at', ''),
                'updated_at': row.get('updated_at', ''),
                'message_count': row.get('message_count', 0),
                'grades': [g for g in row.get('grades', []) if g]
            }
            for row in results
        ]

        # 총 개수 조회
        count_query = """
        MATCH (s:ChatSession)
        WHERE s.brand_id = $brand_id
          AND s.created_at >= datetime() - duration({days: $days})
        RETURN count(s) as total
        """

        count_result = neo4j.query(count_query, {
            'brand_id': brand_id,
            'days': days
        })

        total = count_result[0].get('total', 0) if count_result else 0

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
    특정 세션의 메시지 조회

    Args:
        brand_id: 브랜드 ID
        session_id: 세션 ID
        limit: 최대 개수

    Returns:
        메시지 목록
    """
    try:
        from app.services.shared.neo4j import get_neo4j_client

        neo4j = get_neo4j_client()

        query = """
        MATCH (s:ChatSession {id: $session_id, brand_id: $brand_id})-[:HAS_MESSAGE]->(m:ChatMessage)
        RETURN
            m.id as message_id,
            m.role as role,
            m.content as content,
            m.grade as grade,
            m.score as score,
            m.question_type as question_type,
            toString(m.created_at) as timestamp
        ORDER BY m.created_at ASC
        LIMIT $limit
        """

        results = neo4j.query(query, {
            'brand_id': brand_id,
            'session_id': session_id,
            'limit': limit
        }) or []

        messages = [
            {
                'message_id': row.get('message_id', ''),
                'role': row.get('role', ''),
                'content': row.get('content', ''),
                'grade': row.get('grade'),
                'score': row.get('score'),
                'question_type': row.get('question_type'),
                'timestamp': row.get('timestamp', '')
            }
            for row in results
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
