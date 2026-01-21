### **app/api/v1/admin.py**
"""
Admin API
관리 엔드포인트 - RBAC 적용
"""

from fastapi import APIRouter, HTTPException, Query, Request, Depends
from fastapi.responses import StreamingResponse
from typing import Optional, List
from app.services.platform.monitoring import get_monitoring_service, get_system_stats as get_sys_stats
from app.core.engine import UniversalEngine
from app.services.platform.config_manager import ConfigManager
from app.services.platform.analytics import get_analytics_service
from app.services.chat.chat_storage import ChatStorageService
from app.core.security import log_audit, AuditAction
from app.core.auth import (
    User, get_current_user, require_permission, Permission,
    get_user_brand_filter, filter_brands_by_user
)
import logging
import json
import csv
import io
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin")


@router.get("/health")
async def health_check(user: User = Depends(get_current_user)):
    """
    헬스체크 (로그인 필요)

    Returns:
        헬스 상태
    """
    try:
        monitoring = get_monitoring_service()
        health = monitoring.health_check()
        return health
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_system_stats(
    _: User = Depends(require_permission(Permission.SYSTEM_MONITOR))
):
    """
    시스템 통계 (SYSTEM_MONITOR 권한 필요)

    Returns:
        시스템 통계
    """
    try:
        stats = get_sys_stats()
        return stats
    except Exception as e:
        logger.error(f"System stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reload/{brand_id}")
async def reload_brand_config(
    brand_id: str,
    request: Request,
    user: User = Depends(require_permission(Permission.SYSTEM_CONFIG))
):
    """
    브랜드 설정 리로드 (SYSTEM_CONFIG 권한 필요)

    Args:
        brand_id: 브랜드 ID

    Returns:
        성공 메시지
    """
    try:
        # Config 리로드
        ConfigManager.reload_config(brand_id)

        # Engine 캐시 클리어
        UniversalEngine.clear_cache(brand_id)

        # 감사 로그
        log_audit(AuditAction.CONFIG_RELOAD, request, user_id=user.id, details={"brand_id": brand_id})

        return {
            'message': f'Brand config reloaded: {brand_id}',
            'brand_id': brand_id
        }
    except Exception as e:
        logger.error(f"Reload config error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear")
async def clear_all_cache(
    request: Request,
    user: User = Depends(require_permission(Permission.SYSTEM_CONFIG))
):
    """
    전체 캐시 클리어 (SYSTEM_CONFIG 권한 필요)

    Returns:
        성공 메시지
    """
    try:
        # Config 캐시 클리어
        ConfigManager.clear_cache()

        # Engine 캐시 클리어
        UniversalEngine.clear_cache()

        # 감사 로그
        log_audit(AuditAction.CACHE_CLEAR, request, user_id=user.id)

        return {'message': 'All caches cleared'}
    except Exception as e:
        logger.error(f"Clear cache error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics")
async def get_analytics(
    brand_id: Optional[str] = Query(None, description="브랜드 ID (선택)"),
    period: str = Query("7d", description="분석 기간 (1d, 7d, 30d, 90d)"),
    user: User = Depends(get_current_user)
):
    """
    분석 데이터 조회 (역할에 따라 자기 브랜드만 조회)

    Args:
        brand_id: 브랜드 ID (선택)
        period: 분석 기간

    Returns:
        분석 데이터
    """
    try:
        # 권한 체크: 자기 브랜드만 조회 가능
        if brand_id and not user.can_access_brand(brand_id):
            raise HTTPException(status_code=403, detail=f"No access to brand: {brand_id}")

        # 클라이언트 어드민은 브랜드 ID 필수
        accessible_brands = user.get_accessible_brands()
        if accessible_brands is not None and not brand_id:
            # 첫 번째 브랜드로 기본 설정
            brand_id = accessible_brands[0] if accessible_brands else None
            if not brand_id:
                raise HTTPException(status_code=400, detail="No accessible brands")

        analytics_service = get_analytics_service()

        # 기간 파싱
        period_days = {
            "1d": 1,
            "7d": 7,
            "30d": 30,
            "90d": 90
        }.get(period, 7)

        # 분석 데이터 조회
        analytics = analytics_service.get_analytics(
            brand_id=brand_id,
            days=period_days
        )

        return {
            "period": period,
            "brand_id": brand_id,
            "total_messages": analytics.get("total_messages", 0),
            "unique_sessions": analytics.get("unique_sessions", 0),
            "avg_response_time": analytics.get("avg_response_time", 0),
            "feature_usage": analytics.get("feature_usage", {}),
            "hourly_distribution": analytics.get("hourly_distribution", [0] * 24)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/realtime")
async def get_realtime_metrics(
    _: User = Depends(require_permission(Permission.SYSTEM_MONITOR))
):
    """
    실시간 메트릭 조회 (SYSTEM_MONITOR 권한 필요)

    Returns:
        실시간 메트릭
    """
    try:
        analytics_service = get_analytics_service()
        metrics = analytics_service.get_realtime_metrics()

        return {
            "requests_per_minute": metrics.get("requests_per_minute", 0),
            "active_sessions": metrics.get("active_sessions", 0),
            "avg_response_time_ms": metrics.get("avg_response_time_ms", 0),
            "error_rate": metrics.get("error_rate", 0),
            "cache_hit_rate": metrics.get("cache_hit_rate", 0)
        }
    except Exception as e:
        logger.error(f"Realtime metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/compare")
async def compare_brands(
    period: str = Query("7d", description="분석 기간 (1d, 7d, 30d, 90d)"),
    user: User = Depends(get_current_user)
):
    """
    브랜드별 비교 분석 (역할에 따라 자기 브랜드만)

    Args:
        period: 분석 기간

    Returns:
        브랜드별 비교 데이터
    """
    try:
        analytics_service = get_analytics_service()
        all_brands = ConfigManager.list_brands()

        # 권한에 따라 브랜드 필터링
        brands = filter_brands_by_user(user, all_brands)

        period_days = {
            "1d": 1,
            "7d": 7,
            "30d": 30,
            "90d": 90
        }.get(period, 7)

        comparison = []
        for brand_id in brands:
            analytics = analytics_service.get_analytics(
                brand_id=brand_id,
                days=period_days
            )
            comparison.append({
                "brand_id": brand_id,
                "total_messages": analytics.get("total_messages", 0),
                "unique_sessions": analytics.get("unique_sessions", 0),
                "avg_response_time": analytics.get("avg_response_time", 0),
                "feature_usage": analytics.get("feature_usage", {}),
            })

        # 총 통계
        total_messages = sum(b["total_messages"] for b in comparison)
        total_sessions = sum(b["unique_sessions"] for b in comparison)

        return {
            "period": period,
            "brands": comparison,
            "summary": {
                "total_brands": len(brands),
                "total_messages": total_messages,
                "total_sessions": total_sessions,
                "most_active_brand": max(comparison, key=lambda x: x["total_messages"])["brand_id"] if comparison and total_messages > 0 else None,
            }
        }
    except Exception as e:
        logger.error(f"Brand comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/export")
async def export_analytics(
    request: Request,
    brand_id: Optional[str] = Query(None, description="브랜드 ID (없으면 전체)"),
    period: str = Query("7d", description="분석 기간"),
    format: str = Query("json", description="내보내기 형식 (json, csv)"),
    include_messages: bool = Query(True, description="실제 채팅 메시지 포함 여부"),
    user: User = Depends(require_permission(Permission.ANALYTICS_EXPORT))
):
    """
    분석 데이터 내보내기 (ANALYTICS_EXPORT 권한 필요)

    Args:
        brand_id: 브랜드 ID (선택)
        period: 분석 기간
        format: 내보내기 형식 (json/csv)
        include_messages: 실제 채팅 메시지 포함 여부

    Returns:
        다운로드 가능한 파일 (통계 + 실제 질문/응답 데이터)
    """
    try:
        # 권한 체크: 자기 브랜드만 내보내기 가능
        if brand_id and not user.can_access_brand(brand_id):
            raise HTTPException(status_code=403, detail=f"No access to brand: {brand_id}")

        # 감사 로그 (데이터 내보내기는 보안상 중요)
        log_audit(AuditAction.DATA_EXPORT, request, user_id=user.id, details={
            "brand_id": brand_id,
            "period": period,
            "format": format,
            "include_messages": include_messages
        })
        analytics_service = get_analytics_service()
        chat_storage = ChatStorageService()

        # 권한에 따라 브랜드 필터링
        all_brands = ConfigManager.list_brands()
        accessible_brands = filter_brands_by_user(user, all_brands)
        brands = [brand_id] if brand_id else accessible_brands

        period_days = {
            "1d": 1,
            "7d": 7,
            "30d": 30,
            "90d": 90
        }.get(period, 7)

        # 데이터 수집
        export_data = []
        all_chat_messages = []

        for bid in brands:
            # 통계 데이터
            analytics = analytics_service.get_analytics(
                brand_id=bid,
                days=period_days
            )

            brand_data = {
                "brand_id": bid,
                "period": period,
                "total_messages": analytics.get("total_messages", 0),
                "unique_sessions": analytics.get("unique_sessions", 0),
                "avg_response_time": analytics.get("avg_response_time", 0),
                "feature_usage": analytics.get("feature_usage", {}),
                "hourly_distribution": analytics.get("hourly_distribution", [0] * 24),
                "exported_at": datetime.now().isoformat()
            }

            # 실제 채팅 메시지 데이터 가져오기
            if include_messages:
                try:
                    sessions = chat_storage.list_sessions(brand_id=bid, limit=1000)

                    for session in sessions:
                        # 기간 필터링
                        if session.created_at < datetime.utcnow() - timedelta(days=period_days):
                            continue

                        messages = chat_storage.get_messages(session.id, limit=500)

                        for msg in messages:
                            all_chat_messages.append({
                                "brand_id": bid,
                                "session_id": session.id,
                                "message_id": msg.id,
                                "role": msg.role,
                                "content": msg.content,
                                "timestamp": msg.timestamp.isoformat(),
                                "grade": msg.metadata.get("grade"),
                                "question_type": msg.metadata.get("question_type"),
                                "response_time_ms": msg.metadata.get("response_time_ms")
                            })
                except Exception as e:
                    logger.warning(f"Failed to get chat messages for {bid}: {e}")
                    brand_data["chat_data_error"] = str(e)

            export_data.append(brand_data)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format.lower() == "csv":
            # CSV 형식 - 채팅 메시지 중심
            output = io.StringIO()
            writer = csv.writer(output)

            if include_messages and all_chat_messages:
                # 채팅 메시지 CSV
                writer.writerow([
                    "brand_id", "session_id", "message_id", "role", "content",
                    "timestamp", "grade", "question_type", "response_time_ms"
                ])

                for msg in all_chat_messages:
                    writer.writerow([
                        msg["brand_id"],
                        msg["session_id"],
                        msg["message_id"],
                        msg["role"],
                        msg["content"],
                        msg["timestamp"],
                        msg.get("grade", ""),
                        msg.get("question_type", ""),
                        msg.get("response_time_ms", "")
                    ])
            else:
                # 통계 요약 CSV (기존 방식)
                writer.writerow([
                    "brand_id", "period", "total_messages", "unique_sessions",
                    "avg_response_time", "feature_usage", "exported_at"
                ])

                for row in export_data:
                    writer.writerow([
                        row["brand_id"],
                        row["period"],
                        row["total_messages"],
                        row["unique_sessions"],
                        row["avg_response_time"],
                        json.dumps(row["feature_usage"]),
                        row["exported_at"]
                    ])

            output.seek(0)
            filename = f"chat_export_{timestamp}.csv"

            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        else:
            # JSON 형식
            filename = f"chat_export_{timestamp}.json"

            export_content = {
                "export_info": {
                    "exported_at": datetime.now().isoformat(),
                    "period": period,
                    "period_days": period_days,
                    "brand_filter": brand_id,
                    "total_brands": len(export_data),
                    "include_messages": include_messages
                },
                "analytics_summary": export_data,
            }

            if include_messages:
                export_content["chat_messages"] = all_chat_messages
                export_content["export_info"]["total_messages_exported"] = len(all_chat_messages)

            content = json.dumps(export_content, indent=2, ensure_ascii=False)

            return StreamingResponse(
                iter([content]),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
    except Exception as e:
        logger.error(f"Analytics export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
