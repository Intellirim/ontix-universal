"""
Alert Management API - Production Grade v1.0
알림 관리 API

Features:
- 알림 생성/조회/업데이트/삭제
- 알림 규칙 관리
- 알림 히스토리
- 실시간 알림 상태

Author: ONTIX Universal Team
"""

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alerts")


# ============================================================
# Enums
# ============================================================

class AlertType(str, Enum):
    """알림 유형"""
    MENTION = "mention"          # 멘션 알림
    KEYWORD = "keyword"          # 키워드 알림
    SENTIMENT = "sentiment"      # 감정 변화 알림
    ENGAGEMENT = "engagement"    # 인게이지먼트 알림
    THRESHOLD = "threshold"      # 임계값 초과 알림
    ANOMALY = "anomaly"          # 이상 탐지 알림
    SCHEDULED = "scheduled"      # 예약 알림


class AlertSeverity(str, Enum):
    """알림 심각도"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """알림 상태"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SNOOZED = "snoozed"


class NotificationChannel(str, Enum):
    """알림 채널"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    IN_APP = "in_app"


# ============================================================
# Request/Response Models
# ============================================================

class AlertRuleCreate(BaseModel):
    """알림 규칙 생성 요청"""
    name: str = Field(..., min_length=1, max_length=100, description="규칙 이름")
    description: Optional[str] = Field(None, max_length=500, description="규칙 설명")
    alert_type: AlertType = Field(..., description="알림 유형")
    conditions: Dict[str, Any] = Field(..., description="트리거 조건")
    severity: AlertSeverity = Field(AlertSeverity.MEDIUM, description="심각도")
    channels: List[NotificationChannel] = Field(
        default=[NotificationChannel.IN_APP],
        description="알림 채널"
    )
    enabled: bool = Field(True, description="활성화 여부")
    cooldown_minutes: int = Field(60, ge=0, le=1440, description="재알림 대기 시간(분)")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "부정 감정 급증 알림",
                "description": "부정 감정 비율이 30%를 초과하면 알림",
                "alert_type": "sentiment",
                "conditions": {
                    "metric": "negative_ratio",
                    "operator": "gt",
                    "threshold": 0.3,
                    "time_window_minutes": 60,
                },
                "severity": "high",
                "channels": ["email", "slack"],
                "enabled": True,
                "cooldown_minutes": 120,
            }
        }


class AlertRuleUpdate(BaseModel):
    """알림 규칙 업데이트 요청"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    conditions: Optional[Dict[str, Any]] = None
    severity: Optional[AlertSeverity] = None
    channels: Optional[List[NotificationChannel]] = None
    enabled: Optional[bool] = None
    cooldown_minutes: Optional[int] = Field(None, ge=0, le=1440)


class AlertRuleResponse(BaseModel):
    """알림 규칙 응답"""
    id: str
    brand_id: str
    name: str
    description: Optional[str]
    alert_type: AlertType
    conditions: Dict[str, Any]
    severity: AlertSeverity
    channels: List[NotificationChannel]
    enabled: bool
    cooldown_minutes: int
    created_at: datetime
    updated_at: datetime
    last_triggered_at: Optional[datetime]
    trigger_count: int


class AlertCreate(BaseModel):
    """알림 수동 생성 요청"""
    rule_id: Optional[str] = Field(None, description="연관된 규칙 ID")
    alert_type: AlertType = Field(..., description="알림 유형")
    title: str = Field(..., min_length=1, max_length=200, description="알림 제목")
    message: str = Field(..., min_length=1, max_length=2000, description="알림 내용")
    severity: AlertSeverity = Field(AlertSeverity.MEDIUM, description="심각도")
    metadata: Optional[Dict[str, Any]] = Field(None, description="추가 메타데이터")


class AlertResponse(BaseModel):
    """알림 응답"""
    id: str
    brand_id: str
    rule_id: Optional[str]
    alert_type: AlertType
    title: str
    message: str
    severity: AlertSeverity
    status: AlertStatus
    metadata: Optional[Dict[str, Any]]
    created_at: datetime
    acknowledged_at: Optional[datetime]
    resolved_at: Optional[datetime]
    acknowledged_by: Optional[str]
    resolved_by: Optional[str]


class AlertStatusUpdate(BaseModel):
    """알림 상태 업데이트 요청"""
    status: AlertStatus
    comment: Optional[str] = Field(None, max_length=500)


class AlertsListResponse(BaseModel):
    """알림 목록 응답"""
    alerts: List[AlertResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class AlertRulesListResponse(BaseModel):
    """알림 규칙 목록 응답"""
    rules: List[AlertRuleResponse]
    total: int


class AlertStats(BaseModel):
    """알림 통계"""
    total_alerts: int
    by_status: Dict[str, int]
    by_severity: Dict[str, int]
    by_type: Dict[str, int]
    active_rules: int
    alerts_today: int
    alerts_this_week: int


# ============================================================
# In-Memory Storage (Production에서는 DB 사용)
# ============================================================

# 임시 인메모리 저장소
_alert_rules: Dict[str, Dict[str, Any]] = {}
_alerts: Dict[str, Dict[str, Any]] = {}
_rule_counter = 0
_alert_counter = 0


def _generate_rule_id() -> str:
    global _rule_counter
    _rule_counter += 1
    return f"rule_{_rule_counter:06d}"


def _generate_alert_id() -> str:
    global _alert_counter
    _alert_counter += 1
    return f"alert_{_alert_counter:06d}"


# ============================================================
# API Endpoints - Alert Rules
# ============================================================

@router.post("/rules", response_model=AlertRuleResponse, status_code=201)
async def create_alert_rule(
    brand_id: str,
    rule: AlertRuleCreate,
):
    """
    알림 규칙 생성

    Args:
        brand_id: 브랜드 ID
        rule: 규칙 생성 요청

    Returns:
        생성된 알림 규칙
    """
    rule_id = _generate_rule_id()
    now = datetime.now()

    rule_data = {
        "id": rule_id,
        "brand_id": brand_id,
        "name": rule.name,
        "description": rule.description,
        "alert_type": rule.alert_type,
        "conditions": rule.conditions,
        "severity": rule.severity,
        "channels": rule.channels,
        "enabled": rule.enabled,
        "cooldown_minutes": rule.cooldown_minutes,
        "created_at": now,
        "updated_at": now,
        "last_triggered_at": None,
        "trigger_count": 0,
    }

    _alert_rules[rule_id] = rule_data

    logger.info(f"[AlertAPI] Created rule: {rule_id} for brand {brand_id}")

    return AlertRuleResponse(**rule_data)


@router.get("/rules", response_model=AlertRulesListResponse)
async def list_alert_rules(
    brand_id: str,
    enabled_only: bool = Query(False, description="활성화된 규칙만 조회"),
    alert_type: Optional[AlertType] = Query(None, description="알림 유형 필터"),
):
    """
    알림 규칙 목록 조회

    Args:
        brand_id: 브랜드 ID
        enabled_only: 활성화된 규칙만 조회
        alert_type: 알림 유형 필터

    Returns:
        알림 규칙 목록
    """
    rules = [
        r for r in _alert_rules.values()
        if r["brand_id"] == brand_id
    ]

    if enabled_only:
        rules = [r for r in rules if r["enabled"]]

    if alert_type:
        rules = [r for r in rules if r["alert_type"] == alert_type]

    return AlertRulesListResponse(
        rules=[AlertRuleResponse(**r) for r in rules],
        total=len(rules),
    )


@router.get("/rules/{rule_id}", response_model=AlertRuleResponse)
async def get_alert_rule(rule_id: str):
    """
    알림 규칙 상세 조회

    Args:
        rule_id: 규칙 ID

    Returns:
        알림 규칙 상세
    """
    if rule_id not in _alert_rules:
        raise HTTPException(status_code=404, detail="Rule not found")

    return AlertRuleResponse(**_alert_rules[rule_id])


@router.patch("/rules/{rule_id}", response_model=AlertRuleResponse)
async def update_alert_rule(rule_id: str, update: AlertRuleUpdate):
    """
    알림 규칙 업데이트

    Args:
        rule_id: 규칙 ID
        update: 업데이트 요청

    Returns:
        업데이트된 알림 규칙
    """
    if rule_id not in _alert_rules:
        raise HTTPException(status_code=404, detail="Rule not found")

    rule = _alert_rules[rule_id]

    # 업데이트 적용
    update_data = update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        if value is not None:
            rule[key] = value

    rule["updated_at"] = datetime.now()

    logger.info(f"[AlertAPI] Updated rule: {rule_id}")

    return AlertRuleResponse(**rule)


@router.delete("/rules/{rule_id}", status_code=204)
async def delete_alert_rule(rule_id: str):
    """
    알림 규칙 삭제

    Args:
        rule_id: 규칙 ID
    """
    if rule_id not in _alert_rules:
        raise HTTPException(status_code=404, detail="Rule not found")

    del _alert_rules[rule_id]

    logger.info(f"[AlertAPI] Deleted rule: {rule_id}")


@router.post("/rules/{rule_id}/toggle", response_model=AlertRuleResponse)
async def toggle_alert_rule(rule_id: str):
    """
    알림 규칙 활성화/비활성화 토글

    Args:
        rule_id: 규칙 ID

    Returns:
        업데이트된 알림 규칙
    """
    if rule_id not in _alert_rules:
        raise HTTPException(status_code=404, detail="Rule not found")

    rule = _alert_rules[rule_id]
    rule["enabled"] = not rule["enabled"]
    rule["updated_at"] = datetime.now()

    logger.info(f"[AlertAPI] Toggled rule: {rule_id} -> enabled={rule['enabled']}")

    return AlertRuleResponse(**rule)


# ============================================================
# API Endpoints - Alerts
# ============================================================

@router.post("", response_model=AlertResponse, status_code=201)
async def create_alert(
    brand_id: str,
    alert: AlertCreate,
    background_tasks: BackgroundTasks,
):
    """
    알림 수동 생성

    Args:
        brand_id: 브랜드 ID
        alert: 알림 생성 요청
        background_tasks: 백그라운드 태스크

    Returns:
        생성된 알림
    """
    alert_id = _generate_alert_id()
    now = datetime.now()

    alert_data = {
        "id": alert_id,
        "brand_id": brand_id,
        "rule_id": alert.rule_id,
        "alert_type": alert.alert_type,
        "title": alert.title,
        "message": alert.message,
        "severity": alert.severity,
        "status": AlertStatus.ACTIVE,
        "metadata": alert.metadata,
        "created_at": now,
        "acknowledged_at": None,
        "resolved_at": None,
        "acknowledged_by": None,
        "resolved_by": None,
    }

    _alerts[alert_id] = alert_data

    # 백그라운드에서 알림 전송
    background_tasks.add_task(_send_alert_notifications, alert_data)

    logger.info(f"[AlertAPI] Created alert: {alert_id} for brand {brand_id}")

    return AlertResponse(**alert_data)


@router.get("", response_model=AlertsListResponse)
async def list_alerts(
    brand_id: str,
    status: Optional[AlertStatus] = Query(None, description="상태 필터"),
    severity: Optional[AlertSeverity] = Query(None, description="심각도 필터"),
    alert_type: Optional[AlertType] = Query(None, description="유형 필터"),
    page: int = Query(1, ge=1, description="페이지 번호"),
    page_size: int = Query(20, ge=1, le=100, description="페이지 크기"),
):
    """
    알림 목록 조회

    Args:
        brand_id: 브랜드 ID
        status: 상태 필터
        severity: 심각도 필터
        alert_type: 유형 필터
        page: 페이지 번호
        page_size: 페이지 크기

    Returns:
        알림 목록
    """
    alerts = [
        a for a in _alerts.values()
        if a["brand_id"] == brand_id
    ]

    # 필터 적용
    if status:
        alerts = [a for a in alerts if a["status"] == status]
    if severity:
        alerts = [a for a in alerts if a["severity"] == severity]
    if alert_type:
        alerts = [a for a in alerts if a["alert_type"] == alert_type]

    # 정렬 (최신순)
    alerts.sort(key=lambda x: x["created_at"], reverse=True)

    # 페이지네이션
    total = len(alerts)
    start = (page - 1) * page_size
    end = start + page_size
    paginated = alerts[start:end]

    return AlertsListResponse(
        alerts=[AlertResponse(**a) for a in paginated],
        total=total,
        page=page,
        page_size=page_size,
        has_more=end < total,
    )


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(alert_id: str):
    """
    알림 상세 조회

    Args:
        alert_id: 알림 ID

    Returns:
        알림 상세
    """
    if alert_id not in _alerts:
        raise HTTPException(status_code=404, detail="Alert not found")

    return AlertResponse(**_alerts[alert_id])


@router.patch("/{alert_id}/status", response_model=AlertResponse)
async def update_alert_status(
    alert_id: str,
    update: AlertStatusUpdate,
    user_id: str = Query(..., description="사용자 ID"),
):
    """
    알림 상태 업데이트

    Args:
        alert_id: 알림 ID
        update: 상태 업데이트 요청
        user_id: 사용자 ID

    Returns:
        업데이트된 알림
    """
    if alert_id not in _alerts:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert = _alerts[alert_id]
    now = datetime.now()

    alert["status"] = update.status

    if update.status == AlertStatus.ACKNOWLEDGED:
        alert["acknowledged_at"] = now
        alert["acknowledged_by"] = user_id
    elif update.status == AlertStatus.RESOLVED:
        alert["resolved_at"] = now
        alert["resolved_by"] = user_id

    logger.info(f"[AlertAPI] Updated alert status: {alert_id} -> {update.status.value}")

    return AlertResponse(**alert)


@router.post("/{alert_id}/acknowledge", response_model=AlertResponse)
async def acknowledge_alert(
    alert_id: str,
    user_id: str = Query(..., description="사용자 ID"),
):
    """
    알림 확인 처리

    Args:
        alert_id: 알림 ID
        user_id: 사용자 ID

    Returns:
        업데이트된 알림
    """
    if alert_id not in _alerts:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert = _alerts[alert_id]
    alert["status"] = AlertStatus.ACKNOWLEDGED
    alert["acknowledged_at"] = datetime.now()
    alert["acknowledged_by"] = user_id

    logger.info(f"[AlertAPI] Acknowledged alert: {alert_id} by {user_id}")

    return AlertResponse(**alert)


@router.post("/{alert_id}/resolve", response_model=AlertResponse)
async def resolve_alert(
    alert_id: str,
    user_id: str = Query(..., description="사용자 ID"),
    comment: Optional[str] = Query(None, description="해결 코멘트"),
):
    """
    알림 해결 처리

    Args:
        alert_id: 알림 ID
        user_id: 사용자 ID
        comment: 해결 코멘트

    Returns:
        업데이트된 알림
    """
    if alert_id not in _alerts:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert = _alerts[alert_id]
    alert["status"] = AlertStatus.RESOLVED
    alert["resolved_at"] = datetime.now()
    alert["resolved_by"] = user_id

    if comment:
        if alert["metadata"] is None:
            alert["metadata"] = {}
        alert["metadata"]["resolve_comment"] = comment

    logger.info(f"[AlertAPI] Resolved alert: {alert_id} by {user_id}")

    return AlertResponse(**alert)


@router.delete("/{alert_id}", status_code=204)
async def delete_alert(alert_id: str):
    """
    알림 삭제

    Args:
        alert_id: 알림 ID
    """
    if alert_id not in _alerts:
        raise HTTPException(status_code=404, detail="Alert not found")

    del _alerts[alert_id]

    logger.info(f"[AlertAPI] Deleted alert: {alert_id}")


# ============================================================
# API Endpoints - Statistics
# ============================================================

@router.get("/stats", response_model=AlertStats)
async def get_alert_stats(brand_id: str):
    """
    알림 통계 조회

    Args:
        brand_id: 브랜드 ID

    Returns:
        알림 통계
    """
    from datetime import timedelta

    alerts = [
        a for a in _alerts.values()
        if a["brand_id"] == brand_id
    ]
    rules = [
        r for r in _alert_rules.values()
        if r["brand_id"] == brand_id
    ]

    now = datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=today_start.weekday())

    # 상태별 집계
    by_status = {}
    for status in AlertStatus:
        by_status[status.value] = len([a for a in alerts if a["status"] == status])

    # 심각도별 집계
    by_severity = {}
    for severity in AlertSeverity:
        by_severity[severity.value] = len([a for a in alerts if a["severity"] == severity])

    # 유형별 집계
    by_type = {}
    for alert_type in AlertType:
        by_type[alert_type.value] = len([a for a in alerts if a["alert_type"] == alert_type])

    # 오늘/이번 주 알림
    alerts_today = len([a for a in alerts if a["created_at"] >= today_start])
    alerts_this_week = len([a for a in alerts if a["created_at"] >= week_start])

    return AlertStats(
        total_alerts=len(alerts),
        by_status=by_status,
        by_severity=by_severity,
        by_type=by_type,
        active_rules=len([r for r in rules if r["enabled"]]),
        alerts_today=alerts_today,
        alerts_this_week=alerts_this_week,
    )


# ============================================================
# Background Tasks
# ============================================================

async def _send_alert_notifications(alert_data: Dict[str, Any]):
    """
    알림 전송 (백그라운드)

    Args:
        alert_data: 알림 데이터
    """
    try:
        rule_id = alert_data.get("rule_id")

        if rule_id and rule_id in _alert_rules:
            rule = _alert_rules[rule_id]
            channels = rule.get("channels", [NotificationChannel.IN_APP])
        else:
            channels = [NotificationChannel.IN_APP]

        for channel in channels:
            if channel == NotificationChannel.IN_APP:
                logger.info(f"[AlertAPI] In-app notification sent for alert: {alert_data['id']}")
            elif channel == NotificationChannel.EMAIL:
                logger.info(f"[AlertAPI] Email notification queued for alert: {alert_data['id']}")
            elif channel == NotificationChannel.SLACK:
                logger.info(f"[AlertAPI] Slack notification queued for alert: {alert_data['id']}")
            elif channel == NotificationChannel.WEBHOOK:
                logger.info(f"[AlertAPI] Webhook notification queued for alert: {alert_data['id']}")

    except Exception as e:
        logger.error(f"[AlertAPI] Failed to send notifications: {e}")


# ============================================================
# Utility Functions
# ============================================================

def trigger_alert_from_rule(rule_id: str, trigger_data: Dict[str, Any]) -> Optional[str]:
    """
    규칙에 의한 알림 트리거 (내부 사용)

    Args:
        rule_id: 규칙 ID
        trigger_data: 트리거 데이터

    Returns:
        생성된 알림 ID 또는 None
    """
    if rule_id not in _alert_rules:
        return None

    rule = _alert_rules[rule_id]

    if not rule["enabled"]:
        return None

    # 쿨다운 체크
    if rule["last_triggered_at"]:
        from datetime import timedelta
        cooldown = timedelta(minutes=rule["cooldown_minutes"])
        if datetime.now() - rule["last_triggered_at"] < cooldown:
            logger.debug(f"[AlertAPI] Rule {rule_id} in cooldown period")
            return None

    # 알림 생성
    alert_id = _generate_alert_id()
    now = datetime.now()

    alert_data = {
        "id": alert_id,
        "brand_id": rule["brand_id"],
        "rule_id": rule_id,
        "alert_type": rule["alert_type"],
        "title": f"[{rule['severity'].value.upper()}] {rule['name']}",
        "message": f"규칙 '{rule['name']}'이 트리거되었습니다.",
        "severity": rule["severity"],
        "status": AlertStatus.ACTIVE,
        "metadata": trigger_data,
        "created_at": now,
        "acknowledged_at": None,
        "resolved_at": None,
        "acknowledged_by": None,
        "resolved_by": None,
    }

    _alerts[alert_id] = alert_data

    # 규칙 트리거 정보 업데이트
    rule["last_triggered_at"] = now
    rule["trigger_count"] += 1

    logger.info(f"[AlertAPI] Alert triggered from rule: {rule_id} -> {alert_id}")

    return alert_id
