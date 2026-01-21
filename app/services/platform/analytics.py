"""
분석 서비스
- 이벤트 추적 및 집계
- 사용량 모니터링
- 실시간 통계
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
from functools import lru_cache
import threading
import time
import logging

logger = logging.getLogger(__name__)


# ============================================================
# Enums
# ============================================================

class EventType(str, Enum):
    """이벤트 타입"""
    QUERY = "query"
    GENERATION = "generation"
    RETRIEVAL = "retrieval"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    ERROR = "error"
    API_CALL = "api_call"
    USER_ACTION = "user_action"

    @classmethod
    def from_string(cls, value: str) -> "EventType":
        """문자열에서 변환"""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.USER_ACTION  # 기본값


class AggregationType(str, Enum):
    """집계 타입"""
    COUNT = "count"
    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"


class TimeWindow(str, Enum):
    """시간 윈도우"""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class Event:
    """이벤트"""
    event_type: EventType
    brand_id: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = ""

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"{self.timestamp.strftime('%Y%m%d%H%M%S')}-{id(self)}"

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "event_id": self.event_id,
            "type": self.event_type.value,
            "brand_id": self.brand_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AnalyticsConfig:
    """분석 설정"""
    max_events: int = 10000
    retention_hours: int = 24
    aggregation_interval: int = 60  # seconds
    enable_realtime: bool = True

    @classmethod
    def from_env(cls) -> "AnalyticsConfig":
        """환경변수에서 설정 로드"""
        import os
        return cls(
            max_events=int(os.getenv("ANALYTICS_MAX_EVENTS", "10000")),
            retention_hours=int(os.getenv("ANALYTICS_RETENTION_HOURS", "24")),
            aggregation_interval=int(os.getenv("ANALYTICS_AGGREGATION_INTERVAL", "60")),
            enable_realtime=os.getenv("ANALYTICS_REALTIME", "true").lower() == "true",
        )


@dataclass
class AnalyticsMetrics:
    """분석 메트릭"""
    total_events: int = 0
    events_by_type: Dict[str, int] = field(default_factory=dict)
    events_by_brand: Dict[str, int] = field(default_factory=dict)
    errors_count: int = 0
    last_event_time: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

    def record_event(self, event: Event):
        """이벤트 기록"""
        self.total_events += 1
        self.last_event_time = event.timestamp

        # 타입별 집계
        event_type = event.event_type.value
        self.events_by_type[event_type] = self.events_by_type.get(event_type, 0) + 1

        # 브랜드별 집계
        self.events_by_brand[event.brand_id] = self.events_by_brand.get(event.brand_id, 0) + 1

        if event.event_type == EventType.ERROR:
            self.errors_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "total_events": self.total_events,
            "events_by_type": dict(self.events_by_type),
            "events_by_brand": dict(self.events_by_brand),
            "errors_count": self.errors_count,
            "error_rate": round(self.errors_count / self.total_events, 4) if self.total_events > 0 else 0.0,
            "last_event_time": self.last_event_time.isoformat() if self.last_event_time else None,
            "uptime_seconds": (datetime.now() - self.created_at).total_seconds(),
        }


# ============================================================
# Aggregators
# ============================================================

class TimeSeriesAggregator:
    """시계열 집계기"""

    def __init__(self, window: TimeWindow, max_points: int = 1000):
        self.window = window
        self.max_points = max_points
        self._data: Dict[str, List[tuple]] = defaultdict(list)
        self._lock = threading.Lock()

    def add(self, metric_name: str, value: float, timestamp: datetime = None):
        """데이터 추가"""
        ts = timestamp or datetime.now()
        bucket = self._get_bucket(ts)

        with self._lock:
            self._data[metric_name].append((bucket, value))

            # 최대 포인트 제한
            if len(self._data[metric_name]) > self.max_points:
                self._data[metric_name] = self._data[metric_name][-self.max_points:]

    def get_series(self, metric_name: str,
                   aggregation: AggregationType = AggregationType.SUM) -> List[Dict]:
        """시계열 데이터 조회"""
        with self._lock:
            data = self._data.get(metric_name, [])

        # 버킷별 집계
        buckets: Dict[str, List[float]] = defaultdict(list)
        for bucket, value in data:
            buckets[bucket].append(value)

        # 집계 함수 적용
        result = []
        for bucket, values in sorted(buckets.items()):
            agg_value = self._aggregate(values, aggregation)
            result.append({
                "timestamp": bucket,
                "value": agg_value,
                "count": len(values),
            })

        return result

    def _get_bucket(self, ts: datetime) -> str:
        """시간 버킷 생성"""
        if self.window == TimeWindow.MINUTE:
            return ts.strftime("%Y-%m-%d %H:%M")
        elif self.window == TimeWindow.HOUR:
            return ts.strftime("%Y-%m-%d %H:00")
        elif self.window == TimeWindow.DAY:
            return ts.strftime("%Y-%m-%d")
        elif self.window == TimeWindow.WEEK:
            return ts.strftime("%Y-W%W")
        return ts.strftime("%Y-%m-%d %H:%M")

    def _aggregate(self, values: List[float], aggregation: AggregationType) -> float:
        """집계 함수"""
        if not values:
            return 0.0

        if aggregation == AggregationType.COUNT:
            return float(len(values))
        elif aggregation == AggregationType.SUM:
            return sum(values)
        elif aggregation == AggregationType.AVERAGE:
            return sum(values) / len(values)
        elif aggregation == AggregationType.MIN:
            return min(values)
        elif aggregation == AggregationType.MAX:
            return max(values)

        return sum(values)


# ============================================================
# Event Handlers
# ============================================================

class EventHandler:
    """이벤트 핸들러 베이스"""

    def handle(self, event: Event):
        """이벤트 처리"""
        raise NotImplementedError


class LoggingEventHandler(EventHandler):
    """로깅 이벤트 핸들러"""

    def __init__(self, level: int = logging.DEBUG):
        self.level = level

    def handle(self, event: Event):
        """이벤트 로깅"""
        logger.log(
            self.level,
            f"Event: {event.event_type.value} | {event.brand_id} | {event.data}"
        )


class CallbackEventHandler(EventHandler):
    """콜백 이벤트 핸들러"""

    def __init__(self, callback: Callable[[Event], None]):
        self.callback = callback

    def handle(self, event: Event):
        """콜백 실행"""
        try:
            self.callback(event)
        except Exception as e:
            logger.error(f"Event callback error: {e}")


# ============================================================
# Main Service
# ============================================================

class AnalyticsService:
    """프로덕션 급 분석 서비스"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config: Optional[AnalyticsConfig] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[AnalyticsConfig] = None):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.config = config or AnalyticsConfig.from_env()
        self.events: List[Event] = []
        self.metrics = AnalyticsMetrics()
        self.handlers: List[EventHandler] = []
        self.aggregators: Dict[TimeWindow, TimeSeriesAggregator] = {
            TimeWindow.MINUTE: TimeSeriesAggregator(TimeWindow.MINUTE),
            TimeWindow.HOUR: TimeSeriesAggregator(TimeWindow.HOUR),
            TimeWindow.DAY: TimeSeriesAggregator(TimeWindow.DAY),
        }
        self._events_lock = threading.Lock()

        self._initialized = True
        logger.info(f"AnalyticsService initialized: max_events={self.config.max_events}")

    # --------------------------------------------------
    # Event Tracking
    # --------------------------------------------------

    def track_event(self, event_type: str, brand_id: str, data: Dict[str, Any] = None):
        """이벤트 추적"""
        # 이벤트 타입 변환
        try:
            etype = EventType(event_type.lower())
        except ValueError:
            etype = EventType.USER_ACTION

        event = Event(
            event_type=etype,
            brand_id=brand_id,
            data=data or {},
        )

        with self._events_lock:
            self.events.append(event)
            self.metrics.record_event(event)

            # 최대 이벤트 수 제한
            if len(self.events) > self.config.max_events:
                self.events = self.events[-self.config.max_events:]

        # 핸들러 실행
        for handler in self.handlers:
            handler.handle(event)

        # 시계열 집계
        for aggregator in self.aggregators.values():
            aggregator.add(f"events.{etype.value}", 1.0, event.timestamp)
            aggregator.add(f"brand.{brand_id}", 1.0, event.timestamp)

        logger.debug(f"Event tracked: {etype.value} | {brand_id}")

    def track(self, event_type: EventType, brand_id: str, **kwargs):
        """타입 안전한 이벤트 추적"""
        self.track_event(event_type.value, brand_id, kwargs)

    def track_query(self, brand_id: str, query: str, latency_ms: float = None):
        """쿼리 이벤트 추적"""
        self.track_event(EventType.QUERY.value, brand_id, {
            "query": query[:200],  # 최대 200자
            "latency_ms": latency_ms,
        })

    def track_generation(self, brand_id: str, gen_type: str,
                        tokens: int = None, cost: float = None):
        """생성 이벤트 추적"""
        self.track_event(EventType.GENERATION.value, brand_id, {
            "type": gen_type,
            "tokens": tokens,
            "cost": cost,
        })

    def track_error(self, brand_id: str, error: str, context: Dict = None):
        """에러 이벤트 추적"""
        self.track_event(EventType.ERROR.value, brand_id, {
            "error": error[:500],
            "context": context,
        })

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    def get_stats(self, brand_id: str = None) -> Dict[str, Any]:
        """통계 조회"""
        with self._events_lock:
            events = self.events.copy()

        if brand_id:
            events = [e for e in events if e.brand_id == brand_id]

        stats = {
            "total_events": len(events),
            "by_type": {},
            "by_brand": {},
            "recent_events": [],
        }

        for event in events:
            # 타입별
            etype = event.event_type.value
            stats["by_type"][etype] = stats["by_type"].get(etype, 0) + 1

            # 브랜드별
            stats["by_brand"][event.brand_id] = stats["by_brand"].get(event.brand_id, 0) + 1

        # 최근 이벤트 (최대 10개)
        stats["recent_events"] = [e.to_dict() for e in events[-10:]]

        return stats

    def get_time_series(self, metric: str, window: TimeWindow = TimeWindow.HOUR,
                        aggregation: AggregationType = AggregationType.COUNT) -> List[Dict]:
        """시계열 데이터 조회"""
        aggregator = self.aggregators.get(window)
        if not aggregator:
            return []

        return aggregator.get_series(metric, aggregation)

    def get_brand_stats(self, brand_id: str) -> Dict[str, Any]:
        """브랜드별 상세 통계"""
        with self._events_lock:
            events = [e for e in self.events if e.brand_id == brand_id]

        if not events:
            return {"brand_id": brand_id, "total_events": 0}

        # 타입별 집계
        by_type = {}
        total_latency = 0.0
        latency_count = 0

        for event in events:
            etype = event.event_type.value
            by_type[etype] = by_type.get(etype, 0) + 1

            # 레이턴시 평균
            if "latency_ms" in event.data and event.data["latency_ms"]:
                total_latency += event.data["latency_ms"]
                latency_count += 1

        return {
            "brand_id": brand_id,
            "total_events": len(events),
            "by_type": by_type,
            "avg_latency_ms": round(total_latency / latency_count, 2) if latency_count > 0 else None,
            "error_rate": round(by_type.get("error", 0) / len(events), 4) if events else 0.0,
            "first_event": events[0].timestamp.isoformat(),
            "last_event": events[-1].timestamp.isoformat(),
        }

    # --------------------------------------------------
    # Handler Management
    # --------------------------------------------------

    def add_handler(self, handler: EventHandler):
        """핸들러 추가"""
        self.handlers.append(handler)
        logger.debug(f"Event handler added: {type(handler).__name__}")

    def remove_handler(self, handler: EventHandler):
        """핸들러 제거"""
        if handler in self.handlers:
            self.handlers.remove(handler)

    def add_callback(self, callback: Callable[[Event], None]):
        """콜백 핸들러 추가"""
        self.add_handler(CallbackEventHandler(callback))

    # --------------------------------------------------
    # Cleanup & Maintenance
    # --------------------------------------------------

    def cleanup_old_events(self, hours: int = None):
        """오래된 이벤트 정리"""
        hours = hours or self.config.retention_hours
        cutoff = datetime.now() - timedelta(hours=hours)

        with self._events_lock:
            original_count = len(self.events)
            self.events = [e for e in self.events if e.timestamp > cutoff]
            removed = original_count - len(self.events)

        if removed > 0:
            logger.info(f"Cleaned up {removed} old events")

        return removed

    def clear_events(self):
        """모든 이벤트 삭제"""
        with self._events_lock:
            self.events.clear()
        logger.info("All events cleared")

    # --------------------------------------------------
    # Health & Metrics
    # --------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """헬스 체크"""
        return {
            "status": "healthy",
            "event_count": len(self.events),
            "max_events": self.config.max_events,
            "utilization": round(len(self.events) / self.config.max_events, 4),
            "handler_count": len(self.handlers),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """메트릭 조회"""
        return self.metrics.to_dict()

    def reset_metrics(self):
        """메트릭 리셋"""
        self.metrics = AnalyticsMetrics()
        logger.info("Analytics metrics reset")

    # --------------------------------------------------
    # Admin API Methods
    # --------------------------------------------------

    def get_analytics(self, brand_id: str = None, days: int = 7) -> Dict[str, Any]:
        """
        분석 데이터 조회 (Admin API용)

        Args:
            brand_id: 브랜드 ID (선택)
            days: 분석 기간

        Returns:
            분석 데이터 딕셔너리
        """
        cutoff = datetime.now() - timedelta(days=days)

        with self._events_lock:
            events = self.events.copy()

        # 브랜드 필터링
        if brand_id:
            events = [e for e in events if e.brand_id == brand_id]

        # 기간 필터링
        events = [e for e in events if e.timestamp > cutoff]

        if not events:
            return {
                "total_messages": 0,
                "unique_sessions": 0,
                "avg_response_time": 0,
                "feature_usage": {},
                "hourly_distribution": [0] * 24
            }

        # 총 메시지 수
        total_messages = len([e for e in events if e.event_type == EventType.QUERY])

        # 고유 세션 (브랜드별)
        unique_sessions = len(set(e.brand_id for e in events))

        # 평균 응답 시간
        latencies = [
            e.data.get("latency_ms", 0)
            for e in events
            if e.event_type == EventType.QUERY and e.data.get("latency_ms")
        ]
        avg_response_time = sum(latencies) / len(latencies) / 1000 if latencies else 0

        # 기능별 사용량
        feature_usage = {}
        for event in events:
            feature = event.data.get("feature") or event.data.get("type") or event.event_type.value
            feature_usage[feature] = feature_usage.get(feature, 0) + 1

        # 시간대별 분포
        hourly_distribution = [0] * 24
        for event in events:
            hour = event.timestamp.hour
            hourly_distribution[hour] += 1

        return {
            "total_messages": total_messages,
            "unique_sessions": unique_sessions,
            "avg_response_time": round(avg_response_time, 2),
            "feature_usage": feature_usage,
            "hourly_distribution": hourly_distribution
        }

    def get_realtime_metrics(self) -> Dict[str, Any]:
        """
        실시간 메트릭 조회 (Admin API용)

        Returns:
            실시간 메트릭 딕셔너리
        """
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)

        with self._events_lock:
            events = self.events.copy()

        # 최근 1분간 이벤트
        recent_events = [e for e in events if e.timestamp > one_minute_ago]

        # 분당 요청 수
        requests_per_minute = len([e for e in recent_events if e.event_type == EventType.QUERY])

        # 활성 세션 (최근 5분)
        five_minutes_ago = now - timedelta(minutes=5)
        active_events = [e for e in events if e.timestamp > five_minutes_ago]
        active_sessions = len(set(e.brand_id for e in active_events))

        # 평균 응답 시간 (최근 1분)
        latencies = [
            e.data.get("latency_ms", 0)
            for e in recent_events
            if e.event_type == EventType.QUERY and e.data.get("latency_ms")
        ]
        avg_response_time_ms = sum(latencies) / len(latencies) if latencies else 0

        # 에러율 (최근 1분)
        error_count = len([e for e in recent_events if e.event_type == EventType.ERROR])
        total_count = len(recent_events) or 1
        error_rate = round(error_count / total_count, 4)

        # 캐시 히트율
        cache_hits = len([e for e in recent_events if e.event_type == EventType.CACHE_HIT])
        cache_misses = len([e for e in recent_events if e.event_type == EventType.CACHE_MISS])
        cache_total = cache_hits + cache_misses
        cache_hit_rate = round(cache_hits / cache_total, 4) if cache_total > 0 else 0

        return {
            "requests_per_minute": requests_per_minute,
            "active_sessions": active_sessions,
            "avg_response_time_ms": round(avg_response_time_ms, 2),
            "error_rate": error_rate,
            "cache_hit_rate": cache_hit_rate
        }


# ============================================================
# Factory Functions
# ============================================================

@lru_cache()
def get_analytics_service() -> AnalyticsService:
    """싱글톤 분석 서비스"""
    return AnalyticsService()


def create_analytics_service(config: AnalyticsConfig) -> AnalyticsService:
    """커스텀 설정으로 분석 서비스 생성"""
    AnalyticsService._instance = None
    return AnalyticsService(config)


# ============================================================
# Convenience Functions
# ============================================================

def track_event(event_type: str, brand_id: str, data: Dict = None):
    """이벤트 추적 (편의 함수)"""
    get_analytics_service().track_event(event_type, brand_id, data)


def track_query(brand_id: str, query: str, latency_ms: float = None):
    """쿼리 추적 (편의 함수)"""
    get_analytics_service().track_query(brand_id, query, latency_ms)


def track_error(brand_id: str, error: str, context: Dict = None):
    """에러 추적 (편의 함수)"""
    get_analytics_service().track_error(brand_id, error, context)
