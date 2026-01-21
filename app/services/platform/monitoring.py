"""
모니터링 서비스
- 통합 헬스체크
- 시스템 상태 모니터링
- 알림 및 리포팅
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
import threading
import time
import logging

logger = logging.getLogger(__name__)


# ============================================================
# Enums
# ============================================================

class HealthStatus(str, Enum):
    """헬스 상태"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

    @property
    def is_ok(self) -> bool:
        """정상 여부"""
        return self in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]


class ServiceType(str, Enum):
    """서비스 타입"""
    NEO4J = "neo4j"
    REDIS = "redis"
    LLM = "llm"
    VECTOR = "vector"
    ANALYTICS = "analytics"


class AlertLevel(str, Enum):
    """알림 레벨"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class ServiceHealth:
    """서비스 헬스 상태"""
    service: ServiceType
    status: HealthStatus
    latency_ms: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "service": self.service.value,
            "status": self.status.value,
            "latency_ms": round(self.latency_ms, 2),
            "message": self.message,
            "details": self.details,
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class SystemHealth:
    """시스템 전체 헬스"""
    status: HealthStatus
    services: List[ServiceHealth] = field(default_factory=list)
    total_latency_ms: float = 0.0
    checked_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "status": self.status.value,
            "services": {s.service.value: s.to_dict() for s in self.services},
            "total_latency_ms": round(self.total_latency_ms, 2),
            "healthy_count": sum(1 for s in self.services if s.status == HealthStatus.HEALTHY),
            "unhealthy_count": sum(1 for s in self.services if s.status == HealthStatus.UNHEALTHY),
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class Alert:
    """알림"""
    level: AlertLevel
    service: Optional[ServiceType]
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "level": self.level.value,
            "service": self.service.value if self.service else None,
            "message": self.message,
            "details": self.details,
            "created_at": self.created_at.isoformat(),
            "acknowledged": self.acknowledged,
        }


@dataclass
class MonitoringConfig:
    """모니터링 설정"""
    check_interval: int = 60  # seconds
    timeout: float = 5.0  # seconds per check
    max_alerts: int = 100
    enable_auto_check: bool = False

    @classmethod
    def from_env(cls) -> "MonitoringConfig":
        """환경변수에서 설정 로드"""
        import os
        return cls(
            check_interval=int(os.getenv("MONITORING_CHECK_INTERVAL", "60")),
            timeout=float(os.getenv("MONITORING_TIMEOUT", "5.0")),
            max_alerts=int(os.getenv("MONITORING_MAX_ALERTS", "100")),
            enable_auto_check=os.getenv("MONITORING_AUTO_CHECK", "false").lower() == "true",
        )


@dataclass
class MonitoringMetrics:
    """모니터링 메트릭"""
    total_checks: int = 0
    successful_checks: int = 0
    failed_checks: int = 0
    total_alerts: int = 0
    last_check_time: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

    def record_check(self, success: bool):
        """체크 결과 기록"""
        self.total_checks += 1
        self.last_check_time = datetime.now()
        if success:
            self.successful_checks += 1
        else:
            self.failed_checks += 1

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "total_checks": self.total_checks,
            "successful_checks": self.successful_checks,
            "failed_checks": self.failed_checks,
            "success_rate": round(self.successful_checks / self.total_checks, 4) if self.total_checks > 0 else 0.0,
            "total_alerts": self.total_alerts,
            "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
            "uptime_seconds": (datetime.now() - self.created_at).total_seconds(),
        }


# ============================================================
# Health Checkers
# ============================================================

class HealthChecker:
    """헬스 체커 베이스"""

    def __init__(self, service_type: ServiceType, timeout: float = 5.0):
        self.service_type = service_type
        self.timeout = timeout

    def check(self) -> ServiceHealth:
        """헬스 체크 수행"""
        raise NotImplementedError


class Neo4jHealthChecker(HealthChecker):
    """Neo4j 헬스 체커"""

    def __init__(self, timeout: float = 5.0):
        super().__init__(ServiceType.NEO4J, timeout)

    def check(self) -> ServiceHealth:
        """Neo4j 헬스 체크"""
        start = time.time()
        try:
            from app.services.shared.neo4j import get_neo4j_client

            neo4j = get_neo4j_client()
            health = neo4j.health_check()
            latency = (time.time() - start) * 1000

            status = HealthStatus(health.get("status", "unknown"))

            return ServiceHealth(
                service=self.service_type,
                status=status,
                latency_ms=latency,
                details=health,
            )

        except Exception as e:
            return ServiceHealth(
                service=self.service_type,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=str(e),
            )


class RedisHealthChecker(HealthChecker):
    """Redis 헬스 체커"""

    def __init__(self, timeout: float = 5.0):
        super().__init__(ServiceType.REDIS, timeout)

    def check(self) -> ServiceHealth:
        """Redis 헬스 체크"""
        start = time.time()
        try:
            from app.services.shared.cache import get_cache_client

            cache = get_cache_client()
            latency = (time.time() - start) * 1000

            if cache.available:
                cache.client.ping()
                return ServiceHealth(
                    service=self.service_type,
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency,
                    details={"available": True},
                )
            else:
                return ServiceHealth(
                    service=self.service_type,
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency,
                    message="Redis unavailable, using fallback",
                    details={"available": False},
                )

        except Exception as e:
            return ServiceHealth(
                service=self.service_type,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=str(e),
            )


class LLMHealthChecker(HealthChecker):
    """LLM 헬스 체커"""

    def __init__(self, timeout: float = 5.0):
        super().__init__(ServiceType.LLM, timeout)

    def check(self) -> ServiceHealth:
        """LLM 헬스 체크"""
        start = time.time()
        try:
            from app.services.shared.llm import get_llm_client

            llm = get_llm_client()
            health = llm.health_check()
            latency = (time.time() - start) * 1000

            status = HealthStatus(health.get("status", "unknown"))

            return ServiceHealth(
                service=self.service_type,
                status=status,
                latency_ms=latency,
                details=health,
            )

        except Exception as e:
            return ServiceHealth(
                service=self.service_type,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=str(e),
            )


class VectorHealthChecker(HealthChecker):
    """Vector 헬스 체커"""

    def __init__(self, timeout: float = 5.0):
        super().__init__(ServiceType.VECTOR, timeout)

    def check(self) -> ServiceHealth:
        """Vector 헬스 체크"""
        start = time.time()
        try:
            from app.services.shared.vector import get_vector_service

            vector = get_vector_service()
            health = vector.health_check()
            latency = (time.time() - start) * 1000

            status = HealthStatus(health.get("status", "unknown"))

            return ServiceHealth(
                service=self.service_type,
                status=status,
                latency_ms=latency,
                details=health,
            )

        except Exception as e:
            return ServiceHealth(
                service=self.service_type,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.time() - start) * 1000,
                message=str(e),
            )


# ============================================================
# Main Service
# ============================================================

class MonitoringService:
    """프로덕션 급 모니터링 서비스"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config: Optional[MonitoringConfig] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[MonitoringConfig] = None):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.config = config or MonitoringConfig.from_env()
        self.metrics = MonitoringMetrics()
        self.alerts: List[Alert] = []
        self._alerts_lock = threading.Lock()

        # 헬스 체커 등록
        self.checkers: Dict[ServiceType, HealthChecker] = {
            ServiceType.NEO4J: Neo4jHealthChecker(self.config.timeout),
            ServiceType.REDIS: RedisHealthChecker(self.config.timeout),
            ServiceType.LLM: LLMHealthChecker(self.config.timeout),
            ServiceType.VECTOR: VectorHealthChecker(self.config.timeout),
        }

        # 마지막 헬스 상태 캐시
        self._last_health: Optional[SystemHealth] = None

        self._initialized = True
        logger.info(f"MonitoringService initialized: check_interval={self.config.check_interval}s")

    # --------------------------------------------------
    # Health Check
    # --------------------------------------------------

    def health_check(self, services: List[ServiceType] = None) -> SystemHealth:
        """전체 헬스 체크"""
        services_to_check = services or list(self.checkers.keys())
        service_results: List[ServiceHealth] = []
        total_latency = 0.0

        for service_type in services_to_check:
            checker = self.checkers.get(service_type)
            if checker:
                result = checker.check()
                service_results.append(result)
                total_latency += result.latency_ms

                # 비정상 서비스 알림 생성
                if result.status == HealthStatus.UNHEALTHY:
                    self._create_alert(
                        AlertLevel.ERROR,
                        service_type,
                        f"Service unhealthy: {result.message or 'Unknown error'}",
                    )

        # 전체 상태 결정
        unhealthy_count = sum(1 for s in service_results if s.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for s in service_results if s.status == HealthStatus.DEGRADED)

        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        system_health = SystemHealth(
            status=overall_status,
            services=service_results,
            total_latency_ms=total_latency,
        )

        # 메트릭 기록
        self.metrics.record_check(overall_status.is_ok)

        # 캐시 업데이트
        self._last_health = system_health

        return system_health

    def check_service(self, service_type: ServiceType) -> ServiceHealth:
        """단일 서비스 헬스 체크"""
        checker = self.checkers.get(service_type)
        if not checker:
            return ServiceHealth(
                service=service_type,
                status=HealthStatus.UNKNOWN,
                message="No checker registered",
            )

        return checker.check()

    def get_last_health(self) -> Optional[SystemHealth]:
        """마지막 헬스 상태 조회"""
        return self._last_health

    # --------------------------------------------------
    # System Stats
    # --------------------------------------------------

    def get_system_stats(self) -> Dict[str, Any]:
        """시스템 통계"""
        stats = {}

        # LLM 통계
        try:
            from app.services.shared.llm import get_llm_client
            llm = get_llm_client()
            stats["llm"] = llm.get_metrics()
        except:
            stats["llm"] = {}

        # Cache 통계
        try:
            from app.services.shared.cache import get_cache_client
            cache = get_cache_client()
            stats["cache"] = cache.get_metrics()
        except:
            stats["cache"] = {}

        # Vector 통계
        try:
            from app.services.shared.vector import get_vector_service
            vector = get_vector_service()
            stats["vector"] = vector.get_metrics()
        except:
            stats["vector"] = {}

        # Analytics 통계
        try:
            from app.services.platform.analytics import get_analytics_service
            analytics = get_analytics_service()
            stats["analytics"] = analytics.get_metrics()
        except:
            stats["analytics"] = {}

        # Neo4j 통계
        try:
            from app.services.shared.neo4j import get_neo4j_client
            neo4j = get_neo4j_client()
            stats["neo4j"] = neo4j.get_metrics()
        except:
            stats["neo4j"] = {}

        return stats

    # --------------------------------------------------
    # Alerts
    # --------------------------------------------------

    def _create_alert(self, level: AlertLevel, service: Optional[ServiceType],
                      message: str, details: Dict = None):
        """알림 생성"""
        alert = Alert(
            level=level,
            service=service,
            message=message,
            details=details or {},
        )

        with self._alerts_lock:
            self.alerts.append(alert)
            self.metrics.total_alerts += 1

            # 최대 알림 수 제한
            if len(self.alerts) > self.config.max_alerts:
                self.alerts = self.alerts[-self.config.max_alerts:]

        logger.log(
            logging.ERROR if level in [AlertLevel.ERROR, AlertLevel.CRITICAL] else logging.WARNING,
            f"Alert [{level.value}]: {message}"
        )

    def get_alerts(self, level: AlertLevel = None,
                   service: ServiceType = None,
                   unacknowledged_only: bool = False) -> List[Alert]:
        """알림 조회"""
        with self._alerts_lock:
            alerts = self.alerts.copy()

        if level:
            alerts = [a for a in alerts if a.level == level]

        if service:
            alerts = [a for a in alerts if a.service == service]

        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        return alerts

    def acknowledge_alert(self, index: int) -> bool:
        """알림 확인 처리"""
        with self._alerts_lock:
            if 0 <= index < len(self.alerts):
                self.alerts[index].acknowledged = True
                return True
        return False

    def clear_alerts(self, acknowledged_only: bool = False):
        """알림 클리어"""
        with self._alerts_lock:
            if acknowledged_only:
                self.alerts = [a for a in self.alerts if not a.acknowledged]
            else:
                self.alerts.clear()

    # --------------------------------------------------
    # Checker Management
    # --------------------------------------------------

    def register_checker(self, checker: HealthChecker):
        """커스텀 헬스 체커 등록"""
        self.checkers[checker.service_type] = checker
        logger.info(f"Health checker registered: {checker.service_type.value}")

    def unregister_checker(self, service_type: ServiceType):
        """헬스 체커 해제"""
        if service_type in self.checkers:
            del self.checkers[service_type]

    # --------------------------------------------------
    # Health & Metrics
    # --------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """메트릭 조회"""
        return self.metrics.to_dict()

    def reset_metrics(self):
        """메트릭 리셋"""
        self.metrics = MonitoringMetrics()


# ============================================================
# Factory Functions
# ============================================================

@lru_cache()
def get_monitoring_service() -> MonitoringService:
    """싱글톤 모니터링 서비스"""
    return MonitoringService()


def create_monitoring_service(config: MonitoringConfig) -> MonitoringService:
    """커스텀 설정으로 모니터링 서비스 생성"""
    MonitoringService._instance = None
    return MonitoringService(config)


# ============================================================
# Convenience Functions
# ============================================================

def health_check() -> Dict[str, Any]:
    """헬스 체크 (편의 함수)"""
    return get_monitoring_service().health_check().to_dict()


def get_system_stats() -> Dict[str, Any]:
    """시스템 통계 (편의 함수)"""
    return get_monitoring_service().get_system_stats()


def get_alerts() -> List[Dict[str, Any]]:
    """알림 조회 (편의 함수)"""
    return [a.to_dict() for a in get_monitoring_service().get_alerts()]
