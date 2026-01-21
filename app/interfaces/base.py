"""
Base Interface - Production Grade v2.0
모든 컴포넌트의 기본 인터페이스 정의

Features:
    - 타입 안전한 구성
    - 라이프사이클 관리
    - 헬스체크 지원
    - 메트릭 추적
    - 구성 검증
    - 비동기 지원
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TypeVar, Generic, List, ClassVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
import logging
import time

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ============================================================
# Enums
# ============================================================

class ComponentStatus(str, Enum):
    """컴포넌트 상태"""
    INITIALIZED = "initialized"
    READY = "ready"
    RUNNING = "running"
    DEGRADED = "degraded"
    ERROR = "error"
    STOPPED = "stopped"


class ComponentType(str, Enum):
    """컴포넌트 타입"""
    RETRIEVER = "retriever"
    GENERATOR = "generator"
    VALIDATOR = "validator"
    PROCESSOR = "processor"
    ROUTER = "router"
    FILTER = "filter"


# ============================================================
# Metrics
# ============================================================

@dataclass
class ComponentMetrics:
    """컴포넌트 메트릭"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_latency_ms: float = 0.0
    last_call_time: Optional[datetime] = None
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None

    @property
    def average_latency_ms(self) -> float:
        """평균 지연 시간"""
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls

    @property
    def success_rate(self) -> float:
        """성공률"""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls

    @property
    def error_rate(self) -> float:
        """에러율"""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls

    def record_call(self, success: bool, latency_ms: float, error: str = None):
        """호출 기록"""
        self.total_calls += 1
        self.total_latency_ms += latency_ms
        self.last_call_time = datetime.now()

        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
            self.last_error = error
            self.last_error_time = datetime.now()

    def reset(self):
        """메트릭 초기화"""
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_latency_ms = 0.0
        self.last_call_time = None
        self.last_error = None
        self.last_error_time = None

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'total_calls': self.total_calls,
            'successful_calls': self.successful_calls,
            'failed_calls': self.failed_calls,
            'average_latency_ms': round(self.average_latency_ms, 2),
            'success_rate': round(self.success_rate, 4),
            'error_rate': round(self.error_rate, 4),
            'last_call_time': self.last_call_time.isoformat() if self.last_call_time else None,
            'last_error': self.last_error,
            'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None,
        }


# ============================================================
# Health Check
# ============================================================

@dataclass
class HealthCheckResult:
    """헬스체크 결과"""
    healthy: bool
    status: ComponentStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=datetime.now)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'healthy': self.healthy,
            'status': self.status.value,
            'message': self.message,
            'details': self.details,
            'checked_at': self.checked_at.isoformat(),
            'latency_ms': round(self.latency_ms, 2),
        }


# ============================================================
# Configuration
# ============================================================

@dataclass
class BaseConfig:
    """기본 설정"""
    enabled: bool = True
    timeout_ms: int = 30000
    retry_count: int = 2
    retry_delay_ms: int = 500
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseConfig':
        """딕셔너리에서 생성"""
        return cls(
            enabled=data.get('enabled', True),
            timeout_ms=data.get('timeout_ms', 30000),
            retry_count=data.get('retry_count', 2),
            retry_delay_ms=data.get('retry_delay_ms', 500),
            cache_enabled=data.get('cache_enabled', True),
            cache_ttl_seconds=data.get('cache_ttl_seconds', 300),
        )

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'enabled': self.enabled,
            'timeout_ms': self.timeout_ms,
            'retry_count': self.retry_count,
            'retry_delay_ms': self.retry_delay_ms,
            'cache_enabled': self.cache_enabled,
            'cache_ttl_seconds': self.cache_ttl_seconds,
        }


# ============================================================
# Base Interface
# ============================================================

class BaseInterface(ABC):
    """
    모든 컴포넌트의 프로덕션급 기본 인터페이스

    Features:
        - 라이프사이클 관리 (initialize, start, stop)
        - 헬스체크 지원
        - 메트릭 수집
        - 구성 검증
        - 스레드 안전

    Usage:
        class MyRetriever(BaseInterface):
            def __init__(self, brand_config):
                super().__init__(brand_config, ComponentType.RETRIEVER)

            def _validate_config(self) -> bool:
                return 'retrieval' in self.brand_config

            def _do_health_check(self) -> HealthCheckResult:
                return HealthCheckResult(healthy=True, status=ComponentStatus.READY)
    """

    # 클래스 레벨 레지스트리
    _registry: ClassVar[Dict[str, type]] = {}

    def __init__(
        self,
        brand_config: Dict[str, Any],
        component_type: ComponentType = ComponentType.PROCESSOR
    ):
        """
        Args:
            brand_config: 브랜드 설정 딕셔너리
            component_type: 컴포넌트 타입
        """
        # 기본 속성
        self.brand_config = brand_config
        self.component_type = component_type

        # 브랜드 정보 추출
        brand_info = brand_config.get('brand', {})
        self.brand_id = brand_info.get('id', 'unknown')
        self.brand_name = brand_info.get('name', 'Unknown')

        # 상태 관리
        self._status = ComponentStatus.INITIALIZED
        self._status_lock = threading.RLock()

        # 메트릭
        self._metrics = ComponentMetrics()
        self._metrics_lock = threading.Lock()

        # 초기화 시간
        self._created_at = datetime.now()
        self._started_at: Optional[datetime] = None

        # 설정 로드
        self._base_config = self._load_base_config()

        # 설정 검증
        if not self._validate_config():
            logger.warning(
                f"{self.__class__.__name__}: Config validation failed for brand {self.brand_id}"
            )

    def _load_base_config(self) -> BaseConfig:
        """기본 설정 로드"""
        config_key = self.component_type.value
        config_data = self.brand_config.get(config_key, {}).get('config', {})
        return BaseConfig.from_dict(config_data)

    # === Lifecycle Methods ===

    def initialize(self) -> bool:
        """
        컴포넌트 초기화

        Returns:
            초기화 성공 여부
        """
        try:
            with self._status_lock:
                if self._status != ComponentStatus.INITIALIZED:
                    logger.warning(f"Cannot initialize: current status is {self._status}")
                    return False

                # 서브클래스 초기화 로직
                success = self._do_initialize()

                if success:
                    self._status = ComponentStatus.READY
                    logger.info(f"{self.__class__.__name__} initialized for brand {self.brand_id}")
                else:
                    self._status = ComponentStatus.ERROR

                return success

        except Exception as e:
            logger.error(f"Initialization error: {e}", exc_info=True)
            self._set_error_status(str(e))
            return False

    def start(self) -> bool:
        """
        컴포넌트 시작

        Returns:
            시작 성공 여부
        """
        try:
            with self._status_lock:
                if self._status not in (ComponentStatus.READY, ComponentStatus.STOPPED):
                    logger.warning(f"Cannot start: current status is {self._status}")
                    return False

                success = self._do_start()

                if success:
                    self._status = ComponentStatus.RUNNING
                    self._started_at = datetime.now()
                    logger.info(f"{self.__class__.__name__} started")

                return success

        except Exception as e:
            logger.error(f"Start error: {e}", exc_info=True)
            self._set_error_status(str(e))
            return False

    def stop(self) -> bool:
        """
        컴포넌트 중지

        Returns:
            중지 성공 여부
        """
        try:
            with self._status_lock:
                if self._status == ComponentStatus.STOPPED:
                    return True

                success = self._do_stop()

                if success:
                    self._status = ComponentStatus.STOPPED
                    logger.info(f"{self.__class__.__name__} stopped")

                return success

        except Exception as e:
            logger.error(f"Stop error: {e}", exc_info=True)
            return False

    # === Health Check ===

    def health_check(self) -> HealthCheckResult:
        """
        헬스체크 수행

        Returns:
            헬스체크 결과
        """
        start_time = time.time()

        try:
            # 기본 상태 체크
            with self._status_lock:
                if self._status == ComponentStatus.ERROR:
                    return HealthCheckResult(
                        healthy=False,
                        status=self._status,
                        message="Component is in error state",
                        latency_ms=(time.time() - start_time) * 1000
                    )

                if self._status == ComponentStatus.STOPPED:
                    return HealthCheckResult(
                        healthy=False,
                        status=self._status,
                        message="Component is stopped",
                        latency_ms=(time.time() - start_time) * 1000
                    )

            # 서브클래스 헬스체크
            result = self._do_health_check()
            result.latency_ms = (time.time() - start_time) * 1000

            return result

        except Exception as e:
            logger.error(f"Health check error: {e}")
            return HealthCheckResult(
                healthy=False,
                status=ComponentStatus.ERROR,
                message=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )

    @property
    def is_healthy(self) -> bool:
        """건강 상태 (간단 체크)"""
        with self._status_lock:
            return self._status in (ComponentStatus.READY, ComponentStatus.RUNNING)

    @property
    def status(self) -> ComponentStatus:
        """현재 상태"""
        with self._status_lock:
            return self._status

    # === Metrics ===

    def get_metrics(self) -> Dict[str, Any]:
        """메트릭 조회"""
        with self._metrics_lock:
            return self._metrics.to_dict()

    def reset_metrics(self):
        """메트릭 초기화"""
        with self._metrics_lock:
            self._metrics.reset()

    def _record_call(self, success: bool, latency_ms: float, error: str = None):
        """호출 기록"""
        with self._metrics_lock:
            self._metrics.record_call(success, latency_ms, error)

    # === Configuration ===

    def get_config(self) -> Dict[str, Any]:
        """설정 조회"""
        return {
            'base': self._base_config.to_dict(),
            'component': self._get_component_config(),
        }

    def update_config(self, **kwargs) -> bool:
        """
        런타임 설정 업데이트

        Args:
            **kwargs: 업데이트할 설정값

        Returns:
            성공 여부
        """
        try:
            for key, value in kwargs.items():
                if hasattr(self._base_config, key):
                    setattr(self._base_config, key, value)
                    logger.debug(f"Config updated: {key}={value}")
            return True
        except Exception as e:
            logger.error(f"Config update error: {e}")
            return False

    # === Debug Info ===

    def get_debug_info(self) -> Dict[str, Any]:
        """디버그 정보"""
        with self._status_lock:
            status = self._status

        return {
            'class': self.__class__.__name__,
            'component_type': self.component_type.value,
            'brand_id': self.brand_id,
            'brand_name': self.brand_name,
            'status': status.value,
            'created_at': self._created_at.isoformat(),
            'started_at': self._started_at.isoformat() if self._started_at else None,
            'config': self.get_config(),
            'metrics': self.get_metrics(),
        }

    # === Registry ===

    @classmethod
    def register(cls, name: str):
        """
        컴포넌트 등록 데코레이터

        Usage:
            @BaseInterface.register("my_retriever")
            class MyRetriever(RetrieverInterface):
                pass
        """
        def decorator(component_cls):
            cls._registry[name] = component_cls
            return component_cls
        return decorator

    @classmethod
    def get_registered(cls, name: str) -> Optional[type]:
        """등록된 컴포넌트 조회"""
        return cls._registry.get(name)

    @classmethod
    def list_registered(cls) -> List[str]:
        """등록된 컴포넌트 목록"""
        return list(cls._registry.keys())

    # === Abstract/Override Methods ===

    def _validate_config(self) -> bool:
        """
        설정 검증 (오버라이드 가능)

        Returns:
            검증 성공 여부
        """
        return True

    def _do_initialize(self) -> bool:
        """
        초기화 로직 (오버라이드 가능)

        Returns:
            초기화 성공 여부
        """
        return True

    def _do_start(self) -> bool:
        """
        시작 로직 (오버라이드 가능)

        Returns:
            시작 성공 여부
        """
        return True

    def _do_stop(self) -> bool:
        """
        중지 로직 (오버라이드 가능)

        Returns:
            중지 성공 여부
        """
        return True

    def _do_health_check(self) -> HealthCheckResult:
        """
        헬스체크 로직 (오버라이드 가능)

        Returns:
            헬스체크 결과
        """
        return HealthCheckResult(
            healthy=True,
            status=self._status,
            message="OK"
        )

    def _get_component_config(self) -> Dict[str, Any]:
        """
        컴포넌트별 설정 조회 (오버라이드 가능)

        Returns:
            컴포넌트 설정
        """
        return {}

    # === Private Helpers ===

    def _set_error_status(self, error: str):
        """에러 상태 설정"""
        with self._status_lock:
            self._status = ComponentStatus.ERROR
        with self._metrics_lock:
            self._metrics.last_error = error
            self._metrics.last_error_time = datetime.now()

    # === Magic Methods ===

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"brand={self.brand_id}, "
            f"type={self.component_type.value}, "
            f"status={self._status.value})"
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}[{self.brand_id}]"


# ============================================================
# Context Manager Support
# ============================================================

class ManagedComponent(BaseInterface):
    """컨텍스트 매니저를 지원하는 컴포넌트"""

    def __enter__(self):
        """컨텍스트 진입"""
        self.initialize()
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 종료"""
        self.stop()
        return False


# ============================================================
# Exports
# ============================================================

__all__ = [
    'BaseInterface',
    'ManagedComponent',
    'ComponentStatus',
    'ComponentType',
    'ComponentMetrics',
    'HealthCheckResult',
    'BaseConfig',
]
