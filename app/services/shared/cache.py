"""
Redis Cache Service - Production Grade v2.0
캐시 관리 서비스

Features:
    - 다중 캐시 전략
    - TTL 관리
    - 자동 직렬화/역직렬화
    - 메트릭 추적
    - 헬스체크
    - 분산 락
    - 배치 연산
"""

import redis
import json
import hashlib
import threading
import time
from typing import Optional, Any, Dict, List, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
import os
import logging

logger = logging.getLogger(__name__)


# ============================================================
# Enums & Types
# ============================================================

class CacheStrategy(str, Enum):
    """캐시 전략"""
    WRITE_THROUGH = "write_through"     # 즉시 쓰기
    WRITE_BEHIND = "write_behind"       # 지연 쓰기
    READ_THROUGH = "read_through"       # 읽기 시 로드
    CACHE_ASIDE = "cache_aside"         # 캐시 우선


class SerializationType(str, Enum):
    """직렬화 타입"""
    JSON = "json"
    STRING = "string"
    PICKLE = "pickle"


@dataclass
class CacheConfig:
    """캐시 설정"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    default_ttl: int = 3600  # 1시간
    max_connections: int = 50
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    serialization: SerializationType = SerializationType.JSON

    @classmethod
    def from_env(cls) -> 'CacheConfig':
        """환경변수에서 로드"""
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

        # URL 파싱
        host = "localhost"
        port = 6379

        if redis_url.startswith("redis://"):
            url_part = redis_url.replace("redis://", "")
            if "@" in url_part:
                url_part = url_part.split("@")[1]
            if ":" in url_part:
                parts = url_part.split(":")
                host = parts[0]
                port = int(parts[1].split("/")[0]) if "/" in parts[1] else int(parts[1])

        return cls(
            host=host,
            port=port,
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
            default_ttl=int(os.getenv("CACHE_TTL", "3600")),
        )


@dataclass
class CacheMetrics:
    """캐시 메트릭"""
    hits: int = 0
    misses: int = 0
    writes: int = 0
    deletes: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    start_time: float = field(default_factory=time.time)

    @property
    def hit_rate(self) -> float:
        """캐시 히트율"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        """평균 레이턴시"""
        total_ops = self.hits + self.misses + self.writes
        return (self.total_latency_ms / total_ops) if total_ops > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'hits': self.hits,
            'misses': self.misses,
            'writes': self.writes,
            'deletes': self.deletes,
            'errors': self.errors,
            'hit_rate': round(self.hit_rate, 2),
            'avg_latency_ms': round(self.avg_latency_ms, 2),
            'uptime_seconds': round(time.time() - self.start_time, 0),
        }


@dataclass
class CacheEntry:
    """캐시 엔트리"""
    key: str
    value: Any
    ttl: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Serializer
# ============================================================

class CacheSerializer:
    """캐시 직렬화"""

    @staticmethod
    def serialize(value: Any, stype: SerializationType = SerializationType.JSON) -> str:
        """직렬화"""
        if stype == SerializationType.JSON:
            return json.dumps(value, ensure_ascii=False, default=str)
        elif stype == SerializationType.STRING:
            return str(value)
        elif stype == SerializationType.PICKLE:
            import pickle
            import base64
            return base64.b64encode(pickle.dumps(value)).decode('utf-8')
        return str(value)

    @staticmethod
    def deserialize(data: str, stype: SerializationType = SerializationType.JSON) -> Any:
        """역직렬화"""
        if not data:
            return None

        if stype == SerializationType.JSON:
            return json.loads(data)
        elif stype == SerializationType.STRING:
            return data
        elif stype == SerializationType.PICKLE:
            import pickle
            import base64
            return pickle.loads(base64.b64decode(data.encode('utf-8')))
        return data


# ============================================================
# Distributed Lock
# ============================================================

class DistributedLock:
    """분산 락"""

    def __init__(self, client: redis.Redis, name: str, timeout: int = 10):
        self.client = client
        self.name = f"lock:{name}"
        self.timeout = timeout
        self._token: Optional[str] = None

    def acquire(self, blocking: bool = True, blocking_timeout: int = None) -> bool:
        """락 획득"""
        import uuid
        self._token = str(uuid.uuid4())

        if blocking:
            end_time = time.time() + (blocking_timeout or self.timeout)
            while time.time() < end_time:
                if self.client.set(self.name, self._token, nx=True, ex=self.timeout):
                    return True
                time.sleep(0.1)
            return False
        else:
            return bool(self.client.set(self.name, self._token, nx=True, ex=self.timeout))

    def release(self) -> bool:
        """락 해제"""
        if self._token is None:
            return False

        # Lua 스크립트로 원자적 해제
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        result = self.client.eval(script, 1, self.name, self._token)
        self._token = None
        return bool(result)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


# ============================================================
# Cache Client
# ============================================================

class CacheClient:
    """
    Redis 캐시 클라이언트 - Production Grade

    Features:
        - 싱글톤 패턴
        - 자동 직렬화
        - 메트릭 추적
        - 헬스체크
        - 분산 락
        - 배치 연산
        - 폴백 처리
    """

    _instance: Optional['CacheClient'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'initialized'):
            return

        self.config = CacheConfig.from_env()
        self.metrics = CacheMetrics()
        self.client: Optional[redis.Redis] = None
        self.available = False

        self._connect()
        self.initialized = True

    def _connect(self):
        """Redis 연결"""
        try:
            self.client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                decode_responses=True,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                max_connections=self.config.max_connections,
            )

            # 연결 테스트
            self.client.ping()
            self.available = True
            logger.info(f"Redis connected: {self.config.host}:{self.config.port}")

        except Exception as e:
            logger.warning(f"Redis unavailable: {e}")
            self.client = None
            self.available = False

    def _make_key(self, prefix: str, data: Any) -> str:
        """캐시 키 생성"""
        if isinstance(data, str):
            hash_input = data
        else:
            hash_input = json.dumps(data, sort_keys=True, default=str)

        hash_str = hashlib.md5(hash_input.encode()).hexdigest()[:16]
        return f"{prefix}:{hash_str}"

    def get(
        self,
        key: str,
        prefix: str = "cache",
        deserialize: bool = True
    ) -> Optional[Any]:
        """
        캐시 조회

        Args:
            key: 캐시 키 또는 키 데이터
            prefix: 키 접두사
            deserialize: 역직렬화 여부

        Returns:
            캐시된 값 또는 None
        """
        if not self.available:
            self.metrics.misses += 1
            return None

        start = time.time()
        full_key = self._make_key(prefix, key)

        try:
            value = self.client.get(full_key)

            latency = (time.time() - start) * 1000
            self.metrics.total_latency_ms += latency

            if value is None:
                self.metrics.misses += 1
                return None

            self.metrics.hits += 1

            if deserialize:
                return CacheSerializer.deserialize(value, self.config.serialization)
            return value

        except Exception as e:
            self.metrics.errors += 1
            logger.warning(f"Cache get error: {e}")
            return None

    def set(
        self,
        key: str,
        value: Any,
        prefix: str = "cache",
        ttl: Optional[int] = None,
        serialize: bool = True
    ) -> bool:
        """
        캐시 저장

        Args:
            key: 캐시 키 또는 키 데이터
            value: 저장할 값
            prefix: 키 접두사
            ttl: TTL (초)
            serialize: 직렬화 여부

        Returns:
            성공 여부
        """
        if not self.available:
            return False

        start = time.time()
        full_key = self._make_key(prefix, key)

        try:
            if serialize:
                value = CacheSerializer.serialize(value, self.config.serialization)

            self.client.setex(
                full_key,
                ttl or self.config.default_ttl,
                value
            )

            latency = (time.time() - start) * 1000
            self.metrics.total_latency_ms += latency
            self.metrics.writes += 1

            return True

        except Exception as e:
            self.metrics.errors += 1
            logger.warning(f"Cache set error: {e}")
            return False

    def delete(self, key: str, prefix: str = "cache") -> bool:
        """캐시 삭제"""
        if not self.available:
            return False

        try:
            full_key = self._make_key(prefix, key)
            result = self.client.delete(full_key)
            self.metrics.deletes += 1
            return bool(result)

        except Exception as e:
            self.metrics.errors += 1
            logger.warning(f"Cache delete error: {e}")
            return False

    def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        prefix: str = "cache",
        ttl: Optional[int] = None
    ) -> Any:
        """
        캐시에서 가져오거나 없으면 생성하여 저장

        Args:
            key: 캐시 키
            factory: 값 생성 함수
            prefix: 키 접두사
            ttl: TTL

        Returns:
            캐시된 값 또는 새로 생성된 값
        """
        value = self.get(key, prefix)

        if value is not None:
            return value

        # 새로 생성
        value = factory()
        self.set(key, value, prefix, ttl)
        return value

    def mget(self, keys: List[str], prefix: str = "cache") -> Dict[str, Any]:
        """
        다중 키 조회

        Args:
            keys: 키 리스트
            prefix: 키 접두사

        Returns:
            키-값 딕셔너리
        """
        if not self.available or not keys:
            return {}

        try:
            full_keys = [self._make_key(prefix, k) for k in keys]
            values = self.client.mget(full_keys)

            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = CacheSerializer.deserialize(value, self.config.serialization)
                    self.metrics.hits += 1
                else:
                    self.metrics.misses += 1

            return result

        except Exception as e:
            self.metrics.errors += 1
            logger.warning(f"Cache mget error: {e}")
            return {}

    def mset(
        self,
        items: Dict[str, Any],
        prefix: str = "cache",
        ttl: Optional[int] = None
    ) -> bool:
        """
        다중 키 저장

        Args:
            items: 키-값 딕셔너리
            prefix: 키 접두사
            ttl: TTL

        Returns:
            성공 여부
        """
        if not self.available or not items:
            return False

        try:
            pipeline = self.client.pipeline()

            for key, value in items.items():
                full_key = self._make_key(prefix, key)
                serialized = CacheSerializer.serialize(value, self.config.serialization)
                pipeline.setex(full_key, ttl or self.config.default_ttl, serialized)

            pipeline.execute()
            self.metrics.writes += len(items)
            return True

        except Exception as e:
            self.metrics.errors += 1
            logger.warning(f"Cache mset error: {e}")
            return False

    def flush(self, pattern: Optional[str] = None) -> int:
        """
        캐시 플러시

        Args:
            pattern: 삭제할 키 패턴 (None이면 전체)

        Returns:
            삭제된 키 수
        """
        if not self.available:
            return 0

        try:
            if pattern:
                deleted = 0
                for key in self.client.scan_iter(match=pattern):
                    self.client.delete(key)
                    deleted += 1
                return deleted
            else:
                self.client.flushdb()
                logger.info("Cache flushed")
                return -1  # 전체 삭제

        except Exception as e:
            self.metrics.errors += 1
            logger.error(f"Cache flush error: {e}")
            return 0

    def lock(self, name: str, timeout: int = 10) -> DistributedLock:
        """분산 락 생성"""
        if not self.available:
            raise RuntimeError("Redis not available for locking")
        return DistributedLock(self.client, name, timeout)

    def incr(self, key: str, prefix: str = "counter", amount: int = 1) -> int:
        """카운터 증가"""
        if not self.available:
            return 0

        try:
            full_key = self._make_key(prefix, key)
            return self.client.incrby(full_key, amount)
        except Exception as e:
            self.metrics.errors += 1
            logger.warning(f"Cache incr error: {e}")
            return 0

    def expire(self, key: str, prefix: str = "cache", ttl: int = None) -> bool:
        """TTL 설정"""
        if not self.available:
            return False

        try:
            full_key = self._make_key(prefix, key)
            return bool(self.client.expire(full_key, ttl or self.config.default_ttl))
        except Exception as e:
            self.metrics.errors += 1
            return False

    def ttl(self, key: str, prefix: str = "cache") -> int:
        """남은 TTL 조회"""
        if not self.available:
            return -2

        try:
            full_key = self._make_key(prefix, key)
            return self.client.ttl(full_key)
        except Exception as e:
            self.metrics.errors += 1
            return -2

    def health_check(self) -> Dict[str, Any]:
        """헬스체크"""
        if not self.available:
            return {
                'status': 'unavailable',
                'metrics': self.metrics.to_dict()
            }

        try:
            start = time.time()
            self.client.ping()
            latency = (time.time() - start) * 1000

            info = self.client.info('memory')

            return {
                'status': 'healthy',
                'latency_ms': round(latency, 2),
                'used_memory': info.get('used_memory_human', 'unknown'),
                'connected_clients': self.client.info('clients').get('connected_clients', 0),
                'metrics': self.metrics.to_dict()
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'metrics': self.metrics.to_dict()
            }

    def get_metrics(self) -> CacheMetrics:
        """메트릭 조회"""
        return self.metrics

    def reset_metrics(self):
        """메트릭 리셋"""
        self.metrics = CacheMetrics()

    def close(self):
        """연결 종료"""
        if self.client:
            self.client.close()
            logger.info("Redis connection closed")


# ============================================================
# Decorator
# ============================================================

def cached(
    prefix: str = "func",
    ttl: int = 3600,
    key_builder: Optional[Callable] = None
):
    """
    캐싱 데코레이터

    Args:
        prefix: 캐시 키 접두사
        ttl: TTL (초)
        key_builder: 커스텀 키 빌더

    Usage:
        @cached(prefix="user", ttl=300)
        def get_user(user_id: str):
            return fetch_user(user_id)
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache_client()

            # 키 생성
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{args}:{kwargs}"

            # 캐시 조회
            cached_value = cache.get(cache_key, prefix)
            if cached_value is not None:
                return cached_value

            # 함수 실행
            result = func(*args, **kwargs)

            # 캐시 저장
            cache.set(cache_key, result, prefix, ttl)

            return result

        return wrapper
    return decorator


# ============================================================
# Factory
# ============================================================

def get_cache_client() -> CacheClient:
    """싱글톤 캐시 클라이언트"""
    return CacheClient()
