"""
Retriever Interface - Production Grade v2.0
검색기 인터페이스 정의

Features:
    - 타입 안전한 검색 결과
    - 검색 설정 관리
    - 캐싱 지원
    - 비동기 검색
    - 검색 품질 메트릭
    - 배치 검색 지원
"""

from abc import abstractmethod
from typing import Dict, List, Any, Optional, Union, TypeVar, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import hashlib
import time
import logging

from app.interfaces.base import (
    BaseInterface,
    ComponentType,
    ComponentStatus,
    HealthCheckResult,
)
from app.core.context import QueryContext

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ============================================================
# Enums
# ============================================================

class RetrievalStatus(str, Enum):
    """검색 상태"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CACHED = "cached"
    FAILED = "failed"
    TIMEOUT = "timeout"
    NO_RESULTS = "no_results"


class RetrievalSource(str, Enum):
    """검색 소스 타입"""
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    PRODUCT = "product"
    STATS = "stats"
    CUSTOM = "custom"


# ============================================================
# Data Classes
# ============================================================

@dataclass
class RetrievalConfig:
    """검색 설정"""
    # 결과 설정
    top_k: int = 10
    min_score: float = 0.0
    max_results: int = 50

    # 타임아웃
    timeout_ms: int = 10000

    # 캐싱
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300

    # 필터링
    enable_dedup: bool = True
    enable_reranking: bool = False

    # 소스 설정
    source_weights: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrievalConfig':
        """딕셔너리에서 생성"""
        return cls(
            top_k=data.get('top_k', 10),
            min_score=data.get('min_score', 0.0),
            max_results=data.get('max_results', 50),
            timeout_ms=data.get('timeout_ms', 10000),
            cache_enabled=data.get('cache_enabled', True),
            cache_ttl_seconds=data.get('cache_ttl_seconds', 300),
            enable_dedup=data.get('enable_dedup', True),
            enable_reranking=data.get('enable_reranking', False),
            source_weights=data.get('source_weights', {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'top_k': self.top_k,
            'min_score': self.min_score,
            'max_results': self.max_results,
            'timeout_ms': self.timeout_ms,
            'cache_enabled': self.cache_enabled,
            'cache_ttl_seconds': self.cache_ttl_seconds,
            'enable_dedup': self.enable_dedup,
            'enable_reranking': self.enable_reranking,
            'source_weights': self.source_weights,
        }


@dataclass
class RetrievalItem:
    """검색 결과 항목"""
    id: str
    content: str
    score: float
    source: str
    node_type: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    rank: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'id': self.id,
            'content': self.content,
            'score': round(self.score, 4),
            'source': self.source,
            'node_type': self.node_type,
            'metadata': self.metadata,
            'rank': self.rank,
        }


@dataclass
class RetrievalResult:
    """검색 결과"""
    source: str
    items: List[RetrievalItem] = field(default_factory=list)
    status: RetrievalStatus = RetrievalStatus.COMPLETED
    total_count: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.total_count == 0:
            self.total_count = len(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    @property
    def is_success(self) -> bool:
        """성공 여부"""
        return self.status in (RetrievalStatus.COMPLETED, RetrievalStatus.CACHED)

    @property
    def top_score(self) -> float:
        """최고 점수"""
        if not self.items:
            return 0.0
        return max(item.score for item in self.items)

    @property
    def average_score(self) -> float:
        """평균 점수"""
        if not self.items:
            return 0.0
        return sum(item.score for item in self.items) / len(self.items)

    def get_top_k(self, k: int) -> List[RetrievalItem]:
        """상위 k개 항목"""
        return self.items[:k]

    def filter_by_score(self, min_score: float) -> 'RetrievalResult':
        """점수 필터링"""
        filtered = [item for item in self.items if item.score >= min_score]
        return RetrievalResult(
            source=self.source,
            items=filtered,
            status=self.status,
            total_count=len(filtered),
            latency_ms=self.latency_ms,
            cached=self.cached,
            metadata=self.metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'source': self.source,
            'items': [item.to_dict() for item in self.items],
            'status': self.status.value,
            'total_count': self.total_count,
            'latency_ms': round(self.latency_ms, 2),
            'cached': self.cached,
            'error': self.error,
            'top_score': round(self.top_score, 4),
            'average_score': round(self.average_score, 4),
        }

    def to_context_format(self) -> Dict[str, Any]:
        """QueryContext에 추가할 형식으로 변환"""
        return {
            'source': self.source,
            'data': [item.to_dict() for item in self.items],
            'score': self.average_score,
            'metadata': {
                **self.metadata,
                'total_count': self.total_count,
                'latency_ms': self.latency_ms,
                'cached': self.cached,
            }
        }


@dataclass
class RetrievalMetrics:
    """검색 메트릭"""
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_items_retrieved: int = 0
    average_latency_ms: float = 0.0
    average_result_count: float = 0.0
    error_count: int = 0
    timeout_count: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """캐시 히트율"""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    def record_query(
        self,
        result_count: int,
        latency_ms: float,
        cached: bool,
        error: bool = False,
        timeout: bool = False
    ):
        """쿼리 기록"""
        self.total_queries += 1
        self.total_items_retrieved += result_count

        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        if error:
            self.error_count += 1
        if timeout:
            self.timeout_count += 1

        # 이동 평균
        n = self.total_queries
        self.average_latency_ms = (
            (self.average_latency_ms * (n - 1) + latency_ms) / n
        )
        self.average_result_count = (
            (self.average_result_count * (n - 1) + result_count) / n
        )

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'total_queries': self.total_queries,
            'cache_hit_rate': round(self.cache_hit_rate, 4),
            'average_latency_ms': round(self.average_latency_ms, 2),
            'average_result_count': round(self.average_result_count, 2),
            'total_items_retrieved': self.total_items_retrieved,
            'error_count': self.error_count,
            'timeout_count': self.timeout_count,
        }


# ============================================================
# Cache
# ============================================================

class RetrievalCache:
    """검색 결과 캐시"""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: Dict[str, tuple] = {}  # key -> (result, timestamp)
        self._access_order: List[str] = []

    def get(self, key: str) -> Optional[RetrievalResult]:
        """캐시 조회"""
        if key not in self._cache:
            return None

        result, timestamp = self._cache[key]

        # TTL 체크
        if time.time() - timestamp > self.ttl_seconds:
            self._remove(key)
            return None

        # LRU 업데이트
        self._access_order.remove(key)
        self._access_order.append(key)

        return result

    def set(self, key: str, result: RetrievalResult):
        """캐시 저장"""
        # 크기 제한
        while len(self._cache) >= self.max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = (result, time.time())
        self._access_order.append(key)

    def _remove(self, key: str):
        """캐시 항목 제거"""
        if key in self._cache:
            del self._cache[key]
            self._access_order.remove(key)

    def clear(self):
        """캐시 초기화"""
        self._cache.clear()
        self._access_order.clear()

    def generate_key(self, query: str, config: RetrievalConfig) -> str:
        """캐시 키 생성"""
        key_data = f"{query}:{config.top_k}:{config.min_score}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]

    @property
    def size(self) -> int:
        """현재 캐시 크기"""
        return len(self._cache)


# ============================================================
# Retriever Interface
# ============================================================

class RetrieverInterface(BaseInterface):
    """
    프로덕션급 검색기 인터페이스

    모든 검색기(Vector, Graph, Hybrid 등)의 기본 인터페이스.
    QueryContext를 받아 검색을 수행하고 결과를 추가합니다.

    Features:
        - 타입 안전한 검색 결과
        - 캐싱 지원
        - 검색 품질 메트릭
        - 설정 체이닝
        - 배치 검색

    Usage:
        class MyRetriever(RetrieverInterface):
            def _do_retrieve(self, context: QueryContext) -> RetrievalResult:
                # 검색 로직 구현
                items = self._search(context.question)
                return RetrievalResult(source="my_source", items=items)

        retriever = MyRetriever(brand_config)
        context = retriever.configure(top_k=20).retrieve(context)
    """

    def __init__(
        self,
        brand_config: Dict[str, Any],
        source: RetrievalSource = RetrievalSource.CUSTOM
    ):
        """
        Args:
            brand_config: 브랜드 설정
            source: 검색 소스 타입
        """
        super().__init__(brand_config, ComponentType.RETRIEVER)

        self.source = source
        self.source_name = source.value

        # 설정 로드
        retrieval_config = brand_config.get('retrieval', {}).get(source.value, {})
        self._config = RetrievalConfig.from_dict(retrieval_config)

        # 캐시
        self._cache = RetrievalCache(
            ttl_seconds=self._config.cache_ttl_seconds,
            max_size=1000
        )

        # 검색 메트릭
        self._retrieval_metrics = RetrievalMetrics()

        # 후처리 훅
        self._post_process_hooks: List[Callable[[RetrievalResult], RetrievalResult]] = []

    # === Main Methods ===

    def retrieve(self, context: QueryContext) -> QueryContext:
        """
        검색 실행

        Args:
            context: 쿼리 컨텍스트

        Returns:
            검색 결과가 추가된 컨텍스트
        """
        start_time = time.time()

        try:
            # 캐시 체크
            if self._config.cache_enabled:
                cache_key = self._cache.generate_key(
                    context.question,
                    self._config
                )
                cached = self._cache.get(cache_key)

                if cached:
                    cached.cached = True
                    cached.status = RetrievalStatus.CACHED
                    self._add_to_context(context, cached)
                    self._record_retrieval(cached, time.time() - start_time)
                    logger.debug(f"Cache hit for {self.source_name}")
                    return context

            # 실제 검색 수행
            result = self._do_retrieve(context)
            result.latency_ms = (time.time() - start_time) * 1000

            # 후처리
            result = self._post_process(result)

            # 캐시 저장
            if self._config.cache_enabled and result.is_success:
                self._cache.set(cache_key, result)

            # 컨텍스트에 추가
            self._add_to_context(context, result)

            # 메트릭 기록
            self._record_retrieval(result, time.time() - start_time)

            logger.info(
                f"{self.source_name} retrieval: {len(result)} items "
                f"in {result.latency_ms:.1f}ms"
            )

            return context

        except asyncio.TimeoutError:
            error = f"Retrieval timeout after {self._config.timeout_ms}ms"
            logger.error(error)
            self._record_call(False, (time.time() - start_time) * 1000, error)
            self._retrieval_metrics.record_query(0, 0, False, timeout=True)
            return context

        except Exception as e:
            error = f"Retrieval error: {str(e)}"
            logger.error(error, exc_info=True)
            self._record_call(False, (time.time() - start_time) * 1000, error)
            self._retrieval_metrics.record_query(0, 0, False, error=True)
            return context

    async def retrieve_async(self, context: QueryContext) -> QueryContext:
        """
        비동기 검색 실행

        Args:
            context: 쿼리 컨텍스트

        Returns:
            검색 결과가 추가된 컨텍스트
        """
        start_time = time.time()

        try:
            # 비동기 검색 (타임아웃 적용)
            result = await asyncio.wait_for(
                self._do_retrieve_async(context),
                timeout=self._config.timeout_ms / 1000
            )

            result.latency_ms = (time.time() - start_time) * 1000
            result = self._post_process(result)

            self._add_to_context(context, result)
            self._record_retrieval(result, time.time() - start_time)

            return context

        except asyncio.TimeoutError:
            logger.error(f"Async retrieval timeout for {self.source_name}")
            return context
        except Exception as e:
            logger.error(f"Async retrieval error: {e}", exc_info=True)
            return context

    def retrieve_batch(
        self,
        contexts: List[QueryContext]
    ) -> List[QueryContext]:
        """
        배치 검색

        Args:
            contexts: 쿼리 컨텍스트 리스트

        Returns:
            검색 결과가 추가된 컨텍스트 리스트
        """
        return [self.retrieve(ctx) for ctx in contexts]

    # === Abstract Method ===

    @abstractmethod
    def _do_retrieve(self, context: QueryContext) -> RetrievalResult:
        """
        실제 검색 로직 (구현 필수)

        Args:
            context: 쿼리 컨텍스트

        Returns:
            검색 결과
        """
        pass

    async def _do_retrieve_async(self, context: QueryContext) -> RetrievalResult:
        """
        비동기 검색 로직 (오버라이드 가능)

        기본 구현은 동기 메서드를 래핑합니다.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._do_retrieve, context)

    # === Configuration ===

    def configure(self, **kwargs) -> 'RetrieverInterface':
        """
        검색 설정 변경 (체이닝 지원)

        Usage:
            retriever.configure(top_k=20, min_score=0.5).retrieve(context)

        Args:
            **kwargs: 설정 키-값

        Returns:
            self (체이닝용)
        """
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
                logger.debug(f"Retriever config: {key}={value}")
        return self

    def get_retrieval_config(self) -> RetrievalConfig:
        """검색 설정 조회"""
        return self._config

    def _get_component_config(self) -> Dict[str, Any]:
        """컴포넌트 설정 (오버라이드)"""
        return self._config.to_dict()

    # === Hooks ===

    def add_post_process_hook(
        self,
        hook: Callable[[RetrievalResult], RetrievalResult]
    ) -> 'RetrieverInterface':
        """
        후처리 훅 추가

        Args:
            hook: 후처리 함수

        Returns:
            self (체이닝용)
        """
        self._post_process_hooks.append(hook)
        return self

    def clear_hooks(self):
        """훅 초기화"""
        self._post_process_hooks.clear()

    # === Metrics ===

    def get_retrieval_metrics(self) -> Dict[str, Any]:
        """검색 메트릭 조회"""
        return self._retrieval_metrics.to_dict()

    def reset_retrieval_metrics(self):
        """검색 메트릭 초기화"""
        self._retrieval_metrics = RetrievalMetrics()

    # === Cache ===

    def clear_cache(self):
        """캐시 초기화"""
        self._cache.clear()
        logger.info(f"{self.source_name} cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        return {
            'size': self._cache.size,
            'ttl_seconds': self._cache.ttl_seconds,
            'hit_rate': round(self._retrieval_metrics.cache_hit_rate, 4),
        }

    # === Health Check ===

    def _do_health_check(self) -> HealthCheckResult:
        """헬스체크 (오버라이드)"""
        try:
            # 기본 헬스체크
            metrics = self._retrieval_metrics
            error_rate = (
                metrics.error_count / metrics.total_queries
                if metrics.total_queries > 0 else 0
            )

            if error_rate > 0.5:
                return HealthCheckResult(
                    healthy=False,
                    status=ComponentStatus.DEGRADED,
                    message=f"High error rate: {error_rate:.1%}",
                    details={'error_rate': error_rate}
                )

            return HealthCheckResult(
                healthy=True,
                status=ComponentStatus.READY,
                message="OK",
                details={
                    'cache_size': self._cache.size,
                    'total_queries': metrics.total_queries,
                }
            )

        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                status=ComponentStatus.ERROR,
                message=str(e)
            )

    # === Debug ===

    def get_debug_info(self) -> Dict[str, Any]:
        """디버그 정보 (확장)"""
        base_info = super().get_debug_info()
        base_info.update({
            'source': self.source_name,
            'retrieval_config': self._config.to_dict(),
            'retrieval_metrics': self.get_retrieval_metrics(),
            'cache_stats': self.get_cache_stats(),
        })
        return base_info

    # === Private Helpers ===

    def _add_to_context(self, context: QueryContext, result: RetrievalResult):
        """컨텍스트에 결과 추가"""
        if result.is_success and len(result) > 0:
            context.add_retrieval_result(
                source=result.source,
                data=[item.to_dict() for item in result.items],
                metadata=result.metadata,
                score=result.average_score
            )

    def _post_process(self, result: RetrievalResult) -> RetrievalResult:
        """후처리 실행"""
        # 중복 제거
        if self._config.enable_dedup:
            result = self._deduplicate(result)

        # 점수 필터링
        if self._config.min_score > 0:
            result = result.filter_by_score(self._config.min_score)

        # 결과 수 제한
        if len(result.items) > self._config.max_results:
            result.items = result.items[:self._config.max_results]

        # 랭크 할당
        for i, item in enumerate(result.items):
            item.rank = i + 1

        # 사용자 정의 훅
        for hook in self._post_process_hooks:
            try:
                result = hook(result)
            except Exception as e:
                logger.error(f"Post-process hook error: {e}")

        return result

    def _deduplicate(self, result: RetrievalResult) -> RetrievalResult:
        """중복 제거"""
        seen = set()
        unique = []

        for item in result.items:
            key = (item.node_type, item.id)
            if key not in seen:
                seen.add(key)
                unique.append(item)

        return RetrievalResult(
            source=result.source,
            items=unique,
            status=result.status,
            latency_ms=result.latency_ms,
            cached=result.cached,
            metadata=result.metadata,
        )

    def _record_retrieval(self, result: RetrievalResult, elapsed: float):
        """검색 기록"""
        latency_ms = elapsed * 1000
        self._retrieval_metrics.record_query(
            result_count=len(result),
            latency_ms=latency_ms,
            cached=result.cached,
            error=not result.is_success,
        )
        self._record_call(result.is_success, latency_ms)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"brand={self.brand_id}, "
            f"source={self.source_name})"
        )


# ============================================================
# Exports
# ============================================================

__all__ = [
    'RetrieverInterface',
    'RetrievalStatus',
    'RetrievalSource',
    'RetrievalConfig',
    'RetrievalItem',
    'RetrievalResult',
    'RetrievalMetrics',
    'RetrievalCache',
]
