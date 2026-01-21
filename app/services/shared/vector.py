"""
벡터 서비스 (임베딩 생성)
- 프로덕션 급 임베딩 서비스
- 배치 처리, 캐싱, 비용 추적 지원
"""

from langchain_openai import OpenAIEmbeddings
from functools import lru_cache
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib
import threading
import os
import logging
import time

logger = logging.getLogger(__name__)


# ============================================================
# Enums
# ============================================================

class EmbeddingModel(str, Enum):
    """임베딩 모델"""
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"

    @property
    def dimensions(self) -> int:
        """모델별 차원 수"""
        dims = {
            self.TEXT_EMBEDDING_3_SMALL: 1536,
            self.TEXT_EMBEDDING_3_LARGE: 3072,
            self.TEXT_EMBEDDING_ADA_002: 1536,
        }
        return dims.get(self, 1536)

    @property
    def max_tokens(self) -> int:
        """모델별 최대 토큰"""
        return 8191  # 모든 모델 동일


class EmbeddingType(str, Enum):
    """임베딩 타입"""
    QUERY = "query"
    DOCUMENT = "document"
    BATCH = "batch"


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class VectorConfig:
    """벡터 서비스 설정"""
    model: str = "text-embedding-3-small"
    batch_size: int = 100
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_cache: bool = True
    cache_ttl: int = 3600

    @classmethod
    def from_env(cls) -> "VectorConfig":
        """환경변수에서 설정 로드"""
        return cls(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "100")),
            max_retries=int(os.getenv("EMBEDDING_MAX_RETRIES", "3")),
            enable_cache=os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true",
            cache_ttl=int(os.getenv("EMBEDDING_CACHE_TTL", "3600")),
        )


@dataclass
class VectorMetrics:
    """벡터 서비스 메트릭"""
    total_embeddings: int = 0
    query_embeddings: int = 0
    document_embeddings: int = 0
    batch_operations: int = 0
    total_tokens_used: int = 0
    total_cost: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    error_count: int = 0
    total_time_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

    def record_embedding(self, embedding_type: EmbeddingType,
                        token_count: int, duration_ms: float,
                        from_cache: bool = False):
        """임베딩 기록"""
        self.total_embeddings += 1
        self.total_time_ms += duration_ms

        if embedding_type == EmbeddingType.QUERY:
            self.query_embeddings += 1
        elif embedding_type == EmbeddingType.DOCUMENT:
            self.document_embeddings += 1
        elif embedding_type == EmbeddingType.BATCH:
            self.batch_operations += 1

        if from_cache:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            self.total_tokens_used += token_count

    def record_error(self):
        """에러 기록"""
        self.error_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "total_embeddings": self.total_embeddings,
            "query_embeddings": self.query_embeddings,
            "document_embeddings": self.document_embeddings,
            "batch_operations": self.batch_operations,
            "total_tokens_used": self.total_tokens_used,
            "total_cost": round(self.total_cost, 6),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self._cache_hit_rate(),
            "error_count": self.error_count,
            "avg_time_ms": self._avg_time(),
            "uptime_seconds": (datetime.now() - self.created_at).total_seconds(),
        }

    def _cache_hit_rate(self) -> float:
        """캐시 히트율"""
        total = self.cache_hits + self.cache_misses
        return round(self.cache_hits / total, 4) if total > 0 else 0.0

    def _avg_time(self) -> float:
        """평균 처리 시간"""
        return round(self.total_time_ms / self.total_embeddings, 2) if self.total_embeddings > 0 else 0.0


@dataclass
class EmbeddingResult:
    """임베딩 결과"""
    vector: List[float]
    text: str
    token_count: int
    from_cache: bool
    model: str
    dimensions: int
    duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "vector": self.vector[:5] + ["..."] if len(self.vector) > 5 else self.vector,
            "text_preview": self.text[:50] + "..." if len(self.text) > 50 else self.text,
            "token_count": self.token_count,
            "from_cache": self.from_cache,
            "model": self.model,
            "dimensions": self.dimensions,
            "duration_ms": self.duration_ms,
        }


# ============================================================
# Utility Classes
# ============================================================

class EmbeddingCostCalculator:
    """임베딩 비용 계산기"""

    # 가격: $ per 1M tokens
    PRICING = {
        "text-embedding-3-small": 0.020,
        "text-embedding-3-large": 0.130,
        "text-embedding-ada-002": 0.100,
    }

    @classmethod
    def calculate(cls, model: str, token_count: int) -> float:
        """비용 계산"""
        price_per_million = cls.PRICING.get(model, 0.020)
        return (token_count / 1_000_000) * price_per_million

    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        """토큰 수 추정 (간단한 휴리스틱)"""
        # 평균적으로 4글자당 1토큰 (영어 기준)
        # 한글은 대략 2글자당 1토큰
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        non_ascii_chars = len(text) - ascii_chars
        return int(ascii_chars / 4 + non_ascii_chars / 2) + 1


class EmbeddingCache:
    """임베딩 캐시 (인메모리)"""

    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Tuple[List[float], float]] = {}
        self._lock = threading.Lock()

    def _hash_text(self, text: str, model: str) -> str:
        """텍스트 해시 생성"""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, text: str, model: str) -> Optional[List[float]]:
        """캐시에서 임베딩 조회"""
        key = self._hash_text(text, model)
        with self._lock:
            if key in self._cache:
                vector, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl:
                    return vector
                else:
                    del self._cache[key]
        return None

    def set(self, text: str, model: str, vector: List[float]):
        """캐시에 임베딩 저장"""
        key = self._hash_text(text, model)
        with self._lock:
            # LRU 방식으로 오래된 항목 제거
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._cache.keys(),
                               key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

            self._cache[key] = (vector, time.time())

    def clear(self):
        """캐시 초기화"""
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """캐시 크기"""
        return len(self._cache)


# ============================================================
# Main Service
# ============================================================

class VectorService:
    """프로덕션 급 벡터 임베딩 서비스"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, config: Optional[VectorConfig] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[VectorConfig] = None):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.config = config or VectorConfig.from_env()
        self.embeddings = OpenAIEmbeddings(model=self.config.model)
        self.metrics = VectorMetrics()
        self.cache = EmbeddingCache(ttl=self.config.cache_ttl) if self.config.enable_cache else None

        self._initialized = True
        logger.info(f"VectorService initialized: model={self.config.model}, cache={self.config.enable_cache}")

    # --------------------------------------------------
    # Core Methods
    # --------------------------------------------------

    def embed_query(self, text: str, use_cache: bool = True) -> List[float]:
        """쿼리 임베딩 (단일)"""
        start_time = time.time()
        from_cache = False

        # 캐시 확인
        if use_cache and self.cache:
            cached = self.cache.get(text, self.config.model)
            if cached:
                from_cache = True
                duration_ms = (time.time() - start_time) * 1000
                self.metrics.record_embedding(
                    EmbeddingType.QUERY, 0, duration_ms, from_cache=True
                )
                return cached

        # 임베딩 생성
        try:
            vector = self.embeddings.embed_query(text)
            token_count = EmbeddingCostCalculator.estimate_tokens(text)

            # 캐시 저장
            if self.cache:
                self.cache.set(text, self.config.model, vector)

            # 비용 계산
            cost = EmbeddingCostCalculator.calculate(self.config.model, token_count)
            self.metrics.total_cost += cost

            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_embedding(
                EmbeddingType.QUERY, token_count, duration_ms, from_cache=False
            )

            return vector

        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Embedding error: {e}")
            raise

    def embed_documents(self, texts: List[str], use_cache: bool = True) -> List[List[float]]:
        """문서 임베딩 (다중)"""
        if not texts:
            return []

        start_time = time.time()
        results: List[Optional[List[float]]] = [None] * len(texts)
        texts_to_embed: List[Tuple[int, str]] = []
        total_from_cache = 0

        # 캐시 확인
        for i, text in enumerate(texts):
            if use_cache and self.cache:
                cached = self.cache.get(text, self.config.model)
                if cached:
                    results[i] = cached
                    total_from_cache += 1
                    continue
            texts_to_embed.append((i, text))

        # 캐시 미스된 텍스트만 임베딩
        if texts_to_embed:
            try:
                indices, uncached_texts = zip(*texts_to_embed)
                vectors = self._batch_embed(list(uncached_texts))

                for idx, vector, text in zip(indices, vectors, uncached_texts):
                    results[idx] = vector
                    if self.cache:
                        self.cache.set(text, self.config.model, vector)

            except Exception as e:
                self.metrics.record_error()
                logger.error(f"Batch embedding error: {e}")
                raise

        # 메트릭 기록
        duration_ms = (time.time() - start_time) * 1000
        total_tokens = sum(EmbeddingCostCalculator.estimate_tokens(t) for t in texts)

        self.metrics.record_embedding(
            EmbeddingType.DOCUMENT,
            total_tokens - (total_tokens * total_from_cache // len(texts)),
            duration_ms,
            from_cache=(total_from_cache == len(texts))
        )

        # 비용 계산 (캐시 미스분만)
        if texts_to_embed:
            tokens_used = sum(EmbeddingCostCalculator.estimate_tokens(t) for _, t in texts_to_embed)
            cost = EmbeddingCostCalculator.calculate(self.config.model, tokens_used)
            self.metrics.total_cost += cost

        return results

    def embed_with_result(self, text: str, use_cache: bool = True) -> EmbeddingResult:
        """상세 결과와 함께 임베딩"""
        start_time = time.time()
        from_cache = False

        # 캐시 확인
        if use_cache and self.cache:
            cached = self.cache.get(text, self.config.model)
            if cached:
                from_cache = True
                vector = cached
            else:
                vector = self.embeddings.embed_query(text)
                self.cache.set(text, self.config.model, vector)
        else:
            vector = self.embeddings.embed_query(text)

        duration_ms = (time.time() - start_time) * 1000
        token_count = EmbeddingCostCalculator.estimate_tokens(text)

        # 모델 차원 정보
        try:
            model_enum = EmbeddingModel(self.config.model)
            dimensions = model_enum.dimensions
        except ValueError:
            dimensions = len(vector)

        return EmbeddingResult(
            vector=vector,
            text=text,
            token_count=token_count,
            from_cache=from_cache,
            model=self.config.model,
            dimensions=dimensions,
            duration_ms=duration_ms,
        )

    # --------------------------------------------------
    # Batch Operations
    # --------------------------------------------------

    def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        """배치 임베딩 (청크 분할)"""
        if len(texts) <= self.config.batch_size:
            return self.embeddings.embed_documents(texts)

        # 청크로 분할하여 처리
        all_vectors = []
        for i in range(0, len(texts), self.config.batch_size):
            chunk = texts[i:i + self.config.batch_size]
            vectors = self.embeddings.embed_documents(chunk)
            all_vectors.extend(vectors)
            self.metrics.batch_operations += 1

        return all_vectors

    def batch_embed_with_progress(self, texts: List[str],
                                   callback: Optional[callable] = None) -> List[List[float]]:
        """진행 상황 콜백과 함께 배치 임베딩"""
        total = len(texts)
        all_vectors = []

        for i in range(0, total, self.config.batch_size):
            chunk = texts[i:i + self.config.batch_size]
            vectors = self.embeddings.embed_documents(chunk)
            all_vectors.extend(vectors)

            if callback:
                progress = min(i + self.config.batch_size, total) / total * 100
                callback(progress, len(all_vectors), total)

        return all_vectors

    # --------------------------------------------------
    # Utility Methods
    # --------------------------------------------------

    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 계산"""
        if len(vec1) != len(vec2):
            raise ValueError("Vector dimensions must match")

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def find_most_similar(self, query_vector: List[float],
                          candidates: List[List[float]],
                          top_k: int = 5) -> List[Tuple[int, float]]:
        """가장 유사한 벡터 찾기"""
        similarities = [
            (i, self.similarity(query_vector, vec))
            for i, vec in enumerate(candidates)
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 조회"""
        try:
            model_enum = EmbeddingModel(self.config.model)
            return {
                "model": self.config.model,
                "dimensions": model_enum.dimensions,
                "max_tokens": model_enum.max_tokens,
                "price_per_million": EmbeddingCostCalculator.PRICING.get(self.config.model, 0),
            }
        except ValueError:
            return {
                "model": self.config.model,
                "dimensions": "unknown",
                "max_tokens": 8191,
                "price_per_million": EmbeddingCostCalculator.PRICING.get(self.config.model, 0),
            }

    # --------------------------------------------------
    # Cache Management
    # --------------------------------------------------

    def clear_cache(self):
        """캐시 초기화"""
        if self.cache:
            self.cache.clear()
            logger.info("Embedding cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        if not self.cache:
            return {"enabled": False}

        return {
            "enabled": True,
            "size": self.cache.size(),
            "max_size": self.cache.max_size,
            "ttl": self.cache.ttl,
            "hit_rate": self.metrics._cache_hit_rate(),
        }

    # --------------------------------------------------
    # Health & Metrics
    # --------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """헬스 체크"""
        try:
            # 간단한 테스트 임베딩
            start = time.time()
            test_vector = self.embeddings.embed_query("health check")
            latency = (time.time() - start) * 1000

            return {
                "status": "healthy",
                "model": self.config.model,
                "dimensions": len(test_vector),
                "latency_ms": round(latency, 2),
                "cache_enabled": self.config.enable_cache,
                "cache_size": self.cache.size() if self.cache else 0,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.config.model,
            }

    def get_metrics(self) -> Dict[str, Any]:
        """메트릭 조회"""
        return self.metrics.to_dict()

    def reset_metrics(self):
        """메트릭 리셋"""
        self.metrics = VectorMetrics()


# ============================================================
# Factory Functions
# ============================================================

@lru_cache()
def get_vector_service() -> VectorService:
    """싱글톤 벡터 서비스"""
    return VectorService()


def create_vector_service(config: VectorConfig) -> VectorService:
    """커스텀 설정으로 벡터 서비스 생성"""
    # 싱글톤 초기화 (첫 호출시에만 config 적용)
    VectorService._instance = None
    return VectorService(config)


# ============================================================
# Convenience Functions
# ============================================================

def embed_text(text: str) -> List[float]:
    """텍스트 임베딩 (편의 함수)"""
    return get_vector_service().embed_query(text)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """텍스트 목록 임베딩 (편의 함수)"""
    return get_vector_service().embed_documents(texts)


def calculate_similarity(text1: str, text2: str) -> float:
    """두 텍스트 유사도 계산"""
    service = get_vector_service()
    vec1 = service.embed_query(text1)
    vec2 = service.embed_query(text2)
    return service.similarity(vec1, vec2)
