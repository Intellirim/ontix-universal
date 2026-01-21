"""
공통 서비스 패키지
- 프로덕션 급 싱글톤 서비스들
- 캐시, LLM, Neo4j, Vector 서비스
"""

# Cache Service
from app.services.shared.cache import (
    CacheClient,
    CacheConfig,
    CacheMetrics,
    CacheStrategy,
    SerializationType,
    DistributedLock,
    get_cache_client,
    cached,
)

# LLM Service
from app.services.shared.llm import (
    LLMClient,
    LLMConfig,
    LLMMetrics,
    LLMResponse,
    ModelVariant,
    ModelProvider,
    LLMRateLimiter,
    TokenCounter,
    CostCalculator,
    get_llm_client,
)

# Neo4j Service
from app.services.shared.neo4j import (
    Neo4jClient,
    Neo4jConfig,
    QueryMetrics,
    QueryResult,
    QueryType,
    ConnectionState,
    get_neo4j_client,
)

# Vector Service
from app.services.shared.vector import (
    VectorService,
    VectorConfig,
    VectorMetrics,
    EmbeddingResult,
    EmbeddingModel,
    EmbeddingType,
    EmbeddingCostCalculator,
    EmbeddingCache,
    get_vector_service,
    create_vector_service,
    embed_text,
    embed_texts,
    calculate_similarity,
)

__all__ = [
    # Cache
    "CacheClient",
    "CacheConfig",
    "CacheMetrics",
    "CacheStrategy",
    "SerializationType",
    "DistributedLock",
    "get_cache_client",
    "cached",
    # LLM
    "LLMClient",
    "LLMConfig",
    "LLMMetrics",
    "LLMResponse",
    "ModelVariant",
    "ModelProvider",
    "LLMRateLimiter",
    "TokenCounter",
    "CostCalculator",
    "get_llm_client",
    # Neo4j
    "Neo4jClient",
    "Neo4jConfig",
    "QueryMetrics",
    "QueryResult",
    "QueryType",
    "ConnectionState",
    "get_neo4j_client",
    # Vector
    "VectorService",
    "VectorConfig",
    "VectorMetrics",
    "EmbeddingResult",
    "EmbeddingModel",
    "EmbeddingType",
    "EmbeddingCostCalculator",
    "EmbeddingCache",
    "get_vector_service",
    "create_vector_service",
    "embed_text",
    "embed_texts",
    "calculate_similarity",
]
