"""
Services 패키지
- shared: 공통 인프라 서비스 (Cache, LLM, Neo4j, Vector)
- platform: 플랫폼 서비스 (Config, Brand, Feature, Analytics, Monitoring)
"""

# Shared Services - Core Infrastructure
from app.services.shared import (
    # Cache
    CacheClient,
    CacheConfig,
    get_cache_client,
    cached,
    # LLM
    LLMClient,
    LLMConfig,
    get_llm_client,
    # Neo4j
    Neo4jClient,
    Neo4jConfig,
    get_neo4j_client,
    # Vector
    VectorService,
    VectorConfig,
    get_vector_service,
    embed_text,
    embed_texts,
)

# Platform Services - Application Layer
from app.services.platform import (
    # Config
    ConfigManager,
    load_brand_config,
    load_platform_config,
    list_brands,
    # Brand
    BrandManager,
    get_brand_manager,
    get_brand,
    validate_brand,
    # Feature
    FeatureManager,
    get_feature_manager,
    validate_feature_config,
    list_features,
    # Analytics
    AnalyticsService,
    get_analytics_service,
    track_event,
    track_query,
    track_error,
    # Monitoring
    MonitoringService,
    get_monitoring_service,
    health_check,
    get_system_stats,
)

__all__ = [
    # Shared - Cache
    "CacheClient",
    "CacheConfig",
    "get_cache_client",
    "cached",
    # Shared - LLM
    "LLMClient",
    "LLMConfig",
    "get_llm_client",
    # Shared - Neo4j
    "Neo4jClient",
    "Neo4jConfig",
    "get_neo4j_client",
    # Shared - Vector
    "VectorService",
    "VectorConfig",
    "get_vector_service",
    "embed_text",
    "embed_texts",
    # Platform - Config
    "ConfigManager",
    "load_brand_config",
    "load_platform_config",
    "list_brands",
    # Platform - Brand
    "BrandManager",
    "get_brand_manager",
    "get_brand",
    "validate_brand",
    # Platform - Feature
    "FeatureManager",
    "get_feature_manager",
    "validate_feature_config",
    "list_features",
    # Platform - Analytics
    "AnalyticsService",
    "get_analytics_service",
    "track_event",
    "track_query",
    "track_error",
    # Platform - Monitoring
    "MonitoringService",
    "get_monitoring_service",
    "health_check",
    "get_system_stats",
]
