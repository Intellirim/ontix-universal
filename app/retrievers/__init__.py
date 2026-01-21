"""
ONTIX Universal - Retrievers Package
지식 그래프 검색기 구현체

Production Grade v2.0

Usage:
    from app.retrievers import (
        GraphRetriever,
        VectorRetriever,
        HybridRetriever,
        StatsRetriever,
        ProductRetriever,
    )

    # 기본 사용
    retriever = HybridRetriever(brand_config)
    context = retriever.retrieve(query_context)
"""

# Import order matters to avoid circular imports
# 1. First import modules without cross-dependencies

# Graph Retriever (no cross-dependencies within retrievers)
from app.retrievers.graph import (
    GraphRetriever,
    SearchScope,
    GraphSearchConfig,
    SearchResult as GraphSearchResult,
    KeywordExtractor,
)

# Vector Retriever (no cross-dependencies within retrievers)
from app.retrievers.vector import (
    VectorRetriever,
    SearchMode,
    NodeIndex,
    VectorSearchConfig,
    VectorResult,
    QueryPreprocessor,
    ScoreNormalizer,
)

# Stats Retriever (no cross-dependencies within retrievers)
from app.retrievers.stats import (
    StatsRetriever,
    StatsType,
    TimePeriod,
    SortMetric,
    StatsConfig,
    StatsResult,
    MetricsCalculator,
    TimeHelper,
)

# Product Retriever (no cross-dependencies within retrievers)
from app.retrievers.product import (
    ProductRetriever,
    ProductSearchMode,
    ProductSortBy,
    StockStatus,
    ProductSearchConfig,
    ProductResult,
    PriceParser,
    QueryAnalyzer,
)

# 2. Then import modules with cross-dependencies
# Hybrid Retriever (imports from graph and vector)
from app.retrievers.hybrid import (
    HybridRetriever,
    FusionMethod,
    HybridSearchConfig,
    HybridResult,
    RRFCalculator,
    ResultMerger,
)


__all__ = [
    # Retrievers
    "GraphRetriever",
    "VectorRetriever",
    "HybridRetriever",
    "StatsRetriever",
    "ProductRetriever",
    # Graph Types
    "SearchScope",
    "GraphSearchConfig",
    "GraphSearchResult",
    "KeywordExtractor",
    # Vector Types
    "SearchMode",
    "NodeIndex",
    "VectorSearchConfig",
    "VectorResult",
    "QueryPreprocessor",
    "ScoreNormalizer",
    # Hybrid Types
    "FusionMethod",
    "HybridSearchConfig",
    "HybridResult",
    "RRFCalculator",
    "ResultMerger",
    # Stats Types
    "StatsType",
    "TimePeriod",
    "SortMetric",
    "StatsConfig",
    "StatsResult",
    "MetricsCalculator",
    "TimeHelper",
    # Product Types
    "ProductSearchMode",
    "ProductSortBy",
    "StockStatus",
    "ProductSearchConfig",
    "ProductResult",
    "PriceParser",
    "QueryAnalyzer",
]

__version__ = "2.0.0"
