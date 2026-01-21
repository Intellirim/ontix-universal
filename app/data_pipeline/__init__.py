"""
ONTIX Universal - SNS Data Pipeline
범용 SNS 데이터 수집 및 지식그래프 생성 파이프라인

Usage:
    from app.data_pipeline import SNSDataPipeline, PlatformType

    pipeline = SNSDataPipeline()
    await pipeline.run(brand_id="my_brand", platform=PlatformType.INSTAGRAM, ...)
"""

# Domain Models
from .domain.models import (
    PlatformType,
    ContentType,
    ActorDTO,
    ContentDTO,
    InteractionDTO,
    TopicDTO,
)

# Adapters
from .adapters import (
    BaseSNSAdapter,
    InstagramAdapter,
    YouTubeAdapter,
    TikTokAdapter,
    TwitterAdapter,
)

# Crawlers
from .crawlers import ApifyClient

# Processors
from .processors import LLMProcessor

# Repositories
from .repositories import Neo4jRepository

# Pipeline
from .pipeline import (
    SNSDataPipeline,
    PipelineConfig,
    PipelineStage,
)

__all__ = [
    # Domain Models
    "PlatformType",
    "ContentType",
    "ActorDTO",
    "ContentDTO",
    "InteractionDTO",
    "TopicDTO",
    # Adapters
    "BaseSNSAdapter",
    "InstagramAdapter",
    "YouTubeAdapter",
    "TikTokAdapter",
    "TwitterAdapter",
    # Crawlers
    "ApifyClient",
    # Processors
    "LLMProcessor",
    # Repositories
    "Neo4jRepository",
    # Pipeline
    "SNSDataPipeline",
    "PipelineConfig",
    "PipelineStage",
]

__version__ = "1.0.0"
