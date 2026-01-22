"""
SNS Data Pipeline Orchestrator - Production Grade v4.0

ONTIX Universal 범용 SNS 데이터 수집 및 지식그래프 생성 파이프라인.
모든 SNS 플랫폼(Instagram, YouTube, TikTok, Twitter)의 데이터를 수집하고,
LLM을 통해 지식그래프로 변환하여 Neo4j에 저장합니다.

Pipeline Stages:
    1. Crawl - SNS 데이터 크롤링 (Apify)
    2. Transform - 원시 데이터를 DTO로 변환 (Adapters)
    3. Filter - 중복 콘텐츠 필터링 (Neo4j)
    4. Process - 지식그래프 생성 (LLM)
    5. Save - Neo4j에 저장 (Repository)

Author: ONTIX Universal Team
Version: 4.0.0
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
)

from dotenv import load_dotenv

from .adapters import (
    BaseSNSAdapter,
    InstagramAdapter,
    TikTokAdapter,
    TwitterAdapter,
    YouTubeAdapter,
)
from .crawlers import ApifyClient
from .domain.models import (
    ActorDTO,
    ContentDTO,
    ContentType,
    InteractionDTO,
    PlatformType,
    TopicDTO,
)
from .processors import LLMProcessor
from .processors.llm_processor import GraphProcessingResult
from .repositories import Neo4jRepository
from .repositories.neo4j_repo import SaveResult

# Load environment variables
load_dotenv()

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS & CONFIGURATIONS
# ============================================================================

class PipelineStage(Enum):
    """파이프라인 단계 정의"""

    CRAWL = "crawl"
    TRANSFORM = "transform"
    FILTER = "filter"
    PROCESS = "process"
    SAVE = "save"


class PipelineConfig:
    """파이프라인 설정 상수"""

    # Retry settings
    MAX_RETRIES: Final[int] = 3
    RETRY_DELAY_SECONDS: Final[float] = 5.0

    # Batch settings
    DEFAULT_BATCH_SIZE: Final[int] = 50
    MAX_CONCURRENT_TASKS: Final[int] = 5

    # Timeout settings (seconds)
    CRAWL_TIMEOUT: Final[int] = 600  # 10 minutes
    PROCESS_TIMEOUT: Final[int] = 300  # 5 minutes

    # Default Apify actors (실제 사용 중인 Actor)
    APIFY_ACTORS: Final[Dict[PlatformType, str]] = {
        PlatformType.INSTAGRAM: "shu8hvrXbJbY3Eb9W",  # apify/instagram-post-scraper
        PlatformType.YOUTUBE: "h7sDV53CddomktSi5",  # streamers/youtube-scraper
        PlatformType.TIKTOK: "GdWCkxBtKWOsKjdch",  # clockworks/free-tiktok-scraper
        PlatformType.TWITTER: "61RPP7dywgiy0JPD0",  # apidojo/tweet-scraper
    }


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

class PipelineStatistics(TypedDict):
    """파이프라인 실행 통계"""

    brand_id: str
    platform: str
    started_at: str
    completed_at: Optional[str]
    duration_seconds: float
    stage_timings: Dict[str, float]
    crawled_count: int
    transformed_count: int
    filtered_count: int
    processed_count: int
    saved_nodes: int
    saved_relationships: int
    errors: List[str]
    success: bool


class BrandConfig(TypedDict, total=False):
    """브랜드 설정"""

    id: str
    name: str
    name_en: str
    category: str
    philosophy: str
    core_concepts: List[str]
    description: str
    tone: str
    keywords: List[str]


class CrawlConfig(TypedDict, total=False):
    """크롤링 설정"""

    actor_id: str
    max_items: int
    timeout_secs: int
    proxy_configuration: Dict[str, Any]


class PipelineOptions(TypedDict, total=False):
    """파이프라인 실행 옵션"""

    skip_crawl: bool
    skip_filter: bool
    skip_llm: bool
    use_batch: bool
    batch_size: int
    max_items: int
    dry_run: bool


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PipelineContext:
    """
    파이프라인 실행 컨텍스트.

    각 단계 간 데이터 전달 및 상태 관리를 담당합니다.
    """

    # 설정
    brand_config: BrandConfig
    platform: PlatformType
    options: PipelineOptions = field(default_factory=dict)

    # 데이터 (단계별로 채워짐)
    raw_data: List[Dict[str, Any]] = field(default_factory=list)
    transformed_data: List[Dict[str, Any]] = field(default_factory=list)
    filtered_contents: List[ContentDTO] = field(default_factory=list)
    actors: List[ActorDTO] = field(default_factory=list)
    interactions: List[InteractionDTO] = field(default_factory=list)
    topics: List[TopicDTO] = field(default_factory=list)
    graph_results: List[GraphProcessingResult] = field(default_factory=list)
    save_results: List[SaveResult] = field(default_factory=list)

    # 통계
    started_at: datetime = field(default_factory=datetime.now)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def brand_id(self) -> str:
        """브랜드 ID 반환"""
        return self.brand_config.get("id", "unknown")

    @property
    def is_dry_run(self) -> bool:
        """드라이런 여부"""
        return self.options.get("dry_run", False)

    def add_error(self, stage: str, error: str) -> None:
        """에러 추가"""
        self.errors.append(f"[{stage}] {error}")
        logger.error(f"Pipeline error at {stage}: {error}")

    def record_stage_timing(self, stage: str, duration: float) -> None:
        """단계 소요시간 기록"""
        self.stage_timings[stage] = duration

    def get_statistics(self) -> PipelineStatistics:
        """실행 통계 반환"""
        completed_at = datetime.now()
        duration = (completed_at - self.started_at).total_seconds()

        # 저장 결과 집계
        total_nodes = sum(r.get("nodes_created", 0) + r.get("nodes_updated", 0) for r in self.save_results)
        total_rels = sum(r.get("relationships_created", 0) for r in self.save_results)

        return {
            "brand_id": self.brand_id,
            "platform": self.platform.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_seconds": duration,
            "stage_timings": self.stage_timings,
            "crawled_count": len(self.raw_data),
            "transformed_count": len(self.transformed_data),
            "filtered_count": len(self.filtered_contents),
            "processed_count": len(self.graph_results),
            "saved_nodes": total_nodes,
            "saved_relationships": total_rels,
            "errors": self.errors,
            "success": len(self.errors) == 0,
        }


@dataclass
class StageResult:
    """단계 실행 결과"""

    stage: PipelineStage
    success: bool
    duration_seconds: float
    items_processed: int = 0
    error: Optional[str] = None
    data: Any = None


# ============================================================================
# STAGE EXECUTORS
# ============================================================================

class BaseStageExecutor(ABC):
    """파이프라인 단계 실행기 베이스 클래스"""

    def __init__(self, stage: PipelineStage):
        self.stage = stage
        self.max_retries = PipelineConfig.MAX_RETRIES
        self.retry_delay = PipelineConfig.RETRY_DELAY_SECONDS

    @abstractmethod
    async def execute(self, ctx: PipelineContext) -> StageResult:
        """단계 실행"""
        pass

    async def execute_with_retry(self, ctx: PipelineContext) -> StageResult:
        """재시도 로직이 포함된 단계 실행"""
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                start_time = time.time()
                logger.info(f"[{self.stage.value}] Starting (attempt {attempt}/{self.max_retries})")

                result = await self.execute(ctx)

                duration = time.time() - start_time
                ctx.record_stage_timing(self.stage.value, duration)

                if result.success:
                    logger.info(f"[{self.stage.value}] Completed in {duration:.2f}s")
                    return result
                else:
                    last_error = result.error

            except Exception as e:
                last_error = str(e)
                logger.error(f"[{self.stage.value}] Error on attempt {attempt}: {e}")
                logger.debug(traceback.format_exc())

            if attempt < self.max_retries:
                logger.info(f"[{self.stage.value}] Retrying in {self.retry_delay}s...")
                await asyncio.sleep(self.retry_delay)

        ctx.add_error(self.stage.value, last_error or "Unknown error")
        return StageResult(
            stage=self.stage,
            success=False,
            duration_seconds=0,
            error=last_error,
        )


class CrawlStageExecutor(BaseStageExecutor):
    """크롤링 단계 실행기"""

    def __init__(self, apify_client: ApifyClient):
        super().__init__(PipelineStage.CRAWL)
        self.apify_client = apify_client

    async def execute(self, ctx: PipelineContext) -> StageResult:
        """SNS 데이터 크롤링 실행"""
        start_time = time.time()

        # 크롤링 스킵 옵션
        if ctx.options.get("skip_crawl"):
            logger.info("[crawl] Skipped by option")
            return StageResult(
                stage=self.stage,
                success=True,
                duration_seconds=0,
                items_processed=0,
            )

        # Actor ID 결정
        actor_id = PipelineConfig.APIFY_ACTORS.get(ctx.platform)
        if not actor_id:
            return StageResult(
                stage=self.stage,
                success=False,
                duration_seconds=0,
                error=f"Unsupported platform: {ctx.platform.value}",
            )

        # 크롤링 입력 구성
        max_items = ctx.options.get("max_items", 100)
        run_input = self._build_run_input(ctx.platform, ctx.brand_config, max_items)

        # 크롤링 실행
        raw_data = await self.apify_client.run_actor(
            actor_id=actor_id,
            run_input=run_input,
            timeout_secs=PipelineConfig.CRAWL_TIMEOUT,
        )

        ctx.raw_data = raw_data
        duration = time.time() - start_time

        logger.info(f"[crawl] Crawled {len(raw_data)} items from {ctx.platform.value}")

        return StageResult(
            stage=self.stage,
            success=True,
            duration_seconds=duration,
            items_processed=len(raw_data),
            data=raw_data,
        )

    def _build_run_input(
        self,
        platform: PlatformType,
        brand_config: BrandConfig,
        max_items: int,
    ) -> Dict[str, Any]:
        """플랫폼별 크롤링 입력 생성

        brand_config에서 타겟 정보를 우선 사용하고, 없으면 keywords나 brand_name을 fallback으로 사용
        """
        # API에서 전달된 타겟 정보 (pipeline.py API의 _build_crawl_input에서 생성)
        # Instagram: usernames, hashtags, search
        # YouTube: channelHandles, searchKeywords
        # TikTok: profiles, hashtags, searchQueries
        # Twitter: twitterHandles, searchTerms

        keywords = brand_config.get("keywords", [])
        brand_name = brand_config.get("name", "")

        if platform == PlatformType.INSTAGRAM:
            # 타겟 정보 우선 사용 (apify/instagram-post-scraper 입력 형식)
            usernames = brand_config.get("usernames", [])
            hashtags = brand_config.get("hashtags", [])
            search = brand_config.get("search", [])

            run_input: Dict[str, Any] = {
                "resultsType": "posts",
                "resultsLimit": max_items,
                "searchLimit": 1,
                "addParentData": False,
            }

            if usernames:
                # 프로필 URL로 변환: https://www.instagram.com/{username}/
                direct_urls = [
                    f"https://www.instagram.com/{u.lstrip('@')}/"
                    for u in usernames
                ]
                run_input["directUrls"] = direct_urls
            elif hashtags:
                # 해시태그 URL로 변환
                direct_urls = [
                    f"https://www.instagram.com/explore/tags/{h.lstrip('#')}/"
                    for h in hashtags
                ]
                run_input["directUrls"] = direct_urls
                run_input["searchType"] = "hashtag"
            elif search:
                # 검색어는 해시태그로 처리
                direct_urls = [
                    f"https://www.instagram.com/explore/tags/{s}/"
                    for s in search
                ]
                run_input["directUrls"] = direct_urls
                run_input["searchType"] = "hashtag"
            else:
                # fallback: keywords 또는 brand_name 사용
                fallback = keywords[:5] if keywords else [brand_name]
                direct_urls = [
                    f"https://www.instagram.com/explore/tags/{s}/"
                    for s in fallback
                ]
                run_input["directUrls"] = direct_urls
                run_input["searchType"] = "hashtag"

            return run_input

        elif platform == PlatformType.YOUTUBE:
            # streamers/youtube-scraper 입력 형식
            channel_handles = brand_config.get("channelHandles", [])
            search_keywords = brand_config.get("searchKeywords", [])

            run_input: Dict[str, Any] = {
                "maxResults": max_items,
                "maxResultsShorts": 0,
                "maxResultStreams": 0,
                "startUrls": [],
            }

            if channel_handles:
                # 채널 URL로 변환
                start_urls = [
                    {"url": f"https://www.youtube.com/@{h.lstrip('@')}/videos"}
                    for h in channel_handles
                ]
                run_input["startUrls"] = start_urls
            elif search_keywords:
                run_input["searchQueries"] = search_keywords
            else:
                run_input["searchQueries"] = keywords[:5] if keywords else [brand_name]

            return run_input

        elif platform == PlatformType.TIKTOK:
            # clockworks/free-tiktok-scraper 입력 형식
            profiles = brand_config.get("profiles", [])
            hashtags = brand_config.get("hashtags", [])
            search_queries = brand_config.get("searchQueries", [])

            run_input: Dict[str, Any] = {
                "resultsPerPage": max_items,
                "profileScrapeSections": ["videos"],
                "profileSorting": "latest",
                "excludePinnedPosts": False,
                "shouldDownloadVideos": False,
                "shouldDownloadCovers": False,
                "shouldDownloadSubtitles": False,
                "shouldDownloadSlideshowImages": False,
                "commentsPerPost": 0,
            }

            if profiles:
                run_input["profiles"] = profiles
            elif hashtags:
                run_input["hashtags"] = hashtags
            elif search_queries:
                run_input["searchQueries"] = search_queries
            else:
                run_input["hashtags"] = keywords[:5] if keywords else [brand_name]

            return run_input

        elif platform == PlatformType.TWITTER:
            # apidojo/tweet-scraper 입력 형식
            twitter_handles = brand_config.get("twitterHandles", [])
            search_terms = brand_config.get("searchTerms", [])

            run_input: Dict[str, Any] = {
                "maxItems": max_items,
                "sort": "Latest",
            }

            if twitter_handles:
                run_input["twitterHandles"] = twitter_handles
            elif search_terms:
                run_input["searchTerms"] = search_terms
            else:
                run_input["searchTerms"] = keywords[:5] if keywords else [brand_name]

            return run_input

        return {"limit": max_items}


class TransformStageExecutor(BaseStageExecutor):
    """변환 단계 실행기"""

    def __init__(self, adapters: Dict[PlatformType, BaseSNSAdapter]):
        super().__init__(PipelineStage.TRANSFORM)
        self.adapters = adapters

    async def execute(self, ctx: PipelineContext) -> StageResult:
        """원시 데이터를 DTO로 변환"""
        start_time = time.time()

        if not ctx.raw_data:
            logger.warning("[transform] No raw data to transform")
            return StageResult(
                stage=self.stage,
                success=True,
                duration_seconds=0,
                items_processed=0,
            )

        adapter = self.adapters.get(ctx.platform)
        if not adapter:
            return StageResult(
                stage=self.stage,
                success=False,
                duration_seconds=0,
                error=f"No adapter for platform: {ctx.platform.value}",
            )

        transformed_data = []
        actors = []
        all_interactions = []
        contents = []

        for item in ctx.raw_data:
            try:
                # 유효성 검증
                if not adapter.validate_raw_data(item):
                    logger.debug(f"Invalid raw data item: {item.get('id', 'unknown')}")
                    continue

                # 변환
                result = adapter.transform(item)

                actor: ActorDTO = result["actor"]
                content: ContentDTO = result["content"]
                interactions: List[InteractionDTO] = result["interactions"]

                transformed_data.append(result)
                actors.append(actor)
                contents.append(content)
                all_interactions.extend(interactions)

            except Exception as e:
                logger.warning(f"Transform failed for item: {e}")
                continue

        # 컨텍스트 업데이트
        ctx.transformed_data = transformed_data
        ctx.actors = actors
        ctx.filtered_contents = contents  # 필터링 전 임시 저장
        ctx.interactions = all_interactions

        duration = time.time() - start_time

        logger.info(f"[transform] Transformed {len(transformed_data)} items")
        logger.info(f"  Actors: {len(actors)}, Contents: {len(contents)}, Interactions: {len(all_interactions)}")

        return StageResult(
            stage=self.stage,
            success=True,
            duration_seconds=duration,
            items_processed=len(transformed_data),
        )


class FilterStageExecutor(BaseStageExecutor):
    """필터링 단계 실행기"""

    def __init__(self, neo4j_repo: Neo4jRepository):
        super().__init__(PipelineStage.FILTER)
        self.neo4j_repo = neo4j_repo

    async def execute(self, ctx: PipelineContext) -> StageResult:
        """중복 콘텐츠 필터링"""
        start_time = time.time()

        # 필터링 스킵 옵션
        if ctx.options.get("skip_filter"):
            logger.info("[filter] Skipped by option")
            return StageResult(
                stage=self.stage,
                success=True,
                duration_seconds=0,
                items_processed=len(ctx.filtered_contents),
            )

        if not ctx.filtered_contents:
            logger.warning("[filter] No contents to filter")
            return StageResult(
                stage=self.stage,
                success=True,
                duration_seconds=0,
                items_processed=0,
            )

        # 새 콘텐츠만 필터링
        original_count = len(ctx.filtered_contents)
        new_contents = self.neo4j_repo.filter_new_contents(
            contents=ctx.filtered_contents,
            brand_id=ctx.brand_id,
        )

        # 필터링된 콘텐츠에 해당하는 transformed_data만 유지
        new_content_ids = {c.content_id for c in new_contents}
        ctx.transformed_data = [
            d for d in ctx.transformed_data
            if d["content"].content_id in new_content_ids
        ]

        # 인터랙션도 필터링
        ctx.interactions = [
            i for i in ctx.interactions
            if i.content_id in new_content_ids
        ]

        ctx.filtered_contents = new_contents

        duration = time.time() - start_time
        filtered_out = original_count - len(new_contents)

        logger.info(f"[filter] {len(new_contents)} new contents (filtered out {filtered_out} duplicates)")

        return StageResult(
            stage=self.stage,
            success=True,
            duration_seconds=duration,
            items_processed=len(new_contents),
        )


class ProcessStageExecutor(BaseStageExecutor):
    """처리(LLM) 단계 실행기"""

    def __init__(self, llm_processor: LLMProcessor):
        super().__init__(PipelineStage.PROCESS)
        self.llm_processor = llm_processor

    async def execute(self, ctx: PipelineContext) -> StageResult:
        """LLM을 사용하여 지식그래프 생성"""
        start_time = time.time()

        # LLM 스킵 옵션
        if ctx.options.get("skip_llm"):
            logger.info("[process] LLM processing skipped by option")
            return StageResult(
                stage=self.stage,
                success=True,
                duration_seconds=0,
                items_processed=0,
            )

        if not ctx.filtered_contents:
            logger.warning("[process] No contents to process")
            return StageResult(
                stage=self.stage,
                success=True,
                duration_seconds=0,
                items_processed=0,
            )

        # 드라이런 모드
        if ctx.is_dry_run:
            logger.info("[process] Dry run mode - skipping LLM processing")
            return StageResult(
                stage=self.stage,
                success=True,
                duration_seconds=0,
                items_processed=len(ctx.filtered_contents),
            )

        # 배치 처리
        batch_size = ctx.options.get("batch_size", PipelineConfig.DEFAULT_BATCH_SIZE)
        graph_results = []

        for i in range(0, len(ctx.filtered_contents), batch_size):
            batch_contents = ctx.filtered_contents[i:i + batch_size]
            batch_interactions = [
                interaction for interaction in ctx.interactions
                if any(interaction.content_id == c.content_id for c in batch_contents)
            ]
            batch_actors = [
                c.author for c in batch_contents
                if c.author
            ]

            logger.info(f"[process] Processing batch {i // batch_size + 1} ({len(batch_contents)} items)")

            try:
                result = self.llm_processor.process_contents(
                    actors=batch_actors,
                    contents=batch_contents,
                    interactions=batch_interactions,
                    topics=ctx.topics,
                    brand_config=ctx.brand_config,
                )
                graph_results.append(result)

                logger.info(f"  Nodes: {result.nodes_count}, Relationships: {result.relationships_count}")

            except Exception as e:
                logger.error(f"  Batch processing failed: {e}")
                ctx.add_error(self.stage.value, f"Batch {i // batch_size + 1} failed: {e}")
                continue

        ctx.graph_results = graph_results
        duration = time.time() - start_time

        total_nodes = sum(r.nodes_count for r in graph_results)
        total_rels = sum(r.relationships_count for r in graph_results)

        logger.info(f"[process] Created {total_nodes} nodes, {total_rels} relationships")

        return StageResult(
            stage=self.stage,
            success=True,
            duration_seconds=duration,
            items_processed=len(ctx.filtered_contents),
        )


class SaveStageExecutor(BaseStageExecutor):
    """저장 단계 실행기"""

    def __init__(self, neo4j_repo: Neo4jRepository):
        super().__init__(PipelineStage.SAVE)
        self.neo4j_repo = neo4j_repo

    async def execute(self, ctx: PipelineContext) -> StageResult:
        """Neo4j에 데이터 저장"""
        start_time = time.time()

        # 드라이런 모드
        if ctx.is_dry_run:
            logger.info("[save] Dry run mode - skipping save")
            return StageResult(
                stage=self.stage,
                success=True,
                duration_seconds=0,
                items_processed=0,
            )

        # 1. Content 노드 저장 (원본 콘텐츠 + 메트릭스)
        content_saved = 0
        if ctx.filtered_contents:
            logger.info(f"[save] Saving {len(ctx.filtered_contents)} Content nodes with metrics...")
            for content in ctx.filtered_contents:
                try:
                    await self.neo4j_repo.create_content(content, ctx.brand_id)
                    content_saved += 1
                except Exception as e:
                    logger.warning(f"  Content save failed for {content.content_id}: {e}")
            logger.info(f"[save] Content nodes saved: {content_saved}/{len(ctx.filtered_contents)}")

        # 2. Graph 결과 저장 (Concept 노드 등)
        if not ctx.graph_results:
            logger.warning("[save] No graph results to save")
            return StageResult(
                stage=self.stage,
                success=True,
                duration_seconds=time.time() - start_time,
                items_processed=content_saved,
            )

        save_results = []
        use_batch = ctx.options.get("use_batch", True)

        for idx, graph_result in enumerate(ctx.graph_results):
            logger.info(f"[save] Saving batch {idx + 1}/{len(ctx.graph_results)}")

            try:
                result = self.neo4j_repo.save_graph_documents(
                    graph_documents=graph_result.graph_documents,
                    brand_id=ctx.brand_id,
                    use_batch=use_batch,
                )
                save_results.append(result)

                logger.info(f"  Nodes: {result['nodes_created']} created, {result['nodes_updated']} updated")
                logger.info(f"  Relationships: {result['relationships_created']} created")

            except Exception as e:
                logger.error(f"  Save failed: {e}")
                ctx.add_error(self.stage.value, f"Batch {idx + 1} save failed: {e}")
                save_results.append({
                    "success": False,
                    "nodes_created": 0,
                    "nodes_updated": 0,
                    "relationships_created": 0,
                    "relationships_updated": 0,
                    "errors": [str(e)],
                })

        ctx.save_results = save_results
        duration = time.time() - start_time

        total_nodes = sum(r.get("nodes_created", 0) + r.get("nodes_updated", 0) for r in save_results)
        total_rels = sum(r.get("relationships_created", 0) for r in save_results)

        logger.info(f"[save] Total saved: {total_nodes} nodes, {total_rels} relationships, {content_saved} contents")

        return StageResult(
            stage=self.stage,
            success=all(r.get("success", False) for r in save_results),
            duration_seconds=duration,
            items_processed=len(ctx.graph_results) + content_saved,
        )


# ============================================================================
# PIPELINE HOOKS
# ============================================================================

class PipelineHook(Protocol):
    """파이프라인 훅 프로토콜"""

    async def before_stage(self, stage: PipelineStage, ctx: PipelineContext) -> None:
        """단계 실행 전"""
        ...

    async def after_stage(self, stage: PipelineStage, ctx: PipelineContext, result: StageResult) -> None:
        """단계 실행 후"""
        ...

    async def on_error(self, stage: PipelineStage, ctx: PipelineContext, error: Exception) -> None:
        """에러 발생 시"""
        ...

    async def on_complete(self, ctx: PipelineContext) -> None:
        """파이프라인 완료 시"""
        ...


class LoggingHook:
    """로깅 훅 구현"""

    async def before_stage(self, stage: PipelineStage, ctx: PipelineContext) -> None:
        logger.info(f"{'=' * 60}")
        logger.info(f"Stage: {stage.value.upper()} - Starting")
        logger.info(f"Brand: {ctx.brand_id}, Platform: {ctx.platform.value}")

    async def after_stage(self, stage: PipelineStage, ctx: PipelineContext, result: StageResult) -> None:
        status = "SUCCESS" if result.success else "FAILED"
        logger.info(f"Stage: {stage.value.upper()} - {status}")
        logger.info(f"Duration: {result.duration_seconds:.2f}s, Items: {result.items_processed}")
        if result.error:
            logger.error(f"Error: {result.error}")

    async def on_error(self, stage: PipelineStage, ctx: PipelineContext, error: Exception) -> None:
        logger.error(f"Pipeline error at {stage.value}: {error}")
        logger.debug(traceback.format_exc())

    async def on_complete(self, ctx: PipelineContext) -> None:
        stats = ctx.get_statistics()
        logger.info(f"{'=' * 60}")
        logger.info("PIPELINE COMPLETED")
        logger.info(f"Success: {stats['success']}")
        logger.info(f"Duration: {stats['duration_seconds']:.2f}s")
        logger.info(f"Crawled: {stats['crawled_count']}, Processed: {stats['processed_count']}")
        logger.info(f"Saved: {stats['saved_nodes']} nodes, {stats['saved_relationships']} relationships")
        if stats['errors']:
            logger.warning(f"Errors: {len(stats['errors'])}")


# ============================================================================
# MAIN PIPELINE CLASS
# ============================================================================

class SNSDataPipeline:
    """
    프로덕션급 SNS 데이터 파이프라인 오케스트레이터.

    ONTIX Universal의 범용 SNS 데이터 수집 및 지식그래프 생성 파이프라인입니다.
    5단계(Crawl → Transform → Filter → Process → Save)로 구성되며,
    각 단계는 독립적으로 실행 가능합니다.

    Attributes:
        apify_client: Apify API 클라이언트
        neo4j_repo: Neo4j 저장소
        llm_processor: LLM 프로세서
        adapters: 플랫폼별 어댑터

    Example:
        >>> pipeline = SNSDataPipeline()
        >>> stats = await pipeline.run(
        ...     platform=PlatformType.INSTAGRAM,
        ...     brand_config={"id": "my_brand", "name": "My Brand"},
        ...     options={"max_items": 100}
        ... )
        >>> print(f"Processed {stats['processed_count']} items")
    """

    def __init__(
        self,
        apify_token: Optional[str] = None,
        neo4j_uri: Optional[str] = None,
        neo4j_username: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        llm_model: str = None,  # 무시됨, gpt-4o-mini 고정
    ) -> None:
        """
        파이프라인 초기화.

        Args:
            apify_token: Apify API 토큰. None이면 환경변수에서 로드.
            neo4j_uri: Neo4j URI. None이면 환경변수에서 로드.
            neo4j_username: Neo4j 사용자명. None이면 환경변수에서 로드.
            neo4j_password: Neo4j 비밀번호. None이면 환경변수에서 로드.
            openai_api_key: OpenAI API 키. None이면 환경변수에서 로드.
            llm_model: 무시됨 (gpt-4o-mini 고정 사용).
        """
        logger.info("Initializing SNS Data Pipeline...")

        # Apify 클라이언트
        self.apify_client = ApifyClient(apify_token)

        # Neo4j 저장소
        self.neo4j_repo = Neo4jRepository(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
        )

        # LLM 프로세서 (gpt-4o-mini 고정)
        self.llm_processor = LLMProcessor(
            api_key=openai_api_key,
        )

        # 플랫폼별 어댑터
        self.adapters: Dict[PlatformType, BaseSNSAdapter] = {
            PlatformType.INSTAGRAM: InstagramAdapter(),
            PlatformType.YOUTUBE: YouTubeAdapter(),
            PlatformType.TIKTOK: TikTokAdapter(),
            PlatformType.TWITTER: TwitterAdapter(),
        }

        # 단계 실행기
        self._stage_executors: Dict[PipelineStage, BaseStageExecutor] = {
            PipelineStage.CRAWL: CrawlStageExecutor(self.apify_client),
            PipelineStage.TRANSFORM: TransformStageExecutor(self.adapters),
            PipelineStage.FILTER: FilterStageExecutor(self.neo4j_repo),
            PipelineStage.PROCESS: ProcessStageExecutor(self.llm_processor),
            PipelineStage.SAVE: SaveStageExecutor(self.neo4j_repo),
        }

        # 훅
        self._hooks: List[PipelineHook] = [LoggingHook()]

        logger.info("SNS Data Pipeline initialized successfully")

    def add_hook(self, hook: PipelineHook) -> None:
        """훅 추가"""
        self._hooks.append(hook)

    async def _run_hooks_before(self, stage: PipelineStage, ctx: PipelineContext) -> None:
        """before 훅 실행"""
        for hook in self._hooks:
            try:
                await hook.before_stage(stage, ctx)
            except Exception as e:
                logger.warning(f"Hook error (before {stage.value}): {e}")

    async def _run_hooks_after(self, stage: PipelineStage, ctx: PipelineContext, result: StageResult) -> None:
        """after 훅 실행"""
        for hook in self._hooks:
            try:
                await hook.after_stage(stage, ctx, result)
            except Exception as e:
                logger.warning(f"Hook error (after {stage.value}): {e}")

    async def _run_hooks_error(self, stage: PipelineStage, ctx: PipelineContext, error: Exception) -> None:
        """error 훅 실행"""
        for hook in self._hooks:
            try:
                await hook.on_error(stage, ctx, error)
            except Exception as e:
                logger.warning(f"Hook error (on_error {stage.value}): {e}")

    async def _run_hooks_complete(self, ctx: PipelineContext) -> None:
        """complete 훅 실행"""
        for hook in self._hooks:
            try:
                await hook.on_complete(ctx)
            except Exception as e:
                logger.warning(f"Hook error (on_complete): {e}")

    async def run_stage(
        self,
        stage: PipelineStage,
        ctx: PipelineContext,
    ) -> StageResult:
        """
        단일 단계 실행.

        Args:
            stage: 실행할 단계
            ctx: 파이프라인 컨텍스트

        Returns:
            StageResult: 단계 실행 결과
        """
        executor = self._stage_executors.get(stage)
        if not executor:
            raise ValueError(f"Unknown stage: {stage}")

        try:
            await self._run_hooks_before(stage, ctx)
            result = await executor.execute_with_retry(ctx)
            await self._run_hooks_after(stage, ctx, result)
            return result

        except Exception as e:
            await self._run_hooks_error(stage, ctx, e)
            ctx.add_error(stage.value, str(e))
            return StageResult(
                stage=stage,
                success=False,
                duration_seconds=0,
                error=str(e),
            )

    async def run(
        self,
        platform: PlatformType,
        brand_config: BrandConfig,
        options: Optional[PipelineOptions] = None,
        raw_data: Optional[List[Dict[str, Any]]] = None,
    ) -> PipelineStatistics:
        """
        전체 파이프라인 실행.

        5단계(Crawl → Transform → Filter → Process → Save)를 순차적으로 실행합니다.

        Args:
            platform: 플랫폼 타입
            brand_config: 브랜드 설정
            options: 파이프라인 실행 옵션
            raw_data: 사전 크롤링된 데이터 (옵션)

        Returns:
            PipelineStatistics: 실행 통계

        Example:
            >>> stats = await pipeline.run(
            ...     platform=PlatformType.INSTAGRAM,
            ...     brand_config={"id": "brand", "name": "Brand"},
            ...     options={"max_items": 50, "use_batch": True}
            ... )
        """
        logger.info(f"Starting pipeline for {platform.value}")
        logger.info(f"Brand: {brand_config.get('name', 'Unknown')} ({brand_config.get('id', 'unknown')})")

        # 컨텍스트 생성
        ctx = PipelineContext(
            brand_config=brand_config,
            platform=platform,
            options=options or {},
        )

        # 사전 크롤링 데이터가 있으면 설정
        if raw_data:
            ctx.raw_data = raw_data
            logger.info(f"Using pre-crawled data: {len(raw_data)} items")

        # 단계별 실행
        stages = [
            PipelineStage.CRAWL,
            PipelineStage.TRANSFORM,
            PipelineStage.FILTER,
            PipelineStage.PROCESS,
            PipelineStage.SAVE,
        ]

        for stage in stages:
            # 이전 단계가 실패하면 중단
            if ctx.errors and stage != PipelineStage.CRAWL:
                logger.warning(f"Skipping {stage.value} due to previous errors")
                continue

            # 사전 크롤링 데이터가 있으면 크롤링 스킵
            if stage == PipelineStage.CRAWL and raw_data:
                logger.info("Skipping crawl stage (using pre-crawled data)")
                continue

            result = await self.run_stage(stage, ctx)

            if not result.success:
                logger.error(f"Stage {stage.value} failed: {result.error}")
                # 계속 진행 여부 결정 (현재는 계속 진행)

        # 완료 훅 실행
        await self._run_hooks_complete(ctx)

        return ctx.get_statistics()

    async def run_from_dataset(
        self,
        dataset_id: str,
        platform: PlatformType,
        brand_config: BrandConfig,
        options: Optional[PipelineOptions] = None,
    ) -> PipelineStatistics:
        """
        Apify 데이터셋에서 데이터를 로드하여 파이프라인 실행.

        Args:
            dataset_id: Apify 데이터셋 ID
            platform: 플랫폼 타입
            brand_config: 브랜드 설정
            options: 파이프라인 실행 옵션

        Returns:
            PipelineStatistics: 실행 통계
        """
        logger.info(f"Loading data from dataset: {dataset_id}")

        # 데이터셋에서 데이터 로드
        raw_data = await self.apify_client.get_dataset_items(dataset_id)
        logger.info(f"Loaded {len(raw_data)} items from dataset")

        # 파이프라인 실행
        return await self.run(
            platform=platform,
            brand_config=brand_config,
            options=options,
            raw_data=raw_data,
        )

    async def run_transform_only(
        self,
        platform: PlatformType,
        raw_data: List[Dict[str, Any]],
        brand_config: BrandConfig,
    ) -> Dict[str, Any]:
        """
        변환 단계만 실행.

        Args:
            platform: 플랫폼 타입
            raw_data: 원시 데이터
            brand_config: 브랜드 설정

        Returns:
            변환 결과 딕셔너리
        """
        ctx = PipelineContext(
            brand_config=brand_config,
            platform=platform,
            raw_data=raw_data,
        )

        result = await self.run_stage(PipelineStage.TRANSFORM, ctx)

        return {
            "success": result.success,
            "actors": ctx.actors,
            "contents": ctx.filtered_contents,
            "interactions": ctx.interactions,
            "transformed_count": result.items_processed,
        }

    async def run_process_only(
        self,
        contents: List[ContentDTO],
        interactions: List[InteractionDTO],
        brand_config: BrandConfig,
    ) -> List[GraphProcessingResult]:
        """
        LLM 처리 단계만 실행.

        Args:
            contents: 콘텐츠 DTO 리스트
            interactions: 인터랙션 DTO 리스트
            brand_config: 브랜드 설정

        Returns:
            GraphProcessingResult 리스트
        """
        platform = contents[0].platform if contents else PlatformType.INSTAGRAM

        ctx = PipelineContext(
            brand_config=brand_config,
            platform=platform,
            filtered_contents=contents,
            interactions=interactions,
        )

        await self.run_stage(PipelineStage.PROCESS, ctx)

        return ctx.graph_results

    def get_adapter(self, platform: PlatformType) -> BaseSNSAdapter:
        """플랫폼 어댑터 반환"""
        adapter = self.adapters.get(platform)
        if not adapter:
            raise ValueError(f"Unsupported platform: {platform}")
        return adapter

    def get_statistics(self, brand_id: str) -> Dict[str, Any]:
        """브랜드 통계 조회"""
        return self.neo4j_repo.get_brand_statistics(brand_id)

    def get_concept_cloud(self, brand_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """개념 클라우드 조회"""
        return self.neo4j_repo.get_concept_cloud(brand_id, limit)

    async def close(self) -> None:
        """리소스 정리"""
        logger.info("Closing pipeline resources")
        self.neo4j_repo.close()


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_pipeline(
    **kwargs: Any,
) -> SNSDataPipeline:
    """
    파이프라인 팩토리 함수.

    환경에 따라 적절한 설정으로 파이프라인을 생성합니다.

    Args:
        **kwargs: SNSDataPipeline 추가 인자

    Returns:
        설정된 SNSDataPipeline 인스턴스

    Example:
        >>> pipeline = create_pipeline(llm_model="gpt-4o")
    """
    return SNSDataPipeline(**kwargs)


async def run_pipeline(
    platform: PlatformType,
    brand_config: BrandConfig,
    options: Optional[PipelineOptions] = None,
    **pipeline_kwargs: Any,
) -> PipelineStatistics:
    """
    파이프라인 실행 헬퍼 함수.

    파이프라인을 생성하고 실행한 후 리소스를 정리합니다.

    Args:
        platform: 플랫폼 타입
        brand_config: 브랜드 설정
        options: 파이프라인 옵션
        **pipeline_kwargs: SNSDataPipeline 추가 인자

    Returns:
        PipelineStatistics: 실행 통계

    Example:
        >>> stats = await run_pipeline(
        ...     platform=PlatformType.INSTAGRAM,
        ...     brand_config={"id": "brand", "name": "Brand"}
        ... )
    """
    pipeline = create_pipeline(**pipeline_kwargs)

    try:
        return await pipeline.run(
            platform=platform,
            brand_config=brand_config,
            options=options,
        )
    finally:
        await pipeline.close()


# ============================================================================
# MULTI-PLATFORM PIPELINE
# ============================================================================

class MultiPlatformPipeline:
    """
    멀티 플랫폼 파이프라인.

    여러 플랫폼의 데이터를 동시에 수집하고 처리합니다.
    """

    def __init__(self, pipeline: Optional[SNSDataPipeline] = None, **kwargs):
        """
        멀티 플랫폼 파이프라인 초기화.

        Args:
            pipeline: 기존 파이프라인 인스턴스 (재사용)
            **kwargs: SNSDataPipeline 생성 인자
        """
        self.pipeline = pipeline or SNSDataPipeline(**kwargs)
        self._results: Dict[PlatformType, PipelineStatistics] = {}

    async def run_all_platforms(
        self,
        brand_config: BrandConfig,
        platforms: Optional[List[PlatformType]] = None,
        options: Optional[PipelineOptions] = None,
        parallel: bool = False,
    ) -> Dict[PlatformType, PipelineStatistics]:
        """
        여러 플랫폼 파이프라인 실행.

        Args:
            brand_config: 브랜드 설정
            platforms: 실행할 플랫폼 리스트. None이면 전체.
            options: 파이프라인 옵션
            parallel: True면 병렬 실행

        Returns:
            플랫폼별 실행 통계 딕셔너리
        """
        if platforms is None:
            platforms = list(PlatformType)

        logger.info(f"Running multi-platform pipeline for {len(platforms)} platforms")

        if parallel:
            results = await self._run_parallel(brand_config, platforms, options)
        else:
            results = await self._run_sequential(brand_config, platforms, options)

        self._results = results
        return results

    async def _run_sequential(
        self,
        brand_config: BrandConfig,
        platforms: List[PlatformType],
        options: Optional[PipelineOptions],
    ) -> Dict[PlatformType, PipelineStatistics]:
        """순차 실행"""
        results = {}

        for platform in platforms:
            logger.info(f"Processing platform: {platform.value}")
            try:
                stats = await self.pipeline.run(
                    platform=platform,
                    brand_config=brand_config,
                    options=options,
                )
                results[platform] = stats
            except Exception as e:
                logger.error(f"Failed to process {platform.value}: {e}")
                results[platform] = self._create_error_stats(brand_config, platform, str(e))

        return results

    async def _run_parallel(
        self,
        brand_config: BrandConfig,
        platforms: List[PlatformType],
        options: Optional[PipelineOptions],
    ) -> Dict[PlatformType, PipelineStatistics]:
        """병렬 실행"""
        semaphore = asyncio.Semaphore(PipelineConfig.MAX_CONCURRENT_TASKS)

        async def run_with_semaphore(platform: PlatformType):
            async with semaphore:
                return platform, await self.pipeline.run(
                    platform=platform,
                    brand_config=brand_config,
                    options=options,
                )

        tasks = [run_with_semaphore(p) for p in platforms]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        results = {}
        for item in completed:
            if isinstance(item, Exception):
                logger.error(f"Platform processing failed: {item}")
                continue
            platform, stats = item
            results[platform] = stats

        return results

    def _create_error_stats(
        self,
        brand_config: BrandConfig,
        platform: PlatformType,
        error: str,
    ) -> PipelineStatistics:
        """에러 통계 생성"""
        return {
            "brand_id": brand_config.get("id", "unknown"),
            "platform": platform.value,
            "started_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": 0,
            "stage_timings": {},
            "crawled_count": 0,
            "transformed_count": 0,
            "filtered_count": 0,
            "processed_count": 0,
            "saved_nodes": 0,
            "saved_relationships": 0,
            "errors": [error],
            "success": False,
        }

    def get_summary(self) -> Dict[str, Any]:
        """멀티 플랫폼 실행 요약"""
        if not self._results:
            return {"total_platforms": 0, "success_count": 0, "error_count": 0}

        success_count = sum(1 for r in self._results.values() if r["success"])
        total_nodes = sum(r["saved_nodes"] for r in self._results.values())
        total_rels = sum(r["saved_relationships"] for r in self._results.values())

        return {
            "total_platforms": len(self._results),
            "success_count": success_count,
            "error_count": len(self._results) - success_count,
            "total_nodes_saved": total_nodes,
            "total_relationships_saved": total_rels,
            "platforms": {p.value: r["success"] for p, r in self._results.items()},
        }

    async def close(self) -> None:
        """리소스 정리"""
        await self.pipeline.close()


# ============================================================================
# CLI SUPPORT
# ============================================================================

async def main():
    """CLI 엔트리포인트"""
    import argparse

    parser = argparse.ArgumentParser(description="ONTIX Universal SNS Data Pipeline")
    parser.add_argument("--platform", type=str, required=True, help="Platform (instagram/youtube/tiktok/twitter)")
    parser.add_argument("--brand-id", type=str, required=True, help="Brand ID")
    parser.add_argument("--brand-name", type=str, required=True, help="Brand name")
    parser.add_argument("--max-items", type=int, default=50, help="Max items to crawl")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM processing")

    args = parser.parse_args()

    # 플랫폼 매핑
    platform_map = {
        "instagram": PlatformType.INSTAGRAM,
        "youtube": PlatformType.YOUTUBE,
        "tiktok": PlatformType.TIKTOK,
        "twitter": PlatformType.TWITTER,
    }

    platform = platform_map.get(args.platform.lower())
    if not platform:
        print(f"Unknown platform: {args.platform}")
        return

    # 파이프라인 실행
    stats = await run_pipeline(
        platform=platform,
        brand_config={
            "id": args.brand_id,
            "name": args.brand_name,
        },
        options={
            "max_items": args.max_items,
            "dry_run": args.dry_run,
            "skip_llm": args.skip_llm,
        },
    )

    # 결과 출력
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETED")
    print("=" * 60)
    print(f"Success: {stats['success']}")
    print(f"Duration: {stats['duration_seconds']:.2f}s")
    print(f"Crawled: {stats['crawled_count']}")
    print(f"Processed: {stats['processed_count']}")
    print(f"Saved: {stats['saved_nodes']} nodes, {stats['saved_relationships']} relationships")

    if stats["errors"]:
        print(f"\nErrors ({len(stats['errors'])}):")
        for error in stats["errors"]:
            print(f"  - {error}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(main())
