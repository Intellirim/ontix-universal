"""
Pipeline API - SNS 데이터 파이프라인 실행 엔드포인트
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
import asyncio
import logging

from app.data_pipeline.pipeline import SNSDataPipeline, PlatformType, PipelineStatistics
from app.services.platform.config_manager import ConfigManager
from app.services.crawlers.web_crawler import get_web_crawler_service, WebCrawlRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pipeline")

# 파이프라인 실행 상태 저장 (실제 프로덕션에서는 Redis 등 사용)
pipeline_jobs: Dict[str, Dict[str, Any]] = {}


class PlatformEnum(str, Enum):
    instagram = "instagram"
    youtube = "youtube"
    tiktok = "tiktok"
    twitter = "twitter"
    website = "website"  # 웹사이트 크롤러


class TargetTypeEnum(str, Enum):
    accounts = "accounts"
    hashtags = "hashtags"
    search = "search"
    urls = "urls"  # 웹사이트 URL


class PipelineRunRequest(BaseModel):
    """파이프라인 실행 요청"""
    brand_id: str = Field(..., description="브랜드 ID")
    platform: PlatformEnum = Field(..., description="플랫폼")
    target_type: TargetTypeEnum = Field(..., description="수집 대상 유형 (accounts/hashtags/search)")
    targets: List[str] = Field(..., min_length=1, description="수집 대상 목록 (계정명, 해시태그, 검색어)")
    max_items: int = Field(default=50, ge=1, le=500, description="최대 수집 아이템 수")
    dry_run: bool = Field(default=False, description="테스트 모드 (실제 저장 안함)")
    skip_crawl: bool = Field(default=False, description="크롤링 스킵 (기존 데이터 사용)")
    skip_llm: bool = Field(default=False, description="LLM 처리 스킵")


class PipelineRunResponse(BaseModel):
    """파이프라인 실행 응답"""
    job_id: str
    status: str
    message: str


class PipelineStatusResponse(BaseModel):
    """파이프라인 상태 응답"""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: Optional[str] = None
    statistics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PipelineJobListResponse(BaseModel):
    """파이프라인 작업 목록"""
    jobs: List[Dict[str, Any]]


# 글로벌 파이프라인 인스턴스 (싱글톤)
_pipeline_instance: Optional[SNSDataPipeline] = None


def get_pipeline() -> SNSDataPipeline:
    """파이프라인 인스턴스 반환"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = SNSDataPipeline()
    return _pipeline_instance


async def run_pipeline_task(job_id: str, request: PipelineRunRequest):
    """백그라운드 파이프라인 실행 태스크"""
    try:
        pipeline_jobs[job_id]["status"] = "running"
        pipeline_jobs[job_id]["progress"] = "Initializing..."

        # 브랜드 설정 로드
        try:
            brand_config = ConfigManager.load_brand_config(request.brand_id)
        except FileNotFoundError:
            raise ValueError(f"Brand not found: {request.brand_id}")

        # Website 크롤러 처리 (별도 로직)
        if request.platform.value == "website":
            await _run_website_crawler(job_id, request)
            return

        # 타겟 유형에 따른 크롤링 입력 구성
        crawl_input = _build_crawl_input(
            platform=request.platform.value,
            target_type=request.target_type.value,
            targets=request.targets,
            max_items=request.max_items,
        )

        # brand_config 구조: {"brand": {"id", "name", ...}, "neo4j": {...}, ...}
        brand_info = brand_config.get("brand", {})
        brand_dict = {
            "id": request.brand_id,
            "name": brand_info.get("name", request.brand_id),
            "category": brand_info.get("industry", ""),
            # 크롤링 입력을 브랜드 설정에 포함
            **crawl_input,
        }

        # 플랫폼 매핑
        platform_map = {
            "instagram": PlatformType.INSTAGRAM,
            "youtube": PlatformType.YOUTUBE,
            "tiktok": PlatformType.TIKTOK,
            "twitter": PlatformType.TWITTER,
        }
        platform = platform_map[request.platform.value]

        targets_str = ", ".join(request.targets[:3])
        if len(request.targets) > 3:
            targets_str += f" (+{len(request.targets) - 3} more)"
        pipeline_jobs[job_id]["progress"] = f"Crawling {request.target_type.value}: {targets_str}"

        # 파이프라인 실행
        pipeline = get_pipeline()
        stats = await pipeline.run(
            platform=platform,
            brand_config=brand_dict,
            options={
                "max_items": request.max_items,
                "dry_run": request.dry_run,
                "skip_crawl": request.skip_crawl,
                "skip_llm": request.skip_llm,
            }
        )

        pipeline_jobs[job_id]["status"] = "completed"
        pipeline_jobs[job_id]["progress"] = "Done"
        pipeline_jobs[job_id]["statistics"] = stats

        logger.info(f"Pipeline job {job_id} completed: {stats['saved_nodes']} nodes saved")

    except Exception as e:
        logger.error(f"Pipeline job {job_id} failed: {e}")
        pipeline_jobs[job_id]["status"] = "failed"
        pipeline_jobs[job_id]["error"] = str(e)


async def _run_website_crawler(job_id: str, request: PipelineRunRequest):
    """웹사이트 크롤러 실행"""
    try:
        pipeline_jobs[job_id]["progress"] = "Starting website crawler..."

        crawler = get_web_crawler_service()

        # URL 정리 (줄바꿈, 쉼표 구분 처리)
        urls = []
        for target in request.targets:
            # 줄바꿈으로 분리된 경우
            for url in target.replace(",", "\n").split("\n"):
                url = url.strip()
                if url and url.startswith("http"):
                    urls.append(url)

        if not urls:
            raise ValueError("No valid URLs provided")

        crawl_request = WebCrawlRequest(
            urls=urls,
            max_pages=request.max_items,
            max_depth=3,
            save_markdown=True,
        )

        pipeline_jobs[job_id]["progress"] = f"Crawling {len(urls)} URL(s)..."

        # 크롤링 실행 (완료까지 대기)
        job = await crawler.start_crawl(
            request=crawl_request,
            brand_id=request.brand_id,
            wait_for_finish=True,
        )

        if job.status == "completed" and job.result:
            # Neo4j에 저장
            if not request.dry_run:
                pipeline_jobs[job_id]["progress"] = "Saving to Neo4j..."

                from app.services.shared.neo4j import get_neo4j_client
                neo4j = get_neo4j_client()
                saved_count = 0

                for page in job.result.pages:
                    if not page.text and not page.markdown:
                        continue

                    content = page.markdown or page.text
                    neo4j.execute_write("""
                        MERGE (w:WebContent {url: $url})
                        SET w.title = $title,
                            w.content = $content,
                            w.brand_id = $brand_id,
                            w.crawled_at = datetime($crawled_at),
                            w.updated_at = datetime()
                        WITH w
                        MATCH (b:Brand {id: $brand_id})
                        MERGE (b)-[:HAS_WEB_CONTENT]->(w)
                    """, {
                        "url": page.url,
                        "title": page.title or "",
                        "content": content[:10000] if content else "",
                        "brand_id": request.brand_id,
                        "crawled_at": page.crawled_at.isoformat()
                    })
                    saved_count += 1

            pipeline_jobs[job_id]["status"] = "completed"
            pipeline_jobs[job_id]["progress"] = "Done"
            pipeline_jobs[job_id]["statistics"] = {
                "brand_id": request.brand_id,
                "platform": "website",
                "crawled_count": job.result.total_pages,
                "saved_nodes": saved_count if not request.dry_run else 0,
                "duration_seconds": (job.completed_at - job.started_at).total_seconds() if job.completed_at and job.started_at else 0,
                "success": True,
            }

            logger.info(f"Website crawler job {job_id} completed: {job.result.total_pages} pages crawled")
        else:
            raise ValueError(f"Crawler failed: {job.error or 'Unknown error'}")

    except Exception as e:
        logger.error(f"Website crawler job {job_id} failed: {e}")
        pipeline_jobs[job_id]["status"] = "failed"
        pipeline_jobs[job_id]["error"] = str(e)


def _build_crawl_input(
    platform: str,
    target_type: str,
    targets: List[str],
    max_items: int,
) -> Dict[str, Any]:
    """플랫폼 및 타겟 유형에 따른 크롤링 입력 생성"""

    # 타겟 정리 (@ 제거, # 제거 등)
    clean_targets = []
    for target in targets:
        target = target.strip()
        if target.startswith("@"):
            target = target[1:]
        if target.startswith("#"):
            target = target[1:]
        if target:
            clean_targets.append(target)

    result: Dict[str, Any] = {}

    if platform == "instagram":
        if target_type == "accounts":
            result["usernames"] = clean_targets
        elif target_type == "hashtags":
            result["hashtags"] = clean_targets
        else:  # search
            result["search"] = clean_targets

    elif platform == "youtube":
        if target_type == "accounts":
            result["channelHandles"] = clean_targets
        elif target_type == "hashtags":
            result["searchKeywords"] = [f"#{t}" for t in clean_targets]
        else:  # search
            result["searchKeywords"] = clean_targets

    elif platform == "tiktok":
        if target_type == "accounts":
            result["profiles"] = clean_targets
        elif target_type == "hashtags":
            result["hashtags"] = clean_targets
        else:  # search
            result["searchQueries"] = clean_targets

    elif platform == "twitter":
        if target_type == "accounts":
            result["twitterHandles"] = clean_targets
        elif target_type == "hashtags":
            result["searchTerms"] = [f"#{t}" for t in clean_targets]
        else:  # search
            result["searchTerms"] = clean_targets

    return result


@router.post("/run", response_model=PipelineRunResponse)
async def run_pipeline(request: PipelineRunRequest, background_tasks: BackgroundTasks):
    """
    SNS 데이터 파이프라인 실행

    백그라운드에서 실행되며, job_id로 상태를 조회할 수 있습니다.
    """
    import uuid
    job_id = str(uuid.uuid4())[:8]

    # 작업 등록
    pipeline_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "brand_id": request.brand_id,
        "platform": request.platform.value,
        "target_type": request.target_type.value,
        "targets": request.targets,
        "max_items": request.max_items,
        "created_at": __import__("datetime").datetime.now().isoformat(),
    }

    # 백그라운드 태스크 시작
    background_tasks.add_task(run_pipeline_task, job_id, request)

    return PipelineRunResponse(
        job_id=job_id,
        status="pending",
        message=f"Pipeline job started for {request.brand_id} on {request.platform.value}"
    )


@router.get("/status/{job_id}", response_model=PipelineStatusResponse)
async def get_pipeline_status(job_id: str):
    """파이프라인 작업 상태 조회"""
    if job_id not in pipeline_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = pipeline_jobs[job_id]
    return PipelineStatusResponse(
        job_id=job_id,
        status=job.get("status", "unknown"),
        progress=job.get("progress"),
        statistics=job.get("statistics"),
        error=job.get("error"),
    )


@router.get("/jobs", response_model=PipelineJobListResponse)
async def list_pipeline_jobs(brand_id: Optional[str] = None, limit: int = 20):
    """파이프라인 작업 목록 조회"""
    jobs = list(pipeline_jobs.values())

    if brand_id:
        jobs = [j for j in jobs if j.get("brand_id") == brand_id]

    # 최신순 정렬
    jobs = sorted(jobs, key=lambda x: x.get("created_at", ""), reverse=True)[:limit]

    return PipelineJobListResponse(jobs=jobs)


@router.delete("/jobs/{job_id}")
async def delete_pipeline_job(job_id: str):
    """파이프라인 작업 삭제"""
    if job_id not in pipeline_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    del pipeline_jobs[job_id]
    return {"message": "Job deleted", "job_id": job_id}


@router.get("/platforms")
async def get_supported_platforms():
    """지원 플랫폼 목록"""
    return {
        "platforms": [
            {"id": "instagram", "name": "Instagram", "icon": "instagram"},
            {"id": "youtube", "name": "YouTube", "icon": "youtube"},
            {"id": "tiktok", "name": "TikTok", "icon": "music"},
            {"id": "twitter", "name": "Twitter/X", "icon": "twitter"},
            {"id": "website", "name": "Website", "icon": "globe"},
        ]
    }
