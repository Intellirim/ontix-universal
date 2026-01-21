"""
Web Crawler API
웹사이트 크롤링 엔드포인트
"""

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from typing import Optional, List
from pydantic import BaseModel, Field
from app.services.crawlers.web_crawler import (
    WebCrawlerService,
    WebCrawlRequest,
    WebCrawlJob,
    WebCrawlResult,
    CrawledPage,
    CrawlerType,
    get_web_crawler_service
)
from app.core.auth import User, get_current_user, require_permission, Permission
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/crawler", tags=["Web Crawler"])


# ============================================
# Request Models
# ============================================

class StartCrawlRequest(BaseModel):
    """크롤링 시작 요청"""
    urls: List[str] = Field(..., description="크롤링할 URL 목록", min_length=1, max_length=10)
    brand_id: Optional[str] = Field(None, description="연결할 브랜드 ID")
    max_pages: int = Field(default=50, description="최대 페이지 수", ge=1, le=1000)
    max_depth: int = Field(default=3, description="최대 깊이", ge=1, le=10)
    crawler_type: CrawlerType = Field(default=CrawlerType.PLAYWRIGHT_ADAPTIVE)
    include_patterns: List[str] = Field(default_factory=list, description="포함 패턴")
    exclude_patterns: List[str] = Field(default_factory=list, description="제외 패턴")
    save_markdown: bool = Field(default=True)
    wait_for_finish: bool = Field(default=False, description="완료까지 대기 (동기 실행)")


# ============================================
# API Endpoints
# ============================================

@router.post("/start", response_model=dict)
async def start_crawl(
    request: StartCrawlRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(require_permission(Permission.PIPELINE_RUN))
):
    """
    웹 크롤링 시작

    브랜드 관련 웹사이트 콘텐츠를 크롤링합니다.
    """
    try:
        crawler = get_web_crawler_service()

        crawl_request = WebCrawlRequest(
            urls=request.urls,
            max_pages=request.max_pages,
            max_depth=request.max_depth,
            crawler_type=request.crawler_type,
            include_patterns=request.include_patterns,
            exclude_patterns=request.exclude_patterns,
            save_markdown=request.save_markdown
        )

        job = await crawler.start_crawl(
            request=crawl_request,
            brand_id=request.brand_id,
            wait_for_finish=request.wait_for_finish
        )

        logger.info(f"User {user.email} started crawl job {job.job_id}")

        return {
            "job_id": job.job_id,
            "status": job.status,
            "run_id": job.run_id,
            "urls": request.urls,
            "max_pages": request.max_pages,
            "message": "크롤링이 시작되었습니다" if not request.wait_for_finish else "크롤링이 완료되었습니다"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Crawl start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs", response_model=dict)
async def list_crawl_jobs(
    brand_id: Optional[str] = Query(None, description="브랜드 ID 필터"),
    limit: int = Query(20, ge=1, le=100),
    user: User = Depends(get_current_user)
):
    """
    크롤링 작업 목록 조회
    """
    try:
        crawler = get_web_crawler_service()

        # 권한에 따라 브랜드 필터링
        if user.role != "super_admin" and brand_id:
            if not user.can_access_brand(brand_id):
                raise HTTPException(status_code=403, detail="No access to this brand")

        jobs = crawler.list_jobs(brand_id=brand_id, limit=limit)

        return {
            "jobs": [
                {
                    "job_id": job.job_id,
                    "brand_id": job.brand_id,
                    "status": job.status,
                    "urls": job.request.urls,
                    "max_pages": job.request.max_pages,
                    "created_at": job.created_at.isoformat(),
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                    "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    "total_pages": job.result.total_pages if job.result else None,
                    "error": job.error
                }
                for job in jobs
            ],
            "total": len(jobs)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List jobs error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}", response_model=dict)
async def get_crawl_job(
    job_id: str,
    user: User = Depends(get_current_user)
):
    """
    크롤링 작업 상태 조회
    """
    try:
        crawler = get_web_crawler_service()
        job = await crawler.get_job_status(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # 권한 체크
        if job.brand_id and user.role != "super_admin":
            if not user.can_access_brand(job.brand_id):
                raise HTTPException(status_code=403, detail="No access to this job")

        return {
            "job_id": job.job_id,
            "brand_id": job.brand_id,
            "status": job.status,
            "run_id": job.run_id,
            "urls": job.request.urls,
            "max_pages": job.request.max_pages,
            "max_depth": job.request.max_depth,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "error": job.error,
            "result_summary": {
                "total_pages": job.result.total_pages,
                "status": job.result.status,
                "stats": job.result.stats
            } if job.result else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get job error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/results", response_model=dict)
async def get_crawl_results(
    job_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user: User = Depends(get_current_user)
):
    """
    크롤링 결과 조회 (페이지네이션)
    """
    try:
        crawler = get_web_crawler_service()
        job = await crawler.get_job_status(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # 권한 체크
        if job.brand_id and user.role != "super_admin":
            if not user.can_access_brand(job.brand_id):
                raise HTTPException(status_code=403, detail="No access to this job")

        if not job.result:
            return {
                "job_id": job_id,
                "status": job.status,
                "message": "Results not yet available",
                "pages": [],
                "total": 0
            }

        # 페이지네이션
        all_pages = job.result.pages
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated = all_pages[start_idx:end_idx]

        return {
            "job_id": job_id,
            "status": job.status,
            "total": len(all_pages),
            "page": page,
            "page_size": page_size,
            "pages": [
                {
                    "url": p.url,
                    "title": p.title,
                    "text_length": len(p.text) if p.text else 0,
                    "markdown_length": len(p.markdown) if p.markdown else 0,
                    "text_preview": p.text[:500] if p.text else None,
                    "metadata": p.metadata,
                    "crawled_at": p.crawled_at.isoformat()
                }
                for p in paginated
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get results error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/pages/{page_index}", response_model=dict)
async def get_crawled_page(
    job_id: str,
    page_index: int,
    user: User = Depends(get_current_user)
):
    """
    특정 페이지 전체 내용 조회
    """
    try:
        crawler = get_web_crawler_service()
        job = await crawler.get_job_status(job_id)

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        if not job.result or page_index >= len(job.result.pages):
            raise HTTPException(status_code=404, detail="Page not found")

        page = job.result.pages[page_index]

        return {
            "url": page.url,
            "title": page.title,
            "text": page.text,
            "markdown": page.markdown,
            "html": page.html,
            "metadata": page.metadata,
            "crawled_at": page.crawled_at.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get page error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/cancel", response_model=dict)
async def cancel_crawl_job(
    job_id: str,
    user: User = Depends(require_permission(Permission.PIPELINE_RUN))
):
    """
    크롤링 작업 취소
    """
    try:
        crawler = get_web_crawler_service()
        success = await crawler.cancel_job(job_id)

        if not success:
            raise HTTPException(status_code=400, detail="Cannot cancel job")

        logger.info(f"User {user.email} cancelled crawl job {job_id}")

        return {
            "job_id": job_id,
            "status": "cancelled",
            "message": "크롤링이 취소되었습니다"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cancel job error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/save-to-neo4j", response_model=dict)
async def save_to_neo4j(
    job_id: str,
    brand_id: str = Query(..., description="저장할 브랜드 ID"),
    user: User = Depends(require_permission(Permission.PIPELINE_RUN))
):
    """
    크롤링 결과를 Neo4j에 저장

    크롤링된 웹페이지를 브랜드 지식 그래프에 추가합니다.
    """
    try:
        crawler = get_web_crawler_service()
        job = await crawler.get_job_status(job_id)

        if not job or not job.result:
            raise HTTPException(status_code=404, detail="Job or results not found")

        # Neo4j에 저장
        from app.services.shared.neo4j import get_neo4j_client

        neo4j = get_neo4j_client()
        saved_count = 0

        for page in job.result.pages:
            if not page.text and not page.markdown:
                continue

            # WebContent 노드 생성
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
                "content": content[:10000] if content else "",  # 길이 제한
                "brand_id": brand_id,
                "crawled_at": page.crawled_at.isoformat()
            })
            saved_count += 1

        logger.info(f"Saved {saved_count} pages to Neo4j for brand {brand_id}")

        return {
            "job_id": job_id,
            "brand_id": brand_id,
            "saved_pages": saved_count,
            "message": f"{saved_count}개 페이지가 저장되었습니다"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Save to Neo4j error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
