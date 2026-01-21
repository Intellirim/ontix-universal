"""
Website Content Crawler Service
Apify Website Content Crawler를 사용한 웹사이트 콘텐츠 크롤링
"""

import os
import logging
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from apify_client import ApifyClient

logger = logging.getLogger(__name__)


# ============================================
# Configuration
# ============================================

APIFY_TOKEN = os.getenv("APIFY_TOKEN", "")
WEB_CRAWLER_ACTOR_ID = "aYG0l9s7dbB7j3gbS"  # apify/website-content-crawler


class CrawlerType(str, Enum):
    """크롤러 유형"""
    PLAYWRIGHT_ADAPTIVE = "playwright:adaptive"
    PLAYWRIGHT_FIREFOX = "playwright:firefox"
    CHEERIO = "cheerio"
    JSDOM = "jsdom"


class HtmlTransformer(str, Enum):
    """HTML 변환 방식"""
    READABLE_TEXT = "readableText"
    EXTRACT_TEXT = "extractText"
    NONE = "none"


# ============================================
# Request/Response Models
# ============================================

class WebCrawlRequest(BaseModel):
    """웹 크롤링 요청"""
    urls: List[str] = Field(..., description="크롤링할 URL 목록")
    max_pages: int = Field(default=100, description="최대 크롤링 페이지 수", ge=1, le=10000)
    max_depth: int = Field(default=5, description="최대 크롤링 깊이", ge=1, le=20)
    crawler_type: CrawlerType = Field(default=CrawlerType.PLAYWRIGHT_ADAPTIVE)
    include_patterns: List[str] = Field(default_factory=list, description="포함할 URL 패턴 (glob)")
    exclude_patterns: List[str] = Field(default_factory=list, description="제외할 URL 패턴 (glob)")
    save_markdown: bool = Field(default=True, description="마크다운으로 저장")
    save_html: bool = Field(default=False, description="HTML로 저장")
    remove_cookie_warnings: bool = Field(default=True, description="쿠키 경고 제거")
    block_media: bool = Field(default=True, description="미디어 차단 (빠른 크롤링)")
    wait_for_selector: Optional[str] = Field(default=None, description="대기할 CSS 선택자")
    timeout_secs: int = Field(default=60, description="요청 타임아웃 (초)")


class CrawledPage(BaseModel):
    """크롤링된 페이지"""
    url: str
    title: Optional[str] = None
    text: Optional[str] = None
    markdown: Optional[str] = None
    html: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    crawled_at: datetime = Field(default_factory=datetime.utcnow)


class WebCrawlResult(BaseModel):
    """웹 크롤링 결과"""
    run_id: str
    status: str
    total_pages: int
    pages: List[CrawledPage]
    started_at: datetime
    finished_at: Optional[datetime] = None
    stats: Dict[str, Any] = Field(default_factory=dict)


class WebCrawlJob(BaseModel):
    """웹 크롤링 작업"""
    job_id: str
    brand_id: Optional[str] = None
    request: WebCrawlRequest
    status: str = "pending"  # pending, running, completed, failed
    run_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[WebCrawlResult] = None
    error: Optional[str] = None


# ============================================
# Web Crawler Service
# ============================================

class WebCrawlerService:
    """Apify Website Content Crawler 서비스"""

    def __init__(self, token: Optional[str] = None):
        self.token = token or APIFY_TOKEN
        if not self.token:
            logger.warning("APIFY_TOKEN not set - web crawling will not work")
            self.client = None
        else:
            self.client = ApifyClient(self.token)

        # 진행 중인 작업 추적
        self._jobs: Dict[str, WebCrawlJob] = {}

    def _build_input(self, request: WebCrawlRequest) -> Dict[str, Any]:
        """Apify Actor 입력 생성"""
        return {
            "startUrls": [{"url": url} for url in request.urls],
            "useSitemaps": False,
            "useLlmsTxt": False,
            "respectRobotsTxtFile": True,
            "crawlerType": request.crawler_type.value,
            "includeUrlGlobs": request.include_patterns,
            "excludeUrlGlobs": request.exclude_patterns,
            "keepUrlFragments": False,
            "ignoreCanonicalUrl": False,
            "ignoreHttpsErrors": False,
            "maxCrawlDepth": request.max_depth,
            "maxCrawlPages": request.max_pages,
            "initialConcurrency": 0,
            "maxConcurrency": 200,
            "initialCookies": [],
            "customHttpHeaders": {},
            "signHttpRequests": False,
            "pageFunction": "",
            "proxyConfiguration": {
                "useApifyProxy": True
            },
            "maxSessionRotations": 10,
            "maxRequestRetries": 3,
            "requestTimeoutSecs": request.timeout_secs,
            "minFileDownloadSpeedKBps": 128,
            "dynamicContentWaitSecs": 10,
            "waitForSelector": request.wait_for_selector or "",
            "softWaitForSelector": "",
            "maxScrollHeightPixels": 5000,
            "keepElementsCssSelector": "",
            "removeElementsCssSelector": """nav, footer, script, style, noscript, svg, img[src^='data:'],
                [role="alert"],
                [role="banner"],
                [role="dialog"],
                [role="alertdialog"],
                [role="region"][aria-label*="skip" i],
                [aria-modal="true"]""",
            "removeCookieWarnings": request.remove_cookie_warnings,
            "blockMedia": request.block_media,
            "expandIframes": True,
            "clickElementsCssSelector": '[aria-expanded="false"]',
            "htmlTransformer": HtmlTransformer.READABLE_TEXT.value,
            "readableTextCharThreshold": 100,
            "aggressivePrune": False,
            "debugMode": False,
            "storeSkippedUrls": False,
            "debugLog": False,
            "saveHtml": request.save_html,
            "saveHtmlAsFile": False,
            "saveMarkdown": request.save_markdown,
            "saveFiles": False,
            "saveScreenshots": False,
            "maxResults": request.max_pages,
            "clientSideMinChangePercentage": 15,
            "renderingTypeDetectionPercentage": 10
        }

    async def start_crawl(
        self,
        request: WebCrawlRequest,
        brand_id: Optional[str] = None,
        wait_for_finish: bool = False
    ) -> WebCrawlJob:
        """
        웹 크롤링 시작

        Args:
            request: 크롤링 요청
            brand_id: 브랜드 ID (선택)
            wait_for_finish: 완료까지 대기 여부

        Returns:
            크롤링 작업
        """
        if not self.client:
            raise ValueError("APIFY_TOKEN not configured")

        import uuid
        job_id = str(uuid.uuid4())[:8]

        job = WebCrawlJob(
            job_id=job_id,
            brand_id=brand_id,
            request=request,
            status="pending"
        )
        self._jobs[job_id] = job

        try:
            logger.info(f"Starting web crawl job {job_id} for URLs: {request.urls}")
            job.status = "running"
            job.started_at = datetime.utcnow()

            # Actor 입력 생성
            actor_input = self._build_input(request)

            if wait_for_finish:
                # 동기 실행 (완료까지 대기)
                run = self.client.actor(WEB_CRAWLER_ACTOR_ID).call(run_input=actor_input)
                job.run_id = run.get("id")

                # 결과 가져오기
                result = await self._fetch_results(run)
                job.result = result
                job.status = "completed"
                job.completed_at = datetime.utcnow()
            else:
                # 비동기 실행 (바로 반환)
                run = self.client.actor(WEB_CRAWLER_ACTOR_ID).start(run_input=actor_input)
                job.run_id = run.get("id")

            logger.info(f"Web crawl job {job_id} started with run_id: {job.run_id}")
            return job

        except Exception as e:
            logger.error(f"Web crawl job {job_id} failed: {e}")
            job.status = "failed"
            job.error = str(e)
            job.completed_at = datetime.utcnow()
            raise

    async def _fetch_results(self, run: Dict[str, Any]) -> WebCrawlResult:
        """Actor 실행 결과 가져오기"""
        dataset_id = run.get("defaultDatasetId")
        if not dataset_id:
            raise ValueError("No dataset ID in run result")

        # 데이터셋에서 결과 가져오기
        items = self.client.dataset(dataset_id).list_items().items

        pages = []
        for item in items:
            page = CrawledPage(
                url=item.get("url", ""),
                title=item.get("metadata", {}).get("title"),
                text=item.get("text"),
                markdown=item.get("markdown"),
                html=item.get("html"),
                metadata=item.get("metadata", {}),
                crawled_at=datetime.utcnow()
            )
            pages.append(page)

        return WebCrawlResult(
            run_id=run.get("id", ""),
            status=run.get("status", "UNKNOWN"),
            total_pages=len(pages),
            pages=pages,
            started_at=datetime.fromisoformat(run.get("startedAt", datetime.utcnow().isoformat()).replace("Z", "+00:00")),
            finished_at=datetime.fromisoformat(run.get("finishedAt", datetime.utcnow().isoformat()).replace("Z", "+00:00")) if run.get("finishedAt") else None,
            stats=run.get("stats", {})
        )

    async def get_job_status(self, job_id: str) -> Optional[WebCrawlJob]:
        """작업 상태 조회"""
        job = self._jobs.get(job_id)
        if not job or not job.run_id:
            return job

        # Apify에서 최신 상태 가져오기
        if job.status == "running" and self.client:
            try:
                run = self.client.run(job.run_id).get()
                apify_status = run.get("status", "UNKNOWN")

                if apify_status == "SUCCEEDED":
                    job.status = "completed"
                    job.completed_at = datetime.utcnow()
                    job.result = await self._fetch_results(run)
                elif apify_status in ["FAILED", "ABORTED", "TIMED-OUT"]:
                    job.status = "failed"
                    job.error = f"Apify run {apify_status}"
                    job.completed_at = datetime.utcnow()
            except Exception as e:
                logger.error(f"Error checking job status: {e}")

        return job

    async def get_job_results(self, job_id: str) -> Optional[WebCrawlResult]:
        """작업 결과 가져오기"""
        job = await self.get_job_status(job_id)
        if not job:
            return None
        return job.result

    def list_jobs(self, brand_id: Optional[str] = None, limit: int = 20) -> List[WebCrawlJob]:
        """작업 목록 조회"""
        jobs = list(self._jobs.values())

        if brand_id:
            jobs = [j for j in jobs if j.brand_id == brand_id]

        # 최신순 정렬
        jobs.sort(key=lambda x: x.created_at, reverse=True)

        return jobs[:limit]

    async def cancel_job(self, job_id: str) -> bool:
        """작업 취소"""
        job = self._jobs.get(job_id)
        if not job or not job.run_id:
            return False

        if job.status != "running":
            return False

        try:
            if self.client:
                self.client.run(job.run_id).abort()
            job.status = "cancelled"
            job.completed_at = datetime.utcnow()
            return True
        except Exception as e:
            logger.error(f"Error cancelling job: {e}")
            return False


# ============================================
# Singleton Instance
# ============================================

_crawler_service: Optional[WebCrawlerService] = None


def get_web_crawler_service() -> WebCrawlerService:
    """웹 크롤러 서비스 인스턴스 가져오기"""
    global _crawler_service
    if _crawler_service is None:
        _crawler_service = WebCrawlerService()
    return _crawler_service
