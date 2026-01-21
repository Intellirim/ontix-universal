"""Crawler Services"""

from app.services.crawlers.web_crawler import (
    WebCrawlerService,
    WebCrawlRequest,
    WebCrawlResult,
    WebCrawlJob,
    CrawledPage,
    CrawlerType,
    get_web_crawler_service
)

__all__ = [
    "WebCrawlerService",
    "WebCrawlRequest",
    "WebCrawlResult",
    "WebCrawlJob",
    "CrawledPage",
    "CrawlerType",
    "get_web_crawler_service"
]
