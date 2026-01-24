"""
Tavily Search Service
AI-optimized web search for knowledge augmentation

Use cases:
- Fallback when Neo4j knowledge graph lacks information
- Market research and trend analysis
- Competitor analysis
- Real-time news and updates
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Tavily API key from environment
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY', '')


@dataclass
class TavilySearchResult:
    """Tavily 검색 결과"""
    title: str
    url: str
    content: str
    score: float = 0.0
    published_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'url': self.url,
            'content': self.content,
            'score': self.score,
            'published_date': self.published_date,
        }


@dataclass
class TavilySearchResponse:
    """Tavily 검색 응답"""
    query: str
    results: List[TavilySearchResult] = field(default_factory=list)
    answer: Optional[str] = None  # Tavily가 제공하는 AI 요약 답변
    response_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'query': self.query,
            'results': [r.to_dict() for r in self.results],
            'answer': self.answer,
            'response_time_ms': self.response_time_ms,
        }

    def to_context_string(self, max_results: int = 5) -> str:
        """LLM 컨텍스트용 문자열 변환"""
        if not self.results:
            return ""

        parts = [f"[Web Search Results for: {self.query}]"]

        if self.answer:
            parts.append(f"\nAI Summary: {self.answer}\n")

        for i, result in enumerate(self.results[:max_results], 1):
            parts.append(f"\n{i}. {result.title}")
            parts.append(f"   Source: {result.url}")
            parts.append(f"   {result.content[:500]}...")

        return "\n".join(parts)


class TavilyClient:
    """
    Tavily Search Client

    AI-optimized search for RAG applications.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or TAVILY_API_KEY

        if not self.api_key:
            logger.warning("Tavily API key not set. Web search will be disabled.")
            self._enabled = False
        else:
            self._enabled = True
            logger.info("Tavily client initialized")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def search(
        self,
        query: str,
        search_depth: str = "basic",  # "basic" or "advanced"
        max_results: int = 5,
        include_answer: bool = True,
        include_domains: List[str] = None,
        exclude_domains: List[str] = None,
    ) -> TavilySearchResponse:
        """
        Tavily 웹 검색 실행

        Args:
            query: 검색 쿼리
            search_depth: 검색 깊이 ("basic": 빠름, "advanced": 상세)
            max_results: 최대 결과 수
            include_answer: AI 요약 답변 포함 여부
            include_domains: 포함할 도메인 목록
            exclude_domains: 제외할 도메인 목록

        Returns:
            TavilySearchResponse
        """
        if not self._enabled:
            logger.warning("Tavily search called but API key not set")
            return TavilySearchResponse(query=query)

        import time
        import requests

        start_time = time.time()

        try:
            payload = {
                "api_key": self.api_key,
                "query": query,
                "search_depth": search_depth,
                "max_results": max_results,
                "include_answer": include_answer,
            }

            if include_domains:
                payload["include_domains"] = include_domains
            if exclude_domains:
                payload["exclude_domains"] = exclude_domains

            response = requests.post(
                "https://api.tavily.com/search",
                json=payload,
                timeout=30,
            )

            response.raise_for_status()
            data = response.json()

            # Parse results
            results = []
            for item in data.get('results', []):
                results.append(TavilySearchResult(
                    title=item.get('title', ''),
                    url=item.get('url', ''),
                    content=item.get('content', ''),
                    score=item.get('score', 0.0),
                    published_date=item.get('published_date'),
                ))

            elapsed_ms = (time.time() - start_time) * 1000

            logger.info(f"Tavily search completed: {len(results)} results in {elapsed_ms:.0f}ms")

            return TavilySearchResponse(
                query=query,
                results=results,
                answer=data.get('answer'),
                response_time_ms=elapsed_ms,
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Tavily API request failed: {e}")
            return TavilySearchResponse(query=query)
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return TavilySearchResponse(query=query)

    def search_for_brand(
        self,
        brand_name: str,
        topic: str,
        search_type: str = "general",
    ) -> TavilySearchResponse:
        """
        브랜드 관련 검색

        Args:
            brand_name: 브랜드 이름
            topic: 검색 주제
            search_type: 검색 유형 (general, news, competitor, trend)

        Returns:
            TavilySearchResponse
        """
        # 검색 쿼리 구성
        query_templates = {
            "general": f"{brand_name} {topic}",
            "news": f"{brand_name} latest news {topic}",
            "competitor": f"{brand_name} competitors market analysis {topic}",
            "trend": f"{topic} industry trends market research 2024 2025",
            "review": f"{brand_name} reviews customer feedback {topic}",
        }

        query = query_templates.get(search_type, query_templates["general"])

        return self.search(
            query=query,
            search_depth="advanced" if search_type in ["competitor", "trend"] else "basic",
            max_results=5,
            include_answer=True,
        )


# Singleton instance
_tavily_client: Optional[TavilyClient] = None


def get_tavily_client() -> TavilyClient:
    """Tavily 클라이언트 싱글톤 반환"""
    global _tavily_client

    if _tavily_client is None:
        _tavily_client = TavilyClient()

    return _tavily_client


def search_web(
    query: str,
    max_results: int = 5,
    include_answer: bool = True,
) -> TavilySearchResponse:
    """
    간편 웹 검색 함수

    Args:
        query: 검색 쿼리
        max_results: 최대 결과 수
        include_answer: AI 요약 포함

    Returns:
        TavilySearchResponse
    """
    client = get_tavily_client()
    return client.search(
        query=query,
        max_results=max_results,
        include_answer=include_answer,
    )
