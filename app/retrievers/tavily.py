"""
Tavily Retriever - Web Search for Knowledge Augmentation
웹 검색을 통한 지식 보강

Use cases:
- Fallback when Neo4j knowledge graph lacks information
- Market research and trend analysis
- Competitor analysis
- Real-time news and updates
"""

from typing import Dict, List, Any, Optional
import logging

from app.interfaces.retriever import (
    RetrieverInterface,
    RetrievalResult,
    RetrievalItem,
    RetrievalSource,
    RetrievalStatus,
)
from app.core.context import QueryContext
from app.services.shared.tavily import get_tavily_client, TavilySearchResponse

logger = logging.getLogger(__name__)


class TavilyRetriever(RetrieverInterface):
    """
    Tavily Web Search Retriever

    AI-optimized web search for RAG applications.
    Used as a fallback when internal knowledge is insufficient.
    """

    def __init__(self, brand_config: Dict[str, Any]):
        super().__init__(brand_config, RetrievalSource.CUSTOM)
        self.source_name = "tavily_web"

        self.tavily = get_tavily_client()
        self.brand_name = brand_config.get('brand', {}).get('name', '')
        self.brand_id = brand_config.get('brand', {}).get('id', '')

        # Tavily-specific config
        tavily_config = brand_config.get('retrieval', {}).get('tavily', {})
        self.max_results = tavily_config.get('max_results', 5)
        self.search_depth = tavily_config.get('search_depth', 'basic')
        self.include_answer = tavily_config.get('include_answer', True)

        # Question types that should trigger web search
        self.web_search_question_types = [
            'market_research',
            'competitor_analysis',
            'trend_analysis',
            'news',
            'external_info',
        ]

        logger.info(f"TavilyRetriever initialized (enabled: {self.tavily.is_enabled})")

    def _do_retrieve(self, context: QueryContext) -> RetrievalResult:
        """
        Execute Tavily web search

        Args:
            context: Query context

        Returns:
            RetrievalResult with web search results
        """
        if not self.tavily.is_enabled:
            logger.warning("Tavily not enabled, skipping web search")
            return RetrievalResult(
                source=self.source_name,
                status=RetrievalStatus.NO_RESULTS,
            )

        question = context.question
        question_type = context.question_type
        if hasattr(question_type, 'value'):
            question_type_str = question_type.value
        else:
            question_type_str = str(question_type)

        # Determine search type based on question
        search_type = self._determine_search_type(question, question_type_str)

        # Build search query
        query = self._build_search_query(question, search_type)

        logger.info(f"Tavily search: type={search_type}, query={query[:100]}...")

        try:
            # Execute search
            response = self.tavily.search(
                query=query,
                search_depth=self.search_depth,
                max_results=self.max_results,
                include_answer=self.include_answer,
            )

            # Convert to RetrievalItems
            items = self._convert_to_items(response)

            # Build metadata
            metadata = {
                'search_type': search_type,
                'query': query,
                'tavily_answer': response.answer,
                'result_count': len(response.results),
            }

            return RetrievalResult(
                source=self.source_name,
                items=items,
                status=RetrievalStatus.COMPLETED,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Tavily retrieval error: {e}")
            return RetrievalResult(
                source=self.source_name,
                status=RetrievalStatus.FAILED,
                error=str(e),
            )

    def _determine_search_type(self, question: str, question_type: str) -> str:
        """Determine search type based on question content"""
        question_lower = question.lower()

        # Check for specific patterns
        if any(kw in question_lower for kw in ['경쟁사', '경쟁', 'competitor', '시장 점유']):
            return 'competitor'
        if any(kw in question_lower for kw in ['트렌드', 'trend', '최신', '동향', '전망']):
            return 'trend'
        if any(kw in question_lower for kw in ['뉴스', 'news', '소식', '기사']):
            return 'news'
        if any(kw in question_lower for kw in ['리뷰', 'review', '평가', '후기']):
            return 'review'
        if any(kw in question_lower for kw in ['시장', 'market', '산업', 'industry']):
            return 'trend'

        return 'general'

    def _build_search_query(self, question: str, search_type: str) -> str:
        """Build optimized search query"""
        # Include brand name for relevance
        brand_prefix = f"{self.brand_name} " if self.brand_name else ""

        if search_type == 'competitor':
            return f"{brand_prefix}competitors market analysis"
        elif search_type == 'trend':
            return f"{question} industry trends 2024 2025"
        elif search_type == 'news':
            return f"{brand_prefix}latest news {question}"
        elif search_type == 'review':
            return f"{brand_prefix}reviews customer feedback"
        else:
            return f"{brand_prefix}{question}"

    def _convert_to_items(self, response: TavilySearchResponse) -> List[RetrievalItem]:
        """Convert Tavily response to RetrievalItems"""
        items = []

        # Add AI answer as first item if available
        if response.answer:
            items.append(RetrievalItem(
                id="tavily_answer",
                content=response.answer,
                score=1.0,  # Highest relevance
                source=self.source_name,
                node_type="ai_summary",
                metadata={'type': 'tavily_answer'},
            ))

        # Add search results
        for i, result in enumerate(response.results):
            items.append(RetrievalItem(
                id=f"tavily_{i}",
                content=f"[{result.title}]\n{result.content}",
                score=result.score,
                source=self.source_name,
                node_type="web_content",
                metadata={
                    'url': result.url,
                    'title': result.title,
                    'published_date': result.published_date,
                },
            ))

        return items

    def should_search(self, context: QueryContext) -> bool:
        """
        Determine if web search should be performed

        Triggers when:
        - Question type suggests web search (market research, competitor, etc.)
        - Question contains web search keywords
        - Internal data is sparse (fallback mode)
        """
        if not self.tavily.is_enabled:
            return False

        question_type = context.question_type
        if hasattr(question_type, 'value'):
            question_type_str = question_type.value
        else:
            question_type_str = str(question_type)

        # Check if question type requires web search
        if question_type_str in self.web_search_question_types:
            logger.info(f"Web search triggered by question type: {question_type_str}")
            return True

        # Check for web search keywords
        question_lower = context.question.lower()
        web_keywords = [
            # 시장/경쟁 관련
            '경쟁사', '경쟁', 'competitor',
            '트렌드', 'trend', '동향',
            '시장', 'market', '산업', 'industry',
            '뉴스', 'news', '최근', '최신',
            '외부', 'external',
            # 정보 요청 패턴
            '알아', '뭐야', '뭐지', '뭐에요', '뭔가요', '무엇',
            '검색', '찾아', 'search',
            '정보', '설명해', '알려줘',
            # 외부 서비스/플랫폼 질문
            '사이트', '서비스', '플랫폼', '앱', '카페', '커뮤니티',
        ]

        matched_keywords = [kw for kw in web_keywords if kw in question_lower]
        if matched_keywords:
            logger.info(f"Web search triggered by keywords: {matched_keywords}")
            return True

        # Check if existing results are sparse (fallback mode)
        total_results = context.get_total_retrieval_count()
        if total_results < 3:
            logger.info(f"Web search triggered by sparse results ({total_results})")
            return True

        return False
