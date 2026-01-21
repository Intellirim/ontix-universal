"""
Graph Retriever - Production Grade v2.0
Neo4j 그래프 기반 지식 검색

Features:
    - Content, Interaction, Concept, Actor 노드 검색
    - 다중 홉 관계 탐색 (1-hop, 2-hop)
    - 의미 기반 키워드 추출
    - 점수 기반 랭킹
    - 시간 범위 필터링
    - 플랫폼별 필터링
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import re
import logging

from app.interfaces.retriever import RetrieverInterface, RetrievalResult, RetrievalItem, RetrievalSource
from app.core.context import QueryContext
from app.services.shared.neo4j import get_neo4j_client

logger = logging.getLogger(__name__)


class SearchScope(str, Enum):
    """검색 범위"""
    CONCEPTS = "concepts"
    CONTENTS = "contents"
    INTERACTIONS = "interactions"
    ALL = "all"


@dataclass
class GraphSearchConfig:
    """그래프 검색 설정"""
    scope: SearchScope = SearchScope.ALL
    max_results: int = 20
    min_score: float = 0.0
    days_back: Optional[int] = None  # None이면 전체 기간
    platforms: Optional[List[str]] = None  # None이면 전체 플랫폼
    include_relationships: bool = True
    hop_depth: int = 2  # 관계 탐색 깊이


@dataclass
class SearchResult:
    """검색 결과"""
    node_type: str
    node_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_nodes: List[Dict[str, Any]] = field(default_factory=list)


class KeywordExtractor:
    """의미 기반 키워드 추출"""

    # 한국어 불용어
    KOREAN_STOPWORDS: Set[str] = {
        '은', '는', '이', '가', '을', '를', '의', '에', '와', '과', '도', '로', '으로',
        '에서', '까지', '부터', '만', '보다', '처럼', '같이', '라고', '고', '며',
        '그', '저', '이런', '저런', '그런', '어떤', '무슨', '뭐', '뭔가',
        '있다', '없다', '하다', '되다', '있는', '없는', '하는', '되는',
        '것', '수', '등', '때', '곳', '중', '더', '덜', '매우', '정말', '아주',
        '좀', '많이', '조금', '잘', '못', '안', '왜', '어떻게', '언제', '어디',
    }

    # 영어 불용어
    ENGLISH_STOPWORDS: Set[str] = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
        'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and',
        'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
        'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
        'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
        'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    }

    # 도메인 특화 키워드 (스킨케어/뷰티)
    DOMAIN_KEYWORDS: Set[str] = {
        '피부', '스킨케어', '보습', '수분', '세럼', '크림', '토너', '클렌저',
        '선크림', '자외선', 'spf', '여드름', '트러블', '모공', '주름', '탄력',
        '미백', '브라이트닝', '각질', '필링', '마스크팩', '앰플', '에센스',
        '성분', '히알루론산', '나이아신아마이드', '레티놀', '비타민c', 'aha', 'bha',
        '민감성', '지성', '건성', '복합성', '피부타입', '루틴', '순서',
    }

    @classmethod
    def extract(cls, text: str, max_keywords: int = 10) -> List[str]:
        """
        텍스트에서 의미 있는 키워드 추출

        Args:
            text: 입력 텍스트
            max_keywords: 최대 키워드 수

        Returns:
            키워드 리스트 (중요도 순)
        """
        if not text:
            return []

        # 소문자 변환
        text_lower = text.lower()

        # 토큰화 (한글, 영어, 숫자)
        tokens = re.findall(r'[가-힣]+|[a-zA-Z]+|\d+', text_lower)

        # 불용어 제거 및 필터링
        keywords = []
        seen = set()

        for token in tokens:
            # 너무 짧은 토큰 제외
            if len(token) < 2:
                continue

            # 불용어 제외
            if token in cls.KOREAN_STOPWORDS or token in cls.ENGLISH_STOPWORDS:
                continue

            # 중복 제외
            if token in seen:
                continue

            seen.add(token)

            # 도메인 키워드는 높은 우선순위
            if token in cls.DOMAIN_KEYWORDS:
                keywords.insert(0, token)
            else:
                keywords.append(token)

        return keywords[:max_keywords]

    @classmethod
    def extract_with_scores(cls, text: str, max_keywords: int = 10) -> List[tuple]:
        """키워드와 점수 함께 추출"""
        keywords = cls.extract(text, max_keywords * 2)

        scored = []
        for i, kw in enumerate(keywords):
            # 도메인 키워드는 높은 점수
            if kw in cls.DOMAIN_KEYWORDS:
                score = 1.0
            else:
                # 위치 기반 점수 (앞에 나올수록 높음)
                score = max(0.3, 1.0 - (i * 0.1))
            scored.append((kw, score))

        # 점수 순 정렬
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:max_keywords]


class GraphRetriever(RetrieverInterface):
    """
    프로덕션급 그래프 검색기

    Neo4j에서 Content, Interaction, Concept, Actor 노드를 검색하고
    관계 기반 탐색을 통해 관련 정보를 수집합니다.
    """

    def __init__(self, brand_config):
        super().__init__(brand_config, RetrievalSource.GRAPH)
        self.neo4j = get_neo4j_client()
        self.graph_config = GraphSearchConfig()

    def _do_retrieve(self, context: QueryContext) -> RetrievalResult:
        """
        그래프 검색 실행 (추상 메소드 구현)

        Args:
            context: 쿼리 컨텍스트

        Returns:
            검색 결과
        """
        try:
            question = context.question
            keywords = KeywordExtractor.extract(question)

            if not keywords:
                logger.warning("No keywords extracted from question")
                return RetrievalResult(source='graph_search', items=[])

            logger.info(f"Graph search keywords: {keywords}")

            results = []

            # 1. Concept 검색
            concepts = self._search_concepts(keywords)
            results.extend(concepts)

            # 2. Content 검색
            contents = self._search_contents(keywords)
            results.extend(contents)

            # 3. 관계 기반 확장 검색
            if self.graph_config.include_relationships and results:
                related = self._expand_with_relationships(results[:5])
                results.extend(related)

            # 중복 제거 및 정렬
            results = self._deduplicate_and_rank(results)

            # SearchResult를 RetrievalItem으로 변환
            items = [
                RetrievalItem(
                    id=r.node_id,
                    content=r.content,
                    score=r.score,
                    source='graph_search',
                    node_type=r.node_type,
                    metadata=r.metadata
                )
                for r in results[:self.graph_config.max_results]
            ]

            logger.info(f"Graph retrieval: {len(items)} results found")

            return RetrievalResult(
                source='graph_search',
                items=items,
                metadata={
                    'keywords': keywords,
                    'total_found': len(results),
                    'scope': self.graph_config.scope.value,
                }
            )

        except Exception as e:
            logger.error(f"Graph retrieval error: {e}", exc_info=True)
            return RetrievalResult(source='graph_search', items=[], error=str(e))

    def _search_concepts(self, keywords: List[str]) -> List[SearchResult]:
        """Concept 노드 검색"""
        if self.graph_config.scope not in [SearchScope.CONCEPTS, SearchScope.ALL]:
            return []

        query = """
        MATCH (c:Concept)
        WHERE c.brand_id = $brand_id
          AND (
            any(kw IN $keywords WHERE toLower(c.id) CONTAINS kw)
            OR any(kw IN $keywords WHERE toLower(coalesce(c.description, '')) CONTAINS kw)
          )
        WITH c,
             size([kw IN $keywords WHERE toLower(c.id) CONTAINS kw]) * 2 +
             size([kw IN $keywords WHERE toLower(coalesce(c.description, '')) CONTAINS kw]) as match_score
        OPTIONAL MATCH (c)<-[:MENTIONS_CONCEPT]-(content:Content)
        WITH c, match_score, count(content) as mention_count
        RETURN c.id as id,
               c.description as description,
               c.type as type,
               labels(c) as labels,
               match_score + (mention_count * 0.1) as score,
               mention_count
        ORDER BY score DESC
        LIMIT $limit
        """

        results = self.neo4j.query(query, {
            'brand_id': self.brand_id,
            'keywords': keywords,
            'limit': self.graph_config.max_results,
        })

        return [
            SearchResult(
                node_type='Concept',
                node_id=r['id'],
                content=r.get('description') or r['id'],
                score=float(r.get('score', 0)),
                metadata={
                    'type': r.get('type'),
                    'mention_count': r.get('mention_count', 0),
                }
            )
            for r in results
        ]

    def _search_contents(self, keywords: List[str]) -> List[SearchResult]:
        """Content 노드 검색"""
        if self.graph_config.scope not in [SearchScope.CONTENTS, SearchScope.ALL]:
            return []

        # 시간 필터 조건
        time_filter = ""
        if self.graph_config.days_back:
            time_filter = "AND c.created_at >= datetime() - duration({days: $days_back})"

        # 플랫폼 필터 조건
        platform_filter = ""
        if self.graph_config.platforms:
            platform_filter = "AND c.platform IN $platforms"

        query = f"""
        MATCH (c:Content)
        WHERE c.brand_id = $brand_id
          AND any(kw IN $keywords WHERE toLower(coalesce(c.text, '')) CONTAINS kw)
          {time_filter}
          {platform_filter}
        WITH c,
             size([kw IN $keywords WHERE toLower(coalesce(c.text, '')) CONTAINS kw]) as match_score
        RETURN c.id as id,
               c.text as text,
               c.url as url,
               c.platform as platform,
               c.content_type as content_type,
               c.like_count as likes,
               c.comment_count as comments,
               c.view_count as views,
               c.created_at as created_at,
               match_score + (coalesce(c.like_count, 0) / 1000.0) as score
        ORDER BY score DESC
        LIMIT $limit
        """

        params = {
            'brand_id': self.brand_id,
            'keywords': keywords,
            'limit': self.graph_config.max_results,
        }
        if self.graph_config.days_back:
            params['days_back'] = self.graph_config.days_back
        if self.graph_config.platforms:
            params['platforms'] = self.graph_config.platforms

        results = self.neo4j.query(query, params)

        return [
            SearchResult(
                node_type='Content',
                node_id=r['id'],
                content=r.get('text', '')[:500],  # 텍스트 잘라내기
                score=float(r.get('score', 0)),
                metadata={
                    'url': r.get('url'),
                    'platform': r.get('platform'),
                    'content_type': r.get('content_type'),
                    'likes': r.get('likes', 0),
                    'comments': r.get('comments', 0),
                    'views': r.get('views', 0),
                    'created_at': str(r.get('created_at')) if r.get('created_at') else None,
                }
            )
            for r in results
        ]

    def _expand_with_relationships(self, seed_results: List[SearchResult]) -> List[SearchResult]:
        """관계 기반 확장 검색"""
        expanded = []

        # Concept에서 관련 Content 찾기
        concept_ids = [r.node_id for r in seed_results if r.node_type == 'Concept']
        if concept_ids:
            query = """
            MATCH (c:Concept)<-[:MENTIONS_CONCEPT]-(content:Content)
            WHERE c.id IN $concept_ids AND c.brand_id = $brand_id
            RETURN content.id as id,
                   content.text as text,
                   content.platform as platform,
                   content.like_count as likes,
                   c.id as related_concept,
                   0.5 as score
            LIMIT $limit
            """
            results = self.neo4j.query(query, {
                'concept_ids': concept_ids,
                'brand_id': self.brand_id,
                'limit': 10,
            })

            for r in results:
                expanded.append(SearchResult(
                    node_type='Content',
                    node_id=r['id'],
                    content=r.get('text', '')[:500],
                    score=0.5,
                    metadata={
                        'platform': r.get('platform'),
                        'likes': r.get('likes', 0),
                        'related_concept': r.get('related_concept'),
                        'source': 'relationship_expansion',
                    }
                ))

        # Content에서 관련 Interaction(댓글) 찾기
        content_ids = [r.node_id for r in seed_results if r.node_type == 'Content']
        if content_ids:
            query = """
            MATCH (c:Content)-[:HAS_INTERACTION]->(i:Interaction)
            WHERE c.id IN $content_ids AND c.brand_id = $brand_id
            RETURN i.id as id,
                   i.text as text,
                   i.like_count as likes,
                   c.id as parent_content,
                   0.4 as score
            LIMIT $limit
            """
            results = self.neo4j.query(query, {
                'content_ids': content_ids,
                'brand_id': self.brand_id,
                'limit': 10,
            })

            for r in results:
                expanded.append(SearchResult(
                    node_type='Interaction',
                    node_id=r['id'],
                    content=r.get('text', '')[:300],
                    score=0.4,
                    metadata={
                        'likes': r.get('likes', 0),
                        'parent_content': r.get('parent_content'),
                        'source': 'relationship_expansion',
                    }
                ))

        return expanded

    def _deduplicate_and_rank(self, results: List[SearchResult]) -> List[SearchResult]:
        """중복 제거 및 점수 기반 정렬"""
        seen = set()
        unique = []

        for r in results:
            key = (r.node_type, r.node_id)
            if key not in seen:
                seen.add(key)
                unique.append(r)

        # 점수 기반 정렬
        unique.sort(key=lambda x: x.score, reverse=True)
        return unique

    def _result_to_dict(self, result: SearchResult) -> Dict[str, Any]:
        """SearchResult를 딕셔너리로 변환"""
        return {
            'node_type': result.node_type,
            'node_id': result.node_id,
            'content': result.content,
            'score': result.score,
            'metadata': result.metadata,
        }

    def configure(
        self,
        scope: SearchScope = None,
        max_results: int = None,
        days_back: int = None,
        platforms: List[str] = None,
        include_relationships: bool = None,
    ) -> 'GraphRetriever':
        """검색 설정 변경 (체이닝 지원)"""
        if scope is not None:
            self.graph_config.scope = scope
        if max_results is not None:
            self.graph_config.max_results = max_results
        if days_back is not None:
            self.graph_config.days_back = days_back
        if platforms is not None:
            self.graph_config.platforms = platforms
        if include_relationships is not None:
            self.graph_config.include_relationships = include_relationships
        return self
