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


def _parse_metrics(metrics_str: Optional[str]) -> Dict[str, int]:
    """
    Parse metrics string like "likes:188,comments:0,shares:0,views:3696"
    Returns dict with parsed values
    """
    result = {'likes': 0, 'comments': 0, 'shares': 0, 'views': 0}

    if not metrics_str or not isinstance(metrics_str, str):
        return result

    try:
        for part in metrics_str.split(','):
            if ':' in part:
                key, value = part.split(':', 1)
                key = key.strip().lower()
                if key in result:
                    result[key] = int(value.strip())
    except Exception:
        pass

    return result


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


class BrandKeywordCache:
    """브랜드별 도메인 키워드 캐시"""

    _cache: Dict[str, Set[str]] = {}
    _cache_ttl: Dict[str, datetime] = {}
    _ttl_minutes: int = 30  # 캐시 유효 시간

    @classmethod
    def get_domain_keywords(cls, brand_id: str, neo4j_client=None) -> Set[str]:
        """
        브랜드별 도메인 키워드 조회 (Neo4j Concept 노드에서 추출)

        Args:
            brand_id: 브랜드 ID
            neo4j_client: Neo4j 클라이언트 (없으면 자동 생성)

        Returns:
            도메인 키워드 Set
        """
        # 캐시 확인
        if brand_id in cls._cache:
            cache_time = cls._cache_ttl.get(brand_id)
            if cache_time and datetime.now() - cache_time < timedelta(minutes=cls._ttl_minutes):
                return cls._cache[brand_id]

        # Neo4j에서 Concept 노드의 키워드 추출
        keywords = set()
        try:
            if neo4j_client is None:
                neo4j_client = get_neo4j_client()

            # Concept 노드의 id와 description에서 키워드 추출
            query = """
            MATCH (c:Concept)
            WHERE c.brand_id = $brand_id
            RETURN c.id as id, c.description as description, c.type as type
            LIMIT 100
            """
            results = neo4j_client.query(query, {'brand_id': brand_id})

            for r in results:
                # Concept ID를 키워드로 추가 (소문자로)
                concept_id = r.get('id', '')
                if concept_id:
                    # 공백으로 분리된 단어들도 추가
                    for word in concept_id.lower().split():
                        if len(word) >= 2:
                            keywords.add(word)
                    # 전체 ID도 추가
                    keywords.add(concept_id.lower())

                # description에서 주요 단어 추출
                desc = r.get('description', '')
                if desc:
                    # 한글, 영어 단어 추출
                    words = re.findall(r'[가-힣]+|[a-zA-Z]+', desc.lower())
                    for word in words:
                        if len(word) >= 2:
                            keywords.add(word)

            logger.info(f"Loaded {len(keywords)} domain keywords for brand '{brand_id}'")

        except Exception as e:
            logger.warning(f"Failed to load domain keywords for {brand_id}: {e}")

        # 캐시 저장
        cls._cache[brand_id] = keywords
        cls._cache_ttl[brand_id] = datetime.now()

        return keywords

    @classmethod
    def clear_cache(cls, brand_id: str = None):
        """캐시 클리어"""
        if brand_id:
            cls._cache.pop(brand_id, None)
            cls._cache_ttl.pop(brand_id, None)
        else:
            cls._cache.clear()
            cls._cache_ttl.clear()


class BrandKeywordMappingCache:
    """
    브랜드별 한국어-영어 키워드 매핑 캐시

    브랜드 config의 keyword_mappings를 읽어서 KeywordExtractor에 적용합니다.
    새 브랜드 추가 시 config에 keyword_mappings를 추가하면 자동으로 적용됩니다.
    """

    _brand_mappings: Dict[str, Dict[str, str]] = {}
    _loaded_brands: Set[str] = set()

    @classmethod
    def load_brand_mappings(cls, brand_id: str, brand_config: Dict) -> Dict[str, str]:
        """
        브랜드 config에서 키워드 매핑 로드

        Args:
            brand_id: 브랜드 ID
            brand_config: 브랜드 설정 딕셔너리

        Returns:
            키워드 매핑 딕셔너리
        """
        if brand_id in cls._loaded_brands:
            return cls._brand_mappings.get(brand_id, {})

        # 브랜드 config에서 keyword_mappings 로드
        brand_mappings = brand_config.get('keyword_mappings', {})

        # 브랜드 이름도 자동으로 매핑에 추가
        brand_name_kr = brand_config.get('brand', {}).get('name', '')
        brand_name_id = brand_config.get('brand', {}).get('id', '')

        if brand_name_kr and brand_name_id:
            # 브랜드 이름의 다양한 변형을 ID로 매핑
            brand_mappings[brand_name_kr.lower()] = brand_name_id.lower()

        if brand_mappings:
            cls._brand_mappings[brand_id] = brand_mappings
            logger.info(f"Loaded {len(brand_mappings)} keyword mappings for brand '{brand_id}'")

        cls._loaded_brands.add(brand_id)
        return brand_mappings

    @classmethod
    def get_combined_mappings(cls, brand_id: str) -> Dict[str, str]:
        """
        기본 매핑 + 브랜드별 매핑 결합

        Args:
            brand_id: 브랜드 ID

        Returns:
            결합된 키워드 매핑
        """
        # 기본 매핑 복사
        combined = dict(KeywordExtractor.KEYWORD_MAPPINGS)

        # 브랜드별 매핑 추가 (덮어쓰기)
        brand_mappings = cls._brand_mappings.get(brand_id, {})
        combined.update(brand_mappings)

        return combined

    @classmethod
    def clear_cache(cls, brand_id: str = None):
        """캐시 클리어"""
        if brand_id:
            cls._brand_mappings.pop(brand_id, None)
            cls._loaded_brands.discard(brand_id)
        else:
            cls._brand_mappings.clear()
            cls._loaded_brands.clear()


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

    # 기본 도메인 키워드 (fallback)
    DEFAULT_DOMAIN_KEYWORDS: Set[str] = {
        '피부', '스킨케어', '보습', '수분', '세럼', '크림', '토너', '클렌저',
        '데이터', '통합', '솔루션', 'api', '분석', '인사이트', '대시보드',
    }

    # 플랫폼 키워드 매핑
    PLATFORM_KEYWORDS: Dict[str, str] = {
        '인스타': 'instagram',
        '인스타그램': 'instagram',
        'instagram': 'instagram',
        'insta': 'instagram',
        '유튜브': 'youtube',
        'youtube': 'youtube',
        '틱톡': 'tiktok',
        'tiktok': 'tiktok',
        '트위터': 'twitter',
        'twitter': 'twitter',
        '페이스북': 'facebook',
        'facebook': 'facebook',
        '블로그': 'blog',
        'blog': 'blog',
    }

    # 한국어-영어 브랜드명/키워드 매핑
    KEYWORD_MAPPINGS: Dict[str, str] = {
        '온틱스': 'ontix',
        '퓨처비': 'futurebi',
        '퓨쳐비': 'futurebi',
        '스킨케어': 'skincare',
        '브랜드': 'brand',
        '콘텐츠': 'content',
        '내용': 'content',
        '게시물': 'post',
        '포스트': 'post',
        '피드': 'feed',
    }

    # 추가 불용어 (쿼리에서 제외할 단어)
    QUERY_STOPWORDS: Set[str] = {
        '알려줘', '알려', '보여줘', '보여', '뭐야', '뭐', '무엇', '어떤',
        '좀', '해줘', '줘', '주세요', '해', '실제', '진짜', '정말',
    }

    @classmethod
    def detect_platforms(cls, text: str) -> List[str]:
        """
        텍스트에서 플랫폼 언급 감지

        Args:
            text: 입력 텍스트

        Returns:
            감지된 플랫폼 리스트
        """
        if not text:
            return []

        text_lower = text.lower()
        platforms = []

        for keyword, platform in cls.PLATFORM_KEYWORDS.items():
            if keyword in text_lower and platform not in platforms:
                platforms.append(platform)

        return platforms

    @classmethod
    def extract(cls, text: str, max_keywords: int = 10, domain_keywords: Set[str] = None) -> List[str]:
        """
        텍스트에서 의미 있는 키워드 추출

        Args:
            text: 입력 텍스트
            max_keywords: 최대 키워드 수
            domain_keywords: 브랜드별 도메인 키워드 (없으면 기본값 사용)

        Returns:
            키워드 리스트 (중요도 순)
        """
        if not text:
            return []

        # 도메인 키워드 설정
        active_domain_keywords = domain_keywords or cls.DEFAULT_DOMAIN_KEYWORDS

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

            # 쿼리 불용어 제외
            if token in cls.QUERY_STOPWORDS:
                continue

            # 불용어 제외
            if token in cls.KOREAN_STOPWORDS or token in cls.ENGLISH_STOPWORDS:
                continue

            # 한국어-영어 매핑 적용
            mapped_token = cls.KEYWORD_MAPPINGS.get(token, token)

            # 중복 제외 (매핑된 토큰 기준)
            if mapped_token in seen:
                continue

            seen.add(mapped_token)

            # 도메인 키워드는 높은 우선순위
            if mapped_token in active_domain_keywords:
                keywords.insert(0, mapped_token)
            else:
                keywords.append(mapped_token)

        return keywords[:max_keywords]

    @classmethod
    def extract_with_scores(cls, text: str, max_keywords: int = 10, domain_keywords: Set[str] = None) -> List[tuple]:
        """키워드와 점수 함께 추출"""
        active_domain_keywords = domain_keywords or cls.DEFAULT_DOMAIN_KEYWORDS
        keywords = cls.extract(text, max_keywords * 2, domain_keywords)

        scored = []
        for i, kw in enumerate(keywords):
            # 도메인 키워드는 높은 점수
            if kw in active_domain_keywords:
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

        # 브랜드별 도메인 키워드 로드 (Neo4j Concept 노드에서 자동 추출)
        self.domain_keywords = BrandKeywordCache.get_domain_keywords(
            self.brand_id, self.neo4j
        )

        # 브랜드별 키워드 매핑 로드 (config에서)
        BrandKeywordMappingCache.load_brand_mappings(self.brand_id, brand_config)
        self.keyword_mappings = BrandKeywordMappingCache.get_combined_mappings(self.brand_id)

        logger.info(f"Loaded {len(self.domain_keywords)} domain keywords for brand '{self.brand_id}'")

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

            # 브랜드별 도메인 키워드를 사용하여 키워드 추출
            keywords = KeywordExtractor.extract(question, domain_keywords=self.domain_keywords)

            # 브랜드별 추가 키워드 매핑 적용
            if self.keyword_mappings:
                mapped_keywords = []
                for kw in keywords:
                    # 매핑이 있으면 적용
                    mapped = self.keyword_mappings.get(kw.lower(), kw)
                    if mapped not in mapped_keywords:
                        mapped_keywords.append(mapped)
                keywords = mapped_keywords

            # 플랫폼 감지
            detected_platforms = KeywordExtractor.detect_platforms(question)
            if detected_platforms:
                logger.info(f"Detected platforms: {detected_platforms}")
                self.graph_config.platforms = detected_platforms

            # 도메인 키워드 폴백: 추출된 키워드 중 도메인 키워드가 없으면 상위 도메인 키워드 추가
            domain_matched = [kw for kw in keywords if kw.lower() in self.domain_keywords]
            if not domain_matched and self.domain_keywords:
                # 상위 5개 도메인 키워드를 폴백으로 추가
                fallback_keywords = list(self.domain_keywords)[:5]
                keywords = keywords + fallback_keywords
                logger.info(f"Added domain keyword fallback: {fallback_keywords}")

            if not keywords and not detected_platforms:
                logger.warning("No keywords or platforms extracted from question")
                return RetrievalResult(source='graph_search', items=[])

            logger.info(f"Graph search keywords: {keywords}, platforms: {detected_platforms}")

            results = []

            # 1. Concept 검색 (플랫폼이 감지되지 않은 경우에만)
            if not detected_platforms:
                concepts = self._search_concepts(keywords)
                results.extend(concepts)

            # 2. Content 검색 (플랫폼 필터 적용)
            contents = self._search_contents(keywords, detected_platforms)
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

    def _search_contents(self, keywords: List[str], platforms: List[str] = None) -> List[SearchResult]:
        """Content 노드 검색 (플랫폼 기반 검색 지원)"""
        if self.graph_config.scope not in [SearchScope.CONTENTS, SearchScope.ALL]:
            return []

        # 시간 필터 조건
        time_filter = ""
        if self.graph_config.days_back:
            time_filter = "AND c.created_at >= datetime() - duration({days: $days_back})"

        # 플랫폼 필터 조건
        platform_filter = ""
        use_platforms = platforms or self.graph_config.platforms
        if use_platforms:
            platform_filter = "AND c.platform IN $platforms"

        # 키워드 필터 조건 (플랫폼이 지정된 경우 선택적)
        if keywords:
            keyword_filter = "AND any(kw IN $keywords WHERE toLower(coalesce(c.text, '')) CONTAINS kw)"
            score_calc = "size([kw IN $keywords WHERE toLower(coalesce(c.text, '')) CONTAINS kw])"
        elif use_platforms:
            # 플랫폼만 지정된 경우 - 키워드 필터 없이 플랫폼의 모든 콘텐츠 검색
            keyword_filter = ""
            score_calc = "1"  # 기본 점수
        else:
            return []

        query = f"""
        MATCH (c:Content)
        WHERE c.brand_id = $brand_id
          {keyword_filter}
          {time_filter}
          {platform_filter}
        WITH c,
             {score_calc} as match_score
        RETURN c.id as id,
               c.text as text,
               c.url as url,
               c.platform as platform,
               c.content_type as content_type,
               c.like_count as likes,
               c.comment_count as comments,
               c.view_count as views,
               c.metrics as metrics,
               c.created_at as created_at,
               match_score + (coalesce(c.like_count, 0) / 1000.0) as score
        ORDER BY score DESC, c.created_at DESC
        LIMIT $limit
        """

        params = {
            'brand_id': self.brand_id,
            'limit': self.graph_config.max_results,
        }
        if keywords:
            params['keywords'] = keywords
        if self.graph_config.days_back:
            params['days_back'] = self.graph_config.days_back
        if use_platforms:
            params['platforms'] = use_platforms

        results = self.neo4j.query(query, params)

        search_results = []
        for r in results:
            # metrics 문자열 파싱 (fallback)
            parsed_metrics = _parse_metrics(r.get('metrics'))
            likes = r.get('likes', 0) or parsed_metrics['likes']
            comments = r.get('comments', 0) or parsed_metrics['comments']
            views = r.get('views', 0) or parsed_metrics['views']
            shares = parsed_metrics['shares']

            search_results.append(SearchResult(
                node_type='Content',
                node_id=r['id'],
                content=r.get('text', '')[:500],  # 텍스트 잘라내기
                score=float(r.get('score', 0)),
                metadata={
                    'url': r.get('url'),
                    'platform': r.get('platform'),
                    'content_type': r.get('content_type'),
                    'likes': likes,
                    'comments': comments,
                    'views': views,
                    'shares': shares,
                    'created_at': str(r.get('created_at')) if r.get('created_at') else None,
                }
            ))

        return search_results

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
