"""
Vector Retriever - Production Grade v2.0
벡터 유사도 기반 시맨틱 검색

Features:
    - 멀티 인덱스 검색 (Concept, Content)
    - 쿼리 확장 및 전처리
    - 메타데이터 필터링
    - 점수 정규화 및 재랭킹
    - 키워드 검색 폴백
    - 임베딩 캐싱
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import re
import logging
import hashlib

from app.interfaces.retriever import RetrieverInterface, RetrievalResult, RetrievalItem, RetrievalSource
from app.core.context import QueryContext
from app.services.shared.neo4j import get_neo4j_client

logger = logging.getLogger(__name__)


class SearchMode(str, Enum):
    """검색 모드"""
    SEMANTIC = "semantic"  # 순수 의미 검색
    HYBRID = "hybrid"  # 벡터 + 키워드
    FILTERED = "filtered"  # 메타데이터 필터 적용


class NodeIndex(str, Enum):
    """검색 대상 노드 인덱스"""
    CONCEPT = "concept"
    CONTENT = "content"
    ALL = "all"


@dataclass
class VectorSearchConfig:
    """벡터 검색 설정"""
    mode: SearchMode = SearchMode.SEMANTIC
    node_index: NodeIndex = NodeIndex.CONCEPT
    top_k: int = 20
    min_score: float = 0.65
    max_score: float = 1.0
    enable_query_expansion: bool = True
    enable_score_boost: bool = True
    platforms: Optional[List[str]] = None
    days_back: Optional[int] = None


@dataclass
class VectorResult:
    """벡터 검색 결과"""
    node_type: str
    node_id: str
    content: str
    score: float
    raw_score: float  # 정규화 전 점수
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueryPreprocessor:
    """쿼리 전처리 및 확장"""

    # 동의어 매핑 (도메인 특화)
    SYNONYMS: Dict[str, List[str]] = {
        # 스킨케어
        '보습': ['수분', '하이드레이팅', 'moisturizing', 'hydrating'],
        '미백': ['브라이트닝', '화이트닝', 'brightening', 'whitening'],
        '주름': ['안티에이징', '링클', 'wrinkle', 'anti-aging'],
        '여드름': ['트러블', '피지', 'acne', 'pimple', 'blemish'],
        '민감성': ['민감', 'sensitive', '자극'],
        # 제품 타입
        '세럼': ['에센스', 'serum', 'essence'],
        '크림': ['모이스처라이저', 'cream', 'moisturizer'],
        '선크림': ['자외선차단제', 'sunscreen', 'sunblock', 'spf'],
        # 성분
        '비타민c': ['아스코르빅애시드', 'vitamin c', 'ascorbic acid'],
        '레티놀': ['비타민a', 'retinol', 'retinoid', 'vitamin a'],
        '히알루론산': ['히알루로닉애시드', 'hyaluronic acid', 'ha'],
    }

    @classmethod
    def preprocess(cls, query: str) -> str:
        """
        쿼리 전처리

        - 소문자 변환
        - 특수문자 정리
        - 공백 정규화
        """
        if not query:
            return ""

        # 소문자 변환
        query = query.lower()

        # 특수문자 정리 (한글, 영어, 숫자, 공백 유지)
        query = re.sub(r'[^\w\s가-힣]', ' ', query)

        # 공백 정규화
        query = ' '.join(query.split())

        return query

    @classmethod
    def expand(cls, query: str) -> str:
        """
        쿼리 확장 (동의어 추가)

        Args:
            query: 원본 쿼리

        Returns:
            확장된 쿼리
        """
        query_lower = query.lower()
        expansions = []

        for term, synonyms in cls.SYNONYMS.items():
            if term in query_lower:
                # 동의어 중 일부만 추가 (너무 많으면 노이즈)
                expansions.extend(synonyms[:2])

        if expansions:
            return f"{query} {' '.join(expansions)}"
        return query

    @classmethod
    def extract_filters(cls, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        쿼리에서 필터 조건 추출

        예: "인스타그램 보습 세럼 추천" -> ("보습 세럼 추천", {"platform": "instagram"})

        Returns:
            (정제된 쿼리, 필터 딕셔너리)
        """
        filters = {}
        cleaned_query = query

        # 플랫폼 필터
        platform_patterns = {
            r'\b(인스타그램|인스타|instagram|ig)\b': 'instagram',
            r'\b(유튜브|youtube|yt)\b': 'youtube',
            r'\b(틱톡|tiktok)\b': 'tiktok',
            r'\b(트위터|twitter|x)\b': 'twitter',
        }

        for pattern, platform in platform_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                filters['platform'] = platform
                cleaned_query = re.sub(pattern, '', cleaned_query, flags=re.IGNORECASE)

        # 시간 필터
        time_patterns = {
            r'\b(오늘|today)\b': 1,
            r'\b(이번\s*주|this\s*week)\b': 7,
            r'\b(이번\s*달|this\s*month)\b': 30,
            r'\b(최근|recently)\b': 14,
        }

        for pattern, days in time_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                filters['days_back'] = days
                cleaned_query = re.sub(pattern, '', cleaned_query, flags=re.IGNORECASE)

        cleaned_query = ' '.join(cleaned_query.split())
        return cleaned_query, filters


class ScoreNormalizer:
    """점수 정규화 및 부스팅"""

    @staticmethod
    def normalize(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """점수를 0-1 범위로 정규화"""
        if max_val == min_val:
            return 0.5
        normalized = (score - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))

    @staticmethod
    def boost_by_engagement(base_score: float, metadata: Dict[str, Any]) -> float:
        """
        인게이지먼트 기반 점수 부스팅

        좋아요, 조회수 등에 따라 점수 조정
        """
        boost = 0.0

        likes = metadata.get('likes', 0) or 0
        views = metadata.get('views', 0) or 0
        comments = metadata.get('comments', 0) or 0

        # 로그 스케일 부스트 (너무 큰 값 방지)
        if likes > 0:
            import math
            boost += min(0.1, math.log10(likes + 1) / 50)
        if views > 0:
            import math
            boost += min(0.05, math.log10(views + 1) / 100)
        if comments > 0:
            import math
            boost += min(0.05, math.log10(comments + 1) / 30)

        return min(1.0, base_score + boost)

    @staticmethod
    def boost_by_recency(base_score: float, created_at: Optional[str]) -> float:
        """최신 콘텐츠 부스팅"""
        if not created_at:
            return base_score

        try:
            from datetime import datetime, timedelta

            # 다양한 날짜 형식 파싱 시도
            for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d', '%Y-%m-%dT%H:%M:%SZ']:
                try:
                    dt = datetime.strptime(str(created_at)[:19], fmt)
                    break
                except ValueError:
                    continue
            else:
                return base_score

            days_old = (datetime.now() - dt).days

            # 7일 이내: +0.1, 30일 이내: +0.05, 이후 감소
            if days_old <= 7:
                return min(1.0, base_score + 0.1)
            elif days_old <= 30:
                return min(1.0, base_score + 0.05)
            elif days_old > 180:
                return max(0.0, base_score - 0.05)

        except Exception:
            pass

        return base_score


class VectorRetriever(RetrieverInterface):
    """
    프로덕션급 벡터 검색기

    Neo4j 벡터 인덱스를 활용한 시맨틱 검색을 수행합니다.
    키워드 폴백, 점수 정규화, 필터링 등 프로덕션 기능을 제공합니다.
    """

    # 임베딩 캐시 (쿼리 -> 임베딩)
    _embedding_cache: Dict[str, List[float]] = {}
    _cache_max_size: int = 100

    def __init__(self, brand_config):
        super().__init__(brand_config, RetrievalSource.VECTOR)
        self.neo4j = get_neo4j_client()
        self.vector_config = VectorSearchConfig()

    def _do_retrieve(self, context: QueryContext) -> RetrievalResult:
        """
        벡터 검색 실행 (추상 메서드 구현)

        Args:
            context: 쿼리 컨텍스트

        Returns:
            검색 결과
        """
        try:
            question = context.question

            # 1. 쿼리 전처리
            processed_query, extracted_filters = QueryPreprocessor.extract_filters(question)
            processed_query = QueryPreprocessor.preprocess(processed_query)

            if not processed_query:
                logger.warning("Empty query after preprocessing")
                return RetrievalResult(source='vector_search', items=[])

            # 필터 적용
            if extracted_filters.get('platform'):
                self.vector_config.platforms = [extracted_filters['platform']]
            if extracted_filters.get('days_back'):
                self.vector_config.days_back = extracted_filters['days_back']

            # 2. 쿼리 확장
            if self.vector_config.enable_query_expansion:
                expanded_query = QueryPreprocessor.expand(processed_query)
            else:
                expanded_query = processed_query

            logger.info(f"Vector search: '{processed_query}' -> '{expanded_query}'")

            # 3. 벡터 검색 실행
            results = []

            if self.vector_config.node_index in [NodeIndex.CONCEPT, NodeIndex.ALL]:
                concept_results = self._search_concepts(expanded_query)
                results.extend(concept_results)

            if self.vector_config.node_index in [NodeIndex.CONTENT, NodeIndex.ALL]:
                content_results = self._search_contents(expanded_query)
                results.extend(content_results)

            # 4. 점수 정규화 및 정렬
            results = self._normalize_and_rank(results)

            # 5. 필터링
            results = self._apply_filters(results)

            # 6. 최종 결과 제한
            results = results[:self.vector_config.top_k]

            # VectorResult를 RetrievalItem으로 변환
            items = [
                RetrievalItem(
                    id=r.node_id,
                    content=r.content,
                    score=r.score,
                    source='vector_search',
                    node_type=r.node_type,
                    metadata={**r.metadata, 'raw_score': r.raw_score}
                )
                for r in results
            ]

            logger.info(f"Vector retrieval: {len(items)} results found")

            return RetrievalResult(
                source='vector_search',
                items=items,
                metadata={
                    'query': processed_query,
                    'expanded_query': expanded_query,
                    'mode': self.vector_config.mode.value,
                    'total_found': len(results),
                }
            )

        except Exception as e:
            logger.error(f"Vector retrieval error: {e}", exc_info=True)
            return RetrievalResult(source='vector_search', items=[], error=str(e))

    def _get_embedding(self, query: str) -> Optional[List[float]]:
        """
        쿼리 임베딩 조회 (캐싱 지원)
        """
        if not self.neo4j.vector_available:
            return None

        # 캐시 키 생성
        cache_key = hashlib.md5(query.encode()).hexdigest()

        if cache_key in self._embedding_cache:
            logger.debug(f"Embedding cache hit: {query[:30]}...")
            return self._embedding_cache[cache_key]

        try:
            embedding = self.neo4j.vector_index.embedding.embed_query(query)

            # 캐시에 저장 (크기 제한)
            if len(self._embedding_cache) >= self._cache_max_size:
                # 가장 오래된 항목 제거 (단순 구현)
                oldest_key = next(iter(self._embedding_cache))
                del self._embedding_cache[oldest_key]

            self._embedding_cache[cache_key] = embedding
            return embedding

        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return None

    def _search_concepts(self, query: str) -> List[VectorResult]:
        """Concept 노드 벡터 검색"""
        embedding = self._get_embedding(query)

        if embedding:
            return self._vector_search_concepts(embedding)
        else:
            return self._keyword_search_concepts(query)

    def _vector_search_concepts(self, embedding: List[float]) -> List[VectorResult]:
        """벡터 기반 Concept 검색"""
        try:
            cypher = """
            CALL db.index.vector.queryNodes(
                'ontix_global_concept_index',
                $top_k,
                $embedding
            ) YIELD node, score
            WHERE node.brand_id = $brand_id
            RETURN node.id as id,
                   node.description as description,
                   node.type as type,
                   score
            ORDER BY score DESC
            """

            results = self.neo4j.query(cypher, {
                'embedding': embedding,
                'brand_id': self.brand_id,
                'top_k': self.vector_config.top_k * 2,  # 필터링 후 감소 대비
            })

            return [
                VectorResult(
                    node_type='Concept',
                    node_id=r['id'],
                    content=r.get('description') or r['id'],
                    score=float(r.get('score', 0)),
                    raw_score=float(r.get('score', 0)),
                    metadata={'type': r.get('type')},
                )
                for r in results
                if float(r.get('score', 0)) >= self.vector_config.min_score
            ]

        except Exception as e:
            logger.error(f"Vector search concepts error: {e}")
            return []

    def _keyword_search_concepts(self, query: str) -> List[VectorResult]:
        """키워드 기반 Concept 검색 (폴백)"""
        # 쿼리에서 키워드 추출
        keywords = re.findall(r'[가-힣]+|[a-zA-Z]+', query.lower())
        keywords = [kw for kw in keywords if len(kw) >= 2]

        if not keywords:
            return []

        cypher = """
        MATCH (c:Concept)
        WHERE c.brand_id = $brand_id
          AND (
            any(kw IN $keywords WHERE toLower(c.id) CONTAINS kw)
            OR any(kw IN $keywords WHERE toLower(coalesce(c.description, '')) CONTAINS kw)
          )
        WITH c,
             size([kw IN $keywords WHERE toLower(c.id) CONTAINS kw]) * 2 +
             size([kw IN $keywords WHERE toLower(coalesce(c.description, '')) CONTAINS kw]) as match_score
        RETURN c.id as id,
               c.description as description,
               c.type as type,
               toFloat(match_score) / $keyword_count as score
        ORDER BY score DESC
        LIMIT $limit
        """

        results = self.neo4j.query(cypher, {
            'brand_id': self.brand_id,
            'keywords': keywords,
            'keyword_count': len(keywords),
            'limit': self.vector_config.top_k,
        })

        return [
            VectorResult(
                node_type='Concept',
                node_id=r['id'],
                content=r.get('description') or r['id'],
                score=min(0.8, float(r.get('score', 0))),  # 키워드 검색은 최대 0.8
                raw_score=float(r.get('score', 0)),
                metadata={'type': r.get('type'), 'search_type': 'keyword_fallback'},
            )
            for r in results
        ]

    def _search_contents(self, query: str) -> List[VectorResult]:
        """Content 노드 검색 (키워드 기반)"""
        # Content는 벡터 인덱스가 없을 수 있으므로 키워드 검색 사용
        keywords = re.findall(r'[가-힣]+|[a-zA-Z]+', query.lower())
        keywords = [kw for kw in keywords if len(kw) >= 2]

        if not keywords:
            return []

        # 필터 조건 구성
        time_filter = ""
        if self.vector_config.days_back:
            time_filter = "AND c.created_at >= datetime() - duration({days: $days_back})"

        platform_filter = ""
        if self.vector_config.platforms:
            platform_filter = "AND c.platform IN $platforms"

        cypher = f"""
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
               c.like_count as likes,
               c.view_count as views,
               c.comment_count as comments,
               c.created_at as created_at,
               toFloat(match_score) / $keyword_count as score
        ORDER BY score DESC, c.like_count DESC
        LIMIT $limit
        """

        params = {
            'brand_id': self.brand_id,
            'keywords': keywords,
            'keyword_count': len(keywords),
            'limit': self.vector_config.top_k,
        }
        if self.vector_config.days_back:
            params['days_back'] = self.vector_config.days_back
        if self.vector_config.platforms:
            params['platforms'] = self.vector_config.platforms

        try:
            results = self.neo4j.query(cypher, params)

            return [
                VectorResult(
                    node_type='Content',
                    node_id=r['id'],
                    content=r.get('text', '')[:500],
                    score=min(0.75, float(r.get('score', 0))),  # Content 키워드 검색은 최대 0.75
                    raw_score=float(r.get('score', 0)),
                    metadata={
                        'url': r.get('url'),
                        'platform': r.get('platform'),
                        'likes': r.get('likes', 0),
                        'views': r.get('views', 0),
                        'comments': r.get('comments', 0),
                        'created_at': str(r.get('created_at')) if r.get('created_at') else None,
                    },
                )
                for r in results
            ]

        except Exception as e:
            logger.error(f"Content search error: {e}")
            return []

    def _normalize_and_rank(self, results: List[VectorResult]) -> List[VectorResult]:
        """점수 정규화 및 재랭킹"""
        if not results:
            return []

        # 점수 부스팅 적용
        if self.vector_config.enable_score_boost:
            for result in results:
                # 인게이지먼트 부스트
                result.score = ScoreNormalizer.boost_by_engagement(
                    result.score, result.metadata
                )
                # 최신성 부스트
                result.score = ScoreNormalizer.boost_by_recency(
                    result.score, result.metadata.get('created_at')
                )

        # 점수 기반 정렬
        results.sort(key=lambda x: x.score, reverse=True)

        return results

    def _apply_filters(self, results: List[VectorResult]) -> List[VectorResult]:
        """필터 적용"""
        filtered = []

        for result in results:
            # 최소 점수 필터
            if result.score < self.vector_config.min_score:
                continue

            # 최대 점수 필터 (이상치 제거)
            if result.score > self.vector_config.max_score:
                continue

            filtered.append(result)

        return filtered

    def _result_to_dict(self, result: VectorResult) -> Dict[str, Any]:
        """VectorResult를 딕셔너리로 변환"""
        return {
            'node_type': result.node_type,
            'node_id': result.node_id,
            'content': result.content,
            'score': round(result.score, 4),
            'raw_score': round(result.raw_score, 4),
            'metadata': result.metadata,
        }

    def configure(
        self,
        mode: SearchMode = None,
        node_index: NodeIndex = None,
        top_k: int = None,
        min_score: float = None,
        enable_query_expansion: bool = None,
        enable_score_boost: bool = None,
        platforms: List[str] = None,
        days_back: int = None,
    ) -> 'VectorRetriever':
        """
        검색 설정 변경 (체이닝 지원)

        Usage:
            retriever.configure(top_k=30, min_score=0.7).retrieve(context)
        """
        if mode is not None:
            self.vector_config.mode = mode
        if node_index is not None:
            self.vector_config.node_index = node_index
        if top_k is not None:
            self.vector_config.top_k = top_k
        if min_score is not None:
            self.vector_config.min_score = min_score
        if enable_query_expansion is not None:
            self.vector_config.enable_query_expansion = enable_query_expansion
        if enable_score_boost is not None:
            self.vector_config.enable_score_boost = enable_score_boost
        if platforms is not None:
            self.vector_config.platforms = platforms
        if days_back is not None:
            self.vector_config.days_back = days_back
        return self

    @classmethod
    def clear_embedding_cache(cls):
        """임베딩 캐시 초기화"""
        cls._embedding_cache.clear()
        logger.info("Embedding cache cleared")
