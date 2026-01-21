"""
Product Retriever - Production Grade v2.0
상품 검색 및 추천

Features:
    - 키워드/카테고리 기반 상품 검색
    - SNS 언급 기반 인기 상품 추천
    - Concept-Product 관계 기반 추천
    - 가격 필터링
    - 재고 상태 필터링
    - 다양한 정렬 옵션
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import logging

from app.interfaces.retriever import RetrieverInterface, RetrievalResult, RetrievalItem, RetrievalSource
from app.core.context import QueryContext
from app.services.shared.neo4j import get_neo4j_client

logger = logging.getLogger(__name__)


class ProductSearchMode(str, Enum):
    """상품 검색 모드"""
    KEYWORD = "keyword"  # 키워드 검색
    CATEGORY = "category"  # 카테고리 검색
    CONCEPT = "concept"  # 컨셉 기반 추천
    POPULAR = "popular"  # 인기 상품 (SNS 언급 기반)
    PRICE_RANGE = "price_range"  # 가격대 검색
    ALL = "all"  # 전체


class ProductSortBy(str, Enum):
    """정렬 기준"""
    RELEVANCE = "relevance"
    PRICE_LOW = "price_low"
    PRICE_HIGH = "price_high"
    POPULARITY = "popularity"
    NAME = "name"
    STOCK = "stock"


class StockStatus(str, Enum):
    """재고 상태"""
    IN_STOCK = "in_stock"
    LOW_STOCK = "low_stock"
    OUT_OF_STOCK = "out_of_stock"
    ALL = "all"


@dataclass
class ProductSearchConfig:
    """상품 검색 설정"""
    mode: ProductSearchMode = ProductSearchMode.KEYWORD
    sort_by: ProductSortBy = ProductSortBy.RELEVANCE
    stock_status: StockStatus = StockStatus.IN_STOCK
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    categories: Optional[List[str]] = None
    limit: int = 20
    include_mentions: bool = True  # SNS 언급 수 포함


@dataclass
class ProductResult:
    """상품 검색 결과"""
    product_id: str
    name: str
    price: float
    stock: int
    score: float = 0.0
    category: Optional[str] = None
    description: Optional[str] = None
    image_url: Optional[str] = None
    mention_count: int = 0
    avg_sentiment: float = 0.0
    related_concepts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PriceParser:
    """가격 관련 파싱"""

    @staticmethod
    def extract_price_range(query: str) -> Tuple[Optional[float], Optional[float]]:
        """
        쿼리에서 가격 범위 추출

        예: "5만원대 세럼" -> (50000, 60000)
            "10만원 이하" -> (None, 100000)
            "3만원 이상" -> (30000, None)
        """
        min_price, max_price = None, None

        # 만원 단위 패턴
        patterns = [
            # X만원대
            (r'(\d+)만원?\s*대', lambda m: (int(m.group(1)) * 10000, (int(m.group(1)) + 1) * 10000)),
            # X만원 이하
            (r'(\d+)만원?\s*이하', lambda m: (None, int(m.group(1)) * 10000)),
            # X만원 이상
            (r'(\d+)만원?\s*이상', lambda m: (int(m.group(1)) * 10000, None)),
            # X원 이하
            (r'(\d+)원?\s*이하', lambda m: (None, int(m.group(1)))),
            # X원 이상
            (r'(\d+)원?\s*이상', lambda m: (int(m.group(1)), None)),
            # X~Y만원
            (r'(\d+)\s*[~\-]\s*(\d+)만원?', lambda m: (int(m.group(1)) * 10000, int(m.group(2)) * 10000)),
        ]

        for pattern, extractor in patterns:
            match = re.search(pattern, query)
            if match:
                min_price, max_price = extractor(match)
                break

        return min_price, max_price


class QueryAnalyzer:
    """쿼리 분석"""

    # 카테고리 키워드
    CATEGORY_KEYWORDS = {
        '스킨케어': ['스킨', '스킨케어', '기초', '기초케어'],
        '세럼': ['세럼', '에센스', '앰플'],
        '크림': ['크림', '모이스처라이저', '수분크림', '보습크림'],
        '선케어': ['선크림', '자외선', 'spf', '선케어'],
        '클렌저': ['클렌저', '클렌징', '폼', '오일'],
        '토너': ['토너', '스킨', '화장수'],
        '마스크': ['마스크', '마스크팩', '시트마스크'],
        '메이크업': ['메이크업', '립', '아이', '파운데이션'],
    }

    # 의도 키워드
    INTENT_KEYWORDS = {
        'recommend': ['추천', '좋은', '인기', '베스트', 'best', 'popular'],
        'cheap': ['저렴', '싼', '가성비', 'cheap', 'budget'],
        'expensive': ['고급', '프리미엄', '럭셔리', 'premium', 'luxury'],
        'new': ['신상', '새로운', '신제품', 'new', 'latest'],
    }

    @classmethod
    def detect_categories(cls, query: str) -> List[str]:
        """쿼리에서 카테고리 감지"""
        query_lower = query.lower()
        detected = []

        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                detected.append(category)

        return detected

    @classmethod
    def detect_intent(cls, query: str) -> Dict[str, bool]:
        """쿼리에서 의도 감지"""
        query_lower = query.lower()
        intents = {}

        for intent, keywords in cls.INTENT_KEYWORDS.items():
            intents[intent] = any(kw in query_lower for kw in keywords)

        return intents

    @classmethod
    def extract_keywords(cls, query: str) -> List[str]:
        """쿼리에서 검색 키워드 추출"""
        # 불용어 제거
        stopwords = {'의', '을', '를', '이', '가', '은', '는', '좀', '추천', '해줘', '알려줘'}
        words = re.findall(r'[가-힣]+|[a-zA-Z]+', query.lower())
        keywords = [w for w in words if len(w) >= 2 and w not in stopwords]
        return keywords


class ProductRetriever(RetrieverInterface):
    """
    프로덕션급 상품 검색기

    Neo4j에서 상품을 검색하고 SNS 데이터와 연계하여
    인기도, 언급 수, 감성 분석 결과를 함께 제공합니다.
    """

    def __init__(self, brand_config):
        super().__init__(brand_config, RetrievalSource.PRODUCT)
        self.neo4j = get_neo4j_client()
        self.product_config = ProductSearchConfig()

    def _do_retrieve(self, context: QueryContext) -> RetrievalResult:
        """
        상품 검색 실행 (추상 메서드 구현)

        Args:
            context: 쿼리 컨텍스트

        Returns:
            검색 결과
        """
        try:
            question = context.question

            # 1. 쿼리 분석
            categories = QueryAnalyzer.detect_categories(question)
            intents = QueryAnalyzer.detect_intent(question)
            keywords = QueryAnalyzer.extract_keywords(question)
            price_range = PriceParser.extract_price_range(question)

            # 설정 업데이트
            if categories:
                self.product_config.categories = categories
            if price_range[0] is not None:
                self.product_config.min_price = price_range[0]
            if price_range[1] is not None:
                self.product_config.max_price = price_range[1]

            # 정렬 방식 결정
            if intents.get('cheap'):
                self.product_config.sort_by = ProductSortBy.PRICE_LOW
            elif intents.get('expensive'):
                self.product_config.sort_by = ProductSortBy.PRICE_HIGH
            elif intents.get('recommend'):
                self.product_config.sort_by = ProductSortBy.POPULARITY

            logger.info(
                f"Product search: keywords={keywords}, "
                f"categories={categories}, price={price_range}"
            )

            # 2. 상품 검색
            results = self._search_products(keywords)

            # 3. SNS 언급 정보 추가
            if self.product_config.include_mentions and results:
                results = self._add_mention_info(results)

            # 4. 정렬
            results = self._sort_results(results)

            # 5. 결과 제한
            results = results[:self.product_config.limit]

            # ProductResult를 RetrievalItem으로 변환
            items = [
                RetrievalItem(
                    id=r.product_id,
                    content=f"{r.name}: {r.description or ''} (${r.price})",
                    score=r.score,
                    source='product_search',
                    node_type='Product',
                    metadata={
                        'name': r.name,
                        'price': r.price,
                        'stock': r.stock,
                        'category': r.category,
                        'image_url': r.image_url,
                        'mention_count': r.mention_count,
                        'avg_sentiment': r.avg_sentiment,
                        'related_concepts': r.related_concepts,
                    }
                )
                for r in results
            ]

            logger.info(f"Product retrieval: {len(items)} products found")

            return RetrievalResult(
                source='product_search',
                items=items,
                metadata={
                    'keywords': keywords,
                    'categories': categories,
                    'price_range': price_range,
                    'sort_by': self.product_config.sort_by.value,
                    'total_found': len(results),
                }
            )

        except Exception as e:
            logger.error(f"Product retrieval error: {e}", exc_info=True)
            return RetrievalResult(source='product_search', items=[], error=str(e))

    def _search_products(self, keywords: List[str]) -> List[ProductResult]:
        """상품 검색"""
        # 필터 조건 구성
        stock_filter = self._get_stock_filter()
        price_filter = self._get_price_filter()
        category_filter = self._get_category_filter()

        # 키워드 검색 조건
        keyword_filter = ""
        if keywords:
            keyword_filter = """
            AND (
                any(kw IN $keywords WHERE toLower(p.name) CONTAINS kw)
                OR any(kw IN $keywords WHERE toLower(coalesce(p.description, '')) CONTAINS kw)
                OR any(kw IN $keywords WHERE toLower(coalesce(p.category, '')) CONTAINS kw)
            )
            """

        query = f"""
        MATCH (p:Product)
        WHERE p.brand_id = $brand_id
          {stock_filter}
          {price_filter}
          {category_filter}
          {keyword_filter}
        WITH p,
             CASE WHEN $keywords IS NOT NULL AND size($keywords) > 0
                  THEN size([kw IN $keywords WHERE toLower(p.name) CONTAINS kw]) * 2 +
                       size([kw IN $keywords WHERE toLower(coalesce(p.description, '')) CONTAINS kw])
                  ELSE 1
             END as relevance_score
        RETURN p.id as id,
               p.name as name,
               p.price as price,
               p.stock as stock,
               p.category as category,
               p.description as description,
               p.image_url as image_url,
               relevance_score as score
        ORDER BY relevance_score DESC
        LIMIT $limit
        """

        params = {
            'brand_id': self.brand_id,
            'keywords': [kw.lower() for kw in keywords] if keywords else [],
            'limit': self.product_config.limit * 2,  # 후처리용 여유분
        }

        if self.product_config.min_price is not None:
            params['min_price'] = self.product_config.min_price
        if self.product_config.max_price is not None:
            params['max_price'] = self.product_config.max_price
        if self.product_config.categories:
            params['categories'] = self.product_config.categories

        results = self.neo4j.query(query, params)

        return [
            ProductResult(
                product_id=r['id'],
                name=r.get('name', ''),
                price=float(r.get('price', 0) or 0),
                stock=int(r.get('stock', 0) or 0),
                score=float(r.get('score', 0) or 0),
                category=r.get('category'),
                description=r.get('description'),
                image_url=r.get('image_url'),
            )
            for r in results
        ]

    def _get_stock_filter(self) -> str:
        """재고 필터 조건"""
        if self.product_config.stock_status == StockStatus.IN_STOCK:
            return "AND p.stock > 0"
        elif self.product_config.stock_status == StockStatus.LOW_STOCK:
            return "AND p.stock > 0 AND p.stock < 10"
        elif self.product_config.stock_status == StockStatus.OUT_OF_STOCK:
            return "AND p.stock = 0"
        else:
            return ""

    def _get_price_filter(self) -> str:
        """가격 필터 조건"""
        filters = []
        if self.product_config.min_price is not None:
            filters.append("AND p.price >= $min_price")
        if self.product_config.max_price is not None:
            filters.append("AND p.price <= $max_price")
        return " ".join(filters)

    def _get_category_filter(self) -> str:
        """카테고리 필터 조건"""
        if self.product_config.categories:
            return "AND p.category IN $categories"
        return ""

    def _add_mention_info(self, products: List[ProductResult]) -> List[ProductResult]:
        """SNS 언급 정보 추가"""
        product_ids = [p.product_id for p in products]

        if not product_ids:
            return products

        # 상품-Concept 관계 및 언급 수 조회
        query = """
        UNWIND $product_ids as pid
        OPTIONAL MATCH (p:Product {id: pid})<-[:MENTIONS_PRODUCT]-(c:Content)
        OPTIONAL MATCH (p)-[:RELATED_TO]->(concept:Concept)
        WITH pid,
             count(DISTINCT c) as mention_count,
             collect(DISTINCT concept.id)[..5] as concepts
        RETURN pid as product_id,
               mention_count,
               concepts
        """

        results = self.neo4j.query(query, {'product_ids': product_ids})

        # 결과를 딕셔너리로 변환
        mention_map = {
            r['product_id']: {
                'mention_count': r.get('mention_count', 0),
                'concepts': r.get('concepts', []),
            }
            for r in results
        }

        # 상품 정보 업데이트
        for product in products:
            if product.product_id in mention_map:
                info = mention_map[product.product_id]
                product.mention_count = info['mention_count']
                product.related_concepts = info['concepts']

                # 언급이 많으면 점수 부스트
                if product.mention_count > 0:
                    import math
                    product.score += math.log10(product.mention_count + 1) * 0.5

        return products

    def _sort_results(self, results: List[ProductResult]) -> List[ProductResult]:
        """결과 정렬"""
        if self.product_config.sort_by == ProductSortBy.PRICE_LOW:
            results.sort(key=lambda x: x.price)
        elif self.product_config.sort_by == ProductSortBy.PRICE_HIGH:
            results.sort(key=lambda x: x.price, reverse=True)
        elif self.product_config.sort_by == ProductSortBy.POPULARITY:
            results.sort(key=lambda x: x.mention_count, reverse=True)
        elif self.product_config.sort_by == ProductSortBy.NAME:
            results.sort(key=lambda x: x.name)
        elif self.product_config.sort_by == ProductSortBy.STOCK:
            results.sort(key=lambda x: x.stock, reverse=True)
        else:  # RELEVANCE
            results.sort(key=lambda x: x.score, reverse=True)

        return results

    def _result_to_dict(self, result: ProductResult) -> Dict[str, Any]:
        """ProductResult를 딕셔너리로 변환"""
        return {
            'product_id': result.product_id,
            'name': result.name,
            'price': result.price,
            'price_formatted': self._format_price(result.price),
            'stock': result.stock,
            'stock_status': self._get_stock_status_label(result.stock),
            'category': result.category,
            'description': result.description,
            'image_url': result.image_url,
            'mention_count': result.mention_count,
            'related_concepts': result.related_concepts,
            'score': round(result.score, 4),
        }

    def _format_price(self, price: float) -> str:
        """가격 포맷팅"""
        if price >= 10000:
            return f"{int(price // 10000)}만 {int(price % 10000):,}원" if price % 10000 else f"{int(price // 10000)}만원"
        return f"{int(price):,}원"

    def _get_stock_status_label(self, stock: int) -> str:
        """재고 상태 레이블"""
        if stock == 0:
            return "품절"
        elif stock < 10:
            return "재고 부족"
        else:
            return "재고 있음"

    def get_popular_products(self, limit: int = 10) -> List[ProductResult]:
        """인기 상품 조회 (SNS 언급 기반)"""
        query = """
        MATCH (p:Product)<-[:MENTIONS_PRODUCT]-(c:Content)
        WHERE p.brand_id = $brand_id AND p.stock > 0
        WITH p, count(c) as mentions
        RETURN p.id as id,
               p.name as name,
               p.price as price,
               p.stock as stock,
               p.category as category,
               mentions
        ORDER BY mentions DESC
        LIMIT $limit
        """

        results = self.neo4j.query(query, {
            'brand_id': self.brand_id,
            'limit': limit,
        })

        return [
            ProductResult(
                product_id=r['id'],
                name=r.get('name', ''),
                price=float(r.get('price', 0) or 0),
                stock=int(r.get('stock', 0) or 0),
                category=r.get('category'),
                mention_count=r.get('mentions', 0),
            )
            for r in results
        ]

    def get_products_by_concept(self, concept_id: str, limit: int = 10) -> List[ProductResult]:
        """특정 컨셉과 관련된 상품 조회"""
        query = """
        MATCH (p:Product)-[:RELATED_TO]->(c:Concept {id: $concept_id})
        WHERE p.brand_id = $brand_id AND p.stock > 0
        RETURN p.id as id,
               p.name as name,
               p.price as price,
               p.stock as stock,
               p.category as category
        ORDER BY p.price ASC
        LIMIT $limit
        """

        results = self.neo4j.query(query, {
            'brand_id': self.brand_id,
            'concept_id': concept_id,
            'limit': limit,
        })

        return [
            ProductResult(
                product_id=r['id'],
                name=r.get('name', ''),
                price=float(r.get('price', 0) or 0),
                stock=int(r.get('stock', 0) or 0),
                category=r.get('category'),
                related_concepts=[concept_id],
            )
            for r in results
        ]

    def configure(
        self,
        mode: ProductSearchMode = None,
        sort_by: ProductSortBy = None,
        stock_status: StockStatus = None,
        min_price: float = None,
        max_price: float = None,
        categories: List[str] = None,
        limit: int = None,
        include_mentions: bool = None,
    ) -> 'ProductRetriever':
        """
        설정 변경 (체이닝 지원)

        Usage:
            retriever.configure(sort_by=ProductSortBy.PRICE_LOW, max_price=50000).retrieve(context)
        """
        if mode is not None:
            self.product_config.mode = mode
        if sort_by is not None:
            self.product_config.sort_by = sort_by
        if stock_status is not None:
            self.product_config.stock_status = stock_status
        if min_price is not None:
            self.product_config.min_price = min_price
        if max_price is not None:
            self.product_config.max_price = max_price
        if categories is not None:
            self.product_config.categories = categories
        if limit is not None:
            self.product_config.limit = limit
        if include_mentions is not None:
            self.product_config.include_mentions = include_mentions
        return self
