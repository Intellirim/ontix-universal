"""
Products API
상품 및 추천 데이터 엔드포인트
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/products")


# ============================================================
# Response Models
# ============================================================

class Product(BaseModel):
    """상품 정보"""
    id: str
    name: str
    category: Optional[str] = None
    price: Optional[float] = None
    stock: Optional[int] = None
    sales: Optional[int] = None
    rating: Optional[float] = None
    image_url: Optional[str] = None


class ProductRecommendation(BaseModel):
    """상품 추천"""
    product_id: str
    name: str
    score: float
    reason: str
    category: Optional[str] = None


class ProductStats(BaseModel):
    """상품 통계"""
    total_products: int
    active_products: int
    total_sales: int
    avg_price: float
    top_category: Optional[str] = None


# ============================================================
# API Endpoints
# ============================================================

@router.get("/{brand_id}", response_model=Dict[str, Any])
async def get_products(
    brand_id: str,
    category: Optional[str] = Query(None, description="카테고리 필터"),
    limit: int = Query(50, ge=1, le=200, description="최대 개수"),
    offset: int = Query(0, ge=0, description="시작 위치")
):
    """
    브랜드 상품 목록 조회

    Args:
        brand_id: 브랜드 ID
        category: 카테고리 필터
        limit: 최대 개수
        offset: 시작 위치

    Returns:
        상품 목록 및 통계
    """
    try:
        from app.services.shared.neo4j import get_neo4j_client

        neo4j = get_neo4j_client()

        # 상품 조회
        where_clause = "p.brand_id = $brand_id"
        params = {'brand_id': brand_id, 'limit': limit, 'offset': offset}

        if category:
            where_clause += " AND p.category = $category"
            params['category'] = category

        query = f"""
        MATCH (p:Product)
        WHERE {where_clause}
        RETURN
            p.id as id,
            p.name as name,
            p.category as category,
            p.price as price,
            p.stock as stock,
            p.sales as sales,
            p.rating as rating,
            p.image_url as image_url
        ORDER BY coalesce(p.sales, 0) DESC
        SKIP $offset
        LIMIT $limit
        """

        products = neo4j.query(query, params) or []

        # 통계 조회
        stats_query = """
        MATCH (p:Product)
        WHERE p.brand_id = $brand_id
        RETURN
            count(p) as total,
            sum(CASE WHEN coalesce(p.stock, 0) > 0 THEN 1 ELSE 0 END) as active,
            sum(coalesce(p.sales, 0)) as total_sales,
            avg(coalesce(p.price, 0)) as avg_price
        """

        stats_result = neo4j.query(stats_query, {'brand_id': brand_id})

        stats = ProductStats(
            total_products=stats_result[0].get('total', 0) if stats_result else 0,
            active_products=stats_result[0].get('active', 0) if stats_result else 0,
            total_sales=stats_result[0].get('total_sales', 0) if stats_result else 0,
            avg_price=round(stats_result[0].get('avg_price', 0) or 0, 0) if stats_result else 0
        )

        # 카테고리별 집계
        category_query = """
        MATCH (p:Product)
        WHERE p.brand_id = $brand_id AND p.category IS NOT NULL
        RETURN p.category as category, count(*) as count
        ORDER BY count DESC
        """

        categories = neo4j.query(category_query, {'brand_id': brand_id}) or []

        return {
            'brand_id': brand_id,
            'products': [
                Product(
                    id=str(p.get('id', '')),
                    name=p.get('name', ''),
                    category=p.get('category'),
                    price=p.get('price'),
                    stock=p.get('stock'),
                    sales=p.get('sales'),
                    rating=p.get('rating'),
                    image_url=p.get('image_url')
                ).model_dump()
                for p in products
            ],
            'stats': stats.model_dump(),
            'categories': [
                {'category': c.get('category'), 'count': c.get('count')}
                for c in categories
            ],
            'pagination': {
                'limit': limit,
                'offset': offset,
                'total': stats.total_products
            }
        }

    except Exception as e:
        logger.error(f"Products error for {brand_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{brand_id}/recommendations")
async def get_product_recommendations(
    brand_id: str,
    segment: Optional[str] = Query(None, description="고객 세그먼트 (new, returning, vip, dormant)"),
    limit: int = Query(10, ge=1, le=50, description="최대 개수")
):
    """
    상품 추천 조회

    Neo4j 데이터 기반 추천 또는 AI 기반 추천

    Args:
        brand_id: 브랜드 ID
        segment: 고객 세그먼트
        limit: 최대 개수

    Returns:
        추천 상품 목록
    """
    try:
        from app.services.shared.neo4j import get_neo4j_client

        neo4j = get_neo4j_client()

        # 세그먼트별 추천 로직
        if segment == "new":
            # 신규 고객: 인기 상품 + 입문용 가격대
            query = """
            MATCH (p:Product)
            WHERE p.brand_id = $brand_id
              AND coalesce(p.stock, 0) > 0
            WITH p, coalesce(p.sales, 0) as sales, coalesce(p.price, 0) as price
            ORDER BY sales DESC, price ASC
            LIMIT $limit
            RETURN p.id as id, p.name as name, p.category as category,
                   sales, price,
                   (sales * 0.7 + (1.0 / (price + 1)) * 0.3) as score
            """
            reason_prefix = "신규 고객에게 인기 있는"

        elif segment == "vip":
            # VIP: 프리미엄 상품
            query = """
            MATCH (p:Product)
            WHERE p.brand_id = $brand_id
              AND coalesce(p.stock, 0) > 0
            WITH p, coalesce(p.price, 0) as price, coalesce(p.rating, 0) as rating
            ORDER BY price DESC, rating DESC
            LIMIT $limit
            RETURN p.id as id, p.name as name, p.category as category,
                   coalesce(p.sales, 0) as sales, price,
                   (price * 0.5 + rating * 50) as score
            """
            reason_prefix = "VIP 고객에게 추천하는 프리미엄"

        elif segment == "dormant":
            # 휴면 고객: 할인/이벤트 상품
            query = """
            MATCH (p:Product)
            WHERE p.brand_id = $brand_id
              AND coalesce(p.stock, 0) > 0
            WITH p, coalesce(p.sales, 0) as sales, coalesce(p.price, 0) as price
            ORDER BY price ASC, sales DESC
            LIMIT $limit
            RETURN p.id as id, p.name as name, p.category as category,
                   sales, price,
                   (1.0 / (price + 1) * 100 + sales * 0.3) as score
            """
            reason_prefix = "재방문 유도를 위한 가성비"

        else:
            # 기본/재구매: 베스트셀러
            query = """
            MATCH (p:Product)
            WHERE p.brand_id = $brand_id
              AND coalesce(p.stock, 0) > 0
            WITH p, coalesce(p.sales, 0) as sales, coalesce(p.rating, 0) as rating
            ORDER BY sales DESC, rating DESC
            LIMIT $limit
            RETURN p.id as id, p.name as name, p.category as category,
                   sales, coalesce(p.price, 0) as price,
                   (sales * 0.6 + rating * 40) as score
            """
            reason_prefix = "베스트셀러"

        results = neo4j.query(query, {'brand_id': brand_id, 'limit': limit}) or []

        # 점수 정규화
        max_score = max((r.get('score', 0) or 1) for r in results) if results else 1

        recommendations = [
            ProductRecommendation(
                product_id=str(r.get('id', '')),
                name=r.get('name', ''),
                score=round((r.get('score', 0) or 0) / max_score, 2),
                reason=f"{reason_prefix} 상품 - {r.get('category', '기타')} 카테고리",
                category=r.get('category')
            ).model_dump()
            for r in results
        ]

        return {
            'brand_id': brand_id,
            'segment': segment or 'default',
            'recommendations': recommendations,
            'total': len(recommendations)
        }

    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{brand_id}/categories")
async def get_product_categories(brand_id: str):
    """
    브랜드 상품 카테고리 목록

    Args:
        brand_id: 브랜드 ID

    Returns:
        카테고리 목록 및 통계
    """
    try:
        from app.services.shared.neo4j import get_neo4j_client

        neo4j = get_neo4j_client()

        query = """
        MATCH (p:Product)
        WHERE p.brand_id = $brand_id AND p.category IS NOT NULL
        WITH p.category as category,
             count(*) as product_count,
             sum(coalesce(p.sales, 0)) as total_sales,
             avg(coalesce(p.price, 0)) as avg_price
        RETURN category, product_count, total_sales, avg_price
        ORDER BY total_sales DESC
        """

        results = neo4j.query(query, {'brand_id': brand_id}) or []

        return {
            'brand_id': brand_id,
            'categories': [
                {
                    'category': r.get('category'),
                    'product_count': r.get('product_count', 0),
                    'total_sales': r.get('total_sales', 0),
                    'avg_price': round(r.get('avg_price', 0) or 0, 0)
                }
                for r in results
            ],
            'total_categories': len(results)
        }

    except Exception as e:
        logger.error(f"Categories error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{brand_id}/top-sellers")
async def get_top_sellers(
    brand_id: str,
    days: int = Query(30, ge=1, le=90, description="기간"),
    limit: int = Query(10, ge=1, le=50, description="최대 개수")
):
    """
    베스트셀러 조회

    Args:
        brand_id: 브랜드 ID
        days: 기간
        limit: 최대 개수

    Returns:
        베스트셀러 목록
    """
    try:
        from app.services.shared.neo4j import get_neo4j_client

        neo4j = get_neo4j_client()

        query = """
        MATCH (p:Product)
        WHERE p.brand_id = $brand_id
        WITH p, coalesce(p.sales, 0) as sales
        ORDER BY sales DESC
        LIMIT $limit
        RETURN
            p.id as id,
            p.name as name,
            p.category as category,
            p.price as price,
            sales,
            p.rating as rating
        """

        results = neo4j.query(query, {'brand_id': brand_id, 'limit': limit}) or []

        return {
            'brand_id': brand_id,
            'period_days': days,
            'top_sellers': [
                {
                    'rank': i + 1,
                    'id': str(r.get('id', '')),
                    'name': r.get('name', ''),
                    'category': r.get('category'),
                    'price': r.get('price'),
                    'sales': r.get('sales', 0),
                    'rating': r.get('rating')
                }
                for i, r in enumerate(results)
            ]
        }

    except Exception as e:
        logger.error(f"Top sellers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
