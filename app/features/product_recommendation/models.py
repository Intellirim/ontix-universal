### **app/features/product_recommendation/models.py**
"""
Product Recommendation Feature Models
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class ProductFilter(BaseModel):
    """상품 필터 조건"""
    min_price: Optional[int] = None
    max_price: Optional[int] = None
    category: Optional[str] = None
    in_stock_only: bool = True
    brands: Optional[List[str]] = None


class ProductInfo(BaseModel):
    """상품 정보"""
    id: str
    name: str
    price: int
    stock: int
    category: Optional[str] = None
    features: List[str] = []
    image_url: Optional[str] = None
    
    
class ProductRecommendationRequest(BaseModel):
    """상품 추천 요청"""
    question: str
    filters: Optional[ProductFilter] = None
    limit: int = Field(default=10, ge=1, le=50)
    sort_by: str = Field(default="stock", pattern="^(stock|price|popularity)$")


class ProductRecommendationResponse(BaseModel):
    """상품 추천 응답"""
    products: List[ProductInfo]
    total_count: int
    recommendation_reason: str
    metadata: dict = {}
