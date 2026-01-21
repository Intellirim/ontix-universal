
"""
브랜드 모델
"""

from pydantic import Field, field_validator
from typing import List, Optional
from app.models.base import BaseModel
import re


class BrandCreate(BaseModel):
    """브랜드 생성 요청 모델"""

    id: str = Field(..., min_length=2, max_length=50, description="브랜드 ID (영문, 숫자, 하이픈만)")
    name: str = Field(..., min_length=1, max_length=100, description="브랜드 이름")
    description: Optional[str] = Field(None, max_length=500, description="브랜드 설명")
    industry: Optional[str] = Field(None, max_length=100, description="산업 분류")
    features: List[str] = Field(default_factory=list, description="활성화된 기능 목록")
    neo4j_brand_id: Optional[str] = Field(None, description="Neo4j 브랜드 ID (미입력시 id와 동일)")
    neo4j_namespaces: List[str] = Field(default_factory=list, description="Neo4j 네임스페이스")

    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str) -> str:
        if not re.match(r'^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$', v):
            raise ValueError('Brand ID must contain only lowercase letters, numbers, and hyphens')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "id": "new-brand",
                "name": "New Brand",
                "description": "새로운 브랜드 설명",
                "industry": "테크",
                "features": ["product_recommendation"],
                "neo4j_brand_id": "new-brand",
                "neo4j_namespaces": ["new-brand"]
            }
        }


class BrandUpdate(BaseModel):
    """브랜드 수정 요청 모델"""

    name: Optional[str] = Field(None, min_length=1, max_length=100, description="브랜드 이름")
    description: Optional[str] = Field(None, max_length=500, description="브랜드 설명")
    industry: Optional[str] = Field(None, max_length=100, description="산업 분류")
    features: Optional[List[str]] = Field(None, description="활성화된 기능 목록")
    neo4j_brand_id: Optional[str] = Field(None, description="Neo4j 브랜드 ID")
    neo4j_namespaces: Optional[List[str]] = Field(None, description="Neo4j 네임스페이스")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Updated Brand Name",
                "description": "수정된 설명",
                "features": ["product_recommendation", "analytics"]
            }
        }


class BrandInfo(BaseModel):
    """브랜드 정보 모델"""

    id: str = Field(..., description="브랜드 ID")
    name: str = Field(..., description="브랜드 이름")
    description: Optional[str] = Field(None, description="브랜드 설명")
    industry: Optional[str] = Field(None, description="산업 분류")
    features: List[str] = Field(default_factory=list, description="활성화된 기능 목록")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "raceon",
                "name": "RACEON",
                "description": "복싱 용품 브랜드",
                "industry": "스포츠",
                "features": ["product_recommendation", "analytics"]
            }
        }


class BrandStats(BaseModel):
    """브랜드 통계 모델"""

    brand_id: str
    total_nodes: int = 0
    nodes: dict = Field(default_factory=dict, description="노드 타입별 카운트")

    class Config:
        json_schema_extra = {
            "example": {
                "brand_id": "raceon",
                "total_nodes": 1250,
                "nodes": {
                    "Post": 850,
                    "Product": 120,
                    "Concept": 280
                }
            }
        }
