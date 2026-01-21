### **app/api/v1/brands.py**
"""
Brands API
브랜드 관리 엔드포인트 (CRUD)
"""

from fastapi import APIRouter, HTTPException, status, Query
from typing import List, Optional
from app.models.brand import BrandInfo, BrandStats, BrandCreate, BrandUpdate
from app.services.platform.brand_manager import get_brand_manager
from app.services.platform.config_manager import ConfigManager
from app.data_pipeline.repositories.neo4j_repo import Neo4jRepository
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/brands")


@router.get("", response_model=List[BrandInfo])
async def list_brands():
    """
    브랜드 목록 조회

    Returns:
        브랜드 목록
    """
    try:
        brands = get_brand_manager().list_brands()
        return brands
    except Exception as e:
        logger.error(f"List brands error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=BrandInfo, status_code=status.HTTP_201_CREATED)
async def create_brand(brand: BrandCreate):
    """
    새 브랜드 생성

    Args:
        brand: 브랜드 생성 데이터

    Returns:
        생성된 브랜드 정보
    """
    try:
        # 브랜드 설정 파일 생성
        ConfigManager.create_brand(
            brand_id=brand.id,
            brand_data={
                "name": brand.name,
                "description": brand.description,
                "industry": brand.industry,
                "features": brand.features,
                "neo4j_brand_id": brand.neo4j_brand_id,
                "neo4j_namespaces": brand.neo4j_namespaces,
            }
        )

        # 브랜드 매니저 캐시 클리어
        get_brand_manager().clear_cache(brand.id)

        # 생성된 브랜드 정보 반환
        return get_brand_manager().get_brand(brand.id)

    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.error(f"Create brand error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{brand_id}", response_model=BrandInfo)
async def get_brand(brand_id: str):
    """
    브랜드 정보 조회

    Args:
        brand_id: 브랜드 ID

    Returns:
        브랜드 정보
    """
    try:
        brand = get_brand_manager().get_brand(brand_id)
        return brand
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Get brand error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{brand_id}", response_model=BrandInfo)
async def update_brand(brand_id: str, updates: BrandUpdate):
    """
    브랜드 정보 수정

    Args:
        brand_id: 브랜드 ID
        updates: 수정할 필드

    Returns:
        수정된 브랜드 정보
    """
    try:
        # 업데이트할 필드만 추출
        update_data = updates.model_dump(exclude_unset=True)

        if not update_data:
            raise HTTPException(status_code=400, detail="No fields to update")

        # 설정 파일 업데이트
        ConfigManager.update_brand(brand_id, update_data)

        # 브랜드 매니저 캐시 클리어
        get_brand_manager().clear_cache(brand_id)

        # 업데이트된 브랜드 정보 반환
        return get_brand_manager().get_brand(brand_id)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Update brand error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{brand_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_brand(brand_id: str):
    """
    브랜드 삭제

    Args:
        brand_id: 브랜드 ID
    """
    try:
        # 설정 파일 삭제
        ConfigManager.delete_brand(brand_id)

        # 브랜드 매니저 캐시 클리어
        get_brand_manager().clear_cache(brand_id)

        return None

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Delete brand error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{brand_id}/stats", response_model=BrandStats)
async def get_brand_stats(brand_id: str):
    """
    브랜드 통계 조회

    Args:
        brand_id: 브랜드 ID

    Returns:
        브랜드 통계
    """
    try:
        stats = get_brand_manager().get_brand_stats(brand_id)
        return stats
    except Exception as e:
        logger.error(f"Get brand stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{brand_id}/validate")
async def validate_brand(brand_id: str):
    """
    브랜드 설정 검증

    Args:
        brand_id: 브랜드 ID

    Returns:
        검증 결과
    """
    try:
        result = get_brand_manager().validate_brand(brand_id)
        return result
    except Exception as e:
        logger.error(f"Validate brand error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{brand_id}/graph-summary")
async def get_brand_graph_summary(brand_id: str):
    """
    브랜드 그래프 요약 조회

    Args:
        brand_id: 브랜드 ID

    Returns:
        그래프 요약 (노드/관계 수, 타입별 통계, 주요 컨셉 등)
    """
    try:
        repo = Neo4jRepository()
        result = repo.get_brand_graph_summary(brand_id)
        return result
    except Exception as e:
        logger.error(f"Get brand graph summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{brand_id}/graph")
async def get_brand_graph(
    brand_id: str,
    limit: int = Query(default=100, ge=1, le=1000, description="최대 노드 수"),
    node_types: Optional[str] = Query(default=None, description="노드 타입 필터 (콤마로 구분)"),
):
    """
    브랜드 지식그래프 시각화 데이터 조회

    Neo4j에서 실제 노드와 관계를 가져와 프론트엔드 시각화용 데이터를 반환합니다.

    Args:
        brand_id: 브랜드 ID
        limit: 최대 노드 수 (기본값 100, 최대 1000)
        node_types: 필터링할 노드 타입 (콤마로 구분, 예: "Content,Concept,Topic")

    Returns:
        nodes: 노드 리스트 (id, label, type, properties)
        relationships: 관계 리스트 (source, target, type)
        stats: 통계 정보
    """
    try:
        # Neo4j 연결
        repo = Neo4jRepository()

        # 노드 타입 파싱
        types_list = None
        if node_types:
            types_list = [t.strip() for t in node_types.split(",") if t.strip()]

        # 그래프 데이터 조회
        result = repo.get_graph_visualization_data(
            brand_id=brand_id,
            limit=limit,
            node_types=types_list,
        )

        return result

    except Exception as e:
        logger.error(f"Get brand graph error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
