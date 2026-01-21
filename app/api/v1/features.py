### **app/api/v1/features.py**
"""
Features API
기능 관리 엔드포인트
"""

from fastapi import APIRouter
from typing import List
from app.features.registry import FeatureRegistry

router = APIRouter(prefix="/features")


@router.get("", response_model=List[str])
async def list_features():
    """
    등록된 기능 목록
    
    Returns:
        기능 이름 리스트
    """
    return FeatureRegistry.list_features()


@router.get("/{feature_name}")
async def get_feature_info(feature_name: str):
    """
    기능 정보 조회
    
    Args:
        feature_name: 기능 이름
    
    Returns:
        기능 정보
    """
    exists = FeatureRegistry.has_feature(feature_name)
    
    return {
        'feature_name': feature_name,
        'exists': exists
    }
