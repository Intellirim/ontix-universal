
"""
기능 모델
"""

from pydantic import Field
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from app.models.base import BaseModel


class FeatureConfig(BaseModel):
    """기능 설정"""
    
    enabled: bool = Field(True, description="기능 활성화 여부")
    config: Dict[str, Any] = Field(default_factory=dict, description="기능별 설정")


class FeatureHandler(ABC):
    """
    기능 핸들러 추상 클래스
    모든 기능 핸들러는 이 클래스를 상속받아야 함
    """
    
    def __init__(self, brand_config: Dict[str, Any]):
        """
        Args:
            brand_config: 브랜드 전체 설정
        """
        self.brand_config = brand_config
        self.brand_id = brand_config['brand']['id']
        self.feature_config = self._extract_feature_config()
    
    @abstractmethod
    def _extract_feature_config(self) -> Dict[str, Any]:
        """
        브랜드 설정에서 이 기능에 필요한 설정 추출
        
        Returns:
            기능별 설정
        """
        pass
    
    @abstractmethod
    def can_handle(self, question: str, context: Dict[str, Any]) -> bool:
        """
        이 핸들러가 질문을 처리할 수 있는지 판단
        
        Args:
            question: 사용자 질문
            context: 컨텍스트 정보
        
        Returns:
            처리 가능 여부
        """
        pass
    
    @abstractmethod
    def process(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        질문 처리
        
        Args:
            question: 사용자 질문
            context: 컨텍스트 정보
        
        Returns:
            처리 결과
        """
        pass
