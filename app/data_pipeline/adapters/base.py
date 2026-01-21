"""
Base SNS Adapter
모든 SNS 어댑터의 추상 클래스
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..domain.models import ActorDTO, ContentDTO, InteractionDTO, PlatformType


class BaseSNSAdapter(ABC):
    """SNS 플랫폼 어댑터 베이스 클래스"""

    def __init__(self, platform: PlatformType):
        self.platform = platform

    @abstractmethod
    def parse_actor(self, raw_data: Dict[str, Any]) -> ActorDTO:
        """원시 데이터에서 ActorDTO 추출"""
        pass

    @abstractmethod
    def parse_content(self, raw_data: Dict[str, Any]) -> ContentDTO:
        """원시 데이터에서 ContentDTO 추출"""
        pass

    @abstractmethod
    def parse_interactions(self, raw_data: Dict[str, Any]) -> List[InteractionDTO]:
        """원시 데이터에서 InteractionDTO 리스트 추출"""
        pass

    @abstractmethod
    def validate_raw_data(self, raw_data: Dict[str, Any]) -> bool:
        """원시 데이터 유효성 검증"""
        pass

    def transform(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        원시 데이터를 공통 포맷으로 변환
        Returns:
            {
                "actor": ActorDTO,
                "content": ContentDTO,
                "interactions": List[InteractionDTO]
            }
        """
        if not self.validate_raw_data(raw_data):
            raise ValueError(f"Invalid raw data for platform {self.platform}")

        actor = self.parse_actor(raw_data)
        content = self.parse_content(raw_data)
        interactions = self.parse_interactions(raw_data)

        return {
            "actor": actor,
            "content": content,
            "interactions": interactions,
        }
