"""
Apify Crawler Base Class
Apify API 호출을 위한 추상 클래스
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ApifyCrawlerBase(ABC):
    """Apify 크롤러 베이스 클래스"""

    def __init__(self, api_token: str):
        self.api_token = api_token

    @abstractmethod
    def get_actor_id(self) -> str:
        """Apify Actor ID 반환"""
        pass

    @abstractmethod
    def build_input(self, **kwargs) -> Dict[str, Any]:
        """Actor 실행을 위한 입력 데이터 생성"""
        pass

    @abstractmethod
    async def fetch_data(self, **kwargs) -> List[Dict[str, Any]]:
        """데이터 크롤링 실행"""
        pass

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """입력 데이터 유효성 검증"""
        return True

    async def run(self, **kwargs) -> List[Dict[str, Any]]:
        """
        크롤링 실행 및 결과 반환

        Args:
            **kwargs: Actor별 입력 파라미터

        Returns:
            크롤링된 원시 데이터 리스트
        """
        logger.info(f"Starting crawler for actor: {self.get_actor_id()}")

        input_data = self.build_input(**kwargs)

        if not self.validate_input(input_data):
            raise ValueError("Invalid input data")

        results = await self.fetch_data(**kwargs)

        logger.info(f"Crawling completed. Found {len(results)} items")

        return results
