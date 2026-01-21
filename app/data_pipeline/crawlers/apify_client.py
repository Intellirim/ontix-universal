"""
Apify API Client
Apify API를 호출하여 크롤링 데이터를 가져오는 클라이언트
"""
import os
from typing import Dict, Any, List, Optional
import logging
from apify_client import ApifyClient as ApifySDK

logger = logging.getLogger(__name__)


class ApifyClient:
    """Apify API 클라이언트"""

    def __init__(self, api_token: Optional[str] = None):
        """
        Args:
            api_token: Apify API 토큰. None이면 환경변수에서 읽음
        """
        self.api_token = api_token or os.getenv("APIFY_TOKEN") or os.getenv("APIFY_API_TOKEN")
        if not self.api_token:
            raise ValueError("APIFY_TOKEN is required")

        self.client = ApifySDK(self.api_token)

    async def run_actor(
        self,
        actor_id: str,
        run_input: Dict[str, Any],
        timeout_secs: int = 300,
    ) -> List[Dict[str, Any]]:
        """
        Apify Actor 실행

        Args:
            actor_id: Actor ID (예: "apify/instagram-scraper")
            run_input: Actor 입력 데이터
            timeout_secs: 타임아웃 (초)

        Returns:
            크롤링된 데이터 리스트
        """
        logger.info(f"Running Apify actor: {actor_id}")
        logger.debug(f"Input: {run_input}")

        # Actor 실행
        run = self.client.actor(actor_id).call(
            run_input=run_input,
            timeout_secs=timeout_secs,
        )

        # 결과 가져오기
        items = []
        for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
            items.append(item)

        logger.info(f"Retrieved {len(items)} items from actor: {actor_id}")
        return items

    async def get_dataset_items(
        self,
        dataset_id: str,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        기존 데이터셋에서 데이터 가져오기

        Args:
            dataset_id: 데이터셋 ID
            offset: 시작 오프셋
            limit: 최대 개수

        Returns:
            데이터 리스트
        """
        logger.info(f"Fetching dataset: {dataset_id}")

        items = []
        for item in self.client.dataset(dataset_id).iterate_items(
            offset=offset, limit=limit
        ):
            items.append(item)

        logger.info(f"Retrieved {len(items)} items from dataset: {dataset_id}")
        return items

    def list_runs(self, actor_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Actor의 최근 실행 목록 조회

        Args:
            actor_id: Actor ID
            limit: 최대 개수

        Returns:
            실행 목록
        """
        runs = self.client.actor(actor_id).runs().list(limit=limit)
        return runs.items
