"""
YouTube Adapter
streamers/youtube-scraper 결과를 공통 포맷으로 변환

실제 데이터 형식:
{
    "title": "Video Title",
    "id": "g1Ll9OlFwEQ",
    "url": "https://www.youtube.com/watch?v=g1Ll9OlFwEQ",
    "viewCount": 11974,
    "date": "2022-08-22T15:26:13.000Z",
    "likes": 154,
    "channelName": "Apify",
    "channelUrl": "https://www.youtube.com/@Apify",
    "numberOfSubscribers": 11600,
    "duration": "00:03:14"
}
"""
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from .base import BaseSNSAdapter
from ..domain.models import (
    ActorDTO,
    ContentDTO,
    InteractionDTO,
    PlatformType,
    ContentType,
)


class YouTubeAdapter(BaseSNSAdapter):
    """YouTube 플랫폼 어댑터 (streamers/youtube-scraper)"""

    def __init__(self):
        super().__init__(PlatformType.YOUTUBE)

    def validate_raw_data(self, raw_data: Dict[str, Any]) -> bool:
        """YouTube 데이터 유효성 검증"""
        required_fields = ["id"]
        return all(field in raw_data for field in required_fields)

    def _extract_channel_id_from_url(self, channel_url: str) -> str:
        """채널 URL에서 ID 또는 handle 추출"""
        if not channel_url:
            return "unknown"

        # https://www.youtube.com/@Apify 형식
        match = re.search(r'youtube\.com/@([^/]+)', channel_url)
        if match:
            return match.group(1)

        # https://www.youtube.com/channel/UC... 형식
        match = re.search(r'youtube\.com/channel/([^/]+)', channel_url)
        if match:
            return match.group(1)

        return "unknown"

    def _parse_duration_to_seconds(self, duration: str) -> int:
        """duration 문자열을 초로 변환 (HH:MM:SS 또는 MM:SS)"""
        if not duration:
            return 0

        try:
            parts = duration.split(":")
            if len(parts) == 3:  # HH:MM:SS
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            elif len(parts) == 2:  # MM:SS
                return int(parts[0]) * 60 + int(parts[1])
            else:
                return int(parts[0])
        except:
            return 0

    def parse_actor(self, raw_data: Dict[str, Any]) -> ActorDTO:
        """YouTube 채널 정보 파싱"""
        channel_name = raw_data.get("channelName", "")
        channel_url = raw_data.get("channelUrl", "")
        channel_id = self._extract_channel_id_from_url(channel_url)
        subscriber_count = raw_data.get("numberOfSubscribers")

        return ActorDTO(
            platform=self.platform,
            actor_id=channel_id,
            username=channel_name,
            display_name=channel_name,
            follower_count=subscriber_count,
            profile_url=channel_url,
            metadata={
                "subscriber_count": subscriber_count,
            },
        )

    def parse_content(self, raw_data: Dict[str, Any]) -> ContentDTO:
        """YouTube 비디오 정보 파싱"""
        actor = self.parse_actor(raw_data)

        # 비디오 ID와 URL
        video_id = raw_data.get("id", "")
        video_url = raw_data.get("url", "")
        if not video_url and video_id:
            video_url = f"https://www.youtube.com/watch?v={video_id}"

        # 콘텐츠 타입 결정 (duration 기반으로 Shorts 판별)
        duration_str = raw_data.get("duration", "")
        duration_seconds = self._parse_duration_to_seconds(duration_str)

        content_type = ContentType.VIDEO
        if duration_seconds > 0 and duration_seconds < 60:  # 60초 미만은 Shorts
            content_type = ContentType.SHORT

        # 날짜 파싱 - "2022-08-22T15:26:13.000Z" 형식
        created_at = None
        if raw_data.get("date"):
            try:
                date_str = raw_data["date"].replace("Z", "+00:00")
                created_at = datetime.fromisoformat(date_str)
            except:
                pass

        # 텍스트는 title 사용
        text = raw_data.get("title", "")

        # 해시태그 추출 (title에서)
        hashtags = []
        if text:
            hashtags = [word[1:] for word in text.split() if word.startswith("#")]

        # 메트릭
        view_count = raw_data.get("viewCount", 0) or 0
        like_count = raw_data.get("likes", 0) or 0

        # 썸네일 URL 생성 (YouTube 표준 형식)
        thumbnail_url = None
        if video_id:
            thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"

        return ContentDTO(
            platform=self.platform,
            content_id=video_id,
            content_type=content_type,
            author=actor,
            text=text,
            url=video_url,
            created_at=created_at,
            like_count=like_count,
            comment_count=0,  # streamers/youtube-scraper에서 제공하지 않음
            share_count=0,
            view_count=view_count,
            media_urls=[video_url],
            thumbnail_url=thumbnail_url,
            hashtags=hashtags,
            metadata={
                "duration": duration_str,
                "duration_seconds": duration_seconds,
                "subscriber_count": raw_data.get("numberOfSubscribers"),
            },
        )

    def parse_interactions(self, raw_data: Dict[str, Any]) -> List[InteractionDTO]:
        """YouTube 댓글 파싱 (streamers/youtube-scraper는 댓글을 제공하지 않음)"""
        # streamers/youtube-scraper는 댓글 데이터를 포함하지 않음
        return []
