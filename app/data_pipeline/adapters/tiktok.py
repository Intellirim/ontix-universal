"""
TikTok Adapter
clockworks/tiktok-scraper 결과를 공통 포맷으로 변환

실제 데이터 형식 (flat 구조 - 점 표기법 키):
{
    "authorMeta.avatar": "https://...",
    "authorMeta.name": "themalachibarton",
    "text": "Certified #fyp @Mk🇪🇹 ",
    "diggCount": 3100000,
    "shareCount": 52800,
    "playCount": 27500000,
    "commentCount": 7320,
    "collectCount": 256837,
    "videoMeta.duration": 15,
    "musicMeta.musicName": "Not Like Us",
    "musicMeta.musicAuthor": "Kendrick Lamar",
    "musicMeta.musicOriginal": false,
    "createTimeISO": "2024-05-19T01:44:18.000Z",
    "webVideoUrl": "https://www.tiktok.com/@themalachibarton/video/7370520570070338859"
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


class TikTokAdapter(BaseSNSAdapter):
    """TikTok 플랫폼 어댑터 (clockworks/tiktok-scraper, flat 구조)"""

    def __init__(self):
        super().__init__(PlatformType.TIKTOK)

    def validate_raw_data(self, raw_data: Dict[str, Any]) -> bool:
        """TikTok 데이터 유효성 검증"""
        # webVideoUrl 또는 authorMeta.name이 있으면 유효
        has_url = "webVideoUrl" in raw_data
        has_author = "authorMeta.name" in raw_data
        return has_url or has_author

    def _extract_video_id_from_url(self, url: str) -> str:
        """URL에서 비디오 ID 추출"""
        if not url:
            return ""

        # https://www.tiktok.com/@username/video/7370520570070338859 형식
        match = re.search(r'/video/(\d+)', url)
        if match:
            return match.group(1)

        return ""

    def parse_actor(self, raw_data: Dict[str, Any]) -> ActorDTO:
        """TikTok 사용자 정보 파싱 (flat 구조)"""
        # flat 구조에서 authorMeta.* 필드 추출
        username = raw_data.get("authorMeta.name", "")
        avatar_url = raw_data.get("authorMeta.avatar", "")

        # nickname은 제공되지 않으면 username 사용
        display_name = raw_data.get("authorMeta.nickName") or raw_data.get("authorMeta.nickname") or username

        # 프로필 URL
        profile_url = f"https://www.tiktok.com/@{username}" if username else ""

        return ActorDTO(
            platform=self.platform,
            actor_id=username,  # username을 ID로 사용
            username=username,
            display_name=display_name,
            profile_url=profile_url,
            avatar_url=avatar_url,
            metadata={
                "avatar": avatar_url,
            },
        )

    def parse_content(self, raw_data: Dict[str, Any]) -> ContentDTO:
        """TikTok 비디오 정보 파싱 (flat 구조)"""
        actor = self.parse_actor(raw_data)

        # 비디오 URL과 ID
        video_url = raw_data.get("webVideoUrl", "")
        video_id = self._extract_video_id_from_url(video_url)

        if not video_url and actor.username:
            video_url = f"https://www.tiktok.com/@{actor.username}"

        # 날짜 파싱 - "2024-05-19T01:44:18.000Z" 형식
        created_at = None
        if raw_data.get("createTimeISO"):
            try:
                date_str = raw_data["createTimeISO"].replace("Z", "+00:00")
                created_at = datetime.fromisoformat(date_str)
            except:
                pass

        # 텍스트 및 해시태그/멘션 추출
        text = raw_data.get("text", "")
        hashtags = []
        mentions = []

        if text:
            hashtags = [word[1:] for word in text.split() if word.startswith("#")]
            mentions = [word[1:] for word in text.split() if word.startswith("@")]

        # 메트릭 (flat 구조)
        like_count = raw_data.get("diggCount", 0) or 0
        share_count = raw_data.get("shareCount", 0) or 0
        view_count = raw_data.get("playCount", 0) or 0
        comment_count = raw_data.get("commentCount", 0) or 0
        collect_count = raw_data.get("collectCount", 0) or 0

        # 비디오 duration (flat 구조)
        duration = raw_data.get("videoMeta.duration", 0) or 0

        # 음악 정보 (flat 구조)
        music_name = raw_data.get("musicMeta.musicName", "")
        music_author = raw_data.get("musicMeta.musicAuthor", "")
        music_original = raw_data.get("musicMeta.musicOriginal", False)

        return ContentDTO(
            platform=self.platform,
            content_id=video_id,
            content_type=ContentType.VIDEO,
            author=actor,
            text=text,
            url=video_url,
            created_at=created_at,
            like_count=like_count,
            comment_count=comment_count,
            share_count=share_count,
            view_count=view_count,
            media_urls=[video_url] if video_url else [],
            hashtags=hashtags,
            mentions=mentions,
            metadata={
                "duration": duration,
                "collect_count": collect_count,
                "music_name": music_name,
                "music_author": music_author,
                "music_original": music_original,
            },
        )

    def parse_interactions(self, raw_data: Dict[str, Any]) -> List[InteractionDTO]:
        """TikTok 댓글 파싱 (clockworks/tiktok-scraper는 댓글을 제공하지 않음)"""
        # clockworks/tiktok-scraper는 댓글 데이터를 포함하지 않음
        return []
