"""
Instagram Adapter
apify/instagram-scraper 결과를 공통 포맷으로 변환

실제 데이터 형식:
{
    "id": "3029218572739856426",
    "type": "Sidecar",
    "shortCode": "CoJ8j4PLMgq",
    "caption": "Whilst the selfie is not high quality...",
    "hashtags": [],
    "mentions": [],
    "url": "https://www.instagram.com/p/CoJ8j4PLMgq/",
    "commentsCount": 61,
    "firstComment": "Cute",
    "latestComments": [
        {
            "id": "18073818227593872",
            "text": "Cute",
            "ownerUsername": "therealklopp",
            "ownerProfilePicUrl": "...",
            "timestamp": "2025-12-22T17:24:48.000Z",
            "repliesCount": 0,
            "replies": [],
            "likesCount": 0,
            "owner": {
                "id": "9169100897",
                "is_verified": false,
                "profile_pic_url": "...",
                "username": "therealklopp"
            }
        }
    ]
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


class InstagramAdapter(BaseSNSAdapter):
    """Instagram 플랫폼 어댑터 (apify/instagram-scraper)"""

    def __init__(self):
        super().__init__(PlatformType.INSTAGRAM)

    def validate_raw_data(self, raw_data: Dict[str, Any]) -> bool:
        """Instagram 데이터 유효성 검증"""
        required_fields = ["id"]
        return all(field in raw_data for field in required_fields)

    def _extract_username_from_url(self, url: str) -> str:
        """URL에서 username 추출 시도"""
        if not url:
            return "unknown"

        # https://www.instagram.com/p/CoJ8j4PLMgq/ 형식에서는 username을 알 수 없음
        # https://www.instagram.com/username/ 형식인 경우만 추출
        match = re.search(r'instagram\.com/([^/p][^/]*?)(?:/|$)', url)
        if match:
            return match.group(1)

        return "unknown"

    def _determine_content_type(self, raw_type: str) -> ContentType:
        """Instagram 타입을 ContentType으로 변환"""
        type_mapping = {
            "Image": ContentType.POST,
            "Sidecar": ContentType.POST,  # 여러 이미지/비디오
            "Video": ContentType.VIDEO,
            "Reel": ContentType.REEL,
            "Story": ContentType.STORY,
        }
        return type_mapping.get(raw_type, ContentType.POST)

    def parse_actor(self, raw_data: Dict[str, Any]) -> ActorDTO:
        """Instagram 사용자 정보 파싱"""
        # apify/instagram-scraper는 게시물 레벨에서 owner 정보가 제한적
        # ownerUsername, ownerId 등이 있을 수 있음
        username = raw_data.get("ownerUsername", "")
        owner_id = raw_data.get("ownerId", "")

        if not username:
            # URL에서 username 추출 시도
            username = self._extract_username_from_url(raw_data.get("url", ""))

        profile_url = f"https://instagram.com/{username}" if username and username != "unknown" else ""

        return ActorDTO(
            platform=self.platform,
            actor_id=owner_id or username,
            username=username,
            display_name=raw_data.get("ownerFullName") or username,
            profile_url=profile_url,
            metadata={
                "owner_id": owner_id,
            },
        )

    def parse_content(self, raw_data: Dict[str, Any]) -> ContentDTO:
        """Instagram 콘텐츠 정보 파싱"""
        actor = self.parse_actor(raw_data)

        # 콘텐츠 ID와 URL
        content_id = raw_data.get("id", "")
        short_code = raw_data.get("shortCode", "")
        url = raw_data.get("url", "")

        if not url and short_code:
            url = f"https://www.instagram.com/p/{short_code}/"

        # 콘텐츠 타입 결정
        raw_type = raw_data.get("type", "Image")
        content_type = self._determine_content_type(raw_type)

        # 날짜 파싱 - "timestamp" 필드 (ISO 형식)
        created_at = None
        if raw_data.get("timestamp"):
            try:
                ts = raw_data["timestamp"]
                if isinstance(ts, str):
                    created_at = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                elif isinstance(ts, (int, float)):
                    created_at = datetime.fromtimestamp(ts)
            except:
                pass

        # 텍스트 (caption)
        text = raw_data.get("caption", "")

        # 해시태그와 멘션 (배열로 직접 제공됨)
        hashtags = raw_data.get("hashtags", [])
        mentions = raw_data.get("mentions", [])

        # 텍스트에서도 추출 (빈 배열인 경우)
        if not hashtags and text:
            hashtags = [word[1:] for word in text.split() if word.startswith("#")]
        if not mentions and text:
            mentions = [word[1:] for word in text.split() if word.startswith("@")]

        # 메트릭
        like_count = raw_data.get("likesCount", 0) or raw_data.get("likes", 0) or 0
        comment_count = raw_data.get("commentsCount", 0) or 0
        view_count = raw_data.get("videoViewCount", 0) or raw_data.get("viewCount", 0) or 0

        # 미디어 URL
        media_urls = []
        if raw_data.get("displayUrl"):
            media_urls.append(raw_data["displayUrl"])
        if raw_data.get("videoUrl"):
            media_urls.append(raw_data["videoUrl"])

        # 썸네일
        thumbnail_url = raw_data.get("displayUrl") or raw_data.get("thumbnailUrl")

        return ContentDTO(
            platform=self.platform,
            content_id=content_id,
            content_type=content_type,
            author=actor,
            text=text,
            url=url,
            created_at=created_at,
            like_count=like_count,
            comment_count=comment_count,
            share_count=0,  # Instagram에서 제공하지 않음
            view_count=view_count,
            media_urls=media_urls,
            thumbnail_url=thumbnail_url,
            hashtags=hashtags,
            mentions=mentions,
            location=raw_data.get("locationName"),
            metadata={
                "short_code": short_code,
                "type": raw_type,
                "first_comment": raw_data.get("firstComment"),
                "dimensions": {
                    "width": raw_data.get("dimensionsWidth"),
                    "height": raw_data.get("dimensionsHeight"),
                },
            },
        )

    def parse_interactions(self, raw_data: Dict[str, Any]) -> List[InteractionDTO]:
        """Instagram 댓글 파싱"""
        interactions = []
        latest_comments = raw_data.get("latestComments", [])
        content_id = raw_data.get("id", "")

        for comment_data in latest_comments:
            # 댓글 작성자 정보
            owner = comment_data.get("owner", {})
            owner_username = comment_data.get("ownerUsername") or owner.get("username", "")

            actor = ActorDTO(
                platform=self.platform,
                actor_id=owner.get("id", owner_username),
                username=owner_username,
                verified=owner.get("is_verified", False),
                profile_url=f"https://instagram.com/{owner_username}" if owner_username else "",
                avatar_url=comment_data.get("ownerProfilePicUrl") or owner.get("profile_pic_url"),
            )

            # 날짜 파싱 - "timestamp" 필드 (ISO 형식)
            created_at = None
            if comment_data.get("timestamp"):
                try:
                    ts = comment_data["timestamp"]
                    if isinstance(ts, str):
                        created_at = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    elif isinstance(ts, (int, float)):
                        created_at = datetime.fromtimestamp(ts)
                except:
                    pass

            interaction = InteractionDTO(
                platform=self.platform,
                interaction_id=comment_data.get("id", ""),
                content_id=content_id,
                author=actor,
                text=comment_data.get("text", ""),
                created_at=created_at,
                like_count=comment_data.get("likesCount", 0) or 0,
                metadata={
                    "replies_count": comment_data.get("repliesCount", 0),
                    "replies": comment_data.get("replies", []),
                },
            )
            interactions.append(interaction)

        return interactions
