"""
Twitter/X Adapter
apidojo/tweet-scraper 결과를 공통 포맷으로 변환

실제 데이터 형식:
{
    "url": "https://x.com/username/status/123",
    "twitterUrl": "https://twitter.com/username/status/123",
    "id": "123",
    "text": "...",
    "retweetCount": 0,
    "replyCount": 0,
    "likeCount": 0,
    "quoteCount": 0,
    "createdAt": "Tue Jan 13 11:39:04 +0000 2026",
    "bookmarkCount": 0,
    "isRetweet": false,
    "isQuote": false
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


class TwitterAdapter(BaseSNSAdapter):
    """Twitter/X 플랫폼 어댑터 (apidojo/tweet-scraper)"""

    def __init__(self):
        super().__init__(PlatformType.TWITTER)

    def validate_raw_data(self, raw_data: Dict[str, Any]) -> bool:
        """Twitter 데이터 유효성 검증"""
        required_fields = ["id"]
        return all(field in raw_data for field in required_fields)

    def _extract_username_from_url(self, url: str) -> str:
        """URL에서 username 추출"""
        if not url:
            return "unknown"

        # https://x.com/username/status/123 또는
        # https://twitter.com/username/status/123 형식
        patterns = [
            r'(?:x\.com|twitter\.com)/([^/]+)/status/',
            r'(?:x\.com|twitter\.com)/([^/]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return "unknown"

    def parse_actor(self, raw_data: Dict[str, Any]) -> ActorDTO:
        """Twitter 사용자 정보 파싱 (URL에서 추출)"""
        # apidojo/tweet-scraper는 사용자 정보를 별도로 제공하지 않음
        # URL에서 username 추출
        url = raw_data.get("url") or raw_data.get("twitterUrl", "")
        username = self._extract_username_from_url(url)

        profile_url = f"https://twitter.com/{username}"

        return ActorDTO(
            platform=self.platform,
            actor_id=username,  # username을 ID로 사용
            username=username,
            display_name=username,
            profile_url=profile_url,
            metadata={
                "source_url": url,
            },
        )

    def parse_content(self, raw_data: Dict[str, Any]) -> ContentDTO:
        """Twitter 트윗 정보 파싱"""
        actor = self.parse_actor(raw_data)

        # 트윗 ID와 URL
        tweet_id = raw_data.get("id", "")
        tweet_url = raw_data.get("url") or raw_data.get("twitterUrl", "")

        if not tweet_url and tweet_id:
            tweet_url = f"https://twitter.com/{actor.username}/status/{tweet_id}"

        # 날짜 파싱 - "Tue Jan 13 11:39:04 +0000 2026" 형식
        created_at = None
        if raw_data.get("createdAt"):
            try:
                created_at = datetime.strptime(
                    raw_data["createdAt"],
                    "%a %b %d %H:%M:%S %z %Y"
                )
            except:
                pass

        # 트윗 텍스트
        text = raw_data.get("text", "")

        # 해시태그 및 멘션 추출 (텍스트에서)
        hashtags = []
        mentions = []

        if text:
            hashtags = [word[1:] for word in text.split() if word.startswith("#")]
            mentions = [word[1:] for word in text.split() if word.startswith("@")]

        # 미디어 URL (텍스트에서 t.co 링크 추출)
        media_urls = []
        if text:
            urls = re.findall(r'https?://t\.co/\w+', text)
            media_urls = urls

        # 메트릭
        like_count = raw_data.get("likeCount", 0) or 0
        retweet_count = raw_data.get("retweetCount", 0) or 0
        reply_count = raw_data.get("replyCount", 0) or 0
        quote_count = raw_data.get("quoteCount", 0) or 0
        bookmark_count = raw_data.get("bookmarkCount", 0) or 0

        # share_count = retweet + quote
        share_count = retweet_count + quote_count

        return ContentDTO(
            platform=self.platform,
            content_id=tweet_id,
            content_type=ContentType.TWEET,
            author=actor,
            text=text,
            url=tweet_url,
            created_at=created_at,
            like_count=like_count,
            comment_count=reply_count,
            share_count=share_count,
            view_count=0,  # apidojo/tweet-scraper에서 제공하지 않음
            media_urls=media_urls,
            hashtags=hashtags,
            mentions=mentions,
            metadata={
                "retweet_count": retweet_count,
                "quote_count": quote_count,
                "bookmark_count": bookmark_count,
                "is_retweet": raw_data.get("isRetweet", False),
                "is_quote": raw_data.get("isQuote", False),
            },
        )

    def parse_interactions(self, raw_data: Dict[str, Any]) -> List[InteractionDTO]:
        """Twitter 답글 파싱 (apidojo/tweet-scraper는 답글을 제공하지 않음)"""
        # apidojo/tweet-scraper는 답글 데이터를 포함하지 않음
        return []
