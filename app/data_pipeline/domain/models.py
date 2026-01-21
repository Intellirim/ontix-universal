"""
범용 SNS 데이터 모델
모든 SNS 플랫폼의 데이터를 추상화하는 공통 DTO
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum


class PlatformType(str, Enum):
    """SNS 플랫폼 타입"""
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    TWITTER = "twitter"


class ContentType(str, Enum):
    """콘텐츠 타입"""
    POST = "post"
    VIDEO = "video"
    SHORT = "short"
    STORY = "story"
    REEL = "reel"
    TWEET = "tweet"


@dataclass
class ActorDTO:
    """크리에이터/사용자 정보"""
    platform: PlatformType
    actor_id: str
    username: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    follower_count: Optional[int] = None
    following_count: Optional[int] = None
    verified: bool = False
    profile_url: Optional[str] = None
    avatar_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_metadata: bool = True) -> Dict[str, Any]:
        """ActorDTO를 딕셔너리로 변환"""
        result = {
            "platform": self.platform.value,
            "actor_id": self.actor_id,
            "username": self.username,
            "display_name": self.display_name,
            "description": self.description,
            "follower_count": self.follower_count,
            "following_count": self.following_count,
            "verified": self.verified,
            "profile_url": self.profile_url,
            "avatar_url": self.avatar_url,
        }
        if include_metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class ContentDTO:
    """콘텐츠 정보"""
    platform: PlatformType
    content_id: str
    content_type: ContentType
    author: ActorDTO
    text: Optional[str] = None
    url: str = ""
    created_at: Optional[datetime] = None

    # 인터랙션 메트릭
    like_count: int = 0
    comment_count: int = 0
    share_count: int = 0
    view_count: int = 0

    # 미디어
    media_urls: List[str] = field(default_factory=list)
    thumbnail_url: Optional[str] = None

    # 추가 정보
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    location: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_metadata: bool = True) -> Dict[str, Any]:
        """
        ContentDTO를 딕셔너리로 변환 (metadata 보존)

        Args:
            include_metadata: metadata 필드 포함 여부

        Returns:
            딕셔너리 형태의 콘텐츠 데이터
        """
        result = {
            "platform": self.platform.value,
            "content_id": self.content_id,
            "content_type": self.content_type.value if self.content_type else None,
            "author": self.author.to_dict() if hasattr(self.author, 'to_dict') else {
                "actor_id": self.author.actor_id,
                "username": self.author.username,
                "platform": self.author.platform.value,
            } if self.author else None,
            "text": self.text,
            "url": self.url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "like_count": self.like_count,
            "comment_count": self.comment_count,
            "share_count": self.share_count,
            "view_count": self.view_count,
            "media_urls": self.media_urls,
            "thumbnail_url": self.thumbnail_url,
            "hashtags": self.hashtags,
            "mentions": self.mentions,
            "location": self.location,
        }
        if include_metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentDTO':
        """
        딕셔너리에서 ContentDTO 생성 (metadata 복원)

        Args:
            data: 콘텐츠 데이터 딕셔너리

        Returns:
            ContentDTO 인스턴스
        """
        # Actor 복원
        author_data = data.get("author", {})
        author = ActorDTO(
            platform=PlatformType(author_data.get("platform", "instagram")),
            actor_id=author_data.get("actor_id", ""),
            username=author_data.get("username", ""),
            metadata=author_data.get("metadata", {}),
        )

        # created_at 파싱
        created_at = None
        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        return cls(
            platform=PlatformType(data.get("platform", "instagram")),
            content_id=data.get("content_id", ""),
            content_type=ContentType(data.get("content_type", "post")) if data.get("content_type") else ContentType.POST,
            author=author,
            text=data.get("text"),
            url=data.get("url", ""),
            created_at=created_at,
            like_count=data.get("like_count", 0),
            comment_count=data.get("comment_count", 0),
            share_count=data.get("share_count", 0),
            view_count=data.get("view_count", 0),
            media_urls=data.get("media_urls", []),
            thumbnail_url=data.get("thumbnail_url"),
            hashtags=data.get("hashtags", []),
            mentions=data.get("mentions", []),
            location=data.get("location"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class InteractionDTO:
    """사용자 인터랙션 (댓글, 답글 등)"""
    platform: PlatformType
    interaction_id: str
    content_id: str
    author: ActorDTO
    text: str
    created_at: Optional[datetime] = None
    like_count: int = 0
    parent_id: Optional[str] = None  # 답글의 경우
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_metadata: bool = True) -> Dict[str, Any]:
        """InteractionDTO를 딕셔너리로 변환"""
        result = {
            "platform": self.platform.value,
            "interaction_id": self.interaction_id,
            "content_id": self.content_id,
            "author": self.author.to_dict() if self.author else None,
            "text": self.text,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "like_count": self.like_count,
            "parent_id": self.parent_id,
        }
        if include_metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class TopicDTO:
    """추출된 토픽/주제"""
    name: str
    category: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_metadata: bool = True) -> Dict[str, Any]:
        """TopicDTO를 딕셔너리로 변환"""
        result = {
            "name": self.name,
            "category": self.category,
            "confidence": self.confidence,
        }
        if include_metadata:
            result["metadata"] = self.metadata
        return result
