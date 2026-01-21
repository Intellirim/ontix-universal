### **app/features/social_monitoring/models.py**

"""
Social Monitoring Feature Models
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum


class Platform(str, Enum):
    """소셜 플랫폼"""
    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    YOUTUBE = "youtube"


class SentimentType(str, Enum):
    """감정 분석"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class SocialMention(BaseModel):
    """소셜 멘션"""
    id: str
    platform: Platform
    content: str
    author: str
    sentiment: SentimentType
    engagement: int
    created_at: datetime
    url: Optional[str] = None


class MonitoringRequest(BaseModel):
    """모니터링 요청"""
    keywords: List[str]
    platforms: List[Platform]
    time_range_hours: int = Field(default=24, ge=1, le=168)
    sentiment_filter: Optional[SentimentType] = None
    min_engagement: int = 0


class MonitoringAlert(BaseModel):
    """모니터링 알림"""
    alert_type: str
    severity: str  # "low", "medium", "high"
    message: str
    mentions: List[SocialMention]
    created_at: datetime


class MonitoringResponse(BaseModel):
    """모니터링 응답"""
    total_mentions: int
    sentiment_breakdown: dict  # {"positive": 10, "neutral": 5, "negative": 2}
    top_mentions: List[SocialMention]
    alerts: List[MonitoringAlert]
    trends: List[str] = []
    metadata: dict = {}
