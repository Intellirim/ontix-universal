### **app/features/analytics/models.py**

"""
Analytics Feature Models
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class PostMetrics(BaseModel):
    """게시물 메트릭"""
    id: str
    content: str
    likes: int
    comments: int
    shares: int
    engagement_rate: float
    created_at: datetime


class AnalyticsTimeRange(BaseModel):
    """분석 기간"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    days: int = Field(default=30, ge=1, le=365)


class AnalyticsRequest(BaseModel):
    """분석 요청"""
    question: str
    time_range: Optional[AnalyticsTimeRange] = None
    metrics: List[str] = ["likes", "comments", "engagement_rate"]
    top_n: int = Field(default=10, ge=1, le=100)


class AnalyticsInsight(BaseModel):
    """분석 인사이트"""
    metric: str
    value: float
    trend: str  # "increasing", "decreasing", "stable"
    description: str


class AnalyticsResponse(BaseModel):
    """분석 응답"""
    summary: str
    top_posts: List[PostMetrics]
    insights: List[AnalyticsInsight]
    total_metrics: dict
    metadata: dict = {}
