### **app/features/content_generation/models.py**


"""
Content Generation Feature Models
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class ContentType(str, Enum):
    """콘텐츠 타입"""
    BLOG_POST = "blog_post"
    SOCIAL_CAPTION = "social_caption"
    PRODUCT_DESCRIPTION = "product_description"
    EMAIL = "email"
    ADVERTISEMENT = "advertisement"


class ContentTone(str, Enum):
    """콘텐츠 톤"""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    FORMAL = "formal"
    CREATIVE = "creative"
    PHILOSOPHICAL = "philosophical"


class ContentGenerationRequest(BaseModel):
    """콘텐츠 생성 요청"""
    prompt: str
    content_type: ContentType
    tone: ContentTone = ContentTone.PROFESSIONAL
    max_length: int = Field(default=500, ge=100, le=5000)
    include_hashtags: bool = False
    target_audience: Optional[str] = None
    keywords: Optional[List[str]] = None


class GeneratedContent(BaseModel):
    """생성된 콘텐츠"""
    content: str
    title: Optional[str] = None
    hashtags: List[str] = []
    word_count: int
    estimated_read_time: int  # minutes


class ContentGenerationResponse(BaseModel):
    """콘텐츠 생성 응답"""
    generated: GeneratedContent
    alternatives: List[str] = []
    metadata: dict = {}
