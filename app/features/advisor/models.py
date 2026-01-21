### **app/features/advisor/models.py**

"""
Advisor Feature Models
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class ExpertiseLevel(str, Enum):
    """전문성 레벨"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    PROFESSIONAL = "professional"
    PHILOSOPHICAL = "philosophical"


class AdvisorRequest(BaseModel):
    """조언 요청"""
    question: str
    expertise_level: ExpertiseLevel = ExpertiseLevel.INTERMEDIATE
    include_examples: bool = True
    focus_areas: Optional[List[str]] = None
    context: Optional[str] = None


class AdvisorStep(BaseModel):
    """조언 단계"""
    step_number: int
    title: str
    description: str
    action_items: List[str] = []


class AdvisorResponse(BaseModel):
    """조언 응답"""
    advice: str
    understanding: str
    key_considerations: List[str]
    recommended_approach: str
    next_steps: List[AdvisorStep]
    references: Optional[List[str]] = None
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: dict = {}
