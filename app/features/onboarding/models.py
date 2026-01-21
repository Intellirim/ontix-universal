### **app/features/onboarding/models.py**

"""
Onboarding Feature Models
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class OnboardingStage(str, Enum):
    """온보딩 단계"""
    WELCOME = "welcome"
    FEATURE_INTRO = "feature_intro"
    FIRST_ACTION = "first_action"
    COMPLETION = "completion"


class OnboardingStep(BaseModel):
    """온보딩 스텝"""
    step_number: int
    title: str
    description: str
    action_required: Optional[str] = None
    completed: bool = False


class OnboardingRequest(BaseModel):
    """온보딩 요청"""
    user_id: str
    current_stage: OnboardingStage = OnboardingStage.WELCOME
    user_goals: Optional[List[str]] = None
    skip_intro: bool = False


class OnboardingProgress(BaseModel):
    """온보딩 진행 상황"""
    total_steps: int
    completed_steps: int
    current_step: int
    percentage: float = Field(ge=0.0, le=100.0)


class OnboardingResponse(BaseModel):
    """온보딩 응답"""
    message: str
    current_stage: OnboardingStage
    next_stage: Optional[OnboardingStage] = None
    steps: List[OnboardingStep]
    progress: OnboardingProgress
    tips: List[str] = []
    metadata: dict = {}
