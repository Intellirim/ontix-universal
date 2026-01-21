
"""
기본 모델
"""

from pydantic import BaseModel as PydanticBaseModel, ConfigDict
from typing import Optional
from datetime import datetime


class BaseModel(PydanticBaseModel):
    """
    기본 모델 클래스
    모든 모델의 베이스
    """
    
    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
