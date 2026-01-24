
"""
채팅 모델
"""

from pydantic import Field
from typing import List, Optional, Dict, Any
from enum import Enum
from app.models.base import BaseModel


class MessageRole(str, Enum):
    """메시지 역할"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """채팅 메시지"""
    
    role: MessageRole = Field(..., description="메시지 역할")
    content: str = Field(..., description="메시지 내용")
    
    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "가장 인기있는 상품 추천해줘"
            }
        }


class ChatRequest(BaseModel):
    """채팅 요청"""

    brand_id: str = Field(..., description="브랜드 ID")
    message: str = Field(..., description="사용자 메시지")
    conversation_history: List[Message] = Field(
        default_factory=list,
        description="대화 히스토리"
    )
    question_type: Optional[str] = Field(
        default=None,
        description="질문 타입 (지정시 라우팅 건너뜀). 예: advisor, analytics, content_generation"
    )
    stream: bool = Field(default=False, description="스트리밍 여부")

    class Config:
        json_schema_extra = {
            "example": {
                "brand_id": "raceon",
                "message": "가장 인기있는 글러브 추천해줘",
                "conversation_history": [],
                "question_type": None,
                "stream": False
            }
        }


class RetrievalContext(BaseModel):
    """검색된 컨텍스트"""
    
    source: str = Field(..., description="검색 소스 (vector, graph, stats 등)")
    data: List[Dict[str, Any]] = Field(..., description="검색 결과 데이터")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="메타데이터")


class ChatResponse(BaseModel):
    """채팅 응답"""
    
    brand_id: str
    message: str = Field(..., description="응답 메시지")
    question_type: Optional[str] = Field(None, description="질문 타입")
    retrieval_contexts: List[RetrievalContext] = Field(
        default_factory=list,
        description="검색된 컨텍스트 목록"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="응답 메타데이터"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "brand_id": "raceon",
                "message": "WINNING 글러브를 추천합니다. 프로복서들이 선호하는...",
                "question_type": "product_recommendation",
                "retrieval_contexts": [
                    {
                        "source": "neo4j_products",
                        "data": [{"name": "WINNING Gloves", "price": 350000}],
                        "metadata": {"count": 1}
                    }
                ],
                "metadata": {
                    "response_time": 1.23,
                    "tokens_used": 450
                }
            }
        }
