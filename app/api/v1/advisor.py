"""
AI Advisor API - Dedicated Endpoint v1.0
비즈니스 어드바이저 전용 엔드포인트

일반 채팅(/chat)과 완전히 분리된 AI 어드바이저 전용 API
항상 BusinessAdvisorGenerator를 사용하여 전체 브랜드 데이터 분석 제공
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from app.generators.business_advisor import BusinessAdvisorGenerator
from app.core.context import QueryContext
from app.services.brand.config import get_brand_config
from app.services.chat.chat_storage import get_chat_storage
import logging
import time
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/advisor")


# ============================================================
# Request/Response Models
# ============================================================

class AdvisorRequest(BaseModel):
    """AI 어드바이저 요청"""
    brand_id: str = Field(..., description="브랜드 ID")
    message: str = Field(..., description="질문 메시지")
    session_id: Optional[str] = Field(None, description="세션 ID (선택)")


class AdvisorResponse(BaseModel):
    """AI 어드바이저 응답"""
    brand_id: str
    message: str
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationMessage(BaseModel):
    """대화 메시지"""
    role: str
    content: str


# ============================================================
# API Endpoints
# ============================================================

@router.post("", response_model=AdvisorResponse)
async def get_advice(
    request: AdvisorRequest,
    background_tasks: BackgroundTasks
):
    """
    AI 비즈니스 어드바이저 - 전용 엔드포인트

    이 엔드포인트는 항상 BusinessAdvisorGenerator를 사용합니다.
    일반 채팅과 달리 전체 브랜드 데이터를 분석하여 비즈니스 인사이트를 제공합니다.

    Features:
        - Neo4j 전체 브랜드 데이터 활용
        - 감정 분석 인사이트
        - 트렌드 및 성과 분석
        - 실행 가능한 비즈니스 추천

    Args:
        request: 어드바이저 요청 (brand_id, message)

    Returns:
        AdvisorResponse: 비즈니스 인사이트 응답
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]

    try:
        # 브랜드 설정 로드
        brand_config = get_brand_config(request.brand_id)
        if not brand_config:
            raise HTTPException(
                status_code=404,
                detail=f"Brand not found: {request.brand_id}"
            )

        # BusinessAdvisorGenerator 직접 사용 (라우팅 우회)
        advisor = BusinessAdvisorGenerator(brand_config)

        # 쿼리 컨텍스트 생성
        context = QueryContext(
            question=request.message,
            brand_id=request.brand_id,
            brand_config=brand_config
        )

        # 응답 생성
        response_message = advisor.generate(context)

        # 메트릭 수집
        elapsed_time = (time.time() - start_time) * 1000
        metrics = advisor.get_last_metrics()

        metadata = {
            'request_id': request_id,
            'response_time_ms': elapsed_time,
            'handled_by': 'BusinessAdvisorGenerator',
            'question_type': 'advisor',
        }

        if metrics:
            metadata['generation_time_ms'] = metrics.generation_time_ms
            metadata['llm_tokens'] = metrics.llm_tokens

        # 채팅 저장 (백그라운드)
        if request.session_id:
            background_tasks.add_task(
                _save_advisor_chat,
                request.session_id,
                request.brand_id,
                request.message,
                response_message,
                metadata
            )

        logger.info(
            f"Advisor response generated for {request.brand_id} "
            f"in {elapsed_time:.1f}ms"
        )

        return AdvisorResponse(
            brand_id=request.brand_id,
            message=response_message,
            session_id=request.session_id,
            metadata=metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Advisor API error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/with-session", response_model=AdvisorResponse)
async def get_advice_with_session(
    request: AdvisorRequest,
    background_tasks: BackgroundTasks
):
    """
    세션 기반 AI 어드바이저

    세션 ID가 없으면 자동으로 생성합니다.
    이전 대화 히스토리를 컨텍스트로 활용합니다.

    Args:
        request: 어드바이저 요청

    Returns:
        AdvisorResponse: 세션 ID가 포함된 응답
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())[:8]

    try:
        # 브랜드 설정 로드
        brand_config = get_brand_config(request.brand_id)
        if not brand_config:
            raise HTTPException(
                status_code=404,
                detail=f"Brand not found: {request.brand_id}"
            )

        # 세션 관리
        storage = get_chat_storage()
        session = storage.get_or_create_session(
            session_id=request.session_id,
            brand_id=request.brand_id
        )

        # 이전 대화 컨텍스트 (어드바이저 세션용)
        conversation_history = []
        history = storage.get_recent_context(session.id, max_messages=6)
        conversation_history = [
            {"role": h["role"], "content": h["content"]}
            for h in history
        ]

        # BusinessAdvisorGenerator 직접 사용
        advisor = BusinessAdvisorGenerator(brand_config)

        # 쿼리 컨텍스트 생성 (대화 히스토리 포함)
        context = QueryContext(
            question=request.message,
            brand_id=request.brand_id,
            brand_config=brand_config,
            conversation_history=conversation_history
        )

        # 응답 생성
        response_message = advisor.generate(context)

        # 메트릭 수집
        elapsed_time = (time.time() - start_time) * 1000
        metrics = advisor.get_last_metrics()

        metadata = {
            'request_id': request_id,
            'response_time_ms': elapsed_time,
            'handled_by': 'BusinessAdvisorGenerator',
            'question_type': 'advisor',
            'session_id': session.id,
        }

        if metrics:
            metadata['generation_time_ms'] = metrics.generation_time_ms
            metadata['llm_tokens'] = metrics.llm_tokens

        # 채팅 저장 (백그라운드)
        background_tasks.add_task(
            _save_advisor_chat,
            session.id,
            request.brand_id,
            request.message,
            response_message,
            metadata
        )

        logger.info(
            f"Advisor session response for {request.brand_id} "
            f"session={session.id} in {elapsed_time:.1f}ms"
        )

        return AdvisorResponse(
            brand_id=request.brand_id,
            message=response_message,
            session_id=session.id,
            metadata=metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Advisor with session error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Helper Functions
# ============================================================

def _save_advisor_chat(
    session_id: str,
    brand_id: str,
    user_message: str,
    assistant_message: str,
    metadata: Dict[str, Any]
):
    """어드바이저 채팅 저장 (백그라운드)"""
    try:
        storage = get_chat_storage()

        # 세션 조회 또는 생성
        session = storage.get_or_create_session(session_id, brand_id)

        # 대화 턴 저장
        storage.save_conversation_turn(
            session_id=session.id,
            user_message=user_message,
            assistant_message=assistant_message,
            metadata={**metadata, 'type': 'advisor'}
        )

        logger.debug(f"Advisor chat saved to session {session.id}")
    except Exception as e:
        logger.error(f"Advisor chat storage error: {e}")
