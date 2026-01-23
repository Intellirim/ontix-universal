"""
Chat API - Production Grade v2.0
채팅 엔드포인트

Features:
- Validation Filter v2.0 통합
- 등급 기반 응답 품질 관리
- 자동 재생성 및 사과 메시지
- 상세 메트릭 추적
- 채팅 히스토리 저장 및 분석
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from app.models.chat import ChatRequest, ChatResponse
from app.core.engine import UniversalEngine
from app.services.platform.analytics import get_analytics_service
from app.services.chat.chat_storage import get_chat_storage
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat")


# ============================================================
# Extended Response Models
# ============================================================

class ValidationInfo(BaseModel):
    """검증 정보"""
    grade: str = Field(..., description="A~F 등급")
    score: float = Field(..., description="0.0~1.0 점수")
    status: str = Field(..., description="passed/warning/failed")
    retries: int = Field(default=0, description="재생성 시도 횟수")
    suggestions: Optional[List[str]] = Field(default=None, description="개선 제안")


class EnhancedChatResponse(BaseModel):
    """확장된 채팅 응답 (검증 정보 포함)"""
    brand_id: str
    message: str
    question_type: Optional[str] = None
    validation: Optional[ValidationInfo] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "brand_id": "brand_001",
                "message": "안녕하세요! 도움이 필요하신가요?",
                "question_type": "greeting",
                "validation": {
                    "grade": "A",
                    "score": 0.92,
                    "status": "passed",
                    "retries": 0,
                    "suggestions": []
                },
                "metadata": {
                    "request_id": "abc123",
                    "response_time_ms": 245.5
                }
            }
        }


# ============================================================
# API Endpoints
# ============================================================

@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    include_validation: bool = Query(
        default=False,
        description="응답에 검증 정보 포함 여부"
    ),
    session_id: Optional[str] = Query(
        default=None,
        description="세션 ID (없으면 자동 생성)"
    )
):
    """
    채팅 요청 처리

    Args:
        request: 채팅 요청
        include_validation: 검증 정보 포함 여부
        session_id: 세션 ID (선택, 채팅 저장에 사용)

    Returns:
        ChatResponse (또는 검증 정보 포함 시 EnhancedChatResponse)
    """
    try:
        # 엔진 가져오기
        engine = UniversalEngine.get_instance(request.brand_id)

        # 질문 처리 (Validation v2.0 포함)
        response = engine.ask(
            question=request.message,
            conversation_history=[msg.dict() for msg in request.conversation_history],
            use_cache=True
        )

        # 검증 정보 추출
        validation_info = response.metadata.get('validation', {}) if response.metadata else {}

        # 메타데이터 구성
        message_metadata = {
            'response_time_ms': response.metadata.get('response_time_ms') if response.metadata else None,
            'question_type': response.question_type,
            'grade': validation_info.get('grade'),
            'score': validation_info.get('score'),
            'handled_by': response.metadata.get('handled_by') if response.metadata else None,
        }

        # 채팅 저장 (백그라운드) - 모든 채팅 데이터 수집
        background_tasks.add_task(
            _save_chat_to_storage,
            session_id,
            request.brand_id,
            request.message,
            response.message,
            message_metadata
        )

        # 분석 추적 (백그라운드)
        background_tasks.add_task(
            _track_chat_event,
            request.brand_id,
            request.message,
            response
        )

        # 검증 정보 포함 요청 시
        if include_validation and response.metadata:
            validation_data = response.metadata.get('validation')
            if validation_data:
                enhanced_response = {
                    'brand_id': response.brand_id,
                    'message': response.message,
                    'question_type': response.question_type,
                    'validation': validation_data,
                    'metadata': {
                        k: v for k, v in response.metadata.items()
                        if k not in ['validation', 'improvement_suggestions']
                    }
                }

                # 개선 제안 추가
                suggestions = response.metadata.get('improvement_suggestions')
                if suggestions:
                    enhanced_response['validation']['suggestions'] = suggestions

                return JSONResponse(content=enhanced_response)

        return response

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Chat API error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    스트리밍 채팅

    Args:
        request: 채팅 요청

    Returns:
        StreamingResponse
    """
    try:
        engine = UniversalEngine.get_instance(request.brand_id)

        async def generate():
            for chunk in engine.ask_stream(
                question=request.message,
                conversation_history=[msg.dict() for msg in request.conversation_history]
            ):
                yield chunk

        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Chat stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate", response_model=Dict[str, Any])
async def validate_response(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """
    응답 생성 및 상세 검증 결과 반환

    이 엔드포인트는 디버깅 및 품질 모니터링 목적으로
    전체 검증 결과를 반환합니다.

    Args:
        request: 채팅 요청

    Returns:
        상세 검증 결과 포함 응답
    """
    try:
        engine = UniversalEngine.get_instance(request.brand_id)

        # 질문 처리
        response = engine.ask(
            question=request.message,
            conversation_history=[msg.dict() for msg in request.conversation_history],
            use_cache=False  # 캐시 미사용으로 항상 새로 생성
        )

        # 검증 정보 추출
        validation_info = response.metadata.get('validation', {})
        improvement_suggestions = response.metadata.get('improvement_suggestions', [])

        # 상세 결과 구성
        result = {
            'response': {
                'brand_id': response.brand_id,
                'message': response.message,
                'question_type': response.question_type,
            },
            'validation': {
                'grade': validation_info.get('grade', 'N/A'),
                'score': validation_info.get('score', 0.0),
                'status': validation_info.get('status', 'unknown'),
                'retries': validation_info.get('retries', 0),
                'issues': validation_info.get('issues', []),
            },
            'suggestions': improvement_suggestions,
            'metrics': {
                'request_id': response.metadata.get('request_id'),
                'response_time_ms': response.metadata.get('response_time_ms'),
                'validation_time_ms': response.metadata.get('validation_time_ms'),
                'routing_time_ms': response.metadata.get('routing_time_ms'),
                'retrieval_time_ms': response.metadata.get('retrieval_time_ms'),
                'generation_time_ms': response.metadata.get('generation_time_ms'),
            },
            'quality_breakdown': _extract_quality_breakdown(validation_info),
        }

        # 추적
        background_tasks.add_task(
            _track_validation_event,
            request.brand_id,
            request.message,
            result
        )

        return result

    except Exception as e:
        logger.error(f"Validate API error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/{brand_id}")
async def chat_health(brand_id: str):
    """
    특정 브랜드 엔진의 헬스 체크

    Args:
        brand_id: 브랜드 ID

    Returns:
        헬스 상태 정보
    """
    try:
        engine = UniversalEngine.get_instance(brand_id)
        health = engine.health_check()

        # 검증 관련 정보 추가
        health['validation_enabled'] = engine.engine_config.validation_enabled
        health['validation_min_grade'] = engine.engine_config.validation_min_grade

        return health

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Brand not found: {brand_id}")
    except Exception as e:
        return {
            'status': 'unhealthy',
            'brand_id': brand_id,
            'error': str(e)
        }


# ============================================================
# Helper Functions
# ============================================================

def _extract_quality_breakdown(validation_info: Dict[str, Any]) -> Dict[str, Any]:
    """검증 정보에서 품질 세부 분석 추출"""
    # 실제 구현에서는 ValidationResult의 filter_results에서 추출
    return {
        'trust': {
            'score': validation_info.get('trust_score', None),
            'hallucination_risk': validation_info.get('hallucination_risk', None),
        },
        'quality': {
            'score': validation_info.get('quality_score', None),
            'level': validation_info.get('quality_level', None),
        },
        'relevance': {
            'score': validation_info.get('relevance_score', None),
            'response_type': validation_info.get('response_type', None),
        },
    }


def _track_chat_event(brand_id: str, question: str, response: ChatResponse):
    """채팅 이벤트 추적 (백그라운드)"""
    try:
        analytics = get_analytics_service()

        # 검증 정보 추출
        validation_info = response.metadata.get('validation', {}) if response.metadata else {}

        analytics.track_event(
            'chat',
            brand_id,
            {
                'question': question[:100],
                'question_type': response.question_type,
                'response_length': len(response.message),
                'validation_grade': validation_info.get('grade'),
                'validation_score': validation_info.get('score'),
                'validation_retries': validation_info.get('retries', 0),
            }
        )
    except Exception as e:
        logger.error(f"Analytics tracking error: {e}")


def _track_validation_event(
    brand_id: str,
    question: str,
    result: Dict[str, Any]
):
    """검증 이벤트 추적 (백그라운드)"""
    try:
        analytics = get_analytics_service()
        analytics.track_event(
            'validation',
            brand_id,
            {
                'question': question[:100],
                'grade': result['validation']['grade'],
                'score': result['validation']['score'],
                'issues_count': len(result['validation'].get('issues', [])),
                'suggestions_count': len(result.get('suggestions', [])),
            }
        )
    except Exception as e:
        logger.error(f"Validation tracking error: {e}")


def _save_chat_to_storage(
    session_id: str,
    brand_id: str,
    user_message: str,
    assistant_message: str,
    metadata: Dict[str, Any]
):
    """채팅을 PostgreSQL에 저장 (백그라운드) - Neo4j 오염 방지"""
    try:
        storage = get_chat_storage()

        # 세션이 없으면 생성
        session = storage.get_or_create_session(session_id, brand_id)

        # 대화 턴 저장
        storage.save_conversation_turn(
            session_id=session.id,
            user_message=user_message,
            assistant_message=assistant_message,
            metadata=metadata
        )

        logger.debug(f"Chat saved to PostgreSQL session {session.id}")
    except Exception as e:
        logger.error(f"Chat storage error: {e}")


# ============================================================
# Session & History Endpoints
# ============================================================

class CreateSessionRequest(BaseModel):
    """세션 생성 요청"""
    brand_id: str = Field(..., description="브랜드 ID")
    user_id: Optional[str] = Field(None, description="사용자 ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="추가 메타데이터")


class SessionResponse(BaseModel):
    """세션 응답"""
    id: str
    brand_id: str
    user_id: Optional[str]
    created_at: str
    updated_at: str
    message_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MessageResponse(BaseModel):
    """메시지 응답"""
    id: str
    session_id: str
    role: str
    content: str
    timestamp: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatWithSessionRequest(BaseModel):
    """세션 포함 채팅 요청"""
    brand_id: str = Field(..., description="브랜드 ID")
    message: str = Field(..., description="사용자 메시지")
    session_id: Optional[str] = Field(None, description="세션 ID (없으면 자동 생성)")
    user_id: Optional[str] = Field(None, description="사용자 ID")
    use_history: bool = Field(True, description="이전 대화 컨텍스트 사용 여부")


class ChatWithSessionResponse(BaseModel):
    """세션 포함 채팅 응답"""
    brand_id: str
    message: str
    session_id: str
    question_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


@router.post("/session", response_model=SessionResponse)
async def create_session(request: CreateSessionRequest):
    """
    새 채팅 세션 생성

    Args:
        request: 세션 생성 요청

    Returns:
        생성된 세션 정보
    """
    try:
        storage = get_chat_storage()
        session = storage.create_session(
            brand_id=request.brand_id,
            user_id=request.user_id,
            metadata=request.metadata
        )

        return SessionResponse(
            id=session.id,
            brand_id=session.brand_id,
            user_id=session.user_id,
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
            message_count=session.message_count,
            metadata=session.metadata
        )

    except Exception as e:
        logger.error(f"Create session error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """
    세션 정보 조회

    Args:
        session_id: 세션 ID

    Returns:
        세션 정보
    """
    try:
        storage = get_chat_storage()
        session = storage.get_session(session_id)

        if not session:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        return SessionResponse(
            id=session.id,
            brand_id=session.brand_id,
            user_id=session.user_id,
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
            message_count=session.message_count,
            metadata=session.metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get session error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{brand_id}", response_model=List[SessionResponse])
async def list_sessions(
    brand_id: str,
    user_id: Optional[str] = Query(None, description="사용자 ID로 필터링"),
    limit: int = Query(50, ge=1, le=100, description="최대 개수"),
    offset: int = Query(0, ge=0, description="시작 위치")
):
    """
    브랜드의 세션 목록 조회

    Args:
        brand_id: 브랜드 ID
        user_id: 사용자 ID (선택)
        limit: 최대 개수
        offset: 시작 위치

    Returns:
        세션 목록
    """
    try:
        storage = get_chat_storage()
        sessions = storage.list_sessions(
            brand_id=brand_id,
            user_id=user_id,
            limit=limit,
            offset=offset
        )

        return [
            SessionResponse(
                id=s.id,
                brand_id=s.brand_id,
                user_id=s.user_id,
                created_at=s.created_at.isoformat(),
                updated_at=s.updated_at.isoformat(),
                message_count=s.message_count,
                metadata=s.metadata
            )
            for s in sessions
        ]

    except Exception as e:
        logger.error(f"List sessions error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}/messages", response_model=List[MessageResponse])
async def get_session_messages(
    session_id: str,
    limit: int = Query(100, ge=1, le=500, description="최대 개수")
):
    """
    세션의 메시지 목록 조회

    Args:
        session_id: 세션 ID
        limit: 최대 개수

    Returns:
        메시지 목록 (시간순)
    """
    try:
        storage = get_chat_storage()
        messages = storage.get_messages(session_id=session_id, limit=limit)

        return [
            MessageResponse(
                id=m.id,
                session_id=m.session_id,
                role=m.role,
                content=m.content,
                timestamp=m.timestamp.isoformat(),
                metadata=m.metadata
            )
            for m in messages
        ]

    except Exception as e:
        logger.error(f"Get messages error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/with-session", response_model=ChatWithSessionResponse)
async def chat_with_session(
    request: ChatWithSessionRequest,
    background_tasks: BackgroundTasks
):
    """
    세션 기반 채팅 (히스토리 자동 저장)

    세션 ID가 제공되지 않으면 자동으로 새 세션을 생성합니다.
    이전 대화 히스토리를 자동으로 불러와 컨텍스트로 사용합니다.

    Args:
        request: 세션 포함 채팅 요청

    Returns:
        세션 ID가 포함된 채팅 응답
    """
    try:
        storage = get_chat_storage()

        # 세션 조회 또는 생성
        session = storage.get_or_create_session(
            session_id=request.session_id,
            brand_id=request.brand_id,
            user_id=request.user_id
        )

        # 이전 대화 컨텍스트 불러오기
        conversation_history = []
        if request.use_history:
            history = storage.get_recent_context(session.id, max_messages=10)
            conversation_history = [
                {"role": h["role"], "content": h["content"]}
                for h in history
            ]

        # 엔진에서 응답 생성
        engine = UniversalEngine.get_instance(request.brand_id)
        response = engine.ask(
            question=request.message,
            conversation_history=conversation_history,
            use_cache=True
        )

        # 검증 정보 추출
        validation_info = response.metadata.get('validation', {}) if response.metadata else {}

        # 메타데이터 구성
        message_metadata = {
            'response_time_ms': response.metadata.get('response_time_ms'),
            'question_type': response.question_type,
            'grade': validation_info.get('grade'),
            'score': validation_info.get('score'),
            'handled_by': response.metadata.get('handled_by')
        }

        # 백그라운드에서 저장
        background_tasks.add_task(
            _save_chat_to_storage,
            session.id,
            request.brand_id,
            request.message,
            response.message,
            message_metadata
        )

        # 분석 추적
        background_tasks.add_task(
            _track_chat_event,
            request.brand_id,
            request.message,
            response
        )

        return ChatWithSessionResponse(
            brand_id=response.brand_id,
            message=response.message,
            session_id=session.id,
            question_type=response.question_type,
            metadata={
                **response.metadata,
                'session_id': session.id
            }
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Chat with session error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}/summary")
async def get_session_summary(session_id: str):
    """
    세션 요약 정보 조회

    Args:
        session_id: 세션 ID

    Returns:
        세션 요약 (메시지 수, 주요 질문, 등급 분포 등)
    """
    try:
        storage = get_chat_storage()
        summary = storage.get_session_summary(session_id)

        if not summary:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        return summary

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get session summary error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Analytics Endpoints
# ============================================================

@router.get("/analytics/{brand_id}")
async def get_chat_analytics(
    brand_id: str,
    days: int = Query(30, ge=1, le=365, description="분석 기간 (일)")
):
    """
    채팅 분석 데이터 조회

    Args:
        brand_id: 브랜드 ID
        days: 분석 기간

    Returns:
        채팅 분석 데이터 (세션 수, 메시지 수, 등급 분포, 일별 추이 등)
    """
    try:
        storage = get_chat_storage()
        analytics = storage.get_analytics(brand_id=brand_id, days=days)

        return analytics.to_dict()

    except Exception as e:
        logger.error(f"Get analytics error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
