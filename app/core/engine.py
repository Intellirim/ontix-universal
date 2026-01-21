"""
Universal Engine - Production Grade v2.0
전체 파이프라인 조율 및 브랜드별 처리

Features:
    - 브랜드별 싱글톤 인스턴스 관리
    - 멀티 스테이지 처리 파이프라인
    - 성능 메트릭 추적
    - 헬스 체크 및 상태 관리
    - Graceful 에러 핸들링
    - 요청 제한 (Rate Limiting)
    - 미들웨어 지원
"""

from typing import Dict, List, Optional, Generator, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import threading
import asyncio
import time
import logging
import json
import hashlib

from app.core.context import QueryContext, QuestionType, ProcessingStage
from app.core.routing import QuestionRouter
from app.core.pipeline import Pipeline
from app.services.platform.config_manager import ConfigManager
from app.features.registry import FeatureRegistry
from app.services.platform.analytics import get_analytics_service
from app.models.chat import ChatRequest, ChatResponse, RetrievalContext
from app.services.shared.cache import get_cache_client

# Production Grade v2.0 Filters
from app.filters import (
    ValidationFilter,
    ValidationConfig,
    ValidationResult,
    ValidationStatus,
    OverallGrade,
    FilterType,
    QualityConfig,
    TrustConfig,
    RelevanceConfig,
)

logger = logging.getLogger(__name__)


# ============================================================
# Feature Handler Manager
# ============================================================

class FeatureHandlerManager:
    """
    Feature Handler 관리자
    FeatureRegistry를 래핑하여 engine에서 사용
    """

    def __init__(self, brand_id: str, brand_config: Dict[str, Any]):
        self.brand_id = brand_id
        self.brand_config = brand_config
        self._handler_cache: Dict[str, Any] = {}

    def get_handler(self, feature_name: str):
        """
        Feature handler 인스턴스 반환

        Args:
            feature_name: 기능 이름 (question_type)

        Returns:
            FeatureHandler 인스턴스 또는 None
        """
        # 캐시에서 확인
        if feature_name in self._handler_cache:
            return self._handler_cache[feature_name]

        # FeatureRegistry에서 핸들러 가져오기
        handler = FeatureRegistry.get_handler(feature_name, self.brand_config)

        if handler:
            self._handler_cache[feature_name] = handler

        return handler

    def has_handler(self, feature_name: str) -> bool:
        """Feature handler 존재 여부"""
        return FeatureRegistry.has_feature(feature_name)

    def list_features(self) -> List[str]:
        """등록된 feature 목록"""
        return FeatureRegistry.list_features()


# ============================================================
# Enums
# ============================================================

class EngineState(str, Enum):
    """엔진 상태"""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    DEGRADED = "degraded"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


class ErrorType(str, Enum):
    """에러 유형"""
    VALIDATION = "validation"
    ROUTING = "routing"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    CACHE = "cache"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    INTERNAL = "internal"


# ============================================================
# Configuration
# ============================================================

@dataclass
class EngineConfig:
    """엔진 설정"""
    # 캐시 설정
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1시간

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100  # 분당 요청 수
    rate_limit_window: int = 60  # 초

    # 타임아웃
    request_timeout: float = 30.0  # 초
    retrieval_timeout: float = 10.0
    generation_timeout: float = 20.0

    # 재시도
    max_retries: int = 2
    retry_delay: float = 0.5

    # 스트리밍
    stream_chunk_size: int = 100

    # 디버그
    debug_mode: bool = False

    # === Validation Filter v2.0 설정 ===
    validation_enabled: bool = True
    validation_min_grade: str = "D"  # 최소 허용 등급 (A, B, C, D, F)
    validation_auto_retry: bool = True  # 낮은 등급시 자동 재생성
    validation_max_retries: int = 2  # 재생성 최대 시도 횟수
    validation_include_suggestions: bool = True  # 응답에 개선 제안 포함

    @classmethod
    def from_dict(cls, data: Dict) -> 'EngineConfig':
        """딕셔너리에서 생성"""
        return cls(
            cache_enabled=data.get('cache_enabled', True),
            cache_ttl=data.get('cache_ttl', 3600),
            rate_limit_enabled=data.get('rate_limit_enabled', True),
            rate_limit_requests=data.get('rate_limit_requests', 100),
            rate_limit_window=data.get('rate_limit_window', 60),
            request_timeout=data.get('request_timeout', 30.0),
            retrieval_timeout=data.get('retrieval_timeout', 10.0),
            generation_timeout=data.get('generation_timeout', 20.0),
            max_retries=data.get('max_retries', 2),
            retry_delay=data.get('retry_delay', 0.5),
            debug_mode=data.get('debug_mode', False),
            # Validation v2.0
            validation_enabled=data.get('validation_enabled', True),
            validation_min_grade=data.get('validation_min_grade', 'D'),
            validation_auto_retry=data.get('validation_auto_retry', True),
            validation_max_retries=data.get('validation_max_retries', 2),
            validation_include_suggestions=data.get('validation_include_suggestions', True),
        )


# ============================================================
# Metrics
# ============================================================

@dataclass
class RequestMetrics:
    """요청별 메트릭"""
    request_id: str
    start_time: float
    end_time: Optional[float] = None
    routing_time: float = 0.0
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    validation_time: float = 0.0
    cache_hit: bool = False
    error: Optional[str] = None
    question_type: Optional[str] = None
    retrieval_count: int = 0
    # Validation v2.0 메트릭
    validation_score: Optional[float] = None
    validation_grade: Optional[str] = None
    validation_retries: int = 0

    @property
    def total_time(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            'request_id': self.request_id,
            'total_time_ms': round(self.total_time * 1000, 2),
            'routing_time_ms': round(self.routing_time * 1000, 2),
            'retrieval_time_ms': round(self.retrieval_time * 1000, 2),
            'generation_time_ms': round(self.generation_time * 1000, 2),
            'validation_time_ms': round(self.validation_time * 1000, 2),
            'cache_hit': self.cache_hit,
            'question_type': self.question_type,
            'retrieval_count': self.retrieval_count,
            'validation_score': self.validation_score,
            'validation_grade': self.validation_grade,
            'validation_retries': self.validation_retries,
            'error': self.error,
        }


class EngineMetrics:
    """엔진 전체 메트릭"""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._requests: deque = deque(maxlen=max_history)
        self._lock = threading.Lock()

        # 카운터
        self.total_requests: int = 0
        self.successful_requests: int = 0
        self.failed_requests: int = 0
        self.cache_hits: int = 0
        self.rate_limited_requests: int = 0

        # 시작 시간
        self.start_time: float = time.time()

    def record_request(self, metrics: RequestMetrics):
        """요청 기록"""
        with self._lock:
            self._requests.append(metrics)
            self.total_requests += 1

            if metrics.error:
                self.failed_requests += 1
            else:
                self.successful_requests += 1

            if metrics.cache_hit:
                self.cache_hits += 1

    def record_rate_limit(self):
        """Rate limit 기록"""
        with self._lock:
            self.rate_limited_requests += 1

    def get_average_response_time(self, last_n: int = 100) -> float:
        """평균 응답 시간 (ms)"""
        with self._lock:
            recent = list(self._requests)[-last_n:]
            if not recent:
                return 0.0
            times = [r.total_time for r in recent if r.end_time]
            return round(sum(times) / len(times) * 1000, 2) if times else 0.0

    def get_error_rate(self, last_n: int = 100) -> float:
        """에러율 (%)"""
        with self._lock:
            recent = list(self._requests)[-last_n:]
            if not recent:
                return 0.0
            errors = sum(1 for r in recent if r.error)
            return round(errors / len(recent) * 100, 2)

    def get_cache_hit_rate(self) -> float:
        """캐시 히트율 (%)"""
        if self.total_requests == 0:
            return 0.0
        return round(self.cache_hits / self.total_requests * 100, 2)

    def get_requests_per_minute(self) -> float:
        """분당 요청 수"""
        elapsed_minutes = (time.time() - self.start_time) / 60
        if elapsed_minutes < 0.1:
            return 0.0
        return round(self.total_requests / elapsed_minutes, 2)

    def get_summary(self) -> Dict[str, Any]:
        """메트릭 요약"""
        return {
            'uptime_seconds': round(time.time() - self.start_time, 2),
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.get_cache_hit_rate(),
            'rate_limited_requests': self.rate_limited_requests,
            'avg_response_time_ms': self.get_average_response_time(),
            'error_rate': self.get_error_rate(),
            'requests_per_minute': self.get_requests_per_minute(),
        }


# ============================================================
# Rate Limiter
# ============================================================

class RateLimiter:
    """Rate Limiter (Token Bucket)"""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: deque = deque()
        self._lock = threading.Lock()

    def is_allowed(self) -> bool:
        """요청 허용 여부"""
        with self._lock:
            now = time.time()
            window_start = now - self.window_seconds

            # 윈도우 밖의 요청 제거
            while self._requests and self._requests[0] < window_start:
                self._requests.popleft()

            # 제한 확인
            if len(self._requests) >= self.max_requests:
                return False

            # 현재 요청 기록
            self._requests.append(now)
            return True

    def get_remaining(self) -> int:
        """남은 요청 수"""
        with self._lock:
            now = time.time()
            window_start = now - self.window_seconds

            # 윈도우 밖의 요청 제거
            while self._requests and self._requests[0] < window_start:
                self._requests.popleft()

            return max(0, self.max_requests - len(self._requests))

    def reset(self):
        """리셋"""
        with self._lock:
            self._requests.clear()


# ============================================================
# Engine Error
# ============================================================

class EngineError(Exception):
    """엔진 에러"""

    def __init__(
        self,
        message: str,
        error_type: ErrorType = ErrorType.INTERNAL,
        details: Dict = None
    ):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'message': self.message,
            'type': self.error_type.value,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
        }


# ============================================================
# Middleware
# ============================================================

class Middleware:
    """미들웨어 베이스 클래스"""

    def before_request(self, context: QueryContext) -> QueryContext:
        """요청 전처리"""
        return context

    def after_response(self, context: QueryContext, response: ChatResponse) -> ChatResponse:
        """응답 후처리"""
        return response

    def on_error(self, context: QueryContext, error: Exception) -> Optional[ChatResponse]:
        """에러 처리 (None 반환시 에러 전파)"""
        return None


class LoggingMiddleware(Middleware):
    """로깅 미들웨어"""

    def before_request(self, context: QueryContext) -> QueryContext:
        logger.info(f"[Request] brand={context.brand_id}, question={context.question[:50]}...")
        return context

    def after_response(self, context: QueryContext, response: ChatResponse) -> ChatResponse:
        logger.info(f"[Response] brand={context.brand_id}, status=success")
        return response

    def on_error(self, context: QueryContext, error: Exception) -> Optional[ChatResponse]:
        logger.error(f"[Error] brand={context.brand_id}, error={str(error)}")
        return None


class ValidationMiddleware(Middleware):
    """검증 미들웨어"""

    MAX_QUESTION_LENGTH = 2000
    MIN_QUESTION_LENGTH = 2

    def before_request(self, context: QueryContext) -> QueryContext:
        # 질문 길이 검증
        if len(context.question) < self.MIN_QUESTION_LENGTH:
            raise EngineError(
                "질문이 너무 짧습니다.",
                ErrorType.VALIDATION,
                {'min_length': self.MIN_QUESTION_LENGTH}
            )

        if len(context.question) > self.MAX_QUESTION_LENGTH:
            raise EngineError(
                "질문이 너무 깁니다.",
                ErrorType.VALIDATION,
                {'max_length': self.MAX_QUESTION_LENGTH}
            )

        return context


# ============================================================
# Universal Engine
# ============================================================

class UniversalEngine:
    """
    Universal Engine - Production Grade

    모든 브랜드에 대한 통합 처리 엔진

    Features:
        - 브랜드별 자동 설정 로드
        - 멀티 스테이지 질문 라우팅
        - RAG 파이프라인 실행
        - 성능 메트릭 추적
        - 캐싱 및 Rate Limiting
        - 미들웨어 체인
        - 헬스 체크
    """

    # 브랜드별 엔진 인스턴스 캐시
    _instances: Dict[str, 'UniversalEngine'] = {}
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, brand_id: str) -> 'UniversalEngine':
        """
        브랜드별 엔진 인스턴스 반환 (Thread-safe 싱글톤)

        Args:
            brand_id: 브랜드 ID

        Returns:
            엔진 인스턴스
        """
        with cls._lock:
            if brand_id not in cls._instances:
                cls._instances[brand_id] = cls(brand_id)
            return cls._instances[brand_id]

    @classmethod
    def clear_cache(cls, brand_id: str = None):
        """
        엔진 캐시 클리어

        Args:
            brand_id: 브랜드 ID (None이면 전체 클리어)
        """
        with cls._lock:
            if brand_id:
                if brand_id in cls._instances:
                    engine = cls._instances[brand_id]
                    engine._shutdown()
                    del cls._instances[brand_id]
                    logger.info(f"Cleared engine cache for: {brand_id}")
            else:
                for engine in cls._instances.values():
                    engine._shutdown()
                cls._instances.clear()
                logger.info("Cleared all engine caches")

    @classmethod
    def get_all_instances(cls) -> Dict[str, 'UniversalEngine']:
        """모든 엔진 인스턴스 반환"""
        with cls._lock:
            return dict(cls._instances)

    def __init__(self, brand_id: str):
        """
        Args:
            brand_id: 브랜드 ID
        """
        self._state = EngineState.INITIALIZING
        self.brand_id = brand_id

        # 설정 로드
        self.brand_config = ConfigManager.load_brand_config(brand_id)
        self.engine_config = EngineConfig.from_dict(
            self.brand_config.get('engine', {})
        )

        # 컴포넌트 초기화
        self.router = QuestionRouter(self.brand_config)
        self.pipeline = Pipeline(self.brand_config)
        self.feature_manager = FeatureHandlerManager(
            brand_id=brand_id,
            brand_config=self.brand_config
        )

        # 서비스
        self.cache = get_cache_client()
        self.analytics = get_analytics_service()

        # Rate Limiter
        self.rate_limiter = RateLimiter(
            max_requests=self.engine_config.rate_limit_requests,
            window_seconds=self.engine_config.rate_limit_window
        )

        # === Validation Filter v2.0 초기화 ===
        self.validation_filter = ValidationFilter(
            config=ValidationConfig(
                trust_config=TrustConfig(),
                quality_config=QualityConfig(language="ko"),
                relevance_config=RelevanceConfig(language="ko"),
            )
        )
        # 등급 매핑 (문자열 -> 열거형)
        self._grade_priority = {
            'A': 5, 'B': 4, 'C': 3, 'D': 2, 'F': 1
        }

        # 메트릭
        self.metrics = EngineMetrics()

        # 미들웨어
        self._middlewares: List[Middleware] = [
            LoggingMiddleware(),
            ValidationMiddleware(),
        ]

        # 상태 업데이트
        self._state = EngineState.READY
        self._initialized_at = datetime.now()

        logger.info(f"Engine initialized for brand: {brand_id} (Validation v2.0 enabled)")

    # === Public API ===

    def ask(
        self,
        question: str,
        conversation_history: List[Dict] = None,
        use_cache: bool = True,
        request_id: str = None
    ) -> ChatResponse:
        """
        질문 처리 (동기)

        Args:
            question: 사용자 질문
            conversation_history: 대화 히스토리
            use_cache: 캐시 사용 여부
            request_id: 요청 ID (없으면 자동 생성)

        Returns:
            ChatResponse
        """
        # 요청 ID 생성
        if not request_id:
            request_id = self._generate_request_id(question)

        # 메트릭 초기화
        request_metrics = RequestMetrics(
            request_id=request_id,
            start_time=time.time()
        )

        try:
            # 상태 확인
            self._check_engine_state()

            # Rate Limiting
            if self.engine_config.rate_limit_enabled:
                if not self.rate_limiter.is_allowed():
                    self.metrics.record_rate_limit()
                    raise EngineError(
                        "요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.",
                        ErrorType.RATE_LIMIT,
                        {'remaining': 0}
                    )

            # 캐시 확인
            use_cache = use_cache and self.engine_config.cache_enabled
            if use_cache:
                cached_response = self._get_cached_response(question)
                if cached_response:
                    request_metrics.cache_hit = True
                    request_metrics.end_time = time.time()
                    self.metrics.record_request(request_metrics)

                    self.analytics.track_event(
                        'cache_hit',
                        self.brand_id,
                        {'question': question[:100], 'request_id': request_id}
                    )
                    return cached_response

            # 컨텍스트 생성
            context = QueryContext(
                brand_id=self.brand_id,
                question=question,
                conversation_history=conversation_history or []
            )
            context.add_metadata('request_id', request_id)

            # 미들웨어 - before_request
            for middleware in self._middlewares:
                context = middleware.before_request(context)

            # 1. 질문 라우팅
            routing_start = time.time()
            question_type = self.router.route(context)
            request_metrics.routing_time = time.time() - routing_start
            request_metrics.question_type = question_type

            # 2. Feature Handler 확인
            feature_handler = self.feature_manager.get_handler(question_type)

            # Feature handler 결과 저장
            feature_result = None
            use_pipeline = True

            if feature_handler and feature_handler.can_handle(question, context.to_dict()):
                logger.info(f"Using feature handler: {question_type}")

                feature_result = feature_handler.process(question, context.to_dict())

                # Feature handler가 응답을 생성한 경우
                if feature_result.get('response'):
                    context.set_response(feature_result.get('response'))
                    context.add_metadata('handled_by', f'feature_{question_type}')
                    use_pipeline = False
                else:
                    # 응답이 None이면 파이프라인에 위임
                    logger.info(f"Feature handler {question_type} delegated to pipeline")
                    context.add_metadata('feature_preprocessed', question_type)

            # 파이프라인 실행 (feature handler가 없거나 응답을 위임한 경우)
            if use_pipeline:
                # 3. 기본 파이프라인 실행
                retrieval_start = time.time()
                context = self.pipeline._retrieve(context)
                request_metrics.retrieval_time = time.time() - retrieval_start
                request_metrics.retrieval_count = context.get_total_retrieval_count()

                generation_start = time.time()
                context = self.pipeline._generate(context)
                request_metrics.generation_time = time.time() - generation_start

                context.add_metadata('handled_by', 'pipeline')

            # Feature handler의 메타데이터/필터 결과 병합
            if feature_result:
                if feature_result.get('metadata'):
                    for key, value in feature_result['metadata'].items():
                        context.add_metadata(key, value)
                if feature_result.get('filter_results'):
                    context.add_metadata('filter_results', feature_result['filter_results'])

            # === 4. Validation Filter v2.0 적용 ===
            if self.engine_config.validation_enabled and context.response:
                validation_start = time.time()
                context, request_metrics = self._validate_and_retry(
                    context, request_metrics, question
                )
                request_metrics.validation_time = time.time() - validation_start

            # 응답 생성
            response = self._build_response(context, request_metrics)

            # 미들웨어 - after_response
            for middleware in reversed(self._middlewares):
                response = middleware.after_response(context, response)

            # 캐싱
            if use_cache and context.response:
                self._cache_response(question, response)

            # 메트릭 기록
            request_metrics.end_time = time.time()
            self.metrics.record_request(request_metrics)

            # 분석 추적
            self.analytics.track_event(
                'query',
                self.brand_id,
                {
                    'question': question[:100],
                    'question_type': question_type,
                    'request_id': request_id,
                    **request_metrics.to_dict()
                }
            )

            return response

        except EngineError as e:
            return self._handle_error(e, request_metrics, question)

        except Exception as e:
            engine_error = EngineError(str(e), ErrorType.INTERNAL)
            return self._handle_error(engine_error, request_metrics, question)

    def ask_stream(
        self,
        question: str,
        conversation_history: List[Dict] = None
    ) -> Generator[str, None, None]:
        """
        질문 처리 (스트리밍)

        Args:
            question: 사용자 질문
            conversation_history: 대화 히스토리

        Yields:
            응답 청크
        """
        try:
            # 상태 확인
            self._check_engine_state()

            # Rate Limiting
            if self.engine_config.rate_limit_enabled:
                if not self.rate_limiter.is_allowed():
                    yield "Error: Rate limit exceeded"
                    return

            # 컨텍스트 생성
            context = QueryContext(
                brand_id=self.brand_id,
                question=question,
                conversation_history=conversation_history or []
            )

            # 라우팅
            question_type = self.router.route(context)

            # Retrieval
            context = self.pipeline._retrieve(context)

            # 스트리밍 Generation
            from app.services.shared.llm import get_llm_client
            llm = get_llm_client()

            # 프롬프트 구성
            generation_config = self.brand_config.get('generation', {}).get(question_type, {})
            prompt_path = generation_config.get('prompt', 'conversational/default.txt')

            system_prompt = self.pipeline._load_prompt(prompt_path) or "You are a helpful assistant."

            # 컨텍스트 포맷팅
            context_str = self._format_retrieval_context(context)
            user_prompt = f"""Question: {question}

Context:
{context_str}

Answer the question based on the context above."""

            # 스트리밍
            for chunk in llm.stream(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model_variant="full"
            ):
                yield chunk

        except EngineError as e:
            yield f"Error: {e.message}"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"Error: {str(e)}"

    async def ask_async(
        self,
        question: str,
        conversation_history: List[Dict] = None,
        use_cache: bool = True
    ) -> ChatResponse:
        """
        질문 처리 (비동기)

        Args:
            question: 사용자 질문
            conversation_history: 대화 히스토리
            use_cache: 캐시 사용 여부

        Returns:
            ChatResponse
        """
        # 동기 함수를 스레드풀에서 실행
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.ask(question, conversation_history, use_cache)
        )

    # === Health & Status ===

    def health_check(self) -> Dict[str, Any]:
        """
        헬스 체크

        Returns:
            헬스 상태 정보
        """
        health = {
            'status': 'healthy',
            'brand_id': self.brand_id,
            'engine_state': self._state.value,
            'initialized_at': self._initialized_at.isoformat(),
            'uptime_seconds': round(time.time() - self.metrics.start_time, 2),
            'checks': {}
        }

        # 각 컴포넌트 체크
        try:
            # Router 체크
            health['checks']['router'] = 'ok' if self.router else 'missing'

            # Pipeline 체크
            health['checks']['pipeline'] = 'ok' if self.pipeline else 'missing'

            # Cache 체크
            try:
                self.cache.ping() if hasattr(self.cache, 'ping') else None
                health['checks']['cache'] = 'ok'
            except Exception:
                health['checks']['cache'] = 'degraded'

            # 메트릭 요약
            health['metrics'] = self.metrics.get_summary()

            # 에러율 확인
            error_rate = self.metrics.get_error_rate()
            if error_rate > 50:
                health['status'] = 'unhealthy'
            elif error_rate > 20:
                health['status'] = 'degraded'

        except Exception as e:
            health['status'] = 'unhealthy'
            health['error'] = str(e)

        return health

    def get_status(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return {
            'brand_id': self.brand_id,
            'state': self._state.value,
            'initialized_at': self._initialized_at.isoformat(),
            'rate_limit_remaining': self.rate_limiter.get_remaining(),
            'metrics': self.metrics.get_summary(),
        }

    # === Middleware Management ===

    def add_middleware(self, middleware: Middleware):
        """미들웨어 추가"""
        self._middlewares.append(middleware)
        logger.info(f"Added middleware: {middleware.__class__.__name__}")

    def remove_middleware(self, middleware_class: type):
        """미들웨어 제거"""
        self._middlewares = [
            m for m in self._middlewares
            if not isinstance(m, middleware_class)
        ]

    # === Configuration ===

    def reload_config(self):
        """설정 리로드"""
        logger.info(f"Reloading config for: {self.brand_id}")

        # 설정 리로드
        self.brand_config = ConfigManager.reload_config(self.brand_id)
        self.engine_config = EngineConfig.from_dict(
            self.brand_config.get('engine', {})
        )

        # 컴포넌트 재초기화
        self.router = QuestionRouter(self.brand_config)
        self.pipeline = Pipeline(self.brand_config)
        self.feature_manager = FeatureHandlerManager(
            brand_id=self.brand_id,
            brand_config=self.brand_config
        )

        # Rate Limiter 업데이트
        self.rate_limiter = RateLimiter(
            max_requests=self.engine_config.rate_limit_requests,
            window_seconds=self.engine_config.rate_limit_window
        )

        logger.info(f"Config reloaded for: {self.brand_id}")

    def get_debug_info(self) -> Dict[str, Any]:
        """디버그 정보"""
        if not self.engine_config.debug_mode:
            return {'debug_mode': False}

        return {
            'debug_mode': True,
            'brand_id': self.brand_id,
            'state': self._state.value,
            'config': {
                'cache_enabled': self.engine_config.cache_enabled,
                'rate_limit_enabled': self.engine_config.rate_limit_enabled,
                'rate_limit_requests': self.engine_config.rate_limit_requests,
                'request_timeout': self.engine_config.request_timeout,
            },
            'router_info': self.router.get_debug_info() if hasattr(self.router, 'get_debug_info') else {},
            'metrics': self.metrics.get_summary(),
            'middlewares': [m.__class__.__name__ for m in self._middlewares],
        }

    # === Private Methods ===

    def _check_engine_state(self):
        """엔진 상태 확인"""
        if self._state == EngineState.STOPPED:
            raise EngineError(
                "엔진이 중지되었습니다.",
                ErrorType.INTERNAL
            )

        if self._state == EngineState.SHUTTING_DOWN:
            raise EngineError(
                "엔진이 종료 중입니다.",
                ErrorType.INTERNAL
            )

    def _generate_request_id(self, question: str) -> str:
        """요청 ID 생성"""
        data = f"{self.brand_id}:{question}:{time.time()}"
        return hashlib.md5(data.encode()).hexdigest()[:12]

    def _build_response(
        self,
        context: QueryContext,
        metrics: RequestMetrics
    ) -> ChatResponse:
        """ChatResponse 객체 생성"""
        # Retrieval 컨텍스트 변환
        retrieval_contexts = []

        for result in context.retrieval_results:
            retrieval_contexts.append(
                RetrievalContext(
                    source=result.source,
                    data=result.data,
                    metadata=result.metadata
                )
            )

        # 메타데이터
        metadata = {
            'request_id': metrics.request_id,
            'response_time_ms': round(metrics.total_time * 1000, 2),
            'routing_time_ms': round(metrics.routing_time * 1000, 2),
            'retrieval_time_ms': round(metrics.retrieval_time * 1000, 2),
            'generation_time_ms': round(metrics.generation_time * 1000, 2),
            'retrieval_count': metrics.retrieval_count,
            **context.metadata
        }

        return ChatResponse(
            brand_id=self.brand_id,
            message=context.response or "",
            question_type=context.question_type.value if hasattr(context.question_type, 'value') else str(context.question_type),
            retrieval_contexts=retrieval_contexts,
            metadata=metadata
        )

    def _format_retrieval_context(self, context: QueryContext) -> str:
        """검색 컨텍스트 포맷팅"""
        if not context.retrieval_results:
            return "No relevant information found."

        formatted_parts = []

        for result in context.retrieval_results:
            if not result.data:
                continue

            formatted_parts.append(f"\n[{result.source.upper()}]")

            for i, item in enumerate(result.data[:5], 1):
                formatted_parts.append(f"{i}. {json.dumps(item, ensure_ascii=False)}")

        return "\n".join(formatted_parts)

    def _get_cached_response(self, question: str) -> Optional[ChatResponse]:
        """캐시된 응답 조회"""
        try:
            cache_key = f"{self.brand_id}:question:{hashlib.md5(question.encode()).hexdigest()[:16]}"
            cached = self.cache.get(cache_key)

            if cached:
                return ChatResponse.model_validate_json(cached)

            return None

        except Exception as e:
            logger.debug(f"Cache get error: {e}")
            return None

    def _cache_response(self, question: str, response: ChatResponse):
        """응답 캐싱"""
        try:
            cache_key = f"{self.brand_id}:question:{hashlib.md5(question.encode()).hexdigest()[:16]}"
            self.cache.set(
                cache_key,
                response.model_dump_json(),
                ttl=self.engine_config.cache_ttl
            )
        except Exception as e:
            logger.debug(f"Cache set error: {e}")

    def _handle_error(
        self,
        error: EngineError,
        metrics: RequestMetrics,
        question: str
    ) -> ChatResponse:
        """에러 처리"""
        metrics.error = error.message
        metrics.end_time = time.time()
        self.metrics.record_request(metrics)

        # 미들웨어 에러 처리
        context = QueryContext(brand_id=self.brand_id, question=question)
        for middleware in self._middlewares:
            result = middleware.on_error(context, error)
            if result:
                return result

        # 에러 추적
        self.analytics.track_event(
            'error',
            self.brand_id,
            {
                'question': question[:100],
                'error_type': error.error_type.value,
                'error_message': error.message,
                **error.details
            }
        )

        # 사용자 친화적 메시지
        user_messages = {
            ErrorType.VALIDATION: error.message,
            ErrorType.RATE_LIMIT: "요청이 너무 많습니다. 잠시 후 다시 시도해주세요.",
            ErrorType.TIMEOUT: "요청 처리 시간이 초과되었습니다.",
            ErrorType.ROUTING: "질문을 분석하는 중 문제가 발생했습니다.",
            ErrorType.RETRIEVAL: "정보를 검색하는 중 문제가 발생했습니다.",
            ErrorType.GENERATION: "응답을 생성하는 중 문제가 발생했습니다.",
        }

        message = user_messages.get(
            error.error_type,
            "죄송합니다. 처리 중 오류가 발생했습니다."
        )

        return ChatResponse(
            brand_id=self.brand_id,
            message=message,
            metadata={
                'error': True,
                'error_type': error.error_type.value,
                'request_id': metrics.request_id,
            }
        )

    # === Validation v2.0 Methods ===

    def _validate_and_retry(
        self,
        context: QueryContext,
        metrics: RequestMetrics,
        question: str
    ) -> tuple:
        """
        응답 검증 및 필요시 재생성

        Args:
            context: 현재 컨텍스트
            metrics: 요청 메트릭
            question: 원본 질문

        Returns:
            (updated_context, updated_metrics)
        """
        min_grade_priority = self._grade_priority.get(
            self.engine_config.validation_min_grade.upper(), 2
        )
        max_retries = self.engine_config.validation_max_retries
        retry_count = 0

        while retry_count <= max_retries:
            # 검증 컨텍스트 준비
            validation_context = {
                'question': question,
                'brand_id': self.brand_id,
                'retrieval_results': {
                    r.source: r.data for r in context.retrieval_results
                } if context.retrieval_results else {},
            }

            # 검증 실행
            validation_result: ValidationResult = self.validation_filter.validate(
                response=context.response or "",
                context=validation_context
            )

            # 메트릭 업데이트
            metrics.validation_score = validation_result.score
            metrics.validation_grade = validation_result.grade.value
            metrics.validation_retries = retry_count

            # 등급 확인
            current_grade_priority = self._grade_priority.get(
                validation_result.grade.value, 1
            )

            # 로깅
            logger.info(
                f"[Validation] Grade={validation_result.grade.value}, "
                f"Score={validation_result.score:.2f}, "
                f"Retry={retry_count}/{max_retries}"
            )

            # 통과 조건 확인
            if current_grade_priority >= min_grade_priority:
                # 검증 통과 - 메타데이터 추가
                context.add_metadata('validation', {
                    'grade': validation_result.grade.value,
                    'score': validation_result.score,
                    'status': validation_result.status.value,
                    'retries': retry_count,
                })

                # 개선 제안 추가 (옵션)
                if self.engine_config.validation_include_suggestions and validation_result.suggestions:
                    context.add_metadata('improvement_suggestions', validation_result.suggestions)

                # 경고가 있으면 로그에 기록
                if validation_result.all_warnings:
                    logger.warning(
                        f"[Validation Warnings] {', '.join(validation_result.all_warnings[:3])}"
                    )

                return context, metrics

            # 재시도 가능 여부 확인
            if not self.engine_config.validation_auto_retry or retry_count >= max_retries:
                break

            # 재생성 시도
            logger.warning(
                f"[Validation] Low grade ({validation_result.grade.value}), "
                f"attempting regeneration ({retry_count + 1}/{max_retries})"
            )

            # 이슈를 프롬프트 힌트로 추가
            issues_hint = "; ".join(validation_result.all_issues[:3]) if validation_result.all_issues else ""
            context.add_metadata('regeneration_hint', issues_hint)

            # 재생성
            try:
                context = self.pipeline._generate(context)
                retry_count += 1
            except Exception as e:
                logger.error(f"[Validation] Regeneration failed: {e}")
                break

        # 최소 등급 미달 - 사과 메시지 추가
        if current_grade_priority < min_grade_priority:
            context = self._append_apology_message(context, validation_result)
            context.add_metadata('validation', {
                'grade': validation_result.grade.value,
                'score': validation_result.score,
                'status': 'below_threshold',
                'retries': retry_count,
                'issues': validation_result.all_issues[:5],
            })

            logger.warning(
                f"[Validation] Response below minimum grade after {retry_count} retries. "
                f"Final grade: {validation_result.grade.value}"
            )

        return context, metrics

    def _append_apology_message(
        self,
        context: QueryContext,
        validation_result: ValidationResult
    ) -> QueryContext:
        """낮은 품질 응답에 사과 메시지 추가"""
        apology_messages = {
            'F': "\n\n---\n⚠️ 죄송합니다. 충분히 정확한 답변을 드리지 못했을 수 있습니다. 추가 정보가 필요하시면 다시 질문해 주세요.",
            'D': "\n\n---\n💡 참고: 이 답변은 제한된 정보를 바탕으로 작성되었습니다. 더 정확한 정보가 필요하시면 구체적으로 질문해 주세요.",
        }

        grade = validation_result.grade.value
        if grade in apology_messages:
            current_response = context.response or ""
            context.set_response(current_response + apology_messages[grade])

        return context

    def _is_grade_acceptable(self, grade: str) -> bool:
        """등급이 허용 가능한지 확인"""
        min_priority = self._grade_priority.get(
            self.engine_config.validation_min_grade.upper(), 2
        )
        current_priority = self._grade_priority.get(grade.upper(), 1)
        return current_priority >= min_priority

    def _shutdown(self):
        """엔진 종료"""
        self._state = EngineState.SHUTTING_DOWN
        logger.info(f"Shutting down engine for: {self.brand_id}")

        # 정리 작업
        self.rate_limiter.reset()

        self._state = EngineState.STOPPED

    def __repr__(self) -> str:
        return (
            f"UniversalEngine(brand={self.brand_id}, "
            f"state={self._state.value}, "
            f"requests={self.metrics.total_requests})"
        )
