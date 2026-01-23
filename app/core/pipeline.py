"""
Pipeline - Production Grade v2.0
RAG 파이프라인 (Retrieval -> Generation)

Features:
    - 멀티 스테이지 파이프라인
    - 병렬/순차 Retrieval 지원
    - 프롬프트 캐싱
    - 후처리 훅
    - 폴백 전략
    - 성능 추적
"""

from typing import Dict, List, Any, Optional, Callable, TYPE_CHECKING, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import threading
import time
import logging

from app.core.context import QueryContext, QuestionType, ProcessingStage

# Type hints without import to avoid circular import
if TYPE_CHECKING:
    from app.retrievers.graph import GraphRetriever
    from app.retrievers.vector import VectorRetriever
    from app.retrievers.stats import StatsRetriever
    from app.retrievers.product import ProductRetriever

logger = logging.getLogger(__name__)


# ============================================================
# Enums
# ============================================================

class PipelineStage(str, Enum):
    """파이프라인 단계"""
    INIT = "init"
    PRE_PROCESS = "pre_process"
    RETRIEVAL = "retrieval"
    POST_RETRIEVAL = "post_retrieval"
    GENERATION = "generation"
    POST_PROCESS = "post_process"
    COMPLETED = "completed"
    ERROR = "error"


class RetrievalMode(str, Enum):
    """Retrieval 실행 모드"""
    SEQUENTIAL = "sequential"  # 순차 실행
    PARALLEL = "parallel"  # 병렬 실행


class FallbackStrategy(str, Enum):
    """폴백 전략"""
    NONE = "none"  # 폴백 없음
    DEFAULT_RESPONSE = "default_response"  # 기본 응답
    PARTIAL_RESULT = "partial_result"  # 부분 결과 사용
    RETRY = "retry"  # 재시도


# ============================================================
# Configuration
# ============================================================

@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    # Retrieval 설정
    retrieval_mode: RetrievalMode = RetrievalMode.SEQUENTIAL
    max_parallel_retrievers: int = 4
    retrieval_timeout: float = 10.0

    # Generation 설정
    generation_timeout: float = 30.0
    max_context_length: int = 8000

    # 폴백 설정
    fallback_strategy: FallbackStrategy = FallbackStrategy.DEFAULT_RESPONSE
    default_error_message: str = "죄송합니다. 응답 생성 중 오류가 발생했습니다."

    # 캐싱
    cache_prompts: bool = True
    prompt_cache_size: int = 50

    # 디버그
    debug_mode: bool = False
    trace_enabled: bool = False

    @classmethod
    def from_dict(cls, data: Dict) -> 'PipelineConfig':
        """딕셔너리에서 생성"""
        return cls(
            retrieval_mode=RetrievalMode(data.get('retrieval_mode', 'sequential')),
            max_parallel_retrievers=data.get('max_parallel_retrievers', 4),
            retrieval_timeout=data.get('retrieval_timeout', 10.0),
            generation_timeout=data.get('generation_timeout', 30.0),
            max_context_length=data.get('max_context_length', 8000),
            fallback_strategy=FallbackStrategy(data.get('fallback_strategy', 'default_response')),
            default_error_message=data.get('default_error_message', "죄송합니다. 응답 생성 중 오류가 발생했습니다."),
            cache_prompts=data.get('cache_prompts', True),
            prompt_cache_size=data.get('prompt_cache_size', 50),
            debug_mode=data.get('debug_mode', False),
            trace_enabled=data.get('trace_enabled', False),
        )


# ============================================================
# Pipeline Metrics
# ============================================================

@dataclass
class StepMetrics:
    """단계별 메트릭"""
    step_name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000

    def complete(self, success: bool = True, error: str = None):
        """단계 완료"""
        self.end_time = time.time()
        self.success = success
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_name': self.step_name,
            'duration_ms': round(self.duration_ms, 2),
            'success': self.success,
            'error': self.error,
            'metadata': self.metadata,
        }


@dataclass
class PipelineTrace:
    """파이프라인 실행 추적"""
    pipeline_id: str
    brand_id: str
    question: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    stages: List[StepMetrics] = field(default_factory=list)
    current_stage: PipelineStage = PipelineStage.INIT

    def add_stage(self, stage_name: str) -> StepMetrics:
        """단계 추가"""
        metrics = StepMetrics(step_name=stage_name)
        self.stages.append(metrics)
        return metrics

    def get_current_stage_metrics(self) -> Optional[StepMetrics]:
        """현재 단계 메트릭"""
        if self.stages:
            return self.stages[-1]
        return None

    @property
    def total_duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pipeline_id': self.pipeline_id,
            'brand_id': self.brand_id,
            'question': self.question[:100],
            'total_duration_ms': round(self.total_duration_ms, 2),
            'current_stage': self.current_stage.value,
            'stages': [s.to_dict() for s in self.stages],
        }


# ============================================================
# Prompt Manager
# ============================================================

class PromptManager:
    """프롬프트 관리자"""

    def __init__(self, base_path: str = "prompts", cache_size: int = 50):
        self.base_path = Path(base_path)
        self.cache_size = cache_size
        self._cache: Dict[str, str] = {}
        self._cache_order: List[str] = []
        self._lock = threading.Lock()

    def load(self, prompt_path: str, use_cache: bool = True) -> Optional[str]:
        """
        프롬프트 로드

        Args:
            prompt_path: 프롬프트 파일 경로
            use_cache: 캐시 사용 여부

        Returns:
            프롬프트 텍스트
        """
        if use_cache:
            with self._lock:
                if prompt_path in self._cache:
                    return self._cache[prompt_path]

        try:
            path = self.base_path / prompt_path

            if not path.exists():
                logger.warning(f"Prompt file not found: {prompt_path}")
                return None

            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 캐시 저장
            if use_cache:
                self._add_to_cache(prompt_path, content)

            return content

        except Exception as e:
            logger.error(f"Failed to load prompt: {e}")
            return None

    def _add_to_cache(self, key: str, value: str):
        """캐시에 추가 (LRU)"""
        with self._lock:
            if key in self._cache:
                self._cache_order.remove(key)
            elif len(self._cache) >= self.cache_size:
                oldest = self._cache_order.pop(0)
                del self._cache[oldest]

            self._cache[key] = value
            self._cache_order.append(key)

    def clear_cache(self):
        """캐시 클리어"""
        with self._lock:
            self._cache.clear()
            self._cache_order.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계"""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.cache_size,
                'keys': list(self._cache.keys()),
            }


# ============================================================
# Hook System
# ============================================================

class PipelineHook:
    """파이프라인 훅 베이스 클래스"""

    def pre_retrieval(self, context: QueryContext) -> QueryContext:
        """Retrieval 전 처리"""
        return context

    def post_retrieval(self, context: QueryContext) -> QueryContext:
        """Retrieval 후 처리"""
        return context

    def pre_generation(self, context: QueryContext) -> QueryContext:
        """Generation 전 처리"""
        return context

    def post_generation(self, context: QueryContext) -> QueryContext:
        """Generation 후 처리"""
        return context


class ContentFilterHook(PipelineHook):
    """콘텐츠 필터링 훅"""

    def post_retrieval(self, context: QueryContext) -> QueryContext:
        """검색 결과 필터링"""
        # 예: 중복 제거, 관련성 필터링 등
        return context

    def post_generation(self, context: QueryContext) -> QueryContext:
        """응답 필터링"""
        # 예: 민감 정보 필터링, 포맷팅 등
        return context


class ContextTruncationHook(PipelineHook):
    """컨텍스트 길이 제한 훅"""

    def __init__(self, max_length: int = 8000):
        self.max_length = max_length

    def pre_generation(self, context: QueryContext) -> QueryContext:
        """컨텍스트 길이 제한"""
        # 검색 결과가 너무 많으면 truncate
        total_items = context.get_total_retrieval_count()
        if total_items > 20:
            logger.info(f"Truncating retrieval results from {total_items} to 20")
            for result in context.retrieval_results:
                result.data = result.data[:5]
        return context


# ============================================================
# Pipeline
# ============================================================

class Pipeline:
    """
    RAG 파이프라인 - Production Grade

    Retrieval -> Generation

    Features:
        - 병렬/순차 Retrieval
        - 프롬프트 캐싱
        - 훅 시스템
        - 폴백 전략
        - 성능 추적
    """

    def __init__(self, brand_config: Dict):
        """
        Args:
            brand_config: 브랜드 설정
        """
        self.brand_config = brand_config
        self.brand_id = brand_config.get('brand', {}).get('id', 'unknown')

        # 파이프라인 설정
        self.config = PipelineConfig.from_dict(
            brand_config.get('pipeline', {})
        )

        # Lazy imports to avoid circular dependency
        from app.retrievers.graph import GraphRetriever
        from app.retrievers.vector import VectorRetriever
        from app.retrievers.stats import StatsRetriever
        from app.retrievers.product import ProductRetriever
        from app.generators.factual import FactualGenerator
        from app.generators.insight import InsightGenerator
        from app.generators.conversational import ConversationalGenerator

        # Retrievers 초기화
        self.retrievers = {
            'graph': GraphRetriever(brand_config),
            'vector': VectorRetriever(brand_config),
            'stats': StatsRetriever(brand_config),
            'product': ProductRetriever(brand_config),
        }

        # Generators 초기화
        self.generators = {
            'factual': FactualGenerator(brand_config),
            'insight': InsightGenerator(brand_config),
            'conversational': ConversationalGenerator(brand_config),
        }

        # 프롬프트 매니저
        self.prompt_manager = PromptManager(
            cache_size=self.config.prompt_cache_size
        )

        # 훅
        self._hooks: List[PipelineHook] = [
            ContextTruncationHook(max_length=self.config.max_context_length),
        ]

        # 스레드풀 (병렬 retrieval용)
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.max_parallel_retrievers
        )

        # 추적
        self._current_trace: Optional[PipelineTrace] = None

        logger.info(f"Pipeline initialized for brand: {self.brand_id}")

    # === Public API ===

    def execute(self, context: QueryContext) -> QueryContext:
        """
        파이프라인 실행

        Args:
            context: 쿼리 컨텍스트

        Returns:
            업데이트된 컨텍스트
        """
        # 추적 시작
        if self.config.trace_enabled:
            self._start_trace(context)

        try:
            # 1. Pre-process
            context = self._pre_process(context)

            # 2. Retrieval
            context = self._retrieve(context)

            # 3. Post-retrieval
            context = self._post_retrieval(context)

            # 4. Pre-generation
            context = self._pre_generation(context)

            # 5. Generation
            context = self._generate(context)

            # 6. Post-process
            context = self._post_process(context)

            # 추적 완료
            if self._current_trace:
                self._current_trace.end_time = time.time()
                self._current_trace.current_stage = PipelineStage.COMPLETED

            return context

        except Exception as e:
            logger.error(f"Pipeline error: {e}")

            # 추적에 에러 기록
            if self._current_trace:
                current = self._current_trace.get_current_stage_metrics()
                if current:
                    current.complete(success=False, error=str(e))
                self._current_trace.current_stage = PipelineStage.ERROR

            # 폴백 처리
            return self._handle_fallback(context, e)

    def _retrieve(self, context: QueryContext) -> QueryContext:
        """
        Retrieval 단계

        Args:
            context: 쿼리 컨텍스트

        Returns:
            업데이트된 컨텍스트
        """
        if self._current_trace:
            stage = self._current_trace.add_stage("retrieval")
            self._current_trace.current_stage = PipelineStage.RETRIEVAL

        question_type = context.question_type
        if hasattr(question_type, 'value'):
            question_type_str = question_type.value
        else:
            question_type_str = str(question_type)

        # 질문 타입별 retrieval 설정
        retrieval_config = self.brand_config.get('retrieval', {})
        qtype_config = retrieval_config.get(question_type_str, {})

        # 사용할 retriever 목록
        retrievers_to_use = qtype_config.get('retrievers', ['graph'])

        logger.info(f"Retrieving with: {retrievers_to_use}")

        # 병렬 또는 순차 실행
        if self.config.retrieval_mode == RetrievalMode.PARALLEL:
            context = self._retrieve_parallel(context, retrievers_to_use)
        else:
            context = self._retrieve_sequential(context, retrievers_to_use)

        # 메트릭 업데이트
        total_results = context.get_total_retrieval_count()
        context.mark_retrieval_complete()

        if self._current_trace:
            stage.complete()
            stage.metadata['retrieval_count'] = total_results
            stage.metadata['retrievers'] = retrievers_to_use

        logger.info(f"Retrieved {total_results} total results")

        return context

    def _retrieve_sequential(
        self,
        context: QueryContext,
        retrievers: List[str]
    ) -> QueryContext:
        """순차 Retrieval"""
        for retriever_name in retrievers:
            retriever = self.retrievers.get(retriever_name)

            if not retriever:
                logger.warning(f"Unknown retriever: {retriever_name}")
                continue

            try:
                retriever.retrieve(context)
            except Exception as e:
                logger.error(f"Retriever error ({retriever_name}): {e}")

        return context

    def _retrieve_parallel(
        self,
        context: QueryContext,
        retrievers: List[str]
    ) -> QueryContext:
        """병렬 Retrieval"""
        futures = {}

        for retriever_name in retrievers:
            retriever = self.retrievers.get(retriever_name)

            if not retriever:
                logger.warning(f"Unknown retriever: {retriever_name}")
                continue

            future = self._executor.submit(
                self._safe_retrieve,
                retriever,
                context,
                retriever_name
            )
            futures[future] = retriever_name

        # 결과 수집
        for future in as_completed(futures, timeout=self.config.retrieval_timeout):
            retriever_name = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Parallel retriever error ({retriever_name}): {e}")

        return context

    def _safe_retrieve(
        self,
        retriever,
        context: QueryContext,
        name: str
    ):
        """안전한 Retrieval (에러 격리)"""
        try:
            retriever.retrieve(context)
        except Exception as e:
            logger.error(f"Retriever error ({name}): {e}")

    def _generate(self, context: QueryContext) -> QueryContext:
        """
        Generation 단계

        Args:
            context: 쿼리 컨텍스트

        Returns:
            업데이트된 컨텍스트
        """
        if self._current_trace:
            stage = self._current_trace.add_stage("generation")
            self._current_trace.current_stage = PipelineStage.GENERATION

        question_type = context.question_type
        if hasattr(question_type, 'value'):
            question_type_str = question_type.value
        else:
            question_type_str = str(question_type)

        # 질문 타입별 generation 설정
        generation_config = self.brand_config.get('generation', {})
        qtype_config = generation_config.get(question_type_str, {})

        # Generator 선택
        generator_type = qtype_config.get('type', 'conversational')
        generator = self.generators.get(generator_type)

        if not generator:
            logger.warning(f"Unknown generator: {generator_type}, using conversational")
            generator = self.generators['conversational']

        # AI Advisor와 Content Generation은 gpt-5-mini (feature variant) 사용
        feature_question_types = ['advisor', 'content_generation']
        if question_type_str in feature_question_types:
            model_variant = 'feature'  # gpt-5-mini
            logger.info(f"Using feature model (gpt-5-mini) for {question_type_str}")
        else:
            model_variant = qtype_config.get('model_variant', 'mini')  # 기본값 gpt-4o-mini

        # Generator에 model_variant 설정
        generator.configure(model_variant=model_variant)

        logger.info(f"Generating with: {generator_type} (model_variant: {model_variant})")

        try:
            response = generator.generate(context)
            context.set_response(response)

            if self._current_trace:
                stage.complete()
                stage.metadata['generator'] = generator_type
                stage.metadata['response_length'] = len(response) if response else 0

        except Exception as e:
            logger.error(f"Generator error: {e}")

            if self._current_trace:
                stage.complete(success=False, error=str(e))

            # 폴백 응답
            context.set_response(self.config.default_error_message)

        return context

    # === Hook Methods ===

    def _pre_process(self, context: QueryContext) -> QueryContext:
        """전처리"""
        if self._current_trace:
            stage = self._current_trace.add_stage("pre_process")
            self._current_trace.current_stage = PipelineStage.PRE_PROCESS

        # 훅 호출 - pre_retrieval
        for hook in self._hooks:
            context = hook.pre_retrieval(context)

        if self._current_trace:
            stage.complete()

        return context

    def _post_retrieval(self, context: QueryContext) -> QueryContext:
        """Retrieval 후처리"""
        if self._current_trace:
            stage = self._current_trace.add_stage("post_retrieval")
            self._current_trace.current_stage = PipelineStage.POST_RETRIEVAL

        # 훅 호출 - post_retrieval
        for hook in self._hooks:
            context = hook.post_retrieval(context)

        if self._current_trace:
            stage.complete()

        return context

    def _pre_generation(self, context: QueryContext) -> QueryContext:
        """Generation 전처리"""
        # 훅 호출 - pre_generation
        for hook in self._hooks:
            context = hook.pre_generation(context)

        return context

    def _post_process(self, context: QueryContext) -> QueryContext:
        """후처리"""
        if self._current_trace:
            stage = self._current_trace.add_stage("post_process")
            self._current_trace.current_stage = PipelineStage.POST_PROCESS

        # 훅 호출 - post_generation
        for hook in self._hooks:
            context = hook.post_generation(context)

        if self._current_trace:
            stage.complete()

        return context

    # === Fallback ===

    def _handle_fallback(
        self,
        context: QueryContext,
        error: Exception
    ) -> QueryContext:
        """폴백 처리"""
        strategy = self.config.fallback_strategy

        if strategy == FallbackStrategy.DEFAULT_RESPONSE:
            context.set_response(self.config.default_error_message)
            context.set_error(str(error))

        elif strategy == FallbackStrategy.PARTIAL_RESULT:
            # 이미 있는 검색 결과로 간단한 응답 생성
            if context.get_total_retrieval_count() > 0:
                context.set_response(
                    "검색 결과를 찾았으나 전체 응답 생성에 실패했습니다."
                )
            else:
                context.set_response(self.config.default_error_message)
            context.set_error(str(error))

        elif strategy == FallbackStrategy.RETRY:
            # 재시도 로직 (간단한 구현)
            try:
                context = self._retrieve(context)
                context = self._generate(context)
            except Exception as retry_error:
                context.set_response(self.config.default_error_message)
                context.set_error(f"Retry failed: {retry_error}")

        else:  # NONE
            context.set_error(str(error))
            raise error

        return context

    # === Tracing ===

    def _start_trace(self, context: QueryContext):
        """추적 시작"""
        import hashlib
        trace_id = hashlib.md5(
            f"{self.brand_id}:{context.question}:{time.time()}".encode()
        ).hexdigest()[:12]

        self._current_trace = PipelineTrace(
            pipeline_id=trace_id,
            brand_id=self.brand_id,
            question=context.question
        )

    def get_trace(self) -> Optional[Dict[str, Any]]:
        """현재 추적 정보"""
        if self._current_trace:
            return self._current_trace.to_dict()
        return None

    # === Hooks Management ===

    def add_hook(self, hook: PipelineHook):
        """훅 추가"""
        self._hooks.append(hook)
        logger.info(f"Added hook: {hook.__class__.__name__}")

    def remove_hook(self, hook_class: type):
        """훅 제거"""
        self._hooks = [h for h in self._hooks if not isinstance(h, hook_class)]

    def get_hooks(self) -> List[str]:
        """등록된 훅 목록"""
        return [h.__class__.__name__ for h in self._hooks]

    # === Utilities ===

    def _load_prompt(self, prompt_path: str) -> Optional[str]:
        """
        프롬프트 로드 (호환성 유지)

        Args:
            prompt_path: 프롬프트 파일 경로

        Returns:
            프롬프트 텍스트
        """
        return self.prompt_manager.load(
            prompt_path,
            use_cache=self.config.cache_prompts
        )

    def clear_prompt_cache(self):
        """프롬프트 캐시 클리어"""
        self.prompt_manager.clear_cache()

    def get_debug_info(self) -> Dict[str, Any]:
        """디버그 정보"""
        return {
            'brand_id': self.brand_id,
            'config': {
                'retrieval_mode': self.config.retrieval_mode.value,
                'max_parallel_retrievers': self.config.max_parallel_retrievers,
                'fallback_strategy': self.config.fallback_strategy.value,
                'cache_prompts': self.config.cache_prompts,
            },
            'retrievers': list(self.retrievers.keys()),
            'generators': list(self.generators.keys()),
            'hooks': self.get_hooks(),
            'prompt_cache': self.prompt_manager.get_cache_stats(),
            'trace': self.get_trace(),
        }

    def shutdown(self):
        """파이프라인 종료"""
        self._executor.shutdown(wait=False)
        logger.info(f"Pipeline shutdown for brand: {self.brand_id}")

    def __repr__(self) -> str:
        return (
            f"Pipeline(brand={self.brand_id}, "
            f"retrievers={len(self.retrievers)}, "
            f"generators={len(self.generators)})"
        )
