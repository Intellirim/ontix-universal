"""
Generator Interface - Production Grade v2.0
응답 생성기 인터페이스 정의

Features:
    - 타입 안전한 생성 결과
    - 스트리밍 지원
    - 프롬프트 관리
    - 생성 메트릭
    - 품질 검증
    - 폴백 처리
"""

from abc import abstractmethod
from typing import Dict, List, Any, Optional, Generator as TypingGenerator, AsyncGenerator, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import asyncio
import time
import logging

from app.interfaces.base import (
    BaseInterface,
    ComponentType,
    ComponentStatus,
    HealthCheckResult,
)
from app.core.context import QueryContext

logger = logging.getLogger(__name__)


# ============================================================
# Enums
# ============================================================

class GenerationStatus(str, Enum):
    """생성 상태"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    STREAMING = "streaming"
    FAILED = "failed"
    TIMEOUT = "timeout"
    FALLBACK = "fallback"


class GeneratorType(str, Enum):
    """생성기 유형"""
    FACTUAL = "factual"
    INSIGHT = "insight"
    CONVERSATIONAL = "conversational"
    RECOMMENDATION = "recommendation"
    SUMMARY = "summary"
    CREATIVE = "creative"
    CUSTOM = "custom"


class OutputFormat(str, Enum):
    """출력 포맷"""
    PLAIN = "plain"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    STRUCTURED = "structured"


class ResponseTone(str, Enum):
    """응답 톤"""
    FORMAL = "formal"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    EMPATHETIC = "empathetic"


# ============================================================
# Data Classes
# ============================================================

@dataclass
class GenerationConfig:
    """생성 설정"""
    # LLM 설정
    model_variant: str = "full"  # mini, full, premium
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.9

    # 출력 설정
    output_format: OutputFormat = OutputFormat.MARKDOWN
    response_tone: ResponseTone = ResponseTone.FRIENDLY
    language: str = "ko"

    # 컨텍스트 설정
    max_context_items: int = 10
    max_context_length: int = 4000
    include_sources: bool = True

    # 검증
    validate_output: bool = True
    min_response_length: int = 10
    max_response_length: int = 5000

    # 폴백
    fallback_enabled: bool = True
    retry_count: int = 2
    timeout_ms: int = 30000

    # 캐싱
    cache_prompts: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenerationConfig':
        """딕셔너리에서 생성"""
        return cls(
            model_variant=data.get('model_variant', 'full'),
            temperature=data.get('temperature', 0.7),
            max_tokens=data.get('max_tokens', 2000),
            top_p=data.get('top_p', 0.9),
            output_format=OutputFormat(data.get('output_format', 'markdown')),
            response_tone=ResponseTone(data.get('response_tone', 'friendly')),
            language=data.get('language', 'ko'),
            max_context_items=data.get('max_context_items', 10),
            max_context_length=data.get('max_context_length', 4000),
            include_sources=data.get('include_sources', True),
            validate_output=data.get('validate_output', True),
            min_response_length=data.get('min_response_length', 10),
            max_response_length=data.get('max_response_length', 5000),
            fallback_enabled=data.get('fallback_enabled', True),
            retry_count=data.get('retry_count', 2),
            timeout_ms=data.get('timeout_ms', 30000),
            cache_prompts=data.get('cache_prompts', True),
        )

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'model_variant': self.model_variant,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'output_format': self.output_format.value,
            'response_tone': self.response_tone.value,
            'language': self.language,
            'max_context_items': self.max_context_items,
            'max_context_length': self.max_context_length,
            'include_sources': self.include_sources,
            'validate_output': self.validate_output,
            'min_response_length': self.min_response_length,
            'max_response_length': self.max_response_length,
            'fallback_enabled': self.fallback_enabled,
            'retry_count': self.retry_count,
            'timeout_ms': self.timeout_ms,
        }


@dataclass
class GenerationResult:
    """생성 결과"""
    content: str
    status: GenerationStatus = GenerationStatus.COMPLETED
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    retries: int = 0
    cached: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """성공 여부"""
        return self.status in (
            GenerationStatus.COMPLETED,
            GenerationStatus.FALLBACK
        )

    @property
    def content_length(self) -> int:
        """컨텐츠 길이"""
        return len(self.content)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'content': self.content,
            'status': self.status.value,
            'tokens_used': self.tokens_used,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'latency_ms': round(self.latency_ms, 2),
            'model': self.model,
            'retries': self.retries,
            'cached': self.cached,
            'error': self.error,
            'content_length': self.content_length,
        }


@dataclass
class GenerationMetrics:
    """생성 메트릭"""
    total_generations: int = 0
    successful_generations: int = 0
    failed_generations: int = 0
    total_tokens_used: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_retries: int = 0
    average_latency_ms: float = 0.0
    average_content_length: float = 0.0

    @property
    def success_rate(self) -> float:
        """성공률"""
        if self.total_generations == 0:
            return 0.0
        return self.successful_generations / self.total_generations

    @property
    def average_tokens_per_generation(self) -> float:
        """생성당 평균 토큰"""
        if self.total_generations == 0:
            return 0.0
        return self.total_tokens_used / self.total_generations

    def record_generation(self, result: GenerationResult):
        """생성 기록"""
        self.total_generations += 1

        if result.is_success:
            self.successful_generations += 1
        else:
            self.failed_generations += 1

        self.total_tokens_used += result.tokens_used
        self.total_prompt_tokens += result.prompt_tokens
        self.total_completion_tokens += result.completion_tokens
        self.total_retries += result.retries

        # 이동 평균
        n = self.total_generations
        self.average_latency_ms = (
            (self.average_latency_ms * (n - 1) + result.latency_ms) / n
        )
        self.average_content_length = (
            (self.average_content_length * (n - 1) + result.content_length) / n
        )

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'total_generations': self.total_generations,
            'success_rate': round(self.success_rate, 4),
            'total_tokens_used': self.total_tokens_used,
            'average_tokens_per_generation': round(self.average_tokens_per_generation, 2),
            'average_latency_ms': round(self.average_latency_ms, 2),
            'average_content_length': round(self.average_content_length, 2),
            'total_retries': self.total_retries,
        }


@dataclass
class PromptTemplate:
    """프롬프트 템플릿"""
    name: str
    system_prompt: str
    user_prompt_template: str
    variables: List[str] = field(default_factory=list)
    description: str = ""

    def render(self, **kwargs) -> str:
        """템플릿 렌더링"""
        result = self.user_prompt_template
        for var in self.variables:
            if var in kwargs:
                result = result.replace(f"{{{{{var}}}}}", str(kwargs[var]))
        return result


# ============================================================
# Generator Interface
# ============================================================

class GeneratorInterface(BaseInterface):
    """
    프로덕션급 응답 생성기 인터페이스

    모든 생성기(Factual, Insight, Recommendation 등)의 기본 인터페이스.
    QueryContext를 받아 응답을 생성합니다.

    Features:
        - 스트리밍 지원
        - 프롬프트 템플릿 관리
        - 생성 메트릭
        - 품질 검증
        - 폴백 처리

    Usage:
        class MyGenerator(GeneratorInterface):
            def _do_generate(self, context: QueryContext) -> GenerationResult:
                # 생성 로직 구현
                response = self._invoke_llm(context)
                return GenerationResult(content=response)

        generator = MyGenerator(brand_config)
        response = generator.configure(temperature=0.8).generate(context)
    """

    def __init__(
        self,
        brand_config: Dict[str, Any],
        generator_type: GeneratorType = GeneratorType.CONVERSATIONAL
    ):
        """
        Args:
            brand_config: 브랜드 설정
            generator_type: 생성기 유형
        """
        super().__init__(brand_config, ComponentType.GENERATOR)

        self.generator_type = generator_type

        # 설정 로드
        gen_config = brand_config.get('generation', {}).get(generator_type.value, {})
        self._config = GenerationConfig.from_dict(gen_config)

        # 생성 메트릭
        self._generation_metrics = GenerationMetrics()

        # 프롬프트 템플릿
        self._templates: Dict[str, PromptTemplate] = {}

        # LLM 클라이언트 (lazy load)
        self._llm = None

        # 후처리 훅
        self._post_process_hooks: List[Callable[[str], str]] = []

    # === Properties ===

    @property
    def llm(self):
        """LLM 클라이언트 lazy loading"""
        if self._llm is None:
            from app.services.shared.llm import get_llm_client
            self._llm = get_llm_client()
        return self._llm

    # === Main Methods ===

    def generate(self, context: QueryContext) -> str:
        """
        응답 생성

        Args:
            context: 쿼리 컨텍스트

        Returns:
            생성된 응답 텍스트
        """
        start_time = time.time()
        retries = 0

        try:
            # 생성 수행 (재시도 포함)
            while retries <= self._config.retry_count:
                try:
                    result = self._do_generate(context)
                    result.retries = retries
                    result.latency_ms = (time.time() - start_time) * 1000

                    # 검증
                    if self._config.validate_output:
                        valid, error = self._validate_response(result.content)
                        if not valid:
                            if retries < self._config.retry_count:
                                retries += 1
                                logger.warning(f"Validation failed, retry {retries}: {error}")
                                continue
                            result.content = self._truncate(result.content)

                    # 후처리
                    result.content = self._post_process(result.content)

                    # 컨텍스트 업데이트
                    context.set_response(result.content, result.tokens_used)

                    # 메트릭 기록
                    self._record_generation(result)

                    logger.info(
                        f"{self.generator_type.value} generation: "
                        f"{result.content_length} chars in {result.latency_ms:.1f}ms"
                    )

                    return result.content

                except Exception as e:
                    retries += 1
                    logger.error(f"Generation error (attempt {retries}): {e}")

                    if retries > self._config.retry_count:
                        raise

                    time.sleep(0.5 * retries)

            # 폴백
            if self._config.fallback_enabled:
                return self._get_fallback_response(context)

            return ""

        except asyncio.TimeoutError:
            error = f"Generation timeout after {self._config.timeout_ms}ms"
            logger.error(error)
            self._record_call(False, (time.time() - start_time) * 1000, error)

            if self._config.fallback_enabled:
                return self._get_fallback_response(context)
            return ""

        except Exception as e:
            error = f"Generation error: {str(e)}"
            logger.error(error, exc_info=True)
            self._record_call(False, (time.time() - start_time) * 1000, error)

            if self._config.fallback_enabled:
                return self._get_fallback_response(context)
            return ""

    async def generate_async(self, context: QueryContext) -> str:
        """
        비동기 응답 생성

        Args:
            context: 쿼리 컨텍스트

        Returns:
            생성된 응답 텍스트
        """
        try:
            result = await asyncio.wait_for(
                self._do_generate_async(context),
                timeout=self._config.timeout_ms / 1000
            )

            result.content = self._post_process(result.content)
            context.set_response(result.content, result.tokens_used)
            self._record_generation(result)

            return result.content

        except asyncio.TimeoutError:
            logger.error(f"Async generation timeout")
            return self._get_fallback_response(context) if self._config.fallback_enabled else ""
        except Exception as e:
            logger.error(f"Async generation error: {e}", exc_info=True)
            return self._get_fallback_response(context) if self._config.fallback_enabled else ""

    def generate_stream(
        self,
        context: QueryContext
    ) -> TypingGenerator[str, None, None]:
        """
        스트리밍 응답 생성

        Args:
            context: 쿼리 컨텍스트

        Yields:
            응답 청크
        """
        start_time = time.time()

        try:
            for chunk in self._do_generate_stream(context):
                yield chunk

        except Exception as e:
            logger.error(f"Stream generation error: {e}")
            yield f"Error: {str(e)}"

        finally:
            latency_ms = (time.time() - start_time) * 1000
            self._record_call(True, latency_ms)

    async def generate_stream_async(
        self,
        context: QueryContext
    ) -> AsyncGenerator[str, None]:
        """
        비동기 스트리밍 응답 생성

        Args:
            context: 쿼리 컨텍스트

        Yields:
            응답 청크
        """
        try:
            async for chunk in self._do_generate_stream_async(context):
                yield chunk
        except Exception as e:
            logger.error(f"Async stream generation error: {e}")
            yield f"Error: {str(e)}"

    # === Abstract Methods ===

    @abstractmethod
    def _do_generate(self, context: QueryContext) -> GenerationResult:
        """
        실제 생성 로직 (구현 필수)

        Args:
            context: 쿼리 컨텍스트

        Returns:
            생성 결과
        """
        pass

    async def _do_generate_async(self, context: QueryContext) -> GenerationResult:
        """
        비동기 생성 로직 (오버라이드 가능)

        기본 구현은 동기 메서드를 래핑합니다.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._do_generate, context)

    def _do_generate_stream(
        self,
        context: QueryContext
    ) -> TypingGenerator[str, None, None]:
        """
        스트리밍 생성 로직 (오버라이드 가능)

        기본 구현은 전체 응답을 한 번에 반환합니다.
        """
        result = self._do_generate(context)
        yield result.content

    async def _do_generate_stream_async(
        self,
        context: QueryContext
    ) -> AsyncGenerator[str, None]:
        """
        비동기 스트리밍 생성 로직 (오버라이드 가능)
        """
        result = await self._do_generate_async(context)
        yield result.content

    # === Prompt Methods ===

    def build_system_prompt(self, context: QueryContext) -> str:
        """
        시스템 프롬프트 구성 (오버라이드 가능)

        Args:
            context: 쿼리 컨텍스트

        Returns:
            시스템 프롬프트
        """
        return self._get_default_system_prompt()

    def build_user_prompt(self, context: QueryContext) -> str:
        """
        사용자 프롬프트 구성 (오버라이드 가능)

        Args:
            context: 쿼리 컨텍스트

        Returns:
            사용자 프롬프트
        """
        parts = []

        # 질문
        parts.append(f"질문: {context.question}")

        # 컨텍스트 정보
        if context.retrieval_results:
            parts.append("\n관련 정보:")
            parts.append(self._format_context(context))

        # 대화 히스토리
        if context.conversation_history:
            parts.append("\n이전 대화:")
            parts.append(self._format_history(context))

        return "\n".join(parts)

    def register_template(self, template: PromptTemplate):
        """템플릿 등록"""
        self._templates[template.name] = template
        logger.debug(f"Template registered: {template.name}")

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """템플릿 조회"""
        return self._templates.get(name)

    # === Configuration ===

    def configure(self, **kwargs) -> 'GeneratorInterface':
        """
        생성 설정 변경 (체이닝 지원)

        Usage:
            generator.configure(temperature=0.8, max_tokens=3000).generate(context)

        Args:
            **kwargs: 설정 키-값

        Returns:
            self (체이닝용)
        """
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
                logger.debug(f"Generator config: {key}={value}")
        return self

    def get_generation_config(self) -> GenerationConfig:
        """생성 설정 조회"""
        return self._config

    def _get_component_config(self) -> Dict[str, Any]:
        """컴포넌트 설정 (오버라이드)"""
        return self._config.to_dict()

    # === Hooks ===

    def add_post_process_hook(
        self,
        hook: Callable[[str], str]
    ) -> 'GeneratorInterface':
        """
        후처리 훅 추가

        Args:
            hook: 후처리 함수

        Returns:
            self (체이닝용)
        """
        self._post_process_hooks.append(hook)
        return self

    def clear_hooks(self):
        """훅 초기화"""
        self._post_process_hooks.clear()

    # === Metrics ===

    def get_generation_metrics(self) -> Dict[str, Any]:
        """생성 메트릭 조회"""
        return self._generation_metrics.to_dict()

    def reset_generation_metrics(self):
        """생성 메트릭 초기화"""
        self._generation_metrics = GenerationMetrics()

    # === Health Check ===

    def _do_health_check(self) -> HealthCheckResult:
        """헬스체크 (오버라이드)"""
        try:
            metrics = self._generation_metrics

            if metrics.total_generations > 0:
                if metrics.success_rate < 0.5:
                    return HealthCheckResult(
                        healthy=False,
                        status=ComponentStatus.DEGRADED,
                        message=f"Low success rate: {metrics.success_rate:.1%}",
                        details={'success_rate': metrics.success_rate}
                    )

            return HealthCheckResult(
                healthy=True,
                status=ComponentStatus.READY,
                message="OK",
                details={
                    'total_generations': metrics.total_generations,
                    'success_rate': round(metrics.success_rate, 4),
                }
            )

        except Exception as e:
            return HealthCheckResult(
                healthy=False,
                status=ComponentStatus.ERROR,
                message=str(e)
            )

    # === Debug ===

    def get_debug_info(self) -> Dict[str, Any]:
        """디버그 정보 (확장)"""
        base_info = super().get_debug_info()
        base_info.update({
            'generator_type': self.generator_type.value,
            'generation_config': self._config.to_dict(),
            'generation_metrics': self.get_generation_metrics(),
            'registered_templates': list(self._templates.keys()),
        })
        return base_info

    # === Private Helpers ===

    def _get_default_system_prompt(self) -> str:
        """기본 시스템 프롬프트"""
        return f"""You are a helpful assistant for {self.brand_name}.

Answer questions accurately and concisely based on the provided context.
If you don't have enough information, say so clearly.

Guidelines:
- Be {self._config.response_tone.value} in tone
- Output in {self._config.output_format.value} format
- Answer in {self._config.language}
"""

    def _get_fallback_response(self, context: QueryContext) -> str:
        """폴백 응답"""
        logger.warning("Using fallback response")
        return "죄송합니다. 현재 요청을 처리할 수 없습니다. 잠시 후 다시 시도해 주세요."

    def _format_context(self, context: QueryContext) -> str:
        """컨텍스트 포맷팅"""
        parts = []

        for result in context.retrieval_results:
            source = result.source if hasattr(result, 'source') else 'unknown'
            data = result.data if hasattr(result, 'data') else []

            if data:
                parts.append(f"\n[{source.upper()}]")
                for item in data[:self._config.max_context_items]:
                    content = item.get('content', str(item))[:500]
                    parts.append(f"- {content}")

        return "\n".join(parts) if parts else "No context available."

    def _format_history(self, context: QueryContext) -> str:
        """히스토리 포맷팅"""
        parts = []
        recent = context.conversation_history[-5:]

        for msg in recent:
            if hasattr(msg, 'role'):
                role = msg.role.upper()
                content = msg.content
            elif isinstance(msg, dict):
                role = msg.get('role', 'user').upper()
                content = msg.get('content', '')
            else:
                continue

            parts.append(f"{role}: {content}")

        return "\n".join(parts)

    def _validate_response(self, response: str) -> tuple:
        """응답 검증"""
        if not response:
            return False, "Empty response"

        if len(response) < self._config.min_response_length:
            return False, f"Response too short: {len(response)}"

        if len(response) > self._config.max_response_length:
            return False, f"Response too long: {len(response)}"

        return True, None

    def _truncate(self, response: str) -> str:
        """응답 자르기"""
        if len(response) <= self._config.max_response_length:
            return response

        max_len = self._config.max_response_length - 3
        truncated = response[:max_len]

        # 문장 단위로 자르기
        last_period = max(
            truncated.rfind('.'),
            truncated.rfind('。'),
            truncated.rfind('!'),
            truncated.rfind('?')
        )

        if last_period > max_len * 0.7:
            return truncated[:last_period + 1]

        return truncated + "..."

    def _post_process(self, response: str) -> str:
        """후처리 실행"""
        result = response.strip()

        # 사용자 정의 훅
        for hook in self._post_process_hooks:
            try:
                result = hook(result)
            except Exception as e:
                logger.error(f"Post-process hook error: {e}")

        return result

    def _record_generation(self, result: GenerationResult):
        """생성 기록"""
        self._generation_metrics.record_generation(result)
        self._record_call(result.is_success, result.latency_ms)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"brand={self.brand_id}, "
            f"type={self.generator_type.value})"
        )


# ============================================================
# Exports
# ============================================================

__all__ = [
    'GeneratorInterface',
    'GenerationStatus',
    'GeneratorType',
    'OutputFormat',
    'ResponseTone',
    'GenerationConfig',
    'GenerationResult',
    'GenerationMetrics',
    'PromptTemplate',
]
