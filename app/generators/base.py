"""
Base Generator - Production Grade v2.0
응답 생성기 기반 클래스

Features:
    - 프롬프트 템플릿 관리
    - 응답 포맷팅 유틸리티
    - 출력 검증
    - 스트리밍 지원
    - 생성 메트릭 추적
    - 폴백 처리
"""

from typing import Dict, List, Any, Optional, Generator, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod
from jinja2 import Template, Environment, FileSystemLoader, TemplateError
import threading
import hashlib
import time
import json
import re
import logging

logger = logging.getLogger(__name__)


# ============================================================
# Enums
# ============================================================

class GeneratorType(str, Enum):
    """생성기 유형"""
    FACTUAL = "factual"
    INSIGHT = "insight"
    CONVERSATIONAL = "conversational"
    RECOMMENDATION = "recommendation"
    SUMMARY = "summary"
    CREATIVE = "creative"


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
# Configuration
# ============================================================

@dataclass
class GeneratorConfig:
    """생성기 설정"""
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

    # 캐싱
    cache_prompts: bool = True

    @classmethod
    def from_dict(cls, data: Dict) -> 'GeneratorConfig':
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
            cache_prompts=data.get('cache_prompts', True),
        )


# ============================================================
# Metrics
# ============================================================

@dataclass
class GenerationMetrics:
    """생성 메트릭"""
    generator_type: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    context_items: int = 0
    retries: int = 0
    success: bool = True
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000

    def complete(self, success: bool = True, error: str = None):
        """완료 마킹"""
        self.end_time = time.time()
        self.success = success
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            'generator_type': self.generator_type,
            'duration_ms': round(self.duration_ms, 2),
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'context_items': self.context_items,
            'retries': self.retries,
            'success': self.success,
            'error': self.error,
        }


# ============================================================
# Template Manager
# ============================================================

class TemplateManager:
    """프롬프트 템플릿 관리자"""

    def __init__(self, base_path: str = "prompts", cache_size: int = 50):
        self.base_path = Path(base_path)
        self.cache_size = cache_size
        self._cache: Dict[str, Template] = {}
        self._cache_order: List[str] = []
        self._lock = threading.Lock()

        # Jinja2 환경 설정
        if self.base_path.exists():
            self._env = Environment(
                loader=FileSystemLoader(str(self.base_path)),
                autoescape=False
            )
        else:
            self._env = None

    def load(
        self,
        template_path: str,
        use_cache: bool = True
    ) -> Optional[Template]:
        """
        템플릿 로드

        Args:
            template_path: 템플릿 파일 경로
            use_cache: 캐시 사용 여부

        Returns:
            Template 객체 또는 None
        """
        if use_cache:
            with self._lock:
                if template_path in self._cache:
                    return self._cache[template_path]

        try:
            full_path = self.base_path / template_path

            if not full_path.exists():
                logger.debug(f"Template not found: {template_path}")
                return None

            with open(full_path, 'r', encoding='utf-8') as f:
                template = Template(f.read())

            if use_cache:
                self._add_to_cache(template_path, template)

            return template

        except TemplateError as e:
            logger.error(f"Template error ({template_path}): {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load template ({template_path}): {e}")
            return None

    def render(
        self,
        template_path: str,
        variables: Dict[str, Any],
        use_cache: bool = True
    ) -> Optional[str]:
        """
        템플릿 렌더링

        Args:
            template_path: 템플릿 파일 경로
            variables: 템플릿 변수
            use_cache: 캐시 사용 여부

        Returns:
            렌더링된 문자열 또는 None
        """
        template = self.load(template_path, use_cache)

        if not template:
            return None

        try:
            return template.render(**variables)
        except TemplateError as e:
            logger.error(f"Template render error: {e}")
            return None

    def _add_to_cache(self, key: str, template: Template):
        """캐시에 추가 (LRU)"""
        with self._lock:
            if key in self._cache:
                self._cache_order.remove(key)
            elif len(self._cache) >= self.cache_size:
                oldest = self._cache_order.pop(0)
                del self._cache[oldest]

            self._cache[key] = template
            self._cache_order.append(key)

    def clear_cache(self):
        """캐시 클리어"""
        with self._lock:
            self._cache.clear()
            self._cache_order.clear()


# ============================================================
# Response Formatter
# ============================================================

class ResponseFormatter:
    """응답 포맷터"""

    @staticmethod
    def format_context(
        retrieval_results: List[Any],
        max_items: int = 10,
        max_length: int = 4000
    ) -> str:
        """
        검색 결과를 컨텍스트 문자열로 포맷

        Args:
            retrieval_results: 검색 결과 리스트
            max_items: 최대 항목 수
            max_length: 최대 문자열 길이

        Returns:
            포맷된 컨텍스트 문자열
        """
        if not retrieval_results:
            return "No relevant information found."

        formatted_parts = []
        total_length = 0
        item_count = 0

        for result in retrieval_results:
            if item_count >= max_items:
                break

            source = result.source if hasattr(result, 'source') else 'unknown'
            items = result.items if hasattr(result, 'items') else []

            if not items:
                continue

            formatted_parts.append(f"\n[{source.upper()}]")

            for item in items[:5]:
                # RetrievalItem 객체 또는 딕셔너리 처리
                if hasattr(item, 'to_dict'):
                    item_dict = item.to_dict()
                    item_str = f"[{item_dict.get('node_type', 'unknown')}] {item_dict.get('content', '')}"
                elif isinstance(item, dict):
                    item_str = json.dumps(item, ensure_ascii=False)
                else:
                    item_str = str(item)

                if total_length + len(item_str) > max_length:
                    formatted_parts.append("... (truncated)")
                    break

                formatted_parts.append(f"- {item_str}")
                total_length += len(item_str)
                item_count += 1

        return "\n".join(formatted_parts) if formatted_parts else "No context available."

    @staticmethod
    def format_history(
        history: List[Any],
        max_turns: int = 5
    ) -> str:
        """
        대화 히스토리 포맷

        Args:
            history: 대화 히스토리
            max_turns: 최대 턴 수

        Returns:
            포맷된 히스토리 문자열
        """
        if not history:
            return ""

        parts = []
        recent = history[-max_turns:]

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

    @staticmethod
    def format_products(
        products: List[Dict],
        max_items: int = 10
    ) -> str:
        """
        상품 목록 포맷

        Args:
            products: 상품 리스트
            max_items: 최대 항목 수

        Returns:
            포맷된 상품 문자열
        """
        if not products:
            return "No products available."

        parts = []
        for i, prod in enumerate(products[:max_items], 1):
            name = prod.get('name', 'Unknown')
            price = prod.get('price', 0)
            stock = prod.get('stock', 'N/A')
            description = prod.get('description', '')[:100]

            parts.append(
                f"{i}. {name}\n"
                f"   가격: ₩{price:,}\n"
                f"   재고: {stock}\n"
                f"   설명: {description}..."
            )

        return "\n".join(parts)

    @staticmethod
    def format_stats(
        stats: List[Dict],
        max_items: int = 10
    ) -> str:
        """
        통계 데이터 포맷

        Args:
            stats: 통계 리스트
            max_items: 최대 항목 수

        Returns:
            포맷된 통계 문자열
        """
        if not stats:
            return "No statistics available."

        parts = []
        for i, item in enumerate(stats[:max_items], 1):
            date = item.get('date', '')
            likes = item.get('likes', 0)
            views = item.get('views', 0)
            content = item.get('content', '')[:80]

            date_str = f"[{date}] " if date else ""
            content_str = f"\n   Content: {content}..." if content else ""

            parts.append(
                f"{i}. {date_str}Likes: {likes:,} | Views: {views:,}{content_str}"
            )

        return "\n".join(parts)


# ============================================================
# Output Validator
# ============================================================

class OutputValidator:
    """출력 검증기"""

    # 필터링할 패턴 (보안 관련만 - "내부"는 일반적인 비즈니스 용어이므로 제외)
    FORBIDDEN_PATTERNS = [
        r'(?i)(api.?key|password|secret|token)',
        r'(?i)(confidential|private)',
    ]

    @classmethod
    def validate(
        cls,
        response: str,
        min_length: int = 10,
        max_length: int = 5000
    ) -> tuple[bool, Optional[str]]:
        """
        응답 검증

        Args:
            response: 검증할 응답
            min_length: 최소 길이
            max_length: 최대 길이

        Returns:
            (valid, error_message)
        """
        if not response:
            return False, "Empty response"

        if len(response) < min_length:
            return False, f"Response too short: {len(response)} < {min_length}"

        if len(response) > max_length:
            return False, f"Response too long: {len(response)} > {max_length}"

        # 금지 패턴 체크
        for pattern in cls.FORBIDDEN_PATTERNS:
            if re.search(pattern, response):
                return False, f"Forbidden pattern detected"

        return True, None

    @classmethod
    def sanitize(cls, response: str) -> str:
        """
        응답 정제

        Args:
            response: 정제할 응답

        Returns:
            정제된 응답
        """
        # 제어 문자 제거 (탭, 개행 제외)
        result = ''.join(char for char in response if ord(char) >= 32 or char in '\t\n\r')

        # 앞뒤 공백 제거
        result = result.strip()

        # 연속 개행 제거
        result = re.sub(r'\n{3,}', '\n\n', result)

        # 연속 공백 제거
        result = re.sub(r' {2,}', ' ', result)

        return result

    @classmethod
    def truncate(cls, response: str, max_length: int) -> str:
        """
        응답 자르기

        Args:
            response: 자를 응답
            max_length: 최대 길이

        Returns:
            잘린 응답 (max_length 이하)
        """
        if len(response) <= max_length:
            return response

        # 말줄임표를 고려한 최대 길이
        effective_max = max_length - 3  # "..." 길이

        # 문장 단위로 자르기 시도
        truncated = response[:effective_max]
        last_period = max(
            truncated.rfind('.'),
            truncated.rfind('。'),
            truncated.rfind('!'),
            truncated.rfind('?')
        )

        if last_period > effective_max * 0.7:
            return truncated[:last_period + 1]

        return truncated + "..."


# ============================================================
# Base Generator
# ============================================================

class BaseGenerator(ABC):
    """
    프로덕션급 응답 생성기 베이스 클래스

    Features:
        - 템플릿 기반 프롬프트 관리
        - 응답 포맷팅 및 검증
        - 생성 메트릭 추적
        - 스트리밍 지원
        - 폴백 처리
    """

    # 클래스 레벨 템플릿 매니저 (공유)
    _template_manager: Optional[TemplateManager] = None
    _template_lock = threading.Lock()

    def __init__(self, brand_config: Dict, generator_type: GeneratorType = None):
        """
        Args:
            brand_config: 브랜드 설정
            generator_type: 생성기 유형
        """
        self.brand_config = brand_config
        self.brand_id = brand_config.get('brand', {}).get('id', 'unknown')
        self.brand_name = brand_config.get('brand', {}).get('name', 'Unknown')
        self.generator_type = generator_type or GeneratorType.CONVERSATIONAL

        # 설정 로드
        gen_config = brand_config.get('generation', {}).get(
            self.generator_type.value, {}
        )
        self.config = GeneratorConfig.from_dict(gen_config)

        # 템플릿 매니저 초기화 (싱글톤)
        self._init_template_manager()

        # LLM 클라이언트 (lazy load)
        self._llm = None

        # 메트릭
        self._last_metrics: Optional[GenerationMetrics] = None

    @classmethod
    def _init_template_manager(cls):
        """템플릿 매니저 초기화 (싱글톤)"""
        if cls._template_manager is None:
            with cls._template_lock:
                if cls._template_manager is None:
                    cls._template_manager = TemplateManager()

    @property
    def llm(self):
        """LLM 클라이언트 lazy loading"""
        if self._llm is None:
            from app.services.shared.llm import get_llm_client
            self._llm = get_llm_client()
        return self._llm

    @property
    def template_manager(self) -> TemplateManager:
        """템플릿 매니저"""
        return self._template_manager

    # === Abstract Methods ===

    @abstractmethod
    def generate(self, context: 'QueryContext') -> str:
        """
        응답 생성 (구현 필수)

        Args:
            context: 쿼리 컨텍스트

        Returns:
            생성된 응답
        """
        pass

    @abstractmethod
    def _build_user_prompt(self, context: 'QueryContext') -> str:
        """
        사용자 프롬프트 구성 (구현 필수)

        Args:
            context: 쿼리 컨텍스트

        Returns:
            사용자 프롬프트
        """
        pass

    # === Template Methods ===

    def _load_system_prompt(
        self,
        question_type: str,
        custom_variables: Optional[Dict] = None
    ) -> str:
        """
        시스템 프롬프트 로드 (우선순위 기반)

        우선순위:
        1. 브랜드 전용 프롬프트 (prompts/{brand_id}/...)
        2. 공통 프롬프트 (prompts/shared/...)
        3. 하드코딩 폴백

        Args:
            question_type: 질문 타입
            custom_variables: 추가 템플릿 변수

        Returns:
            렌더링된 프롬프트
        """
        # 질문 타입 문자열 변환
        if hasattr(question_type, 'value'):
            qtype_str = question_type.value
        else:
            qtype_str = str(question_type)

        generation_config = self.brand_config.get('generation', {})
        qtype_config = generation_config.get(qtype_str, {})

        # 1순위: 브랜드 전용 프롬프트
        brand_prompt = qtype_config.get('prompt')

        # 2순위: 공통 프롬프트 (폴백)
        shared_prompt = qtype_config.get('fallback_prompt')

        # 템플릿 변수 준비
        variables = self._prepare_template_variables(qtype_config, custom_variables)

        # 1순위 시도
        if brand_prompt:
            rendered = self.template_manager.render(brand_prompt, variables)
            if rendered:
                logger.debug(f"Loaded brand prompt: {brand_prompt}")
                return rendered

        # 2순위 시도
        if shared_prompt:
            rendered = self.template_manager.render(shared_prompt, variables)
            if rendered:
                logger.debug(f"Loaded shared prompt: {shared_prompt}")
                return rendered

        # 폴백
        logger.warning(f"No prompt found for {qtype_str}, using fallback")
        return self._get_fallback_prompt()

    def _prepare_template_variables(
        self,
        qtype_config: Dict,
        custom_variables: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        템플릿 변수 준비

        Args:
            qtype_config: 질문 타입 설정
            custom_variables: 추가 변수

        Returns:
            템플릿 변수 딕셔너리
        """
        # 기본 브랜드 정보
        variables = {
            'brand_id': self.brand_id,
            'brand_name': self.brand_name,
            'brand_description': self.brand_config.get('brand', {}).get('description', ''),
            'brand_industry': self.brand_config.get('brand', {}).get('industry', ''),
            'language': self.config.language,
            'tone': self.config.response_tone.value,
            'output_format': self.config.output_format.value,
        }

        # YAML에서 정의한 프롬프트 변수
        prompt_variables = qtype_config.get('prompt_variables', {})
        variables.update(prompt_variables)

        # 커스텀 변수 (우선순위 최상위)
        if custom_variables:
            variables.update(custom_variables)

        return variables

    def _get_fallback_prompt(self) -> str:
        """최종 폴백 프롬프트"""
        return f"""You are a helpful assistant for {self.brand_name}.

Answer questions accurately and concisely based on the provided context.
If you don't have enough information, say so clearly.

Guidelines:
- Be {self.config.response_tone.value} in tone
- Output in {self.config.output_format.value} format
- Answer in Korean
"""

    # === Generation Helpers ===

    def _invoke_llm(
        self,
        user_prompt: str,
        system_prompt: str,
        metrics: GenerationMetrics
    ) -> str:
        """
        LLM 호출 (재시도 포함)

        Args:
            user_prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트
            metrics: 메트릭 객체

        Returns:
            LLM 응답
        """
        last_error = None

        for attempt in range(self.config.retry_count + 1):
            try:
                response = self.llm.invoke(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    model_variant=self.config.model_variant,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )

                # 응답 정제
                response = OutputValidator.sanitize(response)

                # 검증
                if self.config.validate_output:
                    valid, error = OutputValidator.validate(
                        response,
                        self.config.min_response_length,
                        self.config.max_response_length
                    )

                    if not valid:
                        logger.warning(f"Response validation failed: {error}")
                        if attempt < self.config.retry_count:
                            metrics.retries += 1
                            continue

                        # 마지막 시도에서도 실패하면 truncate
                        response = OutputValidator.truncate(
                            response,
                            self.config.max_response_length
                        )

                return response

            except Exception as e:
                last_error = e
                logger.error(f"LLM invoke error (attempt {attempt + 1}): {e}")
                metrics.retries += 1

                if attempt < self.config.retry_count:
                    time.sleep(0.5 * (attempt + 1))

        # 모든 재시도 실패
        metrics.complete(success=False, error=str(last_error))
        raise last_error

    def _invoke_llm_stream(
        self,
        user_prompt: str,
        system_prompt: str
    ) -> Generator[str, None, None]:
        """
        LLM 스트리밍 호출

        Args:
            user_prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트

        Yields:
            응답 청크
        """
        try:
            for chunk in self.llm.stream(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model_variant=self.config.model_variant,
                temperature=self.config.temperature,
            ):
                yield chunk

        except Exception as e:
            logger.error(f"LLM stream error: {e}")
            yield f"Error: {str(e)}"

    # === Metrics ===

    def _create_metrics(self, context: 'QueryContext') -> GenerationMetrics:
        """메트릭 생성"""
        metrics = GenerationMetrics(
            generator_type=self.generator_type.value,
            context_items=context.get_total_retrieval_count() if hasattr(context, 'get_total_retrieval_count') else 0
        )
        return metrics

    def get_last_metrics(self) -> Optional[Dict[str, Any]]:
        """마지막 생성 메트릭 조회"""
        if self._last_metrics:
            return self._last_metrics.to_dict()
        return None

    # === Utilities ===

    def format_context(self, context: 'QueryContext') -> str:
        """컨텍스트 포맷팅 (편의 메서드)"""
        return ResponseFormatter.format_context(
            context.retrieval_results if hasattr(context, 'retrieval_results') else [],
            max_items=self.config.max_context_items,
            max_length=self.config.max_context_length
        )

    def format_history(self, context: 'QueryContext') -> str:
        """히스토리 포맷팅 (편의 메서드)"""
        history = context.conversation_history if hasattr(context, 'conversation_history') else []
        return ResponseFormatter.format_history(history)

    def configure(self, **kwargs):
        """런타임 설정 변경"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.debug(f"Config updated: {key}={value}")

    def get_debug_info(self) -> Dict[str, Any]:
        """디버그 정보"""
        return {
            'generator_type': self.generator_type.value,
            'brand_id': self.brand_id,
            'config': {
                'model_variant': self.config.model_variant,
                'temperature': self.config.temperature,
                'output_format': self.config.output_format.value,
                'response_tone': self.config.response_tone.value,
                'validate_output': self.config.validate_output,
            },
            'last_metrics': self.get_last_metrics(),
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"brand={self.brand_id}, "
            f"type={self.generator_type.value})"
        )
