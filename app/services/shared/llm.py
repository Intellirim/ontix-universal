"""
LLM Service - Production Grade v2.0
LLM 프로덕션 서비스

Features:
    - 멀티 모델 지원
    - 스트리밍
    - 토큰 카운팅
    - 비용 추적
    - 에러 핸들링
    - 재시도 로직
    - 레이트 리미팅
    - 메트릭 추적
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from functools import lru_cache, wraps
from typing import Optional, List, Dict, Any, Generator, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
import time
import os
import logging

logger = logging.getLogger(__name__)


# ============================================================
# Enums & Types
# ============================================================

class ModelVariant(str, Enum):
    """모델 변형"""
    MINI = "mini"           # 빠른 응답, 저비용 (GPT-4o-mini)
    FULL = "full"           # 표준 (GPT-4o-mini)
    CREATIVE = "creative"   # 높은 온도 (GPT-4o-mini)
    PREMIUM = "premium"     # 최고 품질 (GPT-4o)
    FEATURE = "feature"     # Feature 핸들러용 (GPT-5-mini)


class ModelProvider(str, Enum):
    """모델 제공자"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"


@dataclass
class ModelConfig:
    """모델 설정"""
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.9
    request_timeout: int = 60
    max_retries: int = 3

    # 가격 (USD per 1M tokens)
    input_price: float = 0.0
    output_price: float = 0.0

    # 컨텍스트 윈도우
    context_window: int = 128000

    # GPT-5 계열은 temperature/top_p 지원 안함
    supports_sampling_params: bool = True


@dataclass
class LLMConfig:
    """LLM 클라이언트 설정"""
    api_key: str
    default_model: str = "gpt-4o-mini"
    provider: ModelProvider = ModelProvider.OPENAI
    enable_cost_tracking: bool = True
    enable_rate_limiting: bool = True
    rate_limit_rpm: int = 60  # Requests per minute

    # GPT-5 모델 prefix (temperature/top_p 파라미터 제거 대상)
    GPT5_MODEL_PREFIX: str = "gpt-5"

    @classmethod
    def is_gpt5_model(cls, model: str) -> bool:
        """GPT-5 계열 모델인지 확인 (temperature/top_p 제거 필요)"""
        return model.startswith(cls.GPT5_MODEL_PREFIX)

    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """환경변수에서 로드"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        return cls(
            api_key=api_key,
            default_model=os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini"),
            enable_cost_tracking=os.getenv("LLM_COST_TRACKING", "true").lower() == "true",
            rate_limit_rpm=int(os.getenv("LLM_RATE_LIMIT_RPM", "60")),
        )


@dataclass
class LLMMetrics:
    """LLM 메트릭"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    start_time: float = field(default_factory=time.time)

    # 모델별 통계
    by_model: Dict[str, Dict] = field(default_factory=dict)

    @property
    def avg_latency_ms(self) -> float:
        """평균 레이턴시"""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    @property
    def success_rate(self) -> float:
        """성공률"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        latency_ms: float,
        success: bool = True
    ):
        """요청 기록"""
        self.total_requests += 1
        self.total_latency_ms += latency_ms

        if success:
            self.successful_requests += 1
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost_usd += cost
        else:
            self.failed_requests += 1

        # 모델별 통계
        if model not in self.by_model:
            self.by_model[model] = {
                'requests': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'cost': 0.0
            }

        self.by_model[model]['requests'] += 1
        if success:
            self.by_model[model]['input_tokens'] += input_tokens
            self.by_model[model]['output_tokens'] += output_tokens
            self.by_model[model]['cost'] += cost

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': round(self.success_rate, 2),
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'total_cost_usd': round(self.total_cost_usd, 4),
            'avg_latency_ms': round(self.avg_latency_ms, 2),
            'uptime_seconds': round(time.time() - self.start_time, 0),
            'by_model': self.by_model,
        }


@dataclass
class LLMResponse:
    """LLM 응답"""
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    finish_reason: str = "stop"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'model': self.model,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_tokens': self.input_tokens + self.output_tokens,
            'cost_usd': self.cost_usd,
            'latency_ms': self.latency_ms,
            'finish_reason': self.finish_reason,
        }


# ============================================================
# Rate Limiter
# ============================================================

class LLMRateLimiter:
    """LLM 레이트 리미터"""

    def __init__(self, rpm: int = 60):
        self.rpm = rpm
        self.requests: List[float] = []
        self._lock = threading.Lock()

    def is_allowed(self) -> bool:
        """요청 허용 여부"""
        with self._lock:
            now = time.time()
            # 1분 이전 요청 제거
            self.requests = [t for t in self.requests if now - t < 60]

            if len(self.requests) < self.rpm:
                self.requests.append(now)
                return True
            return False

    def wait_if_needed(self):
        """필요시 대기"""
        while not self.is_allowed():
            time.sleep(0.5)


# ============================================================
# Token Counter
# ============================================================

class TokenCounter:
    """토큰 카운터"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self._encoder = None

    @property
    def encoder(self):
        if self._encoder is None:
            try:
                import tiktoken
                self._encoder = tiktoken.encoding_for_model(self.model)
            except Exception:
                import tiktoken
                self._encoder = tiktoken.get_encoding("cl100k_base")
        return self._encoder

    def count(self, text: str) -> int:
        """토큰 수 계산"""
        try:
            return len(self.encoder.encode(text))
        except Exception:
            # 폴백: 대략적인 계산
            return len(text) // 4

    def count_messages(self, messages: List[Dict]) -> int:
        """메시지 리스트의 토큰 수"""
        total = 0
        for msg in messages:
            # 메시지 오버헤드 (role, separators)
            total += 4
            content = msg.get('content', '')
            if isinstance(content, str):
                total += self.count(content)
        return total + 2  # 시작/끝 토큰


# ============================================================
# Cost Calculator
# ============================================================

class CostCalculator:
    """비용 계산기"""

    # 모델별 가격 (USD per 1M tokens)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.150, "output": 0.600},
        "gpt-5-mini": {"input": 0.150, "output": 0.600},  # GPT-5-mini 추가
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }

    @classmethod
    def calculate(
        cls,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """비용 계산"""
        pricing = cls.PRICING.get(model, cls.PRICING["gpt-5-mini"])

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost


# ============================================================
# LLM Client
# ============================================================

class LLMClient:
    """
    LLM 클라이언트 - Production Grade

    Features:
        - 싱글톤 패턴
        - 멀티 모델 지원
        - 토큰 카운팅
        - 비용 추적
        - 레이트 리미팅
        - 메트릭 추적
        - 스트리밍 지원
    """

    _instance: Optional['LLMClient'] = None
    _lock = threading.Lock()

    # 모델 설정 - GPT-4o-mini 사용 (gpt-5-mini가 빈 응답 반환 이슈)
    MODEL_CONFIGS = {
        ModelVariant.MINI: ModelConfig(
            model_name="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000,
            input_price=0.150,
            output_price=0.600,
            supports_sampling_params=True,
        ),
        ModelVariant.FULL: ModelConfig(
            model_name="gpt-4o-mini",
            temperature=0.7,
            max_tokens=2000,
            input_price=0.150,
            output_price=0.600,
            supports_sampling_params=True,
        ),
        ModelVariant.CREATIVE: ModelConfig(
            model_name="gpt-5-mini",
            temperature=0.9,  # GPT-5에서는 무시됨
            max_tokens=3000,
            input_price=0.150,
            output_price=0.600,
            supports_sampling_params=False,  # GPT-5는 temperature/top_p 미지원
        ),
        ModelVariant.PREMIUM: ModelConfig(
            model_name="gpt-4o",
            temperature=0.7,
            max_tokens=4000,
            input_price=2.50,
            output_price=10.00,
            supports_sampling_params=True,  # GPT-4o는 sampling params 지원
        ),
        # GPT-5: Feature 핸들러용 (AI Advisor, 콘텐츠 생성 등 중요 기능)
        ModelVariant.FEATURE: ModelConfig(
            model_name="gpt-5",
            temperature=0.7,
            max_tokens=3000,
            input_price=1.00,
            output_price=3.00,
            supports_sampling_params=False,  # GPT-5는 sampling params 비활성화
        ),
    }

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'initialized'):
            return

        self.config = LLMConfig.from_env()
        self.metrics = LLMMetrics()
        self.token_counter = TokenCounter(self.config.default_model)
        self.rate_limiter = LLMRateLimiter(self.config.rate_limit_rpm)

        # 모델 초기화
        self.models: Dict[ModelVariant, ChatOpenAI] = {}
        self._init_models()

        self.initialized = True
        logger.info(f"LLM initialized: {self.config.default_model}")

    def _init_models(self):
        """모델 초기화"""
        for variant, config in self.MODEL_CONFIGS.items():
            # GPT-5 계열은 temperature/top_p 파라미터를 지원하지 않음
            if config.supports_sampling_params:
                self.models[variant] = ChatOpenAI(
                    api_key=self.config.api_key,
                    model=config.model_name,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    request_timeout=config.request_timeout,
                    max_retries=config.max_retries,
                )
            else:
                # GPT-5-mini: sampling parameters 제거
                self.models[variant] = ChatOpenAI(
                    api_key=self.config.api_key,
                    model=config.model_name,
                    max_tokens=config.max_tokens,
                    request_timeout=config.request_timeout,
                    max_retries=config.max_retries,
                    # temperature, top_p 제거됨
                )

    def count_tokens(self, text: str) -> int:
        """토큰 수 계산"""
        return self.token_counter.count(text)

    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_variant: Union[str, ModelVariant] = ModelVariant.MINI,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        LLM 호출

        Args:
            prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트
            model_variant: 모델 변형
            temperature: 온도 (오버라이드)
            max_tokens: 최대 토큰 (오버라이드)
            **kwargs: 추가 파라미터

        Returns:
            생성된 응답 텍스트
        """
        # 레이트 리미팅
        if self.config.enable_rate_limiting:
            self.rate_limiter.wait_if_needed()

        start_time = time.time()

        # 모델 변형 파싱
        if isinstance(model_variant, str):
            try:
                model_variant = ModelVariant(model_variant)
            except ValueError:
                model_variant = ModelVariant.MINI

        try:
            llm = self.models.get(model_variant, self.models[ModelVariant.MINI])
            model_config = self.MODEL_CONFIGS.get(model_variant, self.MODEL_CONFIGS[ModelVariant.MINI])

            # 온도/토큰 오버라이드 (GPT-5 계열은 temperature 지원 안함)
            if temperature is not None and model_config.supports_sampling_params:
                llm = llm.bind(temperature=temperature)
            if max_tokens is not None:
                llm = llm.bind(max_tokens=max_tokens)

            # 메시지 구성
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))

            # 토큰 카운팅
            input_tokens = self.token_counter.count((system_prompt or "") + prompt)

            # 호출
            response = llm.invoke(messages, **kwargs)
            output_text = response.content

            # 메트릭
            output_tokens = self.token_counter.count(output_text)
            latency_ms = (time.time() - start_time) * 1000
            cost = CostCalculator.calculate(
                model_config.model_name,
                input_tokens,
                output_tokens
            )

            self.metrics.record(
                model=model_config.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                latency_ms=latency_ms,
                success=True
            )

            logger.debug(
                f"LLM: {model_variant.value} | "
                f"Tokens: {input_tokens}+{output_tokens} | "
                f"Cost: ${cost:.4f} | "
                f"Time: {latency_ms:.0f}ms"
            )

            return output_text

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.record(
                model=model_config.model_name if 'model_config' in dir() else "unknown",
                input_tokens=0,
                output_tokens=0,
                cost=0,
                latency_ms=latency_ms,
                success=False
            )
            logger.error(f"LLM error: {e}")
            raise

    def invoke_with_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_variant: Union[str, ModelVariant] = ModelVariant.MINI,
        **kwargs
    ) -> LLMResponse:
        """
        LLM 호출 (상세 응답)

        Returns:
            LLMResponse 객체
        """
        start_time = time.time()

        if isinstance(model_variant, str):
            try:
                model_variant = ModelVariant(model_variant)
            except ValueError:
                model_variant = ModelVariant.MINI

        model_config = self.MODEL_CONFIGS.get(model_variant, self.MODEL_CONFIGS[ModelVariant.MINI])

        content = self.invoke(prompt, system_prompt, model_variant, **kwargs)

        input_tokens = self.token_counter.count((system_prompt or "") + prompt)
        output_tokens = self.token_counter.count(content)
        latency_ms = (time.time() - start_time) * 1000
        cost = CostCalculator.calculate(model_config.model_name, input_tokens, output_tokens)

        return LLMResponse(
            content=content,
            model=model_config.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model_variant: Union[str, ModelVariant] = ModelVariant.FULL,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        스트리밍 응답

        Yields:
            응답 청크
        """
        if self.config.enable_rate_limiting:
            self.rate_limiter.wait_if_needed()

        if isinstance(model_variant, str):
            try:
                model_variant = ModelVariant(model_variant)
            except ValueError:
                model_variant = ModelVariant.FULL

        try:
            llm = self.models.get(model_variant, self.models[ModelVariant.FULL])

            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))

            for chunk in llm.stream(messages, **kwargs):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content

        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            raise

    def batch_invoke(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        model_variant: Union[str, ModelVariant] = ModelVariant.MINI,
        **kwargs
    ) -> List[str]:
        """
        배치 호출

        Args:
            prompts: 프롬프트 리스트

        Returns:
            응답 리스트
        """
        results = []
        for prompt in prompts:
            try:
                result = self.invoke(prompt, system_prompt, model_variant, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch invoke error: {e}")
                results.append("")
        return results

    def get_model_info(self, variant: ModelVariant = None) -> Dict[str, Any]:
        """모델 정보"""
        if variant:
            config = self.MODEL_CONFIGS.get(variant)
            if config:
                return {
                    'variant': variant.value,
                    'model': config.model_name,
                    'temperature': config.temperature,
                    'max_tokens': config.max_tokens,
                    'context_window': config.context_window,
                }
            return {}

        return {
            variant.value: {
                'model': config.model_name,
                'temperature': config.temperature,
                'max_tokens': config.max_tokens,
            }
            for variant, config in self.MODEL_CONFIGS.items()
        }

    def get_stats(self) -> Dict[str, Any]:
        """통계 조회"""
        return self.metrics.to_dict()

    def get_metrics(self) -> LLMMetrics:
        """메트릭 조회"""
        return self.metrics

    def reset_metrics(self):
        """메트릭 리셋"""
        self.metrics = LLMMetrics()

    def health_check(self) -> Dict[str, Any]:
        """헬스체크"""
        try:
            start = time.time()
            # 간단한 테스트 호출
            self.invoke("Say 'OK'", model_variant=ModelVariant.MINI)
            latency = (time.time() - start) * 1000

            return {
                'status': 'healthy',
                'latency_ms': round(latency, 2),
                'metrics': self.metrics.to_dict(),
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'metrics': self.metrics.to_dict(),
            }


# ============================================================
# Factory
# ============================================================

@lru_cache()
def get_llm_client() -> LLMClient:
    """싱글톤 LLM 클라이언트"""
    return LLMClient()
