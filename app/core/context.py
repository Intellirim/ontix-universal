"""
Query Context - Production Grade v2.0
쿼리 처리 컨텍스트 관리

Features:
    - 타입 안전한 데이터 구조
    - 성능 메트릭 추적
    - 직렬화/역직렬화
    - 검색 결과 집계
    - 대화 히스토리 관리
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json
import hashlib
import logging
import time

logger = logging.getLogger(__name__)


class QuestionType(str, Enum):
    """질문 유형"""
    PRODUCT_RECOMMENDATION = "product_recommendation"
    ANALYTICS = "analytics"
    ADVISOR = "advisor"
    FACTUAL = "factual"
    CONVERSATIONAL = "conversational"
    UNKNOWN = "unknown"


class ProcessingStage(str, Enum):
    """처리 단계"""
    CREATED = "created"
    ROUTING = "routing"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class RetrievalResult:
    """검색 결과 단위"""
    source: str
    data: List[Dict[str, Any]]
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def __len__(self) -> int:
        return len(self.data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'data': self.data,
            'score': self.score,
            'metadata': self.metadata,
            'count': len(self.data),
        }


@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    start_time: float = field(default_factory=time.time)
    routing_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0

    def mark_routing_complete(self):
        """라우팅 완료 마킹"""
        self.routing_time_ms = (time.time() - self.start_time) * 1000

    def mark_retrieval_complete(self):
        """검색 완료 마킹"""
        elapsed = (time.time() - self.start_time) * 1000
        self.retrieval_time_ms = elapsed - self.routing_time_ms

    def mark_generation_complete(self):
        """생성 완료 마킹"""
        elapsed = (time.time() - self.start_time) * 1000
        self.generation_time_ms = elapsed - self.routing_time_ms - self.retrieval_time_ms
        self.total_time_ms = elapsed

    def to_dict(self) -> Dict[str, float]:
        return {
            'routing_time_ms': round(self.routing_time_ms, 2),
            'retrieval_time_ms': round(self.retrieval_time_ms, 2),
            'generation_time_ms': round(self.generation_time_ms, 2),
            'total_time_ms': round(self.total_time_ms, 2),
        }


@dataclass
class ConversationMessage:
    """대화 메시지"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class QueryContext:
    """
    쿼리 컨텍스트 - Production Grade

    질문 처리 과정 전체를 추적하고 관리하는 핵심 클래스
    """

    # === 필수 정보 ===
    brand_id: str
    question: str

    # === 대화 히스토리 ===
    conversation_history: List[ConversationMessage] = field(default_factory=list)

    # === 분석 결과 ===
    question_type: QuestionType = QuestionType.UNKNOWN
    intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0

    # === 검색 결과 ===
    retrieval_results: List[RetrievalResult] = field(default_factory=list)

    # === 프롬프트 ===
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None

    # === 응답 ===
    response: Optional[str] = None
    response_tokens: int = 0

    # === 처리 상태 ===
    stage: ProcessingStage = ProcessingStage.CREATED
    error: Optional[str] = None

    # === 메트릭 & 메타데이터 ===
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    # === 캐싱 ===
    cache_key: Optional[str] = None

    def __post_init__(self):
        """초기화 후 처리"""
        # 캐시 키 생성
        self.cache_key = self._generate_cache_key()

        # 대화 히스토리 변환 (dict -> ConversationMessage)
        if self.conversation_history and isinstance(self.conversation_history[0], dict):
            self.conversation_history = [
                ConversationMessage(
                    role=msg.get('role', 'user'),
                    content=msg.get('content', ''),
                    metadata=msg.get('metadata', {})
                )
                for msg in self.conversation_history
            ]

    def _generate_cache_key(self) -> str:
        """캐시 키 생성"""
        key_data = f"{self.brand_id}:{self.question}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]

    # === 검색 결과 관리 ===

    def add_retrieval_result(
        self,
        source: str,
        data: List[Dict],
        metadata: Dict = None,
        score: float = 0.0
    ):
        """
        검색 결과 추가

        Args:
            source: 검색 소스 (예: "vector_search", "graph_search")
            data: 검색 결과 데이터
            metadata: 메타데이터
            score: 검색 점수
        """
        result = RetrievalResult(
            source=source,
            data=data,
            score=score,
            metadata=metadata or {}
        )
        self.retrieval_results.append(result)

        logger.debug(f"Added retrieval: {source} ({len(data)} items, score={score:.2f})")

    def get_retrieval_result(self, source: str) -> Optional[RetrievalResult]:
        """특정 소스의 검색 결과 조회"""
        for result in self.retrieval_results:
            if result.source == source:
                return result
        return None

    def get_retrieval_data(self, source: str) -> List[Dict]:
        """특정 소스의 검색 데이터 조회"""
        result = self.get_retrieval_result(source)
        return result.data if result else []

    def has_retrieval_result(self, source: str) -> bool:
        """검색 결과 존재 여부"""
        result = self.get_retrieval_result(source)
        return result is not None and len(result.data) > 0

    def get_all_retrieval_data(self) -> List[Dict]:
        """모든 검색 결과 데이터 통합 조회"""
        all_data = []
        for result in self.retrieval_results:
            # RetrievalResult.items를 사용 (data 대신)
            if hasattr(result, 'items'):
                all_data.extend(result.items)
            elif hasattr(result, 'data'):
                all_data.extend(result.data)
        return all_data

    def get_total_retrieval_count(self) -> int:
        """전체 검색 결과 개수"""
        total = 0
        for r in self.retrieval_results:
            if hasattr(r, 'items'):
                total += len(r.items)
            elif hasattr(r, 'data'):
                total += len(r.data)
        return total

    def get_retrieval_sources(self) -> List[str]:
        """검색 소스 목록"""
        return [r.source for r in self.retrieval_results]

    # === 상태 관리 ===

    def set_question_type(self, question_type: Union[str, QuestionType], confidence: float = 0.0):
        """질문 타입 설정"""
        if isinstance(question_type, str):
            try:
                self.question_type = QuestionType(question_type)
            except ValueError:
                self.question_type = QuestionType.UNKNOWN
        else:
            self.question_type = question_type

        self.confidence = confidence
        self.stage = ProcessingStage.ROUTING
        self.metrics.mark_routing_complete()

        logger.debug(f"Question type: {self.question_type.value} (confidence={confidence:.2f})")

    def set_prompts(self, system_prompt: str, user_prompt: str):
        """프롬프트 설정"""
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def set_response(self, response: str, tokens: int = 0):
        """응답 설정"""
        self.response = response
        self.response_tokens = tokens
        self.stage = ProcessingStage.COMPLETED
        self.metrics.mark_generation_complete()

    def set_error(self, error: str):
        """에러 설정"""
        self.error = error
        self.stage = ProcessingStage.ERROR
        logger.error(f"Context error: {error}")

    def mark_retrieval_complete(self):
        """검색 완료 마킹"""
        self.stage = ProcessingStage.RETRIEVAL
        self.metrics.mark_retrieval_complete()

    # === 메타데이터 ===

    def add_metadata(self, key: str, value: Any):
        """메타데이터 추가"""
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """메타데이터 조회"""
        return self.metadata.get(key, default)

    # === 대화 히스토리 ===

    def add_to_history(self, role: str, content: str):
        """대화 히스토리에 추가"""
        self.conversation_history.append(
            ConversationMessage(role=role, content=content)
        )

    def get_history_as_list(self) -> List[Dict[str, str]]:
        """히스토리를 리스트로 변환"""
        return [msg.to_dict() for msg in self.conversation_history]

    def get_recent_history(self, n: int = 5) -> List[ConversationMessage]:
        """최근 n개 히스토리"""
        return self.conversation_history[-n:]

    # === 직렬화 ===

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'brand_id': self.brand_id,
            'question': self.question,
            'question_type': self.question_type.value,
            'intent': self.intent,
            'confidence': self.confidence,
            'entities': self.entities,
            'retrieval_results': [r.to_dict() for r in self.retrieval_results],
            'response': self.response,
            'response_tokens': self.response_tokens,
            'stage': self.stage.value,
            'error': self.error,
            'metrics': self.metrics.to_dict(),
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'cache_key': self.cache_key,
        }

    def to_json(self) -> str:
        """JSON 문자열로 변환"""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryContext':
        """딕셔너리에서 생성"""
        # 기본 필드
        context = cls(
            brand_id=data['brand_id'],
            question=data['question'],
        )

        # 선택적 필드
        if 'question_type' in data:
            context.question_type = QuestionType(data['question_type'])
        if 'intent' in data:
            context.intent = data['intent']
        if 'confidence' in data:
            context.confidence = data['confidence']
        if 'entities' in data:
            context.entities = data['entities']
        if 'response' in data:
            context.response = data['response']
        if 'metadata' in data:
            context.metadata = data['metadata']

        return context

    @classmethod
    def from_json(cls, json_str: str) -> 'QueryContext':
        """JSON 문자열에서 생성"""
        return cls.from_dict(json.loads(json_str))

    # === 요약 ===

    def get_summary(self) -> Dict[str, Any]:
        """컨텍스트 요약"""
        return {
            'brand_id': self.brand_id,
            'question': self.question[:100] + '...' if len(self.question) > 100 else self.question,
            'question_type': self.question_type.value,
            'confidence': self.confidence,
            'stage': self.stage.value,
            'retrieval_count': self.get_total_retrieval_count(),
            'retrieval_sources': self.get_retrieval_sources(),
            'has_response': self.response is not None,
            'response_tokens': self.response_tokens,
            'metrics': self.metrics.to_dict(),
            'has_error': self.error is not None,
        }

    def __repr__(self) -> str:
        return (
            f"QueryContext(brand={self.brand_id}, "
            f"type={self.question_type.value}, "
            f"stage={self.stage.value}, "
            f"retrievals={self.get_total_retrieval_count()})"
        )
