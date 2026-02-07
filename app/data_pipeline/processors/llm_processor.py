"""
LLM Knowledge Graph Processor - Production Grade v4.0

ONTIX Universal 범용 SNS 데이터 파이프라인을 위한 프로덕션급 LLM 프로세서.
LangChain의 LLMGraphTransformer를 사용하여 SNS 데이터에서 지식그래프를 생성.

Supported Platforms:
    - Instagram
    - YouTube
    - TikTok
    - Twitter

4-Layer Knowledge Graph Structure:
    Brand → Concept → Content → Interaction

Author: ONTIX Universal Team
Version: 4.0.0
"""

from __future__ import annotations

import os
import re
import time
import logging
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    Union,
    cast,
)

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Optional: LLMGraphTransformer (may not be available in all langchain-experimental versions)
try:
    from langchain_experimental.graph_transformers import LLMGraphTransformer
except ImportError:
    LLMGraphTransformer = None  # type: ignore

from ..domain.models import (
    ActorDTO,
    ContentDTO,
    InteractionDTO,
    PlatformType,
    TopicDTO,
)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS & CONFIGURATIONS
# ============================================================================

class ModelConfig:
    """LLM 모델 설정 상수"""

    # Default Model - gpt-5-mini 사용
    DEFAULT_MODEL: Final[str] = "gpt-5-mini"
    DEFAULT_TEMPERATURE: Final[float] = 0.0  # GPT-5 모델에서는 무시됨

    # GPT-5 모델 prefix (temperature/top_p 파라미터 제거 대상)
    GPT5_MODEL_PREFIX: Final[str] = "gpt-5"

    # Retry Settings
    MAX_RETRIES: Final[int] = 3
    RETRY_DELAY_SECONDS: Final[float] = 5.0

    # Cost Estimation (per 1K tokens, USD) - gpt-5-mini 기준
    COST_PER_1K_INPUT: Final[float] = 0.00015
    COST_PER_1K_OUTPUT: Final[float] = 0.0006

    @classmethod
    def is_gpt5_model(cls, model: str) -> bool:
        """GPT-5 계열 모델인지 확인 (temperature/top_p 제거 필요)"""
        return model.startswith(cls.GPT5_MODEL_PREFIX)


class GraphNodeTypes:
    """그래프 노드 타입 상수"""

    BRAND: Final[str] = "Brand"
    CONCEPT: Final[str] = "Concept"
    CONTENT: Final[str] = "Content"
    INTERACTION: Final[str] = "Interaction"
    ACTOR: Final[str] = "Actor"
    TOPIC: Final[str] = "Topic"

    @classmethod
    def all_types(cls) -> List[str]:
        """모든 노드 타입 반환"""
        return [cls.BRAND, cls.CONCEPT, cls.CONTENT, cls.INTERACTION, cls.ACTOR, cls.TOPIC]


class GraphRelationshipTypes:
    """그래프 관계 타입 상수"""

    # Brand relationships
    HAS_CONCEPT: Final[str] = "HAS_CONCEPT"
    HAS_CONTENT: Final[str] = "HAS_CONTENT"

    # Content relationships
    MENTIONS_CONCEPT: Final[str] = "MENTIONS_CONCEPT"
    HAS_INTERACTION: Final[str] = "HAS_INTERACTION"
    CREATED_BY: Final[str] = "CREATED_BY"

    # Concept relationships
    RELATED_TO: Final[str] = "RELATED_TO"
    CAUSES: Final[str] = "CAUSES"
    LEADS_TO: Final[str] = "LEADS_TO"
    STEP_OF_ROUTINE: Final[str] = "STEP_OF_ROUTINE"
    TARGETS_PROBLEM: Final[str] = "TARGETS_PROBLEM"
    PROVIDES_EFFECT: Final[str] = "PROVIDES_EFFECT"

    # Interaction relationships
    EXPRESSES_SENTIMENT: Final[str] = "EXPRESSES_SENTIMENT"
    REPLY_TO: Final[str] = "REPLY_TO"

    # Actor relationships
    AUTHORED: Final[str] = "AUTHORED"
    INTERACTED: Final[str] = "INTERACTED"

    @classmethod
    def all_types(cls) -> List[str]:
        """모든 관계 타입 반환"""
        return [
            cls.HAS_CONCEPT, cls.HAS_CONTENT, cls.MENTIONS_CONCEPT, cls.HAS_INTERACTION,
            cls.CREATED_BY, cls.RELATED_TO, cls.CAUSES, cls.LEADS_TO, cls.STEP_OF_ROUTINE,
            cls.TARGETS_PROBLEM, cls.PROVIDES_EFFECT, cls.EXPRESSES_SENTIMENT,
            cls.REPLY_TO, cls.AUTHORED, cls.INTERACTED,
        ]


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

class ProcessingStatistics(TypedDict):
    """처리 통계 타입 정의"""

    total_contents_processed: int
    total_interactions_processed: int
    total_nodes_created: int
    total_relationships_created: int
    total_cost_estimate_usd: float
    total_processing_time_seconds: float
    errors_count: int
    retries_count: int


class BrandConfig(TypedDict, total=False):
    """브랜드 설정 타입 정의"""

    id: str
    name: str
    name_en: str
    category: str
    philosophy: str
    core_concepts: List[str]
    description: str
    tone: str
    keywords: List[str]


class PlatformMetrics(TypedDict, total=False):
    """플랫폼별 메트릭 타입 정의"""

    # Common
    like_count: int
    comment_count: int
    share_count: int

    # Instagram specific
    saves_count: int

    # YouTube specific
    view_count: int
    dislike_count: int
    subscriber_count: int

    # TikTok specific
    play_count: int

    # Twitter specific
    retweet_count: int
    quote_count: int
    reply_count: int


# ============================================================================
# CONTENT ID EXTRACTION
# ============================================================================

class ContentIdExtractor:
    """
    플랫폼별 콘텐츠 ID 추출기.

    각 SNS 플랫폼의 URL 패턴에서 고유 콘텐츠 ID를 추출합니다.

    Supported URL Patterns:
        - Instagram: /p/ABC123/, /reel/ABC123/
        - YouTube: /watch?v=XYZ456, youtu.be/XYZ456, /shorts/XYZ456
        - TikTok: /video/123456789, /@user/video/123456789
        - Twitter: /status/987654321
    """

    # URL 패턴 정규식
    _PATTERNS: Dict[PlatformType, List[re.Pattern]] = {
        PlatformType.INSTAGRAM: [
            re.compile(r'/p/([A-Za-z0-9_-]+)'),
            re.compile(r'/reel/([A-Za-z0-9_-]+)'),
            re.compile(r'/tv/([A-Za-z0-9_-]+)'),
        ],
        PlatformType.YOUTUBE: [
            re.compile(r'[?&]v=([A-Za-z0-9_-]{11})'),
            re.compile(r'youtu\.be/([A-Za-z0-9_-]{11})'),
            re.compile(r'/shorts/([A-Za-z0-9_-]{11})'),
            re.compile(r'/embed/([A-Za-z0-9_-]{11})'),
            re.compile(r'/v/([A-Za-z0-9_-]{11})'),
        ],
        PlatformType.TIKTOK: [
            re.compile(r'/video/(\d+)'),
            re.compile(r'/@[\w.]+/video/(\d+)'),
        ],
        PlatformType.TWITTER: [
            re.compile(r'/status/(\d+)'),
            re.compile(r'/statuses/(\d+)'),
        ],
    }

    @classmethod
    def extract(cls, url: str, platform: PlatformType) -> str:
        """
        URL에서 플랫폼별 콘텐츠 ID를 추출합니다.

        Args:
            url: 콘텐츠 URL
            platform: 플랫폼 타입

        Returns:
            추출된 콘텐츠 ID. 추출 실패 시 URL 해시 기반 대체 ID 반환.

        Examples:
            >>> ContentIdExtractor.extract(
            ...     "https://www.instagram.com/p/ABC123xyz/",
            ...     PlatformType.INSTAGRAM
            ... )
            'ABC123xyz'

            >>> ContentIdExtractor.extract(
            ...     "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            ...     PlatformType.YOUTUBE
            ... )
            'dQw4w9WgXcQ'
        """
        if not url:
            return cls._generate_fallback_id(url, platform)

        patterns = cls._PATTERNS.get(platform, [])

        for pattern in patterns:
            match = pattern.search(url)
            if match:
                content_id = match.group(1)
                logger.debug(f"Extracted content ID '{content_id}' from URL using pattern {pattern.pattern}")
                return content_id

        # 패턴 매칭 실패 시 URL 해시 기반 ID 생성
        logger.warning(
            f"Could not extract content ID from URL '{url}' for platform {platform.value}. "
            f"Using fallback hash-based ID."
        )
        return cls._generate_fallback_id(url, platform)

    @classmethod
    def _generate_fallback_id(cls, url: str, platform: PlatformType) -> str:
        """URL 해시 기반 대체 ID 생성"""
        url_hash = hashlib.md5((url or "").encode()).hexdigest()[:12]
        return f"{platform.value}_{url_hash}"

    @classmethod
    def validate_content_id(cls, content_id: str, platform: PlatformType) -> bool:
        """
        콘텐츠 ID 형식 유효성 검증.

        Args:
            content_id: 검증할 콘텐츠 ID
            platform: 플랫폼 타입

        Returns:
            유효하면 True, 그렇지 않으면 False
        """
        if not content_id:
            return False

        validation_patterns = {
            PlatformType.INSTAGRAM: r'^[A-Za-z0-9_-]{6,}$',
            PlatformType.YOUTUBE: r'^[A-Za-z0-9_-]{11}$',
            PlatformType.TIKTOK: r'^\d{15,}$',
            PlatformType.TWITTER: r'^\d{10,}$',
        }

        pattern = validation_patterns.get(platform)
        if pattern:
            return bool(re.match(pattern, content_id))

        return True


# ============================================================================
# PLATFORM METRICS MAPPING
# ============================================================================

class PlatformMetricsMapper:
    """
    플랫폼별 메트릭 매핑기.

    각 플랫폼의 고유 메트릭 필드명을 통합 스키마로 매핑합니다.
    """

    # 플랫폼별 메트릭 필드 매핑
    _METRIC_MAPPINGS: Dict[PlatformType, Dict[str, str]] = {
        PlatformType.INSTAGRAM: {
            "likesCount": "like_count",
            "likes": "like_count",
            "commentsCount": "comment_count",
            "comments_count": "comment_count",
            "savesCount": "saves_count",
            "sharesCount": "share_count",
            "videoViewCount": "view_count",
            "viewCount": "view_count",
        },
        PlatformType.YOUTUBE: {
            "viewCount": "view_count",
            "views": "view_count",
            "likeCount": "like_count",
            "likes": "like_count",
            "dislikeCount": "dislike_count",
            "commentCount": "comment_count",
            "comments": "comment_count",
            "subscriberCount": "subscriber_count",
        },
        PlatformType.TIKTOK: {
            "playCount": "play_count",
            "plays": "play_count",
            "diggCount": "like_count",
            "likes": "like_count",
            "shareCount": "share_count",
            "shares": "share_count",
            "commentCount": "comment_count",
            "comments": "comment_count",
        },
        PlatformType.TWITTER: {
            "retweetCount": "retweet_count",
            "retweets": "retweet_count",
            "likeCount": "like_count",
            "likes": "like_count",
            "quoteCount": "quote_count",
            "quotes": "quote_count",
            "replyCount": "reply_count",
            "replies": "reply_count",
        },
    }

    # 플랫폼별 기본 메트릭 키
    _DEFAULT_METRICS: Dict[PlatformType, List[str]] = {
        PlatformType.INSTAGRAM: ["like_count", "comment_count", "share_count", "view_count"],
        PlatformType.YOUTUBE: ["view_count", "like_count", "comment_count"],
        PlatformType.TIKTOK: ["play_count", "like_count", "share_count", "comment_count"],
        PlatformType.TWITTER: ["retweet_count", "like_count", "reply_count", "quote_count"],
    }

    @classmethod
    def map_metrics(
        cls,
        raw_metrics: Dict[str, Any],
        platform: PlatformType,
    ) -> PlatformMetrics:
        """
        원시 메트릭 데이터를 통합 스키마로 매핑합니다.

        Args:
            raw_metrics: 원시 메트릭 딕셔너리
            platform: 플랫폼 타입

        Returns:
            통합 메트릭 딕셔너리
        """
        mapped_metrics: PlatformMetrics = {}
        mappings = cls._METRIC_MAPPINGS.get(platform, {})

        for raw_key, value in raw_metrics.items():
            mapped_key = mappings.get(raw_key, raw_key)
            if isinstance(value, (int, float)):
                mapped_metrics[mapped_key] = int(value)

        return mapped_metrics

    @classmethod
    def get_display_metrics(
        cls,
        content: ContentDTO,
    ) -> Dict[str, int]:
        """
        콘텐츠 DTO에서 표시용 메트릭을 추출합니다.

        Args:
            content: ContentDTO 객체

        Returns:
            메트릭 딕셔너리
        """
        platform = content.platform
        metrics: Dict[str, int] = {}

        # 공통 메트릭
        metrics["like_count"] = content.like_count
        metrics["comment_count"] = content.comment_count
        metrics["share_count"] = content.share_count
        metrics["view_count"] = content.view_count

        # 메타데이터에서 추가 메트릭 추출
        if content.metadata:
            raw_mapped = cls.map_metrics(content.metadata, platform)
            metrics.update(raw_mapped)

        return metrics

    @classmethod
    def format_metrics_for_prompt(
        cls,
        content: ContentDTO,
    ) -> str:
        """
        프롬프트용 메트릭 문자열을 생성합니다.

        Args:
            content: ContentDTO 객체

        Returns:
            포맷된 메트릭 문자열
        """
        platform = content.platform
        metrics = cls.get_display_metrics(content)
        default_keys = cls._DEFAULT_METRICS.get(platform, [])

        lines = []
        for key in default_keys:
            value = metrics.get(key, 0)
            display_name = key.replace("_", " ").title()
            lines.append(f"- {display_name}: {value:,}")

        return "\n".join(lines)


# ============================================================================
# PROMPT BUILDER
# ============================================================================

class PromptBuilder:
    """
    동적 프롬프트 생성기.

    플랫폼과 브랜드 설정에 따라 LLM 프롬프트를 동적으로 생성합니다.
    엔티티 추출 프롬프트는 외부 파일에서 로드합니다.
    """

    _cached_prompt: Optional[str] = None

    @classmethod
    def _load_prompt_file(cls) -> str:
        """외부 프롬프트 파일 로드 (캐싱)"""
        if cls._cached_prompt is not None:
            return cls._cached_prompt

        prompt_path = os.getenv(
            "ENTITY_EXTRACTION_PROMPT_PATH",
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "prompts", "default_extraction.txt"),
        )
        prompt_path = os.path.abspath(prompt_path)

        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                cls._cached_prompt = f.read().strip()
            logger.info(f"Loaded extraction prompt from: {prompt_path}")
        except FileNotFoundError:
            logger.warning(f"Prompt file not found: {prompt_path}. Using built-in default.")
            cls._cached_prompt = cls._default_prompt()

        return cls._cached_prompt

    @staticmethod
    def _default_prompt() -> str:
        """파일이 없을 때 사용하는 최소 기본 프롬프트"""
        return (
            "You are a knowledge graph builder. "
            "Extract entities and relationships from the given social media text.\n\n"
            "Create nodes with labels: Brand, Concept, Content, Interaction.\n"
            "Create relationships: HAS_CONCEPT, HAS_CONTENT, MENTIONS_CONCEPT, "
            "HAS_INTERACTION, CREATED_BY, RELATED_TO.\n\n"
            "Every node MUST have a 'brand_id' property extracted from the "
            "'=== BRAND INFORMATION ===' section in the input.\n"
            "Content ID must be extracted from the URL, not generated."
        )

    @staticmethod
    def build_system_prompt() -> str:
        """시스템 프롬프트 생성 (외부 파일에서 로드)"""
        return PromptBuilder._load_prompt_file()

    @classmethod
    def build_brand_header(cls, brand_config: BrandConfig) -> str:
        """브랜드 정보 헤더 생성"""
        brand_id = brand_config.get("id", "unknown")
        brand_name = brand_config.get("name", "Unknown Brand")
        brand_name_en = brand_config.get("name_en", "")
        category = brand_config.get("category", "")
        philosophy = brand_config.get("philosophy", "")
        core_concepts = brand_config.get("core_concepts", [])
        description = brand_config.get("description", "")
        tone = brand_config.get("tone", "")
        keywords = brand_config.get("keywords", [])

        return f"""
=== BRAND INFORMATION ===
Brand ID: {brand_id}
Brand Name: {brand_name}
Brand Name (English): {brand_name_en}
Category: {category}
Philosophy: {philosophy}
Core Concepts: {', '.join(core_concepts) if core_concepts else 'N/A'}
Keywords: {', '.join(keywords) if keywords else 'N/A'}
Description: {description}
Tone: {tone}
=========================

"""

    @classmethod
    def build_content_block(
        cls,
        content: ContentDTO,
        interactions: List[InteractionDTO],
        index: int,
        brand_config: BrandConfig,
    ) -> str:
        """콘텐츠 블록 생성"""
        platform = content.platform.value.upper()
        content_type = content.content_type.value if content.content_type else "content"

        # 콘텐츠 ID 추출
        content_id = ContentIdExtractor.extract(content.url, content.platform)
        if content.content_id and ContentIdExtractor.validate_content_id(content.content_id, content.platform):
            content_id = content.content_id

        # 메트릭 포맷
        metrics_str = PlatformMetricsMapper.format_metrics_for_prompt(content)

        # 해시태그
        hashtags_str = ", ".join(content.hashtags) if content.hashtags else "N/A"

        # 멘션
        mentions_str = ", ".join(content.mentions) if content.mentions else "N/A"

        # 콘텐츠 블록 생성
        block = f"""
=== {platform} {content_type.upper()} #{index} ===
Platform: {platform.lower()}
Content Type: {content_type}
Account: {content.author.username if content.author else 'unknown'}
URL: {content.url}
CONTENT ID: {content_id}
Created: {content.created_at.isoformat() if content.created_at else 'N/A'}

CONTENT TEXT:
{content.text or '(No text content)'}

HASHTAGS: {hashtags_str}
MENTIONS: {mentions_str}

METRICS:
{metrics_str}
"""

        # 인터랙션 추가
        content_interactions = [i for i in interactions if i.content_id == content.content_id]

        if content_interactions:
            block += f"\nINTERACTIONS ({len(content_interactions)}):\n"
            for i_idx, interaction in enumerate(content_interactions[:20], 1):
                interaction_id = f"interaction_{content_id}_{i_idx:03d}"
                block += f"""
Interaction #{i_idx}:
- ID: {interaction_id}
- Author: {interaction.author.username if interaction.author else 'anonymous'}
- Text: {interaction.text}
- Likes: {interaction.like_count}
"""
                if interaction.parent_id:
                    block += f"- Reply To: {interaction.parent_id}\n"
        else:
            block += "\n(No interactions)\n"

        block += f"\n=== END OF {platform} {content_type.upper()} #{index} ===\n"

        return block

    @classmethod
    def build_topic_context(cls, topics: List[TopicDTO]) -> str:
        """토픽 컨텍스트 생성"""
        if not topics:
            return ""

        topic_lines = []
        for topic in topics:
            category_str = f" ({topic.category})" if topic.category else ""
            confidence_str = f" [confidence: {topic.confidence:.2f}]" if topic.confidence < 1.0 else ""
            topic_lines.append(f"- {topic.name}{category_str}{confidence_str}")

        return f"""
=== PRE-EXTRACTED TOPICS ===
{chr(10).join(topic_lines)}
===========================

"""


# ============================================================================
# GRAPH DOCUMENT PROCESSOR
# ============================================================================

@dataclass
class GraphProcessingResult:
    """그래프 처리 결과"""

    graph_documents: List[Any]
    nodes_count: int = 0
    relationships_count: int = 0
    processing_time_seconds: float = 0.0
    cost_estimate_usd: float = 0.0
    node_types: Dict[str, int] = field(default_factory=dict)
    relationship_types: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    retries: int = 0


class BrandIdInjector:
    """
    브랜드 ID 안전장치.

    모든 노드와 관계에 brand_id를 강제 주입하여 데이터 무결성을 보장합니다.
    """

    @staticmethod
    def inject_to_nodes(graph_documents: List[Any], brand_id: str) -> None:
        """모든 노드에 brand_id 주입"""
        for doc in graph_documents:
            if not hasattr(doc, 'nodes'):
                continue

            for node in doc.nodes:
                if not hasattr(node, 'properties'):
                    node.properties = {}
                node.properties['brand_id'] = brand_id

    @staticmethod
    def inject_to_relationships(graph_documents: List[Any], brand_id: str) -> None:
        """모든 관계의 source/target에 brand_id 주입"""
        for doc in graph_documents:
            if not hasattr(doc, 'relationships'):
                continue

            for rel in doc.relationships:
                # Source node
                if hasattr(rel, 'source'):
                    if not hasattr(rel.source, 'properties'):
                        rel.source.properties = {}
                    rel.source.properties['brand_id'] = brand_id

                # Target node
                if hasattr(rel, 'target'):
                    if not hasattr(rel.target, 'properties'):
                        rel.target.properties = {}
                    rel.target.properties['brand_id'] = brand_id

    @classmethod
    def inject_all(cls, graph_documents: List[Any], brand_id: str) -> None:
        """
        모든 그래프 문서에 brand_id 강제 주입.

        Args:
            graph_documents: LLMGraphTransformer 결과 문서 리스트
            brand_id: 주입할 브랜드 ID
        """
        if not brand_id:
            logger.warning("Brand ID is empty. Skipping injection.")
            return

        cls.inject_to_nodes(graph_documents, brand_id)
        cls.inject_to_relationships(graph_documents, brand_id)

        logger.debug(f"Injected brand_id '{brand_id}' to all nodes and relationships")


class GraphStatisticsCollector:
    """그래프 통계 수집기"""

    @staticmethod
    def collect(graph_documents: List[Any]) -> Tuple[int, int, Dict[str, int], Dict[str, int]]:
        """
        그래프 문서에서 통계 수집.

        Args:
            graph_documents: 그래프 문서 리스트

        Returns:
            (total_nodes, total_rels, node_types, rel_types) 튜플
        """
        total_nodes = 0
        total_rels = 0
        node_types: Dict[str, int] = {}
        rel_types: Dict[str, int] = {}

        for doc in graph_documents:
            if hasattr(doc, 'nodes'):
                for node in doc.nodes:
                    total_nodes += 1
                    ntype = getattr(node, 'type', 'Unknown')
                    node_types[ntype] = node_types.get(ntype, 0) + 1

            if hasattr(doc, 'relationships'):
                for rel in doc.relationships:
                    total_rels += 1
                    rtype = getattr(rel, 'type', 'Unknown')
                    rel_types[rtype] = rel_types.get(rtype, 0) + 1

        return total_nodes, total_rels, node_types, rel_types


# ============================================================================
# COST ESTIMATOR
# ============================================================================

class CostEstimator:
    """LLM API 비용 추정기"""

    CHARS_PER_TOKEN: Final[float] = 4.0  # 평균 토큰당 문자 수

    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        """텍스트의 토큰 수 추정"""
        if not text:
            return 0
        return int(len(text) / cls.CHARS_PER_TOKEN)

    @classmethod
    def estimate_cost(
        cls,
        input_text: str,
        output_tokens_estimate: int,
        model: str = None,
    ) -> float:
        """
        API 호출 비용 추정 (gpt-5-mini 기준).

        Args:
            input_text: 입력 텍스트
            output_tokens_estimate: 예상 출력 토큰 수
            model: 모델명 (무시됨, gpt-5-mini 고정)

        Returns:
            예상 비용 (USD)
        """
        input_tokens = cls.estimate_tokens(input_text)

        input_cost = (input_tokens / 1000) * ModelConfig.COST_PER_1K_INPUT
        output_cost = (output_tokens_estimate / 1000) * ModelConfig.COST_PER_1K_OUTPUT

        return input_cost + output_cost


# ============================================================================
# MAIN PROCESSOR CLASS
# ============================================================================

class LLMProcessor:
    """
    프로덕션급 LLM 지식그래프 프로세서.

    ONTIX Universal의 범용 SNS 데이터 파이프라인을 위한 LLM 기반 지식그래프 생성기입니다.
    LangChain의 LLMGraphTransformer를 사용하여 다양한 SNS 플랫폼의 데이터를
    4계층 지식그래프(Brand → Concept → Content → Interaction)로 변환합니다.

    Attributes:
        llm: ChatOpenAI 인스턴스
        transformer: LLMGraphTransformer 인스턴스
        model: 사용 중인 모델명
        statistics: 누적 처리 통계

    Example:
        >>> processor = LLMProcessor(model="gpt-5-mini")
        >>> result = processor.process_contents(
        ...     actors=actors,
        ...     contents=contents,
        ...     interactions=interactions,
        ...     topics=topics,
        ...     brand_config={"id": "my_brand", "name": "My Brand"}
        ... )
        >>> print(f"Created {result.nodes_count} nodes")
    """

    def __init__(
        self,
        model: str = None,  # 무시됨, gpt-5-mini 고정
        temperature: float = ModelConfig.DEFAULT_TEMPERATURE,
        api_key: Optional[str] = None,
        max_retries: int = ModelConfig.MAX_RETRIES,
        retry_delay: float = ModelConfig.RETRY_DELAY_SECONDS,
    ) -> None:
        """
        LLM 프로세서 초기화.

        Args:
            model: 무시됨 (gpt-5-mini 고정 사용)
            temperature: 생성 온도 (0.0 ~ 1.0). 낮을수록 결정적.
            api_key: OpenAI API 키. None이면 환경변수에서 로드.
            max_retries: 최대 재시도 횟수.
            retry_delay: 재시도 간 대기 시간(초).

        Raises:
            ValueError: API 키가 없거나 유효하지 않은 경우.
            RuntimeError: LLM 초기화 실패 시.
        """
        # gpt-5-mini 고정 사용
        self.model = ModelConfig.DEFAULT_MODEL
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # API 키 검증
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        if not resolved_api_key.startswith("sk-"):
            raise ValueError(
                "Invalid OpenAI API key format. "
                "API key should start with 'sk-'."
            )

        logger.info(f"Initializing LLMProcessor with model: {self.model}")

        try:
            # LLM 초기화
            # GPT-5 모델은 temperature/top_p 파라미터를 지원하지 않음
            llm_kwargs = {
                "model": self.model,
                "api_key": resolved_api_key,
            }

            # GPT-5 계열이 아닌 경우에만 temperature 추가
            if not ModelConfig.is_gpt5_model(self.model):
                llm_kwargs["temperature"] = temperature

            self.llm = ChatOpenAI(**llm_kwargs)

            # 프롬프트 생성
            custom_prompt = ChatPromptTemplate.from_messages([
                ("system", PromptBuilder.build_system_prompt()),
                ("human", "Extract knowledge graph from the following text:\n\n{input}"),
            ])

            # Transformer 초기화
            if LLMGraphTransformer is None:
                logger.warning("LLMGraphTransformer not available, graph transformation disabled")
                self.transformer = None
            else:
                self.transformer = LLMGraphTransformer(
                    llm=self.llm,
                    prompt=custom_prompt,
                    allowed_nodes=GraphNodeTypes.all_types(),
                    allowed_relationships=GraphRelationshipTypes.all_types(),
                    node_properties=[
                        "name", "brand_id", "platform", "content_type",
                        "description", "type", "canonical_name", "synonyms",
                        "url", "text", "created_at", "metrics",
                        "author", "philosophy", "category", "sentiment",
                        "confidence", "hashtags", "mentions",
                    ],
                    relationship_properties=["confidence", "polarity", "score", "timestamp"],
                    strict_mode=False,
                )

            # 통계 초기화
            self._statistics: ProcessingStatistics = {
                "total_contents_processed": 0,
                "total_interactions_processed": 0,
                "total_nodes_created": 0,
                "total_relationships_created": 0,
                "total_cost_estimate_usd": 0.0,
                "total_processing_time_seconds": 0.0,
                "errors_count": 0,
                "retries_count": 0,
            }

            logger.info(f"LLMProcessor initialized successfully. Model: {model}")

        except Exception as e:
            logger.error(f"Failed to initialize LLMProcessor: {e}")
            raise RuntimeError(f"LLM initialization failed: {e}") from e

    @property
    def statistics(self) -> ProcessingStatistics:
        """누적 처리 통계 반환"""
        return self._statistics.copy()

    def reset_statistics(self) -> None:
        """통계 초기화"""
        self._statistics = {
            "total_contents_processed": 0,
            "total_interactions_processed": 0,
            "total_nodes_created": 0,
            "total_relationships_created": 0,
            "total_cost_estimate_usd": 0.0,
            "total_processing_time_seconds": 0.0,
            "errors_count": 0,
            "retries_count": 0,
        }
        logger.info("Processing statistics reset")

    def process_contents(
        self,
        actors: List[ActorDTO],
        contents: List[ContentDTO],
        interactions: List[InteractionDTO],
        topics: List[TopicDTO],
        brand_config: BrandConfig,
    ) -> GraphProcessingResult:
        """
        SNS 콘텐츠를 지식그래프로 변환합니다.

        DTO 기반의 범용 인터페이스로, 모든 지원 플랫폼(Instagram, YouTube, TikTok, Twitter)의
        데이터를 통합 처리합니다.

        Args:
            actors: 크리에이터/사용자 DTO 리스트
            contents: 콘텐츠 DTO 리스트
            interactions: 인터랙션(댓글/답글) DTO 리스트
            topics: 사전 추출된 토픽 DTO 리스트
            brand_config: 브랜드 설정 딕셔너리. 필수 키:
                - id: 브랜드 고유 ID
                - name: 브랜드명

        Returns:
            GraphProcessingResult: 처리 결과 객체

        Raises:
            ValueError: 필수 파라미터가 누락된 경우

        Example:
            >>> result = processor.process_contents(
            ...     actors=[ActorDTO(...)],
            ...     contents=[ContentDTO(...)],
            ...     interactions=[InteractionDTO(...)],
            ...     topics=[TopicDTO(...)],
            ...     brand_config={"id": "my_brand", "name": "My Brand"}
            ... )
        """
        if not contents:
            logger.warning("No contents to process")
            return GraphProcessingResult(graph_documents=[])

        brand_id = brand_config.get("id", "unknown")
        brand_name = brand_config.get("name", "Unknown Brand")

        logger.info(f"Processing {len(contents)} contents for brand: {brand_name} ({brand_id})")
        logger.info(f"  Actors: {len(actors)}, Interactions: {len(interactions)}, Topics: {len(topics)}")

        # 플랫폼별 콘텐츠 분류
        platform_counts = self._count_by_platform(contents)
        for platform, count in platform_counts.items():
            logger.info(f"  {platform}: {count} contents")

        # 텍스트 문서 생성
        documents = self._build_documents(
            contents=contents,
            interactions=interactions,
            topics=topics,
            brand_config=brand_config,
        )

        if not documents:
            logger.warning("No valid documents generated")
            return GraphProcessingResult(graph_documents=[])

        # 그래프 변환 (재시도 로직 포함)
        result = self._process_with_retry(documents, brand_id)

        # 통계 업데이트
        self._update_statistics(result, len(contents), len(interactions))

        # 결과 로깅
        self._log_result(result, brand_id)

        return result

    def _count_by_platform(self, contents: List[ContentDTO]) -> Dict[str, int]:
        """플랫폼별 콘텐츠 수 집계"""
        counts: Dict[str, int] = {}
        for content in contents:
            platform_name = content.platform.value
            counts[platform_name] = counts.get(platform_name, 0) + 1
        return counts

    def _build_documents(
        self,
        contents: List[ContentDTO],
        interactions: List[InteractionDTO],
        topics: List[TopicDTO],
        brand_config: BrandConfig,
    ) -> List[Document]:
        """LangChain Document 리스트 생성"""
        documents = []
        brand_id = brand_config.get("id", "unknown")

        # 브랜드 헤더
        brand_header = PromptBuilder.build_brand_header(brand_config)

        # 토픽 컨텍스트
        topic_context = PromptBuilder.build_topic_context(topics)

        for idx, content in enumerate(contents, 1):
            try:
                # 콘텐츠 블록 생성
                content_block = PromptBuilder.build_content_block(
                    content=content,
                    interactions=interactions,
                    index=idx,
                    brand_config=brand_config,
                )

                # 전체 텍스트 조합
                full_text = brand_header + topic_context + content_block

                # 최소 길이 검증
                if len(full_text.strip()) < 50:
                    logger.warning(f"Content {idx} text too short, skipping")
                    continue

                # Document 생성
                document = Document(
                    page_content=full_text,
                    metadata={
                        "brand_id": brand_id,
                        "content_id": content.content_id,
                        "platform": content.platform.value,
                        "index": idx,
                    }
                )
                documents.append(document)

            except Exception as e:
                logger.error(f"Failed to build document for content {idx}: {e}")
                continue

        logger.info(f"Built {len(documents)} documents from {len(contents)} contents")
        return documents

    def _process_with_retry(
        self,
        documents: List[Document],
        brand_id: str,
    ) -> GraphProcessingResult:
        """재시도 로직이 포함된 그래프 처리"""
        result = GraphProcessingResult(graph_documents=[])

        if self.transformer is None:
            logger.warning("LLMGraphTransformer not available, skipping graph processing")
            result.processing_time_seconds = 0.0
            return result

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Processing attempt {attempt}/{self.max_retries}")
                start_time = time.time()

                # LLM으로 그래프 변환
                graph_documents = self.transformer.convert_to_graph_documents(documents)

                # Brand ID 안전장치 적용
                BrandIdInjector.inject_all(graph_documents, brand_id)

                elapsed_time = time.time() - start_time

                # 통계 수집
                total_nodes, total_rels, node_types, rel_types = GraphStatisticsCollector.collect(
                    graph_documents
                )

                # 결과 검증
                if total_nodes == 0:
                    logger.warning(f"No nodes created on attempt {attempt}")
                    if attempt < self.max_retries:
                        result.retries += 1
                        time.sleep(self.retry_delay)
                        continue

                # 비용 추정
                total_text = "".join(doc.page_content for doc in documents)
                cost_estimate = CostEstimator.estimate_cost(
                    input_text=total_text,
                    output_tokens_estimate=total_nodes * 50,  # 노드당 평균 50토큰 추정
                    model=self.model,
                )

                result.graph_documents = graph_documents
                result.nodes_count = total_nodes
                result.relationships_count = total_rels
                result.processing_time_seconds = elapsed_time
                result.cost_estimate_usd = cost_estimate
                result.node_types = node_types
                result.relationship_types = rel_types

                return result

            except Exception as e:
                error_msg = f"Processing failed on attempt {attempt}: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)
                result.retries += 1

                if attempt < self.max_retries:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"All {self.max_retries} attempts failed")

        return result

    def _update_statistics(
        self,
        result: GraphProcessingResult,
        contents_count: int,
        interactions_count: int,
    ) -> None:
        """누적 통계 업데이트"""
        self._statistics["total_contents_processed"] += contents_count
        self._statistics["total_interactions_processed"] += interactions_count
        self._statistics["total_nodes_created"] += result.nodes_count
        self._statistics["total_relationships_created"] += result.relationships_count
        self._statistics["total_cost_estimate_usd"] += result.cost_estimate_usd
        self._statistics["total_processing_time_seconds"] += result.processing_time_seconds
        self._statistics["errors_count"] += len(result.errors)
        self._statistics["retries_count"] += result.retries

    def _log_result(self, result: GraphProcessingResult, brand_id: str) -> None:
        """처리 결과 로깅"""
        logger.info(f"Processing completed for brand: {brand_id}")
        logger.info(f"  Nodes: {result.nodes_count}, Relationships: {result.relationships_count}")
        logger.info(f"  Time: {result.processing_time_seconds:.2f}s")
        logger.info(f"  Estimated cost: ${result.cost_estimate_usd:.4f}")

        if result.node_types:
            logger.info("  Node types:")
            for ntype, count in sorted(result.node_types.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"    - {ntype}: {count}")

        if result.relationship_types:
            logger.info("  Relationship types:")
            for rtype, count in sorted(result.relationship_types.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"    - {rtype}: {count}")

        # 4계층 구조 검증
        has_brand = GraphNodeTypes.BRAND in result.node_types
        has_concept = GraphNodeTypes.CONCEPT in result.node_types
        has_content = GraphNodeTypes.CONTENT in result.node_types
        has_interaction = GraphNodeTypes.INTERACTION in result.node_types

        logger.info("  4-Layer structure validation:")
        logger.info(f"    Brand: {'OK' if has_brand else 'MISSING'}")
        logger.info(f"    Concept: {'OK' if has_concept else 'MISSING'}")
        logger.info(f"    Content: {'OK' if has_content else 'MISSING'}")
        logger.info(f"    Interaction: {'OK' if has_interaction else 'MISSING'}")

        if result.errors:
            logger.warning(f"  Errors: {len(result.errors)}")
            for error in result.errors[:5]:  # 최대 5개만 표시
                logger.warning(f"    - {error}")

    # ========================================================================
    # LEGACY COMPATIBILITY METHODS
    # ========================================================================

    def process_posts(
        self,
        posts_data: List[Dict[str, Any]],
        brand_config: Optional[Dict[str, Any]] = None,
        brand_id: Optional[str] = None,
    ) -> List[Any]:
        """
        레거시 호환성을 위한 Dict 기반 처리 메서드.

        DEPRECATED: 새 코드에서는 process_contents()를 사용하세요.

        기존 Instagram 전용 인터페이스를 유지하면서 내부적으로
        DTO 기반 처리로 변환합니다.

        Args:
            posts_data: 레거시 포맷의 게시물 데이터 리스트
            brand_config: 브랜드 설정 (우선순위)
            brand_id: 브랜드 ID (brand_config 없을 때 대체)

        Returns:
            그래프 문서 리스트 (LLMGraphTransformer 결과)
        """
        import warnings
        warnings.warn(
            "process_posts() is deprecated. Use process_contents() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if not posts_data:
            logger.warning("No posts to process")
            return []

        # brand_config 구성
        if brand_config:
            _brand_config: BrandConfig = {
                "id": brand_config.get("id", brand_id or "unknown"),
                "name": brand_config.get("name", "Unknown Brand"),
                "name_en": brand_config.get("name_en", ""),
                "category": brand_config.get("category", ""),
                "philosophy": brand_config.get("philosophy", ""),
                "core_concepts": brand_config.get("core_concepts", []),
                "description": brand_config.get("description", ""),
                "tone": brand_config.get("tone", ""),
            }
        else:
            _brand_config = {
                "id": brand_id or "unknown",
                "name": "Unknown Brand",
            }

        # Dict를 DTO로 변환
        actors: List[ActorDTO] = []
        contents: List[ContentDTO] = []
        interactions: List[InteractionDTO] = []

        for post in posts_data:
            if not isinstance(post, dict):
                continue

            try:
                # Actor 생성
                actor = ActorDTO(
                    platform=PlatformType.INSTAGRAM,
                    actor_id=post.get("account", "unknown"),
                    username=post.get("account", "unknown"),
                )
                actors.append(actor)

                # Content 생성
                from ..domain.models import ContentType
                content = ContentDTO(
                    platform=PlatformType.INSTAGRAM,
                    content_id=ContentIdExtractor.extract(
                        post.get("url", ""),
                        PlatformType.INSTAGRAM
                    ),
                    content_type=ContentType.POST,
                    author=actor,
                    text=post.get("content", ""),
                    url=post.get("url", ""),
                    created_at=self._parse_datetime(post.get("timestamp")),
                    like_count=post.get("likes", 0),
                    comment_count=post.get("comments_count", 0),
                )
                contents.append(content)

                # Interactions 생성
                for comment in post.get("comments", []):
                    comment_author = ActorDTO(
                        platform=PlatformType.INSTAGRAM,
                        actor_id=comment.get("author", "user"),
                        username=comment.get("author", "user"),
                    )

                    interaction = InteractionDTO(
                        platform=PlatformType.INSTAGRAM,
                        interaction_id=f"comment_{content.content_id}_{len(interactions)+1}",
                        content_id=content.content_id,
                        author=comment_author,
                        text=comment.get("text", ""),
                    )
                    interactions.append(interaction)

            except Exception as e:
                logger.error(f"Failed to convert post to DTO: {e}")
                continue

        # 새 인터페이스로 처리
        result = self.process_contents(
            actors=actors,
            contents=contents,
            interactions=interactions,
            topics=[],
            brand_config=_brand_config,
        )

        return result.graph_documents

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        """다양한 형식의 datetime 파싱"""
        if value is None:
            return None

        if isinstance(value, datetime):
            return value

        if isinstance(value, str):
            # ISO 포맷 시도
            for fmt in [
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ]:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        레거시 호환성을 위한 통계 반환 메서드.

        DEPRECATED: statistics 프로퍼티를 사용하세요.
        """
        return {
            "total_processed": self._statistics["total_contents_processed"],
            "total_cost_estimate": self._statistics["total_cost_estimate_usd"],
        }

    # ========================================================================
    # ASYNC METHODS FOR PIPELINE INTEGRATION
    # ========================================================================

    async def extract_entities(
        self,
        content: ContentDTO,
        interactions: Optional[List[InteractionDTO]] = None,
    ) -> Dict[str, Any]:
        """
        콘텐츠에서 엔티티 추출 (비동기).

        기존 파이프라인 호환성을 위한 메서드입니다.

        Args:
            content: 콘텐츠 DTO
            interactions: 인터랙션 DTO 리스트 (선택)

        Returns:
            추출된 엔티티 딕셔너리
        """
        logger.info(f"Extracting entities from content: {content.content_id}")

        # 단일 콘텐츠를 process_contents로 처리
        result = self.process_contents(
            actors=[content.author] if content.author else [],
            contents=[content],
            interactions=interactions or [],
            topics=[],
            brand_config={"id": "extract", "name": "Entity Extraction"},
        )

        # 결과에서 엔티티 추출
        entities: Dict[str, List[str]] = {
            "concepts": [],
            "persons": [],
            "organizations": [],
            "locations": [],
            "products": [],
        }

        for doc in result.graph_documents:
            if hasattr(doc, 'nodes'):
                for node in doc.nodes:
                    node_type = getattr(node, 'type', '')
                    node_id = getattr(node, 'id', '')

                    if node_type == GraphNodeTypes.CONCEPT:
                        entities["concepts"].append(node_id)

        return {
            "topics": [TopicDTO(name=c) for c in entities["concepts"]],
            "keywords": entities["concepts"],
            "sentiment": "neutral",  # TODO: 감정 분석 추가
            "summary": "",
            "entities": entities,
        }

    async def generate_knowledge_graph(
        self,
        content: ContentDTO,
        interactions: Optional[List[InteractionDTO]] = None,
        extracted_entities: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        지식그래프 생성 (비동기).

        기존 파이프라인 호환성을 위한 메서드입니다.

        Args:
            content: 콘텐츠 DTO
            interactions: 인터랙션 DTO 리스트 (선택)
            extracted_entities: 사전 추출된 엔티티 (선택)

        Returns:
            노드와 관계를 포함한 딕셔너리
        """
        logger.info(f"Generating knowledge graph for content: {content.content_id}")

        # 토픽 변환
        topics: List[TopicDTO] = []
        if extracted_entities and "topics" in extracted_entities:
            topics = extracted_entities["topics"]

        result = self.process_contents(
            actors=[content.author] if content.author else [],
            contents=[content],
            interactions=interactions or [],
            topics=topics,
            brand_config={"id": "kg_gen", "name": "Knowledge Graph Generation"},
        )

        # LangChain 형식을 표준 형식으로 변환
        nodes = []
        relationships = []

        for doc in result.graph_documents:
            if hasattr(doc, 'nodes'):
                for node in doc.nodes:
                    nodes.append({
                        "type": getattr(node, 'type', 'Unknown'),
                        "properties": getattr(node, 'properties', {}),
                    })

            if hasattr(doc, 'relationships'):
                for rel in doc.relationships:
                    relationships.append({
                        "type": getattr(rel, 'type', 'Unknown'),
                        "from": getattr(rel.source, 'id', '') if hasattr(rel, 'source') else '',
                        "to": getattr(rel.target, 'id', '') if hasattr(rel, 'target') else '',
                        "properties": getattr(rel, 'properties', {}),
                    })

        return {
            "nodes": nodes,
            "relationships": relationships,
        }

    async def analyze_trends(
        self,
        contents: List[ContentDTO],
    ) -> Dict[str, Any]:
        """
        트렌드 분석 (비동기).

        여러 콘텐츠에서 트렌드를 분석합니다.

        Args:
            contents: 콘텐츠 DTO 리스트

        Returns:
            트렌드 분석 결과 딕셔너리
        """
        logger.info(f"Analyzing trends for {len(contents)} contents")

        result = self.process_contents(
            actors=[c.author for c in contents if c.author],
            contents=contents,
            interactions=[],
            topics=[],
            brand_config={"id": "trend", "name": "Trend Analysis"},
        )

        # 개념별 빈도 집계
        concept_counts: Dict[str, int] = {}

        for doc in result.graph_documents:
            if hasattr(doc, 'nodes'):
                for node in doc.nodes:
                    if getattr(node, 'type', '') == GraphNodeTypes.CONCEPT:
                        concept_id = getattr(node, 'id', '')
                        concept_counts[concept_id] = concept_counts.get(concept_id, 0) + 1

        # 상위 트렌드 추출
        sorted_concepts = sorted(
            concept_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]

        trending_topics = [
            TopicDTO(
                name=name,
                confidence=min(count / len(contents), 1.0),
            )
            for name, count in sorted_concepts
        ]

        return {
            "trending_topics": trending_topics,
            "sentiment_distribution": {},  # TODO: 감정 분포 추가
            "engagement_metrics": {
                "total_contents": len(contents),
                "total_concepts_extracted": len(concept_counts),
            },
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_llm_processor(
    **kwargs: Any,
) -> LLMProcessor:
    """
    LLMProcessor 팩토리 함수.

    gpt-5-mini 모델로 프로세서를 생성합니다.

    Args:
        **kwargs: LLMProcessor 추가 인자

    Returns:
        설정된 LLMProcessor 인스턴스 (gpt-5-mini 고정)

    Example:
        >>> processor = create_llm_processor()
    """
    return LLMProcessor(**kwargs)
