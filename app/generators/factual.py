"""
Factual Generator - Production Grade v2.0
사실 기반 응답 생성

Features:
    - 정확한 사실 기반 응답
    - 출처 인용
    - 신뢰도 점수
    - 검색 결과 기반 응답
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
import logging

from app.generators.base import (
    BaseGenerator,
    GeneratorType,
    GeneratorConfig,
    GenerationMetrics,
    ResponseFormatter,
    OutputValidator,
)
from app.core.context import QueryContext

logger = logging.getLogger(__name__)


# ============================================================
# Factual Response Types
# ============================================================

@dataclass
class SourceCitation:
    """출처 인용"""
    source: str
    content: str
    relevance: float = 0.0
    index: int = 0

    def to_markdown(self) -> str:
        return f"[{self.index}] {self.source}: {self.content[:100]}..."


@dataclass
class FactualResponse:
    """사실 응답 구조"""
    answer: str
    confidence: float
    sources: List[SourceCitation] = field(default_factory=list)
    has_sufficient_context: bool = True

    def to_formatted_string(self, include_sources: bool = True) -> str:
        """포맷된 응답 문자열"""
        result = self.answer

        if include_sources and self.sources:
            result += "\n\n---\n**참고 출처:**\n"
            for src in self.sources[:5]:
                result += f"- {src.source}\n"

        return result


# ============================================================
# Factual Generator
# ============================================================

class FactualGenerator(BaseGenerator):
    """
    사실 기반 응답 생성기 - Production Grade

    Features:
        - Temperature: 0 (일관성)
        - 짧고 정확한 응답
        - 검색 결과에만 의존
        - 출처 인용
        - 신뢰도 점수
    """

    # 기본 설정 오버라이드
    DEFAULT_TEMPERATURE = 0.0
    DEFAULT_MODEL = "feature"  # GPT-5-mini for accurate factual responses
    DEFAULT_MAX_TOKENS = 1000

    def __init__(self, brand_config: Dict):
        super().__init__(brand_config, GeneratorType.FACTUAL)

        # Factual 전용 설정 오버라이드
        self.config.temperature = self.DEFAULT_TEMPERATURE
        self.config.model_variant = self.DEFAULT_MODEL
        self.config.max_tokens = self.DEFAULT_MAX_TOKENS

        # 출처 인용 설정
        self.include_sources = self.config.include_sources
        self.min_confidence = 0.3

    def generate(self, context: QueryContext) -> str:
        """
        사실 기반 응답 생성

        Args:
            context: 쿼리 컨텍스트

        Returns:
            생성된 응답
        """
        # 메트릭 시작
        metrics = self._create_metrics(context)

        try:
            # 컨텍스트 충분성 확인
            if not self._has_sufficient_context(context):
                response = self._generate_insufficient_context_response(context)
                metrics.complete()
                self._last_metrics = metrics
                return response

            # 프롬프트 로드
            system_prompt = self._load_system_prompt(context.question_type)
            user_prompt = self._build_user_prompt(context)

            # 프롬프트 저장 (디버깅용)
            context.set_prompts(system_prompt, user_prompt)

            # LLM 호출
            response = self._invoke_llm(user_prompt, system_prompt, metrics)

            # 출처 추가
            if self.include_sources:
                response = self._append_sources(response, context)

            # 메트릭 완료
            metrics.complete()
            self._last_metrics = metrics

            return response

        except Exception as e:
            logger.error(f"Factual generation error: {e}")
            metrics.complete(success=False, error=str(e))
            self._last_metrics = metrics

            return "죄송합니다. 정확한 정보를 제공할 수 없습니다."

    def generate_with_confidence(self, context: QueryContext) -> FactualResponse:
        """
        신뢰도 점수 포함 응답 생성

        Args:
            context: 쿼리 컨텍스트

        Returns:
            FactualResponse 객체
        """
        # 컨텍스트 충분성 확인
        has_context = self._has_sufficient_context(context)
        confidence = self._calculate_confidence(context)

        if not has_context:
            return FactualResponse(
                answer="죄송합니다. 해당 질문에 대한 충분한 정보가 없습니다.",
                confidence=0.0,
                has_sufficient_context=False
            )

        # 응답 생성
        answer = self.generate(context)

        # 출처 추출
        sources = self._extract_sources(context)

        return FactualResponse(
            answer=answer,
            confidence=confidence,
            sources=sources,
            has_sufficient_context=True
        )

    def _build_user_prompt(self, context: QueryContext) -> str:
        """사용자 프롬프트 구성"""
        # 검색 결과 포맷팅
        context_str = self._format_retrieval_results(context)

        # 엔티티 정보
        entities_str = ""
        if context.entities:
            entities_str = f"\nDetected entities: {json.dumps(context.entities, ensure_ascii=False)}"

        # 프롬프트 구성
        prompt = f"""Question: {context.question}
{entities_str}

Context:
{context_str}

Instructions:
- Answer the question based ONLY on the context above
- Be concise and factual (2-3 sentences max)
- If context is insufficient, say "정보가 부족합니다"
- Do NOT make up information
- Answer in Korean

Answer:"""

        return prompt

    def _format_retrieval_results(self, context: QueryContext) -> str:
        """검색 결과 포맷팅"""
        if not context.retrieval_results:
            return "No relevant information found."

        parts = []

        for result in context.retrieval_results:
            source = result.source
            data = result.data

            if not data:
                continue

            parts.append(f"\n=== {source.upper()} ===")

            for i, item in enumerate(data[:5], 1):
                # 주요 필드 추출
                content = self._extract_content(item)
                parts.append(f"{i}. {content}")

        if not parts:
            return "No relevant information found."

        return "\n".join(parts)

    def _extract_content(self, item: Dict) -> str:
        """항목에서 주요 콘텐츠 추출"""
        # 우선순위: content > text > description > summary > 전체
        for field in ['content', 'text', 'description', 'summary', 'name']:
            if field in item and item[field]:
                value = str(item[field])
                if len(value) > 200:
                    value = value[:200] + "..."
                return value

        # 전체 JSON (짧게)
        return json.dumps(item, ensure_ascii=False)[:200]

    def _has_sufficient_context(self, context: QueryContext) -> bool:
        """컨텍스트 충분성 확인"""
        total_results = context.get_total_retrieval_count()

        if total_results == 0:
            return False

        # 최소 관련 결과 필요
        if total_results < 1:
            return False

        return True

    def _calculate_confidence(self, context: QueryContext) -> float:
        """신뢰도 점수 계산"""
        if not context.retrieval_results:
            return 0.0

        # 검색 결과 수 기반 점수
        total_results = context.get_total_retrieval_count()
        result_score = min(total_results / 10, 0.4)

        # 검색 점수 기반 (있으면)
        score_sum = 0.0
        score_count = 0
        for result in context.retrieval_results:
            if result.score > 0:
                score_sum += result.score
                score_count += 1

        avg_score = (score_sum / score_count) if score_count > 0 else 0.5
        score_component = avg_score * 0.4

        # 엔티티 매칭 점수
        entity_score = 0.2 if context.entities else 0.0

        confidence = result_score + score_component + entity_score
        return min(confidence, 1.0)

    def _generate_insufficient_context_response(self, context: QueryContext) -> str:
        """컨텍스트 부족 시 응답"""
        question = context.question

        # 간단한 응답
        return f"죄송합니다. '{question[:50]}...'에 대한 정확한 정보를 찾지 못했습니다. 다른 방식으로 질문해 주시거나, 더 구체적인 키워드를 사용해 주세요."

    def _extract_sources(self, context: QueryContext) -> List[SourceCitation]:
        """출처 추출"""
        sources = []

        for i, result in enumerate(context.retrieval_results, 1):
            if not result.data:
                continue

            for item in result.data[:3]:
                content = self._extract_content(item)
                sources.append(SourceCitation(
                    source=result.source,
                    content=content,
                    relevance=result.score,
                    index=len(sources) + 1
                ))

        return sources[:5]  # 최대 5개

    def _append_sources(self, response: str, context: QueryContext) -> str:
        """응답에 출처 추가"""
        sources = self._extract_sources(context)

        if not sources:
            return response

        # 마크다운 형식 출처
        source_lines = ["\n\n---", "**참고:**"]
        for src in sources[:3]:
            source_lines.append(f"- [{src.source}] {src.content[:50]}...")

        return response + "\n".join(source_lines)

    def get_debug_info(self) -> Dict[str, Any]:
        """디버그 정보"""
        base_info = super().get_debug_info()
        base_info.update({
            'include_sources': self.include_sources,
            'min_confidence': self.min_confidence,
            'default_temperature': self.DEFAULT_TEMPERATURE,
        })
        return base_info
