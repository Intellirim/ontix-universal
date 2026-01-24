"""
Business Advisor Generator - Production Grade v1.0
비즈니스 컨설팅 스타일 응답 생성

Features:
    - Neo4j 전체 브랜드 데이터 활용
    - 감정 분석 인사이트
    - 트렌드 및 성과 분석
    - 실행 가능한 비즈니스 추천
    - 데이터 기반 컨설팅
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
)
from app.core.context import QueryContext
from app.services.shared.neo4j import get_neo4j_client

logger = logging.getLogger(__name__)


@dataclass
class BrandMetrics:
    """브랜드 전체 메트릭"""
    total_content: int = 0
    total_interactions: int = 0
    sentiment_positive: int = 0
    sentiment_neutral: int = 0
    sentiment_negative: int = 0
    total_likes: int = 0
    total_comments: int = 0
    total_views: int = 0
    top_content: List[Dict] = field(default_factory=list)
    recent_content: List[Dict] = field(default_factory=list)
    negative_interactions: List[Dict] = field(default_factory=list)
    positive_interactions: List[Dict] = field(default_factory=list)

    @property
    def sentiment_score(self) -> float:
        """감정 점수 (0-100)"""
        total = self.sentiment_positive + self.sentiment_neutral + self.sentiment_negative
        if total == 0:
            return 50.0
        return (self.sentiment_positive * 100 + self.sentiment_neutral * 50) / total

    @property
    def positive_ratio(self) -> float:
        """긍정 비율"""
        total = self.sentiment_positive + self.sentiment_neutral + self.sentiment_negative
        if total == 0:
            return 0.0
        return self.sentiment_positive / total * 100

    @property
    def negative_ratio(self) -> float:
        """부정 비율"""
        total = self.sentiment_positive + self.sentiment_neutral + self.sentiment_negative
        if total == 0:
            return 0.0
        return self.sentiment_negative / total * 100

    def to_summary(self) -> str:
        """요약 문자열"""
        return f"""
=== 브랜드 성과 요약 ===
콘텐츠 수: {self.total_content}개
고객 반응(댓글): {self.total_interactions}개
총 좋아요: {self.total_likes:,}
총 조회수: {self.total_views:,}

=== 감정 분석 결과 ===
긍정적 반응: {self.sentiment_positive}개 ({self.positive_ratio:.1f}%)
중립적 반응: {self.sentiment_neutral}개
부정적 반응: {self.sentiment_negative}개 ({self.negative_ratio:.1f}%)
감정 점수: {self.sentiment_score:.1f}/100
"""


class BusinessAdvisorGenerator(BaseGenerator):
    """
    비즈니스 어드바이저 생성기 - Production Grade

    Features:
        - Temperature: 0.7 (분석적이면서 창의적)
        - 비즈니스 컨설팅 스타일
        - 전체 브랜드 데이터 활용
        - 실행 가능한 인사이트
    """

    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MODEL = "feature"  # GPT-5-mini
    DEFAULT_MAX_TOKENS = 4000

    def __init__(self, brand_config: Dict):
        super().__init__(brand_config, GeneratorType.INSIGHT)

        # 설정 오버라이드
        self.config.temperature = self.DEFAULT_TEMPERATURE
        self.config.model_variant = self.DEFAULT_MODEL
        self.config.max_tokens = self.DEFAULT_MAX_TOKENS

        # Neo4j 클라이언트
        self.neo4j = get_neo4j_client()

        # 캐시된 브랜드 메트릭
        self._cached_metrics: Optional[BrandMetrics] = None

    def generate(self, context: QueryContext) -> str:
        """
        비즈니스 어드바이저 응답 생성

        Args:
            context: 쿼리 컨텍스트

        Returns:
            생성된 응답
        """
        metrics = self._create_metrics(context)

        try:
            # 1. 브랜드 전체 데이터 수집
            brand_metrics = self._collect_brand_metrics()

            # 2. 시스템 프롬프트 구성
            system_prompt = self._build_system_prompt(brand_metrics)

            # 3. 사용자 프롬프트 구성
            user_prompt = self._build_user_prompt(context, brand_metrics)

            context.set_prompts(system_prompt, user_prompt)

            # 4. LLM 호출
            response = self._invoke_llm(user_prompt, system_prompt, metrics)

            # 메트릭 완료
            metrics.complete()
            self._last_metrics = metrics

            return response

        except Exception as e:
            logger.error(f"Business advisor generation error: {e}")
            metrics.complete(success=False, error=str(e))
            self._last_metrics = metrics

            return "죄송합니다. 분석 결과를 생성할 수 없습니다. 잠시 후 다시 시도해 주세요."

    def _collect_brand_metrics(self) -> BrandMetrics:
        """브랜드 전체 메트릭 수집"""
        metrics = BrandMetrics()

        try:
            # 1. 콘텐츠 통계
            content_stats = self.neo4j.query("""
                MATCH (c:Content)
                WHERE c.brand_id = $brand_id
                RETURN
                    count(c) as total,
                    sum(coalesce(c.like_count, 0)) as likes,
                    sum(coalesce(c.comment_count, 0)) as comments,
                    sum(coalesce(c.view_count, 0)) as views
            """, {'brand_id': self.brand_id})

            if content_stats:
                row = content_stats[0]
                metrics.total_content = row.get('total', 0)
                metrics.total_likes = row.get('likes', 0) or 0
                metrics.total_comments = row.get('comments', 0) or 0
                metrics.total_views = row.get('views', 0) or 0

            # 2. 인터랙션 감정 분석
            sentiment_stats = self.neo4j.query("""
                MATCH (i:Interaction)
                WHERE i.brand_id = $brand_id
                RETURN
                    coalesce(i.sentiment, 'neutral') as sentiment,
                    count(*) as count
            """, {'brand_id': self.brand_id})

            for row in sentiment_stats or []:
                sentiment = row.get('sentiment', 'neutral')
                count = row.get('count', 0)
                if sentiment == 'positive':
                    metrics.sentiment_positive = count
                elif sentiment == 'negative':
                    metrics.sentiment_negative = count
                else:
                    metrics.sentiment_neutral = count

            metrics.total_interactions = (
                metrics.sentiment_positive +
                metrics.sentiment_neutral +
                metrics.sentiment_negative
            )

            # 3. 상위 성과 콘텐츠 (좋아요 기준)
            top_content = self.neo4j.query("""
                MATCH (c:Content)
                WHERE c.brand_id = $brand_id
                RETURN
                    c.text as text,
                    c.url as url,
                    c.platform as platform,
                    coalesce(c.like_count, 0) as likes,
                    coalesce(c.view_count, 0) as views
                ORDER BY likes DESC
                LIMIT 5
            """, {'brand_id': self.brand_id})

            metrics.top_content = [
                {
                    'text': (r.get('text') or '')[:200],
                    'url': r.get('url'),
                    'platform': r.get('platform'),
                    'likes': r.get('likes', 0),
                    'views': r.get('views', 0),
                }
                for r in (top_content or [])
            ]

            # 4. 최근 콘텐츠
            recent_content = self.neo4j.query("""
                MATCH (c:Content)
                WHERE c.brand_id = $brand_id
                RETURN
                    c.text as text,
                    c.created_at as created_at,
                    coalesce(c.like_count, 0) as likes
                ORDER BY c.created_at DESC
                LIMIT 5
            """, {'brand_id': self.brand_id})

            metrics.recent_content = [
                {
                    'text': (r.get('text') or '')[:200],
                    'created_at': str(r.get('created_at')) if r.get('created_at') else None,
                    'likes': r.get('likes', 0),
                }
                for r in (recent_content or [])
            ]

            # 5. 부정적 반응 샘플
            negative_interactions = self.neo4j.query("""
                MATCH (i:Interaction)
                WHERE i.brand_id = $brand_id AND i.sentiment = 'negative'
                RETURN i.text as text
                LIMIT 5
            """, {'brand_id': self.brand_id})

            metrics.negative_interactions = [
                {'text': (r.get('text') or '')[:150]}
                for r in (negative_interactions or [])
            ]

            # 6. 긍정적 반응 샘플
            positive_interactions = self.neo4j.query("""
                MATCH (i:Interaction)
                WHERE i.brand_id = $brand_id AND i.sentiment = 'positive'
                RETURN i.text as text
                LIMIT 5
            """, {'brand_id': self.brand_id})

            metrics.positive_interactions = [
                {'text': (r.get('text') or '')[:150]}
                for r in (positive_interactions or [])
            ]

            self._cached_metrics = metrics
            logger.info(f"Collected brand metrics: {metrics.total_content} content, {metrics.total_interactions} interactions")

        except Exception as e:
            logger.error(f"Failed to collect brand metrics: {e}")

        return metrics

    def _build_system_prompt(self, brand_metrics: BrandMetrics) -> str:
        """비즈니스 컨설턴트 시스템 프롬프트"""

        brand_desc = self.brand_config.get('brand', {}).get('description', '')
        brand_industry = self.brand_config.get('brand', {}).get('industry', '')

        return f"""당신은 {self.brand_name}의 전문 비즈니스 컨설턴트입니다.

## 역할
- 브랜드 오너에게 데이터 기반 비즈니스 인사이트 제공
- SNS 성과 분석 및 개선 전략 제안
- 고객 반응 분석을 통한 마케팅 인사이트 도출
- 실행 가능한 구체적 액션 아이템 제시

## 브랜드 정보
- 브랜드명: {self.brand_name}
- 산업/분야: {brand_industry}
- 설명: {brand_desc}

## 현재 브랜드 성과 데이터
{brand_metrics.to_summary()}

## 응답 가이드라인
1. **데이터 기반**: 모든 인사이트는 실제 데이터를 근거로 제시
2. **구체적 수치**: 가능한 경우 구체적인 숫자와 비율 포함
3. **실행 가능성**: 브랜드 오너가 바로 실행할 수 있는 구체적 액션 제안
4. **우선순위**: 가장 중요한 인사이트부터 순서대로 제시
5. **전문적 톤**: 비즈니스 컨설턴트로서 전문적이면서도 이해하기 쉽게

## 출력 형식
- 핵심 인사이트를 먼저 요약
- 데이터 분석 결과 제시
- 구체적인 개선 방안 또는 추천 액션
- 필요시 주의점이나 리스크 언급

응답은 한국어로 작성하세요.
"""

    def _build_user_prompt(
        self,
        context: QueryContext,
        brand_metrics: BrandMetrics
    ) -> str:
        """사용자 프롬프트 구성"""

        # 검색 결과 포맷팅
        context_str = self._format_retrieval_results(context)

        # 상위 성과 콘텐츠 포맷팅
        top_content_str = ""
        if brand_metrics.top_content:
            top_content_str = "\n\n## 상위 성과 콘텐츠 (좋아요 기준)"
            for i, c in enumerate(brand_metrics.top_content[:3], 1):
                top_content_str += f"\n{i}. [{c.get('platform', 'SNS')}] 좋아요 {c.get('likes', 0):,}개"
                if c.get('text'):
                    top_content_str += f"\n   내용: {c['text'][:100]}..."

        # 부정적 반응 샘플
        negative_str = ""
        if brand_metrics.negative_interactions:
            negative_str = "\n\n## 부정적 고객 반응 샘플"
            for i, n in enumerate(brand_metrics.negative_interactions[:3], 1):
                negative_str += f"\n{i}. \"{n.get('text', '')}\""

        # 긍정적 반응 샘플
        positive_str = ""
        if brand_metrics.positive_interactions:
            positive_str = "\n\n## 긍정적 고객 반응 샘플"
            for i, p in enumerate(brand_metrics.positive_interactions[:3], 1):
                positive_str += f"\n{i}. \"{p.get('text', '')}\""

        prompt = f"""## 브랜드 오너의 질문
{context.question}

## 현재 데이터 요약
- 총 콘텐츠: {brand_metrics.total_content}개
- 총 고객 반응(댓글): {brand_metrics.total_interactions}개
- 감정 분석: 긍정 {brand_metrics.positive_ratio:.1f}% / 부정 {brand_metrics.negative_ratio:.1f}%
- 감정 점수: {brand_metrics.sentiment_score:.1f}/100
{top_content_str}
{negative_str}
{positive_str}

{context_str}

## 요청
위 데이터를 바탕으로 브랜드 오너의 질문에 대해 전문적인 비즈니스 인사이트를 제공해주세요.
데이터에 기반한 분석과 실행 가능한 추천을 포함해주세요.
"""

        return prompt

    def _format_retrieval_results(self, context: QueryContext) -> str:
        """검색 결과 포맷팅"""
        if not context.retrieval_results:
            return ""

        total = context.get_total_retrieval_count()
        if total == 0:
            return ""

        parts = ["\n## 관련 데이터"]

        for result in context.retrieval_results:
            items = getattr(result, 'items', None) or getattr(result, 'data', None) or []
            for item in items[:5]:
                if isinstance(item, dict):
                    content = item.get('content', '') or item.get('text', '')
                    if content:
                        parts.append(f"- {content[:150]}...")
                else:
                    content = getattr(item, 'content', '') or getattr(item, 'text', '')
                    if content:
                        parts.append(f"- {content[:150]}...")

        return "\n".join(parts) if len(parts) > 1 else ""

    def get_debug_info(self) -> Dict[str, Any]:
        """디버그 정보"""
        base_info = super().get_debug_info()
        if self._cached_metrics:
            base_info['brand_metrics'] = {
                'total_content': self._cached_metrics.total_content,
                'total_interactions': self._cached_metrics.total_interactions,
                'sentiment_score': self._cached_metrics.sentiment_score,
            }
        return base_info
