"""
Insight Generator - Production Grade v2.0
인사이트 기반 응답 생성

Features:
    - 데이터 분석 및 해석
    - 트렌드 감지
    - 통계 포맷팅
    - 액션 가능한 인사이트
    - 시각화 힌트 제공
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
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

logger = logging.getLogger(__name__)


# ============================================================
# Insight Types
# ============================================================

class InsightType(str, Enum):
    """인사이트 유형"""
    TREND = "trend"
    COMPARISON = "comparison"
    ANOMALY = "anomaly"
    CORRELATION = "correlation"
    PERFORMANCE = "performance"
    RECOMMENDATION = "recommendation"


class TrendDirection(str, Enum):
    """트렌드 방향"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"
    VOLATILE = "volatile"


@dataclass
class InsightItem:
    """개별 인사이트"""
    insight_type: InsightType
    title: str
    description: str
    importance: float = 0.5  # 0-1
    metrics: Dict[str, Any] = field(default_factory=dict)
    trend: Optional[TrendDirection] = None

    def to_markdown(self) -> str:
        """마크다운 형식 변환"""
        icon = self._get_icon()
        trend_str = f" ({self.trend.value})" if self.trend else ""
        return f"{icon} **{self.title}**{trend_str}\n   {self.description}"

    def _get_icon(self) -> str:
        icons = {
            InsightType.TREND: "[T]",
            InsightType.COMPARISON: "[C]",
            InsightType.ANOMALY: "[!]",
            InsightType.CORRELATION: "[~]",
            InsightType.PERFORMANCE: "[P]",
            InsightType.RECOMMENDATION: "[R]",
        }
        return icons.get(self.insight_type, "[*]")


@dataclass
class InsightResponse:
    """인사이트 응답 구조"""
    summary: str
    insights: List[InsightItem] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    data_quality: float = 1.0
    visualization_hints: List[str] = field(default_factory=list)

    def to_formatted_string(self) -> str:
        """포맷된 응답 문자열"""
        parts = [self.summary, "\n\n**주요 인사이트:**"]

        for insight in self.insights[:5]:
            parts.append(insight.to_markdown())

        if self.recommendations:
            parts.append("\n**추천 액션:**")
            for i, rec in enumerate(self.recommendations[:3], 1):
                parts.append(f"{i}. {rec}")

        return "\n".join(parts)


# ============================================================
# Statistics Helpers
# ============================================================

class StatisticsHelper:
    """통계 계산 헬퍼"""

    @staticmethod
    def calculate_growth_rate(current: float, previous: float) -> float:
        """성장률 계산"""
        if previous == 0:
            return 0.0 if current == 0 else 100.0
        return ((current - previous) / previous) * 100

    @staticmethod
    def detect_trend(values: List[float]) -> TrendDirection:
        """트렌드 감지"""
        if len(values) < 2:
            return TrendDirection.STABLE

        # 간단한 선형 회귀
        n = len(values)
        avg_increase = (values[-1] - values[0]) / (n - 1) if n > 1 else 0

        volatility = sum(abs(values[i] - values[i-1]) for i in range(1, n)) / (n - 1) if n > 1 else 0
        avg_value = sum(values) / n if n > 0 else 0

        # 변동성이 높으면 volatile
        if avg_value > 0 and volatility / avg_value > 0.3:
            return TrendDirection.VOLATILE

        # 증가/감소 판단
        if avg_increase > avg_value * 0.05:
            return TrendDirection.UP
        elif avg_increase < -avg_value * 0.05:
            return TrendDirection.DOWN

        return TrendDirection.STABLE

    @staticmethod
    def find_top_performers(
        items: List[Dict],
        metric_key: str,
        n: int = 5
    ) -> List[Dict]:
        """상위 성과 항목 추출"""
        sorted_items = sorted(
            items,
            key=lambda x: x.get(metric_key, 0),
            reverse=True
        )
        return sorted_items[:n]

    @staticmethod
    def calculate_engagement_rate(likes: int, comments: int, views: int) -> float:
        """참여율 계산"""
        if views == 0:
            return 0.0
        return ((likes + comments * 2) / views) * 100

    @staticmethod
    def format_number(value: float) -> str:
        """숫자 포맷팅"""
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        elif value >= 1_000:
            return f"{value / 1_000:.1f}K"
        return str(int(value))


# ============================================================
# Insight Generator
# ============================================================

class InsightGenerator(BaseGenerator):
    """
    인사이트 생성기 - Production Grade

    Features:
        - Temperature: 0.7 (창의성)
        - 분석적 응답
        - 통계 데이터 해석
        - 트렌드 분석
        - 액션 제안
    """

    # 기본 설정
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MODEL = "feature"  # GPT-5-mini for better analytical insights
    DEFAULT_MAX_TOKENS = 2000

    def __init__(self, brand_config: Dict):
        super().__init__(brand_config, GeneratorType.INSIGHT)

        # Insight 전용 설정
        self.config.temperature = self.DEFAULT_TEMPERATURE
        self.config.model_variant = self.DEFAULT_MODEL
        self.config.max_tokens = self.DEFAULT_MAX_TOKENS

        # 분석 설정
        self.max_insights = 5
        self.include_visualizations = True

    def generate(self, context: QueryContext) -> str:
        """
        인사이트 응답 생성

        Args:
            context: 쿼리 컨텍스트

        Returns:
            생성된 응답
        """
        metrics = self._create_metrics(context)

        try:
            # 데이터 분석
            analyzed_data = self._analyze_data(context)

            # 프롬프트 로드
            system_prompt = self._load_system_prompt(context.question_type)
            user_prompt = self._build_user_prompt(context, analyzed_data)

            context.set_prompts(system_prompt, user_prompt)

            # LLM 호출
            response = self._invoke_llm(user_prompt, system_prompt, metrics)

            # 메트릭 완료
            metrics.complete()
            self._last_metrics = metrics

            return response

        except Exception as e:
            logger.error(f"Insight generation error: {e}")
            metrics.complete(success=False, error=str(e))
            self._last_metrics = metrics

            return "죄송합니다. 분석 결과를 생성할 수 없습니다."

    def generate_structured(self, context: QueryContext) -> InsightResponse:
        """
        구조화된 인사이트 응답 생성

        Args:
            context: 쿼리 컨텍스트

        Returns:
            InsightResponse 객체
        """
        # 데이터 분석
        analyzed_data = self._analyze_data(context)

        # 인사이트 추출
        insights = self._extract_insights(context, analyzed_data)

        # 추천 생성
        recommendations = self._generate_recommendations(insights)

        # 시각화 힌트
        viz_hints = self._generate_visualization_hints(analyzed_data)

        # 요약 생성
        summary = self.generate(context)

        return InsightResponse(
            summary=summary,
            insights=insights,
            recommendations=recommendations,
            data_quality=self._assess_data_quality(context),
            visualization_hints=viz_hints
        )

    def _build_user_prompt(
        self,
        context: QueryContext,
        analyzed_data: Dict[str, Any]
    ) -> str:
        """사용자 프롬프트 구성"""
        # 데이터 포맷팅
        data_str = self._format_analyzed_data(context, analyzed_data)

        # 프롬프트 구성
        prompt = f"""Question: {context.question}

Data Analysis:
{data_str}

Pre-computed Insights:
{json.dumps(analyzed_data.get('insights', []), ensure_ascii=False, indent=2)}

Instructions:
- Analyze the data above thoroughly
- Provide 3-5 key insights with specific numbers
- Each insight should be actionable
- Suggest 2-3 recommendations
- Use Korean for response
- Use bullet points for clarity
- Highlight important trends or anomalies

Analysis:"""

        return prompt

    def _analyze_data(self, context: QueryContext) -> Dict[str, Any]:
        """데이터 사전 분석"""
        analysis = {
            'total_items': 0,
            'sources': [],
            'insights': [],
            'metrics': {},
            'trends': {},
        }

        if not context.retrieval_results:
            return analysis

        # 각 소스별 분석
        for result in context.retrieval_results:
            source = result.source
            data = result.data

            if not data:
                continue

            analysis['sources'].append(source)
            analysis['total_items'] += len(data)

            # 통계 데이터 분석
            if 'stats' in source.lower():
                stats_analysis = self._analyze_stats(data)
                analysis['metrics'][source] = stats_analysis

            # 포스트 데이터 분석
            if 'post' in source.lower() or 'content' in source.lower():
                content_analysis = self._analyze_content(data)
                analysis['insights'].extend(content_analysis.get('insights', []))

        return analysis

    def _analyze_stats(self, data: List[Dict]) -> Dict[str, Any]:
        """통계 데이터 분석"""
        if not data:
            return {}

        analysis = {
            'count': len(data),
            'top_performers': [],
            'averages': {},
            'trends': {}
        }

        # 숫자 필드 추출
        numeric_fields = []
        if data:
            for key, value in data[0].items():
                if isinstance(value, (int, float)):
                    numeric_fields.append(key)

        # 평균 계산
        for field in numeric_fields:
            values = [item.get(field, 0) for item in data]
            if values:
                analysis['averages'][field] = sum(values) / len(values)
                analysis['trends'][field] = StatisticsHelper.detect_trend(values).value

        # 상위 성과자 (likes 기준)
        if 'likes' in numeric_fields:
            analysis['top_performers'] = StatisticsHelper.find_top_performers(data, 'likes', 5)

        return analysis

    def _analyze_content(self, data: List[Dict]) -> Dict[str, Any]:
        """콘텐츠 데이터 분석"""
        insights = []

        if not data:
            return {'insights': insights}

        # 참여율 분석
        engagement_rates = []
        for item in data:
            likes = item.get('likes', 0)
            comments = item.get('comments', 0)
            views = item.get('views', 1)
            rate = StatisticsHelper.calculate_engagement_rate(likes, comments, views)
            engagement_rates.append(rate)

        if engagement_rates:
            avg_engagement = sum(engagement_rates) / len(engagement_rates)
            insights.append({
                'type': 'performance',
                'title': '평균 참여율',
                'value': f"{avg_engagement:.2f}%"
            })

        return {'insights': insights}

    def _format_analyzed_data(
        self,
        context: QueryContext,
        analyzed_data: Dict[str, Any]
    ) -> str:
        """분석 데이터 포맷팅"""
        parts = []

        # 통계 데이터
        for result in context.retrieval_results:
            source = result.source
            data = result.data

            if not data:
                continue

            parts.append(f"\n=== {source.upper()} ({len(data)} items) ===")

            # 상위 항목
            for i, item in enumerate(data[:10], 1):
                likes = item.get('likes', 0)
                views = item.get('views', 0)
                content = str(item.get('content', item.get('text', '')))[:80]

                parts.append(
                    f"{i}. Likes: {StatisticsHelper.format_number(likes)} | "
                    f"Views: {StatisticsHelper.format_number(views)}\n"
                    f"   Content: {content}..."
                )

        # 분석 결과 요약
        if analyzed_data.get('metrics'):
            parts.append("\n=== ANALYSIS SUMMARY ===")
            for source, metrics in analyzed_data['metrics'].items():
                if metrics.get('averages'):
                    parts.append(f"\n{source} Averages:")
                    for key, value in metrics['averages'].items():
                        parts.append(f"  - {key}: {StatisticsHelper.format_number(value)}")

        return "\n".join(parts) if parts else "No data available."

    def _extract_insights(
        self,
        context: QueryContext,
        analyzed_data: Dict[str, Any]
    ) -> List[InsightItem]:
        """인사이트 추출"""
        insights = []

        # 메트릭 기반 인사이트
        for source, metrics in analyzed_data.get('metrics', {}).items():
            # 트렌드 인사이트
            for field, trend in metrics.get('trends', {}).items():
                if trend != 'stable':
                    insights.append(InsightItem(
                        insight_type=InsightType.TREND,
                        title=f"{field} 트렌드",
                        description=f"{source}의 {field}이(가) {trend} 추세입니다.",
                        trend=TrendDirection(trend),
                        importance=0.7
                    ))

            # 성과 인사이트
            top_performers = metrics.get('top_performers', [])
            if top_performers:
                insights.append(InsightItem(
                    insight_type=InsightType.PERFORMANCE,
                    title="최고 성과 콘텐츠",
                    description=f"상위 {len(top_performers)}개 콘텐츠가 평균 대비 높은 성과를 보입니다.",
                    importance=0.8,
                    metrics={'count': len(top_performers)}
                ))

        # 중요도 순 정렬
        insights.sort(key=lambda x: x.importance, reverse=True)

        return insights[:self.max_insights]

    def _generate_recommendations(
        self,
        insights: List[InsightItem]
    ) -> List[str]:
        """추천 생성"""
        recommendations = []

        for insight in insights:
            if insight.insight_type == InsightType.TREND:
                if insight.trend == TrendDirection.UP:
                    recommendations.append(
                        f"{insight.title} 상승 추세를 활용해 관련 콘텐츠를 확대하세요."
                    )
                elif insight.trend == TrendDirection.DOWN:
                    recommendations.append(
                        f"{insight.title} 하락 추세를 분석하고 개선 전략을 수립하세요."
                    )

            elif insight.insight_type == InsightType.PERFORMANCE:
                recommendations.append(
                    "높은 성과 콘텐츠의 특성을 분석하여 향후 콘텐츠 전략에 반영하세요."
                )

        return recommendations[:3]

    def _generate_visualization_hints(
        self,
        analyzed_data: Dict[str, Any]
    ) -> List[str]:
        """시각화 힌트 생성"""
        hints = []

        if analyzed_data.get('metrics'):
            for source, metrics in analyzed_data['metrics'].items():
                if metrics.get('trends'):
                    hints.append(f"line_chart: {source} 시계열 트렌드")
                if metrics.get('top_performers'):
                    hints.append(f"bar_chart: {source} 상위 성과자 비교")

        if analyzed_data.get('total_items', 0) > 10:
            hints.append("pie_chart: 소스별 데이터 분포")

        return hints[:5]

    def _assess_data_quality(self, context: QueryContext) -> float:
        """데이터 품질 평가"""
        if not context.retrieval_results:
            return 0.0

        total_items = context.get_total_retrieval_count()

        if total_items == 0:
            return 0.0
        elif total_items < 5:
            return 0.3
        elif total_items < 20:
            return 0.6
        else:
            return 0.9

    def get_debug_info(self) -> Dict[str, Any]:
        """디버그 정보"""
        base_info = super().get_debug_info()
        base_info.update({
            'max_insights': self.max_insights,
            'include_visualizations': self.include_visualizations,
        })
        return base_info
