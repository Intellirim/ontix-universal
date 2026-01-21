"""
Recommendation Generator - Production Grade v2.0
추천 기반 응답 생성

Features:
    - 개인화 추천
    - 다중 추천 전략
    - 스코어링 시스템
    - 다양성 보장
    - 설득력 있는 설명
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from abc import ABC, abstractmethod

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
# Recommendation Types
# ============================================================

class RecommendationStrategy(str, Enum):
    """추천 전략"""
    SIMILARITY = "similarity"       # 유사도 기반
    POPULARITY = "popularity"       # 인기도 기반
    PERSONALIZED = "personalized"   # 개인화
    TRENDING = "trending"           # 트렌딩
    HYBRID = "hybrid"               # 하이브리드


class RecommendationReason(str, Enum):
    """추천 이유"""
    HIGH_RATING = "high_rating"
    BEST_SELLER = "best_seller"
    NEW_ARRIVAL = "new_arrival"
    PRICE_VALUE = "price_value"
    MATCHING_PREFERENCE = "matching_preference"
    SIMILAR_TO_LIKED = "similar_to_liked"
    TRENDING_NOW = "trending_now"


class SortOrder(str, Enum):
    """정렬 순서"""
    RELEVANCE = "relevance"
    PRICE_LOW = "price_low"
    PRICE_HIGH = "price_high"
    RATING = "rating"
    POPULARITY = "popularity"
    NEWEST = "newest"


@dataclass
class RecommendationItem:
    """개별 추천 아이템"""
    id: str
    name: str
    score: float = 0.0
    rank: int = 0
    price: Optional[float] = None
    description: Optional[str] = None
    reasons: List[RecommendationReason] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """마크다운 형식 변환"""
        price_str = f" - ₩{self.price:,.0f}" if self.price else ""
        reasons_str = ", ".join([r.value for r in self.reasons[:2]]) if self.reasons else ""
        reason_tag = f" ({reasons_str})" if reasons_str else ""

        return f"**{self.rank}. {self.name}**{price_str}{reason_tag}\n   {self.description or ''}"

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'id': self.id,
            'name': self.name,
            'score': self.score,
            'rank': self.rank,
            'price': self.price,
            'description': self.description,
            'reasons': [r.value for r in self.reasons],
            'metadata': self.metadata,
        }


@dataclass
class RecommendationResponse:
    """추천 응답 구조"""
    items: List[RecommendationItem] = field(default_factory=list)
    summary: str = ""
    strategy_used: RecommendationStrategy = RecommendationStrategy.HYBRID
    total_candidates: int = 0
    diversity_score: float = 0.0

    def to_formatted_string(self, max_items: int = 5) -> str:
        """포맷된 응답 문자열"""
        if not self.items:
            return self.summary or "추천할 상품이 없습니다."

        parts = [self.summary] if self.summary else []
        parts.append("\n**추천 상품:**\n")

        for item in self.items[:max_items]:
            parts.append(item.to_markdown())
            parts.append("")

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'items': [item.to_dict() for item in self.items],
            'summary': self.summary,
            'strategy': self.strategy_used.value,
            'total_candidates': self.total_candidates,
            'diversity_score': self.diversity_score,
        }


# ============================================================
# Scoring System
# ============================================================

class RecommendationScorer:
    """추천 스코어링 시스템"""

    # 스코어 가중치
    WEIGHTS = {
        'relevance': 0.35,
        'popularity': 0.20,
        'rating': 0.20,
        'recency': 0.10,
        'price_match': 0.15,
    }

    @classmethod
    def calculate_score(
        cls,
        item: Dict[str, Any],
        query_context: Optional[Dict] = None
    ) -> Tuple[float, List[RecommendationReason]]:
        """
        아이템 스코어 계산

        Args:
            item: 상품/콘텐츠 데이터
            query_context: 쿼리 컨텍스트 (엔티티, 선호도 등)

        Returns:
            (점수, 추천 이유 리스트)
        """
        scores = {}
        reasons = []

        # 1. 관련성 점수 (검색 스코어 기반)
        relevance = item.get('score', item.get('relevance', 0.5))
        scores['relevance'] = min(relevance, 1.0)

        # 2. 인기도 점수
        popularity = cls._calculate_popularity(item)
        scores['popularity'] = popularity
        if popularity > 0.7:
            reasons.append(RecommendationReason.BEST_SELLER)

        # 3. 평점 점수
        rating = item.get('rating', item.get('avg_rating', 0))
        if rating:
            scores['rating'] = min(rating / 5.0, 1.0)
            if rating >= 4.5:
                reasons.append(RecommendationReason.HIGH_RATING)
        else:
            scores['rating'] = 0.5

        # 4. 최신성 점수
        recency = cls._calculate_recency(item)
        scores['recency'] = recency
        if recency > 0.8:
            reasons.append(RecommendationReason.NEW_ARRIVAL)

        # 5. 가격 매칭 점수
        price_match = cls._calculate_price_match(item, query_context)
        scores['price_match'] = price_match
        if price_match > 0.8:
            reasons.append(RecommendationReason.PRICE_VALUE)

        # 가중 평균 계산
        total_score = sum(
            scores[key] * cls.WEIGHTS[key]
            for key in cls.WEIGHTS
        )

        return total_score, reasons

    @classmethod
    def _calculate_popularity(cls, item: Dict) -> float:
        """인기도 점수 계산"""
        views = item.get('views', item.get('view_count', 0))
        likes = item.get('likes', item.get('like_count', 0))
        sales = item.get('sales', item.get('order_count', 0))

        # 정규화된 인기도 (로그 스케일)
        import math
        pop_score = math.log1p(views + likes * 2 + sales * 5) / 15
        return min(pop_score, 1.0)

    @classmethod
    def _calculate_recency(cls, item: Dict) -> float:
        """최신성 점수 계산"""
        from datetime import datetime

        created_at = item.get('created_at', item.get('date'))
        if not created_at:
            return 0.5

        try:
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))

            days_old = (datetime.now(created_at.tzinfo) - created_at).days
            # 30일 이내 = 1.0, 365일 이상 = 0.1
            recency = max(0.1, 1.0 - (days_old / 365))
            return recency

        except Exception:
            return 0.5

    @classmethod
    def _calculate_price_match(
        cls,
        item: Dict,
        query_context: Optional[Dict]
    ) -> float:
        """가격 매칭 점수"""
        price = item.get('price', 0)
        if not price:
            return 0.5

        if not query_context:
            return 0.5

        # 가격대 선호도가 있으면 매칭
        price_range = query_context.get('price_range')
        if price_range:
            min_price = price_range.get('min', 0)
            max_price = price_range.get('max', float('inf'))

            if min_price <= price <= max_price:
                return 1.0
            elif price < min_price:
                return 0.7
            else:
                return 0.3

        return 0.5


# ============================================================
# Diversity Optimizer
# ============================================================

class DiversityOptimizer:
    """추천 다양성 최적화"""

    @classmethod
    def diversify(
        cls,
        items: List[RecommendationItem],
        max_items: int = 5,
        category_limit: int = 2
    ) -> Tuple[List[RecommendationItem], float]:
        """
        추천 결과 다양화

        Args:
            items: 스코어 순 정렬된 아이템
            max_items: 최대 추천 수
            category_limit: 카테고리당 최대 아이템 수

        Returns:
            (다양화된 아이템 리스트, 다양성 점수)
        """
        if len(items) <= max_items:
            diversity = cls._calculate_diversity(items)
            return items, diversity

        selected = []
        category_counts: Dict[str, int] = {}

        for item in items:
            if len(selected) >= max_items:
                break

            # 카테고리 추출
            category = item.metadata.get('category', 'default')

            # 카테고리 제한 확인
            if category_counts.get(category, 0) >= category_limit:
                continue

            selected.append(item)
            category_counts[category] = category_counts.get(category, 0) + 1

        # 부족하면 스코어 순으로 채우기
        if len(selected) < max_items:
            for item in items:
                if item not in selected and len(selected) < max_items:
                    selected.append(item)

        diversity = cls._calculate_diversity(selected)
        return selected, diversity

    @classmethod
    def _calculate_diversity(cls, items: List[RecommendationItem]) -> float:
        """다양성 점수 계산"""
        if len(items) <= 1:
            return 1.0

        # 카테고리 다양성
        categories = set(item.metadata.get('category', 'default') for item in items)
        category_diversity = len(categories) / len(items)

        # 가격대 다양성
        prices = [item.price for item in items if item.price]
        if len(prices) >= 2:
            price_range = max(prices) - min(prices)
            avg_price = sum(prices) / len(prices)
            price_diversity = min(price_range / avg_price, 1.0) if avg_price > 0 else 0
        else:
            price_diversity = 0.5

        return (category_diversity * 0.6 + price_diversity * 0.4)


# ============================================================
# Recommendation Generator
# ============================================================

class RecommendationGenerator(BaseGenerator):
    """
    추천 생성기 - Production Grade

    Features:
        - Temperature: 0.7 (자연스러운 설명)
        - 다중 추천 전략
        - 스코어링 시스템
        - 다양성 최적화
        - 개인화 지원
    """

    # 기본 설정
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MODEL = "feature"  # GPT-5-mini for better recommendations
    DEFAULT_MAX_TOKENS = 1500

    def __init__(self, brand_config: Dict):
        super().__init__(brand_config, GeneratorType.RECOMMENDATION)

        # Recommendation 전용 설정
        self.config.temperature = self.DEFAULT_TEMPERATURE
        self.config.model_variant = self.DEFAULT_MODEL
        self.config.max_tokens = self.DEFAULT_MAX_TOKENS

        # 추천 설정
        self.max_recommendations = 5
        self.min_score = 0.3
        self.strategy = RecommendationStrategy.HYBRID
        self.enable_diversity = True

    def generate(self, context: QueryContext) -> str:
        """
        추천 응답 생성

        Args:
            context: 쿼리 컨텍스트

        Returns:
            생성된 응답
        """
        metrics = self._create_metrics(context)

        try:
            # 후보 추출 및 스코어링
            candidates = self._extract_candidates(context)
            scored_items = self._score_candidates(candidates, context)

            # 최소 스코어 필터링
            filtered = [item for item in scored_items if item.score >= self.min_score]

            # 다양성 최적화
            if self.enable_diversity:
                final_items, diversity = DiversityOptimizer.diversify(
                    filtered,
                    max_items=self.max_recommendations
                )
            else:
                final_items = filtered[:self.max_recommendations]
                diversity = 0.0

            # 랭킹 부여
            for i, item in enumerate(final_items, 1):
                item.rank = i

            # 추천 결과가 없으면 기본 응답
            if not final_items:
                metrics.complete()
                self._last_metrics = metrics
                return self._generate_no_recommendation_response(context)

            # 프롬프트 로드
            system_prompt = self._load_system_prompt(context.question_type)
            user_prompt = self._build_user_prompt(context, final_items)

            context.set_prompts(system_prompt, user_prompt)

            # LLM 호출
            response = self._invoke_llm(user_prompt, system_prompt, metrics)

            # 메트릭 완료
            metrics.complete()
            self._last_metrics = metrics

            return response

        except Exception as e:
            logger.error(f"Recommendation generation error: {e}")
            metrics.complete(success=False, error=str(e))
            self._last_metrics = metrics

            return "죄송합니다. 추천을 생성할 수 없습니다."

    def generate_structured(self, context: QueryContext) -> RecommendationResponse:
        """
        구조화된 추천 응답 생성

        Args:
            context: 쿼리 컨텍스트

        Returns:
            RecommendationResponse 객체
        """
        # 후보 추출 및 스코어링
        candidates = self._extract_candidates(context)
        scored_items = self._score_candidates(candidates, context)

        # 필터링 및 다양화
        filtered = [item for item in scored_items if item.score >= self.min_score]
        final_items, diversity = DiversityOptimizer.diversify(
            filtered,
            max_items=self.max_recommendations
        )

        # 랭킹 부여
        for i, item in enumerate(final_items, 1):
            item.rank = i

        # 요약 생성
        summary = self.generate(context)

        return RecommendationResponse(
            items=final_items,
            summary=summary,
            strategy_used=self.strategy,
            total_candidates=len(candidates),
            diversity_score=diversity
        )

    def _build_user_prompt(
        self,
        context: QueryContext,
        items: List[RecommendationItem]
    ) -> str:
        """사용자 프롬프트 구성"""
        # 상품 정보 포맷팅
        products_str = self._format_recommendation_items(items)

        # 사용자 선호도 (있으면)
        preferences_str = ""
        if context.entities:
            preferences_str = f"\nUser Preferences: {json.dumps(context.entities, ensure_ascii=False)}"

        # 프롬프트 구성
        prompt = f"""Question: {context.question}
{preferences_str}

Recommended Products (ranked by relevance):
{products_str}

Instructions:
- Present the top 3-5 recommendations naturally
- Explain why each product is suitable for the user
- Highlight key features, benefits, and price value
- Maintain a friendly, helpful tone
- Answer in Korean
- Use bullet points or numbered list for clarity

Recommendations:"""

        return prompt

    def _format_recommendation_items(self, items: List[RecommendationItem]) -> str:
        """추천 아이템 포맷팅"""
        parts = []

        for item in items:
            price_str = f"₩{item.price:,.0f}" if item.price else "가격 미정"
            reasons = [r.value for r in item.reasons[:2]]
            reasons_str = f" [{', '.join(reasons)}]" if reasons else ""

            parts.append(
                f"{item.rank}. {item.name} - {price_str}{reasons_str}\n"
                f"   Score: {item.score:.2f}\n"
                f"   {item.description or 'No description'}"
            )

        return "\n\n".join(parts)

    def _extract_candidates(self, context: QueryContext) -> List[Dict[str, Any]]:
        """후보 아이템 추출"""
        candidates = []

        if not context.retrieval_results:
            return candidates

        for result in context.retrieval_results:
            source = result.source
            data = result.data

            if not data:
                continue

            # 상품/콘텐츠 데이터 추출
            for item in data:
                # 기본 정보 추출
                candidate = {
                    'id': str(item.get('id', item.get('product_id', ''))),
                    'name': item.get('name', item.get('title', 'Unknown')),
                    'price': item.get('price', item.get('sale_price', 0)),
                    'description': item.get('description', item.get('content', '')),
                    'category': item.get('category', item.get('type', 'default')),
                    'rating': item.get('rating', item.get('avg_rating', 0)),
                    'score': result.score,
                    'source': source,
                    'raw': item,
                }

                # 메타데이터 추가
                for key in ['stock', 'brand', 'tags', 'views', 'likes', 'sales']:
                    if key in item:
                        candidate[key] = item[key]

                candidates.append(candidate)

        return candidates

    def _score_candidates(
        self,
        candidates: List[Dict],
        context: QueryContext
    ) -> List[RecommendationItem]:
        """후보 스코어링"""
        scored_items = []

        # 쿼리 컨텍스트 추출
        query_context = {
            'entities': context.entities,
            'question_type': str(context.question_type.value) if context.question_type else None,
        }

        for candidate in candidates:
            # 스코어 계산
            score, reasons = RecommendationScorer.calculate_score(
                candidate,
                query_context
            )

            # 재고 체크 (0이면 페널티)
            stock = candidate.get('stock', 1)
            if stock == 0:
                score *= 0.5
                reasons = [r for r in reasons if r != RecommendationReason.BEST_SELLER]

            # RecommendationItem 생성
            item = RecommendationItem(
                id=candidate['id'],
                name=candidate['name'],
                score=score,
                price=candidate.get('price'),
                description=self._truncate_description(candidate.get('description', '')),
                reasons=reasons,
                metadata={
                    'category': candidate.get('category', 'default'),
                    'source': candidate.get('source', 'unknown'),
                    'stock': stock,
                }
            )

            scored_items.append(item)

        # 스코어 순 정렬
        scored_items.sort(key=lambda x: x.score, reverse=True)

        return scored_items

    def _truncate_description(self, desc: str, max_length: int = 100) -> str:
        """설명 잘라내기"""
        if not desc:
            return ""
        if len(desc) <= max_length:
            return desc
        return desc[:max_length-3] + "..."

    def _generate_no_recommendation_response(self, context: QueryContext) -> str:
        """추천 결과 없음 응답"""
        question = context.question[:50]

        suggestions = [
            "다른 키워드로 검색해 보시겠어요?",
            "더 넓은 조건으로 찾아볼까요?",
            "인기 상품을 대신 보여드릴까요?",
        ]

        return f"'{question}'에 맞는 추천 상품을 찾지 못했습니다. {suggestions[0]}"

    def set_strategy(self, strategy: RecommendationStrategy):
        """추천 전략 변경"""
        self.strategy = strategy

    def set_max_recommendations(self, max_count: int):
        """최대 추천 수 변경"""
        self.max_recommendations = max(1, min(max_count, 20))

    def get_debug_info(self) -> Dict[str, Any]:
        """디버그 정보"""
        base_info = super().get_debug_info()
        base_info.update({
            'max_recommendations': self.max_recommendations,
            'min_score': self.min_score,
            'strategy': self.strategy.value,
            'enable_diversity': self.enable_diversity,
        })
        return base_info
