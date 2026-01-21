"""
Product Recommendation Handler - Production Grade v2.0
상품 추천 기능

Features:
- Filters v2.0 통합 (Quality + Trust)
- 상품 데이터 기반 신뢰성 검증
- 추천 품질 분석
- 상세 메트릭 로깅
"""

from app.models.feature import FeatureHandler
from app.filters import (
    QualityFilter,
    TrustFilter,
    QualityConfig,
    TrustConfig,
    QualityResult,
    TrustResult,
)
from app.filters.relevance import (
    RelevanceFilter,
    RelevanceConfig,
    RelevanceResult,
    RelevanceLevel,
)
from app.filters.validation import (
    ValidationFilter,
    ValidationConfig,
    ValidationResult,
    ValidationStatus,
    OverallGrade,
    FilterWeight,
)
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

@dataclass
class ProductRecommendationConfig:
    """상품 추천 설정"""
    # 필터 활성화
    quality_filter_enabled: bool = True
    trust_filter_enabled: bool = True
    relevance_filter_enabled: bool = True
    validation_filter_enabled: bool = True

    # 품질 임계값
    min_quality_score: float = 0.5
    min_trust_score: float = 0.5
    min_relevance_score: float = 0.5

    # 종합 검증 임계값
    validation_pass_threshold: float = 0.7
    validation_warning_threshold: float = 0.5

    # 추천 설정
    max_recommendations: int = 5
    min_stock: int = 1
    include_out_of_stock: bool = False

    # 로깅
    log_filter_results: bool = True
    log_recommendations: bool = True

    # 메타데이터
    include_quality_details: bool = True
    include_trust_details: bool = True
    include_relevance_details: bool = True
    include_validation_details: bool = True


# ============================================================
# Product Recommendation Handler
# ============================================================

class ProductRecommendationHandler(FeatureHandler):
    """
    상품 추천 핸들러 - Production Grade v2.0

    Features:
    - 재고 기반 상품 추천
    - 추천 응답 품질 검증
    - 상품 데이터 신뢰성 검증
    - 상세 메트릭 로깅
    """

    def __init__(self, brand_config: Dict[str, Any]):
        super().__init__(brand_config)

        # 설정 로드
        config_dict = brand_config.get('product_recommendation', {})
        self.handler_config = ProductRecommendationConfig(
            quality_filter_enabled=config_dict.get('quality_filter_enabled', True),
            trust_filter_enabled=config_dict.get('trust_filter_enabled', True),
            relevance_filter_enabled=config_dict.get('relevance_filter_enabled', True),
            validation_filter_enabled=config_dict.get('validation_filter_enabled', True),
            min_quality_score=config_dict.get('min_quality_score', 0.5),
            min_trust_score=config_dict.get('min_trust_score', 0.5),
            min_relevance_score=config_dict.get('min_relevance_score', 0.5),
            validation_pass_threshold=config_dict.get('validation_pass_threshold', 0.7),
            validation_warning_threshold=config_dict.get('validation_warning_threshold', 0.5),
            max_recommendations=config_dict.get('max_recommendations', 5),
            min_stock=config_dict.get('min_stock', 1),
            include_out_of_stock=config_dict.get('include_out_of_stock', False),
            log_filter_results=config_dict.get('log_filter_results', True),
            log_recommendations=config_dict.get('log_recommendations', True),
            include_quality_details=config_dict.get('include_quality_details', True),
            include_trust_details=config_dict.get('include_trust_details', True),
            include_relevance_details=config_dict.get('include_relevance_details', True),
            include_validation_details=config_dict.get('include_validation_details', True),
        )

        # Filters v2.0 초기화
        self.quality_filter = QualityFilter(
            config=QualityConfig(
                language="ko",
                min_length=30,
                optimal_min_length=100,
            )
        )
        self.trust_filter = TrustFilter(
            config=TrustConfig(
                min_trust_score=self.handler_config.min_trust_score,
            )
        )
        self.relevance_filter = RelevanceFilter(
            config=RelevanceConfig(
                min_relevance_score=self.handler_config.min_relevance_score,
            )
        )
        self.validation_filter = ValidationFilter(
            config=ValidationConfig(
                trust_config=TrustConfig(
                    min_trust_score=self.handler_config.min_trust_score,
                ),
                quality_config=QualityConfig(
                    language="ko",
                    min_length=30,
                ),
                relevance_config=RelevanceConfig(
                    min_relevance_score=self.handler_config.min_relevance_score,
                ),
                min_pass_score=self.handler_config.validation_pass_threshold,
                warning_threshold=self.handler_config.validation_warning_threshold,
            )
        )

        logger.info(
            f"[ProductRecommendationHandler] Initialized for {self.brand_id} "
            f"(Quality={self.handler_config.quality_filter_enabled}, "
            f"Trust={self.handler_config.trust_filter_enabled}, "
            f"Relevance={self.handler_config.relevance_filter_enabled}, "
            f"Validation={self.handler_config.validation_filter_enabled})"
        )

    def _extract_feature_config(self) -> Dict[str, Any]:
        return self.brand_config.get('product_recommendation', {})

    def can_handle(self, question: str, context: Dict[str, Any]) -> bool:
        """상품 추천 관련 질문인지 판단"""
        keywords = [
            '추천', '상품', '제품', '글러브', '복싱', '어떤', '좋은',
            '구매', '사고 싶', '살까', '뭐가', '괜찮', '인기',
            '베스트', '신상', '할인', '세일', '가격',
        ]
        question_lower = question.lower()

        return any(kw in question_lower for kw in keywords)

    def process(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        상품 추천 처리 및 품질 검증

        Args:
            question: 사용자 질문
            context: 컨텍스트

        Returns:
            처리 결과 (response, metadata, filter_results)
        """
        start_time = datetime.now()

        # 메타데이터 초기화
        metadata = {
            'handled_by': 'product_recommendation',
            'handler_version': '2.0',
            'processed_at': start_time.isoformat(),
        }

        # 검증 결과 저장소
        filter_results = {
            'quality': None,
            'trust': None,
            'relevance': None,
            'validation': None,
        }
        all_suggestions: List[str] = []

        try:
            # 상품 검색
            products = self._fetch_products(question)
            metadata['product_count'] = len(products)

            # 응답 생성
            if products:
                response = self._format_products(products, question)

                # 로그
                if self.handler_config.log_recommendations:
                    logger.info(
                        f"[ProductRecommendationHandler] Found {len(products)} products "
                        f"for question: {question[:50]}..."
                    )
            else:
                response = self._generate_no_product_response()

            # === Quality Filter 적용 ===
            if self.handler_config.quality_filter_enabled and response:
                quality_result = self._apply_quality_filter(response, context)
                filter_results['quality'] = quality_result

                if quality_result:
                    all_suggestions.extend(quality_result.get('suggestions', []))

                    if self.handler_config.include_quality_details:
                        metadata['quality'] = {
                            'score': quality_result.get('score'),
                            'level': quality_result.get('level'),
                            'valid': quality_result.get('valid'),
                        }

            # === Trust Filter 적용 ===
            if self.handler_config.trust_filter_enabled and response:
                trust_context = {
                    **context,
                    'retrieval_results': {'products': products},
                }
                trust_result = self._apply_trust_filter(response, question, trust_context)
                filter_results['trust'] = trust_result

                if trust_result:
                    if self.handler_config.include_trust_details:
                        metadata['trust'] = {
                            'score': trust_result.get('score'),
                            'level': trust_result.get('level'),
                            'hallucination_risk': trust_result.get('hallucination_risk'),
                            'valid': trust_result.get('valid'),
                        }

            # === Relevance Filter 적용 ===
            if self.handler_config.relevance_filter_enabled and response:
                relevance_result = self._apply_relevance_filter(response, question, context)
                filter_results['relevance'] = relevance_result

                if relevance_result:
                    if self.handler_config.include_relevance_details:
                        metadata['relevance'] = {
                            'score': relevance_result.get('score'),
                            'level': relevance_result.get('level'),
                            'response_type': relevance_result.get('response_type'),
                            'valid': relevance_result.get('valid'),
                        }

            # === Validation Filter 적용 (종합 검증) ===
            if self.handler_config.validation_filter_enabled and response:
                validation_result = self._apply_validation_filter(response, question, context)
                filter_results['validation'] = validation_result

                if validation_result:
                    all_suggestions.extend(validation_result.get('suggestions', []))

                    if self.handler_config.include_validation_details:
                        metadata['validation'] = {
                            'score': validation_result.get('score'),
                            'grade': validation_result.get('grade'),
                            'status': validation_result.get('status'),
                            'valid': validation_result.get('valid'),
                        }

                    # 종합 검증 실패 경고
                    if validation_result.get('status') == 'failed':
                        logger.warning(
                            f"[ProductRecommendationHandler] Validation failed: "
                            f"Grade={validation_result.get('grade')}, "
                            f"Score={validation_result.get('score'):.2f}"
                        )
                        metadata['validation_warning'] = True

            # 개선 제안 추가
            if self.handler_config.log_filter_results and all_suggestions:
                logger.info(
                    f"[ProductRecommendationHandler] Suggestions: "
                    f"{', '.join(all_suggestions[:3])}"
                )
                metadata['improvement_suggestions'] = all_suggestions

        except Exception as e:
            logger.error(f"[ProductRecommendationHandler] Error: {e}")
            response = "상품 추천 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
            metadata['error'] = str(e)

        # 처리 시간 기록
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        metadata['processing_time_ms'] = round(processing_time, 2)

        return {
            'response': response,
            'metadata': metadata,
            'filter_results': filter_results,
        }

    def _fetch_products(self, question: str) -> List[Dict[str, Any]]:
        """상품 조회"""
        try:
            from app.services.shared.neo4j import get_neo4j_client

            neo4j = get_neo4j_client()

            # 재고 있는 상품 조회
            stock_condition = "p.stock > 0" if not self.handler_config.include_out_of_stock else "1=1"

            query = f"""
            MATCH (p:Product)
            WHERE p.brand_id = $brand_id
              AND {stock_condition}
            RETURN p.id as id, p.name as name, p.price as price,
                   p.stock as stock, p.description as description,
                   p.category as category
            ORDER BY p.stock DESC
            LIMIT $limit
            """

            products = neo4j.query(query, {
                'brand_id': self.brand_id,
                'limit': self.handler_config.max_recommendations * 2,  # 여유분 확보
            })

            return products or []

        except Exception as e:
            logger.error(f"[ProductRecommendationHandler] Fetch error: {e}")
            return []

    def _format_products(self, products: List[Dict[str, Any]], question: str) -> str:
        """상품 목록 포맷팅"""
        lines = ["추천 상품을 안내해드립니다:\n"]

        for i, prod in enumerate(products[:self.handler_config.max_recommendations], 1):
            name = prod.get('name', 'Unknown')
            price = prod.get('price', 0)
            stock = prod.get('stock', 0)
            description = prod.get('description', '')

            lines.append(f"**{i}. {name}**")
            lines.append(f"   💰 가격: ₩{price:,}")
            lines.append(f"   📦 재고: {stock}개")

            if description:
                lines.append(f"   📝 {description[:50]}...")

            lines.append("")

        lines.append("궁금한 상품이 있으시면 말씀해주세요!")

        return "\n".join(lines)

    def _generate_no_product_response(self) -> str:
        """상품 없을 때 응답"""
        return (
            "죄송합니다. 현재 추천드릴 수 있는 상품이 없습니다.\n\n"
            "다음 방법을 시도해보세요:\n"
            "1. 다른 카테고리의 상품을 검색해보세요\n"
            "2. 구체적인 제품명으로 검색해보세요\n"
            "3. 잠시 후 다시 확인해주세요"
        )

    def _apply_quality_filter(
        self,
        response: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """품질 필터 적용"""
        try:
            result: QualityResult = self.quality_filter.validate(response, context)

            if self.handler_config.log_filter_results:
                logger.info(
                    f"[ProductRecommendationHandler:Quality] "
                    f"Score={result.score:.2f}, Level={result.level.value}"
                )

            return {
                'score': result.score,
                'level': result.level.value,
                'valid': result.valid,
                'suggestions': result.suggestions,
            }

        except Exception as e:
            logger.error(f"[ProductRecommendationHandler:Quality] Error: {e}")
            return None

    def _apply_trust_filter(
        self,
        response: str,
        question: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """신뢰성 필터 적용"""
        try:
            trust_context = {
                **context,
                'question': question,
            }

            result: TrustResult = self.trust_filter.validate(response, trust_context)

            if self.handler_config.log_filter_results:
                logger.info(
                    f"[ProductRecommendationHandler:Trust] "
                    f"Score={result.score:.2f}, "
                    f"HallucinationRisk={result.hallucination_risk:.2f}"
                )

            return {
                'score': result.score,
                'level': result.level.value,
                'hallucination_risk': result.hallucination_risk,
                'valid': result.valid,
            }

        except Exception as e:
            logger.error(f"[ProductRecommendationHandler:Trust] Error: {e}")
            return None

    def _apply_relevance_filter(
        self,
        response: str,
        question: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        관련성 필터 적용

        Args:
            response: 생성된 응답
            question: 원본 질문
            context: 컨텍스트

        Returns:
            관련성 검증 결과
        """
        try:
            # 질문을 컨텍스트에 추가
            relevance_context = {
                **context,
                'question': question,
            }

            result: RelevanceResult = self.relevance_filter.validate(response, relevance_context)

            # 로깅
            if self.handler_config.log_filter_results:
                logger.info(
                    f"[ProductRecommendationHandler:Relevance] "
                    f"Score={result.score:.2f}, "
                    f"Level={result.level.value}, "
                    f"ResponseType={result.response_type.value}, "
                    f"Valid={result.valid}"
                )

                # 이슈가 있으면 경고
                if result.issues:
                    for issue in result.issues[:3]:
                        logger.warning(
                            f"[ProductRecommendationHandler:Relevance] Issue: "
                            f"[{issue.relevance_type.value}] {issue.message}"
                        )

            return {
                'score': result.score,
                'level': result.level.value,
                'response_type': result.response_type.value,
                'valid': result.valid,
                'issues': [
                    {
                        'type': issue.relevance_type.value,
                        'severity': issue.severity.value,
                        'message': issue.message,
                    }
                    for issue in result.issues
                ],
                'scores': {
                    rtype.value: rs.score
                    for rtype, rs in result.scores.items()
                } if hasattr(result, 'scores') and result.scores else {},
            }

        except Exception as e:
            logger.error(f"[ProductRecommendationHandler:Relevance] Filter error: {e}")
            return None

    def _apply_validation_filter(
        self,
        response: str,
        question: str,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        종합 검증 필터 적용

        Args:
            response: 생성된 응답
            question: 원본 질문
            context: 컨텍스트

        Returns:
            종합 검증 결과
        """
        try:
            # 질문을 컨텍스트에 추가
            validation_context = {
                **context,
                'question': question,
            }

            result: ValidationResult = self.validation_filter.validate(response, validation_context)

            # 로깅
            if self.handler_config.log_filter_results:
                logger.info(
                    f"[ProductRecommendationHandler:Validation] "
                    f"Score={result.score:.2f}, "
                    f"Grade={result.grade.value}, "
                    f"Status={result.status.value}, "
                    f"Valid={result.valid}"
                )

                # 이슈가 있으면 경고
                if result.all_issues:
                    for issue in result.all_issues[:3]:
                        logger.warning(f"[ProductRecommendationHandler:Validation] Issue: {issue}")

            return {
                'score': result.score,
                'grade': result.grade.value,
                'status': result.status.value,
                'valid': result.valid,
                'summary': {
                    'total_issues': result.summary.total_issues,
                    'total_warnings': result.summary.total_warnings,
                    'passed_filters': result.summary.passed_filters,
                    'failed_filters': result.summary.failed_filters,
                },
                'issues': result.all_issues,
                'warnings': result.all_warnings,
                'suggestions': result.suggestions,
            }

        except Exception as e:
            logger.error(f"[ProductRecommendationHandler:Validation] Filter error: {e}")
            return None

    def get_filter_stats(self) -> Dict[str, Any]:
        """필터 통계 반환"""
        return {
            'quality_filter_enabled': self.handler_config.quality_filter_enabled,
            'trust_filter_enabled': self.handler_config.trust_filter_enabled,
            'relevance_filter_enabled': self.handler_config.relevance_filter_enabled,
            'validation_filter_enabled': self.handler_config.validation_filter_enabled,
            'min_quality_score': self.handler_config.min_quality_score,
            'min_trust_score': self.handler_config.min_trust_score,
            'min_relevance_score': self.handler_config.min_relevance_score,
            'validation_pass_threshold': self.handler_config.validation_pass_threshold,
            'max_recommendations': self.handler_config.max_recommendations,
        }
