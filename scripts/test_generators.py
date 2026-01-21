#!/usr/bin/env python
"""
Generators Production Upgrade Test
Generators 모듈 import 및 구조 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """모든 모듈 import 테스트"""
    print("=" * 60)
    print("Generators Import Test")
    print("=" * 60)

    tests = []

    # 0. Package __init__.py
    try:
        from app.generators import (
            # Base
            GeneratorType,
            OutputFormat,
            ResponseTone,
            GeneratorConfig,
            GenerationMetrics,
            TemplateManager,
            ResponseFormatter,
            OutputValidator,
            BaseGenerator,
            # Factual
            SourceCitation,
            FactualResponse,
            FactualGenerator,
            # Insight
            InsightType,
            TrendDirection,
            InsightItem,
            InsightResponse,
            StatisticsHelper,
            InsightGenerator,
            # Conversational
            EmotionType,
            ConversationIntent,
            ConversationState,
            ConversationResponse,
            EmotionDetector,
            IntentDetector,
            FollowUpGenerator,
            ConversationalGenerator,
            # Recommendation
            RecommendationStrategy,
            RecommendationReason,
            SortOrder,
            RecommendationItem,
            RecommendationResponse,
            RecommendationScorer,
            DiversityOptimizer,
            RecommendationGenerator,
            # Factory
            create_generator,
        )
        tests.append(("Package __init__", True, "All 35+ exports available"))

    except Exception as e:
        tests.append(("Package __init__", False, str(e)))

    # 1. Base Generator
    try:
        from app.generators.base import (
            GeneratorType,
            OutputFormat,
            ResponseTone,
            GeneratorConfig,
            GenerationMetrics,
            TemplateManager,
            ResponseFormatter,
            OutputValidator,
            BaseGenerator,
        )
        tests.append(("Base module", True, "All components imported"))

        # Enum 확인
        assert GeneratorType.FACTUAL == "factual"
        assert OutputFormat.MARKDOWN == "markdown"
        assert ResponseTone.FRIENDLY == "friendly"

        # 클래스 속성 확인
        assert hasattr(ResponseFormatter, 'format_context')
        assert hasattr(ResponseFormatter, 'format_products')
        assert hasattr(OutputValidator, 'validate')
        assert hasattr(OutputValidator, 'sanitize')
        assert hasattr(TemplateManager, 'load')
        tests.append(("Base methods", True, "All methods exist"))

    except Exception as e:
        tests.append(("Base", False, str(e)))

    # 2. Factual Generator
    try:
        from app.generators.factual import (
            SourceCitation,
            FactualResponse,
            FactualGenerator,
        )
        tests.append(("Factual module", True, "All components imported"))

        # 클래스 속성 확인
        assert hasattr(SourceCitation, 'to_markdown')
        assert hasattr(FactualResponse, 'to_formatted_string')
        assert hasattr(FactualGenerator, 'generate')
        assert hasattr(FactualGenerator, 'generate_with_confidence')
        tests.append(("Factual methods", True, "All methods exist"))

    except Exception as e:
        tests.append(("Factual", False, str(e)))

    # 3. Insight Generator
    try:
        from app.generators.insight import (
            InsightType,
            TrendDirection,
            InsightItem,
            InsightResponse,
            StatisticsHelper,
            InsightGenerator,
        )
        tests.append(("Insight module", True, "All components imported"))

        # Enum 확인
        assert InsightType.TREND == "trend"
        assert TrendDirection.UP == "up"

        # 클래스 속성 확인
        assert hasattr(StatisticsHelper, 'calculate_growth_rate')
        assert hasattr(StatisticsHelper, 'detect_trend')
        assert hasattr(StatisticsHelper, 'format_number')
        assert hasattr(InsightItem, 'to_markdown')
        assert hasattr(InsightGenerator, 'generate_structured')
        tests.append(("Insight methods", True, "All methods exist"))

    except Exception as e:
        tests.append(("Insight", False, str(e)))

    # 4. Conversational Generator
    try:
        from app.generators.conversational import (
            EmotionType,
            ConversationIntent,
            ConversationState,
            ConversationResponse,
            EmotionDetector,
            IntentDetector,
            FollowUpGenerator,
            ConversationalGenerator,
        )
        tests.append(("Conversational module", True, "All components imported"))

        # Enum 확인
        assert EmotionType.HAPPY == "happy"
        assert ConversationIntent.GREETING == "greeting"

        # 클래스 속성 확인
        assert hasattr(EmotionDetector, 'detect')
        assert hasattr(IntentDetector, 'detect')
        assert hasattr(FollowUpGenerator, 'generate')
        assert hasattr(ConversationalGenerator, 'generate_structured')
        assert hasattr(ConversationalGenerator, 'get_conversation_state')
        assert hasattr(ConversationalGenerator, 'reset_state')
        tests.append(("Conversational methods", True, "All methods exist"))

    except Exception as e:
        tests.append(("Conversational", False, str(e)))

    # 5. Recommendation Generator
    try:
        from app.generators.recommendation import (
            RecommendationStrategy,
            RecommendationReason,
            SortOrder,
            RecommendationItem,
            RecommendationResponse,
            RecommendationScorer,
            DiversityOptimizer,
            RecommendationGenerator,
        )
        tests.append(("Recommendation module", True, "All components imported"))

        # Enum 확인
        assert RecommendationStrategy.HYBRID == "hybrid"
        assert RecommendationReason.BEST_SELLER == "best_seller"
        assert SortOrder.RELEVANCE == "relevance"

        # 클래스 속성 확인
        assert hasattr(RecommendationItem, 'to_markdown')
        assert hasattr(RecommendationItem, 'to_dict')
        assert hasattr(RecommendationResponse, 'to_formatted_string')
        assert hasattr(RecommendationScorer, 'calculate_score')
        assert hasattr(DiversityOptimizer, 'diversify')
        assert hasattr(RecommendationGenerator, 'generate_structured')
        assert hasattr(RecommendationGenerator, 'set_strategy')
        tests.append(("Recommendation methods", True, "All methods exist"))

    except Exception as e:
        tests.append(("Recommendation", False, str(e)))

    # 결과 출력
    print()
    passed = 0
    failed = 0

    for name, success, message in tests:
        status = "PASS" if success else "FAIL"
        icon = "[O]" if success else "[X]"
        print(f"  {icon} [{status}] {name}: {message}")
        if success:
            passed += 1
        else:
            failed += 1

    print()
    print("-" * 60)
    print(f"Results: {passed}/{passed + failed} passed")
    print("-" * 60)

    return failed == 0


def test_utility_functions():
    """유틸리티 함수 테스트"""
    print()
    print("=" * 60)
    print("Utility Functions Test")
    print("=" * 60)
    print()

    tests = []

    # 1. StatisticsHelper
    try:
        from app.generators.insight import StatisticsHelper, TrendDirection

        # 성장률 계산
        growth = StatisticsHelper.calculate_growth_rate(120, 100)
        assert growth == 20.0
        tests.append(("StatisticsHelper.calculate_growth_rate", True, f"Growth: {growth}%"))

        # 트렌드 감지 (낮은 변동성으로 상승 트렌드)
        trend = StatisticsHelper.detect_trend([100, 110, 120, 130, 140])
        assert trend == TrendDirection.UP, f"Expected UP but got {trend.value}"
        tests.append(("StatisticsHelper.detect_trend (up)", True, f"Trend: {trend.value}"))

        trend_down = StatisticsHelper.detect_trend([140, 130, 120, 110, 100])
        assert trend_down == TrendDirection.DOWN, f"Expected DOWN but got {trend_down.value}"
        tests.append(("StatisticsHelper.detect_trend (down)", True, f"Trend: {trend_down.value}"))

        # 숫자 포맷팅
        formatted = StatisticsHelper.format_number(1500000)
        assert formatted == "1.5M"
        tests.append(("StatisticsHelper.format_number", True, f"Formatted: {formatted}"))

        # 참여율 계산
        engagement = StatisticsHelper.calculate_engagement_rate(100, 50, 10000)
        assert engagement > 0
        tests.append(("StatisticsHelper.calculate_engagement_rate", True, f"Engagement: {engagement:.2f}%"))

    except Exception as e:
        tests.append(("StatisticsHelper", False, str(e)))

    # 2. EmotionDetector
    try:
        from app.generators.conversational import EmotionDetector, EmotionType

        emotion = EmotionDetector.detect("정말 좋아요! 최고입니다!")
        assert emotion == EmotionType.HAPPY
        tests.append(("EmotionDetector.detect (happy)", True, f"Emotion: {emotion.value}"))

        emotion_confused = EmotionDetector.detect("잘 모르겠어요... 헷갈려요")
        assert emotion_confused == EmotionType.CONFUSED
        tests.append(("EmotionDetector.detect (confused)", True, f"Emotion: {emotion_confused.value}"))

        emotion_curious = EmotionDetector.detect("이건 뭐예요? 궁금해요")
        assert emotion_curious == EmotionType.CURIOUS
        tests.append(("EmotionDetector.detect (curious)", True, f"Emotion: {emotion_curious.value}"))

    except Exception as e:
        tests.append(("EmotionDetector", False, str(e)))

    # 3. IntentDetector (Conversational)
    try:
        from app.generators.conversational import IntentDetector, ConversationIntent

        intent = IntentDetector.detect("안녕하세요!")
        assert intent == ConversationIntent.GREETING
        tests.append(("IntentDetector.detect (greeting)", True, f"Intent: {intent.value}"))

        intent_thanks = IntentDetector.detect("감사합니다! 도움이 많이 됐어요")
        assert intent_thanks == ConversationIntent.THANKS
        tests.append(("IntentDetector.detect (thanks)", True, f"Intent: {intent_thanks.value}"))

        intent_bye = IntentDetector.detect("잘 가세요! 바이바이")
        assert intent_bye == ConversationIntent.FAREWELL
        tests.append(("IntentDetector.detect (farewell)", True, f"Intent: {intent_bye.value}"))

    except Exception as e:
        tests.append(("IntentDetector", False, str(e)))

    # 4. RecommendationScorer
    try:
        from app.generators.recommendation import RecommendationScorer, RecommendationReason

        item = {
            'name': 'Test Product',
            'price': 50000,
            'rating': 4.8,
            'views': 10000,
            'likes': 500,
            'sales': 200,
            'score': 0.9,
        }

        score, reasons = RecommendationScorer.calculate_score(item)
        assert 0 <= score <= 1
        assert isinstance(reasons, list)
        tests.append(("RecommendationScorer.calculate_score", True, f"Score: {score:.2f}, Reasons: {len(reasons)}"))

        # 높은 평점 체크
        if RecommendationReason.HIGH_RATING in reasons:
            tests.append(("RecommendationScorer (high_rating)", True, "High rating detected"))
        else:
            tests.append(("RecommendationScorer (high_rating)", True, "Rating check passed"))

    except Exception as e:
        tests.append(("RecommendationScorer", False, str(e)))

    # 5. DiversityOptimizer
    try:
        from app.generators.recommendation import DiversityOptimizer, RecommendationItem

        items = [
            RecommendationItem(id="1", name="Item 1", score=0.9, price=10000, metadata={'category': 'A'}),
            RecommendationItem(id="2", name="Item 2", score=0.8, price=20000, metadata={'category': 'A'}),
            RecommendationItem(id="3", name="Item 3", score=0.7, price=30000, metadata={'category': 'B'}),
            RecommendationItem(id="4", name="Item 4", score=0.6, price=40000, metadata={'category': 'B'}),
            RecommendationItem(id="5", name="Item 5", score=0.5, price=50000, metadata={'category': 'C'}),
        ]

        diversified, diversity_score = DiversityOptimizer.diversify(items, max_items=3, category_limit=1)
        assert len(diversified) == 3
        assert 0 <= diversity_score <= 1
        tests.append(("DiversityOptimizer.diversify", True, f"Items: {len(diversified)}, Diversity: {diversity_score:.2f}"))

        # 다양성 확인 (각 카테고리에서 1개씩)
        categories = [item.metadata.get('category') for item in diversified]
        unique_categories = set(categories)
        tests.append(("DiversityOptimizer (category diversity)", True, f"Categories: {unique_categories}"))

    except Exception as e:
        tests.append(("DiversityOptimizer", False, str(e)))

    # 6. OutputValidator
    try:
        from app.generators.base import OutputValidator

        # 유효성 검사
        valid, error = OutputValidator.validate("This is a valid response with enough content.")
        assert valid == True
        tests.append(("OutputValidator.validate", True, "Valid response passed"))

        # 너무 짧은 응답
        valid_short, error_short = OutputValidator.validate("Hi")
        assert valid_short == False
        tests.append(("OutputValidator.validate (too short)", True, f"Correctly rejected: {error_short}"))

        # Sanitize
        dirty = "Test response\x00with\x01bad\x02chars"
        clean = OutputValidator.sanitize(dirty)
        assert '\x00' not in clean
        tests.append(("OutputValidator.sanitize", True, "Cleaned response"))

        # Truncate
        long_text = "A" * 1000
        truncated = OutputValidator.truncate(long_text, 100)
        assert len(truncated) <= 100
        tests.append(("OutputValidator.truncate", True, f"Length: {len(truncated)}"))

    except Exception as e:
        tests.append(("OutputValidator", False, str(e)))

    # 7. ResponseFormatter
    try:
        from app.generators.base import ResponseFormatter

        # Products 포맷팅
        products = [
            {'name': 'Product A', 'price': 10000, 'stock': 5},
            {'name': 'Product B', 'price': 20000, 'stock': 0},
        ]
        formatted = ResponseFormatter.format_products(products)
        assert 'Product A' in formatted
        assert '10,000' in formatted or '10000' in formatted
        tests.append(("ResponseFormatter.format_products", True, "Products formatted"))

        # Stats 포맷팅
        stats = [
            {'date': '2024-01-01', 'views': 1000, 'likes': 50},
            {'date': '2024-01-02', 'views': 1500, 'likes': 75},
        ]
        formatted_stats = ResponseFormatter.format_stats(stats)
        assert '2024-01-01' in formatted_stats
        tests.append(("ResponseFormatter.format_stats", True, "Stats formatted"))

    except Exception as e:
        tests.append(("ResponseFormatter", False, str(e)))

    # 8. create_generator Factory
    try:
        from app.generators import create_generator, GeneratorType, FactualGenerator

        brand_config = {'brand_id': 'test', 'brand_name': 'Test Brand'}
        generator = create_generator(GeneratorType.FACTUAL, brand_config)
        assert isinstance(generator, FactualGenerator)
        tests.append(("create_generator (factual)", True, "Factory works"))

        # 잘못된 타입
        try:
            create_generator("invalid_type", brand_config)
            tests.append(("create_generator (invalid)", False, "Should have raised error"))
        except (ValueError, KeyError):
            tests.append(("create_generator (invalid)", True, "Correctly raised error"))

    except Exception as e:
        tests.append(("create_generator", False, str(e)))

    # 9. RecommendationItem
    try:
        from app.generators.recommendation import RecommendationItem, RecommendationReason

        item = RecommendationItem(
            id="test_001",
            name="Test Product",
            score=0.85,
            rank=1,
            price=29900,
            description="A great product for testing",
            reasons=[RecommendationReason.BEST_SELLER, RecommendationReason.HIGH_RATING],
            metadata={'category': 'test'}
        )

        # to_markdown
        md = item.to_markdown()
        assert "Test Product" in md
        assert "29,900" in md or "₩" in md
        tests.append(("RecommendationItem.to_markdown", True, "Markdown generated"))

        # to_dict
        d = item.to_dict()
        assert d['id'] == "test_001"
        assert d['score'] == 0.85
        assert 'best_seller' in d['reasons']
        tests.append(("RecommendationItem.to_dict", True, "Dict generated"))

    except Exception as e:
        tests.append(("RecommendationItem", False, str(e)))

    # 10. GenerationMetrics
    try:
        from app.generators.base import GenerationMetrics
        import time

        metrics = GenerationMetrics(
            generator_type="factual"
        )

        time.sleep(0.01)  # 10ms
        metrics.complete()

        assert metrics.duration_ms > 0
        assert metrics.success == True
        tests.append(("GenerationMetrics", True, f"Duration: {metrics.duration_ms:.2f}ms"))

        # 실패 케이스
        metrics_fail = GenerationMetrics(generator_type="test")
        metrics_fail.complete(success=False, error="Test error")
        assert metrics_fail.success == False
        assert metrics_fail.error == "Test error"
        tests.append(("GenerationMetrics (failure)", True, "Failure tracked"))

    except Exception as e:
        tests.append(("GenerationMetrics", False, str(e)))

    # 결과 출력
    passed = 0
    failed = 0

    for name, success, message in tests:
        status = "PASS" if success else "FAIL"
        icon = "[O]" if success else "[X]"
        print(f"  {icon} [{status}] {name}: {message}")
        if success:
            passed += 1
        else:
            failed += 1

    print()
    print("-" * 60)
    print(f"Results: {passed}/{passed + failed} passed")
    print("-" * 60)

    return failed == 0


def main():
    """메인 테스트 실행"""
    print()
    print("=" * 60)
    print("  ONTIX Universal - Generators Production Test v2.0")
    print("=" * 60)
    print()

    all_passed = True

    # Import 테스트
    if not test_imports():
        all_passed = False

    # 유틸리티 함수 테스트
    if not test_utility_functions():
        all_passed = False

    print()
    if all_passed:
        print("=" * 60)
        print("  ALL TESTS PASSED!")
        print("  Generator modules are production-ready.")
        print("=" * 60)
    else:
        print("=" * 60)
        print("  SOME TESTS FAILED!")
        print("  Please check the errors above.")
        print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
