#!/usr/bin/env python
"""
Retrievers Production Upgrade Test
모든 Retriever 모듈 import 및 구조 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """모든 모듈 import 테스트"""
    print("=" * 60)
    print("Retrievers Import Test")
    print("=" * 60)

    tests = []

    # 0. Package __init__.py FIRST (to avoid circular import)
    try:
        from app.retrievers import (
            GraphRetriever as _GR,
            VectorRetriever as _VR,
            HybridRetriever as _HR,
            StatsRetriever as _SR,
            ProductRetriever as _PR,
            FusionMethod,
            SearchScope,
            StatsType,
            ProductSortBy,
        )
        tests.append(("Package __init__", True, "All exports available"))

    except Exception as e:
        tests.append(("Package __init__", False, str(e)))

    # 1. GraphRetriever
    try:
        from app.retrievers.graph import (
            GraphRetriever,
            SearchScope,
            GraphSearchConfig,
            SearchResult,
            KeywordExtractor,
        )
        tests.append(("GraphRetriever", True, "All components imported"))

        # 클래스 속성 확인
        assert hasattr(GraphRetriever, 'retrieve')
        assert hasattr(GraphRetriever, 'configure')
        assert hasattr(KeywordExtractor, 'extract')
        assert hasattr(KeywordExtractor, 'KOREAN_STOPWORDS')
        tests.append(("GraphRetriever methods", True, "All methods exist"))

    except Exception as e:
        tests.append(("GraphRetriever", False, str(e)))

    # 2. VectorRetriever
    try:
        from app.retrievers.vector import (
            VectorRetriever,
            SearchMode,
            NodeIndex,
            VectorSearchConfig,
            VectorResult,
            QueryPreprocessor,
            ScoreNormalizer,
        )
        tests.append(("VectorRetriever", True, "All components imported"))

        # 클래스 속성 확인
        assert hasattr(VectorRetriever, 'retrieve')
        assert hasattr(VectorRetriever, 'configure')
        assert hasattr(VectorRetriever, 'clear_embedding_cache')
        assert hasattr(QueryPreprocessor, 'preprocess')
        assert hasattr(QueryPreprocessor, 'expand')
        assert hasattr(ScoreNormalizer, 'boost_by_engagement')
        tests.append(("VectorRetriever methods", True, "All methods exist"))

    except Exception as e:
        tests.append(("VectorRetriever", False, str(e)))

    # 3. HybridRetriever
    try:
        from app.retrievers.hybrid import (
            HybridRetriever,
            FusionMethod,
            HybridSearchConfig,
            HybridResult,
            RRFCalculator,
            ResultMerger,
        )
        tests.append(("HybridRetriever", True, "All components imported"))

        # 클래스 속성 확인
        assert hasattr(HybridRetriever, 'retrieve')
        assert hasattr(HybridRetriever, 'configure')
        assert hasattr(HybridRetriever, 'get_debug_info')
        assert hasattr(RRFCalculator, 'calculate')
        assert hasattr(ResultMerger, 'deduplicate')

        # Enum 확인
        assert FusionMethod.RRF == "rrf"
        assert FusionMethod.WEIGHTED_SUM == "weighted_sum"
        tests.append(("HybridRetriever methods", True, "All methods exist"))

    except Exception as e:
        tests.append(("HybridRetriever", False, str(e)))

    # 4. StatsRetriever
    try:
        from app.retrievers.stats import (
            StatsRetriever,
            StatsType,
            TimePeriod,
            SortMetric,
            StatsConfig,
            StatsResult,
            MetricsCalculator,
            TimeHelper,
        )
        tests.append(("StatsRetriever", True, "All components imported"))

        # 클래스 속성 확인
        assert hasattr(StatsRetriever, 'retrieve')
        assert hasattr(StatsRetriever, 'configure')
        assert hasattr(MetricsCalculator, 'engagement_rate')
        assert hasattr(MetricsCalculator, 'growth_rate')
        assert hasattr(TimeHelper, 'get_days')

        # Enum 확인
        assert StatsType.POPULAR == "popular"
        assert TimePeriod.WEEK == "week"
        tests.append(("StatsRetriever methods", True, "All methods exist"))

    except Exception as e:
        tests.append(("StatsRetriever", False, str(e)))

    # 5. ProductRetriever
    try:
        from app.retrievers.product import (
            ProductRetriever,
            ProductSearchMode,
            ProductSortBy,
            StockStatus,
            ProductSearchConfig,
            ProductResult,
            PriceParser,
            QueryAnalyzer,
        )
        tests.append(("ProductRetriever", True, "All components imported"))

        # 클래스 속성 확인
        assert hasattr(ProductRetriever, 'retrieve')
        assert hasattr(ProductRetriever, 'configure')
        assert hasattr(ProductRetriever, 'get_popular_products')
        assert hasattr(ProductRetriever, 'get_products_by_concept')
        assert hasattr(PriceParser, 'extract_price_range')
        assert hasattr(QueryAnalyzer, 'detect_categories')
        tests.append(("ProductRetriever methods", True, "All methods exist"))

    except Exception as e:
        tests.append(("ProductRetriever", False, str(e)))

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

    # 1. KeywordExtractor
    try:
        from app.retrievers.graph import KeywordExtractor

        keywords = KeywordExtractor.extract("보습 세럼 추천해줘")
        assert len(keywords) > 0
        assert "보습" in keywords or "세럼" in keywords
        tests.append(("KeywordExtractor.extract", True, f"Extracted: {keywords}"))

        scored = KeywordExtractor.extract_with_scores("피부 트러블에 좋은 제품")
        assert len(scored) > 0
        tests.append(("KeywordExtractor.extract_with_scores", True, f"Scored: {scored[:3]}"))

    except Exception as e:
        tests.append(("KeywordExtractor", False, str(e)))

    # 2. QueryPreprocessor
    try:
        from app.retrievers.vector import QueryPreprocessor

        processed = QueryPreprocessor.preprocess("  인스타그램  보습  크림!!  ")
        assert "인스타그램" in processed
        tests.append(("QueryPreprocessor.preprocess", True, f"Result: '{processed}'"))

        expanded = QueryPreprocessor.expand("보습 세럼")
        assert len(expanded) > len("보습 세럼")
        tests.append(("QueryPreprocessor.expand", True, f"Expanded: '{expanded}'"))

        query, filters = QueryPreprocessor.extract_filters("인스타그램 이번주 인기 게시물")
        tests.append(("QueryPreprocessor.extract_filters", True, f"Filters: {filters}"))

    except Exception as e:
        tests.append(("QueryPreprocessor", False, str(e)))

    # 3. ScoreNormalizer
    try:
        from app.retrievers.vector import ScoreNormalizer

        normalized = ScoreNormalizer.normalize(0.75, 0.0, 1.0)
        assert 0 <= normalized <= 1
        tests.append(("ScoreNormalizer.normalize", True, f"Result: {normalized}"))

        boosted = ScoreNormalizer.boost_by_engagement(0.5, {'likes': 1000, 'views': 50000})
        assert boosted > 0.5
        tests.append(("ScoreNormalizer.boost_by_engagement", True, f"Boosted: {boosted}"))

    except Exception as e:
        tests.append(("ScoreNormalizer", False, str(e)))

    # 4. RRFCalculator
    try:
        from app.retrievers.hybrid import RRFCalculator

        rankings = [
            [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)],
            [("doc2", 0.95), ("doc1", 0.85), ("doc4", 0.75)],
        ]
        weights = [0.6, 0.4]

        rrf_scores = RRFCalculator.calculate(rankings, weights, k=60)
        assert "doc1" in rrf_scores
        assert "doc2" in rrf_scores
        tests.append(("RRFCalculator.calculate", True, f"Scores: {dict(list(rrf_scores.items())[:3])}"))

    except Exception as e:
        tests.append(("RRFCalculator", False, str(e)))

    # 5. MetricsCalculator
    try:
        from app.retrievers.stats import MetricsCalculator

        engagement = MetricsCalculator.engagement_rate(1000, 50, 10000)
        assert 0 <= engagement <= 100
        tests.append(("MetricsCalculator.engagement_rate", True, f"Rate: {engagement}%"))

        growth = MetricsCalculator.growth_rate(150, 100)
        assert growth == 50.0
        tests.append(("MetricsCalculator.growth_rate", True, f"Growth: {growth}%"))

        trend = MetricsCalculator.trend_score(200, 100)
        assert trend == 2.0
        tests.append(("MetricsCalculator.trend_score", True, f"Trend: {trend}x"))

    except Exception as e:
        tests.append(("MetricsCalculator", False, str(e)))

    # 6. PriceParser
    try:
        from app.retrievers.product import PriceParser

        min_p, max_p = PriceParser.extract_price_range("5만원대 세럼")
        assert min_p == 50000 and max_p == 60000
        tests.append(("PriceParser (만원대)", True, f"Range: {min_p}-{max_p}"))

        min_p, max_p = PriceParser.extract_price_range("10만원 이하")
        assert min_p is None and max_p == 100000
        tests.append(("PriceParser (이하)", True, f"Range: {min_p}-{max_p}"))

    except Exception as e:
        tests.append(("PriceParser", False, str(e)))

    # 7. QueryAnalyzer
    try:
        from app.retrievers.product import QueryAnalyzer

        categories = QueryAnalyzer.detect_categories("세럼 추천해줘")
        assert "세럼" in categories
        tests.append(("QueryAnalyzer.detect_categories", True, f"Categories: {categories}"))

        intents = QueryAnalyzer.detect_intent("저렴한 크림 추천")
        assert intents.get('cheap') == True
        tests.append(("QueryAnalyzer.detect_intent", True, f"Intents: {intents}"))

    except Exception as e:
        tests.append(("QueryAnalyzer", False, str(e)))

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
    print("  ONTIX Universal - Retrievers Production Test v2.0")
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
        print("  Retrievers are production-ready.")
        print("=" * 60)
    else:
        print("=" * 60)
        print("  SOME TESTS FAILED!")
        print("  Please check the errors above.")
        print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
