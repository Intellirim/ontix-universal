#!/usr/bin/env python
"""
Core Production Upgrade Test
Core 모듈 import 및 구조 테스트
"""

import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """모든 모듈 import 테스트"""
    print("=" * 60)
    print("Core Import Test")
    print("=" * 60)

    tests = []

    # 0. Package __init__.py
    try:
        from app.core import (
            QueryContext,
            QuestionType,
            ProcessingStage,
            QuestionRouter,
            IntentType,
            Pipeline,
            PipelineStage,
            UniversalEngine,
            EngineState,
        )
        tests.append(("Package __init__", True, "All exports available"))

    except Exception as e:
        tests.append(("Package __init__", False, str(e)))

    # 1. Context
    try:
        from app.core.context import (
            QueryContext,
            QuestionType,
            ProcessingStage,
            RetrievalResult,
            PerformanceMetrics,
            ConversationMessage,
        )
        tests.append(("Context module", True, "All components imported"))

        # Enum 확인
        assert QuestionType.PRODUCT_RECOMMENDATION == "product_recommendation"
        assert ProcessingStage.COMPLETED == "completed"

        # 클래스 속성 확인
        assert hasattr(QueryContext, 'add_retrieval_result')
        assert hasattr(QueryContext, 'set_question_type')
        assert hasattr(QueryContext, 'to_dict')
        assert hasattr(QueryContext, 'from_dict')
        assert hasattr(PerformanceMetrics, 'mark_routing_complete')
        tests.append(("Context methods", True, "All methods exist"))

    except Exception as e:
        tests.append(("Context", False, str(e)))

    # 2. Routing
    try:
        from app.core.routing import (
            QuestionRouter,
            IntentType,
            ClassificationResult,
            RouterConfig,
            PatternMatcher,
            EntityExtractor,
            IntentDetector,
        )
        tests.append(("Routing module", True, "All components imported"))

        # Enum 확인
        assert IntentType.RECOMMENDATION == "recommendation"
        assert IntentType.COMPARISON == "comparison"

        # 클래스 속성 확인
        assert hasattr(QuestionRouter, 'route')
        assert hasattr(QuestionRouter, 'clear_cache')
        assert hasattr(QuestionRouter, 'get_stats')
        assert hasattr(PatternMatcher, 'match')
        assert hasattr(PatternMatcher, 'PATTERNS')
        assert hasattr(EntityExtractor, 'extract')
        assert hasattr(IntentDetector, 'detect')
        tests.append(("Routing methods", True, "All methods exist"))

    except Exception as e:
        tests.append(("Routing", False, str(e)))

    # 3. Pipeline
    try:
        from app.core.pipeline import (
            Pipeline,
            PipelineStage,
            RetrievalMode,
            FallbackStrategy,
            PipelineConfig,
            StepMetrics,
            PipelineTrace,
            PromptManager,
            PipelineHook,
            ContentFilterHook,
            ContextTruncationHook,
        )
        tests.append(("Pipeline module", True, "All components imported"))

        # Enum 확인
        assert PipelineStage.RETRIEVAL == "retrieval"
        assert RetrievalMode.PARALLEL == "parallel"
        assert FallbackStrategy.DEFAULT_RESPONSE == "default_response"

        # 클래스 속성 확인
        assert hasattr(Pipeline, 'execute')
        assert hasattr(Pipeline, '_retrieve')
        assert hasattr(Pipeline, '_generate')
        assert hasattr(Pipeline, 'add_hook')
        assert hasattr(Pipeline, 'get_debug_info')
        assert hasattr(PromptManager, 'load')
        assert hasattr(PromptManager, 'clear_cache')
        tests.append(("Pipeline methods", True, "All methods exist"))

    except Exception as e:
        tests.append(("Pipeline", False, str(e)))

    # 4. Engine
    try:
        from app.core.engine import (
            UniversalEngine,
            EngineState,
            ErrorType,
            EngineConfig,
            RequestMetrics,
            EngineMetrics,
            RateLimiter,
            EngineError,
            Middleware,
            LoggingMiddleware,
            ValidationMiddleware,
        )
        tests.append(("Engine module", True, "All components imported"))

        # Enum 확인
        assert EngineState.READY == "ready"
        assert ErrorType.RATE_LIMIT == "rate_limit"

        # 클래스 속성 확인
        assert hasattr(UniversalEngine, 'get_instance')
        assert hasattr(UniversalEngine, 'ask')
        assert hasattr(UniversalEngine, 'ask_stream')
        assert hasattr(UniversalEngine, 'ask_async')
        assert hasattr(UniversalEngine, 'health_check')
        assert hasattr(UniversalEngine, 'add_middleware')
        assert hasattr(EngineMetrics, 'record_request')
        assert hasattr(EngineMetrics, 'get_summary')
        assert hasattr(RateLimiter, 'is_allowed')
        tests.append(("Engine methods", True, "All methods exist"))

    except Exception as e:
        tests.append(("Engine", False, str(e)))

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

    # 1. QueryContext
    try:
        from app.core.context import QueryContext, QuestionType

        ctx = QueryContext(brand_id="test_brand", question="테스트 질문")
        assert ctx.brand_id == "test_brand"
        assert ctx.question == "테스트 질문"
        assert ctx.cache_key is not None
        tests.append(("QueryContext creation", True, f"cache_key={ctx.cache_key}"))

        # 검색 결과 추가
        ctx.add_retrieval_result(
            source="test_source",
            data=[{"id": 1, "text": "test"}],
            score=0.95
        )
        assert ctx.get_total_retrieval_count() == 1
        assert ctx.has_retrieval_result("test_source")
        tests.append(("QueryContext retrieval", True, f"count={ctx.get_total_retrieval_count()}"))

        # 직렬화
        ctx_dict = ctx.to_dict()
        assert 'brand_id' in ctx_dict
        assert 'question' in ctx_dict
        tests.append(("QueryContext to_dict", True, "Serializable"))

    except Exception as e:
        tests.append(("QueryContext", False, str(e)))

    # 2. PatternMatcher
    try:
        from app.core.routing import PatternMatcher
        from app.core.context import QuestionType

        qtype, confidence, patterns = PatternMatcher.match("보습 세럼 추천해줘")
        assert qtype == QuestionType.PRODUCT_RECOMMENDATION
        assert confidence > 0.5
        tests.append(("PatternMatcher.match", True, f"Type: {qtype.value}, Conf: {confidence:.2f}"))

        qtype2, confidence2, patterns2 = PatternMatcher.match("인기있는 크림 알려줘")
        if qtype2:
            tests.append(("PatternMatcher (popular query)", True, f"Type: {qtype2.value}, Conf: {confidence2:.2f}"))
        else:
            tests.append(("PatternMatcher (popular query)", True, "No pattern match"))

    except Exception as e:
        tests.append(("PatternMatcher", False, str(e)))

    # 3. EntityExtractor
    try:
        from app.core.routing import EntityExtractor

        entities = EntityExtractor.extract("지성 피부에 좋은 세럼 추천해줘")
        assert 'skin_type' in entities or 'product_category' in entities
        tests.append(("EntityExtractor.extract", True, f"Entities: {entities}"))

    except Exception as e:
        tests.append(("EntityExtractor", False, str(e)))

    # 4. IntentDetector
    try:
        from app.core.routing import IntentDetector, IntentType

        intent = IntentDetector.detect("이 두 제품 비교해줘")
        assert isinstance(intent, IntentType)
        tests.append(("IntentDetector.detect", True, f"Intent: {intent.value}"))

    except Exception as e:
        tests.append(("IntentDetector", False, str(e)))

    # 5. PerformanceMetrics
    try:
        from app.core.context import PerformanceMetrics
        import time

        metrics = PerformanceMetrics()
        time.sleep(0.01)  # 10ms
        metrics.mark_routing_complete()
        assert metrics.routing_time_ms > 0
        tests.append(("PerformanceMetrics", True, f"routing_time={metrics.routing_time_ms:.2f}ms"))

    except Exception as e:
        tests.append(("PerformanceMetrics", False, str(e)))

    # 6. RateLimiter
    try:
        from app.core.engine import RateLimiter

        limiter = RateLimiter(max_requests=5, window_seconds=60)

        # 5번 호출 허용
        for i in range(5):
            assert limiter.is_allowed() == True

        # 6번째는 거부
        assert limiter.is_allowed() == False
        assert limiter.get_remaining() == 0

        tests.append(("RateLimiter", True, "Rate limiting works"))

    except Exception as e:
        tests.append(("RateLimiter", False, str(e)))

    # 7. EngineMetrics
    try:
        from app.core.engine import EngineMetrics, RequestMetrics
        import time

        metrics = EngineMetrics()

        # 요청 기록
        req = RequestMetrics(
            request_id="test_001",
            start_time=time.time() - 0.1,
            end_time=time.time(),
            question_type="product_recommendation"
        )
        metrics.record_request(req)

        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1

        summary = metrics.get_summary()
        assert 'total_requests' in summary
        tests.append(("EngineMetrics", True, f"Summary: {summary['total_requests']} requests"))

    except Exception as e:
        tests.append(("EngineMetrics", False, str(e)))

    # 8. PipelineConfig
    try:
        from app.core.pipeline import PipelineConfig, RetrievalMode

        config = PipelineConfig.from_dict({
            'retrieval_mode': 'parallel',
            'max_parallel_retrievers': 8,
        })

        assert config.retrieval_mode == RetrievalMode.PARALLEL
        assert config.max_parallel_retrievers == 8
        tests.append(("PipelineConfig", True, f"Mode: {config.retrieval_mode.value}"))

    except Exception as e:
        tests.append(("PipelineConfig", False, str(e)))

    # 9. PromptManager
    try:
        from app.core.pipeline import PromptManager

        pm = PromptManager(base_path="prompts", cache_size=10)
        stats = pm.get_cache_stats()
        assert 'size' in stats
        assert 'max_size' in stats
        tests.append(("PromptManager", True, f"Cache: {stats}"))

    except Exception as e:
        tests.append(("PromptManager", False, str(e)))

    # 10. EngineError
    try:
        from app.core.engine import EngineError, ErrorType

        error = EngineError(
            "Test error",
            ErrorType.VALIDATION,
            {'field': 'question'}
        )

        error_dict = error.to_dict()
        assert error_dict['type'] == 'validation'
        assert 'timestamp' in error_dict
        tests.append(("EngineError", True, f"Type: {error.error_type.value}"))

    except Exception as e:
        tests.append(("EngineError", False, str(e)))

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
    print("  ONTIX Universal - Core Production Test v2.0")
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
        print("  Core modules are production-ready.")
        print("=" * 60)
    else:
        print("=" * 60)
        print("  SOME TESTS FAILED!")
        print("  Please check the errors above.")
        print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
