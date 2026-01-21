#!/usr/bin/env python3
"""
Services 테스트 스크립트
- shared 서비스: Cache, LLM, Neo4j, Vector
- platform 서비스: Config, Brand, Feature, Analytics, Monitoring

실행: python scripts/test_services.py
"""

import os
import sys
from datetime import datetime

# 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Colors:
    """터미널 색상"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """헤더 출력"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")


def print_section(text: str):
    """섹션 출력"""
    print(f"\n{Colors.CYAN}>> {text}{Colors.RESET}")


def print_success(text: str):
    """성공 메시지"""
    print(f"  {Colors.GREEN}[PASS] {text}{Colors.RESET}")


def print_error(text: str):
    """에러 메시지"""
    print(f"  {Colors.RED}[FAIL] {text}{Colors.RESET}")


def print_warning(text: str):
    """경고 메시지"""
    print(f"  {Colors.YELLOW}[SKIP] {text}{Colors.RESET}")


def print_info(text: str):
    """정보 메시지"""
    print(f"  {Colors.CYAN}-> {text}{Colors.RESET}")


# ============================================================
# Test Results Tracking
# ============================================================

test_results = {
    "passed": 0,
    "failed": 0,
    "skipped": 0,
    "errors": []
}


def record_test(name: str, passed: bool, error: str = None, skipped: bool = False):
    """테스트 결과 기록"""
    if skipped:
        test_results["skipped"] += 1
        print_warning(f"{name} - SKIPPED")
    elif passed:
        test_results["passed"] += 1
        print_success(f"{name} - PASSED")
    else:
        test_results["failed"] += 1
        test_results["errors"].append((name, error))
        print_error(f"{name} - FAILED: {error}")


# ============================================================
# Shared Services Tests
# ============================================================

def test_cache_service():
    """Cache 서비스 테스트"""
    print_section("Testing Cache Service")

    try:
        from app.services.shared.cache import (
            CacheClient, CacheConfig,
            SerializationType, get_cache_client, cached
        )

        # 1. Config 생성
        config = CacheConfig(
            host="localhost",
            port=6379,
            default_ttl=300,
        )
        record_test("CacheConfig creation", True)

        # 2. Enums 테스트
        assert SerializationType.JSON.value == "json"
        assert SerializationType.STRING.value == "string"
        record_test("Cache enums", True)

        # 3. @cached 데코레이터 테스트 (함수 정의만)
        @cached(ttl=60)
        def sample_cached_func(x):
            return x * 2

        record_test("@cached decorator", True)

        # 4. Factory 함수 테스트 (Redis 없이)
        try:
            client = get_cache_client()
            record_test("get_cache_client()", True)

            # Metrics 확인
            metrics = client.get_metrics()
            assert hasattr(metrics, 'hits')
            # to_dict() 테스트
            metrics_dict = metrics.to_dict()
            assert "hits" in metrics_dict
            record_test("CacheClient.get_metrics()", True)

        except Exception as e:
            record_test("get_cache_client()", False, str(e))

    except Exception as e:
        record_test("Cache service import", False, str(e))


def test_llm_service():
    """LLM 서비스 테스트"""
    print_section("Testing LLM Service")

    try:
        from app.services.shared.llm import (
            LLMClient, LLMConfig, LLMMetrics, LLMResponse,
            ModelVariant, ModelProvider, CostCalculator, TokenCounter,
            get_llm_client
        )

        # 1. Config 생성 (api_key 필수)
        config = LLMConfig(
            api_key="test-api-key",
            default_model="gpt-4o-mini",
        )
        record_test("LLMConfig creation", True)

        # 2. Enums 테스트
        assert ModelVariant.MINI.value == "mini"
        assert ModelVariant.FULL.value == "full"
        assert ModelProvider.OPENAI.value == "openai"
        record_test("LLM enums", True)

        # 3. CostCalculator 테스트
        cost = CostCalculator.calculate("gpt-4o-mini", 1000, 500)
        assert cost > 0
        record_test("CostCalculator.calculate()", True)

        # 4. TokenCounter 테스트
        counter = TokenCounter()
        tokens = counter.count("Hello, world!")
        assert tokens > 0
        record_test("TokenCounter.count()", True)

        # 5. Factory 함수 테스트
        try:
            client = get_llm_client()
            record_test("get_llm_client()", True)

            # Metrics 확인
            metrics = client.get_metrics()
            assert hasattr(metrics, 'total_requests')
            record_test("LLMClient.get_metrics()", True)

        except Exception as e:
            # API 키가 없을 수 있음
            if "OPENAI_API_KEY" in str(e):
                record_test("get_llm_client()", True, skipped=True)
            else:
                record_test("get_llm_client()", False, str(e))

    except Exception as e:
        record_test("LLM service import", False, str(e))


def test_neo4j_service():
    """Neo4j 서비스 테스트"""
    print_section("Testing Neo4j Service")

    try:
        from app.services.shared.neo4j import (
            Neo4jClient, Neo4jConfig, QueryMetrics, QueryResult,
            QueryType, ConnectionState, get_neo4j_client
        )

        # 1. Config 생성
        config = Neo4jConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password",
        )
        record_test("Neo4jConfig creation", True)

        # 2. Enums 테스트
        assert QueryType.READ.value == "read"
        assert ConnectionState.CONNECTED.value == "connected"
        record_test("Neo4j enums", True)

        # 3. QueryMetrics 테스트
        metrics = QueryMetrics()
        metrics.record_query(QueryType.READ, 50.0, 10, True)
        assert metrics.total_queries == 1
        assert metrics.read_queries == 1
        record_test("QueryMetrics.record_query()", True)

        # 4. Factory 함수 테스트 (연결 시도)
        try:
            client = get_neo4j_client()
            record_test("get_neo4j_client()", True)

            # 연결 상태 확인
            health = client.health_check()
            assert "status" in health
            record_test("Neo4jClient.health_check()", True)

        except Exception as e:
            # 연결 실패는 환경에 따라 정상 (Neo4j 서버 없음)
            error_msg = str(e).lower()
            if any(x in error_msg for x in ["connection", "unavailable", "connect", "url"]):
                record_test("get_neo4j_client()", True, skipped=True)
                record_test("Neo4jClient.health_check()", True, skipped=True)
            else:
                record_test("get_neo4j_client()", False, str(e))

    except Exception as e:
        record_test("Neo4j service import", False, str(e))


def test_vector_service():
    """Vector 서비스 테스트"""
    print_section("Testing Vector Service")

    try:
        from app.services.shared.vector import (
            VectorService, VectorConfig, VectorMetrics, EmbeddingResult,
            EmbeddingModel, EmbeddingType, EmbeddingCostCalculator,
            EmbeddingCache, get_vector_service
        )

        # 1. Config 생성
        config = VectorConfig(
            model="text-embedding-3-small",
            batch_size=100,
            enable_cache=True,
        )
        record_test("VectorConfig creation", True)

        # 2. Enums 테스트
        assert EmbeddingModel.TEXT_EMBEDDING_3_SMALL.value == "text-embedding-3-small"
        assert EmbeddingModel.TEXT_EMBEDDING_3_SMALL.dimensions == 1536
        assert EmbeddingType.QUERY.value == "query"
        record_test("Vector enums", True)

        # 3. EmbeddingCostCalculator 테스트
        cost = EmbeddingCostCalculator.calculate("text-embedding-3-small", 1000)
        assert cost > 0
        record_test("EmbeddingCostCalculator.calculate()", True)

        # 4. EmbeddingCache 테스트
        cache = EmbeddingCache(max_size=100, ttl=60)
        cache.set("test", "text-embedding-3-small", [0.1, 0.2, 0.3])
        result = cache.get("test", "text-embedding-3-small")
        assert result == [0.1, 0.2, 0.3]
        record_test("EmbeddingCache", True)

        # 5. VectorMetrics 테스트
        metrics = VectorMetrics()
        metrics.record_embedding(EmbeddingType.QUERY, 100, 50.0, from_cache=False)
        assert metrics.total_embeddings == 1
        record_test("VectorMetrics.record_embedding()", True)

        # 6. Factory 함수 테스트
        try:
            service = get_vector_service()
            record_test("get_vector_service()", True)

            # Metrics 확인
            metrics = service.get_metrics()
            assert "total_embeddings" in metrics
            record_test("VectorService.get_metrics()", True)

        except Exception as e:
            if "OPENAI_API_KEY" in str(e):
                record_test("get_vector_service()", True, skipped=True)
            else:
                record_test("get_vector_service()", False, str(e))

    except Exception as e:
        record_test("Vector service import", False, str(e))


# ============================================================
# Platform Services Tests
# ============================================================

def test_config_manager():
    """Config Manager 테스트"""
    print_section("Testing Config Manager")

    try:
        from app.services.platform.config_manager import (
            ConfigManager, ConfigType, ConfigEntry, EnvVarProcessor,
            ConfigValidator, get_config_manager, list_brands
        )

        # 1. Enums 테스트
        assert ConfigType.BRAND.value == "brand"
        assert ConfigType.PLATFORM.value == "platform"
        record_test("ConfigType enums", True)

        # 2. EnvVarProcessor 테스트
        os.environ["TEST_VAR"] = "test_value"
        result = EnvVarProcessor.substitute({"key": "${TEST_VAR}"})
        assert result["key"] == "test_value"
        record_test("EnvVarProcessor.substitute()", True)

        # 3. 기본값 치환 테스트
        result = EnvVarProcessor.substitute({"key": "${NONEXISTENT:default_value}"})
        assert result["key"] == "default_value"
        record_test("EnvVarProcessor default value", True)

        # 4. list_brands 테스트
        brands = list_brands()
        assert isinstance(brands, list)
        record_test("list_brands()", True)

        # 5. Factory 함수 테스트
        manager = get_config_manager()
        assert manager is not None
        record_test("get_config_manager()", True)

        # 6. Health check
        health = ConfigManager.health_check()
        assert "status" in health
        record_test("ConfigManager.health_check()", True)

    except Exception as e:
        record_test("Config Manager import", False, str(e))


def test_brand_manager():
    """Brand Manager 테스트"""
    print_section("Testing Brand Manager")

    try:
        from app.services.platform.brand_manager import (
            BrandManager, BrandInfo, BrandStatus, BrandStats,
            ValidationResult, ValidationLevel, BrandValidator,
            get_brand_manager
        )

        # 1. Enums 테스트
        assert BrandStatus.ACTIVE.value == "active"
        assert ValidationLevel.FULL.value == "full"
        record_test("Brand enums", True)

        # 2. BrandInfo 테스트
        info = BrandInfo(
            id="test-brand",
            name="Test Brand",
            features=["feature1", "feature2"],
        )
        info_dict = info.to_dict()
        assert info_dict["id"] == "test-brand"
        record_test("BrandInfo.to_dict()", True)

        # 3. BrandStats 테스트
        stats = BrandStats(
            brand_id="test-brand",
            nodes={"Product": 100},
            total_nodes=100,
        )
        assert stats.total_nodes == 100
        record_test("BrandStats", True)

        # 4. ValidationResult 테스트
        result = ValidationResult(
            valid=True,
            brand_id="test-brand",
            issues=[],
            warnings=["Test warning"],
        )
        assert result.valid
        record_test("ValidationResult", True)

        # 5. BrandValidator 테스트
        test_config = {
            "brand": {"id": "test", "name": "Test"},
            "features": [],
            "neo4j": {"brand_id": "test"},
        }
        issues, warnings = BrandValidator.validate_basic("test", test_config)
        assert len(issues) == 0
        record_test("BrandValidator.validate_basic()", True)

        # 6. Factory 함수 테스트
        manager = get_brand_manager()
        assert manager is not None
        record_test("get_brand_manager()", True)

    except Exception as e:
        record_test("Brand Manager import", False, str(e))


def test_feature_manager():
    """Feature Manager 테스트"""
    print_section("Testing Feature Manager")

    try:
        from app.services.platform.feature_manager import (
            FeatureManager, FeatureInfo, FeatureStatus,
            FeatureValidationResult, ValidationIssue, ValidationSeverity,
            SchemaLoader, get_feature_manager, list_features
        )

        # 1. Enums 테스트
        assert FeatureStatus.ENABLED.value == "enabled"
        assert ValidationSeverity.ERROR.value == "error"
        record_test("Feature enums", True)

        # 2. FeatureInfo 테스트
        info = FeatureInfo(
            name="test-feature",
            status=FeatureStatus.ENABLED,
            description="Test feature",
        )
        info_dict = info.to_dict()
        assert info_dict["name"] == "test-feature"
        record_test("FeatureInfo.to_dict()", True)

        # 3. ValidationIssue 테스트
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            message="Test error",
            path="config.field",
        )
        assert issue.severity == ValidationSeverity.ERROR
        record_test("ValidationIssue", True)

        # 4. FeatureValidationResult 테스트
        result = FeatureValidationResult(
            feature_name="test",
            valid=True,
            issues=[],
        )
        assert result.valid
        record_test("FeatureValidationResult", True)

        # 5. Factory 함수 테스트
        manager = get_feature_manager()
        assert manager is not None
        record_test("get_feature_manager()", True)

        # 6. list_features 테스트
        features = list_features()
        assert isinstance(features, list)
        record_test("list_features()", True)

    except Exception as e:
        record_test("Feature Manager import", False, str(e))


def test_analytics_service():
    """Analytics 서비스 테스트"""
    print_section("Testing Analytics Service")

    try:
        from app.services.platform.analytics import (
            AnalyticsService, AnalyticsConfig, AnalyticsMetrics,
            Event, EventType, AggregationType, TimeWindow,
            TimeSeriesAggregator, get_analytics_service, track_event
        )

        # 1. Enums 테스트
        assert EventType.QUERY.value == "query"
        assert AggregationType.COUNT.value == "count"
        assert TimeWindow.HOUR.value == "hour"
        record_test("Analytics enums", True)

        # 2. AnalyticsConfig 테스트
        config = AnalyticsConfig(
            max_events=1000,
            retention_hours=12,
        )
        assert config.max_events == 1000
        record_test("AnalyticsConfig", True)

        # 3. Event 테스트
        event = Event(
            event_type=EventType.QUERY,
            brand_id="test-brand",
            data={"query": "test query"},
        )
        event_dict = event.to_dict()
        assert event_dict["brand_id"] == "test-brand"
        record_test("Event.to_dict()", True)

        # 4. TimeSeriesAggregator 테스트
        aggregator = TimeSeriesAggregator(TimeWindow.MINUTE)
        aggregator.add("test_metric", 1.0)
        aggregator.add("test_metric", 2.0)
        series = aggregator.get_series("test_metric", AggregationType.SUM)
        assert len(series) > 0
        record_test("TimeSeriesAggregator", True)

        # 5. Factory 함수 테스트
        service = get_analytics_service()
        assert service is not None
        record_test("get_analytics_service()", True)

        # 6. track_event 테스트
        service.track_event("query", "test-brand", {"test": True})
        stats = service.get_stats()
        assert stats["total_events"] > 0
        record_test("AnalyticsService.track_event()", True)

        # 7. Metrics 테스트
        metrics = service.get_metrics()
        assert "total_events" in metrics
        record_test("AnalyticsService.get_metrics()", True)

    except Exception as e:
        record_test("Analytics service import", False, str(e))


def test_monitoring_service():
    """Monitoring 서비스 테스트"""
    print_section("Testing Monitoring Service")

    try:
        from app.services.platform.monitoring import (
            MonitoringService, MonitoringConfig, MonitoringMetrics,
            ServiceHealth, SystemHealth, Alert,
            HealthStatus, ServiceType, AlertLevel,
            HealthChecker, get_monitoring_service
        )

        # 1. Enums 테스트
        assert HealthStatus.HEALTHY.value == "healthy"
        assert ServiceType.NEO4J.value == "neo4j"
        assert AlertLevel.ERROR.value == "error"
        record_test("Monitoring enums", True)

        # 2. MonitoringConfig 테스트
        config = MonitoringConfig(
            check_interval=30,
            timeout=3.0,
        )
        assert config.check_interval == 30
        record_test("MonitoringConfig", True)

        # 3. ServiceHealth 테스트
        health = ServiceHealth(
            service=ServiceType.NEO4J,
            status=HealthStatus.HEALTHY,
            latency_ms=50.0,
        )
        health_dict = health.to_dict()
        assert health_dict["status"] == "healthy"
        record_test("ServiceHealth.to_dict()", True)

        # 4. Alert 테스트
        alert = Alert(
            level=AlertLevel.WARNING,
            service=ServiceType.REDIS,
            message="Test alert",
        )
        alert_dict = alert.to_dict()
        assert alert_dict["level"] == "warning"
        record_test("Alert.to_dict()", True)

        # 5. Factory 함수 테스트
        service = get_monitoring_service()
        assert service is not None
        record_test("get_monitoring_service()", True)

        # 6. Metrics 테스트
        metrics = service.get_metrics()
        assert "total_checks" in metrics
        record_test("MonitoringService.get_metrics()", True)

    except Exception as e:
        record_test("Monitoring service import", False, str(e))


# ============================================================
# Import Tests
# ============================================================

def test_package_imports():
    """패키지 import 테스트"""
    print_section("Testing Package Imports")

    # Shared package
    try:
        from app.services.shared import (
            CacheClient, LLMClient, Neo4jClient, VectorService,
            get_cache_client, get_llm_client, get_neo4j_client, get_vector_service
        )
        record_test("app.services.shared imports", True)
    except Exception as e:
        record_test("app.services.shared imports", False, str(e))

    # Platform package
    try:
        from app.services.platform import (
            ConfigManager, BrandManager, FeatureManager,
            AnalyticsService, MonitoringService,
            get_brand_manager, get_feature_manager,
            get_analytics_service, get_monitoring_service
        )
        record_test("app.services.platform imports", True)
    except Exception as e:
        record_test("app.services.platform imports", False, str(e))

    # Main services package
    try:
        from app.services import (
            CacheClient, LLMClient, Neo4jClient, VectorService,
            ConfigManager, BrandManager, FeatureManager,
            AnalyticsService, MonitoringService
        )
        record_test("app.services imports", True)
    except Exception as e:
        record_test("app.services imports", False, str(e))


# ============================================================
# Main
# ============================================================

def print_summary():
    """테스트 결과 요약"""
    print_header("Test Summary")

    total = test_results["passed"] + test_results["failed"] + test_results["skipped"]
    print(f"\n  Total: {total} tests")
    print(f"  {Colors.GREEN}Passed: {test_results['passed']}{Colors.RESET}")
    print(f"  {Colors.RED}Failed: {test_results['failed']}{Colors.RESET}")
    print(f"  {Colors.YELLOW}Skipped: {test_results['skipped']}{Colors.RESET}")

    if test_results["errors"]:
        print(f"\n{Colors.RED}Failed Tests:{Colors.RESET}")
        for name, error in test_results["errors"]:
            print(f"  - {name}: {error}")

    print()

    if test_results["failed"] == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}All tests passed!{Colors.RESET}")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}Some tests failed.{Colors.RESET}")
        return 1


def main():
    """메인 함수"""
    print_header("Services Test Suite")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Package imports
    test_package_imports()

    # Shared services
    print_header("Shared Services Tests")
    test_cache_service()
    test_llm_service()
    test_neo4j_service()
    test_vector_service()

    # Platform services
    print_header("Platform Services Tests")
    test_config_manager()
    test_brand_manager()
    test_feature_manager()
    test_analytics_service()
    test_monitoring_service()

    # Summary
    exit_code = print_summary()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
