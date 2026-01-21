#!/usr/bin/env python3
"""
ONTIX Universal - Integration Tests
통합 테스트 스크립트 (API 키 없이 실행 가능한 테스트)

Usage:
    python scripts/test_integration.py
    python scripts/test_integration.py --with-llm   # LLM 테스트 포함 (API 키 필요)
    python scripts/test_integration.py --with-neo4j # Neo4j 테스트 포함 (연결 필요)
"""
import sys
import os
import argparse
from typing import List, Tuple, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_imports() -> List[Tuple[str, bool, str]]:
    """모든 모듈 import 테스트"""
    results = []

    modules = [
        ("domain.models", "app.data_pipeline.domain.models"),
        ("adapters.base", "app.data_pipeline.adapters.base"),
        ("adapters.instagram", "app.data_pipeline.adapters.instagram"),
        ("adapters.youtube", "app.data_pipeline.adapters.youtube"),
        ("adapters.tiktok", "app.data_pipeline.adapters.tiktok"),
        ("adapters.twitter", "app.data_pipeline.adapters.twitter"),
        ("crawlers.base", "app.data_pipeline.crawlers.base"),
        ("crawlers.apify_client", "app.data_pipeline.crawlers.apify_client"),
        ("processors.llm_processor", "app.data_pipeline.processors.llm_processor"),
        ("repositories.neo4j_repo", "app.data_pipeline.repositories.neo4j_repo"),
        ("pipeline", "app.data_pipeline.pipeline"),
    ]

    for name, module_path in modules:
        try:
            __import__(module_path)
            results.append((f"Import: {name}", True, ""))
        except Exception as e:
            results.append((f"Import: {name}", False, str(e)[:50]))

    return results


def test_domain_models() -> List[Tuple[str, bool, str]]:
    """Domain Models 테스트"""
    results = []

    try:
        from app.data_pipeline.domain.models import (
            PlatformType,
            ContentType,
            ActorDTO,
            ContentDTO,
            InteractionDTO,
            TopicDTO,
        )

        # PlatformType
        assert PlatformType.INSTAGRAM.value == "instagram"
        assert PlatformType.YOUTUBE.value == "youtube"
        assert PlatformType.TIKTOK.value == "tiktok"
        assert PlatformType.TWITTER.value == "twitter"
        results.append(("PlatformType enum", True, "4 platforms"))

        # ContentType
        assert ContentType.POST.value == "post"
        assert ContentType.VIDEO.value == "video"
        assert ContentType.SHORT.value == "short"
        assert ContentType.REEL.value == "reel"
        assert ContentType.TWEET.value == "tweet"
        results.append(("ContentType enum", True, "6 types"))

        # ActorDTO
        actor = ActorDTO(
            platform=PlatformType.INSTAGRAM,
            actor_id="123",
            username="testuser",
            display_name="Test User",
            follower_count=1000,
        )
        assert actor.username == "testuser"
        results.append(("ActorDTO dataclass", True, ""))

        # ContentDTO
        content = ContentDTO(
            platform=PlatformType.INSTAGRAM,
            content_id="456",
            content_type=ContentType.POST,
            author=actor,
            text="Test content",
            like_count=100,
        )
        assert content.like_count == 100
        results.append(("ContentDTO dataclass", True, ""))

        # InteractionDTO
        interaction = InteractionDTO(
            platform=PlatformType.INSTAGRAM,
            interaction_id="789",
            content_id="456",
            author=actor,
            text="Test comment",
        )
        assert interaction.text == "Test comment"
        results.append(("InteractionDTO dataclass", True, ""))

    except Exception as e:
        results.append(("Domain Models", False, str(e)[:50]))

    return results


def test_adapter_factory() -> List[Tuple[str, bool, str]]:
    """어댑터 팩토리 패턴 테스트"""
    results = []

    try:
        from app.data_pipeline.adapters import (
            InstagramAdapter,
            YouTubeAdapter,
            TikTokAdapter,
            TwitterAdapter,
        )
        from app.data_pipeline.domain.models import PlatformType

        adapters = {
            PlatformType.INSTAGRAM: InstagramAdapter,
            PlatformType.YOUTUBE: YouTubeAdapter,
            PlatformType.TIKTOK: TikTokAdapter,
            PlatformType.TWITTER: TwitterAdapter,
        }

        for platform, adapter_class in adapters.items():
            adapter = adapter_class()
            assert adapter.platform == platform
            assert hasattr(adapter, "parse_actor")
            assert hasattr(adapter, "parse_content")
            assert hasattr(adapter, "parse_interactions")
            assert hasattr(adapter, "validate_raw_data")
            assert hasattr(adapter, "transform")

        results.append(("Adapter Factory Pattern", True, "4 adapters"))

    except Exception as e:
        results.append(("Adapter Factory Pattern", False, str(e)[:50]))

    return results


def test_pipeline_config() -> List[Tuple[str, bool, str]]:
    """Pipeline 설정 테스트"""
    results = []

    try:
        from app.data_pipeline.pipeline import PipelineConfig, PipelineStage
        from app.data_pipeline.domain.models import PlatformType

        # Check stages
        stages = list(PipelineStage)
        assert len(stages) == 5
        results.append(("PipelineStage enum", True, "5 stages"))

        # Check config
        assert PipelineConfig.DEFAULT_BATCH_SIZE == 50
        assert PipelineConfig.MAX_CONCURRENT_TASKS == 5
        results.append(("PipelineConfig defaults", True, ""))

        # Check Apify actors
        actors = PipelineConfig.APIFY_ACTORS
        assert actors[PlatformType.INSTAGRAM] == "apify/instagram-scraper"
        assert actors[PlatformType.YOUTUBE] == "streamers/youtube-scraper"
        assert actors[PlatformType.TIKTOK] == "clockworks/tiktok-scraper"
        assert actors[PlatformType.TWITTER] == "apidojo/tweet-scraper"
        results.append(("Apify Actor IDs", True, "4 actors configured"))

    except Exception as e:
        results.append(("Pipeline Config", False, str(e)[:50]))

    return results


def test_llm_processor_config() -> List[Tuple[str, bool, str]]:
    """LLM Processor 설정 테스트"""
    results = []

    try:
        from app.data_pipeline.processors.llm_processor import (
            ModelConfig,
            CostEstimator,
            PromptBuilder,
        )

        # Check ModelConfig
        assert ModelConfig.DEFAULT_MODEL == "gpt-4o-mini"
        assert ModelConfig.DEFAULT_TEMPERATURE == 0.0
        assert ModelConfig.MAX_RETRIES == 3
        results.append(("ModelConfig (gpt-4o-mini fixed)", True, ""))

        # Check CostEstimator
        tokens = CostEstimator.estimate_tokens("Hello world test")
        assert tokens > 0
        cost = CostEstimator.estimate_cost("Hello world", 100)
        assert cost > 0
        results.append(("CostEstimator", True, f"tokens={tokens}"))

        # Check PromptBuilder methods
        assert hasattr(PromptBuilder, "build_system_prompt")
        assert hasattr(PromptBuilder, "build_brand_header")
        assert hasattr(PromptBuilder, "build_content_block")
        results.append(("PromptBuilder methods", True, ""))

    except Exception as e:
        results.append(("LLM Processor Config", False, str(e)[:50]))

    return results


def test_neo4j_repo_structure() -> List[Tuple[str, bool, str]]:
    """Neo4j Repository 구조 테스트"""
    results = []

    try:
        from app.data_pipeline.repositories.neo4j_repo import (
            Neo4jRepository,
            NodeTypes,
            RelationshipTypes,
            SchemaManager,
        )

        # Check NodeTypes
        assert NodeTypes.BRAND == "Brand"
        assert NodeTypes.CONTENT == "Content"
        assert NodeTypes.INTERACTION == "Interaction"
        assert NodeTypes.CONCEPT == "Concept"
        assert NodeTypes.ACTOR == "Actor"
        assert NodeTypes.TOPIC == "Topic"
        results.append(("NodeTypes", True, "6 types"))

        # Check RelationshipTypes
        assert NodeTypes.all_types() is not None
        assert len(RelationshipTypes.all_types()) >= 10
        results.append(("RelationshipTypes", True, f"{len(RelationshipTypes.all_types())} types"))

        # Check SchemaManager
        assert len(SchemaManager.CONSTRAINTS) > 0
        assert len(SchemaManager.INDEXES) > 0
        results.append(("SchemaManager", True, f"{len(SchemaManager.CONSTRAINTS)} constraints"))

        # Check Neo4jRepository methods
        methods = [
            "save_graph_documents",
            "content_exists",
            "content_exists_by_url",
            "filter_new_contents",
            "get_brand_statistics",
            "get_content_statistics",
            "delete_brand_data",
        ]
        for method in methods:
            assert hasattr(Neo4jRepository, method), f"Missing method: {method}"
        results.append(("Neo4jRepository methods", True, f"{len(methods)} methods"))

    except Exception as e:
        results.append(("Neo4j Repo Structure", False, str(e)[:50]))

    return results


def test_end_to_end_transform() -> List[Tuple[str, bool, str]]:
    """End-to-End 변환 테스트 (API 없이)"""
    results = []

    try:
        from app.data_pipeline.adapters import (
            InstagramAdapter,
            YouTubeAdapter,
            TikTokAdapter,
            TwitterAdapter,
        )

        # Sample data
        samples = {
            "Twitter": (
                TwitterAdapter(),
                {
                    "id": "123",
                    "url": "https://x.com/user/status/123",
                    "text": "Test tweet #hashtag",
                    "likeCount": 10,
                    "retweetCount": 5,
                    "replyCount": 2,
                    "quoteCount": 1,
                    "createdAt": "Mon Jan 01 00:00:00 +0000 2024",
                },
            ),
            "YouTube": (
                YouTubeAdapter(),
                {
                    "id": "abc123",
                    "title": "Test Video",
                    "url": "https://youtube.com/watch?v=abc123",
                    "viewCount": 1000,
                    "likes": 100,
                    "channelName": "TestChannel",
                    "channelUrl": "https://youtube.com/@TestChannel",
                    "date": "2024-01-01T00:00:00.000Z",
                    "duration": "00:05:00",
                },
            ),
            "TikTok": (
                TikTokAdapter(),
                {
                    "authorMeta.name": "testuser",
                    "text": "Test video #fyp",
                    "webVideoUrl": "https://tiktok.com/@testuser/video/123",
                    "diggCount": 5000,
                    "playCount": 10000,
                    "shareCount": 100,
                    "commentCount": 50,
                    "createTimeISO": "2024-01-01T00:00:00.000Z",
                },
            ),
            "Instagram": (
                InstagramAdapter(),
                {
                    "id": "post123",
                    "type": "Image",
                    "caption": "Test post #photo",
                    "url": "https://instagram.com/p/post123/",
                    "likesCount": 200,
                    "commentsCount": 20,
                    "latestComments": [],
                },
            ),
        }

        for platform_name, (adapter, data) in samples.items():
            try:
                # Validate
                assert adapter.validate_raw_data(data), f"{platform_name} validation failed"

                # Transform
                result = adapter.transform(data)
                assert "actor" in result
                assert "content" in result
                assert "interactions" in result

                # Check content
                content = result["content"]
                assert content.content_id is not None
                assert content.platform is not None

                results.append((f"E2E Transform: {platform_name}", True, f"id={content.content_id}"))

            except Exception as e:
                results.append((f"E2E Transform: {platform_name}", False, str(e)[:40]))

    except Exception as e:
        results.append(("E2E Transform", False, str(e)[:50]))

    return results


def test_llm_with_api(api_key: str = None) -> List[Tuple[str, bool, str]]:
    """LLM API 테스트 (API 키 필요)"""
    results = []

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        results.append(("LLM API Test", False, "OPENAI_API_KEY not set"))
        return results

    try:
        from app.data_pipeline.processors.llm_processor import LLMProcessor

        processor = LLMProcessor(api_key=api_key)
        assert processor.model == "gpt-4o-mini"
        results.append(("LLM Processor Init", True, f"model={processor.model}"))

    except Exception as e:
        results.append(("LLM Processor Init", False, str(e)[:50]))

    return results


def test_neo4j_connection() -> List[Tuple[str, bool, str]]:
    """Neo4j 연결 테스트"""
    results = []

    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    if not all([uri, username, password]):
        results.append(("Neo4j Connection", False, "NEO4J credentials not set"))
        return results

    try:
        from app.data_pipeline.repositories.neo4j_repo import Neo4jRepository

        repo = Neo4jRepository(uri=uri, username=username, password=password)
        stats = repo.get_statistics()
        results.append(("Neo4j Connection", True, f"nodes={stats.get('nodes', 0)}"))
        repo.close()

    except Exception as e:
        results.append(("Neo4j Connection", False, str(e)[:50]))

    return results


# =============================================================================
# RESULT PRINTER
# =============================================================================

def print_results(all_results: List[Tuple[str, bool, str]], title: str) -> Tuple[int, int]:
    """결과 출력"""
    passed = 0
    failed = 0

    print(f"\n{'=' * 70}")
    print(title)
    print("=" * 70)

    for name, success, detail in all_results:
        status = "PASS" if success else "FAIL"
        if success:
            passed += 1
            detail_str = f" - {detail}" if detail else ""
            print(f"  [{status}] {name}{detail_str}")
        else:
            failed += 1
            print(f"  [{status}] {name}")
            if detail:
                print(f"         Error: {detail}")

    print("-" * 70)
    print(f"Total: {passed + failed} | Passed: {passed} | Failed: {failed}")

    return passed, failed


def print_summary_table(
    section_results: List[Tuple[str, int, int]]
) -> None:
    """요약 테이블 출력"""
    print("\n" + "=" * 70)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 70)
    print(f"{'Section':<30} | {'Passed':>8} | {'Failed':>8} | {'Status':>8}")
    print("-" * 70)

    total_passed = 0
    total_failed = 0

    for section, passed, failed in section_results:
        status = "OK" if failed == 0 else "FAIL"
        print(f"{section:<30} | {passed:>8} | {failed:>8} | {status:>8}")
        total_passed += passed
        total_failed += failed

    print("-" * 70)
    overall = "ALL PASSED" if total_failed == 0 else f"{total_failed} FAILED"
    print(f"{'TOTAL':<30} | {total_passed:>8} | {total_failed:>8} | {overall:>8}")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """메인 실행"""
    parser = argparse.ArgumentParser(description="ONTIX Universal Integration Tests")
    parser.add_argument("--with-llm", action="store_true", help="Include LLM API tests")
    parser.add_argument("--with-neo4j", action="store_true", help="Include Neo4j connection tests")
    args = parser.parse_args()

    print("\nONTIX Universal - Integration Tests")
    print("=" * 70)

    section_results = []

    # Core tests (no external dependencies)
    all_results = []

    all_results.extend(test_imports())
    p, f = print_results(all_results, "1. MODULE IMPORTS")
    section_results.append(("Module Imports", p, f))

    all_results = test_domain_models()
    p, f = print_results(all_results, "2. DOMAIN MODELS")
    section_results.append(("Domain Models", p, f))

    all_results = test_adapter_factory()
    p, f = print_results(all_results, "3. ADAPTER FACTORY")
    section_results.append(("Adapter Factory", p, f))

    all_results = test_pipeline_config()
    p, f = print_results(all_results, "4. PIPELINE CONFIG")
    section_results.append(("Pipeline Config", p, f))

    all_results = test_llm_processor_config()
    p, f = print_results(all_results, "5. LLM PROCESSOR CONFIG")
    section_results.append(("LLM Processor Config", p, f))

    all_results = test_neo4j_repo_structure()
    p, f = print_results(all_results, "6. NEO4J REPOSITORY STRUCTURE")
    section_results.append(("Neo4j Repo Structure", p, f))

    all_results = test_end_to_end_transform()
    p, f = print_results(all_results, "7. END-TO-END TRANSFORM")
    section_results.append(("E2E Transform", p, f))

    # Optional tests
    if args.with_llm:
        all_results = test_llm_with_api()
        p, f = print_results(all_results, "8. LLM API (OPTIONAL)")
        section_results.append(("LLM API", p, f))

    if args.with_neo4j:
        all_results = test_neo4j_connection()
        p, f = print_results(all_results, "9. NEO4J CONNECTION (OPTIONAL)")
        section_results.append(("Neo4j Connection", p, f))

    # Print summary
    print_summary_table(section_results)

    # Calculate total failures
    total_failed = sum(f for _, _, f in section_results)
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
