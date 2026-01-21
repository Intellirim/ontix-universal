"""
Integration Tests for Pipeline Flow
파이프라인 전체 흐름 통합 테스트

Run: pytest tests/integration/test_pipeline_flow.py -v
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, Any, List


class TestPipelineDataFlow:
    """파이프라인 데이터 흐름 통합 테스트"""

    @pytest.fixture
    def mock_apify_client(self):
        """Apify 클라이언트 Mock"""
        with patch('app.data_pipeline.pipeline.ApifyClient') as mock:
            mock_instance = MagicMock()
            mock_instance.run_actor = AsyncMock(return_value=[
                {
                    "id": "test_content_001",
                    "shortCode": "ABC123",
                    "caption": "테스트 콘텐츠 #테스트",
                    "url": "https://www.instagram.com/p/ABC123/",
                    "likesCount": 100,
                    "commentsCount": 10,
                    "timestamp": "2025-01-01T12:00:00Z",
                    "ownerUsername": "test_user",
                    "ownerId": "user_001",
                    "latestComments": [
                        {
                            "id": "comment_001",
                            "text": "좋은 콘텐츠입니다!",
                            "ownerUsername": "commenter",
                            "likesCount": 5,
                        }
                    ]
                }
            ])
            mock.return_value = mock_instance
            yield mock

    @pytest.fixture
    def mock_neo4j_repo(self):
        """Neo4j 저장소 Mock"""
        with patch('app.data_pipeline.pipeline.Neo4jRepository') as mock:
            mock_instance = MagicMock()
            mock_instance.filter_new_contents.return_value = []  # 모든 콘텐츠가 새 콘텐츠
            mock_instance.save_graph_documents.return_value = {
                "success": True,
                "nodes_created": 5,
                "nodes_updated": 0,
                "relationships_created": 3,
                "relationships_updated": 0,
                "errors": [],
            }
            mock_instance.content_exists_by_url.return_value = False
            mock.return_value = mock_instance
            yield mock

    @pytest.fixture
    def mock_llm_processor(self):
        """LLM 프로세서 Mock"""
        with patch('app.data_pipeline.pipeline.LLMProcessor') as mock:
            mock_instance = MagicMock()
            # GraphProcessingResult Mock
            mock_result = MagicMock()
            mock_result.graph_documents = []
            mock_result.nodes_count = 5
            mock_result.relationships_count = 3
            mock_result.processing_time_seconds = 1.0
            mock_result.cost_estimate_usd = 0.01
            mock_result.node_types = {"Brand": 1, "Content": 1, "Concept": 3}
            mock_result.relationship_types = {"HAS_CONTENT": 1, "MENTIONS_CONCEPT": 2}
            mock_result.errors = []
            mock_instance.process_contents.return_value = mock_result
            mock.return_value = mock_instance
            yield mock

    @pytest.mark.asyncio
    async def test_transform_stage_preserves_metadata(self, sample_brand_config):
        """변환 단계에서 metadata가 보존되는지 테스트"""
        from app.data_pipeline.adapters.instagram import InstagramAdapter
        from app.data_pipeline.domain.models import PlatformType

        adapter = InstagramAdapter()

        raw_data = {
            "id": "test_content_001",
            "shortCode": "ABC123",
            "type": "Reel",
            "caption": "테스트 콘텐츠 #테스트",
            "url": "https://www.instagram.com/p/ABC123/",
            "likesCount": 100,
            "commentsCount": 10,
            "timestamp": "2025-01-01T12:00:00.000Z",
            "ownerUsername": "test_user",
            "ownerId": "user_001",
            "displayUrl": "https://example.com/image.jpg",
            "dimensionsWidth": 1080,
            "dimensionsHeight": 1920,
            "firstComment": "첫 번째 댓글",
            "latestComments": [],
        }

        result = adapter.transform(raw_data)
        content = result["content"]

        # metadata가 보존되었는지 확인
        assert content.metadata is not None
        assert "short_code" in content.metadata
        assert content.metadata["short_code"] == "ABC123"
        assert content.metadata["type"] == "Reel"
        assert content.metadata["first_comment"] == "첫 번째 댓글"
        assert "dimensions" in content.metadata

    def test_content_dto_to_dict_preserves_metadata(self):
        """ContentDTO.to_dict()가 metadata를 보존하는지 테스트"""
        from app.data_pipeline.domain.models import (
            ContentDTO, ContentType, PlatformType, ActorDTO
        )

        actor = ActorDTO(
            platform=PlatformType.INSTAGRAM,
            actor_id="user_001",
            username="test_user",
        )

        content = ContentDTO(
            platform=PlatformType.INSTAGRAM,
            content_id="test_001",
            content_type=ContentType.POST,
            author=actor,
            text="테스트",
            metadata={
                "custom_field": "value",
                "nested": {"key": "value"},
            }
        )

        # to_dict 변환
        content_dict = content.to_dict()

        # metadata가 보존되었는지 확인
        assert "metadata" in content_dict
        assert content_dict["metadata"]["custom_field"] == "value"
        assert content_dict["metadata"]["nested"]["key"] == "value"

    def test_content_dto_from_dict_restores_metadata(self):
        """ContentDTO.from_dict()가 metadata를 복원하는지 테스트"""
        from app.data_pipeline.domain.models import ContentDTO

        data = {
            "platform": "instagram",
            "content_id": "test_001",
            "content_type": "post",
            "author": {
                "platform": "instagram",
                "actor_id": "user_001",
                "username": "test_user",
            },
            "text": "테스트",
            "metadata": {
                "custom_field": "value",
                "nested": {"key": "value"},
            }
        }

        content = ContentDTO.from_dict(data)

        # metadata가 복원되었는지 확인
        assert content.metadata is not None
        assert content.metadata["custom_field"] == "value"
        assert content.metadata["nested"]["key"] == "value"

    def test_interaction_dto_preserves_metadata(self):
        """InteractionDTO가 metadata를 보존하는지 테스트"""
        from app.data_pipeline.domain.models import (
            InteractionDTO, PlatformType, ActorDTO
        )

        actor = ActorDTO(
            platform=PlatformType.INSTAGRAM,
            actor_id="commenter_001",
            username="commenter",
        )

        interaction = InteractionDTO(
            platform=PlatformType.INSTAGRAM,
            interaction_id="interaction_001",
            content_id="content_001",
            author=actor,
            text="좋은 글입니다!",
            metadata={
                "replies_count": 5,
                "sentiment": "positive",
            }
        )

        # to_dict 변환
        interaction_dict = interaction.to_dict()

        # metadata가 보존되었는지 확인
        assert "metadata" in interaction_dict
        assert interaction_dict["metadata"]["replies_count"] == 5
        assert interaction_dict["metadata"]["sentiment"] == "positive"


class TestFilterPipelineIntegration:
    """필터 파이프라인 통합 테스트"""

    def test_all_filters_chain_correctly(self, sample_brand_config, sample_response, sample_context):
        """모든 필터가 체인으로 올바르게 동작하는지 테스트"""
        from app.filters.quality import QualityFilter, QualityConfig
        from app.filters.trust import TrustFilter, TrustConfig
        from app.filters.relevance import RelevanceFilter, RelevanceConfig
        from app.filters.validation import ValidationFilter, ValidationConfig

        # 필터 인스턴스 생성
        quality_filter = QualityFilter(config=QualityConfig())
        trust_filter = TrustFilter(config=TrustConfig())
        relevance_filter = RelevanceFilter(config=RelevanceConfig())
        validation_filter = ValidationFilter(config=ValidationConfig())

        # 체인 실행
        quality_result = quality_filter.validate(sample_response, sample_context)
        trust_result = trust_filter.validate(sample_response, sample_context)
        relevance_result = relevance_filter.validate(sample_response, sample_context)

        # 최종 검증 - 이전 결과들을 컨텍스트에 포함
        context_with_filters = {
            **sample_context,
            "quality_result": quality_result,
            "trust_result": trust_result,
            "relevance_result": relevance_result,
        }

        validation_result = validation_filter.validate(sample_response, context_with_filters)

        # 모든 결과가 유효한지 확인
        assert quality_result is not None
        assert trust_result is not None
        assert relevance_result is not None
        assert validation_result is not None

        # 점수가 유효 범위인지 확인
        assert 0.0 <= quality_result.score <= 1.0
        assert 0.0 <= trust_result.score <= 1.0
        assert 0.0 <= relevance_result.score <= 1.0
        assert 0.0 <= validation_result.score <= 1.0


class TestRetrieverPipelineIntegration:
    """Retriever 파이프라인 통합 테스트"""

    @pytest.fixture
    def mock_retrievers(self):
        """벡터/그래프 Retriever Mock"""
        with patch('app.retrievers.hybrid.VectorRetriever') as mock_vector, \
             patch('app.retrievers.hybrid.GraphRetriever') as mock_graph:
            # Vector Retriever Mock
            vector_instance = MagicMock()
            vector_result = MagicMock()
            vector_result.retrieval_results = [{
                'source': 'vector_search',
                'data': [
                    {'node_type': 'Content', 'node_id': 'c1', 'content': 'Test 1', 'score': 0.9},
                    {'node_type': 'Content', 'node_id': 'c2', 'content': 'Test 2', 'score': 0.8},
                ]
            }]
            vector_instance.retrieve.return_value = vector_result
            mock_vector.return_value = vector_instance

            # Graph Retriever Mock
            graph_instance = MagicMock()
            graph_result = MagicMock()
            graph_result.retrieval_results = [{
                'source': 'graph_search',
                'data': [
                    {'node_type': 'Concept', 'node_id': 'c3', 'content': 'Test 3', 'score': 0.85},
                ]
            }]
            graph_instance.retrieve.return_value = graph_result
            mock_graph.return_value = graph_instance

            yield mock_vector, mock_graph

    def test_hybrid_retriever_fuses_results(self, sample_brand_config, mock_retrievers):
        """HybridRetriever가 결과를 올바르게 융합하는지 테스트"""
        from app.retrievers.hybrid import HybridRetriever, FusionMethod

        retriever = HybridRetriever(sample_brand_config)

        # 융합 메서드 설정
        retriever.configure(
            fusion_method=FusionMethod.RRF,
            vector_weight=0.6,
            graph_weight=0.4,
        )

        # 디버그 정보 확인
        debug_info = retriever.get_debug_info()

        assert debug_info['config']['fusion_method'] == 'rrf'
        assert debug_info['config']['vector_weight'] == 0.6
        assert debug_info['config']['graph_weight'] == 0.4


class TestHandlerFilterIntegration:
    """Handler-Filter 통합 테스트"""

    def test_advisor_handler_applies_all_filters(self, sample_brand_config):
        """AdvisorHandler가 모든 필터를 적용하는지 테스트"""
        from app.features.advisor.handler import AdvisorHandler

        handler = AdvisorHandler('test_brand', sample_brand_config)

        # 4개 필터가 모두 존재하는지 확인
        assert handler.quality_filter is not None
        assert handler.trust_filter is not None
        assert handler.relevance_filter is not None
        assert handler.validation_filter is not None

        # 필터 통계 확인
        stats = handler.get_filter_stats()

        assert 'quality_filter_enabled' in stats
        assert 'trust_filter_enabled' in stats
        assert 'relevance_filter_enabled' in stats
        assert 'validation_filter_enabled' in stats

    def test_all_handlers_have_consistent_filter_interface(self, sample_brand_config):
        """모든 Handler가 일관된 필터 인터페이스를 가지는지 테스트"""
        from app.features.advisor.handler import AdvisorHandler
        from app.features.analytics.handler import AnalyticsHandler
        from app.features.product_recommendation.handler import ProductRecommendationHandler
        from app.features.content_generation.handler import ContentGenerationHandler
        from app.features.onboarding.handler import OnboardingHandler
        from app.features.social_monitoring.handler import SocialMonitoringHandler

        handlers = [
            AdvisorHandler('test', sample_brand_config),
            AnalyticsHandler('test', sample_brand_config),
            ProductRecommendationHandler('test', sample_brand_config),
            ContentGenerationHandler('test', sample_brand_config),
            OnboardingHandler('test', sample_brand_config),
            SocialMonitoringHandler('test', sample_brand_config),
        ]

        for handler in handlers:
            # 공통 필터 메서드 확인
            assert hasattr(handler, 'get_filter_stats'), f"{handler.__class__.__name__} missing get_filter_stats"

            stats = handler.get_filter_stats()

            # 공통 통계 키 확인
            required_keys = [
                'quality_filter_enabled',
                'trust_filter_enabled',
                'relevance_filter_enabled',
                'validation_filter_enabled',
            ]

            for key in required_keys:
                assert key in stats, f"{handler.__class__.__name__} missing {key} in filter stats"
