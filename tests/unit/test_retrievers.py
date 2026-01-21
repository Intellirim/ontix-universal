"""
Unit Tests for Retrievers
VectorRetriever, HybridRetriever 테스트

Run: pytest tests/unit/test_retrievers.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any


class TestHybridRetriever:
    """HybridRetriever 단위 테스트"""

    @pytest.fixture
    def mock_vector_retriever(self):
        """VectorRetriever Mock"""
        with patch('app.retrievers.hybrid.VectorRetriever') as mock:
            mock_instance = MagicMock()
            mock_instance.retrieve.return_value = MagicMock(
                retrieval_results=[{
                    'source': 'vector_search',
                    'data': [
                        {'node_type': 'Content', 'node_id': '1', 'content': 'Test 1', 'score': 0.9},
                        {'node_type': 'Content', 'node_id': '2', 'content': 'Test 2', 'score': 0.8},
                    ]
                }]
            )
            mock.return_value = mock_instance
            yield mock

    @pytest.fixture
    def mock_graph_retriever(self):
        """GraphRetriever Mock"""
        with patch('app.retrievers.hybrid.GraphRetriever') as mock:
            mock_instance = MagicMock()
            mock_instance.retrieve.return_value = MagicMock(
                retrieval_results=[{
                    'source': 'graph_search',
                    'data': [
                        {'node_type': 'Concept', 'node_id': '3', 'content': 'Test 3', 'score': 0.85},
                    ]
                }]
            )
            mock.return_value = mock_instance
            yield mock

    def test_hybrid_retriever_initialization(self, sample_brand_config, mock_vector_retriever, mock_graph_retriever):
        """HybridRetriever 초기화 테스트"""
        from app.retrievers.hybrid import HybridRetriever

        retriever = HybridRetriever(sample_brand_config)

        assert retriever is not None
        assert retriever.vector_retriever is not None
        assert retriever.graph_retriever is not None
        assert retriever.hybrid_config is not None

    def test_hybrid_retriever_has_do_retrieve(self, sample_brand_config, mock_vector_retriever, mock_graph_retriever):
        """_do_retrieve 메서드 존재 확인"""
        from app.retrievers.hybrid import HybridRetriever

        retriever = HybridRetriever(sample_brand_config)

        assert hasattr(retriever, '_do_retrieve')
        assert callable(retriever._do_retrieve)

    def test_hybrid_retriever_configure(self, sample_brand_config, mock_vector_retriever, mock_graph_retriever):
        """configure 메서드 테스트"""
        from app.retrievers.hybrid import HybridRetriever, FusionMethod

        retriever = HybridRetriever(sample_brand_config)

        # 체이닝 테스트
        result = retriever.configure(
            fusion_method=FusionMethod.RRF,
            vector_weight=0.7,
            graph_weight=0.3,
            max_results=20
        )

        assert result is retriever  # 체이닝 반환
        assert retriever.hybrid_config.vector_weight == 0.7
        assert retriever.hybrid_config.graph_weight == 0.3
        assert retriever.hybrid_config.max_results == 20

    def test_hybrid_retriever_debug_info(self, sample_brand_config, mock_vector_retriever, mock_graph_retriever):
        """디버그 정보 확인"""
        from app.retrievers.hybrid import HybridRetriever

        retriever = HybridRetriever(sample_brand_config)
        debug_info = retriever.get_debug_info()

        assert 'config' in debug_info
        assert 'fusion_method' in debug_info['config']
        assert 'vector_weight' in debug_info['config']


class TestRRFCalculator:
    """RRF 계산기 테스트"""

    def test_rrf_calculate(self):
        """RRF 점수 계산"""
        from app.retrievers.hybrid import RRFCalculator

        rankings = [
            [('doc1', 0.9), ('doc2', 0.8), ('doc3', 0.7)],
            [('doc2', 0.95), ('doc1', 0.85), ('doc4', 0.75)],
        ]
        weights = [0.6, 0.4]

        scores = RRFCalculator.calculate(rankings, weights, k=60)

        assert 'doc1' in scores
        assert 'doc2' in scores
        assert 'doc3' in scores
        assert 'doc4' in scores
        # doc2는 두 리스트 모두에서 상위이므로 높은 점수
        assert scores['doc2'] > scores['doc3']

    def test_rrf_empty_rankings(self):
        """빈 랭킹 처리"""
        from app.retrievers.hybrid import RRFCalculator

        rankings = [[], []]
        weights = [0.5, 0.5]

        scores = RRFCalculator.calculate(rankings, weights, k=60)

        assert scores == {}

    def test_weighted_sum(self):
        """가중치 합 계산"""
        from app.retrievers.hybrid import RRFCalculator

        scores = {
            'vector': {'doc1': 0.9, 'doc2': 0.8},
            'graph': {'doc2': 0.95, 'doc3': 0.85},
        }
        weights = {'vector': 0.6, 'graph': 0.4}

        final_scores = RRFCalculator.weighted_sum(scores, weights)

        assert 'doc1' in final_scores
        assert 'doc2' in final_scores
        assert 'doc3' in final_scores
        # doc2는 두 소스에서 모두 있으므로 보너스 적용


class TestResultMerger:
    """결과 병합기 테스트"""

    def test_deduplicate(self):
        """중복 제거 테스트"""
        from app.retrievers.hybrid import ResultMerger, HybridResult

        results = [
            HybridResult(node_type='Content', node_id='1', content='Test 1', final_score=0.9),
            HybridResult(node_type='Content', node_id='1', content='Test 1', final_score=0.85),
            HybridResult(node_type='Content', node_id='2', content='Test 2', final_score=0.8),
        ]

        unique = ResultMerger.deduplicate(results)

        assert len(unique) == 2
        # 첫 번째 항목이 유지됨
        assert unique[0].final_score == 0.9

    def test_interleave(self):
        """인터리브 병합 테스트"""
        from app.retrievers.hybrid import ResultMerger

        list_a = [{'id': 'a1'}, {'id': 'a2'}]
        list_b = [{'id': 'b1'}, {'id': 'b2'}, {'id': 'b3'}]

        merged = ResultMerger.interleave(list_a, list_b)

        assert len(merged) == 5
        assert merged[0]['id'] == 'a1'
        assert merged[1]['id'] == 'b1'
        assert merged[2]['id'] == 'a2'
        assert merged[3]['id'] == 'b2'
        assert merged[4]['id'] == 'b3'


class TestFusionMethods:
    """융합 방법 테스트"""

    def test_fusion_method_enum(self):
        """FusionMethod 열거형 테스트"""
        from app.retrievers.hybrid import FusionMethod

        assert FusionMethod.RRF.value == 'rrf'
        assert FusionMethod.WEIGHTED_SUM.value == 'weighted_sum'
        assert FusionMethod.MAX_SCORE.value == 'max_score'
        assert FusionMethod.INTERLEAVE.value == 'interleave'
