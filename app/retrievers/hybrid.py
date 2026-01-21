"""
Hybrid Retriever - Production Grade v2.0
벡터 + 그래프 하이브리드 검색 with RRF 재랭킹

Features:
    - RRF (Reciprocal Rank Fusion) 재랭킹
    - 벡터/그래프 검색 동시 실행
    - 가중치 기반 점수 조합
    - 중복 제거 및 결과 병합
    - 다양한 융합 전략 지원
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging
import time

from app.interfaces.retriever import RetrieverInterface
from app.core.context import QueryContext
from app.retrievers.vector import VectorRetriever, VectorSearchConfig, SearchMode, NodeIndex
from app.retrievers.graph import GraphRetriever, GraphSearchConfig, SearchScope

logger = logging.getLogger(__name__)


class FusionMethod(str, Enum):
    """결과 융합 방법"""
    RRF = "rrf"  # Reciprocal Rank Fusion
    WEIGHTED_SUM = "weighted_sum"  # 가중치 합
    MAX_SCORE = "max_score"  # 최대 점수
    INTERLEAVE = "interleave"  # 번갈아 배치


@dataclass
class HybridSearchConfig:
    """하이브리드 검색 설정"""
    fusion_method: FusionMethod = FusionMethod.RRF
    vector_weight: float = 0.6  # 벡터 검색 가중치
    graph_weight: float = 0.4  # 그래프 검색 가중치
    rrf_k: int = 60  # RRF k 파라미터 (일반적으로 60)
    max_results: int = 30
    min_score: float = 0.0
    enable_dedup: bool = True
    enable_parallel: bool = True  # 병렬 검색


@dataclass
class HybridResult:
    """하이브리드 검색 결과"""
    node_type: str
    node_id: str
    content: str
    final_score: float
    vector_score: Optional[float] = None
    graph_score: Optional[float] = None
    vector_rank: Optional[int] = None
    graph_rank: Optional[int] = None
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RRFCalculator:
    """RRF (Reciprocal Rank Fusion) 계산기"""

    @staticmethod
    def calculate(
        rankings: List[List[Tuple[str, float]]],
        weights: List[float],
        k: int = 60
    ) -> Dict[str, float]:
        """
        RRF 점수 계산

        Args:
            rankings: 각 검색 소스별 (doc_id, score) 리스트
            weights: 각 소스별 가중치
            k: RRF k 파라미터

        Returns:
            doc_id -> RRF 점수 매핑
        """
        rrf_scores = defaultdict(float)

        for ranking, weight in zip(rankings, weights):
            for rank, (doc_id, _) in enumerate(ranking, 1):
                # RRF 공식: 1 / (k + rank) * weight
                rrf_scores[doc_id] += weight * (1.0 / (k + rank))

        return dict(rrf_scores)

    @staticmethod
    def weighted_sum(
        scores: Dict[str, Dict[str, float]],
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        가중치 합 점수 계산

        Args:
            scores: source -> {doc_id: score}
            weights: source -> weight

        Returns:
            doc_id -> 최종 점수
        """
        final_scores = defaultdict(float)
        doc_sources = defaultdict(int)

        for source, source_scores in scores.items():
            weight = weights.get(source, 1.0)
            for doc_id, score in source_scores.items():
                final_scores[doc_id] += score * weight
                doc_sources[doc_id] += 1

        # 여러 소스에서 발견된 문서에 보너스
        for doc_id in final_scores:
            if doc_sources[doc_id] > 1:
                final_scores[doc_id] *= 1.1  # 10% 보너스

        return dict(final_scores)


class ResultMerger:
    """검색 결과 병합기"""

    @staticmethod
    def extract_results(context: QueryContext, source: str) -> List[Dict[str, Any]]:
        """컨텍스트에서 특정 소스의 결과 추출"""
        for retrieval in context.retrieval_results:
            if retrieval.get('source') == source:
                return retrieval.get('data', [])
        return []

    @staticmethod
    def deduplicate(results: List[HybridResult]) -> List[HybridResult]:
        """중복 결과 제거 (node_type + node_id 기준)"""
        seen = set()
        unique = []

        for result in results:
            key = (result.node_type, result.node_id)
            if key not in seen:
                seen.add(key)
                unique.append(result)

        return unique

    @staticmethod
    def interleave(
        results_a: List[Dict[str, Any]],
        results_b: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """두 결과 리스트를 번갈아 병합"""
        merged = []
        i, j = 0, 0

        while i < len(results_a) or j < len(results_b):
            if i < len(results_a):
                merged.append(results_a[i])
                i += 1
            if j < len(results_b):
                merged.append(results_b[j])
                j += 1

        return merged


class HybridRetriever(RetrieverInterface):
    """
    프로덕션급 하이브리드 검색기

    벡터 검색(의미)과 그래프 검색(키워드/관계)을 결합하여
    더 정확하고 다양한 검색 결과를 제공합니다.
    RRF를 통해 두 검색 결과를 효과적으로 융합합니다.
    """

    def __init__(self, brand_config):
        from app.interfaces.retriever import RetrievalSource
        super().__init__(brand_config, source=RetrievalSource.HYBRID)
        self.vector_retriever = VectorRetriever(brand_config)
        self.graph_retriever = GraphRetriever(brand_config)
        self.hybrid_config = HybridSearchConfig()

        # 캐시된 결과 (디버깅/분석용)
        self._last_vector_results: List[Dict] = []
        self._last_graph_results: List[Dict] = []

    def _do_retrieve(self, context: QueryContext):
        """
        RetrieverInterface의 추상 메서드 구현

        Args:
            context: 쿼리 컨텍스트

        Returns:
            RetrievalResult: 하이브리드 검색 결과
        """
        from app.interfaces.retriever import RetrievalResult, RetrievalItem, RetrievalStatus

        try:
            # 1. 벡터 검색 실행
            vector_context = self._run_vector_search(context)

            # 2. 그래프 검색 실행
            graph_context = self._run_graph_search(context)

            # 3. 결과 추출
            vector_results = ResultMerger.extract_results(vector_context, 'vector_search')
            graph_results = ResultMerger.extract_results(graph_context, 'graph_search')

            # 캐시 저장
            self._last_vector_results = vector_results
            self._last_graph_results = graph_results

            logger.info(
                f"Hybrid search: vector={len(vector_results)}, "
                f"graph={len(graph_results)} results"
            )

            # 4. 결과 융합
            fused_results = self._fuse_results(vector_results, graph_results)

            # 5. 최종 결과 제한
            fused_results = fused_results[:self.hybrid_config.max_results]

            # 6. RetrievalResult 형식으로 변환
            items = [
                RetrievalItem(
                    id=r.node_id,
                    content=r.content,
                    score=r.final_score,
                    source='hybrid_search',
                    node_type=r.node_type,
                    metadata={
                        'vector_score': r.vector_score,
                        'graph_score': r.graph_score,
                        'vector_rank': r.vector_rank,
                        'graph_rank': r.graph_rank,
                        'sources': r.sources,
                        **r.metadata,
                    }
                )
                for r in fused_results
            ]

            return RetrievalResult(
                source='hybrid_search',
                items=items,
                status=RetrievalStatus.COMPLETED,
                metadata={
                    'fusion_method': self.hybrid_config.fusion_method.value,
                    'vector_count': len(vector_results),
                    'graph_count': len(graph_results),
                    'fused_count': len(fused_results),
                    'vector_weight': self.hybrid_config.vector_weight,
                    'graph_weight': self.hybrid_config.graph_weight,
                }
            )

        except Exception as e:
            logger.error(f"Hybrid retrieval error: {e}", exc_info=True)
            return RetrievalResult(
                source='hybrid_search',
                items=[],
                status=RetrievalStatus.FAILED,
                error=str(e)
            )

    def retrieve(self, context: QueryContext) -> QueryContext:
        """
        하이브리드 검색 실행

        Args:
            context: 쿼리 컨텍스트

        Returns:
            검색 결과가 추가된 컨텍스트
        """
        start_time = time.time()

        try:
            # 1. 벡터 검색 실행
            vector_context = self._run_vector_search(context)

            # 2. 그래프 검색 실행
            graph_context = self._run_graph_search(context)

            # 3. 결과 추출
            vector_results = ResultMerger.extract_results(vector_context, 'vector_search')
            graph_results = ResultMerger.extract_results(graph_context, 'graph_search')

            # 캐시 저장
            self._last_vector_results = vector_results
            self._last_graph_results = graph_results

            logger.info(
                f"Hybrid search: vector={len(vector_results)}, "
                f"graph={len(graph_results)} results"
            )

            # 4. 결과 융합
            fused_results = self._fuse_results(vector_results, graph_results)

            # 5. 최종 결과 제한
            fused_results = fused_results[:self.config.max_results]

            if fused_results:
                context.add_retrieval_result(
                    source='hybrid_search',
                    data=[self._result_to_dict(r) for r in fused_results],
                    metadata={
                        'fusion_method': self.hybrid_config.fusion_method.value,
                        'vector_count': len(vector_results),
                        'graph_count': len(graph_results),
                        'fused_count': len(fused_results),
                        'vector_weight': self.hybrid_config.vector_weight,
                        'graph_weight': self.hybrid_config.graph_weight,
                        'search_time_ms': round((time.time() - start_time) * 1000, 2),
                    }
                )

            logger.info(
                f"Hybrid retrieval: {len(fused_results)} results in "
                f"{(time.time() - start_time) * 1000:.1f}ms"
            )

            return context

        except Exception as e:
            logger.error(f"Hybrid retrieval error: {e}", exc_info=True)
            return context

    def _run_vector_search(self, context: QueryContext) -> QueryContext:
        """벡터 검색 실행 (별도 컨텍스트)"""
        # 벡터 검색용 컨텍스트 복사
        from copy import deepcopy
        vector_context = deepcopy(context)

        try:
            # 벡터 검색기 설정
            self.vector_retriever.configure(
                node_index=NodeIndex.ALL,
                top_k=self.hybrid_config.max_results,
                min_score=0.5,  # 낮은 임계값 (RRF에서 재랭킹)
            )
            return self.vector_retriever.retrieve(vector_context)

        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return vector_context

    def _run_graph_search(self, context: QueryContext) -> QueryContext:
        """그래프 검색 실행 (별도 컨텍스트)"""
        from copy import deepcopy
        graph_context = deepcopy(context)

        try:
            # 그래프 검색기 설정
            self.graph_retriever.configure(
                scope=SearchScope.ALL,
                max_results=self.hybrid_config.max_results,
                include_relationships=True,
            )
            return self.graph_retriever.retrieve(graph_context)

        except Exception as e:
            logger.error(f"Graph search error: {e}")
            return graph_context

    def _fuse_results(
        self,
        vector_results: List[Dict],
        graph_results: List[Dict]
    ) -> List[HybridResult]:
        """결과 융합"""
        if self.hybrid_config.fusion_method == FusionMethod.RRF:
            return self._fuse_rrf(vector_results, graph_results)
        elif self.hybrid_config.fusion_method == FusionMethod.WEIGHTED_SUM:
            return self._fuse_weighted_sum(vector_results, graph_results)
        elif self.hybrid_config.fusion_method == FusionMethod.MAX_SCORE:
            return self._fuse_max_score(vector_results, graph_results)
        elif self.hybrid_config.fusion_method == FusionMethod.INTERLEAVE:
            return self._fuse_interleave(vector_results, graph_results)
        else:
            return self._fuse_rrf(vector_results, graph_results)

    def _fuse_rrf(
        self,
        vector_results: List[Dict],
        graph_results: List[Dict]
    ) -> List[HybridResult]:
        """RRF 기반 융합"""
        # 문서 정보 수집
        doc_info: Dict[str, Dict] = {}

        # 벡터 결과 랭킹
        vector_ranking = []
        for rank, r in enumerate(vector_results, 1):
            doc_id = self._get_doc_id(r)
            vector_ranking.append((doc_id, r.get('score', 0)))
            if doc_id not in doc_info:
                doc_info[doc_id] = self._create_doc_info(r, 'vector')
            doc_info[doc_id]['vector_score'] = r.get('score', 0)
            doc_info[doc_id]['vector_rank'] = rank
            doc_info[doc_id]['sources'].append('vector')

        # 그래프 결과 랭킹
        graph_ranking = []
        for rank, r in enumerate(graph_results, 1):
            doc_id = self._get_doc_id(r)
            graph_ranking.append((doc_id, r.get('score', 0)))
            if doc_id not in doc_info:
                doc_info[doc_id] = self._create_doc_info(r, 'graph')
            doc_info[doc_id]['graph_score'] = r.get('score', 0)
            doc_info[doc_id]['graph_rank'] = rank
            if 'graph' not in doc_info[doc_id]['sources']:
                doc_info[doc_id]['sources'].append('graph')

        # RRF 점수 계산
        rrf_scores = RRFCalculator.calculate(
            rankings=[vector_ranking, graph_ranking],
            weights=[self.hybrid_config.vector_weight, self.hybrid_config.graph_weight],
            k=self.hybrid_config.rrf_k
        )

        # HybridResult 생성
        results = []
        for doc_id, final_score in rrf_scores.items():
            info = doc_info.get(doc_id, {})
            results.append(HybridResult(
                node_type=info.get('node_type', 'Unknown'),
                node_id=info.get('node_id', doc_id),
                content=info.get('content', ''),
                final_score=final_score,
                vector_score=info.get('vector_score'),
                graph_score=info.get('graph_score'),
                vector_rank=info.get('vector_rank'),
                graph_rank=info.get('graph_rank'),
                sources=info.get('sources', []),
                metadata=info.get('metadata', {}),
            ))

        # 점수순 정렬
        results.sort(key=lambda x: x.final_score, reverse=True)

        # 중복 제거
        if self.hybrid_config.enable_dedup:
            results = ResultMerger.deduplicate(results)

        return results

    def _fuse_weighted_sum(
        self,
        vector_results: List[Dict],
        graph_results: List[Dict]
    ) -> List[HybridResult]:
        """가중치 합 기반 융합"""
        scores = {
            'vector': {self._get_doc_id(r): r.get('score', 0) for r in vector_results},
            'graph': {self._get_doc_id(r): r.get('score', 0) for r in graph_results},
        }
        weights = {
            'vector': self.hybrid_config.vector_weight,
            'graph': self.hybrid_config.graph_weight,
        }

        final_scores = RRFCalculator.weighted_sum(scores, weights)

        # 문서 정보 병합
        doc_info = {}
        for r in vector_results + graph_results:
            doc_id = self._get_doc_id(r)
            if doc_id not in doc_info:
                doc_info[doc_id] = self._create_doc_info(r, 'mixed')

        results = []
        for doc_id, final_score in final_scores.items():
            info = doc_info.get(doc_id, {})
            results.append(HybridResult(
                node_type=info.get('node_type', 'Unknown'),
                node_id=info.get('node_id', doc_id),
                content=info.get('content', ''),
                final_score=final_score,
                sources=['weighted_sum'],
                metadata=info.get('metadata', {}),
            ))

        results.sort(key=lambda x: x.final_score, reverse=True)
        return results

    def _fuse_max_score(
        self,
        vector_results: List[Dict],
        graph_results: List[Dict]
    ) -> List[HybridResult]:
        """최대 점수 기반 융합"""
        doc_info = {}

        for r in vector_results:
            doc_id = self._get_doc_id(r)
            score = r.get('score', 0) * self.hybrid_config.vector_weight
            if doc_id not in doc_info or score > doc_info[doc_id].get('max_score', 0):
                doc_info[doc_id] = self._create_doc_info(r, 'vector')
                doc_info[doc_id]['max_score'] = score

        for r in graph_results:
            doc_id = self._get_doc_id(r)
            score = r.get('score', 0) * self.hybrid_config.graph_weight
            if doc_id not in doc_info or score > doc_info[doc_id].get('max_score', 0):
                doc_info[doc_id] = self._create_doc_info(r, 'graph')
                doc_info[doc_id]['max_score'] = score

        results = []
        for doc_id, info in doc_info.items():
            results.append(HybridResult(
                node_type=info.get('node_type', 'Unknown'),
                node_id=info.get('node_id', doc_id),
                content=info.get('content', ''),
                final_score=info.get('max_score', 0),
                sources=['max_score'],
                metadata=info.get('metadata', {}),
            ))

        results.sort(key=lambda x: x.final_score, reverse=True)
        return results

    def _fuse_interleave(
        self,
        vector_results: List[Dict],
        graph_results: List[Dict]
    ) -> List[HybridResult]:
        """번갈아 배치 융합"""
        merged = ResultMerger.interleave(vector_results, graph_results)

        results = []
        seen = set()

        for rank, r in enumerate(merged, 1):
            doc_id = self._get_doc_id(r)
            if doc_id in seen:
                continue
            seen.add(doc_id)

            info = self._create_doc_info(r, 'interleave')
            results.append(HybridResult(
                node_type=info.get('node_type', 'Unknown'),
                node_id=info.get('node_id', doc_id),
                content=info.get('content', ''),
                final_score=1.0 / rank,  # 순위 기반 점수
                sources=['interleave'],
                metadata=info.get('metadata', {}),
            ))

        return results

    def _get_doc_id(self, result: Dict) -> str:
        """문서 고유 ID 생성"""
        node_type = result.get('node_type', 'unknown')
        node_id = result.get('node_id', result.get('id', ''))
        return f"{node_type}:{node_id}"

    def _create_doc_info(self, result: Dict, source: str) -> Dict:
        """문서 정보 생성"""
        return {
            'node_type': result.get('node_type', 'Unknown'),
            'node_id': result.get('node_id', result.get('id', '')),
            'content': result.get('content', result.get('text', '')),
            'metadata': result.get('metadata', {}),
            'sources': [source],
        }

    def _result_to_dict(self, result: HybridResult) -> Dict[str, Any]:
        """HybridResult를 딕셔너리로 변환"""
        return {
            'node_type': result.node_type,
            'node_id': result.node_id,
            'content': result.content,
            'final_score': round(result.final_score, 4),
            'vector_score': round(result.vector_score, 4) if result.vector_score else None,
            'graph_score': round(result.graph_score, 4) if result.graph_score else None,
            'vector_rank': result.vector_rank,
            'graph_rank': result.graph_rank,
            'sources': result.sources,
            'metadata': result.metadata,
        }

    def configure(
        self,
        fusion_method: FusionMethod = None,
        vector_weight: float = None,
        graph_weight: float = None,
        rrf_k: int = None,
        max_results: int = None,
        enable_dedup: bool = None,
    ) -> 'HybridRetriever':
        """
        검색 설정 변경 (체이닝 지원)

        Usage:
            retriever.configure(fusion_method=FusionMethod.RRF, vector_weight=0.7).retrieve(context)
        """
        if fusion_method is not None:
            self.hybrid_config.fusion_method = fusion_method
        if vector_weight is not None:
            self.hybrid_config.vector_weight = vector_weight
        if graph_weight is not None:
            self.hybrid_config.graph_weight = graph_weight
        if rrf_k is not None:
            self.hybrid_config.rrf_k = rrf_k
        if max_results is not None:
            self.hybrid_config.max_results = max_results
        if enable_dedup is not None:
            self.hybrid_config.enable_dedup = enable_dedup
        return self

    def get_debug_info(self) -> Dict[str, Any]:
        """디버깅 정보 반환"""
        return {
            'config': {
                'fusion_method': self.hybrid_config.fusion_method.value,
                'vector_weight': self.hybrid_config.vector_weight,
                'graph_weight': self.hybrid_config.graph_weight,
                'rrf_k': self.hybrid_config.rrf_k,
            },
            'last_search': {
                'vector_results': len(self._last_vector_results),
                'graph_results': len(self._last_graph_results),
            },
        }
