"""
Neo4j Repository - Production Grade v4.0

ONTIX Universal 범용 SNS 데이터 파이프라인을 위한 프로덕션급 Neo4j 저장소.
모든 SNS 플랫폼(Instagram, YouTube, TikTok, Twitter)의 지식그래프 데이터를 관리.

Features:
    - 멀티 브랜드 완벽 지원 (brand_id 기반 격리)
    - 배치 처리 최적화 (UNWIND)
    - 범용 노드 타입 (Content, Interaction, Concept, Actor, Topic, Brand)
    - 플랫폼별 통계 및 필터링
    - 중복 제거 및 MERGE 기반 Upsert

Author: ONTIX Universal Team
Version: 4.0.0
"""

from __future__ import annotations

import os
import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Dict,
    Final,
    List,
    Optional,
    Set,
    Tuple,
    TypedDict,
    Union,
)

from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph

from ..domain.models import (
    ActorDTO,
    ContentDTO,
    InteractionDTO,
    PlatformType,
    TopicDTO,
)

# Load environment variables
load_dotenv()

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS & CONFIGURATIONS
# ============================================================================

class NodeTypes:
    """그래프 노드 타입 상수"""

    BRAND: Final[str] = "Brand"
    CONTENT: Final[str] = "Content"
    INTERACTION: Final[str] = "Interaction"
    CONCEPT: Final[str] = "Concept"
    ACTOR: Final[str] = "Actor"
    TOPIC: Final[str] = "Topic"

    # 레거시 호환
    POST: Final[str] = "Post"
    COMMENT: Final[str] = "Comment"

    @classmethod
    def brand_id_required_types(cls) -> Set[str]:
        """brand_id가 필수인 노드 타입"""
        return {cls.CONTENT, cls.INTERACTION, cls.CONCEPT, cls.TOPIC, cls.POST, cls.COMMENT}

    @classmethod
    def all_types(cls) -> List[str]:
        """모든 노드 타입"""
        return [cls.BRAND, cls.CONTENT, cls.INTERACTION, cls.CONCEPT, cls.ACTOR, cls.TOPIC]


class RelationshipTypes:
    """그래프 관계 타입 상수"""

    # Brand relationships
    HAS_CONCEPT: Final[str] = "HAS_CONCEPT"
    HAS_CONTENT: Final[str] = "HAS_CONTENT"

    # Content relationships
    MENTIONS_CONCEPT: Final[str] = "MENTIONS_CONCEPT"
    HAS_INTERACTION: Final[str] = "HAS_INTERACTION"
    CREATED_BY: Final[str] = "CREATED_BY"

    # Concept relationships
    RELATED_TO: Final[str] = "RELATED_TO"
    CAUSES: Final[str] = "CAUSES"
    LEADS_TO: Final[str] = "LEADS_TO"
    STEP_OF_ROUTINE: Final[str] = "STEP_OF_ROUTINE"
    TARGETS_PROBLEM: Final[str] = "TARGETS_PROBLEM"
    PROVIDES_EFFECT: Final[str] = "PROVIDES_EFFECT"

    # Interaction relationships
    EXPRESSES_SENTIMENT: Final[str] = "EXPRESSES_SENTIMENT"
    REPLY_TO: Final[str] = "REPLY_TO"

    # Actor relationships
    AUTHORED: Final[str] = "AUTHORED"
    INTERACTED: Final[str] = "INTERACTED"

    # Legacy
    HAS_COMMENT: Final[str] = "HAS_COMMENT"

    @classmethod
    def all_types(cls) -> List[str]:
        """모든 관계 타입"""
        return [
            cls.HAS_CONCEPT, cls.HAS_CONTENT, cls.MENTIONS_CONCEPT, cls.HAS_INTERACTION,
            cls.CREATED_BY, cls.RELATED_TO, cls.CAUSES, cls.LEADS_TO, cls.STEP_OF_ROUTINE,
            cls.TARGETS_PROBLEM, cls.PROVIDES_EFFECT, cls.EXPRESSES_SENTIMENT,
            cls.REPLY_TO, cls.AUTHORED, cls.INTERACTED, cls.HAS_COMMENT,
        ]


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

class BrandStatistics(TypedDict):
    """브랜드 통계 타입"""

    brand_id: str
    total_nodes: int
    total_relationships: int
    node_types: Dict[str, int]
    relationship_types: Dict[str, int]
    platforms: Dict[str, int]
    date_range: Dict[str, Optional[str]]


class ContentStatistics(TypedDict):
    """콘텐츠 통계 타입"""

    total_count: int
    by_platform: Dict[str, int]
    by_type: Dict[str, int]
    engagement: Dict[str, int]
    date_range: Dict[str, Optional[str]]


class InteractionStatistics(TypedDict):
    """인터랙션 통계 타입"""

    total_count: int
    by_platform: Dict[str, int]
    average_per_content: float
    sentiment_distribution: Dict[str, int]


class ConceptCloudItem(TypedDict):
    """개념 클라우드 아이템"""

    name: str
    count: int
    type: Optional[str]
    related_concepts: List[str]


class SaveResult(TypedDict):
    """저장 결과 타입"""

    success: bool
    nodes_created: int
    nodes_updated: int
    relationships_created: int
    relationships_updated: int
    errors: List[str]


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class NodeData:
    """노드 데이터 구조"""

    node_type: str
    node_id: str
    brand_id: Optional[str]
    properties: Dict[str, Any]

    def unique_key(self) -> Tuple:
        """중복 제거용 유니크 키"""
        if self.node_type in NodeTypes.brand_id_required_types():
            return (self.node_type, self.node_id, self.brand_id)
        return (self.node_type, self.node_id)


@dataclass
class RelationshipData:
    """관계 데이터 구조"""

    source_type: str
    source_id: str
    source_brand_id: Optional[str]
    target_type: str
    target_id: str
    target_brand_id: Optional[str]
    rel_type: str
    properties: Dict[str, Any]

    def unique_key(self) -> Tuple:
        """중복 제거용 유니크 키"""
        return (
            self.source_type, self.source_id, self.source_brand_id,
            self.rel_type,
            self.target_type, self.target_id, self.target_brand_id,
        )


# ============================================================================
# QUERY BUILDERS
# ============================================================================

class CypherQueryBuilder:
    """Cypher 쿼리 빌더"""

    @staticmethod
    def build_node_merge_query(node_type: str, use_brand_id: bool) -> str:
        """노드 MERGE 쿼리 생성"""
        if use_brand_id:
            return f"""
            UNWIND $nodes AS nodeData
            MERGE (n:{node_type} {{id: nodeData.id, brand_id: nodeData.brand_id}})
            ON CREATE SET
                n += nodeData.properties,
                n.created_at = datetime(),
                n._is_new = true
            ON MATCH SET
                n += nodeData.properties,
                n.updated_at = datetime(),
                n._is_new = false
            RETURN
                count(n) as total,
                sum(CASE WHEN n._is_new = true THEN 1 ELSE 0 END) as created
            """
        else:
            return f"""
            UNWIND $nodes AS nodeData
            MERGE (n:{node_type} {{id: nodeData.id}})
            ON CREATE SET
                n += nodeData.properties,
                n.created_at = datetime(),
                n._is_new = true
            ON MATCH SET
                n += nodeData.properties,
                n.updated_at = datetime(),
                n._is_new = false
            RETURN
                count(n) as total,
                sum(CASE WHEN n._is_new = true THEN 1 ELSE 0 END) as created
            """

    @staticmethod
    def build_relationship_merge_query(
        source_type: str,
        rel_type: str,
        target_type: str,
        source_has_brand: bool,
        target_has_brand: bool,
    ) -> str:
        """관계 MERGE 쿼리 생성"""
        # Source MATCH
        if source_has_brand:
            source_match = f"MATCH (source:{source_type} {{id: relData.source_id, brand_id: relData.source_brand_id}})"
        else:
            source_match = f"MATCH (source:{source_type} {{id: relData.source_id}})"

        # Target MATCH
        if target_has_brand:
            target_match = f"MATCH (target:{target_type} {{id: relData.target_id, brand_id: relData.target_brand_id}})"
        else:
            target_match = f"MATCH (target:{target_type} {{id: relData.target_id}})"

        return f"""
        UNWIND $rels AS relData
        {source_match}
        {target_match}
        MERGE (source)-[r:{rel_type}]->(target)
        ON CREATE SET
            r += relData.properties,
            r.created_at = datetime()
        ON MATCH SET
            r += relData.properties,
            r.updated_at = datetime()
        RETURN count(r) as count
        """


# ============================================================================
# CONSTRAINT & INDEX MANAGER
# ============================================================================

class SchemaManager:
    """Neo4j 스키마 관리자"""

    # 제약조건 정의
    CONSTRAINTS: List[Tuple[str, str]] = [
        # Brand - id만 유니크
        ("brand_id_unique", "CREATE CONSTRAINT brand_id_unique IF NOT EXISTS FOR (b:Brand) REQUIRE b.id IS UNIQUE"),

        # Content - (id, brand_id) NODE KEY
        ("content_brand_key", "CREATE CONSTRAINT content_brand_key IF NOT EXISTS FOR (c:Content) REQUIRE (c.id, c.brand_id) IS NODE KEY"),

        # Interaction - (id, brand_id) NODE KEY
        ("interaction_brand_key", "CREATE CONSTRAINT interaction_brand_key IF NOT EXISTS FOR (i:Interaction) REQUIRE (i.id, i.brand_id) IS NODE KEY"),

        # Concept - (id, brand_id) NODE KEY
        ("concept_brand_key", "CREATE CONSTRAINT concept_brand_key IF NOT EXISTS FOR (c:Concept) REQUIRE (c.id, c.brand_id) IS NODE KEY"),

        # Actor - (actor_id, platform) NODE KEY
        ("actor_platform_key", "CREATE CONSTRAINT actor_platform_key IF NOT EXISTS FOR (a:Actor) REQUIRE (a.actor_id, a.platform) IS NODE KEY"),

        # Topic - (name, brand_id) NODE KEY
        ("topic_brand_key", "CREATE CONSTRAINT topic_brand_key IF NOT EXISTS FOR (t:Topic) REQUIRE (t.name, t.brand_id) IS NODE KEY"),

        # Legacy: Post
        ("post_brand_key", "CREATE CONSTRAINT post_brand_key IF NOT EXISTS FOR (p:Post) REQUIRE (p.id, p.brand_id) IS NODE KEY"),

        # Legacy: Comment
        ("comment_brand_key", "CREATE CONSTRAINT comment_brand_key IF NOT EXISTS FOR (c:Comment) REQUIRE (c.id, c.brand_id) IS NODE KEY"),
    ]

    # 대체 제약조건 (NODE KEY 미지원 시)
    FALLBACK_CONSTRAINTS: List[Tuple[str, str]] = [
        ("brand_id_unique", "CREATE CONSTRAINT brand_id_unique IF NOT EXISTS FOR (b:Brand) REQUIRE b.id IS UNIQUE"),
        ("content_id_unique", "CREATE CONSTRAINT content_id_unique IF NOT EXISTS FOR (c:Content) REQUIRE c.id IS UNIQUE"),
        ("interaction_id_unique", "CREATE CONSTRAINT interaction_id_unique IF NOT EXISTS FOR (i:Interaction) REQUIRE i.id IS UNIQUE"),
        ("concept_id_unique", "CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE"),
        ("actor_id_unique", "CREATE CONSTRAINT actor_id_unique IF NOT EXISTS FOR (a:Actor) REQUIRE a.actor_id IS UNIQUE"),
        ("topic_name_unique", "CREATE CONSTRAINT topic_name_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE"),
        ("post_id_unique", "CREATE CONSTRAINT post_id_unique IF NOT EXISTS FOR (p:Post) REQUIRE p.id IS UNIQUE"),
        ("comment_id_unique", "CREATE CONSTRAINT comment_id_unique IF NOT EXISTS FOR (c:Comment) REQUIRE c.id IS UNIQUE"),
    ]

    # 인덱스 정의
    INDEXES: List[Tuple[str, str]] = [
        # URL 인덱스
        ("content_url_idx", "CREATE INDEX content_url_idx IF NOT EXISTS FOR (c:Content) ON (c.url)"),

        # Platform 인덱스
        ("content_platform_idx", "CREATE INDEX content_platform_idx IF NOT EXISTS FOR (c:Content) ON (c.platform)"),

        # Brand ID 인덱스 (성능 최적화)
        ("content_brand_id_idx", "CREATE INDEX content_brand_id_idx IF NOT EXISTS FOR (c:Content) ON (c.brand_id)"),
        ("interaction_brand_id_idx", "CREATE INDEX interaction_brand_id_idx IF NOT EXISTS FOR (i:Interaction) ON (i.brand_id)"),
        ("concept_brand_id_idx", "CREATE INDEX concept_brand_id_idx IF NOT EXISTS FOR (c:Concept) ON (c.brand_id)"),
        ("topic_brand_id_idx", "CREATE INDEX topic_brand_id_idx IF NOT EXISTS FOR (t:Topic) ON (t.brand_id)"),

        # Actor 인덱스
        ("actor_platform_idx", "CREATE INDEX actor_platform_idx IF NOT EXISTS FOR (a:Actor) ON (a.platform)"),
        ("actor_username_idx", "CREATE INDEX actor_username_idx IF NOT EXISTS FOR (a:Actor) ON (a.username)"),

        # Legacy 인덱스
        ("post_url_idx", "CREATE INDEX post_url_idx IF NOT EXISTS FOR (p:Post) ON (p.url)"),
        ("post_brand_id_idx", "CREATE INDEX post_brand_id_idx IF NOT EXISTS FOR (p:Post) ON (p.brand_id)"),
    ]

    @classmethod
    def create_constraints(cls, query_func) -> Tuple[int, int]:
        """
        제약조건 생성.

        Args:
            query_func: 쿼리 실행 함수

        Returns:
            (성공 수, 실패 수) 튜플
        """
        success = 0
        failed = 0
        use_fallback = False

        for name, query in cls.CONSTRAINTS:
            try:
                query_func(query)
                success += 1
                logger.debug(f"Constraint created: {name}")
            except Exception as e:
                if "NODE KEY" in str(e).upper() or "SYNTAX" in str(e).upper():
                    use_fallback = True
                    break
                failed += 1
                logger.warning(f"Constraint failed ({name}): {e}")

        # NODE KEY 미지원 시 대체 제약조건 사용
        if use_fallback:
            logger.warning("NODE KEY not supported. Using fallback constraints.")
            success = 0
            failed = 0
            for name, query in cls.FALLBACK_CONSTRAINTS:
                try:
                    query_func(query)
                    success += 1
                except Exception as e:
                    failed += 1
                    logger.warning(f"Fallback constraint failed ({name}): {e}")

        return success, failed

    @classmethod
    def create_indexes(cls, query_func) -> Tuple[int, int]:
        """
        인덱스 생성.

        Args:
            query_func: 쿼리 실행 함수

        Returns:
            (성공 수, 실패 수) 튜플
        """
        success = 0
        failed = 0

        for name, query in cls.INDEXES:
            try:
                query_func(query)
                success += 1
                logger.debug(f"Index created: {name}")
            except Exception as e:
                failed += 1
                logger.warning(f"Index failed ({name}): {e}")

        return success, failed


# ============================================================================
# MAIN REPOSITORY CLASS
# ============================================================================

class Neo4jRepository:
    """
    프로덕션급 Neo4j 지식그래프 저장소.

    ONTIX Universal의 범용 SNS 데이터 파이프라인을 위한 Neo4j 저장소입니다.
    멀티 브랜드 완벽 지원, 배치 처리 최적화, 중복 제거를 제공합니다.

    Attributes:
        graph: Neo4jGraph 인스턴스
        uri: Neo4j 연결 URI
        batch_size: 배치 처리 크기

    Example:
        >>> repo = Neo4jRepository()
        >>> result = repo.save_graph_documents(graph_docs, brand_id="my_brand")
        >>> stats = repo.get_brand_statistics("my_brand")
    """

    DEFAULT_BATCH_SIZE: Final[int] = 500

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """
        Neo4j 저장소 초기화.

        Args:
            uri: Neo4j URI. None이면 환경변수에서 로드.
            username: Neo4j 사용자명. None이면 환경변수에서 로드.
            password: Neo4j 비밀번호. None이면 환경변수에서 로드.
            batch_size: 배치 처리 크기.

        Raises:
            ValueError: 연결 정보가 없는 경우.
            ConnectionError: 연결 실패 시.
        """
        self.batch_size = batch_size

        # 연결 정보 로드
        self.uri = uri or os.getenv("NEO4J_URI")
        _username = username or os.getenv("NEO4J_USERNAME")
        _password = password or os.getenv("NEO4J_PASSWORD")

        if not all([self.uri, _username, _password]):
            raise ValueError(
                "Neo4j connection info required. "
                "Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD environment variables."
            )

        logger.info(f"Connecting to Neo4j: {self.uri}")

        try:
            self.graph = Neo4jGraph(
                url=self.uri,
                username=_username,
                password=_password,
            )

            # 연결 테스트
            self._test_connection()

            # 스키마 초기화
            self._initialize_schema()

            logger.info("Neo4j connection established successfully")

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise ConnectionError(f"Neo4j connection failed: {e}") from e

    def _test_connection(self) -> None:
        """연결 테스트"""
        try:
            result = self.graph.query("RETURN 1 AS test")
            if not result:
                raise ConnectionError("Query execution failed")
            logger.debug("Connection test passed")
        except Exception as e:
            raise ConnectionError(f"Connection test failed: {e}") from e

    def _initialize_schema(self) -> None:
        """스키마 초기화 (제약조건 및 인덱스)"""
        logger.info("Initializing Neo4j schema...")

        # 제약조건 생성
        c_success, c_failed = SchemaManager.create_constraints(self.query)
        logger.info(f"Constraints: {c_success} created, {c_failed} failed")

        # 인덱스 생성
        i_success, i_failed = SchemaManager.create_indexes(self.query)
        logger.info(f"Indexes: {i_success} created, {i_failed} failed")

    def query(
        self,
        query_str: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Cypher 쿼리 실행.

        Args:
            query_str: Cypher 쿼리 문자열
            params: 쿼리 파라미터

        Returns:
            쿼리 결과 리스트
        """
        try:
            return self.graph.query(query_str, params or {})
        except Exception as e:
            logger.error(f"Query failed: {e}")
            logger.debug(f"Query: {query_str[:200]}...")
            return []

    # ========================================================================
    # MAIN SAVE METHODS
    # ========================================================================

    def save_graph_documents(
        self,
        graph_documents: List[Any],
        brand_id: str,
        use_batch: bool = True,
    ) -> SaveResult:
        """
        그래프 문서를 Neo4j에 저장.

        LLMGraphTransformer에서 생성된 그래프 문서를 저장합니다.
        배치 처리와 중복 제거를 통해 최적화된 저장을 수행합니다.

        Args:
            graph_documents: LLMGraphTransformer 결과 문서 리스트
            brand_id: 브랜드 ID (모든 노드에 강제 적용)
            use_batch: True면 배치 처리 사용 (권장)

        Returns:
            SaveResult: 저장 결과

        Example:
            >>> result = repo.save_graph_documents(
            ...     graph_docs,
            ...     brand_id="my_brand",
            ...     use_batch=True
            ... )
            >>> print(f"Created {result['nodes_created']} nodes")
        """
        result: SaveResult = {
            "success": False,
            "nodes_created": 0,
            "nodes_updated": 0,
            "relationships_created": 0,
            "relationships_updated": 0,
            "errors": [],
        }

        if not graph_documents:
            logger.warning("No graph documents to save")
            result["success"] = True
            return result

        # 통계 로깅
        total_nodes = sum(len(doc.nodes) for doc in graph_documents if hasattr(doc, 'nodes'))
        total_rels = sum(len(doc.relationships) for doc in graph_documents if hasattr(doc, 'relationships'))

        logger.info(f"Saving {len(graph_documents)} documents to Neo4j")
        logger.info(f"  Brand ID: {brand_id}")
        logger.info(f"  Nodes: {total_nodes}, Relationships: {total_rels}")

        try:
            # 데이터 추출 및 중복 제거
            nodes_data = self._extract_nodes(graph_documents, brand_id)
            rels_data = self._extract_relationships(graph_documents)

            logger.info(f"  After dedup: Nodes: {len(nodes_data)}, Relationships: {len(rels_data)}")

            # 저장 실행
            if use_batch and len(nodes_data) > 10:
                node_result = self._save_nodes_batch(nodes_data)
                rel_result = self._save_relationships_batch(rels_data)
            else:
                node_result = self._save_nodes_individual(nodes_data)
                rel_result = self._save_relationships_individual(rels_data)

            # 스키마 갱신
            self.graph.refresh_schema()

            # 결과 집계
            result["nodes_created"] = node_result.get("created", 0)
            result["nodes_updated"] = node_result.get("updated", 0)
            result["relationships_created"] = rel_result.get("created", 0)
            result["relationships_updated"] = rel_result.get("updated", 0)
            result["errors"] = node_result.get("errors", []) + rel_result.get("errors", [])
            result["success"] = len(result["errors"]) == 0

            # 결과 로깅
            logger.info("Save completed:")
            logger.info(f"  Nodes: {result['nodes_created']} created, {result['nodes_updated']} updated")
            logger.info(f"  Relationships: {result['relationships_created']} created")

            return result

        except Exception as e:
            error_msg = f"Save failed: {e}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
            return result

    def _extract_nodes(
        self,
        graph_documents: List[Any],
        brand_id: str,
    ) -> List[NodeData]:
        """노드 데이터 추출 및 중복 제거"""
        nodes_dict: Dict[Tuple, NodeData] = {}

        for doc in graph_documents:
            if not hasattr(doc, 'nodes'):
                continue

            for node in doc.nodes:
                properties = dict(node.properties) if hasattr(node, 'properties') else {}
                properties['id'] = node.id

                # brand_id 강제 적용
                node_brand_id = properties.get('brand_id', brand_id)
                if node.type in NodeTypes.brand_id_required_types():
                    properties['brand_id'] = node_brand_id

                node_data = NodeData(
                    node_type=node.type,
                    node_id=node.id,
                    brand_id=node_brand_id if node.type in NodeTypes.brand_id_required_types() else None,
                    properties=properties,
                )

                # 중복 제거 (유니크 키 기준)
                key = node_data.unique_key()
                nodes_dict[key] = node_data

        return list(nodes_dict.values())

    def _extract_relationships(
        self,
        graph_documents: List[Any],
    ) -> List[RelationshipData]:
        """관계 데이터 추출 및 중복 제거"""
        rels_dict: Dict[Tuple, RelationshipData] = {}

        for doc in graph_documents:
            if not hasattr(doc, 'relationships'):
                continue

            for rel in doc.relationships:
                properties = dict(rel.properties) if hasattr(rel, 'properties') else {}

                # Source brand_id 추출
                src_brand_id = None
                if hasattr(rel.source, 'properties'):
                    src_props = rel.source.properties
                    if isinstance(src_props, dict):
                        src_brand_id = src_props.get('brand_id')
                    else:
                        src_brand_id = getattr(src_props, 'brand_id', None)

                # Target brand_id 추출
                tgt_brand_id = None
                if hasattr(rel.target, 'properties'):
                    tgt_props = rel.target.properties
                    if isinstance(tgt_props, dict):
                        tgt_brand_id = tgt_props.get('brand_id')
                    else:
                        tgt_brand_id = getattr(tgt_props, 'brand_id', None)

                rel_data = RelationshipData(
                    source_type=rel.source.type,
                    source_id=rel.source.id,
                    source_brand_id=src_brand_id,
                    target_type=rel.target.type,
                    target_id=rel.target.id,
                    target_brand_id=tgt_brand_id,
                    rel_type=rel.type,
                    properties=properties,
                )

                # 중복 제거
                key = rel_data.unique_key()
                rels_dict[key] = rel_data

        return list(rels_dict.values())

    def _save_nodes_batch(
        self,
        nodes_data: List[NodeData],
    ) -> Dict[str, Any]:
        """배치 노드 저장"""
        logger.debug("Saving nodes in batch mode")

        result = {"created": 0, "updated": 0, "errors": []}

        # 노드 타입별 그룹화
        nodes_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for node in nodes_data:
            if node.node_type not in nodes_by_type:
                nodes_by_type[node.node_type] = []
            nodes_by_type[node.node_type].append({
                "id": node.node_id,
                "brand_id": node.brand_id,
                "properties": node.properties,
            })

        # 타입별 배치 저장
        for node_type, nodes in nodes_by_type.items():
            use_brand_id = node_type in NodeTypes.brand_id_required_types()
            query = CypherQueryBuilder.build_node_merge_query(node_type, use_brand_id)

            # 배치 분할
            for i in range(0, len(nodes), self.batch_size):
                batch = nodes[i:i + self.batch_size]
                try:
                    query_result = self.query(query, {"nodes": batch})
                    if query_result:
                        created = query_result[0].get("created", 0)
                        total = query_result[0].get("total", 0)
                        result["created"] += created
                        result["updated"] += total - created
                        logger.debug(f"  {node_type}: {created} created, {total - created} updated")
                except Exception as e:
                    error_msg = f"Batch save failed for {node_type}: {e}"
                    logger.error(error_msg)
                    result["errors"].append(error_msg)

        return result

    def _save_relationships_batch(
        self,
        rels_data: List[RelationshipData],
    ) -> Dict[str, Any]:
        """배치 관계 저장"""
        logger.debug("Saving relationships in batch mode")

        result = {"created": 0, "updated": 0, "errors": []}

        # 관계 타입별 그룹화
        rels_by_type: Dict[Tuple, List[Dict[str, Any]]] = {}
        for rel in rels_data:
            key = (rel.source_type, rel.rel_type, rel.target_type)
            if key not in rels_by_type:
                rels_by_type[key] = []
            rels_by_type[key].append({
                "source_id": rel.source_id,
                "source_brand_id": rel.source_brand_id,
                "target_id": rel.target_id,
                "target_brand_id": rel.target_brand_id,
                "properties": rel.properties,
            })

        # 타입별 배치 저장
        for (src_type, rel_type, tgt_type), rels in rels_by_type.items():
            src_has_brand = src_type in NodeTypes.brand_id_required_types()
            tgt_has_brand = tgt_type in NodeTypes.brand_id_required_types()

            query = CypherQueryBuilder.build_relationship_merge_query(
                src_type, rel_type, tgt_type, src_has_brand, tgt_has_brand
            )

            # 배치 분할
            for i in range(0, len(rels), self.batch_size):
                batch = rels[i:i + self.batch_size]
                try:
                    query_result = self.query(query, {"rels": batch})
                    if query_result:
                        count = query_result[0].get("count", 0)
                        result["created"] += count
                        logger.debug(f"  {rel_type}: {count} relationships")
                except Exception as e:
                    error_msg = f"Batch save failed for {rel_type}: {e}"
                    logger.error(error_msg)
                    result["errors"].append(error_msg)

        return result

    def _save_nodes_individual(
        self,
        nodes_data: List[NodeData],
    ) -> Dict[str, Any]:
        """개별 노드 저장"""
        logger.debug("Saving nodes in individual mode")

        result = {"created": 0, "updated": 0, "errors": []}

        for node in nodes_data:
            use_brand_id = node.node_type in NodeTypes.brand_id_required_types()

            if use_brand_id:
                query = f"""
                MERGE (n:{node.node_type} {{id: $id, brand_id: $brand_id}})
                ON CREATE SET
                    n += $properties,
                    n.created_at = datetime(),
                    n._is_new = true
                ON MATCH SET
                    n += $properties,
                    n.updated_at = datetime(),
                    n._is_new = false
                RETURN n._is_new as is_new
                """
                params = {"id": node.node_id, "brand_id": node.brand_id, "properties": node.properties}
            else:
                query = f"""
                MERGE (n:{node.node_type} {{id: $id}})
                ON CREATE SET
                    n += $properties,
                    n.created_at = datetime(),
                    n._is_new = true
                ON MATCH SET
                    n += $properties,
                    n.updated_at = datetime(),
                    n._is_new = false
                RETURN n._is_new as is_new
                """
                params = {"id": node.node_id, "properties": node.properties}

            try:
                query_result = self.query(query, params)
                if query_result and query_result[0].get("is_new"):
                    result["created"] += 1
                else:
                    result["updated"] += 1
            except Exception as e:
                result["errors"].append(f"Node save failed ({node.node_id}): {e}")

        return result

    def _save_relationships_individual(
        self,
        rels_data: List[RelationshipData],
    ) -> Dict[str, Any]:
        """개별 관계 저장"""
        logger.debug("Saving relationships in individual mode")

        result = {"created": 0, "updated": 0, "errors": []}

        for rel in rels_data:
            src_has_brand = rel.source_type in NodeTypes.brand_id_required_types()
            tgt_has_brand = rel.target_type in NodeTypes.brand_id_required_types()

            # Source MATCH
            if src_has_brand and rel.source_brand_id:
                src_match = f"MATCH (source:{rel.source_type} {{id: $source_id, brand_id: $source_brand_id}})"
            else:
                src_match = f"MATCH (source:{rel.source_type} {{id: $source_id}})"

            # Target MATCH
            if tgt_has_brand and rel.target_brand_id:
                tgt_match = f"MATCH (target:{rel.target_type} {{id: $target_id, brand_id: $target_brand_id}})"
            else:
                tgt_match = f"MATCH (target:{rel.target_type} {{id: $target_id}})"

            query = f"""
            {src_match}
            {tgt_match}
            MERGE (source)-[r:{rel.rel_type}]->(target)
            ON CREATE SET
                r += $properties,
                r.created_at = datetime()
            ON MATCH SET
                r += $properties,
                r.updated_at = datetime()
            RETURN id(r) as rel_id
            """

            params = {
                "source_id": rel.source_id,
                "target_id": rel.target_id,
                "properties": rel.properties,
            }
            if src_has_brand and rel.source_brand_id:
                params["source_brand_id"] = rel.source_brand_id
            if tgt_has_brand and rel.target_brand_id:
                params["target_brand_id"] = rel.target_brand_id

            try:
                query_result = self.query(query, params)
                if query_result:
                    result["created"] += 1
            except Exception as e:
                result["errors"].append(f"Relationship save failed: {e}")

        return result

    # ========================================================================
    # CONTENT EXISTENCE CHECKS
    # ========================================================================

    def content_exists(
        self,
        content_id: str,
        platform: PlatformType,
        brand_id: str,
    ) -> bool:
        """
        콘텐츠 존재 여부 확인.

        Args:
            content_id: 콘텐츠 ID
            platform: 플랫폼 타입
            brand_id: 브랜드 ID

        Returns:
            존재하면 True
        """
        query = """
        MATCH (c:Content {id: $content_id, brand_id: $brand_id, platform: $platform})
        RETURN count(c) > 0 as exists
        """
        result = self.query(query, {
            "content_id": content_id,
            "brand_id": brand_id,
            "platform": platform.value,
        })
        return result[0]["exists"] if result else False

    def content_exists_by_url(
        self,
        url: str,
        brand_id: str,
    ) -> bool:
        """
        URL로 콘텐츠 존재 여부 확인.

        Args:
            url: 콘텐츠 URL
            brand_id: 브랜드 ID

        Returns:
            존재하면 True
        """
        query = """
        MATCH (c:Content {url: $url, brand_id: $brand_id})
        RETURN count(c) > 0 as exists
        """
        result = self.query(query, {"url": url, "brand_id": brand_id})
        return result[0]["exists"] if result else False

    def filter_new_contents(
        self,
        contents: List[ContentDTO],
        brand_id: str,
    ) -> List[ContentDTO]:
        """
        새 콘텐츠만 필터링.

        이미 DB에 존재하는 콘텐츠를 제외하고 새 콘텐츠만 반환합니다.

        Args:
            contents: 콘텐츠 DTO 리스트
            brand_id: 브랜드 ID

        Returns:
            새 콘텐츠 DTO 리스트
        """
        if not contents:
            return []

        new_contents = []
        for content in contents:
            # URL 기반 체크
            if content.url and self.content_exists_by_url(content.url, brand_id):
                continue

            # ID 기반 체크
            if content.content_id and self.content_exists(content.content_id, content.platform, brand_id):
                continue

            new_contents.append(content)

        logger.info(f"Filtered contents: {len(new_contents)}/{len(contents)} are new")
        return new_contents

    # ========================================================================
    # STATISTICS METHODS
    # ========================================================================

    def get_node_count(
        self,
        brand_id: Optional[str] = None,
        node_type: Optional[str] = None,
    ) -> int:
        """
        노드 수 조회.

        Args:
            brand_id: 브랜드 ID (필터링)
            node_type: 노드 타입 (필터링)

        Returns:
            노드 수
        """
        conditions = []
        params = {}

        if brand_id:
            conditions.append("n.brand_id = $brand_id")
            params["brand_id"] = brand_id

        if node_type:
            query = f"MATCH (n:{node_type})"
        else:
            query = "MATCH (n)"

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " RETURN count(n) as count"

        result = self.query(query, params)
        return result[0]["count"] if result else 0

    def get_relationship_count(
        self,
        brand_id: Optional[str] = None,
        rel_type: Optional[str] = None,
    ) -> int:
        """
        관계 수 조회.

        Args:
            brand_id: 브랜드 ID (필터링)
            rel_type: 관계 타입 (필터링)

        Returns:
            관계 수
        """
        if rel_type:
            rel_pattern = f"[r:{rel_type}]"
        else:
            rel_pattern = "[r]"

        if brand_id:
            query = f"""
            MATCH (n)-{rel_pattern}->()
            WHERE n.brand_id = $brand_id
            RETURN count(r) as count
            """
            result = self.query(query, {"brand_id": brand_id})
        else:
            query = f"MATCH ()-{rel_pattern}->() RETURN count(r) as count"
            result = self.query(query)

        return result[0]["count"] if result else 0

    def get_brand_statistics(self, brand_id: str) -> BrandStatistics:
        """
        브랜드 통계 조회.

        Args:
            brand_id: 브랜드 ID

        Returns:
            BrandStatistics: 브랜드 통계
        """
        # 노드 타입별 수
        node_types_query = """
        MATCH (n)
        WHERE n.brand_id = $brand_id
        RETURN labels(n)[0] as type, count(n) as count
        """
        node_types_result = self.query(node_types_query, {"brand_id": brand_id})
        node_types = {r["type"]: r["count"] for r in node_types_result}

        # 관계 타입별 수
        rel_types_query = """
        MATCH (n)-[r]->()
        WHERE n.brand_id = $brand_id
        RETURN type(r) as type, count(r) as count
        """
        rel_types_result = self.query(rel_types_query, {"brand_id": brand_id})
        rel_types = {r["type"]: r["count"] for r in rel_types_result}

        # 플랫폼별 수
        platforms_query = """
        MATCH (c:Content)
        WHERE c.brand_id = $brand_id
        RETURN c.platform as platform, count(c) as count
        """
        platforms_result = self.query(platforms_query, {"brand_id": brand_id})
        platforms = {r["platform"]: r["count"] for r in platforms_result if r["platform"]}

        # 날짜 범위
        date_range_query = """
        MATCH (c:Content)
        WHERE c.brand_id = $brand_id AND c.created_at IS NOT NULL
        RETURN min(c.created_at) as first, max(c.created_at) as last
        """
        date_range_result = self.query(date_range_query, {"brand_id": brand_id})
        date_range = {"first": None, "last": None}
        if date_range_result:
            first = date_range_result[0].get("first")
            last = date_range_result[0].get("last")
            date_range["first"] = str(first) if first else None
            date_range["last"] = str(last) if last else None

        return {
            "brand_id": brand_id,
            "total_nodes": sum(node_types.values()),
            "total_relationships": sum(rel_types.values()),
            "node_types": node_types,
            "relationship_types": rel_types,
            "platforms": platforms,
            "date_range": date_range,
        }

    def get_platform_statistics(
        self,
        platform: PlatformType,
        brand_id: str,
    ) -> ContentStatistics:
        """
        플랫폼별 통계 조회.

        Args:
            platform: 플랫폼 타입
            brand_id: 브랜드 ID

        Returns:
            ContentStatistics: 콘텐츠 통계
        """
        # 콘텐츠 수
        count_query = """
        MATCH (c:Content)
        WHERE c.brand_id = $brand_id AND c.platform = $platform
        RETURN count(c) as count
        """
        count_result = self.query(count_query, {
            "brand_id": brand_id,
            "platform": platform.value,
        })
        total_count = count_result[0]["count"] if count_result else 0

        # 콘텐츠 타입별 수
        by_type_query = """
        MATCH (c:Content)
        WHERE c.brand_id = $brand_id AND c.platform = $platform
        RETURN c.content_type as type, count(c) as count
        """
        by_type_result = self.query(by_type_query, {
            "brand_id": brand_id,
            "platform": platform.value,
        })
        by_type = {r["type"]: r["count"] for r in by_type_result if r["type"]}

        # 인게이지먼트 합계
        engagement_query = """
        MATCH (c:Content)
        WHERE c.brand_id = $brand_id AND c.platform = $platform
        RETURN
            sum(coalesce(c.like_count, 0)) as likes,
            sum(coalesce(c.comment_count, 0)) as comments,
            sum(coalesce(c.share_count, 0)) as shares,
            sum(coalesce(c.view_count, 0)) as views
        """
        engagement_result = self.query(engagement_query, {
            "brand_id": brand_id,
            "platform": platform.value,
        })
        engagement = {}
        if engagement_result:
            engagement = {
                "likes": engagement_result[0].get("likes", 0) or 0,
                "comments": engagement_result[0].get("comments", 0) or 0,
                "shares": engagement_result[0].get("shares", 0) or 0,
                "views": engagement_result[0].get("views", 0) or 0,
            }

        # 날짜 범위
        date_range_query = """
        MATCH (c:Content)
        WHERE c.brand_id = $brand_id AND c.platform = $platform AND c.created_at IS NOT NULL
        RETURN min(c.created_at) as first, max(c.created_at) as last
        """
        date_range_result = self.query(date_range_query, {
            "brand_id": brand_id,
            "platform": platform.value,
        })
        date_range = {"first": None, "last": None}
        if date_range_result:
            first = date_range_result[0].get("first")
            last = date_range_result[0].get("last")
            date_range["first"] = str(first) if first else None
            date_range["last"] = str(last) if last else None

        return {
            "total_count": total_count,
            "by_platform": {platform.value: total_count},
            "by_type": by_type,
            "engagement": engagement,
            "date_range": date_range,
        }

    def get_content_statistics(
        self,
        brand_id: str,
        platform: Optional[PlatformType] = None,
    ) -> ContentStatistics:
        """
        콘텐츠 통계 조회.

        Args:
            brand_id: 브랜드 ID
            platform: 플랫폼 타입 (선택)

        Returns:
            ContentStatistics: 콘텐츠 통계
        """
        if platform:
            return self.get_platform_statistics(platform, brand_id)

        # 전체 플랫폼 통계
        count_query = """
        MATCH (c:Content)
        WHERE c.brand_id = $brand_id
        RETURN count(c) as count
        """
        count_result = self.query(count_query, {"brand_id": brand_id})
        total_count = count_result[0]["count"] if count_result else 0

        # 플랫폼별 수
        by_platform_query = """
        MATCH (c:Content)
        WHERE c.brand_id = $brand_id
        RETURN c.platform as platform, count(c) as count
        """
        by_platform_result = self.query(by_platform_query, {"brand_id": brand_id})
        by_platform = {r["platform"]: r["count"] for r in by_platform_result if r["platform"]}

        # 타입별 수
        by_type_query = """
        MATCH (c:Content)
        WHERE c.brand_id = $brand_id
        RETURN c.content_type as type, count(c) as count
        """
        by_type_result = self.query(by_type_query, {"brand_id": brand_id})
        by_type = {r["type"]: r["count"] for r in by_type_result if r["type"]}

        # 인게이지먼트
        engagement_query = """
        MATCH (c:Content)
        WHERE c.brand_id = $brand_id
        RETURN
            sum(coalesce(c.like_count, 0)) as likes,
            sum(coalesce(c.comment_count, 0)) as comments,
            sum(coalesce(c.share_count, 0)) as shares,
            sum(coalesce(c.view_count, 0)) as views
        """
        engagement_result = self.query(engagement_query, {"brand_id": brand_id})
        engagement = {}
        if engagement_result:
            engagement = {
                "likes": engagement_result[0].get("likes", 0) or 0,
                "comments": engagement_result[0].get("comments", 0) or 0,
                "shares": engagement_result[0].get("shares", 0) or 0,
                "views": engagement_result[0].get("views", 0) or 0,
            }

        # 날짜 범위
        date_range_query = """
        MATCH (c:Content)
        WHERE c.brand_id = $brand_id AND c.created_at IS NOT NULL
        RETURN min(c.created_at) as first, max(c.created_at) as last
        """
        date_range_result = self.query(date_range_query, {"brand_id": brand_id})
        date_range = {"first": None, "last": None}
        if date_range_result:
            first = date_range_result[0].get("first")
            last = date_range_result[0].get("last")
            date_range["first"] = str(first) if first else None
            date_range["last"] = str(last) if last else None

        return {
            "total_count": total_count,
            "by_platform": by_platform,
            "by_type": by_type,
            "engagement": engagement,
            "date_range": date_range,
        }

    def get_interaction_statistics(
        self,
        brand_id: str,
        platform: Optional[PlatformType] = None,
    ) -> InteractionStatistics:
        """
        인터랙션 통계 조회.

        Args:
            brand_id: 브랜드 ID
            platform: 플랫폼 타입 (선택)

        Returns:
            InteractionStatistics: 인터랙션 통계
        """
        # 기본 쿼리
        conditions = ["i.brand_id = $brand_id"]
        params = {"brand_id": brand_id}

        if platform:
            conditions.append("i.platform = $platform")
            params["platform"] = platform.value

        where_clause = " AND ".join(conditions)

        # 총 수
        count_query = f"""
        MATCH (i:Interaction)
        WHERE {where_clause}
        RETURN count(i) as count
        """
        count_result = self.query(count_query, params)
        total_count = count_result[0]["count"] if count_result else 0

        # 플랫폼별 수
        by_platform_query = f"""
        MATCH (i:Interaction)
        WHERE {where_clause}
        RETURN i.platform as platform, count(i) as count
        """
        by_platform_result = self.query(by_platform_query, params)
        by_platform = {r["platform"]: r["count"] for r in by_platform_result if r["platform"]}

        # 콘텐츠당 평균
        avg_query = f"""
        MATCH (c:Content)-[:HAS_INTERACTION]->(i:Interaction)
        WHERE c.brand_id = $brand_id
        WITH c, count(i) as interactions
        RETURN avg(interactions) as avg_per_content
        """
        avg_result = self.query(avg_query, {"brand_id": brand_id})
        avg_per_content = avg_result[0].get("avg_per_content", 0) if avg_result else 0

        # 감정 분포
        sentiment_query = f"""
        MATCH (i:Interaction)
        WHERE {where_clause}
        RETURN i.sentiment as sentiment, count(i) as count
        """
        sentiment_result = self.query(sentiment_query, params)
        sentiment_distribution = {r["sentiment"]: r["count"] for r in sentiment_result if r["sentiment"]}

        return {
            "total_count": total_count,
            "by_platform": by_platform,
            "average_per_content": float(avg_per_content or 0),
            "sentiment_distribution": sentiment_distribution,
        }

    def get_concept_cloud(
        self,
        brand_id: str,
        limit: int = 50,
    ) -> List[ConceptCloudItem]:
        """
        개념 클라우드 (빈도수 기반) 조회.

        Args:
            brand_id: 브랜드 ID
            limit: 최대 개수

        Returns:
            ConceptCloudItem 리스트
        """
        query = """
        MATCH (c:Concept)
        WHERE c.brand_id = $brand_id
        OPTIONAL MATCH (c)<-[:MENTIONS_CONCEPT]-(content)
        WITH c, count(content) as mention_count
        ORDER BY mention_count DESC
        LIMIT $limit
        OPTIONAL MATCH (c)-[:RELATED_TO]-(related:Concept)
        WHERE related.brand_id = $brand_id
        RETURN
            c.id as name,
            mention_count as count,
            c.type as type,
            collect(DISTINCT related.id)[0..5] as related_concepts
        """

        result = self.query(query, {"brand_id": brand_id, "limit": limit})

        return [
            {
                "name": r["name"],
                "count": r["count"],
                "type": r["type"],
                "related_concepts": r["related_concepts"] or [],
            }
            for r in result
        ]

    def get_brand_graph_summary(self, brand_id: str) -> Dict[str, Any]:
        """
        브랜드 그래프 요약.

        Args:
            brand_id: 브랜드 ID

        Returns:
            그래프 요약 딕셔너리
        """
        stats = self.get_brand_statistics(brand_id)
        concept_cloud = self.get_concept_cloud(brand_id, limit=20)

        return {
            "brand_id": brand_id,
            "summary": {
                "total_nodes": stats["total_nodes"],
                "total_relationships": stats["total_relationships"],
            },
            "node_types": stats["node_types"],
            "relationship_types": stats["relationship_types"],
            "platforms": stats["platforms"],
            "date_range": stats["date_range"],
            "top_concepts": concept_cloud[:10],
        }

    def get_graph_visualization_data(
        self,
        brand_id: str,
        limit: int = 100,
        node_types: Optional[List[str]] = None,
        balanced: bool = False,
    ) -> Dict[str, Any]:
        """
        그래프 시각화를 위한 노드와 관계 데이터 조회.

        Args:
            brand_id: 브랜드 ID
            limit: 최대 노드 수 (기본값 100)
            node_types: 필터링할 노드 타입 리스트 (선택)
            balanced: True면 타입별로 균형있게 노드를 가져옴 (카드용)

        Returns:
            Dict containing:
                - nodes: 노드 리스트 (id, label, type, properties)
                - relationships: 관계 리스트 (source, target, type, properties)
                - stats: 통계 정보
        """
        # Debug logging
        logger.info(f"get_graph_visualization_data: brand_id={brand_id}, limit={limit}, balanced={balanced}")

        if balanced:
            # 타입별로 균형있게 가져오기 (카드 미리보기용)
            logger.info("Using balanced graph data function")
            return self._get_balanced_graph_data(brand_id, limit)

        # 노드 타입 필터 조건
        type_conditions = []
        if node_types:
            type_conditions = [f"n:{t}" for t in node_types] + [f"m:{t}" for t in node_types]

        type_filter = ""
        if type_conditions:
            type_filter = f"AND ({' OR '.join(type_conditions)})"

        # 직접 노드와 관계를 함께 가져오는 쿼리
        # Brand 노드도 brand_id 속성으로 매칭 (Brand.id와 brand_id가 다를 수 있음)
        graph_query = f"""
        MATCH (n)-[r]-(m)
        WHERE (n.brand_id = $brand_id OR (n:Brand AND n.brand_id = $brand_id))
          AND (m.brand_id = $brand_id OR (m:Brand AND m.brand_id = $brand_id))
          {type_filter}
        RETURN DISTINCT
            n.id as source_id,
            labels(n)[0] as source_type,
            COALESCE(n.name, n.text, n.id) as source_name,
            properties(n) as source_props,
            m.id as target_id,
            labels(m)[0] as target_type,
            COALESCE(m.name, m.text, m.id) as target_name,
            properties(m) as target_props,
            type(r) as rel_type,
            startNode(r) = n as is_outgoing
        LIMIT $rel_limit
        """

        rel_limit = limit * 3
        result = self.query(graph_query, {"brand_id": brand_id, "rel_limit": rel_limit})

        # 노드와 관계 추출
        nodes_map: Dict[str, Dict[str, Any]] = {}
        relationships: List[Dict[str, Any]] = []
        seen_rels: Set[str] = set()

        for r in result:
            # Source 노드 추가
            source_id = r["source_id"]
            if source_id and source_id not in nodes_map:
                props = r["source_props"] or {}
                name = r["source_name"] or props.get("name") or source_id
                label = str(name)[:35] if name else r["source_type"]
                if len(str(name)) > 35:
                    label += "..."

                nodes_map[source_id] = {
                    "id": source_id,
                    "label": label,
                    "type": r["source_type"],
                    "properties": props,
                }

            # Target 노드 추가
            target_id = r["target_id"]
            if target_id and target_id not in nodes_map:
                props = r["target_props"] or {}
                name = r["target_name"] or props.get("name") or target_id
                label = str(name)[:35] if name else r["target_type"]
                if len(str(name)) > 35:
                    label += "..."

                nodes_map[target_id] = {
                    "id": target_id,
                    "label": label,
                    "type": r["target_type"],
                    "properties": props,
                }

            # 관계 추가 (방향 고려)
            if source_id and target_id and r["rel_type"]:
                # 실제 관계 방향 결정
                if r["is_outgoing"]:
                    actual_source, actual_target = source_id, target_id
                else:
                    actual_source, actual_target = target_id, source_id

                rel_key = f"{actual_source}-{r['rel_type']}-{actual_target}"
                if rel_key not in seen_rels:
                    seen_rels.add(rel_key)
                    relationships.append({
                        "source": actual_source,
                        "target": actual_target,
                        "type": r["rel_type"],
                        "properties": {},
                    })

        # limit 적용
        nodes = list(nodes_map.values())[:limit]
        node_ids_set = {n["id"] for n in nodes}

        # 양쪽 노드가 모두 있는 관계만 필터링
        final_relationships = [
            rel for rel in relationships
            if rel["source"] in node_ids_set and rel["target"] in node_ids_set
        ]

        # 통계 조회
        stats = self.get_brand_statistics(brand_id)

        return {
            "nodes": nodes,
            "relationships": final_relationships,
            "stats": {
                "total_nodes": stats["total_nodes"],
                "total_relationships": stats["total_relationships"],
                "returned_nodes": len(nodes),
                "returned_relationships": len(final_relationships),
                "node_types": stats["node_types"],
                "relationship_types": stats["relationship_types"],
            },
        }

    def _get_balanced_graph_data(
        self,
        brand_id: str,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        타입별로 균형있게 노드를 가져옵니다 (카드 미리보기용).

        각 노드 타입에서 골고루 가져와 다양한 색상이 표시되도록 합니다.

        Args:
            brand_id: 브랜드 ID
            limit: 총 노드 수

        Returns:
            균형잡힌 노드와 관계 데이터
        """
        # 타입별 할당량 계산 (Brand 1개 + 나머지 균등 분배)
        type_priority = ["Brand", "Content", "Concept", "Interaction", "Topic", "Actor"]
        per_type = max(3, (limit - 1) // (len(type_priority) - 1))

        nodes_map: Dict[str, Dict[str, Any]] = {}
        all_node_ids: Set[str] = set()

        # 1. Brand 노드 (1개) - brand_id 또는 id로 매칭
        brand_query = """
        MATCH (b:Brand)
        WHERE b.brand_id = $brand_id OR toLower(b.id) = toLower($brand_id)
        RETURN b.id as id, labels(b)[0] as type,
               COALESCE(b.name, b.id) as name, properties(b) as props
        LIMIT 1
        """
        brand_result = self.query(brand_query, {"brand_id": brand_id})
        for r in brand_result:
            node_id = r["id"]
            if node_id:
                nodes_map[node_id] = {
                    "id": node_id,
                    "label": str(r["name"])[:35] if r["name"] else "Brand",
                    "type": r["type"],
                    "properties": r["props"] or {},
                }
                all_node_ids.add(node_id)

        # 2. 각 타입별 노드 가져오기 (관계가 있는 노드 우선)
        for node_type in type_priority[1:]:  # Brand 제외
            type_query = f"""
            MATCH (n:{node_type})
            WHERE n.brand_id = $brand_id
            OPTIONAL MATCH (n)-[r]-()
            WITH n, count(r) as rel_count
            ORDER BY rel_count DESC
            LIMIT $per_type
            RETURN n.id as id, labels(n)[0] as type,
                   COALESCE(n.name, n.text, n.id) as name,
                   properties(n) as props
            """
            type_result = self.query(type_query, {
                "brand_id": brand_id,
                "per_type": per_type,
            })

            for r in type_result:
                node_id = r["id"]
                if node_id and node_id not in nodes_map:
                    name = r["name"] or node_type
                    label = str(name)[:35]
                    if len(str(name)) > 35:
                        label += "..."

                    nodes_map[node_id] = {
                        "id": node_id,
                        "label": label,
                        "type": r["type"],
                        "properties": r["props"] or {},
                    }
                    all_node_ids.add(node_id)

        # 3. 수집된 노드들 사이의 관계 가져오기
        if len(all_node_ids) > 1:
            node_ids_list = list(all_node_ids)
            rel_query = """
            MATCH (n)-[r]->(m)
            WHERE n.id IN $node_ids AND m.id IN $node_ids
            RETURN DISTINCT n.id as source, m.id as target, type(r) as rel_type
            """
            rel_result = self.query(rel_query, {"node_ids": node_ids_list})

            relationships = []
            seen_rels: Set[str] = set()

            for r in rel_result:
                if r["source"] and r["target"] and r["rel_type"]:
                    rel_key = f"{r['source']}-{r['rel_type']}-{r['target']}"
                    if rel_key not in seen_rels:
                        seen_rels.add(rel_key)
                        relationships.append({
                            "source": r["source"],
                            "target": r["target"],
                            "type": r["rel_type"],
                            "properties": {},
                        })
        else:
            relationships = []

        # 4. Brand와 다른 노드들 사이의 관계도 가져오기
        # Brand 노드의 id가 brand_id와 다를 수 있으므로 brand_id로 매칭
        brand_node_id = None
        for r in brand_result:
            brand_node_id = r["id"]
            break

        if brand_node_id:
            brand_rel_query = """
            MATCH (b:Brand)-[r]-(n)
            WHERE b.brand_id = $brand_id AND n.id IN $node_ids
            RETURN b.id as brand_id, n.id as node_id, type(r) as rel_type,
                   startNode(r) = b as is_outgoing
            """
            brand_rel_result = self.query(brand_rel_query, {
                "brand_id": brand_id,
                "node_ids": list(all_node_ids - {brand_node_id}),
            })

            for r in brand_rel_result:
                if r["brand_id"] and r["node_id"] and r["rel_type"]:
                    if r["is_outgoing"]:
                        source, target = r["brand_id"], r["node_id"]
                    else:
                        source, target = r["node_id"], r["brand_id"]

                    rel_key = f"{source}-{r['rel_type']}-{target}"
                    if rel_key not in seen_rels:
                        seen_rels.add(rel_key)
                        relationships.append({
                            "source": source,
                            "target": target,
                            "type": r["rel_type"],
                            "properties": {},
                        })

        # 결과
        nodes = list(nodes_map.values())[:limit]
        stats = self.get_brand_statistics(brand_id)

        # 반환된 노드 타입 집계
        returned_types: Dict[str, int] = {}
        for node in nodes:
            t = node["type"]
            returned_types[t] = returned_types.get(t, 0) + 1

        return {
            "nodes": nodes,
            "relationships": relationships,
            "stats": {
                "total_nodes": stats["total_nodes"],
                "total_relationships": stats["total_relationships"],
                "returned_nodes": len(nodes),
                "returned_relationships": len(relationships),
                "node_types": returned_types,
                "relationship_types": stats["relationship_types"],
                "balanced": True,
            },
        }

    # ========================================================================
    # LEGACY COMPATIBILITY METHODS
    # ========================================================================

    def save(
        self,
        graph_documents: List[Any],
        use_batch: bool = True,
        brand_id: Optional[str] = None,
    ) -> bool:
        """
        레거시 호환성을 위한 저장 메서드.

        DEPRECATED: save_graph_documents()를 사용하세요.
        """
        import warnings
        warnings.warn(
            "save() is deprecated. Use save_graph_documents() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # brand_id 추출 시도
        _brand_id = brand_id
        if not _brand_id and graph_documents:
            for doc in graph_documents:
                if hasattr(doc, 'nodes'):
                    for node in doc.nodes:
                        if hasattr(node, 'properties') and isinstance(node.properties, dict):
                            _brand_id = node.properties.get('brand_id')
                            if _brand_id:
                                break
                if _brand_id:
                    break

        result = self.save_graph_documents(
            graph_documents,
            brand_id=_brand_id or "unknown",
            use_batch=use_batch,
        )
        return result["success"]

    def check_post_exists(
        self,
        url: str,
        brand_id: Optional[str] = None,
    ) -> bool:
        """
        레거시 호환성을 위한 게시물 존재 확인.

        DEPRECATED: content_exists_by_url()을 사용하세요.
        """
        import warnings
        warnings.warn(
            "check_post_exists() is deprecated. Use content_exists_by_url() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if brand_id:
            return self.content_exists_by_url(url, brand_id)

        # brand_id 없이 검색 (레거시 호환)
        # Content와 Post 둘 다 확인
        query = """
        MATCH (n)
        WHERE (n:Content OR n:Post) AND n.url = $url
        RETURN count(n) > 0 as exists
        """
        result = self.query(query, {"url": url})
        return result[0]["exists"] if result else False

    def get_new_posts_only(
        self,
        posts_data: List[Dict[str, Any]],
        brand_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        레거시 호환성을 위한 새 게시물 필터링.

        DEPRECATED: filter_new_contents()를 사용하세요.
        """
        import warnings
        warnings.warn(
            "get_new_posts_only() is deprecated. Use filter_new_contents() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        new_posts = []
        for post in posts_data:
            url = post.get("url")
            if url and not self.check_post_exists(url, brand_id):
                new_posts.append(post)

        logger.info(f"Filtered posts: {len(new_posts)}/{len(posts_data)} are new")
        return new_posts

    def get_statistics(
        self,
        brand_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        레거시 호환성을 위한 통계 메서드.

        DEPRECATED: get_brand_statistics()를 사용하세요.
        """
        return {
            "nodes": self.get_node_count(brand_id),
            "relationships": self.get_relationship_count(brand_id),
        }

    # ========================================================================
    # CLEANUP & MAINTENANCE
    # ========================================================================

    def delete_brand_data(
        self,
        brand_id: str,
        confirm: bool = False,
    ) -> Dict[str, int]:
        """
        브랜드 데이터 삭제.

        WARNING: 이 작업은 되돌릴 수 없습니다!

        Args:
            brand_id: 삭제할 브랜드 ID
            confirm: True여야 실제 삭제 실행

        Returns:
            삭제된 노드/관계 수
        """
        if not confirm:
            logger.warning("Delete operation requires confirm=True")
            return {"nodes_deleted": 0, "relationships_deleted": 0}

        logger.warning(f"Deleting all data for brand: {brand_id}")

        # 관계 먼저 삭제
        rel_query = """
        MATCH (n)-[r]->()
        WHERE n.brand_id = $brand_id
        DELETE r
        RETURN count(r) as deleted
        """
        rel_result = self.query(rel_query, {"brand_id": brand_id})
        rels_deleted = rel_result[0]["deleted"] if rel_result else 0

        # 노드 삭제
        node_query = """
        MATCH (n)
        WHERE n.brand_id = $brand_id
        DELETE n
        RETURN count(n) as deleted
        """
        node_result = self.query(node_query, {"brand_id": brand_id})
        nodes_deleted = node_result[0]["deleted"] if node_result else 0

        logger.info(f"Deleted: {nodes_deleted} nodes, {rels_deleted} relationships")

        return {
            "nodes_deleted": nodes_deleted,
            "relationships_deleted": rels_deleted,
        }

    def cleanup_orphan_nodes(self, brand_id: str) -> int:
        """
        고아 노드 정리 (관계 없는 Concept/Topic 삭제).

        Args:
            brand_id: 브랜드 ID

        Returns:
            삭제된 노드 수
        """
        query = """
        MATCH (n)
        WHERE n.brand_id = $brand_id
          AND (n:Concept OR n:Topic)
          AND NOT (n)--()
        DELETE n
        RETURN count(n) as deleted
        """
        result = self.query(query, {"brand_id": brand_id})
        deleted = result[0]["deleted"] if result else 0

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} orphan nodes")

        return deleted

    def close(self) -> None:
        """연결 종료"""
        logger.info("Closing Neo4j connection")
        # Neo4jGraph는 명시적 close가 없으므로 pass
        pass


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_neo4j_repository(
    **kwargs: Any,
) -> Neo4jRepository:
    """
    Neo4jRepository 팩토리 함수.

    환경에 따라 적절한 설정으로 저장소를 생성합니다.

    Args:
        **kwargs: Neo4jRepository 추가 인자

    Returns:
        설정된 Neo4jRepository 인스턴스

    Example:
        >>> repo = create_neo4j_repository()
        >>> repo = create_neo4j_repository(batch_size=1000)
    """
    return Neo4jRepository(**kwargs)


# ============================================================================
# ASYNC WRAPPER (for pipeline integration)
# ============================================================================

class AsyncNeo4jRepository:
    """
    비동기 Neo4j 저장소 래퍼.

    기존 파이프라인 호환성을 위한 비동기 인터페이스입니다.
    내부적으로 동기 Neo4jRepository를 사용합니다.
    """

    def __init__(self, sync_repo: Optional[Neo4jRepository] = None, **kwargs):
        """
        비동기 저장소 초기화.

        Args:
            sync_repo: 기존 동기 저장소 인스턴스 (재사용)
            **kwargs: Neo4jRepository 생성 인자
        """
        self._sync_repo = sync_repo or Neo4jRepository(**kwargs)

    async def create_actor(self, actor: ActorDTO) -> bool:
        """Actor 노드 생성"""
        query = """
        MERGE (a:Actor {actor_id: $actor_id, platform: $platform})
        SET a.username = $username,
            a.display_name = $display_name,
            a.follower_count = $follower_count,
            a.verified = $verified,
            a.profile_url = $profile_url,
            a.updated_at = datetime()
        RETURN a
        """
        result = self._sync_repo.query(query, {
            "actor_id": actor.actor_id,
            "platform": actor.platform.value,
            "username": actor.username,
            "display_name": actor.display_name,
            "follower_count": actor.follower_count,
            "verified": actor.verified,
            "profile_url": actor.profile_url,
        })
        return bool(result)

    async def create_content(self, content: ContentDTO, brand_id: str) -> bool:
        """Content 노드 생성 (metadata 포함)"""
        import json

        query = """
        MERGE (c:Content {id: $id, brand_id: $brand_id})
        SET c.platform = $platform,
            c.text = $text,
            c.url = $url,
            c.content_type = $content_type,
            c.created_at = $created_at,
            c.like_count = $like_count,
            c.comment_count = $comment_count,
            c.share_count = $share_count,
            c.view_count = $view_count,
            c.hashtags = $hashtags,
            c.mentions = $mentions,
            c.media_urls = $media_urls,
            c.thumbnail_url = $thumbnail_url,
            c.location = $location,
            c.metadata = $metadata,
            c.updated_at = datetime()
        RETURN c
        """
        # metadata를 JSON 문자열로 변환 (Neo4j는 중첩 딕셔너리를 직접 저장 불가)
        metadata_str = json.dumps(content.metadata, ensure_ascii=False) if content.metadata else None

        result = self._sync_repo.query(query, {
            "id": content.content_id,
            "brand_id": brand_id,
            "platform": content.platform.value,
            "text": content.text,
            "url": content.url,
            "content_type": content.content_type.value if content.content_type else None,
            "created_at": content.created_at.isoformat() if content.created_at else None,
            "like_count": content.like_count,
            "comment_count": content.comment_count,
            "share_count": content.share_count,
            "view_count": content.view_count,
            "hashtags": content.hashtags,
            "mentions": content.mentions,
            "media_urls": content.media_urls,
            "thumbnail_url": content.thumbnail_url,
            "location": content.location,
            "metadata": metadata_str,
        })
        return bool(result)

    async def create_interaction(self, interaction: InteractionDTO, brand_id: str) -> bool:
        """Interaction 노드 생성 (metadata 포함)"""
        import json

        query = """
        MERGE (i:Interaction {id: $id, brand_id: $brand_id})
        SET i.platform = $platform,
            i.content_id = $content_id,
            i.text = $text,
            i.created_at = $created_at,
            i.like_count = $like_count,
            i.parent_id = $parent_id,
            i.author_username = $author_username,
            i.metadata = $metadata,
            i.updated_at = datetime()
        RETURN i
        """
        # metadata를 JSON 문자열로 변환
        metadata_str = json.dumps(interaction.metadata, ensure_ascii=False) if interaction.metadata else None

        result = self._sync_repo.query(query, {
            "id": interaction.interaction_id,
            "brand_id": brand_id,
            "platform": interaction.platform.value,
            "content_id": interaction.content_id,
            "text": interaction.text,
            "created_at": interaction.created_at.isoformat() if interaction.created_at else None,
            "like_count": interaction.like_count,
            "parent_id": interaction.parent_id,
            "author_username": interaction.author.username if interaction.author else None,
            "metadata": metadata_str,
        })
        return bool(result)

    async def save_knowledge_graph(
        self,
        graph_data: Dict[str, Any],
        brand_id: str,
    ) -> bool:
        """지식그래프 데이터 저장"""
        logger.info(f"Saving knowledge graph for brand: {brand_id}")

        try:
            # 노드 저장
            for node in graph_data.get("nodes", []):
                node_type = node.get("type", "Unknown")
                properties = node.get("properties", {})
                properties["brand_id"] = brand_id

                query = f"""
                MERGE (n:{node_type} {{id: $id, brand_id: $brand_id}})
                SET n += $properties,
                    n.updated_at = datetime()
                """
                self._sync_repo.query(query, {
                    "id": properties.get("id", ""),
                    "brand_id": brand_id,
                    "properties": properties,
                })

            # 관계 저장
            for rel in graph_data.get("relationships", []):
                rel_type = rel.get("type", "RELATED_TO")
                from_id = rel.get("from", "")
                to_id = rel.get("to", "")

                query = f"""
                MATCH (a {{id: $from_id, brand_id: $brand_id}})
                MATCH (b {{id: $to_id, brand_id: $brand_id}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r.updated_at = datetime()
                """
                self._sync_repo.query(query, {
                    "from_id": from_id,
                    "to_id": to_id,
                    "brand_id": brand_id,
                })

            logger.info("Knowledge graph saved successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to save knowledge graph: {e}")
            return False

    async def close(self) -> None:
        """연결 종료"""
        self._sync_repo.close()
