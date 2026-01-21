"""
Neo4j Service - Production Grade v2.0
Neo4j 프로덕션 서비스

Features:
    - 연결 풀 관리
    - 자동 재연결
    - 헬스체크
    - 트랜잭션 관리
    - 쿼리 로깅
    - 성능 모니터링
    - 벡터 인덱스 지원
    - 배치 연산
"""

from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, TransientError, Neo4jError
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from functools import lru_cache, wraps
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
from enum import Enum
import threading
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================
# Enums & Types
# ============================================================

class QueryType(str, Enum):
    """쿼리 유형"""
    READ = "read"
    WRITE = "write"
    BATCH = "batch"
    VECTOR = "vector"


class ConnectionState(str, Enum):
    """연결 상태"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class Neo4jConfig:
    """Neo4j 설정"""
    uri: str
    username: str
    password: str
    database: str = "neo4j"
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50
    connection_timeout: float = 30.0
    max_retry_time: float = 30.0
    vector_index: str = "ontix_global_concept_index"
    embedding_model: str = "text-embedding-3-small"

    @classmethod
    def from_settings(cls) -> 'Neo4jConfig':
        """settings에서 로드"""
        from app.core.config import settings
        return cls(
            uri=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD,
            database=settings.NEO4J_DATABASE,
            vector_index=getattr(settings, 'NEO4J_VECTOR_INDEX', 'ontix_global_concept_index'),
            embedding_model=getattr(settings, 'EMBEDDING_MODEL', 'text-embedding-3-small'),
        )


@dataclass
class QueryMetrics:
    """쿼리 메트릭"""
    total_queries: int = 0
    read_queries: int = 0
    write_queries: int = 0
    vector_queries: int = 0
    batch_operations: int = 0
    total_query_time_ms: float = 0.0
    error_count: int = 0
    start_time: float = field(default_factory=time.time)

    # 최근 쿼리 통계
    recent_queries: List[Dict] = field(default_factory=list)

    @property
    def avg_query_time_ms(self) -> float:
        """평균 쿼리 시간"""
        if self.total_queries == 0:
            return 0.0
        return self.total_query_time_ms / self.total_queries

    def record_query(
        self,
        query_type: QueryType,
        duration_ms: float,
        result_count: int = 0,
        success: bool = True
    ):
        """쿼리 기록"""
        self.total_queries += 1
        self.total_query_time_ms += duration_ms

        if query_type == QueryType.READ:
            self.read_queries += 1
        elif query_type == QueryType.WRITE:
            self.write_queries += 1
        elif query_type == QueryType.VECTOR:
            self.vector_queries += 1
        elif query_type == QueryType.BATCH:
            self.batch_operations += 1

        if not success:
            self.error_count += 1

        # 최근 쿼리 기록 (최대 100개)
        self.recent_queries.append({
            'type': query_type.value,
            'duration_ms': duration_ms,
            'result_count': result_count,
            'success': success,
            'timestamp': datetime.now().isoformat(),
        })
        if len(self.recent_queries) > 100:
            self.recent_queries = self.recent_queries[-100:]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_queries': self.total_queries,
            'read_queries': self.read_queries,
            'write_queries': self.write_queries,
            'vector_queries': self.vector_queries,
            'batch_operations': self.batch_operations,
            'avg_query_time_ms': round(self.avg_query_time_ms, 2),
            'error_count': self.error_count,
            'uptime_seconds': round(time.time() - self.start_time, 0),
        }


@dataclass
class QueryResult:
    """쿼리 결과"""
    data: List[Dict]
    count: int
    duration_ms: float
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'data': self.data,
            'count': self.count,
            'duration_ms': self.duration_ms,
            'success': self.success,
            'error': self.error,
        }


# ============================================================
# Decorators
# ============================================================

def retry_on_transient_error(max_retries: int = 3, delay: float = 1.0):
    """트랜잭션 에러 시 재시도 데코레이터"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except TransientError as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Transient error, retrying ({attempt + 1}/{max_retries}): {e}"
                        )
                        time.sleep(delay * (attempt + 1))
                    else:
                        logger.error(f"Max retries reached: {e}")
            raise last_exception
        return wrapper
    return decorator


def log_query(func: Callable):
    """쿼리 로깅 데코레이터"""
    @wraps(func)
    def wrapper(self, cypher: str, *args, **kwargs):
        start = time.time()
        try:
            result = func(self, cypher, *args, **kwargs)
            duration = (time.time() - start) * 1000
            logger.debug(
                f"Query executed in {duration:.1f}ms | "
                f"Results: {len(result) if isinstance(result, list) else 'N/A'}"
            )
            return result
        except Exception as e:
            duration = (time.time() - start) * 1000
            logger.error(f"Query failed in {duration:.1f}ms: {e}")
            raise
    return wrapper


# ============================================================
# Neo4j Client
# ============================================================

class Neo4jClient:
    """
    Neo4j 클라이언트 - Production Grade

    Features:
        - 싱글톤 패턴
        - 연결 풀 관리
        - 자동 재연결
        - 쿼리 로깅
        - 메트릭 추적
        - 헬스체크
        - 벡터 인덱스
        - 배치 연산
    """

    _instance: Optional['Neo4jClient'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'initialized'):
            return

        self.config = Neo4jConfig.from_settings()
        self.metrics = QueryMetrics()
        self.state = ConnectionState.DISCONNECTED

        # 연결 검증
        if not all([self.config.uri, self.config.username, self.config.password]):
            raise ValueError(
                "Neo4j credentials not configured. "
                "Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD"
            )

        # Driver 초기화
        self.driver: Driver = GraphDatabase.driver(
            self.config.uri,
            auth=(self.config.username, self.config.password),
            max_connection_lifetime=self.config.max_connection_lifetime,
            max_connection_pool_size=self.config.max_connection_pool_size,
            connection_timeout=self.config.connection_timeout,
        )

        # LangChain Graph
        self.graph = Neo4jGraph(
            url=self.config.uri,
            username=self.config.username,
            password=self.config.password,
            database=self.config.database
        )

        # 벡터 인덱스
        self.vector_index: Optional[Neo4jVector] = None
        self.vector_available = False

        # 초기화
        self._verify_connection()
        self._init_vector_index()
        self._create_indexes()

        self.initialized = True
        logger.info(f"Neo4j initialized: {self.config.uri}")

    def _verify_connection(self):
        """연결 검증"""
        try:
            with self.driver.session(database=self.config.database) as session:
                result = session.run("RETURN 1 as test")
                result.single()
            self.state = ConnectionState.CONNECTED
            logger.info("Neo4j connection verified")
        except Exception as e:
            self.state = ConnectionState.FAILED
            logger.error(f"Neo4j connection failed: {e}")
            raise

    def _init_vector_index(self):
        """벡터 인덱스 초기화"""
        try:
            self.vector_index = Neo4jVector.from_existing_graph(
                embedding=OpenAIEmbeddings(model=self.config.embedding_model),
                url=self.config.uri,
                username=self.config.username,
                password=self.config.password,
                database=self.config.database,
                index_name=self.config.vector_index,
                node_label="Concept",
                text_node_properties=["id", "description"],
                embedding_node_property="embedding"
            )
            self.vector_available = True
            logger.info(f"Vector index connected: {self.config.vector_index}")

        except Exception as e:
            logger.warning(f"Vector index unavailable: {e}")
            self.vector_available = False

    def _create_indexes(self):
        """필수 인덱스 생성"""
        indexes = [
            # 브랜드별 필터링
            "CREATE INDEX IF NOT EXISTS FOR (p:Post) ON (p.brand_id)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Concept) ON (c.brand_id)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Product) ON (p.brand_id)",

            # 고유성 제약
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Post) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Product) REQUIRE p.id IS UNIQUE",

            # 성능 최적화
            "CREATE INDEX IF NOT EXISTS FOR (p:Post) ON (p.likes)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Post) ON (p.created_at)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Product) ON (p.stock)",
        ]

        for idx_query in indexes:
            try:
                self.execute_write(idx_query, log_query=False)
            except Exception as e:
                logger.debug(f"Index creation: {e}")

    @contextmanager
    def session(self) -> Session:
        """세션 컨텍스트 매니저"""
        session = self.driver.session(database=self.config.database)
        try:
            yield session
        finally:
            session.close()

    @retry_on_transient_error(max_retries=3)
    def query(
        self,
        cypher: str,
        params: Optional[Dict] = None,
        log_query: bool = True
    ) -> List[Dict]:
        """
        Cypher 쿼리 실행 (읽기)

        Args:
            cypher: Cypher 쿼리
            params: 파라미터
            log_query: 쿼리 로깅 여부

        Returns:
            결과 리스트
        """
        start_time = time.time()

        try:
            result = self.graph.query(cypher, params=params or {})

            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_query(
                QueryType.READ,
                duration_ms,
                len(result),
                True
            )

            if log_query:
                logger.debug(
                    f"Query: {duration_ms:.1f}ms | "
                    f"Results: {len(result)} | "
                    f"{cypher[:80]}..."
                )

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_query(QueryType.READ, duration_ms, 0, False)
            logger.error(f"Query error: {e}\nQuery: {cypher}")
            raise

    @retry_on_transient_error(max_retries=3)
    def execute_write(
        self,
        cypher: str,
        params: Optional[Dict] = None,
        log_query: bool = True
    ) -> List[Dict]:
        """쓰기 트랜잭션 실행"""
        start_time = time.time()

        try:
            with self.session() as session:
                result = session.execute_write(
                    lambda tx: tx.run(cypher, params or {}).data()
                )

            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_query(
                QueryType.WRITE,
                duration_ms,
                len(result) if result else 0,
                True
            )

            if log_query:
                logger.debug(f"Write: {duration_ms:.1f}ms")

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_query(QueryType.WRITE, duration_ms, 0, False)
            logger.error(f"Write error: {e}")
            raise

    def batch_write(
        self,
        cypher: str,
        params_list: List[Dict],
        batch_size: int = 1000
    ) -> int:
        """
        배치 쓰기

        Args:
            cypher: Cypher 쿼리 (UNWIND $batch AS item 사용)
            params_list: 파라미터 리스트
            batch_size: 배치 크기

        Returns:
            처리된 레코드 수
        """
        total = len(params_list)
        processed = 0
        start_time = time.time()

        for i in range(0, total, batch_size):
            batch = params_list[i:i + batch_size]

            try:
                self.execute_write(
                    cypher,
                    {"batch": batch},
                    log_query=False
                )
                processed += len(batch)
                logger.info(f"Batch progress: {processed}/{total}")

            except Exception as e:
                logger.error(f"Batch write error at {processed}: {e}")
                raise

        duration_ms = (time.time() - start_time) * 1000
        self.metrics.record_query(QueryType.BATCH, duration_ms, processed, True)
        logger.info(f"Batch completed: {total} records in {duration_ms:.0f}ms")

        return processed

    def vector_search(
        self,
        query: str,
        brand_id: str,
        top_k: int = 10,
        min_score: float = 0.7
    ) -> List[Dict]:
        """
        벡터 검색

        Args:
            query: 검색 쿼리
            brand_id: 브랜드 ID
            top_k: 상위 K개
            min_score: 최소 유사도 점수

        Returns:
            검색 결과 리스트
        """
        start_time = time.time()

        if not self.vector_available:
            return self._keyword_search(query, brand_id, top_k)

        try:
            embedding = self.vector_index.embedding.embed_query(query)

            cypher = """
            CALL db.index.vector.queryNodes(
                $index_name,
                $top_k,
                $embedding
            ) YIELD node, score
            WHERE node.brand_id = $brand_id AND score >= $min_score
            RETURN node.id as id, node.description as description, score
            ORDER BY score DESC
            """

            results = self.query(cypher, {
                "index_name": self.config.vector_index,
                "embedding": embedding,
                "brand_id": brand_id,
                "top_k": top_k * 2,
                "min_score": min_score
            }, log_query=False)

            duration_ms = (time.time() - start_time) * 1000
            self.metrics.record_query(QueryType.VECTOR, duration_ms, len(results), True)

            return results[:top_k]

        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return self._keyword_search(query, brand_id, top_k)

    def _keyword_search(
        self,
        query: str,
        brand_id: str,
        limit: int
    ) -> List[Dict]:
        """키워드 검색 (폴백)"""
        cypher = """
        MATCH (c:Concept)
        WHERE c.brand_id = $brand_id
          AND (c.id CONTAINS $query OR c.description CONTAINS $query)
        RETURN c.id as id, c.description as description, 0.5 as score
        LIMIT $limit
        """
        return self.query(cypher, {
            "brand_id": brand_id,
            "query": query,
            "limit": limit
        })

    def find_by_id(
        self,
        label: str,
        node_id: str,
        brand_id: Optional[str] = None
    ) -> Optional[Dict]:
        """
        ID로 노드 조회

        Args:
            label: 노드 레이블
            node_id: 노드 ID
            brand_id: 브랜드 ID (선택)

        Returns:
            노드 데이터 또는 None
        """
        brand_filter = "AND n.brand_id = $brand_id" if brand_id else ""
        cypher = f"""
        MATCH (n:{label})
        WHERE n.id = $id {brand_filter}
        RETURN n
        LIMIT 1
        """
        params = {"id": node_id}
        if brand_id:
            params["brand_id"] = brand_id

        results = self.query(cypher, params)
        return results[0]['n'] if results else None

    def count_nodes(
        self,
        label: str,
        brand_id: Optional[str] = None
    ) -> int:
        """노드 수 조회"""
        brand_filter = "WHERE n.brand_id = $brand_id" if brand_id else ""
        cypher = f"""
        MATCH (n:{label})
        {brand_filter}
        RETURN count(n) as count
        """
        params = {"brand_id": brand_id} if brand_id else {}
        result = self.query(cypher, params)
        return result[0]['count'] if result else 0

    def get_schema(self) -> Dict[str, Any]:
        """스키마 정보 조회"""
        try:
            # 노드 레이블
            labels = self.query("CALL db.labels()")

            # 관계 타입
            rel_types = self.query("CALL db.relationshipTypes()")

            # 인덱스
            indexes = self.query("SHOW INDEXES")

            return {
                'labels': [r['label'] for r in labels],
                'relationship_types': [r['relationshipType'] for r in rel_types],
                'indexes': len(indexes),
            }
        except Exception as e:
            logger.error(f"Schema query error: {e}")
            return {}

    def health_check(self) -> Dict[str, Any]:
        """헬스체크"""
        try:
            start = time.time()
            with self.driver.session(database=self.config.database) as session:
                session.run("RETURN 1").single()
            latency = (time.time() - start) * 1000

            return {
                'status': 'healthy',
                'state': self.state.value,
                'latency_ms': round(latency, 2),
                'database': self.config.database,
                'vector_available': self.vector_available,
                'metrics': self.metrics.to_dict()
            }

        except Exception as e:
            self.state = ConnectionState.DISCONNECTED
            return {
                'status': 'unhealthy',
                'state': self.state.value,
                'error': str(e),
                'metrics': self.metrics.to_dict()
            }

    def get_metrics(self) -> QueryMetrics:
        """메트릭 조회"""
        return self.metrics

    def reset_metrics(self):
        """메트릭 리셋"""
        self.metrics = QueryMetrics()

    def close(self):
        """연결 종료"""
        if hasattr(self, 'driver'):
            self.driver.close()
            self.state = ConnectionState.DISCONNECTED
            logger.info("Neo4j connection closed")


# ============================================================
# Factory
# ============================================================

@lru_cache()
def get_neo4j_client() -> Neo4jClient:
    """싱글톤 Neo4j 클라이언트"""
    return Neo4jClient()
