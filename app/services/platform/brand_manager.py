"""
브랜드 관리 서비스
- 브랜드 CRUD 작업
- 브랜드 검증
- 브랜드 통계
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import lru_cache
from pathlib import Path
import threading
import logging

logger = logging.getLogger(__name__)


# ============================================================
# Enums
# ============================================================

class BrandStatus(str, Enum):
    """브랜드 상태"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    ERROR = "error"


class ValidationLevel(str, Enum):
    """검증 레벨"""
    BASIC = "basic"
    FULL = "full"
    STRICT = "strict"


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class BrandInfo:
    """브랜드 정보"""
    id: str
    name: str
    description: str = ""
    industry: str = ""
    status: BrandStatus = BrandStatus.ACTIVE
    features: List[str] = field(default_factory=list)
    neo4j_brand_id: str = ""
    neo4j_namespaces: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "industry": self.industry,
            "status": self.status.value,
            "features": self.features,
            "neo4j": {
                "brand_id": self.neo4j_brand_id,
                "namespaces": self.neo4j_namespaces,
            },
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_config(cls, config: Dict) -> "BrandInfo":
        """설정에서 생성"""
        brand_section = config.get("brand", {})
        neo4j_section = config.get("neo4j", {})

        return cls(
            id=brand_section.get("id", ""),
            name=brand_section.get("name", ""),
            description=brand_section.get("description", ""),
            industry=brand_section.get("industry", ""),
            features=config.get("features", []),
            neo4j_brand_id=neo4j_section.get("brand_id", ""),
            neo4j_namespaces=neo4j_section.get("namespaces", []),
        )


@dataclass
class ValidationResult:
    """검증 결과"""
    valid: bool
    brand_id: str
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "valid": self.valid,
            "brand_id": self.brand_id,
            "issues": self.issues,
            "warnings": self.warnings,
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class BrandStats:
    """브랜드 통계"""
    brand_id: str
    nodes: Dict[str, int] = field(default_factory=dict)
    total_nodes: int = 0
    relationships: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "brand_id": self.brand_id,
            "nodes": self.nodes,
            "total_nodes": self.total_nodes,
            "relationships": self.relationships,
            "last_updated": self.last_updated.isoformat(),
        }


# ============================================================
# Validators
# ============================================================

class BrandValidator:
    """브랜드 검증기"""

    @staticmethod
    def validate_basic(brand_id: str, config: Dict) -> Tuple[List[str], List[str]]:
        """기본 검증"""
        issues = []
        warnings = []

        # 필수 섹션 확인
        if "brand" not in config:
            issues.append("Missing 'brand' section")
        else:
            if not config["brand"].get("id"):
                issues.append("Missing brand.id")
            if not config["brand"].get("name"):
                issues.append("Missing brand.name")
            if config["brand"].get("id") != brand_id:
                issues.append(f"Brand ID mismatch: config={config['brand'].get('id')}, file={brand_id}")

        if "features" not in config:
            issues.append("Missing 'features' section")
        elif not isinstance(config.get("features"), list):
            issues.append("'features' must be a list")

        if "neo4j" not in config:
            issues.append("Missing 'neo4j' section")
        elif not config["neo4j"].get("brand_id"):
            issues.append("Missing neo4j.brand_id")

        return issues, warnings

    @staticmethod
    def validate_features(config: Dict) -> Tuple[List[str], List[str]]:
        """Feature 검증"""
        issues = []
        warnings = []

        try:
            from app.features.registry import FeatureRegistry
            available_features = FeatureRegistry.list_features()

            for feature in config.get("features", []):
                if feature not in available_features:
                    warnings.append(f"Unknown feature: {feature}")
        except ImportError:
            warnings.append("Could not validate features - FeatureRegistry not available")

        return issues, warnings

    @staticmethod
    def validate_prompts(config: Dict) -> Tuple[List[str], List[str]]:
        """프롬프트 검증"""
        issues = []
        warnings = []

        generation_config = config.get("generation", {})
        for qtype, gen_config in generation_config.items():
            prompt_path = gen_config.get("prompt")
            if prompt_path:
                if not Path(f"prompts/{prompt_path}").exists():
                    issues.append(f"Prompt file not found: {prompt_path}")

        return issues, warnings


# ============================================================
# Main Service
# ============================================================

class BrandManager:
    """프로덕션 급 브랜드 관리자"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return

        self._brands_cache: Dict[str, BrandInfo] = {}
        self._stats_cache: Dict[str, BrandStats] = {}
        self._cache_lock = threading.Lock()

        self._initialized = True
        logger.info("BrandManager initialized")

    # --------------------------------------------------
    # Brand Operations
    # --------------------------------------------------

    def get_brand(self, brand_id: str, use_cache: bool = True) -> BrandInfo:
        """브랜드 정보 조회"""
        # 캐시 확인
        if use_cache and brand_id in self._brands_cache:
            return self._brands_cache[brand_id]

        from app.services.platform.config_manager import ConfigManager
        config = ConfigManager.load_brand_config(brand_id)
        brand_info = BrandInfo.from_config(config)

        # 캐시 저장
        with self._cache_lock:
            self._brands_cache[brand_id] = brand_info

        return brand_info

    def get_brand_dict(self, brand_id: str) -> Dict[str, Any]:
        """브랜드 정보 딕셔너리 조회"""
        return self.get_brand(brand_id).to_dict()

    def list_brands(self, status: BrandStatus = None) -> List[BrandInfo]:
        """브랜드 목록 조회"""
        from app.services.platform.config_manager import ConfigManager
        brand_ids = ConfigManager.list_brands()

        brands = []
        for brand_id in brand_ids:
            try:
                brand_info = self.get_brand(brand_id)
                if status is None or brand_info.status == status:
                    brands.append(brand_info)
            except Exception as e:
                logger.warning(f"Failed to load brand {brand_id}: {e}")

        return brands

    def list_brands_dict(self) -> List[Dict[str, Any]]:
        """브랜드 목록 딕셔너리 조회"""
        return [b.to_dict() for b in self.list_brands()]

    def brand_exists(self, brand_id: str) -> bool:
        """브랜드 존재 여부"""
        from app.services.platform.config_manager import ConfigManager
        return brand_id in ConfigManager.list_brands()

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def validate_brand(self, brand_id: str,
                       level: ValidationLevel = ValidationLevel.FULL) -> ValidationResult:
        """브랜드 검증"""
        try:
            from app.services.platform.config_manager import ConfigManager
            config = ConfigManager.load_brand_config(brand_id)

            all_issues = []
            all_warnings = []

            # 기본 검증 (항상 수행)
            issues, warnings = BrandValidator.validate_basic(brand_id, config)
            all_issues.extend(issues)
            all_warnings.extend(warnings)

            # FULL/STRICT: Feature 검증
            if level in [ValidationLevel.FULL, ValidationLevel.STRICT]:
                issues, warnings = BrandValidator.validate_features(config)
                all_issues.extend(issues)
                all_warnings.extend(warnings)

            # STRICT: 프롬프트 검증
            if level == ValidationLevel.STRICT:
                issues, warnings = BrandValidator.validate_prompts(config)
                all_issues.extend(issues)
                all_warnings.extend(warnings)

            return ValidationResult(
                valid=len(all_issues) == 0,
                brand_id=brand_id,
                issues=all_issues,
                warnings=all_warnings,
            )

        except Exception as e:
            return ValidationResult(
                valid=False,
                brand_id=brand_id,
                issues=[str(e)],
            )

    def validate_all_brands(self, level: ValidationLevel = ValidationLevel.BASIC) -> Dict[str, ValidationResult]:
        """모든 브랜드 검증"""
        from app.services.platform.config_manager import ConfigManager
        brand_ids = ConfigManager.list_brands()

        results = {}
        for brand_id in brand_ids:
            results[brand_id] = self.validate_brand(brand_id, level)

        return results

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    def get_brand_stats(self, brand_id: str, use_cache: bool = True) -> BrandStats:
        """브랜드 통계 조회"""
        # 캐시 확인
        if use_cache and brand_id in self._stats_cache:
            cached = self._stats_cache[brand_id]
            # 5분 이내면 캐시 사용
            if (datetime.now() - cached.last_updated).total_seconds() < 300:
                return cached

        from app.services.shared.neo4j import get_neo4j_client

        neo4j = get_neo4j_client()

        # 노드 통계 조회
        stats_query = """
        MATCH (n)
        WHERE n.brand_id = $brand_id
        WITH labels(n) as label, count(*) as count
        RETURN label[0] as node_type, count
        """
        results = neo4j.query(stats_query, {"brand_id": brand_id})

        nodes = {}
        total_nodes = 0
        for row in results:
            node_type = row.get("node_type", "Unknown")
            count = row.get("count", 0)
            nodes[node_type] = count
            total_nodes += count

        # 관계 수 조회
        rel_query = """
        MATCH (n)-[r]-()
        WHERE n.brand_id = $brand_id
        RETURN count(r) as rel_count
        """
        rel_results = neo4j.query(rel_query, {"brand_id": brand_id})
        rel_count = rel_results[0].get("rel_count", 0) if rel_results else 0

        stats = BrandStats(
            brand_id=brand_id,
            nodes=nodes,
            total_nodes=total_nodes,
            relationships=rel_count // 2,  # 양방향 카운트 보정
        )

        # 캐시 저장
        with self._cache_lock:
            self._stats_cache[brand_id] = stats

        return stats

    def get_brand_stats_dict(self, brand_id: str) -> Dict[str, Any]:
        """브랜드 통계 딕셔너리"""
        return self.get_brand_stats(brand_id).to_dict()

    # --------------------------------------------------
    # Cache Management
    # --------------------------------------------------

    def clear_cache(self, brand_id: str = None):
        """캐시 클리어"""
        with self._cache_lock:
            if brand_id:
                self._brands_cache.pop(brand_id, None)
                self._stats_cache.pop(brand_id, None)
            else:
                self._brands_cache.clear()
                self._stats_cache.clear()

        logger.info(f"Brand cache cleared: {brand_id or 'all'}")

    def reload_brand(self, brand_id: str) -> BrandInfo:
        """브랜드 설정 리로드"""
        from app.services.platform.config_manager import ConfigManager

        self.clear_cache(brand_id)
        ConfigManager.reload_config(brand_id)

        return self.get_brand(brand_id, use_cache=False)

    # --------------------------------------------------
    # Health Check
    # --------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """헬스 체크"""
        from app.services.platform.config_manager import ConfigManager

        brand_ids = ConfigManager.list_brands()
        validation_errors = 0

        for brand_id in brand_ids:
            result = self.validate_brand(brand_id, ValidationLevel.BASIC)
            if not result.valid:
                validation_errors += 1

        return {
            "status": "healthy" if validation_errors == 0 else "degraded",
            "total_brands": len(brand_ids),
            "validation_errors": validation_errors,
            "cached_brands": len(self._brands_cache),
        }


# ============================================================
# Factory Functions
# ============================================================

@lru_cache()
def get_brand_manager() -> BrandManager:
    """싱글톤 브랜드 매니저"""
    return BrandManager()


# ============================================================
# Convenience Functions (Static Methods)
# ============================================================

def get_brand(brand_id: str) -> Dict[str, Any]:
    """브랜드 정보 조회 (편의 함수)"""
    return get_brand_manager().get_brand_dict(brand_id)


def list_brands() -> List[Dict[str, Any]]:
    """브랜드 목록 조회 (편의 함수)"""
    return get_brand_manager().list_brands_dict()


def validate_brand(brand_id: str) -> Dict[str, Any]:
    """브랜드 검증 (편의 함수)"""
    return get_brand_manager().validate_brand(brand_id).to_dict()


def get_brand_stats(brand_id: str) -> Dict[str, Any]:
    """브랜드 통계 (편의 함수)"""
    return get_brand_manager().get_brand_stats_dict(brand_id)
