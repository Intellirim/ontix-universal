"""
Feature Manager with Schema Validation
- Feature 설정 검증
- 스키마 기반 검증
- Feature 상태 관리
"""

import yaml
import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import lru_cache

try:
    from jsonschema import validate, ValidationError, Draft7Validator
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    ValidationError = Exception

logger = logging.getLogger(__name__)


# ============================================================
# Enums
# ============================================================

class FeatureStatus(str, Enum):
    """Feature 상태"""
    ENABLED = "enabled"
    DISABLED = "disabled"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"


class ValidationSeverity(str, Enum):
    """검증 심각도"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class ValidationIssue:
    """검증 이슈"""
    severity: ValidationSeverity
    message: str
    path: str = ""
    value: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "severity": self.severity.value,
            "message": self.message,
            "path": self.path,
            "value": str(self.value) if self.value is not None else None,
        }


@dataclass
class FeatureValidationResult:
    """Feature 검증 결과"""
    feature_name: str
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    checked_at: datetime = field(default_factory=datetime.now)

    @property
    def errors(self) -> List[ValidationIssue]:
        """에러만 반환"""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """경고만 반환"""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "feature_name": self.feature_name,
            "valid": self.valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [i.to_dict() for i in self.issues],
            "checked_at": self.checked_at.isoformat(),
        }


@dataclass
class FeatureInfo:
    """Feature 정보"""
    name: str
    status: FeatureStatus = FeatureStatus.ENABLED
    description: str = ""
    version: str = "1.0.0"
    dependencies: List[str] = field(default_factory=list)
    schema_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "name": self.name,
            "status": self.status.value,
            "description": self.description,
            "version": self.version,
            "dependencies": self.dependencies,
            "has_schema": self.schema_path is not None,
        }


@dataclass
class FeatureManagerMetrics:
    """Feature Manager 메트릭"""
    total_validations: int = 0
    successful_validations: int = 0
    failed_validations: int = 0
    schema_cache_hits: int = 0
    schema_cache_misses: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    def record_validation(self, success: bool):
        """검증 결과 기록"""
        self.total_validations += 1
        if success:
            self.successful_validations += 1
        else:
            self.failed_validations += 1

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "total_validations": self.total_validations,
            "successful_validations": self.successful_validations,
            "failed_validations": self.failed_validations,
            "success_rate": round(self.successful_validations / self.total_validations, 4) if self.total_validations > 0 else 0.0,
            "schema_cache_hits": self.schema_cache_hits,
            "schema_cache_misses": self.schema_cache_misses,
            "uptime_seconds": (datetime.now() - self.created_at).total_seconds(),
        }


# ============================================================
# Schema Loader
# ============================================================

class SchemaLoader:
    """스키마 로더"""

    _cache: Dict[str, Dict] = {}
    _lock = threading.Lock()

    @classmethod
    def load_schema(cls, feature_name: str, metrics: FeatureManagerMetrics = None) -> Optional[Dict]:
        """Feature 스키마 로드"""
        # 캐시 확인
        if feature_name in cls._cache:
            if metrics:
                metrics.schema_cache_hits += 1
            return cls._cache[feature_name]

        schema_path = Path(f"app/features/{feature_name}/config_schema.yaml")

        if not schema_path.exists():
            return None

        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_data = yaml.safe_load(f)

            schema = schema_data.get("schema")
            if schema:
                with cls._lock:
                    cls._cache[feature_name] = schema
                if metrics:
                    metrics.schema_cache_misses += 1
                return schema

        except Exception as e:
            logger.error(f"Failed to load schema for {feature_name}: {e}")

        return None

    @classmethod
    def clear_cache(cls):
        """캐시 클리어"""
        with cls._lock:
            cls._cache.clear()


# ============================================================
# Main Service
# ============================================================

class FeatureManager:
    """프로덕션 급 Feature 관리자"""

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

        self.metrics = FeatureManagerMetrics()
        self._features_cache: Dict[str, FeatureInfo] = {}

        self._initialized = True
        logger.info(f"FeatureManager initialized (jsonschema available: {HAS_JSONSCHEMA})")

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def validate_feature_config(self, feature_name: str,
                                config: Dict) -> FeatureValidationResult:
        """Feature 설정 검증"""
        issues = []

        # 스키마 로드
        schema = SchemaLoader.load_schema(feature_name, self.metrics)

        if not schema:
            logger.debug(f"No schema found for {feature_name}, skipping validation")
            return FeatureValidationResult(
                feature_name=feature_name,
                valid=True,
                issues=[ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message="No schema defined, validation skipped",
                )],
            )

        if not HAS_JSONSCHEMA:
            logger.warning("jsonschema not available, skipping validation")
            return FeatureValidationResult(
                feature_name=feature_name,
                valid=True,
                issues=[ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message="jsonschema library not installed",
                )],
            )

        # JSON Schema 검증
        try:
            validate(instance=config, schema=schema)
            self.metrics.record_validation(success=True)

            logger.debug(f"Config validation passed: {feature_name}")
            return FeatureValidationResult(
                feature_name=feature_name,
                valid=True,
            )

        except ValidationError as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=e.message,
                path=".".join(str(p) for p in e.absolute_path),
                value=e.instance,
            ))

            self.metrics.record_validation(success=False)
            logger.error(f"Config validation failed: {feature_name} - {e.message}")

            return FeatureValidationResult(
                feature_name=feature_name,
                valid=False,
                issues=issues,
            )

        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Schema validation error: {str(e)}",
            ))

            self.metrics.record_validation(success=False)
            return FeatureValidationResult(
                feature_name=feature_name,
                valid=False,
                issues=issues,
            )

    def validate_all_features(self, brand_config: Dict) -> Dict[str, FeatureValidationResult]:
        """브랜드의 모든 Feature 설정 검증"""
        results = {}

        features = brand_config.get("features", [])
        generation_config = brand_config.get("generation", {})
        retrieval_config = brand_config.get("retrieval", {})

        for feature_name in features:
            # generation 설정에서 feature 설정 추출
            feature_config = {}

            if feature_name in generation_config:
                feature_config.update(generation_config[feature_name])

            if feature_name in retrieval_config:
                feature_config.update(retrieval_config[feature_name])

            results[feature_name] = self.validate_feature_config(feature_name, feature_config)

        return results

    # --------------------------------------------------
    # Feature Discovery
    # --------------------------------------------------

    def discover_features(self) -> List[FeatureInfo]:
        """사용 가능한 Feature 탐색"""
        features = []
        features_dir = Path("app/features")

        if not features_dir.exists():
            return features

        for feature_path in features_dir.iterdir():
            if feature_path.is_dir() and not feature_path.name.startswith("_"):
                feature_info = self._load_feature_info(feature_path.name)
                if feature_info:
                    features.append(feature_info)

        return features

    def _load_feature_info(self, feature_name: str) -> Optional[FeatureInfo]:
        """Feature 정보 로드"""
        if feature_name in self._features_cache:
            return self._features_cache[feature_name]

        feature_dir = Path(f"app/features/{feature_name}")

        if not feature_dir.exists():
            return None

        # config.yaml에서 정보 로드
        config_path = feature_dir / "config.yaml"
        info = FeatureInfo(name=feature_name)

        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}

                info.description = config.get("description", "")
                info.version = config.get("version", "1.0.0")
                info.dependencies = config.get("dependencies", [])

                status_str = config.get("status", "enabled")
                try:
                    info.status = FeatureStatus(status_str)
                except ValueError:
                    info.status = FeatureStatus.ENABLED

            except Exception as e:
                logger.warning(f"Failed to load feature config: {feature_name} - {e}")

        # 스키마 존재 여부
        schema_path = feature_dir / "config_schema.yaml"
        if schema_path.exists():
            info.schema_path = str(schema_path)

        self._features_cache[feature_name] = info
        return info

    def list_features(self) -> List[str]:
        """Feature 이름 목록"""
        return [f.name for f in self.discover_features()]

    def get_feature_info(self, feature_name: str) -> Optional[FeatureInfo]:
        """Feature 정보 조회"""
        return self._load_feature_info(feature_name)

    # --------------------------------------------------
    # Feature Status
    # --------------------------------------------------

    def is_feature_enabled(self, feature_name: str, brand_config: Dict) -> bool:
        """Feature 활성화 여부"""
        enabled_features = brand_config.get("features", [])
        return feature_name in enabled_features

    def get_enabled_features(self, brand_config: Dict) -> List[str]:
        """활성화된 Feature 목록"""
        return brand_config.get("features", [])

    def check_dependencies(self, feature_name: str, brand_config: Dict) -> Tuple[bool, List[str]]:
        """Feature 의존성 확인"""
        info = self.get_feature_info(feature_name)
        if not info:
            return True, []

        missing = []
        enabled = self.get_enabled_features(brand_config)

        for dep in info.dependencies:
            if dep not in enabled:
                missing.append(dep)

        return len(missing) == 0, missing

    # --------------------------------------------------
    # Cache Management
    # --------------------------------------------------

    def clear_cache(self):
        """캐시 클리어"""
        SchemaLoader.clear_cache()
        self._features_cache.clear()
        logger.info("FeatureManager cache cleared")

    # --------------------------------------------------
    # Health & Metrics
    # --------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """헬스 체크"""
        features = self.discover_features()
        enabled_count = sum(1 for f in features if f.status == FeatureStatus.ENABLED)

        return {
            "status": "healthy",
            "total_features": len(features),
            "enabled_features": enabled_count,
            "jsonschema_available": HAS_JSONSCHEMA,
            "cached_schemas": len(SchemaLoader._cache),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """메트릭 조회"""
        return self.metrics.to_dict()

    def reset_metrics(self):
        """메트릭 리셋"""
        self.metrics = FeatureManagerMetrics()


# ============================================================
# Factory Functions
# ============================================================

@lru_cache()
def get_feature_manager() -> FeatureManager:
    """싱글톤 Feature 관리자"""
    return FeatureManager()


# ============================================================
# Convenience Functions
# ============================================================

def validate_feature_config(feature_name: str, config: Dict) -> Tuple[bool, List[str]]:
    """Feature 설정 검증 (편의 함수) - 기존 API 호환"""
    result = get_feature_manager().validate_feature_config(feature_name, config)
    error_messages = [i.message for i in result.errors]
    return result.valid, error_messages


def list_features() -> List[str]:
    """Feature 목록 (편의 함수)"""
    return get_feature_manager().list_features()


def is_feature_enabled(feature_name: str, brand_config: Dict) -> bool:
    """Feature 활성화 여부 (편의 함수)"""
    return get_feature_manager().is_feature_enabled(feature_name, brand_config)
