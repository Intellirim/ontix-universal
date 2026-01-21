"""
설정 관리 서비스
- YAML 설정 로딩 및 캐싱
- 환경변수 치환
- 설정 검증
"""

import yaml
import re
import os
import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import lru_cache

logger = logging.getLogger(__name__)


# ============================================================
# Enums
# ============================================================

class ConfigType(str, Enum):
    """설정 타입"""
    BRAND = "brand"
    PLATFORM = "platform"
    FEATURE = "feature"


class ValidationStatus(str, Enum):
    """검증 상태"""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class ConfigEntry:
    """캐시된 설정 엔트리"""
    config: Dict[str, Any]
    config_type: ConfigType
    loaded_at: datetime = field(default_factory=datetime.now)
    file_path: str = ""
    file_mtime: float = 0.0

    def is_stale(self) -> bool:
        """파일 변경 여부 확인"""
        if not self.file_path:
            return False

        try:
            current_mtime = Path(self.file_path).stat().st_mtime
            return current_mtime > self.file_mtime
        except:
            return True


@dataclass
class ConfigManagerMetrics:
    """설정 관리자 메트릭"""
    total_loads: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    validation_errors: int = 0
    env_substitutions: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    def record_load(self, from_cache: bool):
        """로드 기록"""
        self.total_loads += 1
        if from_cache:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def record_error(self):
        """에러 기록"""
        self.validation_errors += 1

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        total = self.cache_hits + self.cache_misses
        return {
            "total_loads": self.total_loads,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hits / total, 4) if total > 0 else 0.0,
            "validation_errors": self.validation_errors,
            "env_substitutions": self.env_substitutions,
            "uptime_seconds": (datetime.now() - self.created_at).total_seconds(),
        }


# ============================================================
# Environment Variable Processor
# ============================================================

class EnvVarProcessor:
    """환경변수 처리기"""

    ENV_PATTERN = re.compile(r'\$\{([^}:]+)(?::([^}]*))?\}')

    @classmethod
    def substitute(cls, obj: Any, metrics: ConfigManagerMetrics = None) -> Any:
        """환경변수 치환"""
        if isinstance(obj, dict):
            return {k: cls.substitute(v, metrics) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [cls.substitute(item, metrics) for item in obj]
        elif isinstance(obj, str):
            return cls._substitute_string(obj, metrics)
        else:
            return obj

    @classmethod
    def _substitute_string(cls, value: str, metrics: ConfigManagerMetrics = None) -> str:
        """문자열 내 환경변수 치환"""
        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2)

            env_value = os.getenv(var_name)
            if env_value is not None:
                if metrics:
                    metrics.env_substitutions += 1
                return env_value
            elif default_value is not None:
                return default_value
            else:
                logger.warning(f"Environment variable not found: {var_name}")
                return match.group(0)

        return cls.ENV_PATTERN.sub(replacer, value)

    @classmethod
    def extract_vars(cls, obj: Any) -> Set[str]:
        """사용된 환경변수 목록 추출"""
        vars_found = set()

        if isinstance(obj, dict):
            for v in obj.values():
                vars_found.update(cls.extract_vars(v))
        elif isinstance(obj, list):
            for item in obj:
                vars_found.update(cls.extract_vars(item))
        elif isinstance(obj, str):
            matches = cls.ENV_PATTERN.findall(obj)
            for match in matches:
                vars_found.add(match[0])

        return vars_found


# ============================================================
# Config Validator
# ============================================================

class ConfigValidator:
    """설정 검증기"""

    BRAND_REQUIRED = {
        "brand": ["id", "name"],
        "features": None,  # list여야 함
        "neo4j": ["brand_id"],
    }

    @classmethod
    def validate_brand_config(cls, brand_id: str, config: Dict) -> List[str]:
        """브랜드 설정 검증"""
        errors = []

        # 필수 섹션/필드 검증
        for section, fields in cls.BRAND_REQUIRED.items():
            if section not in config:
                errors.append(f"Missing required section: {section}")
                continue

            if fields is not None:
                for field in fields:
                    if field not in config[section]:
                        errors.append(f"Missing required field: {section}.{field}")

        # features 리스트 타입 검증
        if "features" in config and not isinstance(config["features"], list):
            errors.append("'features' must be a list")

        # brand.id 일치 확인
        if "brand" in config:
            config_brand_id = config["brand"].get("id")
            if config_brand_id and config_brand_id != brand_id:
                errors.append(f"Brand ID mismatch: config={config_brand_id}, file={brand_id}")

        return errors


# ============================================================
# Main Service
# ============================================================

class ConfigManager:
    """프로덕션 급 설정 관리자"""

    _instance = None
    _lock = threading.Lock()
    _cache: Dict[str, ConfigEntry] = {}
    _metrics: ConfigManagerMetrics = ConfigManagerMetrics()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True

    # --------------------------------------------------
    # Brand Config
    # --------------------------------------------------

    @classmethod
    def load_brand_config(cls, brand_id: str, use_cache: bool = True) -> Dict:
        """브랜드 설정 로드"""
        cache_key = f"brand:{brand_id}"

        # 캐시 확인
        if use_cache and cache_key in cls._cache:
            entry = cls._cache[cache_key]
            if not entry.is_stale():
                cls._metrics.record_load(from_cache=True)
                return entry.config

        # 파일 경로
        config_path = Path(f"configs/brands/{brand_id}.yaml")

        if not config_path.exists():
            raise FileNotFoundError(
                f"Brand config not found: {brand_id}\n"
                f"Expected path: {config_path}\n"
                f"Available brands: {cls.list_brands()}"
            )

        # YAML 로드
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            cls._metrics.record_error()
            raise ValueError(f"Failed to parse YAML: {e}")

        # 환경변수 치환
        config = EnvVarProcessor.substitute(config, cls._metrics)

        # 검증
        errors = ConfigValidator.validate_brand_config(brand_id, config)
        if errors:
            cls._metrics.record_error()
            raise ValueError(
                f"Config validation failed for {brand_id}:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

        # 캐시 저장
        cls._cache[cache_key] = ConfigEntry(
            config=config,
            config_type=ConfigType.BRAND,
            file_path=str(config_path),
            file_mtime=config_path.stat().st_mtime,
        )

        cls._metrics.record_load(from_cache=False)
        logger.info(f"Loaded config for brand: {brand_id}")

        return config

    # --------------------------------------------------
    # Platform Config
    # --------------------------------------------------

    @classmethod
    def load_platform_config(cls, config_name: str, use_cache: bool = True) -> Dict:
        """플랫폼 설정 로드"""
        cache_key = f"platform:{config_name}"

        # 캐시 확인
        if use_cache and cache_key in cls._cache:
            entry = cls._cache[cache_key]
            if not entry.is_stale():
                cls._metrics.record_load(from_cache=True)
                return entry.config

        # 파일 경로
        config_path = Path(f"configs/platform/{config_name}.yaml")

        if not config_path.exists():
            raise FileNotFoundError(f"Platform config not found: {config_name}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # 환경변수 치환
            config = EnvVarProcessor.substitute(config, cls._metrics)

            # 캐시 저장
            cls._cache[cache_key] = ConfigEntry(
                config=config,
                config_type=ConfigType.PLATFORM,
                file_path=str(config_path),
                file_mtime=config_path.stat().st_mtime,
            )

            cls._metrics.record_load(from_cache=False)
            logger.info(f"Loaded platform config: {config_name}")

            return config

        except Exception as e:
            cls._metrics.record_error()
            raise ValueError(f"Failed to load platform config: {e}")

    # --------------------------------------------------
    # Feature Config
    # --------------------------------------------------

    @classmethod
    def load_feature_config(cls, feature_name: str) -> Optional[Dict]:
        """Feature 설정 로드"""
        cache_key = f"feature:{feature_name}"

        if cache_key in cls._cache:
            cls._metrics.record_load(from_cache=True)
            return cls._cache[cache_key].config

        config_path = Path(f"app/features/{feature_name}/config.yaml")

        if not config_path.exists():
            return None

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            config = EnvVarProcessor.substitute(config, cls._metrics)

            cls._cache[cache_key] = ConfigEntry(
                config=config,
                config_type=ConfigType.FEATURE,
                file_path=str(config_path),
                file_mtime=config_path.stat().st_mtime,
            )

            cls._metrics.record_load(from_cache=False)
            return config

        except Exception as e:
            logger.error(f"Failed to load feature config {feature_name}: {e}")
            return None

    # --------------------------------------------------
    # Brand Management
    # --------------------------------------------------

    @classmethod
    def list_brands(cls) -> List[str]:
        """등록된 브랜드 목록"""
        brands_dir = Path("configs/brands")

        if not brands_dir.exists():
            return []

        brands = [
            f.stem for f in brands_dir.glob("*.yaml")
            if not f.stem.startswith("_")  # _template.yaml 제외
        ]

        return sorted(brands)

    @classmethod
    def brand_exists(cls, brand_id: str) -> bool:
        """브랜드 존재 여부"""
        return brand_id in cls.list_brands()

    @classmethod
    def get_brand_config_path(cls, brand_id: str) -> Path:
        """브랜드 설정 파일 경로"""
        return Path(f"configs/brands/{brand_id}.yaml")

    @classmethod
    def create_brand(cls, brand_id: str, brand_data: dict) -> dict:
        """
        새 브랜드 설정 파일 생성

        Args:
            brand_id: 브랜드 ID
            brand_data: 브랜드 데이터 (name, description, industry, features, neo4j_brand_id, neo4j_namespaces)

        Returns:
            생성된 설정
        """
        config_path = cls.get_brand_config_path(brand_id)

        # 이미 존재하는지 확인
        if config_path.exists():
            raise ValueError(f"Brand already exists: {brand_id}")

        # 브랜드 설정 구조 생성
        config = {
            "brand": {
                "id": brand_id,
                "name": brand_data.get("name", brand_id),
                "description": brand_data.get("description", ""),
                "industry": brand_data.get("industry", ""),
            },
            "features": brand_data.get("features", []),
            "neo4j": {
                "brand_id": brand_data.get("neo4j_brand_id") or brand_id,
                "namespaces": brand_data.get("neo4j_namespaces") or [brand_id],
            },
            "retrieval": {
                "default_top_k": 5,
                "sources": ["neo4j_products", "vector"],
            },
            "generation": {
                "default": {
                    "model": "gpt-5-mini",
                    "temperature": 0.7,  # GPT-5에서는 무시됨
                    "max_tokens": 1000,
                },
            },
        }

        # 디렉토리 확인
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # YAML 파일 저장
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        logger.info(f"Created brand config: {brand_id}")
        return config

    @classmethod
    def update_brand(cls, brand_id: str, updates: dict) -> dict:
        """
        브랜드 설정 업데이트

        Args:
            brand_id: 브랜드 ID
            updates: 업데이트할 필드 (name, description, industry, features, neo4j_brand_id, neo4j_namespaces)

        Returns:
            업데이트된 설정
        """
        config_path = cls.get_brand_config_path(brand_id)

        if not config_path.exists():
            raise FileNotFoundError(f"Brand not found: {brand_id}")

        # 현재 설정 로드
        config = cls.load_brand_config(brand_id, use_cache=False)

        # 업데이트 적용
        if "name" in updates and updates["name"]:
            config["brand"]["name"] = updates["name"]
        if "description" in updates and updates["description"] is not None:
            config["brand"]["description"] = updates["description"]
        if "industry" in updates and updates["industry"] is not None:
            config["brand"]["industry"] = updates["industry"]
        if "features" in updates and updates["features"] is not None:
            config["features"] = updates["features"]
        if "neo4j_brand_id" in updates and updates["neo4j_brand_id"]:
            config["neo4j"]["brand_id"] = updates["neo4j_brand_id"]
        if "neo4j_namespaces" in updates and updates["neo4j_namespaces"] is not None:
            config["neo4j"]["namespaces"] = updates["neo4j_namespaces"]

        # YAML 파일 저장
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        # 캐시 무효화
        cache_key = f"brand:{brand_id}"
        if cache_key in cls._cache:
            del cls._cache[cache_key]

        logger.info(f"Updated brand config: {brand_id}")
        return config

    @classmethod
    def delete_brand(cls, brand_id: str) -> bool:
        """
        브랜드 설정 삭제

        Args:
            brand_id: 브랜드 ID

        Returns:
            삭제 성공 여부
        """
        config_path = cls.get_brand_config_path(brand_id)

        if not config_path.exists():
            raise FileNotFoundError(f"Brand not found: {brand_id}")

        # 파일 삭제
        config_path.unlink()

        # 캐시 무효화
        cache_key = f"brand:{brand_id}"
        if cache_key in cls._cache:
            del cls._cache[cache_key]

        logger.info(f"Deleted brand config: {brand_id}")
        return True

    # --------------------------------------------------
    # Cache Management
    # --------------------------------------------------

    @classmethod
    def reload_config(cls, brand_id: str) -> Dict:
        """캐시된 설정 리로드"""
        cache_key = f"brand:{brand_id}"
        if cache_key in cls._cache:
            del cls._cache[cache_key]

        return cls.load_brand_config(brand_id)

    @classmethod
    def clear_cache(cls, config_type: ConfigType = None):
        """캐시 클리어"""
        if config_type:
            prefix = f"{config_type.value}:"
            keys_to_remove = [k for k in cls._cache.keys() if k.startswith(prefix)]
            for key in keys_to_remove:
                del cls._cache[key]
            logger.info(f"Config cache cleared: {config_type.value}")
        else:
            cls._cache.clear()
            logger.info("Config cache cleared: all")

    @classmethod
    def get_cache_info(cls) -> Dict[str, Any]:
        """캐시 정보"""
        by_type = {}
        for key in cls._cache.keys():
            config_type = key.split(":")[0]
            by_type[config_type] = by_type.get(config_type, 0) + 1

        return {
            "total_entries": len(cls._cache),
            "by_type": by_type,
            "entries": [
                {
                    "key": k,
                    "type": v.config_type.value,
                    "loaded_at": v.loaded_at.isoformat(),
                    "stale": v.is_stale(),
                }
                for k, v in cls._cache.items()
            ],
        }

    # --------------------------------------------------
    # Utility Methods
    # --------------------------------------------------

    @classmethod
    def get_env_vars_used(cls, brand_id: str) -> Set[str]:
        """브랜드 설정에서 사용된 환경변수 목록"""
        config_path = Path(f"configs/brands/{brand_id}.yaml")

        if not config_path.exists():
            return set()

        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)

        return EnvVarProcessor.extract_vars(raw_config)

    @classmethod
    def validate_env_vars(cls, brand_id: str) -> Dict[str, bool]:
        """환경변수 설정 여부 확인"""
        env_vars = cls.get_env_vars_used(brand_id)

        return {
            var: os.getenv(var) is not None
            for var in env_vars
        }

    # --------------------------------------------------
    # Health & Metrics
    # --------------------------------------------------

    @classmethod
    def health_check(cls) -> Dict[str, Any]:
        """헬스 체크"""
        stale_count = sum(1 for e in cls._cache.values() if e.is_stale())

        return {
            "status": "healthy" if stale_count == 0 else "degraded",
            "cache_size": len(cls._cache),
            "stale_entries": stale_count,
            "available_brands": len(cls.list_brands()),
        }

    @classmethod
    def get_metrics(cls) -> Dict[str, Any]:
        """메트릭 조회"""
        return cls._metrics.to_dict()

    @classmethod
    def reset_metrics(cls):
        """메트릭 리셋"""
        cls._metrics = ConfigManagerMetrics()


# ============================================================
# Factory Functions
# ============================================================

@lru_cache()
def get_config_manager() -> ConfigManager:
    """싱글톤 설정 관리자"""
    return ConfigManager()


# ============================================================
# Convenience Functions
# ============================================================

def load_brand_config(brand_id: str) -> Dict:
    """브랜드 설정 로드 (편의 함수)"""
    return ConfigManager.load_brand_config(brand_id)


def load_platform_config(config_name: str) -> Dict:
    """플랫폼 설정 로드 (편의 함수)"""
    return ConfigManager.load_platform_config(config_name)


def list_brands() -> List[str]:
    """브랜드 목록 (편의 함수)"""
    return ConfigManager.list_brands()
