"""
전역 설정 관리
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """애플리케이션 전역 설정"""

    # ============================================
    # Application Settings
    # ============================================
    APP_NAME: str = "ONTIX Universal Platform"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ============================================
    # Server Settings
    # ============================================
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    APP_RELOAD: bool = True

    # ============================================
    # Anthropic API (Claude)
    # ============================================
    ANTHROPIC_API_KEY: Optional[str] = None

    # ============================================
    # OpenAI API
    # ============================================
    OPENAI_API_KEY: str
    OPENAI_MODEL_FULL: str = "gpt-4-turbo-preview"
    OPENAI_MODEL_MINI: str = "gpt-3.5-turbo"
    OPENAI_TEMPERATURE: float = 0.7
    OPENAI_MAX_TOKENS: int = 2000

    # ============================================
    # Neo4j Configuration
    # ============================================
    NEO4J_URI: str
    NEO4J_USERNAME: str
    NEO4J_PASSWORD: str
    NEO4J_DATABASE: str = "neo4j"
    NEO4J_VECTOR_INDEX: str = "ontix_global_concept_index"

    # ============================================
    # Redis Cache (Optional)
    # ============================================
    REDIS_HOST: Optional[str] = None
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    CACHE_TTL: int = 3600
    FEATURE_CACHE: bool = True

    # ============================================
    # Vector Embeddings
    # ============================================
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536
    VECTOR_SIMILARITY_THRESHOLD: float = 0.7

    # ============================================
    # Features
    # ============================================
    ENABLE_WEB_SEARCH: bool = False
    ENABLE_ANALYTICS: bool = True
    ENABLE_MONITORING: bool = True

    # ============================================
    # Rate Limiting
    # ============================================
    RATE_LIMIT_ENABLED: bool = False
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60

    # ============================================
    # CORS Settings
    # ============================================
    CORS_ORIGINS: str = "*"
    CORS_ALLOW_CREDENTIALS: bool = True

    # ============================================
    # Paths
    # ============================================
    BRAND_CONFIGS_PATH: str = "configs/brands"
    PLATFORM_CONFIGS_PATH: str = "configs/platform"
    PROMPTS_PATH: str = "prompts"

    # ============================================
    # Apify Data Pipeline
    # ============================================
    APIFY_API_TOKEN: Optional[str] = None
    APIFY_INSTAGRAM_ACTOR: str = "apify/instagram-scraper"
    APIFY_YOUTUBE_ACTOR: str = "streamers/youtube-scraper"
    APIFY_TIKTOK_ACTOR: str = "clockworks/tiktok-scraper"
    APIFY_TWITTER_ACTOR: str = "apidojo/tweet-scraper"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"  # 추가 환경변수 허용


# 싱글톤 인스턴스
settings = Settings()
