"""
ONTIX Universal - Main Application
FastAPI 메인 앱
"""

# 환경변수 로드 (가장 먼저 실행)
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
import os

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 보안 모듈 임포트
from app.core.security import (
    SecurityHeadersMiddleware,
    RateLimitMiddleware
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 생명주기 관리
    
    Startup:
    - 서비스 초기화
    - 연결 검증
    
    Shutdown:
    - 리소스 정리
    """
    # ============ STARTUP ============
    logger.info("🚀 Starting ONTIX Universal Platform...")
    
    # Neo4j 연결 확인
    try:
        from app.services.shared.neo4j import get_neo4j_client
        neo4j = get_neo4j_client()
        health = neo4j.health_check()
        
        if health['status'] == 'healthy':
            logger.info(f"✅ Neo4j connected: {health['database']}")
        else:
            logger.warning(f"⚠️ Neo4j unhealthy: {health}")
    except Exception as e:
        logger.error(f"❌ Neo4j connection failed: {e}")
        # Neo4j 없어도 앱은 시작 (개발 모드)
    
    # Cache 연결 확인
    try:
        from app.services.shared.cache import get_cache_client
        cache = get_cache_client()
        
        if cache.available:
            logger.info("✅ Redis cache available")
        else:
            logger.warning("⚠️ Redis cache unavailable (continuing without cache)")
    except Exception as e:
        logger.warning(f"⚠️ Cache initialization failed: {e}")
    
    # LLM 초기화
    try:
        from app.services.shared.llm import get_llm_client
        llm = get_llm_client()
        logger.info(f"✅ LLM initialized: {llm.config.default_model}")
    except Exception as e:
        logger.error(f"❌ LLM initialization failed: {e}")
        raise  # LLM 없으면 앱 시작 불가
    
    # 브랜드 목록 로드
    try:
        from app.services.platform.config_manager import ConfigManager
        brands = ConfigManager.list_brands()
        
        if brands:
            logger.info(f"✅ Loaded {len(brands)} brands: {', '.join(brands)}")
        else:
            logger.warning("⚠️ No brands found")
    except Exception as e:
        logger.warning(f"⚠️ Brand loading failed: {e}")
    
    logger.info("🎉 ONTIX Universal Platform started successfully!")
    
    yield
    
    # ============ SHUTDOWN ============
    logger.info("🛑 Shutting down ONTIX Universal Platform...")
    
    try:
        # Neo4j 연결 종료
        neo4j.close()
        logger.info("✅ Neo4j connection closed")
    except:
        pass
    
    logger.info("👋 ONTIX Universal Platform stopped")


# FastAPI 앱 생성
app = FastAPI(
    title="ONTIX Universal Platform",
    description="완전 범용 멀티 브랜드 AI 플랫폼",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================
# CORS 설정 - 가장 먼저!
# ============================================

ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
# 기본값 추가
if "http://localhost:3000" not in ALLOWED_ORIGINS:
    ALLOWED_ORIGINS.append("http://localhost:3000")
logger.info(f"🌐 CORS Origins: {ALLOWED_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=600,
)

# Rate Limiting 미들웨어
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
app.add_middleware(RateLimitMiddleware, requests_per_minute=RATE_LIMIT)

# 보안 헤더 미들웨어
app.add_middleware(SecurityHeadersMiddleware)


# 요청 로깅 미들웨어
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """요청/응답 로깅"""
    start_time = time.time()
    
    # 요청 로깅
    logger.info(f"📥 {request.method} {request.url.path}")
    
    # 응답 처리
    try:
        response = await call_next(request)
        
        # 응답 시간 계산
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # 응답 로깅
        logger.info(
            f"📤 {request.method} {request.url.path} | "
            f"Status: {response.status_code} | "
            f"Time: {process_time:.3f}s"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Request error: {e}", exc_info=True)
        raise


# 에러 핸들러
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """전역 에러 핸들러"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "path": request.url.path
        }
    )


# ============================================
# API 라우터 등록
# ============================================
from app.api.v1 import chat, brands, features, admin, alerts, pipeline, auth, crawler, social, analytics, products, onboarding, content, advisor, pipeline_control

app.include_router(
    chat.router,
    prefix="/api/v1",
    tags=["Chat"]
)

app.include_router(
    brands.router,
    prefix="/api/v1",
    tags=["Brands"]
)

app.include_router(
    features.router,
    prefix="/api/v1",
    tags=["Features"]
)

# Auth API - JWT 기반 인증 (API Key 불필요)
app.include_router(
    auth.router,
    prefix="/api/v1",
    tags=["Authentication"]
)

# Admin API - JWT 인증으로 변경 (RBAC 적용됨)
app.include_router(
    admin.router,
    prefix="/api/v1",
    tags=["Admin"]
)

app.include_router(
    alerts.router,
    prefix="/api/v1",
    tags=["Alerts"]
)

app.include_router(
    pipeline.router,
    prefix="/api/v1",
    tags=["Pipeline"]
)

app.include_router(
    crawler.router,
    prefix="/api/v1",
    tags=["Web Crawler"]
)

app.include_router(
    social.router,
    prefix="/api/v1",
    tags=["Social Monitoring"]
)

app.include_router(
    analytics.router,
    prefix="/api/v1",
    tags=["Analytics"]
)

app.include_router(
    products.router,
    prefix="/api/v1",
    tags=["Products"]
)

app.include_router(
    onboarding.router,
    prefix="/api/v1",
    tags=["Onboarding"]
)

app.include_router(
    content.router,
    prefix="/api/v1",
    tags=["Content Generation"]
)

app.include_router(
    advisor.router,
    prefix="/api/v1",
    tags=["AI Advisor"]
)

app.include_router(
    pipeline_control.router,
    prefix="/api/v1",
    tags=["Pipeline Control"]
)


# 루트 엔드포인트
@app.get("/", tags=["System"])
async def root():
    """
    루트 엔드포인트
    
    Returns:
        시스템 정보
    """
    return {
        "name": "ONTIX Universal Platform",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


# 헬스체크 (개선됨!)
@app.get("/health", tags=["System"])
async def health():
    """
    상세한 헬스체크
    
    Returns:
        서비스별 상태
    """
    health_status = {
        "status": "healthy",
        "services": {}
    }
    
    # Neo4j 체크
    try:
        from app.services.shared.neo4j import get_neo4j_client
        neo4j = get_neo4j_client()
        neo4j_health = neo4j.health_check()
        health_status["services"]["neo4j"] = neo4j_health
    except Exception as e:
        health_status["services"]["neo4j"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Cache 체크
    try:
        from app.services.shared.cache import get_cache_client
        cache = get_cache_client()
        
        if cache.available:
            health_status["services"]["cache"] = {"status": "healthy"}
        else:
            health_status["services"]["cache"] = {"status": "unavailable"}
    except Exception as e:
        health_status["services"]["cache"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # LLM 체크
    try:
        from app.services.shared.llm import get_llm_client
        llm = get_llm_client()
        health_status["services"]["llm"] = {
            "status": "healthy",
            "model": llm.config.default_model
        }
    except Exception as e:
        health_status["services"]["llm"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    return health_status


# API 문서 리다이렉트
@app.get("/api", tags=["System"])
async def api_docs():
    """API 문서 리다이렉트"""
    return {
        "message": "API Documentation",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }


if __name__ == "__main__":
    import uvicorn
    
    # 환경변수에서 설정 로드
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", 8000))
    reload = os.getenv("APP_RELOAD", "true").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
