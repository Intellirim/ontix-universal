"""
ONTIX Security Module
보안 미들웨어, API 키 인증, 감사 로그
"""

import os
import time
import logging
import hashlib
import secrets
from datetime import datetime
from typing import Optional, Callable
from functools import wraps

from fastapi import Request, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)

# 감사 로그 전용 로거
audit_logger = logging.getLogger("audit")
audit_handler = logging.FileHandler("logs/audit.log")
audit_handler.setFormatter(logging.Formatter(
    "[AUDIT] %(asctime)s | %(message)s"
))
audit_logger.addHandler(audit_handler)
audit_logger.setLevel(logging.INFO)


# ============================================
# API Key Authentication
# ============================================

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_admin_api_keys() -> list:
    """환경변수에서 Admin API 키 목록 가져오기"""
    keys_str = os.getenv("ADMIN_API_KEYS", "")
    if not keys_str:
        # 개발 환경: 기본 키 사용 (경고 출력)
        logger.warning("⚠️ ADMIN_API_KEYS not set, using development key")
        return ["dev-admin-key-change-in-production"]
    return [k.strip() for k in keys_str.split(",") if k.strip()]


def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """
    Admin API 키 검증

    Args:
        api_key: X-API-Key 헤더 값

    Returns:
        검증된 API 키

    Raises:
        HTTPException: 키가 없거나 유효하지 않음
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    valid_keys = get_admin_api_keys()

    # 타이밍 공격 방지를 위한 상수 시간 비교
    key_valid = False
    for valid_key in valid_keys:
        if secrets.compare_digest(api_key, valid_key):
            key_valid = True
            break

    if not key_valid:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )

    return api_key


# ============================================
# Security Headers Middleware
# ============================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """보안 헤더 추가 미들웨어"""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)

        # 보안 헤더 추가
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        # HSTS (프로덕션에서만)
        if os.getenv("ENVIRONMENT", "development") == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # CSP (Content Security Policy)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' https://api.openai.com"
        )

        return response


# ============================================
# Rate Limiting
# ============================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    간단한 인메모리 Rate Limiting

    프로덕션에서는 Redis 기반으로 교체 권장
    """

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts: dict = {}  # IP -> [(timestamp, count)]
        self.cleanup_interval = 60  # 초
        self.last_cleanup = time.time()

    def _cleanup_old_entries(self):
        """오래된 항목 정리"""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return

        cutoff = now - 60
        for ip in list(self.request_counts.keys()):
            self.request_counts[ip] = [
                (ts, count) for ts, count in self.request_counts[ip]
                if ts > cutoff
            ]
            if not self.request_counts[ip]:
                del self.request_counts[ip]

        self.last_cleanup = now

    def _get_client_ip(self, request: Request) -> str:
        """클라이언트 IP 추출"""
        # X-Forwarded-For 헤더 확인 (프록시/로드밸런서 뒤에 있을 때)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next) -> Response:
        self._cleanup_old_entries()

        client_ip = self._get_client_ip(request)
        now = time.time()

        # 현재 IP의 요청 수 계산
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []

        # 최근 1분간 요청 수
        recent_requests = sum(
            count for ts, count in self.request_counts[client_ip]
            if ts > now - 60
        )

        if recent_requests >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return Response(
                content='{"error": "Rate limit exceeded", "retry_after": 60}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": "60"}
            )

        # 요청 카운트 증가
        self.request_counts[client_ip].append((now, 1))

        response = await call_next(request)

        # Rate limit 헤더 추가
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.requests_per_minute - recent_requests - 1)
        )

        return response


# ============================================
# Audit Logging
# ============================================

class AuditAction:
    """감사 로그 액션 타입"""
    LOGIN = "login"
    LOGOUT = "logout"
    CONFIG_RELOAD = "config_reload"
    CACHE_CLEAR = "cache_clear"
    DATA_EXPORT = "data_export"
    BRAND_CREATE = "brand_create"
    BRAND_UPDATE = "brand_update"
    BRAND_DELETE = "brand_delete"
    PIPELINE_RUN = "pipeline_run"
    ALERT_ACKNOWLEDGE = "alert_acknowledge"
    API_KEY_USED = "api_key_used"


def log_audit(
    action: str,
    request: Request,
    user_id: str = "anonymous",
    details: Optional[dict] = None
):
    """
    감사 로그 기록

    Args:
        action: 액션 타입
        request: FastAPI Request 객체
        user_id: 사용자 ID
        details: 추가 정보
    """
    client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    if not client_ip:
        client_ip = request.client.host if request.client else "unknown"

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "user_id": user_id,
        "ip": client_ip,
        "path": request.url.path,
        "method": request.method,
        "user_agent": request.headers.get("User-Agent", "unknown")[:100],
    }

    if details:
        log_entry["details"] = details

    # 로그 출력
    log_str = " | ".join(f"{k}={v}" for k, v in log_entry.items() if k != "details")
    if details:
        log_str += f" | details={details}"

    audit_logger.info(log_str)
    logger.info(f"[AUDIT] {log_str}")


def audit_log(action: str, get_details: Optional[Callable] = None):
    """
    감사 로그 데코레이터

    Usage:
        @audit_log(AuditAction.CONFIG_RELOAD)
        async def reload_config(...):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Request 객체 찾기
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            if not request:
                request = kwargs.get("request")

            # 함수 실행
            result = await func(*args, **kwargs)

            # 감사 로그 기록
            if request:
                details = get_details(kwargs, result) if get_details else None
                log_audit(action, request, details=details)

            return result
        return wrapper
    return decorator


# ============================================
# Input Validation Helpers
# ============================================

def sanitize_string(value: str, max_length: int = 1000) -> str:
    """문자열 새니타이즈"""
    if not value:
        return ""
    # 길이 제한
    value = value[:max_length]
    # 제어 문자 제거
    value = "".join(c for c in value if c.isprintable() or c in "\n\t")
    return value.strip()


def validate_brand_id(brand_id: str) -> bool:
    """브랜드 ID 유효성 검사"""
    if not brand_id:
        return False
    if len(brand_id) > 64:
        return False
    # 알파벳, 숫자, 언더스코어, 하이픈만 허용
    import re
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', brand_id))
