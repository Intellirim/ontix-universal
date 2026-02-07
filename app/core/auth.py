"""
Authentication & Authorization (RBAC)
사용자 인증 및 역할 기반 접근 제어
"""

from datetime import datetime, timedelta
from typing import Optional, List, Set
from enum import Enum
from pydantic import BaseModel, Field
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import hashlib
import bcrypt
import secrets
import os
import logging
import json

logger = logging.getLogger(__name__)

# ============================================
# Configuration
# ============================================

JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    JWT_SECRET = secrets.token_hex(32)
    logger.warning("JWT_SECRET not set! Generated random secret (will change on restart). Set JWT_SECRET env var for production.")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "24"))

security = HTTPBearer(auto_error=False)


# ============================================
# Role & Permission Definitions
# ============================================

class Role(str, Enum):
    """사용자 역할"""
    SUPER_ADMIN = "super_admin"     # 전체 관리자 (당신)
    CLIENT_ADMIN = "client_admin"   # 클라이언트 관리자 (브랜드 담당자)
    VIEWER = "viewer"               # 읽기 전용


class Permission(str, Enum):
    """권한 목록"""
    # 브랜드 관련
    BRAND_VIEW_ALL = "brand:view_all"       # 모든 브랜드 조회
    BRAND_VIEW_OWN = "brand:view_own"       # 자기 브랜드만 조회
    BRAND_CREATE = "brand:create"           # 브랜드 생성
    BRAND_UPDATE = "brand:update"           # 브랜드 수정
    BRAND_DELETE = "brand:delete"           # 브랜드 삭제

    # 채팅/세션 관련
    CHAT_VIEW_ALL = "chat:view_all"         # 모든 채팅 조회
    CHAT_VIEW_OWN = "chat:view_own"         # 자기 브랜드 채팅만 조회
    CHAT_EXPORT = "chat:export"             # 채팅 내보내기

    # 분석 관련
    ANALYTICS_VIEW_ALL = "analytics:view_all"
    ANALYTICS_VIEW_OWN = "analytics:view_own"
    ANALYTICS_EXPORT = "analytics:export"

    # 파이프라인 관련
    PIPELINE_VIEW = "pipeline:view"
    PIPELINE_RUN = "pipeline:run"

    # 시스템 관련
    SYSTEM_CONFIG = "system:config"         # 시스템 설정
    SYSTEM_MONITOR = "system:monitor"       # 시스템 모니터링
    USER_MANAGE = "user:manage"             # 사용자 관리

    # 위젯 관련
    WIDGET_CONFIG = "widget:config"         # 위젯 설정


# 역할별 권한 매핑
ROLE_PERMISSIONS: dict[Role, Set[Permission]] = {
    Role.SUPER_ADMIN: set(Permission),  # 모든 권한

    Role.CLIENT_ADMIN: {
        Permission.BRAND_VIEW_OWN,
        Permission.CHAT_VIEW_OWN,
        Permission.CHAT_EXPORT,
        Permission.ANALYTICS_VIEW_OWN,
        Permission.ANALYTICS_EXPORT,
        Permission.WIDGET_CONFIG,
    },

    Role.VIEWER: {
        Permission.BRAND_VIEW_OWN,
        Permission.CHAT_VIEW_OWN,
        Permission.ANALYTICS_VIEW_OWN,
    },
}


# ============================================
# User Model
# ============================================

class User(BaseModel):
    """사용자 모델"""
    id: str
    email: str
    name: str
    role: Role
    brand_ids: List[str] = Field(default_factory=list)  # 접근 가능한 브랜드 목록
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None

    def has_permission(self, permission: Permission) -> bool:
        """권한 확인"""
        return permission in ROLE_PERMISSIONS.get(self.role, set())

    def can_access_brand(self, brand_id: str) -> bool:
        """브랜드 접근 권한 확인"""
        if self.role == Role.SUPER_ADMIN:
            return True
        return brand_id in self.brand_ids

    def get_accessible_brands(self) -> Optional[List[str]]:
        """접근 가능한 브랜드 목록 (None = 전체)"""
        if self.role == Role.SUPER_ADMIN:
            return None  # 전체 접근
        return self.brand_ids


class UserCreate(BaseModel):
    """사용자 생성 요청"""
    email: str
    password: str
    name: str
    role: Role = Role.CLIENT_ADMIN
    brand_ids: List[str] = Field(default_factory=list)


class UserLogin(BaseModel):
    """로그인 요청"""
    email: str
    password: str


class TokenResponse(BaseModel):
    """토큰 응답"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


# ============================================
# User Storage (Simple JSON-based)
# ============================================

USERS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "users.json")


def _ensure_data_dir():
    """데이터 디렉토리 생성"""
    data_dir = os.path.dirname(USERS_FILE)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


def _hash_password(password: str) -> str:
    """비밀번호 해시 (bcrypt)"""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_password(password: str, password_hash: str) -> bool:
    """비밀번호 검증 (bcrypt + SHA-256 레거시 호환)"""
    # bcrypt 해시인 경우 ($2b$ 접두사)
    if password_hash.startswith("$2b$") or password_hash.startswith("$2a$"):
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    # SHA-256 레거시 해시 호환 (마이그레이션 전)
    return password_hash == hashlib.sha256(password.encode()).hexdigest()


def _generate_secure_password(length: int = 12) -> str:
    """보안 비밀번호 생성"""
    import string
    alphabet = string.ascii_letters + string.digits + "!@#$%"
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def _load_users() -> dict:
    """사용자 목록 로드"""
    _ensure_data_dir()
    if not os.path.exists(USERS_FILE):
        # 슈퍼 어드민 (환경변수 또는 랜덤 생성)
        super_admin_password = os.getenv("ONTIX_ADMIN_PASSWORD") or _generate_secure_password(16)

        default_users = {
            "superadmin": {
                "id": "superadmin",
                "email": os.getenv("ONTIX_ADMIN_EMAIL", "admin@localhost"),
                "name": "ONTIX Super Admin",
                "password_hash": _hash_password(super_admin_password),
                "role": Role.SUPER_ADMIN.value,
                "brand_ids": [],
                "is_active": True,
                "created_at": datetime.utcnow().isoformat(),
                "last_login": None
            }
        }

        _save_users(default_users)
        logger.warning("=" * 60)
        logger.warning("DEFAULT ADMIN ACCOUNT CREATED:")
        logger.warning(f"  Email: {default_users['superadmin']['email']}")
        logger.warning(f"  Password: {super_admin_password}")
        logger.warning("  CHANGE THIS PASSWORD IMMEDIATELY!")
        logger.warning("  Set ONTIX_ADMIN_PASSWORD env var to customize.")
        logger.warning("=" * 60)
        return default_users

    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_users(users: dict):
    """사용자 목록 저장"""
    _ensure_data_dir()
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


def get_user_by_email(email: str) -> Optional[User]:
    """이메일로 사용자 조회"""
    users = _load_users()
    for user_data in users.values():
        if user_data.get("email") == email:
            return User(
                id=user_data["id"],
                email=user_data["email"],
                name=user_data["name"],
                role=Role(user_data["role"]),
                brand_ids=user_data.get("brand_ids", []),
                is_active=user_data.get("is_active", True),
                created_at=datetime.fromisoformat(user_data["created_at"]) if user_data.get("created_at") else datetime.utcnow(),
                last_login=datetime.fromisoformat(user_data["last_login"]) if user_data.get("last_login") else None
            )
    return None


def get_user_by_id(user_id: str) -> Optional[User]:
    """ID로 사용자 조회"""
    users = _load_users()
    user_data = users.get(user_id)
    if user_data:
        return User(
            id=user_data["id"],
            email=user_data["email"],
            name=user_data["name"],
            role=Role(user_data["role"]),
            brand_ids=user_data.get("brand_ids", []),
            is_active=user_data.get("is_active", True),
            created_at=datetime.fromisoformat(user_data["created_at"]) if user_data.get("created_at") else datetime.utcnow(),
            last_login=datetime.fromisoformat(user_data["last_login"]) if user_data.get("last_login") else None
        )
    return None


def verify_password(email: str, password: str) -> Optional[User]:
    """비밀번호 검증 (bcrypt + SHA-256 레거시 자동 마이그레이션)"""
    users = _load_users()
    for user_data in users.values():
        if user_data.get("email") == email:
            stored_hash = user_data.get("password_hash", "")
            if _verify_password(password, stored_hash):
                # SHA-256 레거시 해시인 경우 bcrypt로 자동 마이그레이션
                if not stored_hash.startswith("$2b$") and not stored_hash.startswith("$2a$"):
                    user_data["password_hash"] = _hash_password(password)
                    logger.info(f"Migrated password hash to bcrypt for user: {email}")
                # 마지막 로그인 시간 업데이트
                user_data["last_login"] = datetime.utcnow().isoformat()
                _save_users(users)
                return get_user_by_email(email)
    return None


def create_user(user_create: UserCreate) -> User:
    """사용자 생성"""
    users = _load_users()

    # 이메일 중복 체크
    for user_data in users.values():
        if user_data.get("email") == user_create.email:
            raise HTTPException(status_code=400, detail="Email already registered")

    # 새 사용자 생성
    user_id = secrets.token_hex(8)
    users[user_id] = {
        "id": user_id,
        "email": user_create.email,
        "name": user_create.name,
        "password_hash": _hash_password(user_create.password),
        "role": user_create.role.value,
        "brand_ids": user_create.brand_ids,
        "is_active": True,
        "created_at": datetime.utcnow().isoformat(),
        "last_login": None
    }
    _save_users(users)

    logger.info(f"Created user: {user_create.email} with role {user_create.role}")
    return get_user_by_id(user_id)


def list_users() -> List[User]:
    """모든 사용자 목록"""
    users = _load_users()
    return [
        User(
            id=data["id"],
            email=data["email"],
            name=data["name"],
            role=Role(data["role"]),
            brand_ids=data.get("brand_ids", []),
            is_active=data.get("is_active", True),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            last_login=datetime.fromisoformat(data["last_login"]) if data.get("last_login") else None
        )
        for data in users.values()
    ]


def update_user(user_id: str, updates: dict) -> Optional[User]:
    """사용자 정보 수정"""
    users = _load_users()
    if user_id not in users:
        return None

    user_data = users[user_id]

    # 허용된 필드만 업데이트
    allowed_fields = {"name", "role", "brand_ids", "is_active"}
    for key, value in updates.items():
        if key in allowed_fields:
            if key == "role":
                user_data[key] = value if isinstance(value, str) else value.value
            else:
                user_data[key] = value

    _save_users(users)
    return get_user_by_id(user_id)


def delete_user(user_id: str) -> bool:
    """사용자 삭제"""
    users = _load_users()
    if user_id in users:
        del users[user_id]
        _save_users(users)
        return True
    return False


# ============================================
# JWT Token Management
# ============================================

def create_access_token(user: User) -> str:
    """JWT 토큰 생성"""
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS)
    payload = {
        "sub": user.id,
        "email": user.email,
        "role": user.role.value,
        "brand_ids": user.brand_ids,
        "exp": expire,
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """JWT 토큰 디코드"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        return None


# ============================================
# FastAPI Dependencies
# ============================================

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """현재 로그인된 사용자 가져오기"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")

    payload = decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = get_user_by_id(payload.get("sub"))
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="User is deactivated")

    return user


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[User]:
    """선택적 사용자 인증 (없어도 OK)"""
    if not credentials:
        return None

    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def require_permission(permission: Permission):
    """권한 필요 데코레이터"""
    async def permission_checker(user: User = Depends(get_current_user)) -> User:
        if not user.has_permission(permission):
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied: {permission.value} required"
            )
        return user
    return permission_checker


def require_role(role: Role):
    """역할 필요 데코레이터"""
    async def role_checker(user: User = Depends(get_current_user)) -> User:
        if user.role != role and user.role != Role.SUPER_ADMIN:
            raise HTTPException(
                status_code=403,
                detail=f"Role {role.value} required"
            )
        return user
    return role_checker


def require_brand_access(brand_id: str):
    """브랜드 접근 권한 체크"""
    async def brand_checker(user: User = Depends(get_current_user)) -> User:
        if not user.can_access_brand(brand_id):
            raise HTTPException(
                status_code=403,
                detail=f"No access to brand: {brand_id}"
            )
        return user
    return brand_checker


# ============================================
# Utility Functions
# ============================================

def filter_brands_by_user(user: User, brand_ids: List[str]) -> List[str]:
    """사용자 권한에 따라 브랜드 필터링"""
    accessible = user.get_accessible_brands()
    if accessible is None:
        return brand_ids  # 전체 접근 가능
    return [bid for bid in brand_ids if bid in accessible]


def get_user_brand_filter(user: User) -> Optional[List[str]]:
    """사용자의 브랜드 필터 조건 반환 (None = 전체)"""
    return user.get_accessible_brands()
