"""
Authentication API
인증 관련 엔드포인트
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List, Optional
from app.core.auth import (
    User, UserCreate, UserLogin, TokenResponse,
    Role, Permission,
    get_current_user, require_permission,
    verify_password, create_access_token, create_user,
    list_users, get_user_by_id, update_user, delete_user,
    JWT_EXPIRE_HOURS
)
from app.core.security import log_audit, AuditAction
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ============================================
# Public Endpoints (No Auth Required)
# ============================================

@router.post("/login", response_model=TokenResponse)
async def login(credentials: UserLogin, request: Request):
    """
    로그인

    Returns:
        JWT 토큰 및 사용자 정보
    """
    user = verify_password(credentials.email, credentials.password)
    if not user:
        logger.warning(f"Login failed for: {credentials.email}")
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated")

    token = create_access_token(user)

    # 감사 로그
    log_audit(AuditAction.LOGIN, request, user_id=user.id)

    logger.info(f"User logged in: {user.email}")

    return TokenResponse(
        access_token=token,
        expires_in=JWT_EXPIRE_HOURS * 3600,
        user={
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role": user.role.value,
            "brand_ids": user.brand_ids
        }
    )


@router.post("/logout")
async def logout(request: Request, user: User = Depends(get_current_user)):
    """
    로그아웃 (클라이언트에서 토큰 삭제)
    """
    log_audit(AuditAction.LOGOUT, request, user_id=user.id)
    return {"message": "Logged out successfully"}


# ============================================
# User Profile Endpoints
# ============================================

@router.get("/me")
async def get_my_profile(user: User = Depends(get_current_user)):
    """
    현재 사용자 정보 조회
    """
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "role": user.role.value,
        "brand_ids": user.brand_ids,
        "permissions": [p.value for p in Permission if user.has_permission(p)],
        "is_active": user.is_active,
        "created_at": user.created_at.isoformat(),
        "last_login": user.last_login.isoformat() if user.last_login else None
    }


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


@router.post("/me/password")
async def change_password(
    data: PasswordChange,
    request: Request,
    user: User = Depends(get_current_user)
):
    """
    비밀번호 변경
    """
    # 현재 비밀번호 확인
    if not verify_password(user.email, data.current_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    # 새 비밀번호로 업데이트
    from app.core.auth import _load_users, _save_users, _hash_password
    users = _load_users()
    if user.id in users:
        users[user.id]["password_hash"] = _hash_password(data.new_password)
        _save_users(users)

    return {"message": "Password changed successfully"}


class ProfileUpdate(BaseModel):
    email: Optional[str] = None
    name: Optional[str] = None


@router.patch("/me")
async def update_profile(
    data: ProfileUpdate,
    request: Request,
    user: User = Depends(get_current_user)
):
    """
    프로필 수정 (아이디, 이름)
    """
    from app.core.auth import _load_users, _save_users

    users = _load_users()
    if user.id not in users:
        raise HTTPException(status_code=404, detail="User not found")

    # 아이디(이메일) 중복 확인
    if data.email and data.email != user.email:
        for uid, u in users.items():
            if uid != user.id and u.get("email") == data.email:
                raise HTTPException(status_code=400, detail="이미 사용중인 아이디입니다")
        users[user.id]["email"] = data.email

    if data.name:
        users[user.id]["name"] = data.name

    _save_users(users)

    return {
        "id": user.id,
        "email": users[user.id]["email"],
        "name": users[user.id]["name"],
        "role": user.role.value,
        "brand_ids": user.brand_ids
    }


# ============================================
# User Management (Super Admin Only)
# ============================================

@router.get("/users", response_model=List[dict])
async def get_users(user: User = Depends(require_permission(Permission.USER_MANAGE))):
    """
    사용자 목록 조회 (Super Admin 전용)
    """
    users = list_users()
    return [
        {
            "id": u.id,
            "email": u.email,
            "name": u.name,
            "role": u.role.value,
            "brand_ids": u.brand_ids,
            "is_active": u.is_active,
            "created_at": u.created_at.isoformat(),
            "last_login": u.last_login.isoformat() if u.last_login else None
        }
        for u in users
    ]


@router.post("/users", response_model=dict)
async def create_new_user(
    user_data: UserCreate,
    request: Request,
    current_user: User = Depends(require_permission(Permission.USER_MANAGE))
):
    """
    새 사용자 생성 (Super Admin 전용)
    """
    # Super Admin은 생성 불가 (보안)
    if user_data.role == Role.SUPER_ADMIN:
        raise HTTPException(status_code=400, detail="Cannot create super admin")

    new_user = create_user(user_data)

    log_audit(
        AuditAction.BRAND_CREATE,  # 적절한 액션으로 변경 필요
        request,
        user_id=current_user.id,
        details={"created_user": new_user.email, "role": new_user.role.value}
    )

    return {
        "id": new_user.id,
        "email": new_user.email,
        "name": new_user.name,
        "role": new_user.role.value,
        "brand_ids": new_user.brand_ids
    }


@router.get("/users/{user_id}")
async def get_user(
    user_id: str,
    current_user: User = Depends(require_permission(Permission.USER_MANAGE))
):
    """
    특정 사용자 조회 (Super Admin 전용)
    """
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "role": user.role.value,
        "brand_ids": user.brand_ids,
        "is_active": user.is_active,
        "created_at": user.created_at.isoformat(),
        "last_login": user.last_login.isoformat() if user.last_login else None
    }


class UserUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[Role] = None
    brand_ids: Optional[List[str]] = None
    is_active: Optional[bool] = None


@router.patch("/users/{user_id}")
async def update_user_info(
    user_id: str,
    updates: UserUpdate,
    request: Request,
    current_user: User = Depends(require_permission(Permission.USER_MANAGE))
):
    """
    사용자 정보 수정 (Super Admin 전용)
    """
    # Super Admin 역할 변경 방지
    if updates.role == Role.SUPER_ADMIN:
        raise HTTPException(status_code=400, detail="Cannot assign super admin role")

    update_dict = {k: v for k, v in updates.model_dump().items() if v is not None}
    updated_user = update_user(user_id, update_dict)

    if not updated_user:
        raise HTTPException(status_code=404, detail="User not found")

    log_audit(
        AuditAction.BRAND_UPDATE,
        request,
        user_id=current_user.id,
        details={"updated_user": user_id, "changes": update_dict}
    )

    return {
        "id": updated_user.id,
        "email": updated_user.email,
        "name": updated_user.name,
        "role": updated_user.role.value,
        "brand_ids": updated_user.brand_ids,
        "is_active": updated_user.is_active
    }


@router.delete("/users/{user_id}")
async def delete_user_by_id(
    user_id: str,
    request: Request,
    current_user: User = Depends(require_permission(Permission.USER_MANAGE))
):
    """
    사용자 삭제 (Super Admin 전용)
    """
    # 자기 자신 삭제 방지
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    # Super Admin 삭제 방지
    user = get_user_by_id(user_id)
    if user and user.role == Role.SUPER_ADMIN:
        raise HTTPException(status_code=400, detail="Cannot delete super admin")

    if not delete_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")

    log_audit(
        AuditAction.BRAND_DELETE,
        request,
        user_id=current_user.id,
        details={"deleted_user": user_id}
    )

    return {"message": "User deleted successfully"}


# ============================================
# Role & Permission Info
# ============================================

@router.get("/roles")
async def get_roles(user: User = Depends(get_current_user)):
    """
    사용 가능한 역할 목록
    """
    from app.core.auth import ROLE_PERMISSIONS

    return {
        "roles": [
            {
                "value": role.value,
                "name": {
                    Role.SUPER_ADMIN: "슈퍼 어드민",
                    Role.CLIENT_ADMIN: "클라이언트 어드민",
                    Role.VIEWER: "뷰어 (읽기 전용)"
                }.get(role, role.value),
                "permissions": [p.value for p in ROLE_PERMISSIONS.get(role, [])]
            }
            for role in Role
            if role != Role.SUPER_ADMIN or user.role == Role.SUPER_ADMIN
        ]
    }
