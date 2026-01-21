### **app/api/v1/__init__.py**
"""
API v1
"""
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1")

# Sub-routers
from app.api.v1 import chat, brands, features, admin, alerts, pipeline

router.include_router(chat.router, tags=["chat"])
router.include_router(brands.router, tags=["brands"])
router.include_router(features.router, tags=["features"])
router.include_router(admin.router, tags=["admin"])
router.include_router(alerts.router, tags=["alerts"])
router.include_router(pipeline.router, tags=["pipeline"])
