"""
Onboarding API
브랜드 온보딩 진행 상태 엔드포인트
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/onboarding")


# ============================================================
# Response Models
# ============================================================

class OnboardingStep(BaseModel):
    """온보딩 단계"""
    step: int
    title: str
    description: str
    completed: bool
    required_items: List[str]
    completed_items: List[str]


class BrandOnboardingStatus(BaseModel):
    """브랜드 온보딩 상태"""
    brand_id: str
    brand_name: str
    current_step: int
    total_steps: int
    completed: bool
    progress_percent: int
    steps: List[OnboardingStep]


# ============================================================
# API Endpoints
# ============================================================

@router.get("/{brand_id}", response_model=BrandOnboardingStatus)
async def get_onboarding_status(brand_id: str):
    """
    브랜드 온보딩 상태 조회

    Neo4j 데이터와 설정 파일을 기반으로 온보딩 진행 상태 확인

    Args:
        brand_id: 브랜드 ID

    Returns:
        온보딩 상태
    """
    try:
        from app.services.shared.neo4j import get_neo4j_client
        from app.services.platform.config_manager import ConfigManager

        neo4j = get_neo4j_client()

        # 브랜드 설정 로드
        try:
            brand_config = ConfigManager.load_brand_config(brand_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Brand not found: {brand_id}")

        brand_name = brand_config.get('name', brand_id)

        # 각 단계별 완료 상태 확인
        steps = []

        # Step 1: 브랜드 설정
        step1_required = ['브랜드명', '업종', '설명']
        step1_completed = []
        if brand_config.get('name'):
            step1_completed.append('브랜드명')
        if brand_config.get('industry'):
            step1_completed.append('업종')
        if brand_config.get('description'):
            step1_completed.append('설명')

        steps.append(OnboardingStep(
            step=1,
            title="브랜드 설정",
            description="기본 브랜드 정보 및 아이덴티티 설정",
            completed=len(step1_completed) >= 2,
            required_items=step1_required,
            completed_items=step1_completed
        ))

        # Step 2: 지식 베이스 (Neo4j 데이터 확인)
        step2_required = ['문서', 'FAQ', '상품 카탈로그']
        step2_completed = []

        # 문서 체크
        doc_count = neo4j.query(
            "MATCH (d:Document) WHERE d.brand_id = $brand_id RETURN count(d) as count",
            {'brand_id': brand_id}
        )
        if doc_count and doc_count[0].get('count', 0) > 0:
            step2_completed.append('문서')

        # FAQ 체크
        faq_count = neo4j.query(
            "MATCH (f:FAQ) WHERE f.brand_id = $brand_id RETURN count(f) as count",
            {'brand_id': brand_id}
        )
        if faq_count and faq_count[0].get('count', 0) > 0:
            step2_completed.append('FAQ')

        # 상품 체크
        product_count = neo4j.query(
            "MATCH (p:Product) WHERE p.brand_id = $brand_id RETURN count(p) as count",
            {'brand_id': brand_id}
        )
        if product_count and product_count[0].get('count', 0) > 0:
            step2_completed.append('상품 카탈로그')

        steps.append(OnboardingStep(
            step=2,
            title="지식 베이스",
            description="문서 업로드 및 데이터 소스 설정",
            completed=len(step2_completed) >= 1,
            required_items=step2_required,
            completed_items=step2_completed
        ))

        # Step 3: AI 학습 (설정 확인)
        step3_required = ['톤앤매너', '응답 템플릿', '기능 활성화']
        step3_completed = []

        if brand_config.get('tone_style') or brand_config.get('response_style'):
            step3_completed.append('톤앤매너')
        if brand_config.get('templates') or brand_config.get('prompts'):
            step3_completed.append('응답 템플릿')
        if brand_config.get('features') and len(brand_config.get('features', [])) > 0:
            step3_completed.append('기능 활성화')

        steps.append(OnboardingStep(
            step=3,
            title="AI 학습",
            description="AI 응답 스타일 및 행동 설정",
            completed=len(step3_completed) >= 1,
            required_items=step3_required,
            completed_items=step3_completed
        ))

        # Step 4: 연동 (소셜 데이터 및 채팅 확인)
        step4_required = ['소셜 데이터', '채팅 연동', '분석 활성화']
        step4_completed = []

        # 소셜 데이터 체크
        post_count = neo4j.query(
            "MATCH (p:Post) WHERE p.brand_id = $brand_id RETURN count(p) as count",
            {'brand_id': brand_id}
        )
        if post_count and post_count[0].get('count', 0) > 0:
            step4_completed.append('소셜 데이터')

        # 채팅 세션 체크
        session_count = neo4j.query(
            "MATCH (s:ChatSession) WHERE s.brand_id = $brand_id RETURN count(s) as count",
            {'brand_id': brand_id}
        )
        if session_count and session_count[0].get('count', 0) > 0:
            step4_completed.append('채팅 연동')

        # 분석 활성화 체크
        if 'analytics' in brand_config.get('features', []):
            step4_completed.append('분석 활성화')

        steps.append(OnboardingStep(
            step=4,
            title="연동",
            description="채널 연결 및 기능 활성화",
            completed=len(step4_completed) >= 1,
            required_items=step4_required,
            completed_items=step4_completed
        ))

        # 현재 단계 및 진행률 계산
        completed_steps = sum(1 for s in steps if s.completed)
        current_step = completed_steps + 1 if completed_steps < 4 else 4
        is_completed = completed_steps == 4
        progress = int((completed_steps / 4) * 100)

        return BrandOnboardingStatus(
            brand_id=brand_id,
            brand_name=brand_name,
            current_step=current_step,
            total_steps=4,
            completed=is_completed,
            progress_percent=progress,
            steps=steps
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Onboarding status error for {brand_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=List[Dict[str, Any]])
async def get_all_onboarding_status():
    """
    모든 브랜드의 온보딩 상태 조회

    Returns:
        모든 브랜드의 온보딩 상태 요약
    """
    try:
        from app.services.platform.config_manager import ConfigManager

        brands = ConfigManager.list_brands()
        results = []

        for brand_id in brands:
            try:
                status = await get_onboarding_status(brand_id)
                results.append({
                    'brand_id': status.brand_id,
                    'brand_name': status.brand_name,
                    'current_step': status.current_step,
                    'completed': status.completed,
                    'progress_percent': status.progress_percent
                })
            except Exception as e:
                logger.warning(f"Failed to get onboarding status for {brand_id}: {e}")
                results.append({
                    'brand_id': brand_id,
                    'brand_name': brand_id,
                    'current_step': 1,
                    'completed': False,
                    'progress_percent': 0,
                    'error': str(e)
                })

        return results

    except Exception as e:
        logger.error(f"Get all onboarding status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{brand_id}/complete-step")
async def complete_onboarding_step(
    brand_id: str,
    step: int = Query(..., ge=1, le=4, description="완료할 단계 (1-4)")
):
    """
    온보딩 단계 수동 완료 처리

    Args:
        brand_id: 브랜드 ID
        step: 단계 번호

    Returns:
        업데이트된 온보딩 상태
    """
    try:
        from app.services.platform.config_manager import ConfigManager

        # 브랜드 설정 로드
        try:
            brand_config = ConfigManager.load_brand_config(brand_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Brand not found: {brand_id}")

        # 온보딩 상태 메타데이터 업데이트
        onboarding_meta = brand_config.get('onboarding', {})
        completed_steps = onboarding_meta.get('completed_steps', [])

        if step not in completed_steps:
            completed_steps.append(step)
            completed_steps.sort()

        onboarding_meta['completed_steps'] = completed_steps
        onboarding_meta['last_updated'] = __import__('datetime').datetime.now().isoformat()

        # 설정 업데이트
        ConfigManager.update_brand(brand_id, {'onboarding': onboarding_meta})

        # 업데이트된 상태 반환
        return await get_onboarding_status(brand_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Complete step error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
