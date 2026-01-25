"""
Pipeline EC2 Control API
파이프라인 서버 EC2 인스턴스 제어 엔드포인트
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import httpx
import asyncio
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pipeline-control", tags=["Pipeline Control"])

# Lambda Function URL for EC2 control
LAMBDA_URL = "https://yrzihpjg4du6pxwgdqa2vgcieq0mdxnh.lambda-url.ap-northeast-2.on.aws/"

# Pipeline server health check endpoint
PIPELINE_HEALTH_URL = "http://{ip}:8000/health"


class EC2StatusResponse(BaseModel):
    """EC2 상태 응답"""
    instance_id: str
    state: str
    public_ip: Optional[str] = None
    pipeline_url: Optional[str] = None
    ready: bool = False


class EC2ActionResponse(BaseModel):
    """EC2 액션 응답"""
    message: str
    instance_id: str
    state: str
    public_ip: Optional[str] = None


# ============================================
# EC2 Control Endpoints
# ============================================

@router.get("/status", response_model=EC2StatusResponse)
async def get_pipeline_status():
    """
    파이프라인 서버 상태 조회

    Returns:
        EC2 인스턴스 상태 및 파이프라인 서버 준비 여부
    """
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                LAMBDA_URL,
                json={"action": "status"}
            )
            response.raise_for_status()
            data = response.json()

        result = EC2StatusResponse(
            instance_id=data["instance_id"],
            state=data["state"],
            public_ip=data.get("public_ip"),
            ready=False
        )

        # Check if pipeline server is ready
        if data["state"] == "running" and data.get("public_ip"):
            result.pipeline_url = f"http://{data['public_ip']}:8000"
            result.ready = await _check_pipeline_health(data["public_ip"])

        return result

    except httpx.HTTPError as e:
        logger.error(f"Lambda call failed: {e}")
        raise HTTPException(status_code=503, detail="Failed to get EC2 status")
    except Exception as e:
        logger.error(f"Pipeline status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start", response_model=EC2ActionResponse)
async def start_pipeline_server():
    """
    파이프라인 서버 시작

    EC2 인스턴스를 시작하고 running 상태가 될 때까지 대기합니다.
    """
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                LAMBDA_URL,
                json={"action": "start"}
            )
            response.raise_for_status()
            data = response.json()

        # Parse the nested body if present
        if "body" in data:
            import json
            data = json.loads(data["body"])

        return EC2ActionResponse(
            message=data.get("message", "Instance starting"),
            instance_id=data["instance_id"],
            state=data["state"],
            public_ip=data.get("public_ip")
        )

    except httpx.HTTPError as e:
        logger.error(f"Lambda start call failed: {e}")
        raise HTTPException(status_code=503, detail="Failed to start EC2")
    except Exception as e:
        logger.error(f"Pipeline start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop", response_model=EC2ActionResponse)
async def stop_pipeline_server():
    """
    파이프라인 서버 종료

    EC2 인스턴스를 종료합니다.
    """
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                LAMBDA_URL,
                json={"action": "stop"}
            )
            response.raise_for_status()
            data = response.json()

        # Parse the nested body if present
        if "body" in data:
            import json
            data = json.loads(data["body"])

        return EC2ActionResponse(
            message=data.get("message", "Instance stopping"),
            instance_id=data["instance_id"],
            state=data["state"],
            public_ip=data.get("public_ip")
        )

    except httpx.HTTPError as e:
        logger.error(f"Lambda stop call failed: {e}")
        raise HTTPException(status_code=503, detail="Failed to stop EC2")
    except Exception as e:
        logger.error(f"Pipeline stop error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ensure-running")
async def ensure_pipeline_running():
    """
    파이프라인 서버 준비 보장

    서버가 꺼져있으면 시작하고, 완전히 준비될 때까지 대기합니다.
    파이프라인 작업 전에 호출하세요.

    Returns:
        pipeline_url: 사용 가능한 파이프라인 서버 URL
    """
    try:
        # Check current status
        status = await get_pipeline_status()

        if status.ready:
            return {
                "message": "Pipeline server is ready",
                "pipeline_url": status.pipeline_url,
                "state": "running"
            }

        # Start if not running
        if status.state != "running":
            logger.info("Starting pipeline server...")
            await start_pipeline_server()

            # Wait for running state
            for _ in range(30):  # Max 60 seconds
                await asyncio.sleep(2)
                status = await get_pipeline_status()
                if status.state == "running":
                    break

        # Wait for pipeline service to be ready
        if status.public_ip:
            logger.info(f"Waiting for pipeline service at {status.public_ip}...")
            for _ in range(30):  # Max 60 seconds
                await asyncio.sleep(2)
                if await _check_pipeline_health(status.public_ip):
                    return {
                        "message": "Pipeline server is ready",
                        "pipeline_url": f"http://{status.public_ip}:8000",
                        "state": "running"
                    }

        raise HTTPException(
            status_code=503,
            detail="Pipeline server failed to become ready"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ensure running error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Helper Functions
# ============================================

async def _check_pipeline_health(ip: str) -> bool:
    """파이프라인 서버 헬스체크"""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"http://{ip}:8000/health")
            return response.status_code == 200
    except:
        return False


async def schedule_auto_shutdown(delay_minutes: int = 30):
    """
    자동 종료 스케줄링

    파이프라인 작업 완료 후 일정 시간 후 자동 종료
    """
    logger.info(f"Scheduling auto-shutdown in {delay_minutes} minutes")
    await asyncio.sleep(delay_minutes * 60)

    # Check if any pipeline jobs are running
    status = await get_pipeline_status()
    if status.state == "running":
        logger.info("Auto-shutdown: Stopping pipeline server")
        await stop_pipeline_server()
