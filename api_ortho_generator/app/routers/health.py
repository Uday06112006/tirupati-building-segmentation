from fastapi import APIRouter
from ..models.schemas import HealthResponse, BlendMode, CompressionType, OrthoEngine
from ..config import settings
from ..services.job_manager import ortho_job_manager

router = APIRouter(tags=["Health & System"])

@router.get("/health", response_model=HealthResponse, summary="Check 2D Orthophoto API health")
async def health_check():
    """Returns operational status and capabilities of the 2D Orthophoto generation service."""
    active_jobs = sum(1 for j in ortho_job_manager.jobs.values() if j.status in ["QUEUED", "PROCESSING"])
    
    return HealthResponse(
        status="HEALTHY",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        supported_engines=[e.value for e in OrthoEngine],
        supported_blend_modes=[b.value for b in BlendMode],
        supported_compressions=[c.value for c in CompressionType],
        active_jobs=active_jobs
    )
