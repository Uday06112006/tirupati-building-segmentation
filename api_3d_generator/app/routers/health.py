import torch
import pycolmap
import open3d as o3d
from fastapi import APIRouter
from ..models.schemas import HealthResponse
from ..config import settings
from ..services.job_manager import threed_job_manager

router = APIRouter(tags=["Health & System"])

@router.get("/health", response_model=HealthResponse, summary="Check 3D Reconstruction API health")
async def health_check():
    """Returns operational status, PyCOLMAP & Open3D versions, and GPU/CUDA availability."""
    active_jobs = sum(1 for j in threed_job_manager.jobs.values() if j.status in ["QUEUED", "PROCESSING"])
    cuda_avail = torch.cuda.is_available()
    
    return HealthResponse(
        status="HEALTHY",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        pycolmap_version=getattr(pycolmap, "__version__", "4.0.4"),
        open3d_version=o3d.__version__,
        cuda_available=cuda_avail,
        active_jobs=active_jobs
    )
