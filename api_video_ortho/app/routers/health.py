import os
import cv2
from fastapi import APIRouter
from ..models.schemas import HealthResponse
from ..config import settings
from ..services.job_manager import job_manager

router = APIRouter(tags=["Health & System"])

def get_directory_size(path) -> int:
    """Calculate directory size in bytes."""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_directory_size(entry.path)
    except Exception:
        pass
    return total

@router.get("/health", response_model=HealthResponse, summary="Check API health status")
async def health_check():
    """Returns the operational status of the service, supported codecs, and system metrics."""
    active_jobs = sum(1 for j in job_manager.jobs.values() if j.status in ["QUEUED", "PROCESSING"])
    storage_size = get_directory_size(settings.STORAGE_DIR)
    
    return HealthResponse(
        status="HEALTHY",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        opencv_version=cv2.__version__,
        active_jobs=active_jobs,
        storage_used_bytes=storage_size,
        supported_video_formats=[".mov", ".mp4", ".avi", ".mkv", ".m4v", ".ts"],
        supported_image_formats=[".jpg", ".jpeg", ".png"]
    )
