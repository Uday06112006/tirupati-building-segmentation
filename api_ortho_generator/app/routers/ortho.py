import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse

from ..models.schemas import (
    OrthoRequest,
    OrthoJobResponse,
    OrthoJobStatus,
    OrthoEngine,
    BlendMode,
    CompressionType
)
from ..config import settings
from ..services.job_manager import ortho_job_manager

router = APIRouter(prefix="/ortho", tags=["2D Orthomosaic Generation"])

@router.post("/generate-from-path", response_model=OrthoJobResponse, summary="Generate Accurate Orthophoto from server frames folder (Fastest & Accurate)")
async def generate_from_path(
    request: OrthoRequest,
    async_mode: bool = Query(True, description="Process in background thread")
):
    """
    Generate a Survey-Grade Georeferenced GeoTIFF Orthophoto directly from an existing folder of geotagged frames.
    - `engine`: `'accurate'` (Photogrammetric SfM + 3D DSM + True Orthorectification) or `'fast'` (Rapid planar mosaic)
    - `generate_dsm`: Set to `true` to also output a 3D Digital Surface Model elevation GeoTIFF.
    """
    if not request.frames_dir or not os.path.exists(request.frames_dir):
        raise HTTPException(status_code=400, detail=f"Frames directory not found: {request.frames_dir}")

    job = ortho_job_manager.create_job(request)

    if async_mode:
        ortho_job_manager.submit_job_async(job.job_id, request.frames_dir, request)
        return ortho_job_manager.get_job(job.job_id)
    else:
        return ortho_job_manager.run_job_sync(job.job_id, request.frames_dir, request)

@router.post("/generate-from-upload", response_model=OrthoJobResponse, summary="Generate Accurate Orthophoto from uploaded ZIP of frames")
async def generate_from_upload(
    background_tasks: BackgroundTasks,
    zip_file: UploadFile = File(..., description="ZIP archive containing geotagged image frames"),
    engine: OrthoEngine = Form(OrthoEngine.ACCURATE, description="Processing engine: 'accurate' or 'fast'"),
    target_gsd_cm: float = Form(5.0, description="Target Ground Sampling Distance in cm/pixel (default 5.0)"),
    blend_mode: BlendMode = Form(BlendMode.MULTIBAND, description="Mosaic blending method"),
    generate_dsm: bool = Form(True, description="Generate DSM elevation GeoTIFF"),
    compression: CompressionType = Form(CompressionType.DEFLATE, description="GeoTIFF compression"),
    crs: str = Form("AUTO", description="Coordinate Reference System ('AUTO', 'EPSG:4326', etc.)"),
    max_images: Optional[int] = Form(None, description="Max images limit"),
    job_name: Optional[str] = Form(None, description="Custom job name"),
    async_mode: bool = Form(True, description="Process in background")
):
    """
    Upload a ZIP archive of geotagged image frames to generate a Survey-Grade Georeferenced GeoTIFF and DSM.
    """
    params = OrthoRequest(
        engine=engine,
        target_gsd_cm=target_gsd_cm,
        blend_mode=blend_mode,
        generate_dsm=generate_dsm,
        compression=compression,
        crs=crs,
        max_images=max_images,
        job_name=job_name or Path(zip_file.filename or "ortho").stem
    )
    
    job = ortho_job_manager.create_job(params)
    job_dir = settings.JOBS_DIR / job.job_id
    upload_dir = job_dir / "uploaded_frames"
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save and extract zip
    zip_temp = job_dir / "upload.zip"
    with open(zip_temp, "wb") as f:
        shutil.copyfileobj(zip_file.file, f)
        
    try:
        with zipfile.ZipFile(zip_temp, "r") as z:
            z.extractall(upload_dir)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid ZIP archive: {str(e)}")

    if async_mode:
        ortho_job_manager.submit_job_async(job.job_id, str(upload_dir), params)
        return ortho_job_manager.get_job(job.job_id)
    else:
        return ortho_job_manager.run_job_sync(job.job_id, str(upload_dir), params)

@router.get("/jobs", response_model=List[OrthoJobResponse], summary="List all orthomosaic jobs")
async def list_ortho_jobs(limit: int = Query(50, ge=1, le=200)):
    """Retrieve list of recently executed or active orthomosaic jobs."""
    return ortho_job_manager.list_jobs(limit=limit)

@router.get("/jobs/{job_id}", response_model=OrthoJobResponse, summary="Get orthomosaic job status and metadata")
async def get_ortho_job_status(job_id: str):
    """Get live progress percentage, dimensions, spatial extent, and download URLs."""
    job = ortho_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job

@router.get("/jobs/{job_id}/geotiff", summary="Download Georeferenced Orthophoto GeoTIFF (.tif)")
async def download_geotiff(job_id: str):
    """Download the full georeferenced GeoTIFF orthomosaic image."""
    job = ortho_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
    if not job.geotiff_path or not os.path.exists(job.geotiff_path):
        raise HTTPException(status_code=404, detail="GeoTIFF output not found or job still in progress")
        
    return FileResponse(
        path=job.geotiff_path,
        media_type="image/tiff",
        filename=os.path.basename(job.geotiff_path)
    )

@router.get("/jobs/{job_id}/dsm", summary="Download Digital Surface Model (DSM) Elevation GeoTIFF (.tif)")
async def download_dsm_geotiff(job_id: str):
    """Download the georeferenced 3D terrain elevation model GeoTIFF."""
    job = ortho_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
    if not job.dsm_path or not os.path.exists(job.dsm_path):
        raise HTTPException(status_code=404, detail="DSM GeoTIFF output not found for this job")
        
    return FileResponse(
        path=job.dsm_path,
        media_type="image/tiff",
        filename=os.path.basename(job.dsm_path)
    )

@router.get("/jobs/{job_id}/preview", summary="View/Download PNG preview thumbnail")
async def get_preview_png(job_id: str):
    """View lightweight PNG preview thumbnail of the generated orthomosaic."""
    job = ortho_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
    job_dir = settings.JOBS_DIR / job_id
    png_files = list(job_dir.glob("*_preview.png"))
    if not png_files or not png_files[0].exists():
        raise HTTPException(status_code=404, detail="Preview PNG not found")
        
    return FileResponse(
        path=str(png_files[0]),
        media_type="image/png",
        filename=png_files[0].name
    )
