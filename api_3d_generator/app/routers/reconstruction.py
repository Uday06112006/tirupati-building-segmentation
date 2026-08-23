import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse

from ..models.schemas import (
    ReconstructionRequest,
    ThreeDJobResponse,
    ThreeDJobStatus,
    MeshMethod,
    MatchingMethod
)
from ..config import settings
from ..services.job_manager import threed_job_manager

router = APIRouter(prefix="/3d", tags=["3D Reconstruction & Meshing"])

@router.post("/reconstruct-from-path", response_model=ThreeDJobResponse, summary="Reconstruct 3D Point Cloud & Mesh from server frames folder (Fastest)")
async def reconstruct_from_path(
    request: ReconstructionRequest,
    async_mode: bool = Query(True, description="Process in background thread")
):
    """
    Execute 3D Structure from Motion (SfM) and 3D surface meshing on an existing folder of drone frames.
    """
    if not request.frames_dir or not os.path.exists(request.frames_dir):
        raise HTTPException(status_code=400, detail=f"Frames directory not found: {request.frames_dir}")

    job = threed_job_manager.create_job(request)

    if async_mode:
        threed_job_manager.submit_job_async(job.job_id, request.frames_dir, request)
        return threed_job_manager.get_job(job.job_id)
    else:
        return threed_job_manager.run_job_sync(job.job_id, request.frames_dir, request)

@router.post("/reconstruct-from-upload", response_model=ThreeDJobResponse, summary="Reconstruct 3D Model from uploaded ZIP of frames")
async def reconstruct_from_upload(
    background_tasks: BackgroundTasks,
    zip_file: UploadFile = File(..., description="ZIP archive containing geotagged image frames"),
    matching_method: MatchingMethod = Form(MatchingMethod.SEQUENTIAL, description="Feature matcher"),
    generate_mesh: bool = Form(True, description="Generate 3D Surface Mesh"),
    mesh_method: MeshMethod = Form(MeshMethod.POISSON, description="Meshing method"),
    poisson_depth: int = Form(8, description="Poisson tree depth (5-12)"),
    max_images: Optional[int] = Form(None, description="Limit image count"),
    job_name: Optional[str] = Form(None, description="Custom job name"),
    async_mode: bool = Form(True, description="Process in background")
):
    """
    Upload a ZIP archive of images to generate 3D point cloud (.ply) and 3D mesh (.obj).
    """
    params = ReconstructionRequest(
        matching_method=matching_method,
        generate_mesh=generate_mesh,
        mesh_method=mesh_method,
        poisson_depth=poisson_depth,
        max_images=max_images,
        job_name=job_name or Path(zip_file.filename or "model").stem
    )
    
    job = threed_job_manager.create_job(params)
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
        threed_job_manager.submit_job_async(job.job_id, str(upload_dir), params)
        return threed_job_manager.get_job(job.job_id)
    else:
        return threed_job_manager.run_job_sync(job.job_id, str(upload_dir), params)

@router.get("/jobs", response_model=List[ThreeDJobResponse], summary="List all 3D reconstruction jobs")
async def list_3d_jobs(limit: int = Query(50, ge=1, le=200)):
    """Retrieve list of recently executed or active 3D jobs."""
    return threed_job_manager.list_jobs(limit=limit)

@router.get("/jobs/{job_id}", response_model=ThreeDJobResponse, summary="Get 3D job status, point counts, and download links")
async def get_3d_job_status(job_id: str):
    """Get live progress percentage, vertex/triangle counts, and download URLs."""
    job = threed_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job

@router.get("/jobs/{job_id}/pointcloud", summary="Download 3D Point Cloud (.ply)")
async def download_pointcloud(job_id: str):
    """Download the reconstructed 3D Point Cloud in PLY format."""
    job = threed_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
    ply_path = Path(job.output_dir) / "point_cloud.ply"
    if not ply_path.exists():
        raise HTTPException(status_code=404, detail="Point cloud file not found or job still processing")
        
    return FileResponse(
        path=str(ply_path),
        media_type="application/octet-stream",
        filename=f"{job.job_name}_point_cloud.ply"
    )

@router.get("/jobs/{job_id}/mesh", summary="Download 3D Surface Mesh (.obj)")
async def download_mesh(job_id: str):
    """Download the reconstructed 3D Mesh in OBJ format."""
    job = threed_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
    job_dir = Path(job.output_dir)
    obj_files = list(job_dir.glob("*.obj"))
    if not obj_files or not obj_files[0].exists():
        raise HTTPException(status_code=404, detail="3D mesh OBJ file not found")
        
    return FileResponse(
        path=str(obj_files[0]),
        media_type="application/octet-stream",
        filename=obj_files[0].name
    )

@router.get("/jobs/{job_id}/download", summary="Download full 3D ZIP package")
async def download_3d_zip(job_id: str):
    """Download packaged ZIP containing point clouds (.ply) and 3D meshes (.obj)."""
    job = threed_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
    job_dir = Path(job.output_dir)
    zip_files = list(job_dir.glob("*.zip"))
    if not zip_files or not zip_files[0].exists():
        raise HTTPException(status_code=404, detail="3D ZIP package not found")
        
    return FileResponse(
        path=str(zip_files[0]),
        media_type="application/zip",
        filename=zip_files[0].name
    )
