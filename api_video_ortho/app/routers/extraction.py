import os
import shutil
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse

from ..models.schemas import (
    ExtractionRequest,
    JobResponse,
    JobStatus,
    OutputFormat
)
from ..config import settings
from ..services.job_manager import job_manager

router = APIRouter(prefix="/jobs", tags=["Frame Extraction & Geotagging"])

@router.post("/extract-upload", response_model=JobResponse, summary="Extract frames from uploaded Video + SRT files")
async def extract_from_upload(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="Drone video file (.MOV, .MP4, .AVI)"),
    srt: UploadFile = File(..., description="Telemetry subtitle file (.SRT or .CSV)"),
    frame_step: int = Form(1, description="Extract every N-th frame (1 = all frames, 2 = every 2nd frame)"),
    frame_interval_sec: Optional[float] = Form(None, description="Extract 1 frame every X seconds (e.g. 1.0). Leave empty or 0 to use frame_step"),
    fps: Optional[float] = Form(None, description="Target extraction FPS (e.g. 2.0). Leave empty or 0 to ignore"),
    start_time_sec: Optional[float] = Form(0.0, description="Start offset in seconds (0.0 = start of video)"),
    end_time_sec: Optional[float] = Form(None, description="End offset in seconds (Leave empty or 0 for entire video until the end)"),
    max_frames: Optional[int] = Form(None, description="Max frame limit (Leave empty or 0 for entire video without limit)"),
    output_format: OutputFormat = Form(OutputFormat.JPG, description="Output image format (jpg, jpeg, png)"),
    jpeg_quality: int = Form(100, description="JPEG quality 1-100 (100 = highest visual quality)"),
    include_line_string: bool = Form(True, description="Include LineString trajectory in GeoJSON"),
    export_zip: bool = Form(True, description="Create a downloadable ZIP archive"),
    export_geojson: bool = Form(True, description="Create GeoJSON dataset"),
    job_name: Optional[str] = Form(None, description="Optional custom job name"),
    async_mode: bool = Form(True, description="Process asynchronously in background thread")
):
    """
    Upload a video file and an SRT subtitle file to extract high quality frames,
    embed EXIF GPS & telemetry metadata, and generate a GeoJSON flight trajectory.

    Note: To process the **entire video**, simply leave `end_time_sec` and `max_frames` blank or set to `0`.
    """
    try:
        params = ExtractionRequest(
            frame_step=frame_step,
            frame_interval_sec=frame_interval_sec,
            fps=fps,
            start_time_sec=start_time_sec,
            end_time_sec=end_time_sec,
            max_frames=max_frames,
            output_format=output_format,
            jpeg_quality=jpeg_quality,
            include_line_string=include_line_string,
            export_zip=export_zip,
            export_geojson=export_geojson,
            job_name=job_name or Path(video.filename or "video").stem
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid extraction parameters: {str(e)}")
    
    job = job_manager.create_job(params)
    job_dir = Path(job.output_dir)
    uploads_dir = job_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded files
    video_filename = video.filename or "video.mov"
    srt_filename = srt.filename or "telemetry.srt"
    video_dest = uploads_dir / video_filename
    with open(video_dest, "wb") as f:
        shutil.copyfileobj(video.file, f)
        
    srt_dest = uploads_dir / srt_filename
    with open(srt_dest, "wb") as f:
        shutil.copyfileobj(srt.file, f)

    if async_mode:
        job_manager.submit_job_async(job.job_id, str(video_dest), str(srt_dest), params)
        return job_manager.get_job(job.job_id)
    else:
        return job_manager.run_job_sync(job.job_id, str(video_dest), str(srt_dest), params)

@router.post("/extract-from-path", response_model=JobResponse, summary="Extract frames from local server file paths")
async def extract_from_path(
    request: ExtractionRequest,
    async_mode: bool = Query(True, description="Process in background thread")
):
    """
    Trigger extraction by pointing to existing video and SRT files already residing on the server.
    This eliminates file upload transfer overhead for large 4K/8K drone videos.
    """
    if not request.video_path or not os.path.exists(request.video_path):
        raise HTTPException(status_code=400, detail=f"Video file not found at: {request.video_path}")
    if not request.srt_path or not os.path.exists(request.srt_path):
        raise HTTPException(status_code=400, detail=f"SRT file not found at: {request.srt_path}")

    job = job_manager.create_job(request)

    if async_mode:
        job_manager.submit_job_async(job.job_id, request.video_path, request.srt_path, request)
        return job_manager.get_job(job.job_id)
    else:
        return job_manager.run_job_sync(job.job_id, request.video_path, request.srt_path, request)

@router.get("", response_model=List[JobResponse], summary="List all processing jobs")
async def list_jobs(limit: int = Query(50, ge=1, le=200)):
    """Retrieve list of recently executed or active jobs."""
    return job_manager.list_jobs(limit=limit)

@router.get("/{job_id}", response_model=JobResponse, summary="Get job status and progress")
async def get_job_status(job_id: str):
    """Get the live progress, execution status, and download links for a job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job

@router.get("/{job_id}/geojson", summary="Download or view generated GeoJSON")
async def get_job_geojson(job_id: str):
    """Returns the generated GeoJSON flight path and frame locations."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
    if not job.geojson_path or not os.path.exists(job.geojson_path):
        raise HTTPException(status_code=404, detail="GeoJSON not found for this job")
        
    return FileResponse(
        path=job.geojson_path,
        media_type="application/geo+json",
        filename=os.path.basename(job.geojson_path)
    )

@router.get("/{job_id}/download", summary="Download complete ZIP package")
async def download_job_zip(job_id: str):
    """Download ZIP package containing all geotagged frames and the GeoJSON flight log."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
    if not job.download_zip_path or not os.path.exists(job.download_zip_path):
        raise HTTPException(status_code=404, detail="Download ZIP archive not ready or not found")
        
    return FileResponse(
        path=job.download_zip_path,
        media_type="application/zip",
        filename=os.path.basename(job.download_zip_path)
    )

@router.get("/{job_id}/frames/{filename}", summary="Download / preview individual geotagged frame")
async def get_job_frame(job_id: str, filename: str):
    """View or download a specific geotagged image frame."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
    frame_path = Path(job.output_dir) / "frames" / filename
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail=f"Frame {filename} not found")
        
    return FileResponse(
        path=str(frame_path),
        media_type="image/jpeg",
        filename=filename
    )
