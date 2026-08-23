import os
import uuid
import logging
import traceback
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, List
from pathlib import Path

from ..models.schemas import OrthoJobStatus, OrthoJobResponse, OrthoRequest
from ..config import settings
from .pipeline import OrthoPipeline

logger = logging.getLogger(__name__)

class OrthoJobManager:
    """
    Manages asynchronous and synchronous orthomosaic processing jobs.
    """
    def __init__(self, max_workers: int = 2):
        self.jobs: Dict[str, OrthoJobResponse] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ortho-gen-worker")

    def create_job(self, params: OrthoRequest) -> OrthoJobResponse:
        """Initialize a new job record."""
        job_id = str(uuid.uuid4())[:8]
        now_str = datetime.now(timezone.utc).isoformat()
        
        job_dir = settings.JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        job_resp = OrthoJobResponse(
            job_id=job_id,
            job_name=params.job_name or f"ortho_{job_id}",
            status=OrthoJobStatus.QUEUED,
            engine_used=params.engine.value,
            created_at=now_str,
            progress_percent=0.0,
            message="Job queued for orthomosaic generation",
            geotiff_path=None
        )
        self.jobs[job_id] = job_resp
        return job_resp

    def get_job(self, job_id: str) -> Optional[OrthoJobResponse]:
        """Retrieve job status by ID."""
        return self.jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> List[OrthoJobResponse]:
        """List all recent jobs."""
        return list(self.jobs.values())[-limit:]

    def run_job_sync(self, job_id: str, frames_dir: str, params: OrthoRequest) -> OrthoJobResponse:
        """Run job synchronously in current thread."""
        self._execute_pipeline(job_id, frames_dir, params)
        return self.jobs[job_id]

    def submit_job_async(self, job_id: str, frames_dir: str, params: OrthoRequest):
        """Submit job for background execution in thread pool."""
        self.executor.submit(self._execute_pipeline, job_id, frames_dir, params)

    def _execute_pipeline(self, job_id: str, frames_dir: str, params: OrthoRequest):
        """Worker function executing the pipeline and updating job state."""
        job = self.jobs.get(job_id)
        if not job:
            return

        job.status = OrthoJobStatus.PROCESSING
        job.message = "Orthomosaic processing started"
        
        def update_progress(pct: float, msg: str):
            job.progress_percent = pct
            job.message = msg

        try:
            job_dir = settings.JOBS_DIR / job_id
            result = OrthoPipeline.run_pipeline(
                frames_dir=frames_dir,
                output_dir=str(job_dir),
                params=params,
                progress_callback=update_progress
            )
            
            meta = result["spatial_meta"]
            job.status = OrthoJobStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc).isoformat()
            job.progress_percent = 100.0
            job.message = f"Successfully generated {result['engine_used']} orthomosaic ({meta['width_px']}x{meta['height_px']}px) in {result['elapsed_seconds']}s"
            job.engine_used = result["engine_used"]
            job.total_input_images = result["total_images"]
            job.stitched_images = meta["total_images_processed"]
            job.num_sfm_points = meta.get("num_sfm_points")
            job.gsd_cm_per_pixel = meta["gsd_cm"]
            job.ortho_width_px = meta["width_px"]
            job.ortho_height_px = meta["height_px"]
            job.bounds_utm = meta["bounds_utm"]
            job.bounds_wgs84 = meta["bounds_wgs84"]
            job.crs = meta["crs"]
            job.geotiff_path = result["geotiff_path"]
            job.geotiff_url = f"/api/v1/ortho/jobs/{job_id}/geotiff"
            job.preview_url = f"/api/v1/ortho/jobs/{job_id}/preview"
            
            if result.get("dsm_path"):
                job.dsm_path = result["dsm_path"]
                job.dsm_url = f"/api/v1/ortho/jobs/{job_id}/dsm"
                
            job.elapsed_seconds = result["elapsed_seconds"]

        except Exception as e:
            logger.error(f"Ortho job {job_id} failed: {e}\n{traceback.format_exc()}")
            job.status = OrthoJobStatus.FAILED
            job.completed_at = datetime.now(timezone.utc).isoformat()
            job.error = str(e)
            job.message = f"Failed: {str(e)}"

# Global singleton
ortho_job_manager = OrthoJobManager(max_workers=settings.MAX_CONCURRENT_JOBS)
