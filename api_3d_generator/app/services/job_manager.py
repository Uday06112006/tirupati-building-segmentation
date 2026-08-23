import os
import uuid
import logging
import traceback
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, List
from pathlib import Path

from ..models.schemas import ThreeDJobStatus, ThreeDJobResponse, ReconstructionRequest
from ..config import settings
from .pipeline import ThreeDPipeline

logger = logging.getLogger(__name__)

class ThreeDJobManager:
    """
    Manages asynchronous and synchronous 3D reconstruction jobs.
    """
    def __init__(self, max_workers: int = 1):
        self.jobs: Dict[str, ThreeDJobResponse] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="threed-worker")

    def create_job(self, params: ReconstructionRequest) -> ThreeDJobResponse:
        """Initialize a new 3D job record."""
        job_id = str(uuid.uuid4())[:8]
        now_str = datetime.now(timezone.utc).isoformat()
        
        job_dir = settings.JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        job_resp = ThreeDJobResponse(
            job_id=job_id,
            job_name=params.job_name or f"3d_{job_id}",
            status=ThreeDJobStatus.QUEUED,
            created_at=now_str,
            progress_percent=0.0,
            message="Job queued for 3D reconstruction",
            output_dir=str(job_dir)
        )
        self.jobs[job_id] = job_resp
        return job_resp

    def get_job(self, job_id: str) -> Optional[ThreeDJobResponse]:
        """Retrieve job status by ID."""
        return self.jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> List[ThreeDJobResponse]:
        """List all recent jobs."""
        return list(self.jobs.values())[-limit:]

    def run_job_sync(self, job_id: str, frames_dir: str, params: ReconstructionRequest) -> ThreeDJobResponse:
        """Run job synchronously."""
        self._execute_pipeline(job_id, frames_dir, params)
        return self.jobs[job_id]

    def submit_job_async(self, job_id: str, frames_dir: str, params: ReconstructionRequest):
        """Submit job for background execution."""
        self.executor.submit(self._execute_pipeline, job_id, frames_dir, params)

    def _execute_pipeline(self, job_id: str, frames_dir: str, params: ReconstructionRequest):
        """Worker function executing the pipeline and updating job state."""
        job = self.jobs.get(job_id)
        if not job:
            return

        job.status = ThreeDJobStatus.PROCESSING
        job.message = "3D Reconstruction processing started"
        
        def update_progress(pct: float, msg: str):
            job.progress_percent = pct
            job.message = msg

        try:
            job_dir = settings.JOBS_DIR / job_id
            result = ThreeDPipeline.run_pipeline(
                frames_dir=frames_dir,
                output_dir=str(job_dir),
                params=params,
                progress_callback=update_progress
            )
            
            sfm = result["sfm_stats"]
            mesh = result["mesh_stats"]
            
            job.status = ThreeDJobStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc).isoformat()
            job.progress_percent = 100.0
            job.message = f"Reconstructed {sfm['num_3d_points']} 3D points ({mesh.get('num_triangles', 0)} triangles) in {result['elapsed_seconds']}s"
            job.registered_images = sfm["num_registered_images"]
            job.num_3d_points = sfm["num_3d_points"]
            job.num_mesh_vertices = mesh.get("num_vertices")
            job.num_mesh_triangles = mesh.get("num_triangles")
            job.pointcloud_ply_url = f"/api/v1/3d/jobs/{job_id}/pointcloud"
            job.mesh_obj_url = f"/api/v1/3d/jobs/{job_id}/mesh"
            job.download_zip_url = f"/api/v1/3d/jobs/{job_id}/download"
            job.elapsed_seconds = result["elapsed_seconds"]

        except Exception as e:
            logger.error(f"3D job {job_id} failed: {e}\n{traceback.format_exc()}")
            job.status = ThreeDJobStatus.FAILED
            job.completed_at = datetime.now(timezone.utc).isoformat()
            job.error = str(e)
            job.message = f"Failed: {str(e)}"

# Global singleton
threed_job_manager = ThreeDJobManager(max_workers=settings.MAX_CONCURRENT_JOBS)
