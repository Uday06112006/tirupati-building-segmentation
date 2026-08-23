import os
import uuid
import logging
import traceback
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, List
from pathlib import Path

from ..models.schemas import JobStatus, JobResponse, ExtractionRequest
from ..config import settings
from .pipeline import OrthoPipeline

logger = logging.getLogger(__name__)

class JobManager:
    """
    Manages asynchronous and synchronous extraction jobs.
    """
    def __init__(self, max_workers: int = 4):
        self.jobs: Dict[str, JobResponse] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ortho-worker")

    def create_job(self, params: ExtractionRequest) -> JobResponse:
        """Initialize a new job record."""
        job_id = str(uuid.uuid4())[:8]
        now_str = datetime.now(timezone.utc).isoformat()
        
        job_dir = settings.JOBS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        job_resp = JobResponse(
            job_id=job_id,
            job_name=params.job_name or f"job_{job_id}",
            status=JobStatus.QUEUED,
            created_at=now_str,
            progress_percent=0.0,
            message="Job queued for processing",
            output_dir=str(job_dir)
        )
        self.jobs[job_id] = job_resp
        return job_resp

    def get_job(self, job_id: str) -> Optional[JobResponse]:
        """Retrieve job status by ID."""
        return self.jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> List[JobResponse]:
        """List all recent jobs."""
        return list(self.jobs.values())[-limit:]

    def run_job_sync(self, job_id: str, video_path: str, srt_path: str, params: ExtractionRequest) -> JobResponse:
        """Run job synchronously in current thread."""
        self._execute_pipeline(job_id, video_path, srt_path, params)
        return self.jobs[job_id]

    def submit_job_async(self, job_id: str, video_path: str, srt_path: str, params: ExtractionRequest):
        """Submit job for background execution in thread pool."""
        self.executor.submit(self._execute_pipeline, job_id, video_path, srt_path, params)

    def _execute_pipeline(self, job_id: str, video_path: str, srt_path: str, params: ExtractionRequest):
        """Worker function executing the pipeline and updating job state."""
        job = self.jobs.get(job_id)
        if not job:
            return

        job.status = JobStatus.PROCESSING
        job.message = "Processing started"
        
        def update_progress(pct: float, msg: str):
            job.progress_percent = pct
            job.message = msg

        try:
            job_dir = Path(job.output_dir)
            result = OrthoPipeline.run_pipeline(
                video_path=video_path,
                srt_path=srt_path,
                output_dir=str(job_dir),
                params=params,
                progress_callback=update_progress
            )
            
            v_info = result["video_info"]
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc).isoformat()
            job.progress_percent = 100.0
            job.message = f"Successfully extracted and geotagged {result['extracted_count']} frames in {result['elapsed_sec']}s"
            job.total_video_frames = v_info["total_frames"]
            job.extracted_frames = result["extracted_count"]
            job.video_duration_sec = round(v_info["duration_sec"], 2)
            job.video_fps = round(v_info["fps"], 2)
            job.video_resolution = v_info["resolution"]
            
            if result.get("geojson_path"):
                job.geojson_path = result["geojson_path"]
                job.geojson_url = f"/api/v1/jobs/{job_id}/geojson"
                
            if result.get("zip_path"):
                job.download_zip_path = result["zip_path"]
                job.download_zip_url = f"/api/v1/jobs/{job_id}/download"

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}\n{traceback.format_exc()}")
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now(timezone.utc).isoformat()
            job.error = str(e)
            job.message = f"Failed: {str(e)}"

# Global singleton
job_manager = JobManager(max_workers=settings.MAX_CONCURRENT_JOBS)
