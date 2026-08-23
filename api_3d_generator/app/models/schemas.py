from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator

class MeshMethod(str, Enum):
    POISSON = "poisson"
    BALL_PIVOTING = "ball_pivoting"
    NONE = "none"

class MatchingMethod(str, Enum):
    SEQUENTIAL = "sequential"
    EXHAUSTIVE = "exhaustive"
    SPATIAL = "spatial"

class ThreeDJobStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class ReconstructionRequest(BaseModel):
    frames_dir: Optional[str] = Field(None, description="Absolute path to geotagged drone frames folder on server")
    matching_method: MatchingMethod = Field(MatchingMethod.SEQUENTIAL, description="Feature matcher: sequential (fastest for drone paths), exhaustive, or spatial")
    generate_mesh: bool = Field(True, description="Generate 3D Textured Surface Mesh (.obj)")
    mesh_method: MeshMethod = Field(MeshMethod.POISSON, description="3D meshing algorithm (poisson, ball_pivoting, none)")
    poisson_depth: int = Field(8, ge=5, le=12, description="Poisson reconstruction tree depth (higher = more detailed mesh)")
    max_images: Optional[int] = Field(None, description="Limit number of images to reconstruct (0 or null = all)")
    job_name: Optional[str] = Field(None, description="Custom job name")

    @field_validator("max_images", mode="before")
    @classmethod
    def sanitize_max_images(cls, v):
        if v is None or v == "" or v == "null":
            return None
        try:
            val = int(v)
            return val if val > 0 else None
        except (ValueError, TypeError):
            return None

class ThreeDJobResponse(BaseModel):
    job_id: str
    job_name: Optional[str] = None
    status: ThreeDJobStatus
    created_at: str
    completed_at: Optional[str] = None
    progress_percent: float = Field(0.0, ge=0.0, le=100.0)
    message: str = ""
    registered_images: Optional[int] = None
    num_3d_points: Optional[int] = None
    num_mesh_vertices: Optional[int] = None
    num_mesh_triangles: Optional[int] = None
    pointcloud_ply_url: Optional[str] = None
    mesh_obj_url: Optional[str] = None
    download_zip_url: Optional[str] = None
    output_dir: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    pycolmap_version: str
    open3d_version: str
    cuda_available: bool
    active_jobs: int
