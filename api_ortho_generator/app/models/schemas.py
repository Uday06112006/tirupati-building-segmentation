from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator

class OrthoEngine(str, Enum):
    ACCURATE = "accurate"      # Photogrammetric SfM + 3D DSM + True Orthorectification
    FAST = "fast"              # Fast planar georeferencing & distance feathering

class BlendMode(str, Enum):
    FEATHER = "feather"
    MULTIBAND = "multiband"
    AVERAGE = "average"
    LATEST = "latest"

class OrthoJobStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class CompressionType(str, Enum):
    DEFLATE = "DEFLATE"
    LZW = "LZW"
    JPEG = "JPEG"
    PACKBITS = "PACKBITS"

class OrthoRequest(BaseModel):
    frames_dir: Optional[str] = Field(None, description="Absolute path to folder containing geotagged frames on server")
    engine: OrthoEngine = Field(OrthoEngine.ACCURATE, description="Processing engine: 'accurate' (SfM + DSM + True Orthorectification) or 'fast' (Rapid planar mosaic)")
    target_gsd_cm: Optional[float] = Field(5.0, description="Target Ground Sampling Distance in cm/pixel (default 5.0 cm/px)")
    blend_mode: BlendMode = Field(BlendMode.MULTIBAND, description="Mosaic blending algorithm: multiband, feather, average, latest")
    generate_dsm: bool = Field(True, description="Generate Digital Surface Model (DSM) elevation GeoTIFF alongside orthophoto")
    compression: CompressionType = Field(CompressionType.DEFLATE, description="GeoTIFF compression format")
    crs: str = Field("AUTO", description="Target Coordinate Reference System: 'AUTO' (Auto-calculates local UTM zone), 'EPSG:4326', or specific EPSG code")
    max_images: Optional[int] = Field(None, description="Max images to process (0 or null = all frames)")
    job_name: Optional[str] = Field(None, description="Custom job name identifier")

    @field_validator("target_gsd_cm", mode="before")
    @classmethod
    def sanitize_gsd(cls, v):
        if v is None or v == "" or v == "null":
            return 5.0
        try:
            val = float(v)
            return val if val > 0 else 5.0
        except (ValueError, TypeError):
            return 5.0

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

class OrthoJobResponse(BaseModel):
    job_id: str
    job_name: Optional[str] = None
    status: OrthoJobStatus
    engine_used: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    progress_percent: float = Field(0.0, ge=0.0, le=100.0)
    message: str = ""
    total_input_images: Optional[int] = None
    stitched_images: Optional[int] = None
    num_sfm_points: Optional[int] = None
    gsd_cm_per_pixel: Optional[float] = None
    ortho_width_px: Optional[int] = None
    ortho_height_px: Optional[int] = None
    bounds_wgs84: Optional[List[float]] = None
    bounds_utm: Optional[List[float]] = None
    crs: Optional[str] = None
    geotiff_path: Optional[str] = None
    geotiff_url: Optional[str] = None
    dsm_path: Optional[str] = None
    dsm_url: Optional[str] = None
    preview_url: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    supported_engines: List[str]
    supported_blend_modes: List[str]
    supported_compressions: List[str]
    active_jobs: int
