from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator

class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class OutputFormat(str, Enum):
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"

class FrameMetadata(BaseModel):
    frame_index: int = Field(..., description="0-based or 1-based sequential index of the frame in video")
    frame_id: Optional[int] = Field(None, description="Frame count reported in SRT subtitle")
    timestamp_str: Optional[str] = Field(None, description="ISO formatted or raw subtitle timestamp string")
    timestamp_sec: float = Field(..., description="Timestamp in video in seconds")
    latitude: float = Field(..., description="GPS Latitude in WGS-84 decimal degrees")
    longitude: float = Field(..., description="GPS Longitude in WGS-84 decimal degrees")
    altitude: float = Field(..., description="Altitude in meters (AMSL or Relative)")
    iso: Optional[int] = Field(None, description="Camera ISO speed")
    shutter: Optional[str] = Field(None, description="Camera shutter speed e.g. 1/640.0")
    fnum: Optional[float] = Field(None, description="Aperture f-number e.g. 2.8")
    focal_len: Optional[float] = Field(None, description="Focal length in mm e.g. 28.0")
    ev: Optional[float] = Field(None, description="Exposure value compensation")
    ct: Optional[int] = Field(None, description="Color temperature in Kelvin")
    color_md: Optional[str] = Field(None, description="Color mode")
    yaw: Optional[float] = Field(None, description="Gimbal / Drone yaw angle in degrees")
    pitch: Optional[float] = Field(None, description="Gimbal / Drone pitch angle in degrees")
    roll: Optional[float] = Field(None, description="Gimbal / Drone roll angle in degrees")
    filename: Optional[str] = Field(None, description="Output image filename")
    image_path: Optional[str] = Field(None, description="Absolute file path to saved image")

class ExtractionRequest(BaseModel):
    video_path: Optional[str] = Field(None, description="Path to video file on server")
    srt_path: Optional[str] = Field(None, description="Path to SRT subtitle file on server")
    frame_step: int = Field(1, description="Extract every N-th frame (1 = all frames, 2 = every 2nd frame)")
    frame_interval_sec: Optional[float] = Field(None, description="Extract 1 frame every X seconds (0 or null = use frame_step)")
    fps: Optional[float] = Field(None, description="Target extraction FPS (e.g. 2.0 for 2 frames/sec, 0 or null = ignore)")
    start_time_sec: Optional[float] = Field(0.0, description="Start time offset in video in seconds (default 0.0 = start of video)")
    end_time_sec: Optional[float] = Field(None, description="End time offset in seconds (0 or null = entire video until the end)")
    max_frames: Optional[int] = Field(None, description="Maximum number of frames to extract (0 or null = no limit, whole video)")
    output_format: OutputFormat = Field(OutputFormat.JPG, description="Image format: jpg, jpeg, png")
    jpeg_quality: int = Field(100, description="JPEG compression quality (100 = maximum visual quality)")
    include_line_string: bool = Field(True, description="Include LineString trajectory in GeoJSON")
    export_zip: bool = Field(True, description="Create a downloadable ZIP archive with frames and GeoJSON")
    export_geojson: bool = Field(True, description="Generate a GeoJSON file alongside images")
    job_name: Optional[str] = Field(None, description="Optional custom job identifier or label")

    @field_validator("frame_interval_sec", "fps", "end_time_sec", mode="before")
    @classmethod
    def sanitize_float_fields(cls, v):
        if v is None or v == "" or v == "null":
            return None
        try:
            val = float(v)
            return val if val > 0 else None
        except (ValueError, TypeError):
            return None

    @field_validator("max_frames", mode="before")
    @classmethod
    def sanitize_max_frames(cls, v):
        if v is None or v == "" or v == "null":
            return None
        try:
            val = int(v)
            return val if val > 0 else None
        except (ValueError, TypeError):
            return None

    @field_validator("start_time_sec", mode="before")
    @classmethod
    def sanitize_start_time(cls, v):
        if v is None or v == "" or v == "null":
            return 0.0
        try:
            val = float(v)
            return max(0.0, val)
        except (ValueError, TypeError):
            return 0.0

    @field_validator("frame_step", mode="before")
    @classmethod
    def sanitize_frame_step(cls, v):
        if v is None or v == "" or v == "null":
            return 1
        try:
            val = int(v)
            return val if val >= 1 else 1
        except (ValueError, TypeError):
            return 1

    @field_validator("jpeg_quality", mode="before")
    @classmethod
    def sanitize_quality(cls, v):
        if v is None or v == "" or v == "null":
            return 100
        try:
            val = int(v)
            return max(1, min(100, val))
        except (ValueError, TypeError):
            return 100

class JobResponse(BaseModel):
    job_id: str
    job_name: Optional[str] = None
    status: JobStatus
    created_at: str
    completed_at: Optional[str] = None
    progress_percent: float = Field(0.0, ge=0.0, le=100.0)
    message: str = ""
    total_video_frames: Optional[int] = None
    extracted_frames: Optional[int] = None
    video_duration_sec: Optional[float] = None
    video_fps: Optional[float] = None
    video_resolution: Optional[str] = None
    geojson_path: Optional[str] = None
    geojson_url: Optional[str] = None
    download_zip_path: Optional[str] = None
    download_zip_url: Optional[str] = None
    output_dir: Optional[str] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    opencv_version: str
    active_jobs: int
    storage_used_bytes: int
    supported_video_formats: List[str]
    supported_image_formats: List[str]

class GeoJSONGeometry(BaseModel):
    type: str
    coordinates: Union[List[float], List[List[float]], List[List[List[float]]]]

class GeoJSONFeature(BaseModel):
    type: str = "Feature"
    geometry: GeoJSONGeometry
    properties: Dict[str, Any]

class GeoJSONFeatureCollection(BaseModel):
    type: str = "FeatureCollection"
    properties: Optional[Dict[str, Any]] = None
    features: List[GeoJSONFeature]
