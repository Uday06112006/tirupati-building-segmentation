import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ORTHO_", case_sensitive=False)
    
    APP_NAME: str = "OrthoGenerator Video-to-Frame Microservice"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Base directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    STORAGE_DIR: Path = BASE_DIR / "outputs"
    UPLOAD_DIR: Path = STORAGE_DIR / "uploads"
    JOBS_DIR: Path = STORAGE_DIR / "jobs"
    
    # Defaults
    DEFAULT_JPEG_QUALITY: int = 100
    DEFAULT_FRAME_STEP: int = 1
    MAX_UPLOAD_SIZE_MB: int = 4096  # 4GB max upload
    JOB_EXPIRY_HOURS: int = 24
    
    # Threading & workers
    MAX_CONCURRENT_JOBS: int = 4

settings = Settings()

# Ensure directories exist
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.JOBS_DIR.mkdir(parents=True, exist_ok=True)
