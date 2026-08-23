import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="THREED_", case_sensitive=False)
    
    APP_NAME: str = "OrthoGenerator 3D Reconstruction Microservice"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Base directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    STORAGE_DIR: Path = BASE_DIR / "outputs"
    UPLOAD_DIR: Path = STORAGE_DIR / "uploads"
    JOBS_DIR: Path = STORAGE_DIR / "jobs"
    
    # SfM / 3D defaults
    MAX_CONCURRENT_JOBS: int = 1  # 3D reconstruction is memory/compute intensive
    DEFAULT_SIFT_MAX_NUM_FEATURES: int = 4096
    DEFAULT_POISSON_DEPTH: int = 8

settings = Settings()

# Ensure directories exist
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.JOBS_DIR.mkdir(parents=True, exist_ok=True)
