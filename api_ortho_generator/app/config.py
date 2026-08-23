import os
import sys
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ensure Rasterio / GDAL uses its own bundled PROJ data rather than third-party system variables
try:
    import rasterio
    proj_dir = os.path.join(os.path.dirname(rasterio.__file__), "proj_data")
    gdal_dir = os.path.join(os.path.dirname(rasterio.__file__), "gdal_data")
    if os.path.exists(proj_dir):
        os.environ["PROJ_DATA"] = proj_dir
        os.environ["PROJ_LIB"] = proj_dir
        try:
            from rasterio._env import set_gdal_config
            set_gdal_config("PROJ_LIB", proj_dir)
            set_gdal_config("PROJ_DATA", proj_dir)
        except Exception:
            pass
    if os.path.exists(gdal_dir):
        os.environ["GDAL_DATA"] = gdal_dir
except Exception:
    pass

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ORTHO_", case_sensitive=False)
    
    APP_NAME: str = "OrthoGenerator 2D Orthomosaic Microservice"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Base directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    STORAGE_DIR: Path = BASE_DIR / "outputs"
    UPLOAD_DIR: Path = STORAGE_DIR / "uploads"
    JOBS_DIR: Path = STORAGE_DIR / "jobs"
    
    # Processing defaults
    DEFAULT_GSD_METERS: float = 0.05  # 5cm per pixel
    DEFAULT_BLENDING: str = "multiband"  # 'multiband', 'feather', 'average'
    DEFAULT_CRS: str = "AUTO_UTM"  # Automatically determine UTM Zone
    MAX_CONCURRENT_JOBS: int = 2

settings = Settings()

# Ensure directories exist
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.JOBS_DIR.mkdir(parents=True, exist_ok=True)
