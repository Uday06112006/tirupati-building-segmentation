import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from .config import settings
from .routers import health_router, extraction_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("ortho_microservice")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
# OrthoGenerator: Drone Video-to-Frame Geotagging & Telemetry Microservice 🛰️📸

High-performance microservice for photogrammetry, mapping, and GIS workflows:
- **Frame Extraction**: Extract high quality/lossless frames from drone videos (.MOV, .MP4, .AVI)
- **EXIF Geotagging**: Injects standard WGS-84 GPS coordinates (Latitude, Longitude, Altitude), camera orientation (Yaw, Pitch, Roll), Exposure, ISO, Shutter Speed, and Focal Length into image EXIF metadata
- **GeoJSON Flight Path**: Generates RFC 7946 GeoJSON FeatureCollections containing photo positions (Points) and the sequential flight path (LineString)
- **Ready for Photogrammetry**: Outputs are 100% compatible with WebODM, Agisoft Metashape, Pix4D, RealityCapture, QGIS, and ArcGIS.
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for cross-origin web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(health_router, prefix=settings.API_PREFIX)
app.include_router(extraction_router, prefix=settings.API_PREFIX)

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to OpenAPI interactive Swagger documentation."""
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
