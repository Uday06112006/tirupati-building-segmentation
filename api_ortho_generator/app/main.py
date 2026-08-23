import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from .config import settings
from .routers import health_router, ortho_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("ortho_generator_api")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
# OrthoGenerator: 2D Drone Orthophoto & Orthomosaic Microservice 🛰️🗺️

High-performance microservice for generating georeferenced Orthophoto GeoTIFFs from aerial drone frames:
- **Fast 2D Georeferenced Stitching**: Projects aerial images using WGS-84 / UTM CRS with ground footprint calculation
- **Multi-Band & Distance Feathering**: Seamless distance-transform blending eliminates exposure seams and border artifacts
- **Standard GeoTIFF Output**: Produces RGBA GeoTIFFs (`.tif`) with exact geotransforms and spatial bounding boxes
- **GIS & Mapping Ready**: 100% compatible with QGIS, ArcGIS Pro, Google Earth, WebODM, Mapbox, and Leaflet.
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health_router, prefix=settings.API_PREFIX)
app.include_router(ortho_router, prefix=settings.API_PREFIX)

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)
