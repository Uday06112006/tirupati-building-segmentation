import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from .config import settings
from .routers import health_router, reconstruction_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("threed_generator_api")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
# OrthoGenerator: 3D Point Cloud & Surface Mesh Reconstruction Microservice 🛰️🗿

High-performance 3D photogrammetry microservice for aerial drone surveys:
- **PyCOLMAP SfM**: SIFT feature detection, sequential stereo matching, and bundle adjustment
- **3D Point Clouds**: Generates sparse/dense 3D point clouds in `.ply` format with color and camera poses
- **3D Surface Meshing**: Open3D Poisson & Ball-Pivoting surface reconstruction exporting textured `.obj` models
- **3D Viewing & CAD Ready**: 100% compatible with MeshLab, CloudCompare, Blender, Unreal Engine, and GIS 3D viewers.
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix=settings.API_PREFIX)
app.include_router(reconstruction_router, prefix=settings.API_PREFIX)

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8002, reload=True)
