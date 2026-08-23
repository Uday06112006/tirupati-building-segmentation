import os
import pytest
import numpy as np
from PIL import Image
import piexif
from fastapi.testclient import TestClient

from app.main import app
from app.services.exif_reader import EXIFReader, ImageGeoInfo
from app.services.coordinates import CoordinateService
from app.services.ortho_engine import OrthoMosaicEngine
from app.services.geotiff_writer import GeoTIFFWriter
from app.models.schemas import BlendMode, CompressionType, OrthoRequest

client = TestClient(app)

def test_health_endpoint():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "HEALTHY"
    assert "supported_blend_modes" in data

def test_utm_calculation():
    # Test coordinates in South India (Lon ~79.56, Lat ~13.40) -> UTM Zone 44N
    epsg = CoordinateService.get_utm_epsg_for_lon_lat(79.563553, 13.398962)
    assert epsg == "EPSG:32644"

    # Test coordinate in Paris (Lon ~2.35, Lat ~48.85) -> UTM Zone 31N
    epsg_paris = CoordinateService.get_utm_epsg_for_lon_lat(2.35, 48.85)
    assert epsg_paris == "EPSG:32631"

def test_ground_footprint():
    info = ImageGeoInfo(
        file_path="dummy.jpg",
        filename="dummy.jpg",
        width=3840,
        height=2160,
        latitude=13.398962,
        longitude=79.563553,
        altitude_m=200.0,
        focal_length_mm=28.0,
        sensor_width_mm=6.4,
        sensor_height_mm=4.8
    )
    gw, gh, gsd = CoordinateService.calculate_ground_footprint(info)
    assert gw > 0 and gh > 0
    assert gsd > 0
    assert abs(gw - (200.0 * 6.4 / 28.0)) < 1e-3

def test_geotiff_export(tmp_path):
    # Dummy RGB + Alpha
    h, w = 100, 100
    rgb = np.ones((h, w, 3), dtype=np.uint8) * 128
    alpha = np.ones((h, w), dtype=np.uint8) * 255
    spatial_meta = {
        "crs": "EPSG:32644",
        "min_x": 344400.0,
        "max_y": 1481800.0,
        "gsd_m": 0.5,
        "width_px": w,
        "height_px": h
    }
    
    tif_path = str(tmp_path / "test_ortho.tif")
    png_path = str(tmp_path / "test_preview.png")
    
    GeoTIFFWriter.write_geotiff(rgb, alpha, spatial_meta, tif_path, CompressionType.DEFLATE)
    GeoTIFFWriter.write_preview_png(rgb, alpha, png_path)
    
    assert os.path.exists(tif_path)
    assert os.path.exists(png_path)
    assert os.path.getsize(tif_path) > 0
