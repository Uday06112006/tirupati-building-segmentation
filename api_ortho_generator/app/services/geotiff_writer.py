import os
import logging
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import ColorInterp

from ..models.schemas import CompressionType

logger = logging.getLogger(__name__)

class GeoTIFFWriter:
    """
    Exports orthomosaic arrays as standard Georeferenced GeoTIFFs (with EPSG/UTM CRS)
    and browser-viewable PNG previews.
    """

    @classmethod
    def write_geotiff(
        cls,
        rgb_array: np.ndarray,
        alpha_mask: np.ndarray,
        spatial_meta: Dict[str, Any],
        output_tif_path: str,
        compression: CompressionType = CompressionType.DEFLATE
    ) -> str:
        """
        Write georeferenced 4-band RGBA GeoTIFF file.
        """
        height, width = rgb_array.shape[:2]
        min_x = spatial_meta["min_x"]
        max_y = spatial_meta["max_y"]
        gsd_m = spatial_meta["gsd_m"]
        crs_str = spatial_meta["crs"]

        transform = from_origin(
            west=min_x,
            north=max_y,
            xsize=gsd_m,
            ysize=gsd_m
        )

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 4,  # RGBA
            "dtype": rasterio.uint8,
            "crs": crs_str,
            "transform": transform,
            "compress": compression.value.lower(),
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "photometric": "RGB"
        }

        os.makedirs(os.path.dirname(output_tif_path), exist_ok=True)

        with rasterio.open(output_tif_path, "w", **profile) as dst:
            dst.write(rgb_array[:, :, 0], 1)
            dst.write(rgb_array[:, :, 1], 2)
            dst.write(rgb_array[:, :, 2], 3)
            dst.write(alpha_mask, 4)
            
            dst.colorinterp = [
                ColorInterp.red,
                ColorInterp.green,
                ColorInterp.blue,
                ColorInterp.alpha
            ]

        logger.info(f"Exported True Orthophoto GeoTIFF: {output_tif_path}")
        return output_tif_path

    @classmethod
    def write_dsm_geotiff(
        cls,
        dsm_array: np.ndarray,
        spatial_meta: Dict[str, Any],
        output_dsm_path: str,
        compression: CompressionType = CompressionType.DEFLATE
    ) -> str:
        """
        Write georeferenced 1-band Float32 Digital Surface Model (DSM) GeoTIFF.
        """
        height, width = dsm_array.shape[:2]
        min_x = spatial_meta["min_x"]
        max_y = spatial_meta["max_y"]
        gsd_m = spatial_meta["gsd_m"]
        crs_str = spatial_meta["crs"]

        transform = from_origin(
            west=min_x,
            north=max_y,
            xsize=gsd_m,
            ysize=gsd_m
        )

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,  # 1 Band Elevation (Meters AMSL)
            "dtype": rasterio.float32,
            "crs": crs_str,
            "transform": transform,
            "compress": compression.value.lower(),
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "nodata": -9999.0
        }

        os.makedirs(os.path.dirname(output_dsm_path), exist_ok=True)

        with rasterio.open(output_dsm_path, "w", **profile) as dst:
            dst.write(dsm_array.astype(np.float32), 1)

        logger.info(f"Exported Digital Surface Model (DSM) GeoTIFF: {output_dsm_path}")
        return output_dsm_path

    @classmethod
    def write_preview_png(
        cls,
        rgb_array: np.ndarray,
        alpha_mask: np.ndarray,
        output_png_path: str,
        max_preview_dim: int = 1920
    ) -> str:
        """
        Generate lightweight PNG preview thumbnail for web display.
        """
        h, w = rgb_array.shape[:2]
        scale = min(1.0, max_preview_dim / max(h, w))
        
        rgba = np.dstack((rgb_array, alpha_mask))
        pil_img = Image.fromarray(rgba, "RGBA")
        
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
        pil_img.save(output_png_path, "PNG", optimize=True)
        logger.info(f"Generated preview thumbnail: {output_png_path}")
        return output_png_path
