import os
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable
from pathlib import Path

from .exif_reader import EXIFReader
from .ortho_engine import OrthoMosaicEngine
from .accurate_ortho import AccurateOrthoEngine
from .geotiff_writer import GeoTIFFWriter
from ..models.schemas import OrthoRequest, OrthoEngine

logger = logging.getLogger(__name__)

class OrthoPipeline:
    """
    Unified pipeline orchestrating EXIF inspection, coordinate projection,
    accurate photogrammetric true orthorectification, and GeoTIFF export.
    """

    @classmethod
    def run_pipeline(
        cls,
        frames_dir: str,
        output_dir: str,
        params: OrthoRequest,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute the full orthomosaic generation pipeline.
        """
        start_time = datetime.now(timezone.utc)
        os.makedirs(output_dir, exist_ok=True)

        if progress_callback:
            progress_callback(2.0, f"Scanning geotagged frames in {frames_dir}...")

        # 1. Scan and read EXIF
        images = EXIFReader.scan_directory(frames_dir, max_images=params.max_images)
        if not images:
            raise ValueError(f"No valid geotagged image frames found in: {frames_dir}")

        gsd_m = (params.target_gsd_cm or 5.0) / 100.0
        dsm_array = None
        engine_used = params.engine.value

        # 2. Engine Execution
        if params.engine == OrthoEngine.ACCURATE:
            try:
                rgb_mosaic, alpha_mask, dsm_array, spatial_meta = AccurateOrthoEngine.generate_accurate_ortho(
                    frames_dir=frames_dir,
                    images_info=images,
                    target_gsd_m=gsd_m,
                    target_crs=params.crs,
                    work_dir=os.path.join(output_dir, "_sfm_cache"),
                    progress_callback=progress_callback
                )
            except Exception as e:
                logger.warning(f"Accurate SfM engine failed ({e}), falling back to rapid planar georeferencing...")
                engine_used = "fast_fallback"
                rgb_mosaic, alpha_mask, spatial_meta = OrthoMosaicEngine.generate_mosaic(
                    images=images,
                    target_gsd_m=gsd_m,
                    blend_mode=params.blend_mode,
                    target_crs=params.crs,
                    progress_callback=progress_callback
                )
        else:
            rgb_mosaic, alpha_mask, spatial_meta = OrthoMosaicEngine.generate_mosaic(
                images=images,
                target_gsd_m=gsd_m,
                blend_mode=params.blend_mode,
                target_crs=params.crs,
                progress_callback=progress_callback
            )

        if progress_callback:
            progress_callback(96.0, "Writing georeferenced GeoTIFF and preview thumbnail...")

        # 3. Write Orthophoto GeoTIFF
        tif_filename = f"{params.job_name or 'orthomosaic'}.tif"
        tif_path = os.path.join(output_dir, tif_filename)
        GeoTIFFWriter.write_geotiff(
            rgb_array=rgb_mosaic,
            alpha_mask=alpha_mask,
            spatial_meta=spatial_meta,
            output_tif_path=tif_path,
            compression=params.compression
        )

        # 4. Write DSM GeoTIFF if available
        dsm_path = None
        if params.generate_dsm and dsm_array is not None:
            dsm_filename = f"{params.job_name or 'orthomosaic'}_dsm.tif"
            dsm_path = os.path.join(output_dir, dsm_filename)
            GeoTIFFWriter.write_dsm_geotiff(
                dsm_array=dsm_array,
                spatial_meta=spatial_meta,
                output_dsm_path=dsm_path,
                compression=params.compression
            )

        # 5. Write PNG Preview
        png_filename = f"{params.job_name or 'orthomosaic'}_preview.png"
        png_path = os.path.join(output_dir, png_filename)
        GeoTIFFWriter.write_preview_png(
            rgb_array=rgb_mosaic,
            alpha_mask=alpha_mask,
            output_png_path=png_path
        )

        if progress_callback:
            progress_callback(100.0, "Orthomosaic processing completed successfully.")

        end_time = datetime.now(timezone.utc)
        elapsed_sec = round((end_time - start_time).total_seconds(), 2)

        return {
            "status": "COMPLETED",
            "engine_used": engine_used,
            "spatial_meta": spatial_meta,
            "geotiff_path": tif_path,
            "dsm_path": dsm_path,
            "preview_path": png_path,
            "elapsed_seconds": elapsed_sec,
            "total_images": len(images)
        }
