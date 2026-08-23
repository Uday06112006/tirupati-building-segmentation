import math
import logging
from typing import List, Tuple, Optional, Callable, Dict, Any
import numpy as np
import cv2
from PIL import Image

from .exif_reader import ImageGeoInfo
from .coordinates import CoordinateService
from ..models.schemas import BlendMode

logger = logging.getLogger(__name__)

class OrthoMosaicEngine:
    """
    High-performance 2D Orthomosaic generator.
    Projects, aligns, and blends aerial drone frames into a continuous georeferenced orthophoto.
    """

    @classmethod
    def generate_mosaic(
        cls,
        images: List[ImageGeoInfo],
        target_gsd_m: float = 0.05,  # 5 cm/pixel
        blend_mode: BlendMode = BlendMode.MULTIBAND,
        target_crs: str = "AUTO",
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Generate orthomosaic RGB image and alpha mask.
        Returns (rgb_mosaic_uint8, alpha_mask_uint8, spatial_metadata).
        """
        if not images:
            raise ValueError("No images provided for orthomosaic generation.")

        total_images = len(images)
        if progress_callback:
            progress_callback(5.0, f"Calculating spatial extent for {total_images} frames...")

        # 1. Compute bounding box and UTM CRS
        min_x, min_y, max_x, max_y, crs = CoordinateService.compute_projected_bounds(images, target_crs)
        transformer = CoordinateService.get_transformer("EPSG:4326", crs)

        # 2. Compute canvas pixel dimensions
        width_m = max_x - min_x
        height_m = max_y - min_y
        
        canvas_w = int(math.ceil(width_m / target_gsd_m))
        canvas_h = int(math.ceil(height_m / target_gsd_m))

        # Memory safety cap (e.g. max 16000x16000)
        max_dim = 16000
        if canvas_w > max_dim or canvas_h > max_dim:
            scale_factor = max_dim / max(canvas_w, canvas_h)
            target_gsd_m = target_gsd_m / scale_factor
            canvas_w = int(math.ceil(width_m / target_gsd_m))
            canvas_h = int(math.ceil(height_m / target_gsd_m))
            logger.info(f"Adjusted GSD to {target_gsd_m:.4f} m/px to fit memory bounds: {canvas_w}x{canvas_h}")

        logger.info(f"Allocating mosaic canvas: {canvas_w}x{canvas_h} pixels at GSD {target_gsd_m*100:.2f} cm/px")

        # Accumulator arrays (float32 for seamless blending)
        accum_rgb = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        accum_weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        # Process each image
        for idx, img_info in enumerate(images):
            if progress_callback:
                pct = 10.0 + (idx / total_images) * 80.0
                progress_callback(pct, f"Projecting & blending frame {idx+1}/{total_images} ({img_info.filename})...")

            # Load image
            img_bgr = cv2.imread(img_info.file_path)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            # Projected UTM Center
            cx, cy = transformer.transform(img_info.longitude, img_info.latitude)
            
            # Ground dimensions & nominal GSD
            gw, gh, nominal_gsd = CoordinateService.calculate_ground_footprint(img_info)
            scale = nominal_gsd / target_gsd_m

            # Canvas center in pixel coordinates
            pcx = (cx - min_x) / target_gsd_m
            pcy = (max_y - cy) / target_gsd_m

            # Rotation (Yaw)
            # In image coordinates, Y is down (inverted relative to UTM North)
            yaw_rad = math.radians(img_info.yaw_deg)
            cos_a = math.cos(yaw_rad)
            sin_a = math.sin(yaw_rad)

            # Transformation matrix: Scale, Rotation, Translation
            # Maps point (x, y) in source image to canvas (px, py)
            alpha = scale * cos_a
            beta = scale * sin_a

            src_center_x = w / 2.0
            src_center_y = h / 2.0

            tx = pcx - (alpha * src_center_x - beta * src_center_y)
            ty = pcy - (beta * src_center_x + alpha * src_center_y)

            M = np.array([
                [alpha, -beta, tx],
                [beta,  alpha, ty]
            ], dtype=np.float32)

            # Warp image to canvas coordinates
            warped_rgb = cv2.warpAffine(
                img_rgb,
                M,
                (canvas_w, canvas_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

            # Create binary mask of warped image
            binary_mask = cv2.warpAffine(
                np.ones((h, w), dtype=np.uint8) * 255,
                M,
                (canvas_w, canvas_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

            if blend_mode in (BlendMode.MULTIBAND, BlendMode.FEATHER):
                # Distance transform for smooth feathering
                dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
                max_dist = dist.max()
                if max_dist > 0:
                    weight = dist / max_dist
                else:
                    weight = (binary_mask > 0).astype(np.float32)
            else:
                weight = (binary_mask > 0).astype(np.float32)

            # Accumulate weighted RGB
            weight_3d = np.repeat(weight[:, :, np.newaxis], 3, axis=2)
            accum_rgb += warped_rgb.astype(np.float32) * weight_3d
            accum_weight += weight

        if progress_callback:
            progress_callback(92.0, "Normalizing blended orthomosaic and generating transparency mask...")

        # Normalize accumulated RGB by total weights
        valid_mask = accum_weight > 1e-4
        final_rgb = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        for c in range(3):
            chan = np.zeros((canvas_h, canvas_w), dtype=np.float32)
            chan[valid_mask] = accum_rgb[:, :, c][valid_mask] / accum_weight[valid_mask]
            final_rgb[:, :, c] = np.clip(chan, 0, 255).astype(np.uint8)

        final_alpha = (valid_mask * 255).astype(np.uint8)

        # Compute inverse bounds in WGS-84
        inv_transformer = CoordinateService.get_transformer(crs, "EPSG:4326")
        min_lon, min_lat = inv_transformer.transform(min_x, min_y)
        max_lon, max_lat = inv_transformer.transform(max_x, max_y)

        spatial_meta = {
            "crs": crs,
            "min_x": min_x,
            "max_y": max_y,
            "max_x": max_x,
            "min_y": min_y,
            "width_px": canvas_w,
            "height_px": canvas_h,
            "gsd_m": target_gsd_m,
            "gsd_cm": round(target_gsd_m * 100.0, 2),
            "bounds_utm": [round(min_x, 2), round(min_y, 2), round(max_x, 2), round(max_y, 2)],
            "bounds_wgs84": [round(min_lon, 6), round(min_lat, 6), round(max_lon, 6), round(max_lat, 6)],
            "total_images_processed": total_images
        }

        logger.info(f"Orthomosaic generation completed: {canvas_w}x{canvas_h} px, GSD {spatial_meta['gsd_cm']} cm/px")
        return final_rgb, final_alpha, spatial_meta
