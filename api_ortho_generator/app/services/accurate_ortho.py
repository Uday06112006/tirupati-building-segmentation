import os
import math
import logging
from typing import List, Tuple, Optional, Callable, Dict, Any
from pathlib import Path
import numpy as np
import cv2
import scipy.interpolate
import pycolmap

from .exif_reader import ImageGeoInfo, EXIFReader
from .coordinates import CoordinateService
from ..models.schemas import BlendMode

logger = logging.getLogger(__name__)

class AccurateOrthoEngine:
    """
    Survey-Grade Photogrammetric True Orthophoto & DSM Engine.
    Executes PyCOLMAP SfM Bundle Adjustment, Helmert/Umeyama GPS georeferencing,
    Delaunay DSM interpolation, and True Orthorectification via inverse camera ray-casting.
    """

    @staticmethod
    def umeyama_similarity_alignment(src_pts: np.ndarray, dst_pts: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Computes 3D similarity transformation (scale, R, translation) such that:
        dst = scale * (R @ src) + translation
        """
        assert src_pts.shape == dst_pts.shape
        N, dim = src_pts.shape

        src_mean = np.mean(src_pts, axis=0)
        dst_mean = np.mean(dst_pts, axis=0)

        src_centered = src_pts - src_mean
        dst_centered = dst_pts - dst_mean

        H = src_centered.T @ dst_centered / N
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[dim - 1, :] *= -1
            R = Vt.T @ U.T

        src_var = np.var(src_pts, axis=0).sum()
        scale = (1.0 / max(1e-9, src_var)) * np.sum(S)
        translation = dst_mean - scale * (R @ src_mean)

        return float(scale), R, translation

    @classmethod
    def generate_accurate_ortho(
        cls,
        frames_dir: str,
        images_info: List[ImageGeoInfo],
        target_gsd_m: float = 0.05,
        target_crs: str = "AUTO",
        work_dir: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Execute full photogrammetric true orthorectification pipeline using COLMAP.
        Returns (rgb_mosaic, alpha_mask, dsm_elevation_m, spatial_meta).
        """
        if not work_dir:
            work_dir = os.path.join(frames_dir, "_sfm_temp")
        os.makedirs(work_dir, exist_ok=True)

        database_path = os.path.join(work_dir, "database.db")
        sparse_dir = os.path.join(work_dir, "sparse")
        os.makedirs(sparse_dir, exist_ok=True)

        if os.path.exists(database_path):
            try: os.remove(database_path)
            except Exception: pass

        # 1. Feature Extraction (SIFT with memory safety options)
        if progress_callback:
            progress_callback(5.0, "Extracting high-precision SIFT features with PyCOLMAP...")

        logger.info(f"Extracting SIFT features into {database_path}")
        ext_opts = pycolmap.FeatureExtractionOptions()
        ext_opts.max_image_size = 2000
        ext_opts.num_threads = min(8, max(2, os.cpu_count() or 4))

        pycolmap.extract_features(
            database_path=database_path,
            image_path=frames_dir,
            camera_mode=pycolmap.CameraMode.AUTO,
            extraction_options=ext_opts
        )

        # 2. Sequential Stereo Matching
        if progress_callback:
            progress_callback(25.0, "Executing COLMAP sequential stereo feature matching...")

        logger.info("Matching sequential image pairs with COLMAP...")
        match_opts = pycolmap.FeatureMatchingOptions()
        match_opts.num_threads = min(8, max(2, os.cpu_count() or 4))
        pycolmap.match_sequential(database_path=database_path, matching_options=match_opts)

        # 3. Incremental Structure from Motion & Bundle Adjustment
        if progress_callback:
            progress_callback(45.0, "Executing COLMAP Bundle Adjustment & 3D triangulation...")

        logger.info("Running incremental SfM reconstruction with COLMAP...")
        pipe_opts = pycolmap.IncrementalPipelineOptions()
        pipe_opts.num_threads = min(8, max(2, os.cpu_count() or 4))

        reconstructions = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=frames_dir,
            output_path=sparse_dir,
            options=pipe_opts
        )

        if not reconstructions:
            raise RuntimeError("SfM 3D Reconstruction failed to find valid camera tracks.")

        rec = reconstructions[0]
        num_points = rec.num_points3D()
        num_reg = rec.num_reg_images()
        logger.info(f"COLMAP SfM Success: {num_reg} cameras registered, {num_points} 3D points triangulated.")

        # Determine UTM CRS
        if target_crs.upper() in ("AUTO", "AUTO_UTM"):
            median_lon = sum(img.longitude for img in images_info) / len(images_info)
            median_lat = sum(img.latitude for img in images_info) / len(images_info)
            target_crs = CoordinateService.get_utm_epsg_for_lon_lat(median_lon, median_lat)

        transformer = CoordinateService.get_transformer("EPSG:4326", target_crs)

        # 4. GPS Georeferencing Alignment (Umeyama 3D Similarity Transform)
        if progress_callback:
            progress_callback(60.0, f"Aligning COLMAP 3D model to real-world {target_crs} coordinates...")

        info_map = {img.filename: img for img in images_info}
        src_centers = []
        dst_centers = []

        for img_id, im in rec.images.items():
            if im.name in info_map:
                g_info = info_map[im.name]
                c_sfm = im.projection_center()
                ux, uy = transformer.transform(g_info.longitude, g_info.latitude)
                uz = g_info.altitude_m
                src_centers.append(c_sfm)
                dst_centers.append([ux, uy, uz])

        src_arr = np.array(src_centers, dtype=np.float64)
        dst_arr = np.array(dst_centers, dtype=np.float64)

        if len(src_arr) >= 3:
            scale, R_align, T_align = cls.umeyama_similarity_alignment(src_arr, dst_arr)
        else:
            scale, R_align, T_align = 1.0, np.eye(3), np.mean(dst_arr, axis=0) - np.mean(src_arr, axis=0)

        # Transform 3D SfM points to UTM coordinates
        points_3d_utm = []
        for p_id, pt in rec.points3D.items():
            p_sfm = pt.xyz
            p_utm = scale * (R_align @ p_sfm) + T_align
            points_3d_utm.append(p_utm)

        points_3d_utm = np.array(points_3d_utm, dtype=np.float64)

        # 5. Build Digital Surface Model (DSM) Grid
        if progress_callback:
            progress_callback(70.0, "Interpolating Digital Surface Model (DSM) elevation grid...")

        min_x = float(np.percentile(points_3d_utm[:, 0], 1) - 5.0)
        max_x = float(np.percentile(points_3d_utm[:, 0], 99) + 5.0)
        min_y = float(np.percentile(points_3d_utm[:, 1], 1) - 5.0)
        max_y = float(np.percentile(points_3d_utm[:, 1], 99) + 5.0)

        grid_w = int(math.ceil((max_x - min_x) / target_gsd_m))
        grid_h = int(math.ceil((max_y - min_y) / target_gsd_m))

        max_dim = 16000
        if grid_w > max_dim or grid_h > max_dim:
            s_adj = max_dim / max(grid_w, grid_h)
            target_gsd_m = target_gsd_m / s_adj
            grid_w = int(math.ceil((max_x - min_x) / target_gsd_m))
            grid_h = int(math.ceil((max_y - min_y) / target_gsd_m))

        # 2D Grid Coordinate Arrays (Top-Left origin standard: X increases, Y decreases)
        gx = np.linspace(min_x, max_x, grid_w, dtype=np.float32)
        gy = np.linspace(max_y, min_y, grid_h, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(gx, gy)

        # Interpolate DSM elevation Z(X, Y)
        dsm_linear = scipy.interpolate.griddata(
            points=points_3d_utm[:, :2],
            values=points_3d_utm[:, 2],
            xi=(grid_x, grid_y),
            method="linear"
        )
        dsm_nearest = scipy.interpolate.griddata(
            points=points_3d_utm[:, :2],
            values=points_3d_utm[:, 2],
            xi=(grid_x, grid_y),
            method="nearest"
        )
        dsm = np.where(np.isnan(dsm_linear), dsm_nearest, dsm_linear)
        median_alt = float(np.median(points_3d_utm[:, 2]))
        dsm = np.where(np.isnan(dsm), median_alt, dsm).astype(np.float32)

        # 6. True Orthorectification via Inverse Camera Ray-Casting (Memory-Optimized Subgrids)
        if progress_callback:
            progress_callback(80.0, "Executing photogrammetric true orthorectification & nadir blending...")

        accum_rgb = np.zeros((grid_h, grid_w, 3), dtype=np.float32)
        accum_weight = np.zeros((grid_h, grid_w), dtype=np.float32)

        inv_scale = 1.0 / scale
        total_cams = len([im for im in rec.images.values() if im.has_pose])
        processed_cams = 0

        for img_id, im in rec.images.items():
            if not im.has_pose:
                continue

            img_file = os.path.join(frames_dir, im.name)
            if not os.path.exists(img_file):
                continue

            img_bgr = cv2.imread(img_file)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            im_h, im_w = img_rgb.shape[:2]

            cam = rec.cameras[im.camera_id]
            cam_from_world = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
            R_cw = cam_from_world.rotation.matrix()
            t_cw = cam_from_world.translation

            # Determine ground bounding box for this camera to avoid huge global memory allocations
            c_sfm = im.projection_center()
            c_utm = scale * (R_align @ c_sfm) + T_align
            alt_m = max(10.0, float(c_utm[2] - np.median(dsm)))
            fov_rad = 2.0 * math.atan2(im_w / 2.0, cam.focal_length_x)
            r_cov_m = alt_m * math.tan(fov_rad / 2.0) * 2.0 + 15.0

            x0_idx = max(0, int((c_utm[0] - r_cov_m - min_x) / target_gsd_m))
            x1_idx = min(grid_w, int(math.ceil((c_utm[0] + r_cov_m - min_x) / target_gsd_m)) + 1)
            y0_idx = max(0, int((max_y - (c_utm[1] + r_cov_m)) / target_gsd_m))
            y1_idx = min(grid_h, int(math.ceil((max_y - (c_utm[1] - r_cov_m)) / target_gsd_m)) + 1)

            if x1_idx <= x0_idx or y1_idx <= y0_idx:
                continue

            sub_x = grid_x[y0_idx:y1_idx, x0_idx:x1_idx]
            sub_y = grid_y[y0_idx:y1_idx, x0_idx:x1_idx]
            sub_dsm = dsm[y0_idx:y1_idx, x0_idx:x1_idx]
            sub_h, sub_w = sub_x.shape

            sub_pts_utm = np.stack([sub_x, sub_y, sub_dsm], axis=-1)
            pts_sfm = (sub_pts_utm - T_align) @ R_align * inv_scale

            P_cam = (pts_sfm @ R_cw.T) + t_cw
            Z_cam = P_cam[:, :, 2]

            front_mask = Z_cam > 0.5
            if not np.any(front_mask):
                continue

            X_norm = np.zeros((sub_h, sub_w), dtype=np.float32)
            Y_norm = np.zeros((sub_h, sub_w), dtype=np.float32)
            X_norm[front_mask] = P_cam[:, :, 0][front_mask] / Z_cam[front_mask]
            Y_norm[front_mask] = P_cam[:, :, 1][front_mask] / Z_cam[front_mask]

            fx = cam.focal_length_x
            fy = cam.focal_length_y
            cx = cam.principal_point_x
            cy = cam.principal_point_y

            u_px = fx * X_norm + cx
            v_px = fy * Y_norm + cy

            in_frame = (
                front_mask &
                (u_px >= 2.0) & (u_px <= (im_w - 3.0)) &
                (v_px >= 2.0) & (v_px <= (im_h - 3.0))
            )

            if not np.any(in_frame):
                continue

            map_x = np.where(in_frame, u_px, -1.0).astype(np.float32)
            map_y = np.where(in_frame, v_px, -1.0).astype(np.float32)

            sampled_rgb = cv2.remap(
                img_rgb,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

            dist_3d = np.linalg.norm(P_cam, axis=-1) + 1e-6
            cos_nadir = np.clip(Z_cam / dist_3d, 0.0, 1.0)
            weight = np.where(in_frame, (cos_nadir ** 3).astype(np.float32), 0.0)

            u_norm_edge = np.minimum(u_px / (im_w / 2.0), (im_w - u_px) / (im_w / 2.0))
            v_norm_edge = np.minimum(v_px / (im_h / 2.0), (im_h - v_px) / (im_h / 2.0))
            edge_weight = np.clip(np.minimum(u_norm_edge, v_norm_edge), 0.0, 1.0)
            weight *= edge_weight

            weight_3d = np.repeat(weight[:, :, np.newaxis], 3, axis=2)
            accum_rgb[y0_idx:y1_idx, x0_idx:x1_idx] += sampled_rgb.astype(np.float32) * weight_3d
            accum_weight[y0_idx:y1_idx, x0_idx:x1_idx] += weight

            processed_cams += 1
            if progress_callback and processed_cams % 5 == 0:
                pct = 80.0 + (processed_cams / max(1, total_cams)) * 14.0
                progress_callback(pct, f"Orthorectifying frame {processed_cams}/{total_cams}...")

        if progress_callback:
            progress_callback(95.0, "Normalizing true orthophoto and preparing GeoTIFF metadata...")

        valid_mask = accum_weight > 1e-4
        final_rgb = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for c in range(3):
            chan = np.zeros((grid_h, grid_w), dtype=np.float32)
            chan[valid_mask] = accum_rgb[:, :, c][valid_mask] / accum_weight[valid_mask]
            final_rgb[:, :, c] = np.clip(chan, 0, 255).astype(np.uint8)

        final_alpha = (valid_mask * 255).astype(np.uint8)

        # Inverse bounds in WGS-84
        inv_transformer = CoordinateService.get_transformer(target_crs, "EPSG:4326")
        min_lon, min_lat = inv_transformer.transform(min_x, min_y)
        max_lon, max_lat = inv_transformer.transform(max_x, max_y)

        spatial_meta = {
            "crs": target_crs,
            "min_x": float(min_x),
            "max_y": float(max_y),
            "max_x": float(max_x),
            "min_y": float(min_y),
            "width_px": grid_w,
            "height_px": grid_h,
            "gsd_m": float(target_gsd_m),
            "gsd_cm": round(target_gsd_m * 100.0, 2),
            "bounds_utm": [round(min_x, 2), round(min_y, 2), round(max_x, 2), round(max_y, 2)],
            "bounds_wgs84": [round(min_lon, 6), round(min_lat, 6), round(max_lon, 6), round(max_lat, 6)],
            "num_sfm_points": num_points,
            "num_registered_cameras": num_reg,
            "dsm_min_elevation": float(np.min(dsm)),
            "dsm_max_elevation": float(np.max(dsm)),
            "total_images_processed": len(images_info)
        }

        logger.info(f"Photogrammetric True Orthophoto generated: {grid_w}x{grid_h} px at {spatial_meta['gsd_cm']} cm/px")
        return final_rgb, final_alpha, dsm, spatial_meta
