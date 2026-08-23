import os
import shutil
import logging
from typing import Optional, Callable, Dict, Any, Tuple
from pathlib import Path
import pycolmap

from ..models.schemas import MatchingMethod

logger = logging.getLogger(__name__)

class SfMEngine:
    """
    Structure from Motion (SfM) engine using PyCOLMAP.
    Extracts SIFT features, matches stereo pairs, and executes incremental 3D reconstruction.
    """

    @classmethod
    def run_sfm(
        cls,
        image_dir: str,
        output_dir: str,
        matching_method: MatchingMethod = MatchingMethod.SEQUENTIAL,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Execute PyCOLMAP SfM pipeline on image_dir and export sparse 3D point cloud (.ply).
        """
        os.makedirs(output_dir, exist_ok=True)
        database_path = os.path.join(output_dir, "database.db")
        sparse_dir = os.path.join(output_dir, "sparse")
        os.makedirs(sparse_dir, exist_ok=True)

        if os.path.exists(database_path):
            try: os.remove(database_path)
            except Exception: pass

        # 1. Feature Extraction
        if progress_callback:
            progress_callback(10.0, "Extracting SIFT features with PyCOLMAP...")

        logger.info(f"Extracting SIFT features from {image_dir} to {database_path}")
        ext_opts = pycolmap.FeatureExtractionOptions()
        ext_opts.max_image_size = 2000
        ext_opts.num_threads = min(8, max(2, os.cpu_count() or 4))

        pycolmap.extract_features(
            database_path=database_path,
            image_path=image_dir,
            camera_mode=pycolmap.CameraMode.AUTO,
            extraction_options=ext_opts
        )

        # 2. Feature Matching
        if progress_callback:
            progress_callback(35.0, f"Matching stereo image features ({matching_method.value})...")

        logger.info(f"Performing {matching_method.value} feature matching...")
        match_opts = pycolmap.FeatureMatchingOptions()
        match_opts.num_threads = min(8, max(2, os.cpu_count() or 4))

        if matching_method == MatchingMethod.SEQUENTIAL:
            pycolmap.match_sequential(database_path=database_path, matching_options=match_opts)
        elif matching_method == MatchingMethod.EXHAUSTIVE:
            pycolmap.match_exhaustive(database_path=database_path, matching_options=match_opts)
        else:
            pycolmap.match_spatial(database_path=database_path, matching_options=match_opts)

        # 3. Incremental 3D Reconstruction
        if progress_callback:
            progress_callback(60.0, "Executing incremental 3D Bundle Adjustment and point triangulation...")

        logger.info(f"Running incremental reconstruction into {sparse_dir}...")
        pipe_opts = pycolmap.IncrementalPipelineOptions()
        pipe_opts.num_threads = min(8, max(2, os.cpu_count() or 4))

        reconstructions = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=image_dir,
            output_path=sparse_dir,
            options=pipe_opts
        )

        if not reconstructions:
            raise RuntimeError("SfM 3D Reconstruction failed to find valid camera tracks or 3D points.")

        best_rec = reconstructions[0]
        num_points = best_rec.num_points3D()
        num_reg_images = best_rec.num_reg_images()

        logger.info(f"SfM completed: {num_reg_images} images registered, {num_points} 3D points.")

        # 4. Export Sparse 3D Point Cloud PLY
        ply_path = os.path.join(output_dir, "point_cloud.ply")
        best_rec.export_PLY(ply_path)

        stats = {
            "num_registered_images": num_reg_images,
            "num_3d_points": num_points,
            "num_cameras": len(best_rec.cameras),
            "ply_path": ply_path
        }

        return ply_path, stats
