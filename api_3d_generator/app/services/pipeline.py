import os
import zipfile
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable
from pathlib import Path

from .sfm_engine import SfMEngine
from .mesh_engine import MeshEngine
from ..models.schemas import ReconstructionRequest, MeshMethod

logger = logging.getLogger(__name__)

class ThreeDPipeline:
    """
    Unified pipeline executing 3D Structure from Motion (SfM) + 3D Mesh surface reconstruction.
    """

    @classmethod
    def run_pipeline(
        cls,
        frames_dir: str,
        output_dir: str,
        params: ReconstructionRequest,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute the end-to-end 3D reconstruction pipeline.
        """
        start_time = datetime.now(timezone.utc)
        os.makedirs(output_dir, exist_ok=True)

        # 1. Run SfM with PyCOLMAP
        ply_path, sfm_stats = SfMEngine.run_sfm(
            image_dir=frames_dir,
            output_dir=output_dir,
            matching_method=params.matching_method,
            progress_callback=progress_callback
        )

        mesh_obj_path = None
        mesh_stats = {"num_vertices": 0, "num_triangles": 0}

        # 2. Build 3D Surface Mesh
        if params.generate_mesh and params.mesh_method != MeshMethod.NONE:
            if progress_callback:
                progress_callback(80.0, "Reconstructing 3D Poisson surface mesh...")

            mesh_filename = f"{params.job_name or 'model'}_mesh.obj"
            mesh_obj_path = os.path.join(output_dir, mesh_filename)

            try:
                mesh_obj_path, mesh_stats = MeshEngine.generate_mesh_from_ply(
                    ply_path=ply_path,
                    output_obj_path=mesh_obj_path,
                    method=params.mesh_method,
                    poisson_depth=params.poisson_depth
                )
            except Exception as e:
                logger.warning(f"3D mesh generation encountered non-critical error: {e}")

        # 3. Create Downloadable ZIP package
        if progress_callback:
            progress_callback(95.0, "Packaging 3D models and point clouds into ZIP archive...")

        zip_filename = f"{params.job_name or '3d_reconstruction'}.zip"
        zip_path = os.path.join(output_dir, zip_filename)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            if os.path.exists(ply_path):
                zipf.write(ply_path, "point_cloud.ply")
            if mesh_obj_path and os.path.exists(mesh_obj_path):
                zipf.write(mesh_obj_path, os.path.basename(mesh_obj_path))

        if progress_callback:
            progress_callback(100.0, "3D Reconstruction completed successfully.")

        end_time = datetime.now(timezone.utc)
        elapsed_sec = round((end_time - start_time).total_seconds(), 2)

        return {
            "status": "COMPLETED",
            "sfm_stats": sfm_stats,
            "mesh_stats": mesh_stats,
            "pointcloud_path": ply_path,
            "mesh_path": mesh_obj_path,
            "zip_path": zip_path,
            "elapsed_seconds": elapsed_sec
        }
