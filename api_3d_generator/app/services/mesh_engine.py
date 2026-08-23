import os
import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np
import open3d as o3d

from ..models.schemas import MeshMethod

logger = logging.getLogger(__name__)

class MeshEngine:
    """
    3D Surface Mesh generator using Open3D.
    Generates Poisson or Ball-Pivoting triangle meshes from 3D point clouds.
    """

    @classmethod
    def generate_mesh_from_ply(
        cls,
        ply_path: str,
        output_obj_path: str,
        method: MeshMethod = MeshMethod.POISSON,
        poisson_depth: int = 8
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build a 3D surface mesh from point cloud PLY and save as OBJ.
        """
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"Point cloud file not found: {ply_path}")

        logger.info(f"Loading point cloud from: {ply_path}")
        pcd = o3d.io.read_point_cloud(ply_path)
        if len(pcd.points) < 4:
            raise ValueError(f"Point cloud contains too few points ({len(pcd.points)}) to mesh.")

        # 1. Statistical Outlier Removal
        pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        if len(pcd_clean.points) > 10:
            pcd = pcd_clean

        # 2. Normal Estimation
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=10)

        # 3. Surface Reconstruction
        if method == MeshMethod.BALL_PIVOTING:
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 3.0 * avg_dist
            radii = [radius, radius * 2, radius * 4]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )
        else:  # Poisson default
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=poisson_depth
            )
            # Filter low density outer bounding artifacts
            densities_arr = np.asarray(densities)
            if len(densities_arr) > 0:
                density_threshold = np.quantile(densities_arr, 0.05)
                vertices_to_remove = densities_arr < density_threshold
                mesh.remove_vertices_by_mask(vertices_to_remove)

        mesh.compute_vertex_normals()

        # 4. Save OBJ mesh
        os.makedirs(os.path.dirname(output_obj_path), exist_ok=True)
        o3d.io.write_triangle_mesh(output_obj_path, mesh)
        logger.info(f"Saved 3D mesh ({len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles) to {output_obj_path}")

        mesh_stats = {
            "num_vertices": len(mesh.vertices),
            "num_triangles": len(mesh.triangles),
            "mesh_path": output_obj_path
        }
        return output_obj_path, mesh_stats
