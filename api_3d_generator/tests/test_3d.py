import os
import pytest
import numpy as np
import open3d as o3d
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import MeshMethod, MatchingMethod, ReconstructionRequest
from app.services.mesh_engine import MeshEngine

client = TestClient(app)

def test_health_endpoint():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "HEALTHY"
    assert "pycolmap_version" in data
    assert "open3d_version" in data

def test_mesh_generation_from_pointcloud(tmp_path):
    # Create synthetic sphere point cloud with normals
    pcd = o3d.geometry.PointCloud()
    pts = np.random.randn(200, 3)
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.normals = o3d.utility.Vector3dVector(pts / np.linalg.norm(pts, axis=1, keepdims=True))

    ply_path = str(tmp_path / "test_points.ply")
    obj_path = str(tmp_path / "test_model.obj")
    o3d.io.write_point_cloud(ply_path, pcd)

    out_mesh_path, stats = MeshEngine.generate_mesh_from_ply(
        ply_path=ply_path,
        output_obj_path=obj_path,
        method=MeshMethod.POISSON,
        poisson_depth=6
    )

    assert os.path.exists(out_mesh_path)
    assert stats["num_vertices"] > 0
    assert stats["num_triangles"] > 0

def test_list_jobs_endpoint():
    resp = client.get("/api/v1/3d/jobs")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
