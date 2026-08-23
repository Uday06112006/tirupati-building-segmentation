"""
OrthoGenerator 3D Reconstruction CLI & API Demo Client
"""
import os
import sys
import time
import argparse
import requests
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.services.pipeline import ThreeDPipeline
from app.models.schemas import ReconstructionRequest, MatchingMethod, MeshMethod

def run_direct_pipeline(frames_dir: str, output_dir: str, matching: str = "sequential", mesh: bool = True):
    """Execute 3D reconstruction directly."""
    print("=" * 70)
    print("Running OrthoGenerator 3D Reconstruction Pipeline (Direct)")
    print("=" * 70)
    print(f"Frames Directory: {frames_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Matching Method:  {matching}")
    print(f"Generate Mesh:    {mesh}")

    params = ReconstructionRequest(
        frames_dir=frames_dir,
        matching_method=MatchingMethod(matching),
        generate_mesh=mesh,
        mesh_method=MeshMethod.POISSON,
        poisson_depth=8,
        job_name="3d_model_output"
    )

    def progress_cb(pct, msg):
        print(f"[{pct:5.1f}%] {msg}")

    result = ThreeDPipeline.run_pipeline(
        frames_dir=frames_dir,
        output_dir=output_dir,
        params=params,
        progress_callback=progress_cb
    )

    sfm = result["sfm_stats"]
    mesh_st = result["mesh_stats"]
    print("\n" + "=" * 70)
    print("3D Reconstruction Completed Successfully!")
    print("=" * 70)
    print(f"Registered Images:     {sfm['num_registered_images']}")
    print(f"Reconstructed Points:  {sfm['num_3d_points']}")
    print(f"Mesh Triangles:        {mesh_st.get('num_triangles', 0)}")
    print(f"Point Cloud (.ply):    {result['pointcloud_path']}")
    print(f"Mesh (.obj):           {result['mesh_path']}")
    print(f"Packaged ZIP:          {result['zip_path']}")
    print(f"Processing Time:       {result['elapsed_seconds']} seconds\n")
    return result

def run_api_client(api_url: str, frames_dir: str):
    """Execute via REST API."""
    print("=" * 70)
    print(f"Calling 3D Reconstruction API at: {api_url}")
    print("=" * 70)

    # 1. Health check
    h = requests.get(f"{api_url}/api/v1/health").json()
    print(f"Health Status: {h['status']} | PyCOLMAP: {h['pycolmap_version']} | CUDA: {h['cuda_available']}")

    # 2. Trigger job
    payload = {
        "frames_dir": os.path.abspath(frames_dir),
        "matching_method": "sequential",
        "generate_mesh": True,
        "mesh_method": "poisson",
        "poisson_depth": 8,
        "job_name": "api_3d_job"
    }

    resp = requests.post(f"{api_url}/api/v1/3d/reconstruct-from-path?async_mode=true", json=payload).json()
    job_id = resp["job_id"]
    print(f"Job initiated with ID: {job_id}")

    # 3. Poll
    while True:
        st = requests.get(f"{api_url}/api/v1/3d/jobs/{job_id}").json()
        print(f"[{st.get('progress_percent', 0):5.1f}%] Status: {st['status']} | {st.get('message', '')}")
        if st["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(2.0)

    if st["status"] == "COMPLETED":
        print("\n3D Reconstruction Job Finished Successfully!")
        print(f"Point Cloud PLY URL: {api_url}{st['pointcloud_ply_url']}")
        print(f"Mesh OBJ URL:        {api_url}{st['mesh_obj_url']}")
        print(f"Full ZIP Download:   {api_url}{st['download_zip_url']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OrthoGenerator 3D Reconstruction CLI & API")
    parser.add_argument("--frames", default="D:/ortho_generator/api_video_ortho/outputs/sample_run/frames", help="Folder of geotagged frames")
    parser.add_argument("--output", default="D:/ortho_generator/api_3d_generator/outputs/sample_3d", help="Output directory")
    parser.add_argument("--matching", default="sequential", choices=["sequential", "exhaustive", "spatial"], help="Matching method")
    parser.add_argument("--no-mesh", action="store_true", help="Skip 3D mesh generation")
    parser.add_argument("--api", action="store_true", help="Use REST API")
    parser.add_argument("--api-url", default="http://localhost:8002", help="API URL")

    args = parser.parse_args()

    if args.api:
        run_api_client(args.api_url, args.frames)
    else:
        run_direct_pipeline(args.frames, args.output, matching=args.matching, mesh=not args.no_mesh)
