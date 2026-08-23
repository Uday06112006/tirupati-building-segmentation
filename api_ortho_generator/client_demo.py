"""
OrthoGenerator 2D Orthophoto CLI & API Demo Client (High Accuracy & Fast Modes)
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

from app.services.pipeline import OrthoPipeline
from app.models.schemas import OrthoRequest, OrthoEngine, BlendMode, CompressionType

def run_direct_pipeline(frames_dir: str, output_dir: str, engine: str = "accurate", gsd_cm: float = 5.0, blend_mode: str = "multiband"):
    """Execute 2D orthomosaic generation directly."""
    print("=" * 70)
    print(f"Running OrthoGenerator 2D Orthophoto Pipeline ({engine.upper()} Engine)")
    print("=" * 70)
    print(f"Frames Directory: {frames_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Target GSD:       {gsd_cm} cm/pixel")
    print(f"Engine:           {engine} (accurate = SfM + DSM true orthorectification, fast = planar)")
    print(f"Blend Mode:       {blend_mode}")

    params = OrthoRequest(
        frames_dir=frames_dir,
        engine=OrthoEngine(engine),
        target_gsd_cm=gsd_cm,
        blend_mode=BlendMode(blend_mode),
        generate_dsm=True,
        compression=CompressionType.DEFLATE,
        crs="AUTO",
        job_name="survey_orthophoto"
    )

    def progress_cb(pct, msg):
        print(f"[{pct:5.1f}%] {msg}")

    result = OrthoPipeline.run_pipeline(
        frames_dir=frames_dir,
        output_dir=output_dir,
        params=params,
        progress_callback=progress_cb
    )

    meta = result["spatial_meta"]
    print("\n" + "=" * 70)
    print("Orthophoto Generation Completed Successfully!")
    print("=" * 70)
    print(f"Engine Used:            {result['engine_used']}")
    print(f"Total Frames Processed: {result['total_images']}")
    if "num_sfm_points" in meta:
        print(f"Triangulated 3D Points: {meta['num_sfm_points']}")
    print(f"Canvas Dimensions:      {meta['width_px']} x {meta['height_px']} pixels")
    print(f"Ground Resolution:      {meta['gsd_cm']} cm/pixel")
    print(f"Coordinate System:      {meta['crs']}")
    print(f"UTM Bounds (meters):    {meta['bounds_utm']}")
    print(f"WGS-84 Bounds:          {meta['bounds_wgs84']}")
    print(f"True Orthophoto GeoTIFF: {result['geotiff_path']}")
    if result.get("dsm_path"):
        print(f"DSM Elevation GeoTIFF:   {result['dsm_path']}")
    print(f"Preview PNG File:       {result['preview_path']}")
    print(f"Processing Time:        {result['elapsed_seconds']} seconds\n")
    return result

def run_api_client(api_url: str, frames_dir: str, engine: str = "accurate", gsd_cm: float = 5.0):
    """Execute via REST API."""
    print("=" * 70)
    print(f"Calling 2D Orthophoto API at: {api_url}")
    print("=" * 70)

    # 1. Health check
    h = requests.get(f"{api_url}/api/v1/health").json()
    print(f"Health Status: {h['status']} | Version: {h['version']} | Supported Engines: {h['supported_engines']}")

    # 2. Trigger job
    payload = {
        "frames_dir": os.path.abspath(frames_dir),
        "engine": engine,
        "target_gsd_cm": gsd_cm,
        "blend_mode": "multiband",
        "generate_dsm": True,
        "compression": "DEFLATE",
        "crs": "AUTO",
        "job_name": "api_survey_ortho"
    }

    resp = requests.post(f"{api_url}/api/v1/ortho/generate-from-path?async_mode=true", json=payload).json()
    job_id = resp["job_id"]
    print(f"Job initiated with ID: {job_id}")

    # 3. Poll
    while True:
        st = requests.get(f"{api_url}/api/v1/ortho/jobs/{job_id}").json()
        print(f"[{st.get('progress_percent', 0):5.1f}%] Status: {st['status']} | {st.get('message', '')}")
        if st["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(2.0)

    if st["status"] == "COMPLETED":
        print("\nAPI Orthophoto Job Finished Successfully!")
        print(f"Engine Used:          {st['engine_used']}")
        print(f"GeoTIFF Download URL: {api_url}{st['geotiff_url']}")
        if st.get("dsm_url"):
            print(f"DSM GeoTIFF URL:      {api_url}{st['dsm_url']}")
        print(f"Preview Image URL:    {api_url}{st['preview_url']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OrthoGenerator Accurate 2D Orthophoto CLI & API")
    parser.add_argument("--frames", default="D:/ortho_generator/api_video_ortho/outputs/sample_run/frames", help="Folder of geotagged frames")
    parser.add_argument("--output", default="D:/ortho_generator/api_ortho_generator/outputs/accurate_ortho", help="Output directory")
    parser.add_argument("--engine", default="accurate", choices=["accurate", "fast"], help="Ortho engine ('accurate' = SfM + DSM True Orthorectification, 'fast' = Planar)")
    parser.add_argument("--gsd", type=float, default=5.0, help="Target GSD in cm/pixel (default: 5.0)")
    parser.add_argument("--blend", default="multiband", choices=["multiband", "feather", "average", "latest"], help="Blending mode")
    parser.add_argument("--api", action="store_true", help="Use REST API")
    parser.add_argument("--api-url", default="http://localhost:8001", help="API URL")

    args = parser.parse_args()

    if args.api:
        run_api_client(args.api_url, args.frames, engine=args.engine, gsd_cm=args.gsd)
    else:
        run_direct_pipeline(args.frames, args.output, engine=args.engine, gsd_cm=args.gsd, blend_mode=args.blend)
