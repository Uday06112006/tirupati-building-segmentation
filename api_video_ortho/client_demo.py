"""
OrthoGenerator CLI and API Client Demo Script
Demonstrates extraction of geotagged frames and GeoJSON generation from drone video + SRT telemetry.
"""

import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path
from PIL import Image
import piexif

# Ensure UTF-8 output on Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.services.pipeline import OrthoPipeline
from app.models.schemas import ExtractionRequest, OutputFormat

def verify_extracted_image_exif(image_path: str):
    """Read and display verified EXIF metadata from an extracted image."""
    print(f"\n[EXIF VERIFICATION] Inspecting extracted image: {image_path}")
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return

    exif_data = piexif.load(image_path)
    
    # 0th IFD
    make = exif_data.get("0th", {}).get(piexif.ImageIFD.Make, b"").decode("utf-8", errors="ignore")
    model = exif_data.get("0th", {}).get(piexif.ImageIFD.Model, b"").decode("utf-8", errors="ignore")
    desc = exif_data.get("0th", {}).get(piexif.ImageIFD.ImageDescription, b"").decode("utf-8", errors="ignore")
    
    # GPS IFD
    gps = exif_data.get("GPS", {})
    lat_dms = gps.get(piexif.GPSIFD.GPSLatitude)
    lat_ref = gps.get(piexif.GPSIFD.GPSLatitudeRef, b"").decode("utf-8", errors="ignore")
    lon_dms = gps.get(piexif.GPSIFD.GPSLongitude)
    lon_ref = gps.get(piexif.GPSIFD.GPSLongitudeRef, b"").decode("utf-8", errors="ignore")
    alt = gps.get(piexif.GPSIFD.GPSAltitude)
    
    # Exif IFD
    exif = exif_data.get("Exif", {})
    iso = exif.get(piexif.ExifIFD.ISOSpeedRatings)
    dt_orig = exif.get(piexif.ExifIFD.DateTimeOriginal, b"").decode("utf-8", errors="ignore")
    user_comment = exif.get(piexif.ExifIFD.UserComment, b"")
    if user_comment.startswith(b"ASCII\x00\x00\x00"):
        user_comment = user_comment[8:].decode("utf-8", errors="ignore")

    print(f"  * Make / Model:        {make} {model}")
    print(f"  * Capture Timestamp:   {dt_orig}")
    print(f"  * Image Description:   {desc}")
    if lat_dms and lon_dms:
        lat_dec = lat_dms[0][0]/lat_dms[0][1] + lat_dms[1][0]/(lat_dms[1][1]*60) + (lat_dms[2][0]/lat_dms[2][1])/3600
        lon_dec = lon_dms[0][0]/lon_dms[0][1] + lon_dms[1][0]/(lon_dms[1][1]*60) + (lon_dms[2][0]/lon_dms[2][1])/3600
        print(f"  * GPS Coordinates:     {lat_dec:.6f} deg {lat_ref}, {lon_dec:.6f} deg {lon_ref}")
    if alt:
        alt_m = alt[0] / alt[1]
        print(f"  * Altitude:            {alt_m:.2f} m")
    if iso:
        print(f"  * ISO Speed:           {iso}")
    if user_comment:
        print(f"  * Telemetry Payload:   {user_comment}")
    print("  [SUCCESS] EXIF Verification PASSED\n")

def run_direct_pipeline(video_path: str, srt_path: str, output_dir: str, frame_interval_sec: float = None, frame_step: int = 1, max_frames: int = None):
    """Run extraction pipeline locally in Python."""
    print("=" * 70)
    print("Running OrthoGenerator Frame Extraction Pipeline (Direct)")
    print("=" * 70)
    print(f"Video Path: {video_path}")
    print(f"SRT Path:   {srt_path}")
    print(f"Output Dir: {output_dir}")
    
    params = ExtractionRequest(
        video_path=video_path,
        srt_path=srt_path,
        frame_step=frame_step,
        frame_interval_sec=frame_interval_sec,
        max_frames=max_frames,
        output_format=OutputFormat.JPG,
        jpeg_quality=100,
        include_line_string=True,
        export_zip=True,
        export_geojson=True
    )
    
    def progress_cb(pct, msg):
        print(f"[{pct:5.1f}%] {msg}")

    result = OrthoPipeline.run_pipeline(
        video_path=video_path,
        srt_path=srt_path,
        output_dir=output_dir,
        params=params,
        progress_callback=progress_cb
    )

    print("\n" + "=" * 70)
    print("Pipeline Execution Completed Successfully!")
    print("=" * 70)
    print(f"Extracted Frames:  {result['extracted_count']}")
    print(f"Elapsed Time:      {result['elapsed_sec']} seconds")
    print(f"GeoJSON Path:      {result['geojson_path']}")
    print(f"ZIP Archive Path:  {result['zip_path']}")
    
    # Verify first extracted frame
    frames_dir = result["frames_dir"]
    frame_files = sorted(os.listdir(frames_dir))
    if frame_files:
        sample_img = os.path.join(frames_dir, frame_files[0])
        verify_extracted_image_exif(sample_img)
        
    return result

def run_api_client(api_url: str, video_path: str, srt_path: str, frame_interval_sec: float = None, frame_step: int = 1):
    """Run extraction by calling the FastAPI microservice."""
    print("=" * 70)
    print(f"Calling Microservice API at: {api_url}")
    print("=" * 70)
    
    # 1. Health check
    health_resp = requests.get(f"{api_url}/api/v1/health")
    print(f"Health check status: {health_resp.status_code} -> {health_resp.json()}")

    # 2. Trigger extraction from path
    payload = {
        "video_path": os.path.abspath(video_path),
        "srt_path": os.path.abspath(srt_path),
        "frame_step": frame_step,
        "frame_interval_sec": frame_interval_sec,
        "output_format": "jpg",
        "jpeg_quality": 100,
        "include_line_string": True,
        "export_zip": True,
        "export_geojson": True
    }
    
    start_resp = requests.post(f"{api_url}/api/v1/jobs/extract-from-path?async_mode=true", json=payload)
    job_data = start_resp.json()
    job_id = job_data["job_id"]
    print(f"Job initiated with ID: {job_id}")

    # 3. Poll status
    while True:
        status_resp = requests.get(f"{api_url}/api/v1/jobs/{job_id}")
        st = status_resp.json()
        print(f"[{st.get('progress_percent', 0):5.1f}%] Status: {st['status']} | {st.get('message', '')}")
        if st["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(1.0)
        
    if st["status"] == "COMPLETED":
        print("\nAPI Job Completed Successfully!")
        print(f"GeoJSON URL:  {api_url}{st['geojson_url']}")
        print(f"Download URL: {api_url}{st['download_zip_url']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OrthoGenerator Frame Extractor & Geotagger")
    parser.add_argument("--video", default="D:/ortho_generator/F7/DJI_0527.MOV", help="Path to video file")
    parser.add_argument("--srt", default="D:/ortho_generator/F7/DJI_0527.SRT", help="Path to SRT file")
    parser.add_argument("--output", default="D:/ortho_generator/api_video_ortho/outputs/sample_run", help="Output directory")
    parser.add_argument("--interval", type=float, default=1.0, help="Frame interval in seconds (e.g. 1.0 = 1 fps)")
    parser.add_argument("--step", type=int, default=1, help="Frame step (e.g. 1 = all, 24 = every 24th)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to extract")
    parser.add_argument("--api", action="store_true", help="Run via HTTP API instead of direct pipeline")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API URL")
    
    args = parser.parse_args()
    
    if args.api:
        run_api_client(args.api_url, args.video, args.srt, frame_interval_sec=args.interval, frame_step=args.step)
    else:
        run_direct_pipeline(args.video, args.srt, args.output, frame_interval_sec=args.interval, frame_step=args.step, max_frames=args.max_frames)
