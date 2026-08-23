import os
import shutil
import zipfile
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable
from pathlib import Path

from .srt_parser import SRTTelemetryParser
from .frame_extractor import FrameExtractor
from .geojson_builder import GeoJSONBuilder
from ..models.schemas import ExtractionRequest, FrameMetadata

logger = logging.getLogger(__name__)

class OrthoPipeline:
    """
    Unified end-to-end pipeline for video frame extraction, EXIF geotagging,
    and GeoJSON generation.
    """

    @classmethod
    def run_pipeline(
        cls,
        video_path: str,
        srt_path: str,
        output_dir: str,
        params: ExtractionRequest,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete processing pipeline.
        """
        start_time = datetime.now(timezone.utc)
        if progress_callback:
            progress_callback(5.0, "Parsing SRT telemetry track...")

        # 1. Parse SRT
        parser = SRTTelemetryParser()
        parser.parse_file(srt_path)
        
        if progress_callback:
            progress_callback(10.0, "Inspecting video stream and calculating target frames...")

        # 2. Inspect video
        video_info = FrameExtractor.get_video_info(video_path)
        
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        if progress_callback:
            progress_callback(15.0, "Extracting video frames and injecting EXIF GPS metadata...")

        # 3. Frame Extraction & Geotagging
        def internal_progress(pct: float, msg: str):
            # Scale 15% -> 85%
            scaled = 15.0 + (pct / 100.0) * 70.0
            if progress_callback:
                progress_callback(scaled, msg)

        extracted_frames = FrameExtractor.extract_and_tag(
            video_path=video_path,
            srt_parser=parser,
            output_dir=frames_dir,
            params=params,
            progress_callback=internal_progress
        )

        # 4. Generate GeoJSON
        geojson_path = None
        geojson_data = None
        if params.export_geojson and extracted_frames:
            if progress_callback:
                progress_callback(88.0, "Building RFC 7946 GeoJSON telemetry points and trajectory...")
                
            geojson_data = GeoJSONBuilder.build_geojson(
                frames=extracted_frames,
                include_line_string=params.include_line_string,
                job_info={
                    "video_file": os.path.basename(video_path),
                    "srt_file": os.path.basename(srt_path),
                    "video_resolution": video_info["resolution"],
                    "video_fps": video_info["fps"],
                    "total_video_frames": video_info["total_frames"],
                    "extracted_frames_count": len(extracted_frames),
                    "processed_at": start_time.isoformat()
                }
            )
            
            geojson_filename = f"{Path(video_path).stem}_telemetry.geojson"
            geojson_path = os.path.join(output_dir, geojson_filename)
            GeoJSONBuilder.save_geojson(geojson_data, geojson_path)

        # 5. Create ZIP package
        zip_path = None
        if params.export_zip and extracted_frames:
            if progress_callback:
                progress_callback(93.0, "Archiving geotagged frames and GeoJSON into ZIP...")
                
            zip_filename = f"{Path(video_path).stem}_geotagged_frames.zip"
            zip_path = os.path.join(output_dir, zip_filename)
            
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                # Add all frames
                for root, _, files in os.walk(frames_dir):
                    for file in files:
                        abs_file = os.path.join(root, file)
                        rel_file = os.path.relpath(abs_file, output_dir)
                        zipf.write(abs_file, rel_file)
                # Add GeoJSON if generated
                if geojson_path and os.path.exists(geojson_path):
                    zipf.write(geojson_path, os.path.basename(geojson_path))

        if progress_callback:
            progress_callback(100.0, "Processing completed successfully.")

        end_time = datetime.now(timezone.utc)
        elapsed_sec = round((end_time - start_time).total_seconds(), 2)

        return {
            "status": "COMPLETED",
            "video_info": video_info,
            "extracted_count": len(extracted_frames),
            "output_dir": output_dir,
            "frames_dir": frames_dir,
            "geojson_path": geojson_path,
            "geojson_data": geojson_data,
            "zip_path": zip_path,
            "elapsed_sec": elapsed_sec
        }
