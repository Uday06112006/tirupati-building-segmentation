import os
import cv2
import logging
from typing import List, Optional, Callable, Dict, Any, Tuple
from pathlib import Path
from PIL import Image

from .srt_parser import SRTTelemetryParser, TelemetryRecord
from .exif_writer import EXIFWriter
from ..models.schemas import FrameMetadata, ExtractionRequest

logger = logging.getLogger(__name__)

class FrameExtractor:
    """
    High-fidelity video frame extractor with timestamp-synchronized EXIF geotagging.
    """
    
    @staticmethod
    def get_video_info(video_path: str) -> Dict[str, Any]:
        """Inspect video file and return metadata."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration_sec = total_frames / fps if fps > 0 else 0.0
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        
        cap.release()
        
        return {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
            "duration_sec": duration_sec,
            "fourcc": fourcc
        }

    @classmethod
    def calculate_target_frames(
        cls,
        total_frames: int,
        fps: float,
        params: ExtractionRequest
    ) -> List[Tuple[int, float]]:
        """
        Calculate list of (frame_index, timestamp_sec) to extract based on request parameters.
        Frame indices are 0-based.
        """
        duration = total_frames / fps if fps > 0 else 0.0
        start_sec = max(0.0, params.start_time_sec or 0.0)
        end_sec = min(duration, params.end_time_sec) if params.end_time_sec else duration
        
        start_frame = int(round(start_sec * fps))
        end_frame = min(total_frames, int(round(end_sec * fps))) if end_sec > 0 else total_frames
        
        target_list: List[Tuple[int, float]] = []
        
        if params.frame_interval_sec is not None and params.frame_interval_sec > 0:
            step_sec = params.frame_interval_sec
            curr_sec = start_sec
            while curr_sec <= end_sec:
                f_idx = int(round(curr_sec * fps))
                if f_idx < total_frames:
                    target_list.append((f_idx, curr_sec))
                curr_sec += step_sec
        elif params.fps is not None and params.fps > 0:
            step_sec = 1.0 / params.fps
            curr_sec = start_sec
            while curr_sec <= end_sec:
                f_idx = int(round(curr_sec * fps))
                if f_idx < total_frames:
                    target_list.append((f_idx, curr_sec))
                curr_sec += step_sec
        else:
            step = max(1, params.frame_step)
            for f_idx in range(start_frame, end_frame, step):
                t_sec = f_idx / fps
                target_list.append((f_idx, t_sec))
                
        if params.max_frames and len(target_list) > params.max_frames:
            target_list = target_list[:params.max_frames]
            
        return target_list

    @classmethod
    def extract_and_tag(
        cls,
        video_path: str,
        srt_parser: SRTTelemetryParser,
        output_dir: str,
        params: ExtractionRequest,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[FrameMetadata]:
        """
        Extract video frames, match with SRT telemetry, inject EXIF metadata, and save to output_dir.
        """
        os.makedirs(output_dir, exist_ok=True)
        video_info = cls.get_video_info(video_path)
        fps = video_info["fps"]
        total_video_frames = video_info["total_frames"]
        
        target_frames = cls.calculate_target_frames(total_video_frames, fps, params)
        total_targets = len(target_frames)
        
        if total_targets == 0:
            logger.warning("No frames to extract based on provided filters.")
            return []
            
        target_map = {f_idx: t_sec for f_idx, t_sec in target_frames}
        max_target_f_idx = max(target_map.keys()) if target_map else 0
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        extracted_metadata: List[FrameMetadata] = []
        current_frame_idx = 0
        extracted_count = 0

        try:
            while current_frame_idx <= max_target_f_idx:
                if current_frame_idx in target_map:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    
                    target_sec = target_map[current_frame_idx]
                    save_frame_num = current_frame_idx + 1
                    ext = params.output_format.value.lower()
                    if ext == "jpeg":
                        ext = "jpg"
                    filename = f"frame_{save_frame_num:06d}.{ext}"
                    image_save_path = os.path.join(output_dir, filename)

                    # Match with SRT Telemetry
                    telemetry = srt_parser.get_telemetry_for_frame(save_frame_num)
                    if not telemetry:
                        telemetry = srt_parser.get_telemetry_for_timestamp(target_sec)

                    # Convert OpenCV BGR frame to RGB PIL Image with 100% fidelity
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)

                    # Extract telemetry fields
                    lat = telemetry.latitude if telemetry else 0.0
                    lon = telemetry.longitude if telemetry else 0.0
                    alt = telemetry.altitude if telemetry else 0.0
                    iso = telemetry.iso if telemetry else None
                    shutter = telemetry.shutter if telemetry else None
                    fnum = telemetry.fnum if telemetry else None
                    focal_len = telemetry.focal_len if telemetry else None
                    ts_str = telemetry.timestamp_str if telemetry else None
                    yaw = telemetry.yaw if telemetry else None
                    pitch = telemetry.pitch if telemetry else None
                    roll = telemetry.roll if telemetry else None

                    # Generate EXIF bytes
                    exif_bytes = EXIFWriter.create_exif_bytes(
                        latitude=lat,
                        longitude=lon,
                        altitude=alt,
                        timestamp_str=ts_str,
                        iso=iso,
                        shutter=shutter,
                        fnum=fnum,
                        focal_len=focal_len,
                        yaw=yaw,
                        pitch=pitch,
                        roll=roll,
                        frame_index=save_frame_num,
                        extra_telemetry={
                            "diff_time_ms": telemetry.diff_time_ms if telemetry else None,
                            "ev": telemetry.ev if telemetry else None,
                            "ct": telemetry.ct if telemetry else None,
                            "color_md": telemetry.color_md if telemetry else None
                        }
                    )

                    # Save tagged image
                    EXIFWriter.save_image_with_exif(
                        image_pil=pil_img,
                        output_path=image_save_path,
                        exif_bytes=exif_bytes,
                        quality=params.jpeg_quality
                    )

                    # Record metadata
                    meta = FrameMetadata(
                        frame_index=save_frame_num,
                        frame_id=telemetry.frame_cnt if telemetry else save_frame_num,
                        timestamp_str=ts_str,
                        timestamp_sec=round(target_sec, 4),
                        latitude=lat or 0.0,
                        longitude=lon or 0.0,
                        altitude=alt or 0.0,
                        iso=iso,
                        shutter=shutter,
                        fnum=fnum,
                        focal_len=focal_len,
                        ev=telemetry.ev if telemetry else None,
                        ct=telemetry.ct if telemetry else None,
                        color_md=telemetry.color_md if telemetry else None,
                        yaw=yaw,
                        pitch=pitch,
                        roll=roll,
                        filename=filename,
                        image_path=image_save_path
                    )
                    extracted_metadata.append(meta)
                    
                    extracted_count += 1
                    if progress_callback:
                        pct = round((extracted_count / total_targets) * 100.0, 1)
                        progress_callback(pct, f"Extracted & tagged {extracted_count}/{total_targets} frames")
                else:
                    # Fast-forward without decoding
                    ret = cap.grab()
                    if not ret:
                        break
                        
                current_frame_idx += 1
        finally:
            cap.release()

        logger.info(f"Successfully extracted and geotagged {len(extracted_metadata)} frames.")
        return extracted_metadata
