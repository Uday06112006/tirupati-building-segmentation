"""
One-shot job: extract frames at a fixed cadence and geotag them with full orientation.

    python tools/extract_geotagged.py \
        --video D:/ortho_generator/F7/DJI_0527.MOV \
        --srt   D:/ortho_generator/F7/DJI_0527.SRT \
        --interval 1.0 \
        --out   outputs/DJI_0527_1fps

Each frame gets EXIF (WGS-84 GPS, exposure, timestamp) *and* an XMP packet carrying
gimbal yaw/pitch/roll, which is what WebODM / Pix4D / Metashape actually read.

Attitude comes from ``--attitude-csv`` if given, else from a sibling CSV next to the
output, else it is derived on the spot (a few minutes -- it decodes the whole video).
Pass --no-attitude to skip orientation entirely.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.schemas import ExtractionRequest, OutputFormat
from app.services.pipeline import OrthoPipeline
from app.services.attitude_estimator import AttitudeTrack

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("extract_geotagged")


def obtain_attitude(args) -> AttitudeTrack | None:
    """Load a derived attitude CSV, or derive one now."""
    if args.no_attitude:
        return None

    candidates = []
    if args.attitude_csv:
        candidates.append(args.attitude_csv)
    stem = os.path.splitext(os.path.basename(args.video))[0]
    candidates += [
        os.path.join("outputs", f"{stem}_attitude.csv"),
        os.path.join(args.out, f"{stem}_attitude.csv"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            log.info("using attitude from %s", path)
            return AttitudeTrack.from_csv(path)

    if args.attitude_csv:
        log.error("attitude CSV not found: %s", args.attitude_csv)
        raise SystemExit(2)

    log.info("no attitude CSV found; deriving now (decodes the whole video)")
    import numpy as np
    from app.services.srt_parser import SRTTelemetryParser
    from app.services.attitude_estimator import (
        compute_track_dynamics, estimate_visual_attitude, focal_px_from_35mm,
        fuse, write_csv,
    )

    recs = SRTTelemetryParser().parse_file(args.srt)
    lat = np.array([r.latitude for r in recs], float)
    lon = np.array([r.longitude for r in recs], float)
    alt = np.array([r.altitude for r in recs], float)
    t = np.array([r.start_sec for r in recs], float)
    dyn = compute_track_dynamics(lat, lon, alt, t)
    focal_vals = [r.focal_len for r in recs if r.focal_len]
    focal_35 = float(np.median(focal_vals)) if focal_vals else 24.0
    visual = estimate_visual_attitude(args.video, dyn, t,
                                      focal_px_from_35mm(focal_35, 3840))
    frames = fuse(recs, dyn, visual)
    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, f"{stem}_attitude.csv")
    write_csv(frames, out_csv)
    log.info("derived attitude written to %s", out_csv)
    return AttitudeTrack(frames)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--srt", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="seconds between extracted frames (default 1.0 = 1 fps)")
    ap.add_argument("--attitude-csv", default=None)
    ap.add_argument("--no-attitude", action="store_true")
    ap.add_argument("--quality", type=int, default=100)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    for p in (args.video, args.srt):
        if not os.path.exists(p):
            log.error("not found: %s", p)
            return 2

    attitude = obtain_attitude(args)
    if attitude:
        log.info("attitude track: %d frames, %d with a visual solution",
                 len(attitude), attitude.solved_count)

    params = ExtractionRequest(
        video_path=args.video,
        srt_path=args.srt,
        frame_interval_sec=args.interval,
        output_format=OutputFormat.JPG,
        jpeg_quality=args.quality,
        max_frames=args.max_frames,
        export_geojson=True,
        include_line_string=True,
        export_zip=not args.no_zip,
        job_name=f"{os.path.splitext(os.path.basename(args.video))[0]}_{args.interval}s",
    )

    os.makedirs(args.out, exist_ok=True)
    result = OrthoPipeline.run_pipeline(
        video_path=args.video,
        srt_path=args.srt,
        output_dir=args.out,
        params=params,
        progress_callback=lambda pct, msg: log.info("[%5.1f%%] %s", pct, msg),
        attitude=attitude,
    )

    log.info("-" * 60)
    log.info("extracted %d frames -> %s", result["extracted_count"], result["frames_dir"])
    log.info("video: %s @ %.3f fps, %d frames total",
             result["video_info"]["resolution"], result["video_info"]["fps"],
             result["video_info"]["total_frames"])
    if result.get("geojson_path"):
        log.info("geojson: %s", result["geojson_path"])
    if result.get("zip_path"):
        log.info("zip:     %s", result["zip_path"])
    log.info("elapsed: %.1f s", result["elapsed_sec"])

    summary = os.path.join(args.out, "job_summary.json")
    with open(summary, "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in result.items() if k != "geojson_data"},
                  fh, indent=2, default=str)
    log.info("summary: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
