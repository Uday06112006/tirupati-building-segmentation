"""
Derive per-frame camera attitude for a drone video whose SRT carries no yaw/pitch/roll.

    python tools/derive_attitude.py --video D:\\ortho_generator\\F7\\DJI_0527.MOV \\
                                    --srt   D:\\ortho_generator\\F7\\DJI_0527.SRT \\
                                    --out   outputs/DJI_0527_attitude.csv

Add --verify-focal to re-check the SRT focal_len against scene geometry, or
--no-visual for a GPS-track-only run (fast, but yields course over ground only).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.srt_parser import SRTTelemetryParser
from app.services.attitude_estimator import (
    compute_track_dynamics, estimate_visual_attitude, verify_focal_length,
    focal_px_from_35mm, fuse, write_csv,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("derive_attitude")


def circular_stats(deg: np.ndarray):
    d = deg[~np.isnan(deg)]
    if len(d) == 0:
        return float("nan"), float("nan")
    r = np.radians(d)
    s, c = np.sin(r).mean(), np.cos(r).mean()
    R = math.hypot(s, c)
    mean = math.degrees(math.atan2(s, c)) % 360.0
    std = math.degrees(math.sqrt(-2 * math.log(R))) if 0 < R <= 1 else float("nan")
    return mean, std


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--srt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--focal-35mm", type=float, default=None,
                    help="35mm-equivalent focal length; default: SRT focal_len")
    ap.add_argument("--stride", type=int, default=4, help="emit a visual sample every N frames")
    ap.add_argument("--gap", type=int, default=8, help="homography baseline in frames")
    ap.add_argument("--scale", type=float, default=0.5, help="frame downscale for tracking")
    ap.add_argument("--no-visual", action="store_true", help="GPS track only")
    ap.add_argument("--verify-focal", action="store_true",
                    help="score candidate focal lengths against building verticals")
    ap.add_argument("--report", default=None, help="optional JSON summary path")
    args = ap.parse_args()

    recs = SRTTelemetryParser().parse_file(args.srt)
    if not recs:
        log.error("no telemetry parsed from %s", args.srt)
        return 2
    log.info("parsed %d telemetry records", len(recs))

    have_att = sum(r.yaw is not None for r in recs)
    if have_att:
        log.warning("SRT already carries yaw on %d records - derived values are a cross-check",
                    have_att)
    else:
        log.info("SRT carries no yaw/pitch/roll fields; deriving them")

    lat = np.array([r.latitude for r in recs], float)
    lon = np.array([r.longitude for r in recs], float)
    alt = np.array([r.altitude for r in recs], float)
    t = np.array([r.start_sec for r in recs], float)
    if np.isnan(lat).any() or np.isnan(lon).any():
        log.error("telemetry has missing coordinates")
        return 2

    dyn = compute_track_dynamics(lat, lon, alt, t)
    cmean, cstd = circular_stats(dyn.course_deg)
    log.info("track: mean course %.2f deg (circ.std %.2f), mean speed %.2f m/s",
             cmean, cstd, dyn.ground_speed.mean())

    focal_35 = args.focal_35mm
    if focal_35 is None:
        vals = [r.focal_len for r in recs if r.focal_len]
        focal_35 = float(np.median(vals)) if vals else 24.0
        log.info("focal length from SRT: %.1f mm (35mm-equiv)", focal_35)

    report = {
        "video": args.video, "srt": args.srt, "records": len(recs),
        "srt_has_attitude": bool(have_att),
        "track": {"mean_course_deg": cmean, "course_circ_std_deg": cstd,
                  "mean_speed_mps": float(dyn.ground_speed.mean()),
                  "max_speed_mps": float(dyn.ground_speed.max())},
        "focal_35mm_used": focal_35,
    }

    if args.verify_focal:
        idxs = list(range(72, min(len(recs) - 12, 400), 30))
        cands = [round(20 + 1.0 * i, 1) for i in range(16)]
        log.info("verifying focal length over %d candidates on %d frames", len(cands), len(idxs))
        scored = verify_focal_length(args.video, dyn, cands, idxs,
                                     gap=args.gap, scale=args.scale)
        best = max(scored, key=lambda r: r[1])
        log.info("best-supported focal: %.1f mm (score %.0f, pitch %.2f)", *best)
        report["focal_verification"] = [
            {"focal_35mm": f, "support": s, "pitch": p} for f, s, p in scored]
        report["focal_best"] = {"focal_35mm": best[0], "pitch": best[2]}

    visual = None
    if not args.no_visual:
        fpx = focal_px_from_35mm(focal_35, 3840)
        log.info("solving visual attitude (f=%.0f px, stride=%d, gap=%d, scale=%.2f)",
                 fpx, args.stride, args.gap, args.scale)
        visual = estimate_visual_attitude(
            args.video, dyn, t, fpx,
            stride=args.stride, gap=args.gap, scale=args.scale,
            progress=lambda i: log.info("  decoded %d frames", i) if i % 240 == 0 else None)
        if visual.samples:
            p = np.array([s.pitch for s in visual.samples])
            r = np.array([s.roll for s in visual.samples])
            y = np.array([s.yaw for s in visual.samples])
            a = np.array([s.agl_m for s in visual.samples])
            ymean, ystd = circular_stats(y)
            off = np.array([(s.yaw - s.track_bearing + 540) % 360 - 180
                            for s in visual.samples])
            log.info("visual: %d samples | pitch %.2f+-%.2f | roll %.2f+-%.2f | "
                     "yaw %.2f+-%.2f | AGL %.1f+-%.1f m",
                     len(p), p.mean(), p.std(), r.mean(), r.std(),
                     ymean, ystd, np.nanmean(a), np.nanstd(a))
            log.info("camera yaw minus track bearing: %.2f +- %.2f deg",
                     off.mean(), off.std())
            report["visual"] = {
                "samples": len(p),
                "pitch_deg": {"mean": float(p.mean()), "std": float(p.std())},
                "roll_deg": {"mean": float(r.mean()), "std": float(r.std())},
                "yaw_deg": {"mean": ymean, "circ_std": ystd},
                "agl_m": {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a))},
                "yaw_minus_track_deg": {"mean": float(off.mean()), "std": float(off.std())},
                "median_inliers": int(np.median([s.inliers for s in visual.samples])),
            }
        else:
            log.warning("no visual samples solved; falling back to GPS track only")

    frames = fuse(recs, dyn, visual)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    write_csv(frames, args.out)
    solved = sum(1 for f in frames if f.attitude_source == "visual-homography")
    extrap = sum(1 for f in frames if f.attitude_source == "visual-extrapolated")
    log.info("wrote %s | %d/%d frames solved, %d extrapolated at the ends",
             args.out, solved, len(frames), extrap)
    report["frames_with_visual_attitude"] = solved
    report["frames_extrapolated"] = extrap

    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        log.info("wrote %s", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
