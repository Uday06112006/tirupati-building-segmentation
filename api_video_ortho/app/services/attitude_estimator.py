"""
Attitude recovery for drone video whose telemetry lacks roll/pitch/yaw.

Many DJI SRT tracks carry only position and exposure (lat/lon/alt/iso/shutter/...)
with no ``gb_yaw`` / ``gb_pitch`` / ``gb_roll`` fields. This module recovers camera
exterior orientation anyway, from two independent sources:

1. **Track dynamics** (GPS only) -- course over ground, ground speed, heading rate.
   Consumer DJI SRT latitude/longitude are quantised to 6 decimal places (~0.11 m),
   so differencing *consecutive* frames mostly differences quantisation noise. We
   instead fit a local linear model over a time window long enough that real motion
   dominates the quantisation step.

2. **Visual attitude** (video + GPS) -- the ground-plane homography between two
   frames decomposes into ``(R, t/d, n)`` where ``n`` is the ground normal in the
   camera frame. That normal *is* the gravity direction, giving true pitch and roll.
   Combining ``t`` (motion direction in camera frame) with the GPS bearing (motion
   direction in the world) fixes the remaining degree of freedom, yielding absolute
   yaw. The ``|t|`` scale factor additionally recovers height above ground.

Method 2 supersedes method 1: course over ground is where the aircraft *goes*, not
where the camera *looks*. Method 1 is retained as a fallback and cross-check.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Sensor width assumed by the "35 mm equivalent" convention DJI uses for `focal_len`.
FULL_FRAME_WIDTH_MM = 36.0


# --------------------------------------------------------------------------- #
# Geodesy
# --------------------------------------------------------------------------- #

def meters_per_degree(lat_deg: float) -> Tuple[float, float]:
    """Local metres per degree of latitude / longitude (WGS-84 series expansion)."""
    lat = math.radians(lat_deg)
    m_lat = 111132.92 - 559.82 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat) \
        - 0.0023 * math.cos(6 * lat)
    m_lon = 111412.84 * math.cos(lat) - 93.5 * math.cos(3 * lat) + 0.118 * math.cos(5 * lat)
    return m_lat, m_lon


def geodetic_to_enu(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray,
                    lat0: Optional[float] = None,
                    lon0: Optional[float] = None,
                    alt0: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local flat-earth ENU (metres) about an origin. Adequate for sub-kilometre legs."""
    lat0 = float(np.mean(lat)) if lat0 is None else lat0
    lon0 = float(np.mean(lon)) if lon0 is None else lon0
    alt0 = float(np.mean(alt)) if alt0 is None else alt0
    m_lat, m_lon = meters_per_degree(lat0)
    return (lon - lon0) * m_lon, (lat - lat0) * m_lat, (alt - alt0)


# --------------------------------------------------------------------------- #
# Camera intrinsics
# --------------------------------------------------------------------------- #

def focal_px_from_35mm(focal_35mm: float, image_width_px: int) -> float:
    """Focal length in pixels from a 35 mm-equivalent focal length."""
    half_fov = math.atan(FULL_FRAME_WIDTH_MM / (2.0 * focal_35mm))
    return (image_width_px / 2.0) / math.tan(half_fov)


def intrinsics(focal_px: float, width: int, height: int) -> np.ndarray:
    """Pinhole K with the principal point at image centre."""
    return np.array([[focal_px, 0.0, width / 2.0],
                     [0.0, focal_px, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=float)


# --------------------------------------------------------------------------- #
# Attitude conventions
# --------------------------------------------------------------------------- #
# Camera frame: x right, y down, z along the optical axis.
# pitch: elevation of the optical axis; 0 = horizontal, -90 = nadir.
# roll : rotation about the optical axis; positive = right-hand side down.
# yaw  : azimuth of the optical axis, degrees clockwise from true north.

def pitch_roll_from_down_vector(d_cam: np.ndarray) -> Tuple[float, float]:
    """Camera pitch/roll from the world-'down' direction expressed in camera axes."""
    d = np.array(d_cam, dtype=float)          # copy: never normalise the caller's array
    d /= max(float(np.linalg.norm(d)), 1e-12)  # max(), not +eps, which would shrink a unit vector
    pitch = math.degrees(math.acos(float(np.clip(d[2], -1.0, 1.0)))) - 90.0
    roll = math.degrees(math.atan2(float(d[0]), float(d[1])))
    return pitch, roll


def yaw_from_motion(d_cam: np.ndarray, motion_cam: np.ndarray, track_bearing_deg: float) -> float:
    """
    Absolute camera yaw.

    ``d_cam`` (world down) and ``motion_cam`` (the camera's own direction of travel)
    are both known in camera axes; the same two directions are known in the world
    (down = -Z_up, and the motion bearing from GPS). Two independent correspondences
    determine the full world->camera rotation, hence yaw.

    Note ``motion_cam`` must be the *camera's* motion. ``decomposeHomographyMat``
    returns ``t`` for ``X2 = R X1 + t`` -- the scene's motion relative to the camera,
    which is the opposite direction. See ``_decompose_ground_homography``.
    """
    d = np.array(d_cam, dtype=float); d /= max(float(np.linalg.norm(d)), 1e-12)
    t = np.array(motion_cam, dtype=float); t /= max(float(np.linalg.norm(t)), 1e-12)

    # Camera-frame orthonormal triad from (down, motion).
    b1 = d
    b2 = t - np.dot(t, b1) * b1
    n2 = float(np.linalg.norm(b2))
    if n2 < 1e-6:                       # motion parallel to gravity: yaw unobservable
        return float("nan")
    b2 /= n2
    b3 = np.cross(b1, b2)
    B = np.column_stack([b1, b2, b3])

    # Same triad in world ENU axes (east, north, up).
    brg = math.radians(track_bearing_deg)
    a1 = np.array([0.0, 0.0, -1.0])                      # down
    a2 = np.array([math.sin(brg), math.cos(brg), 0.0])   # horizontal motion
    a2 = a2 - np.dot(a2, a1) * a1
    a2 /= (np.linalg.norm(a2) + 1e-12)
    a3 = np.cross(a1, a2)
    A = np.column_stack([a1, a2, a3])

    R_wc = B @ A.T                       # world -> camera
    axis_world = R_wc.T @ np.array([0.0, 0.0, 1.0])      # optical axis in ENU
    return math.degrees(math.atan2(axis_world[0], axis_world[1])) % 360.0


# --------------------------------------------------------------------------- #
# 1. Track dynamics from GPS
# --------------------------------------------------------------------------- #

@dataclass
class TrackDynamics:
    """Per-frame kinematics derived from the GPS track alone."""
    course_deg: np.ndarray          # course over ground (deg from north), NaN when idle
    ground_speed: np.ndarray        # m/s, horizontal
    heading_change: np.ndarray      # deg/s, signed turn rate
    vertical_speed: np.ndarray      # m/s, positive up
    course_quality: np.ndarray      # 0..1; fraction of window motion vs quantisation
    east: np.ndarray
    north: np.ndarray
    up: np.ndarray


def compute_track_dynamics(lat: Sequence[float], lon: Sequence[float],
                           alt: Sequence[float], t_sec: Sequence[float],
                           window_frames: int = 8,
                           min_speed_mps: float = 0.5,
                           quantisation_m: float = 0.11) -> TrackDynamics:
    """
    Course over ground / speed / turn rate, robust to SRT coordinate quantisation.

    A local linear fit of E(t) and N(t) over +/-``window_frames`` gives a velocity
    whose direction is the course. Over that window the aircraft must travel well
    beyond ``quantisation_m`` for the course to mean anything; ``course_quality``
    reports that ratio and the course is set to NaN when it is hopeless.
    """
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    alt = np.asarray(alt, float); t = np.asarray(t_sec, float)
    n = len(lat)
    E, N, U = geodetic_to_enu(lat, lon, alt)

    course = np.full(n, np.nan)
    speed = np.zeros(n)
    vspeed = np.zeros(n)
    quality = np.zeros(n)

    for i in range(n):
        a, b = max(0, i - window_frames), min(n, i + window_frames + 1)
        tw = t[a:b] - t[i]
        span = float(np.ptp(tw))
        if len(tw) < 3 or span < 1e-6:
            continue
        # slope of a degree-1 polynomial == velocity component
        ve = np.polyfit(tw, E[a:b], 1)[0]
        vn = np.polyfit(tw, N[a:b], 1)[0]
        vu = np.polyfit(tw, U[a:b], 1)[0]
        sp = math.hypot(ve, vn)
        speed[i] = sp
        vspeed[i] = vu
        travelled = sp * span
        quality[i] = min(1.0, travelled / max(quantisation_m * 3.0, 1e-9))
        if sp >= min_speed_mps and travelled > quantisation_m * 3.0:
            course[i] = math.degrees(math.atan2(ve, vn)) % 360.0

    # signed turn rate, unwrapped so the 0/360 seam does not create spikes
    heading_change = np.full(n, np.nan)
    valid = ~np.isnan(course)
    if valid.sum() >= 2:
        idx = np.flatnonzero(valid)
        unwrapped = np.degrees(np.unwrap(np.radians(course[idx])))
        dh = np.gradient(unwrapped, t[idx])
        heading_change[idx] = dh

    return TrackDynamics(course, speed, heading_change, vspeed, quality, E, N, U)


# --------------------------------------------------------------------------- #
# 2. Visual attitude from the ground-plane homography
# --------------------------------------------------------------------------- #

@dataclass
class VisualSample:
    """One homography solution, anchored at frame ``index``."""
    index: int
    frame_a: int
    frame_b: int
    pitch: float
    roll: float
    yaw: float
    agl_m: float
    inliers: int
    down_cam: Tuple[float, float, float]
    track_bearing: float
    baseline_m: float


@dataclass
class VisualAttitudeResult:
    samples: List[VisualSample] = field(default_factory=list)
    focal_px: float = 0.0
    width: int = 0
    height: int = 0
    scale: float = 1.0

    def as_arrays(self):
        idx = np.array([s.index for s in self.samples], float)
        return (idx,
                np.array([s.pitch for s in self.samples], float),
                np.array([s.roll for s in self.samples], float),
                np.array([s.yaw for s in self.samples], float),
                np.array([s.agl_m for s in self.samples], float))


def _decompose_ground_homography(H: np.ndarray, K: np.ndarray,
                                 pts_a: np.ndarray, pts_b: np.ndarray,
                                 max_roll_deg: float = 35.0):
    """
    Pick the physically valid ``(down, motion)`` from the homography's four-fold ambiguity.

    OpenCV's cheirality filter removes solutions that put tracked points behind the
    camera; of the survivors we take the one with the smallest |roll|, since a
    stabilised gimbal does not fly inverted (the rejected twin is typically ~180
    degrees rolled).

    Returns ``(pitch, roll, down_cam, motion_cam)``. OpenCV's ``t`` satisfies
    ``X2 = R X1 + t``, i.e. it describes how the *scene* shifts in camera axes, so
    the camera's own direction of travel is ``-t``; that negated vector is what is
    returned. Its magnitude is still ``|t|/d``, so ``baseline / |motion_cam|``
    recovers the distance to the ground plane.
    """
    import cv2

    num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, K)
    try:
        keep = cv2.filterHomographyDecompByVisibleRefpoints(
            np.array(Rs), np.array(Ns),
            pts_a.reshape(-1, 1, 2).astype(np.float32),
            pts_b.reshape(-1, 1, 2).astype(np.float32), None)
        cand = [int(i) for i in keep.ravel()] if keep is not None else list(range(num))
    except Exception:
        cand = list(range(num))

    best = None
    for s in cand:
        n = np.asarray(Ns[s], float).ravel().copy()
        t = np.asarray(Ts[s], float).ravel().copy()
        if n[2] < 0:                     # plane must lie in front of the camera
            n, t = -n, -t
        d = n / (np.linalg.norm(n) + 1e-12)      # world-down in camera axes
        pitch, roll = pitch_roll_from_down_vector(d)
        if abs(roll) > max_roll_deg:
            continue
        if best is None or abs(roll) < abs(best[1]):
            best = (pitch, roll, d, -t)          # -t: camera motion, not scene motion
    return best


def _track_features(gray_a, gray_b, max_corners=4000, fb_thresh=1.0):
    """Forward-backward checked LK tracks between two frames."""
    import cv2

    p0 = cv2.goodFeaturesToTrack(gray_a, max_corners, 0.01, 12, blockSize=7)
    if p0 is None or len(p0) < 60:
        return None, None
    lk = dict(winSize=(31, 31), maxLevel=4,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(gray_a, gray_b, p0, None, **lk)
    if p1 is None:
        return None, None
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray_b, gray_a, p1, None, **lk)
    if p0r is None:
        return None, None
    fb = np.linalg.norm(p0.reshape(-1, 2) - p0r.reshape(-1, 2), axis=1)
    m = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < fb_thresh)
    a, b = p0.reshape(-1, 2)[m], p1.reshape(-1, 2)[m]
    return (a, b) if len(a) >= 60 else (None, None)


def estimate_visual_attitude(video_path: str,
                             dynamics: TrackDynamics,
                             t_sec: Sequence[float],
                             focal_px_full: float,
                             stride: int = 4,
                             gap: int = 8,
                             scale: float = 0.5,
                             min_baseline_m: float = 0.8,
                             min_inliers: int = 200,
                             progress=None) -> VisualAttitudeResult:
    """
    Solve per-sample camera attitude from consecutive-frame ground homographies.

    ``gap`` sets the baseline in frames (needs real parallax); ``stride`` how often a
    sample is produced; ``scale`` downsamples frames for speed (attitude is a
    direction, so it is scale invariant once K is scaled to match).
    """
    import av
    import cv2
    from collections import deque

    t_sec = np.asarray(t_sec, float)
    container = av.open(video_path)
    vstream = container.streams.video[0]
    vstream.thread_type = "AUTO"
    fw = int(vstream.codec_context.width)
    fh = int(vstream.codec_context.height)
    w, h = int(round(fw * scale)), int(round(fh * scale))
    K = intrinsics(focal_px_full * scale, w, h)

    result = VisualAttitudeResult(focal_px=focal_px_full, width=fw, height=fh, scale=scale)
    buf: deque = deque(maxlen=gap + 1)
    n_tel = len(dynamics.east)
    idx = 0

    def world_move(i0: int, i1: int, half: int = 4) -> Tuple[float, float, float]:
        """Smoothed ENU displacement between two telemetry indices."""
        def pos(k):
            a, b = max(0, k - half), min(n_tel, k + half + 1)
            return np.array([dynamics.east[a:b].mean(),
                             dynamics.north[a:b].mean(),
                             dynamics.up[a:b].mean()])
        i0 = min(max(i0, 0), n_tel - 1)
        i1 = min(max(i1, 0), n_tel - 1)
        v = pos(i1) - pos(i0)
        return float(v[0]), float(v[1]), float(v[2])

    for frame in container.decode(vstream):
        gray = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2GRAY)
        if scale != 1.0:
            gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
        buf.append((idx, gray))

        if len(buf) == gap + 1 and (idx % stride == 0):
            i0, g0 = buf[0]
            i1, g1 = buf[-1]
            de, dn, _ = world_move(i0, i1)
            baseline = math.hypot(de, dn)
            if baseline >= min_baseline_m:
                a, b = _track_features(g0, g1)
                if a is not None:
                    H, inl = cv2.findHomography(a, b, cv2.RANSAC, 2.0,
                                                maxIters=5000, confidence=0.9995)
                    if H is not None and inl is not None and int(inl.sum()) >= min_inliers:
                        mask = inl.ravel().astype(bool)
                        sol = _decompose_ground_homography(H, K, a[mask][:600], b[mask][:600])
                        if sol is not None:
                            pitch, roll, d_cam, motion_cam = sol
                            bearing = math.degrees(math.atan2(de, dn)) % 360.0
                            yaw = yaw_from_motion(d_cam, motion_cam, bearing)
                            tn = float(np.linalg.norm(motion_cam))
                            agl = baseline / tn if tn > 1e-9 else float("nan")
                            mid = (i0 + i1) // 2
                            result.samples.append(VisualSample(
                                index=mid, frame_a=i0, frame_b=i1,
                                pitch=pitch, roll=roll, yaw=yaw, agl_m=agl,
                                inliers=int(mask.sum()),
                                down_cam=(float(d_cam[0]), float(d_cam[1]), float(d_cam[2])),
                                track_bearing=bearing, baseline_m=baseline))
        idx += 1
        if progress is not None and idx % 60 == 0:
            progress(idx)

    container.close()
    logger.info("visual attitude: %d samples from %d frames", len(result.samples), idx)
    return result


# --------------------------------------------------------------------------- #
# Focal-length verification
# --------------------------------------------------------------------------- #

def verify_focal_length(video_path: str, dynamics: TrackDynamics,
                        candidates_35mm: Sequence[float],
                        frame_indices: Sequence[int],
                        gap: int = 8, scale: float = 0.5,
                        image_width: int = 3840,
                        line_min_px: int = 45,
                        tol_deg: float = 1.2) -> List[Tuple[float, float, float]]:
    """
    Score candidate focal lengths against building verticals.

    For a given ``f`` the ground homography predicts exactly where the *vertical*
    vanishing point must fall. Straight edges in the scene are independent evidence
    (they come from image structure, not motion), so the ``f`` whose predicted
    vertical VP attracts the most edge length is the best supported.

    Returns ``[(focal_35mm, support_score, mean_pitch), ...]``.
    """
    import av
    import cv2
    from collections import deque

    want = set(int(i) for i in frame_indices)
    homs, segs = {}, {}
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)

    container = av.open(video_path)
    vstream = container.streams.video[0]
    vstream.thread_type = "AUTO"
    fw = int(vstream.codec_context.width); fh = int(vstream.codec_context.height)
    w, h = int(round(fw * scale)), int(round(fh * scale))
    buf: deque = deque(maxlen=gap + 1)
    idx = 0
    n_tel = len(dynamics.east)

    for frame in container.decode(vstream):
        full = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2GRAY)
        small = cv2.resize(full, (w, h), interpolation=cv2.INTER_AREA) if scale != 1.0 else full
        buf.append((idx, small))
        if idx in want and len(buf) == gap + 1:
            lines = lsd.detect(full)[0]
            s = lines.reshape(-1, 4) if lines is not None else np.empty((0, 4))
            if len(s):
                L = np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1])
                s = s[L >= line_min_px]
            i0, g0 = buf[0]; i1, g1 = buf[-1]
            a, b = _track_features(g0, g1)
            if a is not None:
                H, inl = cv2.findHomography(a, b, cv2.RANSAC, 2.0,
                                            maxIters=5000, confidence=0.9995)
                if H is not None and inl is not None:
                    m = inl.ravel().astype(bool)
                    homs[idx] = (H, a[m][:600], b[m][:600])
                    segs[idx] = s
            want.discard(idx)
        idx += 1
        if not want:
            break
    container.close()

    def support(s: np.ndarray, vp: Tuple[float, float]) -> float:
        if len(s) == 0:
            return 0.0
        mid = np.column_stack([(s[:, 0] + s[:, 2]) / 2, (s[:, 1] + s[:, 3]) / 2])
        dv = s[:, 2:4] - s[:, 0:2]
        ln = np.hypot(dv[:, 0], dv[:, 1])
        u = dv / (ln[:, None] + 1e-12)
        v = np.array(vp) - mid
        nv = np.linalg.norm(v, axis=1)
        ok = nv > 1e-6
        wv = np.zeros_like(v); wv[ok] = v[ok] / nv[ok, None]
        ang = np.degrees(np.arccos(np.abs(np.einsum("ij,ij->i", wv, u)).clip(0, 1)))
        return float(ln[ok & (ang < tol_deg)].sum())

    out = []
    for f35 in candidates_35mm:
        fpx = focal_px_from_35mm(f35, image_width)
        K = intrinsics(fpx * scale, w, h)
        score, pitches = 0.0, []
        for k, (H, pa, pb) in homs.items():
            sol = _decompose_ground_homography(H, K, pa, pb)
            if sol is None:
                continue
            pitch, roll, d, t = sol
            if d[2] <= 1e-6:
                continue
            vp = (fw / 2 + fpx * d[0] / d[2], fh / 2 + fpx * d[1] / d[2])
            score += support(segs.get(k, np.empty((0, 4))), vp)
            pitches.append(pitch)
        out.append((float(f35), score, float(np.mean(pitches)) if pitches else float("nan")))
    return out


# --------------------------------------------------------------------------- #
# Fusion / output
# --------------------------------------------------------------------------- #

@dataclass
class FrameAttitude:
    frame_id: int
    time_sec: float
    timestamp: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    altitude: Optional[float]
    estimated_yaw: Optional[float]      # course over ground (aircraft track)
    ground_speed: Optional[float]
    heading_change: Optional[float]
    vertical_speed: Optional[float]
    camera_yaw: Optional[float] = None  # optical axis azimuth
    camera_pitch: Optional[float] = None
    camera_roll: Optional[float] = None
    agl_m: Optional[float] = None
    attitude_source: str = "none"
    course_quality: Optional[float] = None


def _mad_inliers(values: np.ndarray, mad_k: float = 4.0) -> np.ndarray:
    """
    Robust outlier mask via median absolute deviation.

    A stabilised gimbal does not swing wildly, so large excursions are almost always
    a degenerate homography (near-zero baseline while the aircraft accelerates or
    hovers) rather than real motion. Rejecting them keeps those frames from dragging
    the interpolation at the ends of a flight leg.
    """
    v = np.asarray(values, float)
    good = ~np.isnan(v)
    if good.sum() < 4:
        return good
    med = np.median(v[good])
    mad = np.median(np.abs(v[good] - med))
    if mad < 1e-9:
        return good
    return good & (np.abs(v - med) <= mad_k * 1.4826 * mad)


def _smooth_interp(sample_idx: np.ndarray, values: np.ndarray,
                   target: np.ndarray, median_k: int = 5) -> np.ndarray:
    """Median-filter samples then linearly interpolate onto every frame."""
    if len(sample_idx) == 0:
        return np.full(len(target), np.nan)
    v = values.copy()
    good = ~np.isnan(v)
    if good.sum() == 0:
        return np.full(len(target), np.nan)
    si, v = sample_idx[good], v[good]
    if len(v) >= median_k >= 3:
        k = median_k | 1
        pad = k // 2
        vp = np.pad(v, pad, mode="edge")
        v = np.array([np.median(vp[i:i + k]) for i in range(len(v))])
    return np.interp(target, si, v)


def _smooth_interp_angle(sample_idx: np.ndarray, deg: np.ndarray,
                         target: np.ndarray, median_k: int = 5) -> np.ndarray:
    """Angle-safe variant: interpolate on the unit circle to avoid the 0/360 seam."""
    if len(sample_idx) == 0:
        return np.full(len(target), np.nan)
    good = ~np.isnan(deg)
    if good.sum() == 0:
        return np.full(len(target), np.nan)
    si, d = sample_idx[good], np.radians(deg[good])
    s = _smooth_interp(si, np.sin(d), target, median_k)
    c = _smooth_interp(si, np.cos(d), target, median_k)
    return np.degrees(np.arctan2(s, c)) % 360.0


def fuse(records, dynamics: TrackDynamics,
         visual: Optional[VisualAttitudeResult] = None) -> List[FrameAttitude]:
    """Merge telemetry, track kinematics and (optionally) visual attitude per frame."""
    n = len(records)
    target = np.arange(n, dtype=float)

    cam_p = cam_r = cam_y = agl = np.full(n, np.nan)
    lo = hi = None
    if visual is not None and visual.samples:
        si, ps, rs, ys, ags = visual.as_arrays()
        # Drop degenerate solves before interpolating: a sample must be sane in
        # pitch, roll and scale simultaneously.
        keep = _mad_inliers(ps) & _mad_inliers(rs) & _mad_inliers(ags)
        if keep.sum() >= 4:
            si, ps, rs, ys, ags = si[keep], ps[keep], rs[keep], ys[keep], ags[keep]
        else:
            logger.warning("outlier rejection kept only %d samples; using all", int(keep.sum()))
        n_drop = len(visual.samples) - len(si)
        if n_drop:
            logger.info("rejected %d/%d degenerate visual samples",
                        n_drop, len(visual.samples))
        lo, hi = float(si.min()), float(si.max())
        cam_p = _smooth_interp(si, ps, target)
        cam_r = _smooth_interp(si, rs, target)
        cam_y = _smooth_interp_angle(si, ys, target)
        agl = _smooth_interp(si, ags, target)

    out: List[FrameAttitude] = []
    for i, rec in enumerate(records):
        has_visual = not math.isnan(cam_p[i])
        # Outside the solved span the values are held flat from the nearest sample,
        # so label them rather than passing them off as measurements.
        extrapolated = has_visual and lo is not None and not (lo <= i <= hi)
        source = ("visual-extrapolated" if extrapolated
                  else "visual-homography" if has_visual else "gps-track")
        out.append(FrameAttitude(
            frame_id=rec.frame_cnt if rec.frame_cnt is not None else i + 1,
            time_sec=float(rec.start_sec),
            timestamp=rec.timestamp_str,
            latitude=rec.latitude,
            longitude=rec.longitude,
            altitude=rec.altitude,
            estimated_yaw=None if math.isnan(dynamics.course_deg[i]) else round(float(dynamics.course_deg[i]), 3),
            ground_speed=round(float(dynamics.ground_speed[i]), 4),
            heading_change=None if math.isnan(dynamics.heading_change[i]) else round(float(dynamics.heading_change[i]), 4),
            vertical_speed=round(float(dynamics.vertical_speed[i]), 4),
            camera_yaw=round(float(cam_y[i]), 3) if has_visual and not math.isnan(cam_y[i]) else None,
            camera_pitch=round(float(cam_p[i]), 3) if has_visual else None,
            camera_roll=round(float(cam_r[i]), 3) if has_visual else None,
            agl_m=round(float(agl[i]), 2) if has_visual and not math.isnan(agl[i]) else None,
            attitude_source=source,
            course_quality=round(float(dynamics.course_quality[i]), 3),
        ))
    return out


class AttitudeTrack:
    """
    Per-frame attitude indexed by SRT frame number, for the extraction pipeline.

    Frame numbering matches ``FrameExtractor`` (1-based, aligned with the SRT
    ``FrameCnt``). Lookup falls back to the nearest solved frame so that an
    extraction cadence which lands between samples still gets an orientation.
    """

    def __init__(self, frames: Sequence[FrameAttitude]):
        self._by_id = {f.frame_id: f for f in frames}
        self._ids = np.array(sorted(self._by_id), dtype=int) if self._by_id else np.empty(0, int)

    def __len__(self) -> int:
        return len(self._by_id)

    @property
    def solved_count(self) -> int:
        return sum(1 for f in self._by_id.values()
                   if f.attitude_source.startswith("visual") and f.camera_pitch is not None)

    @classmethod
    def from_csv(cls, path: str) -> "AttitudeTrack":
        """Load a CSV previously written by :func:`write_csv`."""
        import csv

        def num(v):
            if v is None or v == "":
                return None
            try:
                return float(v)
            except ValueError:
                return None

        frames: List[FrameAttitude] = []
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                frames.append(FrameAttitude(
                    frame_id=int(float(row["frame_id"])),
                    time_sec=float(row.get("time_sec") or 0.0),
                    timestamp=row.get("timestamp") or None,
                    latitude=num(row.get("latitude")),
                    longitude=num(row.get("longitude")),
                    altitude=num(row.get("altitude")),
                    estimated_yaw=num(row.get("estimated_yaw")),
                    ground_speed=num(row.get("ground_speed")),
                    heading_change=num(row.get("heading_change")),
                    vertical_speed=num(row.get("vertical_speed")),
                    camera_yaw=num(row.get("camera_yaw")),
                    camera_pitch=num(row.get("camera_pitch")),
                    camera_roll=num(row.get("camera_roll")),
                    agl_m=num(row.get("agl_m")),
                    attitude_source=row.get("attitude_source") or "none",
                    course_quality=num(row.get("course_quality")),
                ))
        logger.info("loaded attitude for %d frames from %s", len(frames), path)
        return cls(frames)

    def get(self, frame_id: int) -> Optional[FrameAttitude]:
        """Exact match, else the nearest solved frame."""
        hit = self._by_id.get(frame_id)
        if hit is not None or len(self._ids) == 0:
            return hit
        pos = int(np.argmin(np.abs(self._ids - frame_id)))
        return self._by_id[int(self._ids[pos])]


CSV_COLUMNS = ["frame_id", "timestamp", "time_sec", "latitude", "longitude", "altitude",
               "estimated_yaw", "ground_speed", "heading_change", "vertical_speed",
               "camera_yaw", "camera_pitch", "camera_roll", "agl_m",
               "attitude_source", "course_quality"]


def write_csv(frames: Sequence[FrameAttitude], path: str) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for fr in frames:
            w.writerow({k: ("" if v is None else v) for k, v in asdict(fr).items()})
    logger.info("wrote %s (%d rows)", path, len(frames))
