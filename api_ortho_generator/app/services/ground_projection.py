"""
Ground-plane rectification for obliquely-mounted drone cameras.

The fast engine's original projection is a *similarity* transform: scale, rotate by
yaw, translate. That is exact only for a perfectly nadir camera, because it assumes
one constant scale across the whole frame. A camera tilted forward has a large scale
gradient -- at -38 degrees pitch with a 40 degree vertical field of view the near edge
of the frame lands ~35 m from the aircraft and the far edge ~172 m, a 4.9x ratio. No
scale-and-rotate can reconcile that, which is why oblique frames ghost when mosaicked
that way.

The correct mapping for flat ground is a **homography**. With the camera pose known
(yaw/pitch/roll) and its height above ground, the map from image pixels to the ground
plane is exact up to terrain relief:

    [u v 1]^T  ~  K [r1 | r2 | -H r3] [e n 1]^T

where ``r1, r2, r3`` are the columns of the world->camera rotation and ``(e, n)`` are
metres east/north of the point directly beneath the camera. Inverting that gives
ground-from-image; composing with the canvas geotransform gives a single 3x3 that
``cv2.warpPerspective`` applies in one pass.

Angle convention matches ``drone-dji`` XMP, i.e. what
``api_video_ortho/app/services/attitude_estimator.py`` produces:

* yaw   -- azimuth of the optical axis, degrees clockwise from true north
* pitch -- elevation of the optical axis; 0 = horizon, **-90 = nadir**
* roll  -- rotation about the optical axis, positive = right-hand side down

Camera axes are x right, y down, z along the optical axis.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FULL_FRAME_WIDTH_MM = 36.0


def intrinsics_from_35mm(focal_35mm: float, width: int, height: int) -> np.ndarray:
    """Pinhole K from a 35 mm-equivalent focal length, principal point at centre."""
    f_px = (width / 2.0) / math.tan(math.atan(FULL_FRAME_WIDTH_MM / (2.0 * focal_35mm)))
    return np.array([[f_px, 0.0, width / 2.0],
                     [0.0, f_px, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=float)


def rotation_world_to_camera(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """
    World (ENU) -> camera rotation for the convention documented above.

    Built by composing, in order: point the optical axis north and level, pitch it
    down, roll about it, then swing to the commanded azimuth.
    """
    th = math.radians(pitch_deg)
    ph = math.radians(roll_deg)
    ps = math.radians(yaw_deg)

    # Level camera looking north, then pitched by `th` about its own x (east) axis.
    z = np.array([0.0, math.cos(th), math.sin(th)])     # optical axis
    y = np.array([0.0, math.sin(th), -math.cos(th)])    # image 'down'
    x = np.cross(y, z)                                   # image 'right' (== east here)

    # Roll about the optical axis: positive tips the right-hand side of the frame down.
    xr = math.cos(ph) * x + math.sin(ph) * y
    yr = -math.sin(ph) * x + math.cos(ph) * y

    # Yaw about world up; azimuth measured clockwise from north.
    c, s = math.cos(ps), math.sin(ps)

    def swing(v: np.ndarray) -> np.ndarray:
        return np.array([v[0] * c + v[1] * s, -v[0] * s + v[1] * c, v[2]])

    # Rows of a world->camera rotation are the camera axes in world coordinates.
    return np.vstack([swing(xr), swing(yr), swing(z)])


def homography_image_from_ground(K: np.ndarray, R_wc: np.ndarray, agl_m: float) -> np.ndarray:
    """Map local ground offsets ``(east, north)`` in metres to image pixels."""
    r1, r2, r3 = R_wc[:, 0], R_wc[:, 1], R_wc[:, 2]
    return K @ np.column_stack([r1, r2, -agl_m * r3])


def homography_canvas_from_image(
    K: np.ndarray,
    R_wc: np.ndarray,
    agl_m: float,
    cam_easting: float,
    cam_northing: float,
    min_x: float,
    max_y: float,
    gsd_m: float,
) -> Optional[np.ndarray]:
    """
    Single 3x3 taking source-image pixels straight to canvas pixels.

    Returns ``None`` if the geometry is degenerate (camera on the ground, or the
    image-to-ground map is singular).
    """
    if agl_m is None or agl_m <= 1e-6:
        return None
    H_img_ground = homography_image_from_ground(K, R_wc, agl_m)
    if abs(np.linalg.det(H_img_ground)) < 1e-12:
        return None
    H_ground_img = np.linalg.inv(H_img_ground)

    # Ground offset (e, n) -> canvas pixel. Canvas rows run north-to-south.
    M_canvas_ground = np.array([
        [1.0 / gsd_m, 0.0, (cam_easting - min_x) / gsd_m],
        [0.0, -1.0 / gsd_m, (max_y - cam_northing) / gsd_m],
        [0.0, 0.0, 1.0],
    ])
    return M_canvas_ground @ H_ground_img


def ground_offsets_for_corners(
    K: np.ndarray, R_wc: np.ndarray, agl_m: float,
    width: int, height: int, max_range_m: float,
) -> Optional[np.ndarray]:
    """
    Ground offsets (metres east/north, camera-relative) of the image corners.

    Rays that pass above the horizon, or land beyond ``max_range_m``, are pulled back
    to ``max_range_m`` along their own azimuth, so the returned quad stays finite for
    steeply oblique frames.
    """
    if agl_m is None or agl_m <= 1e-6:
        return None
    H_img_ground = homography_image_from_ground(K, R_wc, agl_m)
    if abs(np.linalg.det(H_img_ground)) < 1e-12:
        return None
    H_ground_img = np.linalg.inv(H_img_ground)

    corners = np.array([[0.0, 0.0, 1.0], [width, 0.0, 1.0],
                        [width, height, 1.0], [0.0, height, 1.0]]).T
    g = H_ground_img @ corners
    out = []
    for i in range(g.shape[1]):
        w = g[2, i]
        if abs(w) < 1e-9:                       # ray parallel to the ground plane
            continue
        e, n = g[0, i] / w, g[1, i] / w
        r = math.hypot(e, n)
        if not np.isfinite(r):
            continue
        if w < 0:
            # Point is behind the camera: the ray went above the horizon. Keep its
            # azimuth but clamp the range.
            if r > 1e-9:
                e, n = -e / r * max_range_m, -n / r * max_range_m
            else:
                continue
        elif r > max_range_m:
            e, n = e / r * max_range_m, n / r * max_range_m
        out.append((e, n))
    return np.array(out) if len(out) >= 3 else None


def local_scale(M: np.ndarray, x: float, y: float) -> float:
    """
    Canvas pixels per source pixel at source point ``(x, y)`` under homography ``M``.

    A homography's scale varies across the frame, so this is the Jacobian determinant
    at a point rather than a single global factor. Used to decide how much to
    prefilter a frame before warping: sampling a 4K frame straight down to a coarser
    mosaic with bilinear interpolation aliases badly, which reads as blur.
    """
    d = M[2, 0] * x + M[2, 1] * y + M[2, 2]
    if abs(d) < 1e-12:
        return 0.0
    u = (M[0, 0] * x + M[0, 1] * y + M[0, 2]) / d
    v = (M[1, 0] * x + M[1, 1] * y + M[1, 2]) / d
    j = np.array([[M[0, 0] - u * M[2, 0], M[0, 1] - u * M[2, 1]],
                  [M[1, 0] - v * M[2, 0], M[1, 1] - v * M[2, 1]]]) / d
    return float(math.sqrt(abs(np.linalg.det(j))))


def usable_range_factor(nadir_gsd_m: float, target_gsd_m: float,
                        lo: float = 0.4, hi: float = 4.0) -> float:
    """
    How far from the aircraft a tilted frame still resolves at the target GSD.

    Looking out at incidence angle ``theta``, ground range is ``H tan(theta)`` and the
    along-range sample spacing stretches to ``(H/f) * sin(theta)/cos^2(theta)``. Setting
    that equal to the target GSD and solving for ``theta`` gives the range past which a
    frame can no longer honour the requested resolution -- beyond it, pixels only add
    blur. Returned as a multiple of flying height, clamped to ``[lo, hi]``.

    At 55 m AGL with f=2987 px the nadir GSD is 1.84 cm, so a 5 cm target yields
    ``k = 2.72`` -> theta ~ 56 degrees -> ~1.5x height.
    """
    if nadir_gsd_m <= 0 or target_gsd_m <= 0:
        return hi
    k = target_gsd_m / nadir_gsd_m
    if k <= 1.0:                      # target finer than the sensor can ever deliver
        return lo
    a, b = 0.0, math.radians(85.0)
    for _ in range(60):               # bisection on sin(t)/cos^2(t) = k
        m = 0.5 * (a + b)
        if math.sin(m) / (math.cos(m) ** 2) < k:
            a = m
        else:
            b = m
    return float(min(hi, max(lo, math.tan(0.5 * (a + b)))))


def nadir_weight_map(
    shape: Tuple[int, int],
    cam_easting: float, cam_northing: float, agl_m: float,
    min_x: float, max_y: float, gsd_m: float,
    max_range_m: float, roi: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """
    Per-pixel blending weight ``cos^3(theta)``, theta measured from nadir.

    Ground resolution degrades with the cube of the obliquity, so this prefers the
    part of each frame that was imaged most steeply -- the standard nadir weighting.
    Pixels beyond ``max_range_m`` get zero weight, which trims the badly foreshortened
    far field that an oblique frame would otherwise smear across the mosaic.
    """
    h, w = shape
    y0, y1, x0, x1 = roi if roi else (0, h, 0, w)
    ys = np.arange(y0, y1, dtype=np.float32)
    xs = np.arange(x0, x1, dtype=np.float32)
    east = (min_x + (xs + 0.5) * gsd_m) - cam_easting
    north = (max_y - (ys + 0.5) * gsd_m) - cam_northing
    r2 = north[:, None] ** 2 + east[None, :] ** 2
    cos_t = agl_m / np.sqrt(r2 + agl_m * agl_m)
    weight = cos_t ** 3
    weight[r2 > max_range_m * max_range_m] = 0.0
    return weight.astype(np.float32)
