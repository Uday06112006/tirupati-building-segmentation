import math
import logging
from typing import List, Tuple, Optional, Callable, Dict, Any
import numpy as np
import cv2
from PIL import Image

from .exif_reader import ImageGeoInfo
from .coordinates import CoordinateService
from .ground_projection import (
    intrinsics_from_35mm, rotation_world_to_camera,
    homography_canvas_from_image, nadir_weight_map, usable_range_factor, local_scale,
)
from ..models.schemas import BlendMode

logger = logging.getLogger(__name__)

class OrthoMosaicEngine:
    """
    High-performance 2D Orthomosaic generator.
    Projects, aligns, and blends aerial drone frames into a continuous georeferenced orthophoto.
    """

    # Seam cross-fade width in metres. Kept small on purpose: a wide fade re-introduces
    # exactly the averaging blur that best-frame selection exists to avoid.
    SEAM_FEATHER_M = 0.6
    # Canvas pixels of each frame's rim excluded from ownership (interpolation there
    # mixes in the black warp border).
    EDGE_EXCLUDE_PX = 4
    # Seams are chosen on a downscaled copy. Seam placement only needs to be accurate
    # to a metre or so -- the feather covers the rest -- and graph cut over the full
    # canvas would be needlessly slow.
    SEAM_SCALE = 0.4

    @staticmethod
    def _spatial_meta(crs, min_x, min_y, max_x, max_y, canvas_w, canvas_h,
                      gsd_m, total_images) -> Dict[str, Any]:
        inv = CoordinateService.get_transformer(crs, "EPSG:4326")
        min_lon, min_lat = inv.transform(min_x, min_y)
        max_lon, max_lat = inv.transform(max_x, max_y)
        return {
            "crs": crs, "min_x": min_x, "max_y": max_y, "max_x": max_x, "min_y": min_y,
            "width_px": canvas_w, "height_px": canvas_h,
            "gsd_m": gsd_m, "gsd_cm": round(gsd_m * 100.0, 2),
            "bounds_utm": [round(min_x, 2), round(min_y, 2), round(max_x, 2), round(max_y, 2)],
            "bounds_wgs84": [round(min_lon, 6), round(min_lat, 6),
                             round(max_lon, 6), round(max_lat, 6)],
            "total_images_processed": total_images,
        }

    @classmethod
    def _frame_homography(cls, img_info: ImageGeoInfo, transformer,
                          min_x: float, max_y: float, gsd_m: float):
        """Canvas-from-image homography for one posed frame, or None."""
        cx, cy = transformer.transform(img_info.longitude, img_info.latitude)
        K = intrinsics_from_35mm(img_info.focal_35mm or img_info.focal_length_mm,
                                 img_info.width, img_info.height)
        R = rotation_world_to_camera(img_info.yaw_deg, img_info.pitch_deg, img_info.roll_deg)
        M = homography_canvas_from_image(K, R, img_info.agl_m, cx, cy, min_x, max_y, gsd_m)
        return M, cx, cy

    @classmethod
    def _warp_prefiltered(cls, img_rgb: np.ndarray, M: np.ndarray,
                          canvas_w: int, canvas_h: int) -> np.ndarray:
        """
        Warp a frame to the canvas, prefiltering first if it is being minified.

        ``warpPerspective`` has no INTER_AREA, so sampling a 4K frame down to a coarser
        mosaic with bilinear interpolation aliases -- fine detail turns to mush. Shrink
        with INTER_AREA first (a proper box filter) and fold the shrink into ``M``.
        """
        h, w = img_rgb.shape[:2]
        s = np.median([local_scale(M, x, y)
                       for x in (w * 0.25, w * 0.5, w * 0.75)
                       for y in (h * 0.5, h * 0.7, h * 0.9)])
        if 0.05 < s < 0.7:
            nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
            small = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            S_inv = np.array([[w / nw, 0, 0], [0, h / nh, 0], [0, 0, 1]], float)
            # After INTER_AREA the frame is already near 1:1 with the canvas, so linear
            # resampling is enough and avoids cubic's overshoot halos on hard edges.
            return cv2.warpPerspective(small, M @ S_inv, (canvas_w, canvas_h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return cv2.warpPerspective(img_rgb, M, (canvas_w, canvas_h), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    @staticmethod
    def _solve_gains(warps: Dict[int, np.ndarray], sels: Dict[int, np.ndarray],
                     label: np.ndarray, feather_px: float,
                     clamp: Tuple[float, float] = (0.82, 1.22)) -> Dict[int, float]:
        """
        Per-frame brightness gains that flatten exposure steps across seams.

        The clip runs through auto-exposure changes (shutter moves 1/640 -> 1/500),
        so neighbouring frames disagree on brightness and best-frame selection turns
        that disagreement into a visible edge along every seam. Each frame is compared
        against its neighbours *only in the strip either side of their shared seam*,
        where both genuinely observed the same ground -- comparing whole-frame means
        would instead 'correct' for one frame simply containing more dark trees.

        Gains are the least-squares fit to those pairwise ratios in log space, anchored
        so the mean gain is 1 (the mosaic keeps its overall exposure) and clamped so a
        bad pair can never blow out a frame.
        """
        idxs = sorted(warps)
        if len(idxs) < 2:
            return {i: 1.0 for i in idxs}
        band = int(max(2.0, feather_px * 4))
        kern = np.ones((band * 2 + 1,) * 2, np.uint8)

        # Work inside each owner's bounding box. Dilating the full canvas once per
        # ordered pair is O(n^2) over millions of pixels and dominates runtime; seam
        # neighbourhoods are tiny by comparison and only touching frames can share one.
        boxes: Dict[int, Tuple[int, int, int, int]] = {}
        for i in idxs:
            ys, xs = np.nonzero(sels[i])
            boxes[i] = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)

        pos = {i: k for k, i in enumerate(idxs)}
        rows, rhs, wts = [], [], []
        # Only nearby frames in capture order can share a seam: the aircraft flies a
        # track, so ownership bands are laid down in sequence. Checking all pairs is
        # quadratic for no gain -- frame 1 and frame 25 never touch.
        NEIGHBOURS = 2
        for ai, a in enumerate(idxs):
            ay0, ay1, ax0, ax1 = boxes[a]
            for b in idxs[ai + 1: ai + 1 + NEIGHBOURS]:
                by0, by1, bx0, bx1 = boxes[b]
                # Intersection of the two boxes, padded by the seam band.
                y0, y1 = max(0, max(ay0, by0) - band), min(label.shape[0], min(ay1, by1) + band)
                x0, x1 = max(0, max(ax0, bx0) - band), min(label.shape[1], min(ax1, bx1) + band)
                if y1 - y0 < 3 or x1 - x0 < 3:
                    continue
                sa = sels[a][y0:y1, x0:x1]
                sb = sels[b][y0:y1, x0:x1]
                near_a = cv2.dilate(sa.astype(np.uint8), kern) > 0
                shared_b = near_a & sb
                if shared_b.sum() < 400:
                    continue
                shared_a = (cv2.dilate(sb.astype(np.uint8), kern) > 0) & sa
                if shared_a.sum() < 400:
                    continue
                la = warps[a][y0:y1, x0:x1].astype(np.float32).mean(axis=2)
                lb = warps[b][y0:y1, x0:x1].astype(np.float32).mean(axis=2)
                ma, mb = la[shared_a].mean(), lb[shared_b].mean()
                if ma < 4 or mb < 4:
                    continue
                r = np.zeros(len(idxs), np.float64)
                r[pos[a]], r[pos[b]] = 1.0, -1.0
                rows.append(r)
                rhs.append(math.log(mb / ma))
                wts.append(math.sqrt(min(shared_b.sum(), shared_a.sum())))
        if not rows:
            return {i: 1.0 for i in idxs}

        A = np.array(rows) * np.array(wts)[:, None]
        y = np.array(rhs) * np.array(wts)
        # Anchor: mean log-gain = 0, so overall mosaic exposure is preserved.
        A = np.vstack([A, np.ones(len(idxs)) * len(idxs)])
        y = np.append(y, 0.0)
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        gains = {i: float(np.clip(math.exp(sol[pos[i]]), *clamp)) for i in idxs}
        spread = max(gains.values()) / max(1e-6, min(gains.values()))
        logger.info("Exposure gains solved over %d seam pairs, spread %.3f", len(rows), spread)
        return gains

    @classmethod
    def _route_seams(cls, small, corners, sizes, order):
        """
        Choose seam paths that follow ground rather than cutting through buildings.

        Nadir-weight argmax alone puts the seam wherever two frames happen to trade
        places, which is frequently straight across a rooftop -- and because a planar
        mosaic misplaces anything with height, the two frames put that roof metres
        apart, so the building arrives severed and doubled.

        A graph cut instead pays a price proportional to how much the two frames
        *disagree* along the boundary, so the seam is pushed into regions where they
        agree -- flat ground -- and routed around structures. The lean stays (only a
        DSM removes that) but each building is served whole by one frame.

        Operates on downscaled ROI crops; returns refined masks in the same order.
        """
        masks = [cv2.UMat(m) for _, m in small]
        imgs = [im.astype(np.float32) / 255.0 for im, _ in small]
        try:
            finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
            finder.find(imgs, corners, masks)
            out = [m.get() for m in masks]
            logger.info("Graph-cut seam routing over %d frames", len(imgs))
            return out
        except cv2.error as e:
            logger.warning("Graph cut unavailable (%s); falling back to Voronoi seams", e)
            try:
                masks = [cv2.UMat(m) for _, m in small]
                cv2.detail_VoronoiSeamFinder().find(imgs, corners, masks)
                return [m.get() for m in masks]
            except cv2.error:
                return [m for _, m in small]

    @classmethod
    def _composite_best_frame(cls, images, transformer, min_x, max_y,
                              canvas_w, canvas_h, gsd_m, max_range_m,
                              progress_callback=None):
        """
        Mosaic by picking, for every output pixel, the single frame that saw it most
        steeply -- rather than averaging every frame that covers it.

        This matters far more than it sounds. A planar mosaic assumes the world is
        flat, so anything with height lands in the wrong place by ``h*tan(theta)``:
        at 55 m AGL and 52 degrees off nadir a 7 m roof is displaced ~9 m, and it
        displaces a *different way in every frame*. Averaging 28 such views smears
        every roof and treetop across ~180 px while leaving the road (height 0) sharp
        -- which is exactly the blur seen in the first attempt.

        Choosing one frame per pixel removes the smear outright. Picking the *most
        nadir* frame additionally minimises the residual lean, since displacement
        scales with ``tan(theta)``. Buildings still lean -- only a DSM can remove that,
        which is what the ``accurate`` engine is for -- but they lean crisply instead
        of dissolving.
        """
        n = len(images)
        best_score = np.zeros((canvas_h, canvas_w), np.float32)
        label = np.full((canvas_h, canvas_w), -1, np.int16)
        cache = {}

        # Pass 1 -- decide ownership. Only masks and analytic weights, no RGB yet.
        for idx, info in enumerate(images):
            if not info.pose_is_known:
                continue
            M, cx, cy = cls._frame_homography(info, transformer, min_x, max_y, gsd_m)
            if M is None:
                continue
            cache[idx] = (M, cx, cy)
            cover = cv2.warpPerspective(
                np.full((info.height, info.width), 255, np.uint8), M,
                (canvas_w, canvas_h), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            # Pull ownership back from the frame edge. Interpolation there samples the
            # black border, so a pixel owned right at the rim comes out darkened --
            # which shows up as a thin dark line along every seam.
            cover = cv2.erode(cover, np.ones((cls.EDGE_EXCLUDE_PX * 2 + 1,) * 2, np.uint8))
            score = nadir_weight_map((canvas_h, canvas_w), cx, cy, info.agl_m,
                                     min_x, max_y, gsd_m,
                                     max_range_m or (2.5 * info.agl_m))
            score[cover == 0] = 0.0
            win = score > best_score
            best_score[win] = score[win]
            label[win] = idx
            if progress_callback:
                progress_callback(10.0 + (idx / n) * 35.0,
                                  f"Selecting best view {idx+1}/{n} ({info.filename})...")

        feather_px = max(1.0, cls.SEAM_FEATHER_M / gsd_m)

        # Pass 2 -- re-cut the seams. The partition above is geometry-only: it hands
        # over wherever two frames trade nadir advantage, which lands on rooftops as
        # often as not. Graph cut instead prices each boundary by how much the two
        # frames disagree there, pushing seams onto ground they agree on and around
        # structures they do not.
        sc = cls.SEAM_SCALE
        sw, sh = max(1, int(round(canvas_w * sc))), max(1, int(round(canvas_h * sc)))
        S = np.array([[sc, 0, 0], [0, sc, 0], [0, 0, 1]], float)
        gsd_s = gsd_m / sc
        order, small, corners, small_sel = [], [], [], {}
        small_warps: Dict[int, np.ndarray] = {}

        for k, idx in enumerate(sorted(cache)):
            info = images[idx]
            M, cx, cy = cache[idx]
            bgr = cv2.imread(info.file_path)
            if bgr is None:
                continue
            rgb_s = cls._warp_prefiltered(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                                          S @ M, sw, sh)
            cov_s = cv2.warpPerspective(
                np.full((info.height, info.width), 255, np.uint8), S @ M, (sw, sh),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            er = max(1, int(round(cls.EDGE_EXCLUDE_PX * sc)))
            cov_s = cv2.erode(cov_s, np.ones((er * 2 + 1,) * 2, np.uint8))
            sco_s = nadir_weight_map((sh, sw), cx, cy, info.agl_m, min_x, max_y, gsd_s,
                                     max_range_m or (2.5 * info.agl_m))
            cov_s[sco_s <= 0] = 0
            if not cov_s.any():
                continue
            ys, xs = np.nonzero(cov_s)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            order.append(idx)
            corners.append((int(x0), int(y0)))
            small.append((rgb_s[y0:y1, x0:x1], cov_s[y0:y1, x0:x1].copy()))
            small_sel[idx] = (y0, y1, x0, x1)
            small_warps[idx] = rgb_s
            if progress_callback:
                progress_callback(45.0 + (k / n) * 20.0,
                                  f"Preparing seams {k+1}/{n} ({info.filename})...")

        refined = cls._route_seams(small, corners, None, order)

        # Rebuild ownership from the routed masks. Where the cut left a pixel to more
        # than one frame, or to none, fall back to the nadir-weight winner.
        lab_s = np.full((sh, sw), -1, np.int32)
        best_s = np.zeros((sh, sw), np.float32)
        for k, idx in enumerate(order):
            y0, y1, x0, x1 = small_sel[idx]
            m = refined[k]
            if m is None or not np.any(m):
                continue
            info = images[idx]
            _, cx, cy = cache[idx]
            sco = nadir_weight_map((sh, sw), cx, cy, info.agl_m, min_x, max_y, gsd_s,
                                   max_range_m or (2.5 * info.agl_m))
            claim = np.zeros((sh, sw), bool)
            claim[y0:y1, x0:x1] = m > 0
            take = claim & (sco > best_s)
            best_s[take] = sco[take]
            lab_s[take] = idx

        # Upscaling the seam labels nearest-neighbour leaves stair-steps the width of
        # the seam-scale factor. A median filter over the label field snaps those back
        # to a smooth boundary; the median of a set of labels is always one of them, so
        # no pixel is handed to a frame that never claimed it.
        up = cv2.resize((lab_s + 1).astype(np.uint8), (canvas_w, canvas_h),
                        interpolation=cv2.INTER_NEAREST)
        k = int(round(1.0 / cls.SEAM_SCALE)) * 2 + 1
        up = cv2.medianBlur(up, k if k % 2 else k + 1)
        grown = up.astype(np.int16) - 1
        # Never claim a pixel the frame does not actually cover.
        keep = grown >= 0
        label = np.where(keep & (grown != label), grown, label).astype(np.int16)

        sels = {idx: (label == idx) for idx in order}
        sels = {i: m for i, m in sels.items() if m.any()}
        gains = cls._solve_gains(
            {i: small_warps[i] for i in sels},
            {i: cv2.resize(sels[i].astype(np.uint8), (sw, sh),
                           interpolation=cv2.INTER_NEAREST).astype(bool) for i in sels},
            lab_s, max(1.0, feather_px * sc))
        del small, small_warps, refined

        accum_rgb = np.zeros((canvas_h, canvas_w, 3), np.float32)
        accum_w = np.zeros((canvas_h, canvas_w), np.float32)

        # Pass 3 -- each frame paints only the pixels it owns, cross-fading at seams.
        # Full-resolution warps are streamed one at a time; holding all of them would
        # cost ~1.3 GB on a canvas this size for no benefit.
        for idx in sorted(sels):
            sel = sels[idx]
            bgr = cv2.imread(images[idx].file_path)
            if bgr is None:
                continue
            M, _, _ = cache[idx]
            warped = cls._warp_prefiltered(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), M, canvas_w, canvas_h) * gains[idx]

            dist = cv2.distanceTransform(sel.astype(np.uint8), cv2.DIST_L2, 5)
            wgt = np.minimum(dist / feather_px, 1.0).astype(np.float32)
            wgt[sel & (wgt < 1e-3)] = 1e-3          # never zero out an owned pixel
            wgt[~sel] = 0.0

            accum_rgb += warped.astype(np.float32) * wgt[:, :, None]
            accum_w += wgt
            if progress_callback:
                progress_callback(75.0 + (idx / n) * 15.0,
                                  f"Compositing frame {idx+1}/{n}...")

        if progress_callback:
            progress_callback(92.0, "Normalizing blended orthomosaic and generating transparency mask...")

        valid = accum_w > 1e-6
        final_rgb = np.zeros((canvas_h, canvas_w, 3), np.uint8)
        for c in range(3):
            chan = np.zeros((canvas_h, canvas_w), np.float32)
            chan[valid] = accum_rgb[:, :, c][valid] / accum_w[valid]
            final_rgb[:, :, c] = np.clip(chan, 0, 255).astype(np.uint8)

        owned = int((label >= 0).sum())
        logger.info("Best-frame composite: %d frames own %.2f ha; median views/px reduced to 1",
                    len(cache), owned * gsd_m * gsd_m / 10000.0)
        return final_rgb, (valid * 255).astype(np.uint8)

    @classmethod
    def generate_mosaic(
        cls,
        images: List[ImageGeoInfo],
        target_gsd_m: float = 0.05,  # 5 cm/pixel
        blend_mode: BlendMode = BlendMode.MULTIBAND,
        target_crs: str = "AUTO",
        progress_callback: Optional[Callable[[float, str], None]] = None,
        max_range_factor: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Generate orthomosaic RGB image and alpha mask.
        Returns (rgb_mosaic_uint8, alpha_mask_uint8, spatial_metadata).

        When frames carry a full pose (gimbal pitch/roll plus height above ground)
        each one is rectified onto the ground plane with a homography, which is the
        correct mapping for a tilted camera. Frames without a pose fall back to the
        original nadir similarity transform.

        ``max_range_factor`` limits usable ground range to that multiple of the flying
        height. An oblique frame's far field is so foreshortened that including it only
        smears the mosaic; with high along-track overlap it is redundant anyway. Left
        at ``None`` it is derived from the requested GSD -- see
        :func:`ground_projection.usable_range_factor`.
        """
        if not images:
            raise ValueError("No images provided for orthomosaic generation.")

        total_images = len(images)
        if progress_callback:
            progress_callback(5.0, f"Calculating spatial extent for {total_images} frames...")

        posed = [i for i in images if i.pose_is_known]
        use_homography = len(posed) > 0
        median_agl = float(np.median([i.agl_m for i in posed])) if posed else 0.0
        max_range_m = None
        if median_agl > 0:
            factor = max_range_factor
            if factor is None:
                ref = posed[len(posed) // 2]
                _, _, nadir_gsd = CoordinateService.calculate_ground_footprint(ref)
                factor = usable_range_factor(nadir_gsd, target_gsd_m)
                logger.info("Nadir GSD %.2f cm/px vs target %.2f cm/px -> usable range "
                            "%.2fx height", nadir_gsd * 100, target_gsd_m * 100, factor)
            max_range_m = factor * median_agl
        if use_homography:
            oblique = sum(1 for i in posed if not i.is_nadir)
            logger.info(
                "Pose available for %d/%d frames (%d oblique); rectifying by ground-plane "
                "homography, AGL median %.1f m, usable range %.0f m",
                len(posed), total_images, oblique, median_agl, max_range_m or 0.0)
        else:
            logger.info("No gimbal pose found; falling back to nadir similarity transform")

        # 1. Compute bounding box and UTM CRS
        min_x, min_y, max_x, max_y, crs = CoordinateService.compute_projected_bounds(
            images, target_crs, max_range_m=max_range_m)
        transformer = CoordinateService.get_transformer("EPSG:4326", crs)

        # 2. Compute canvas pixel dimensions
        width_m = max_x - min_x
        height_m = max_y - min_y
        
        canvas_w = int(math.ceil(width_m / target_gsd_m))
        canvas_h = int(math.ceil(height_m / target_gsd_m))

        # Memory safety cap (e.g. max 16000x16000)
        max_dim = 16000
        if canvas_w > max_dim or canvas_h > max_dim:
            scale_factor = max_dim / max(canvas_w, canvas_h)
            target_gsd_m = target_gsd_m / scale_factor
            canvas_w = int(math.ceil(width_m / target_gsd_m))
            canvas_h = int(math.ceil(height_m / target_gsd_m))
            logger.info(f"Adjusted GSD to {target_gsd_m:.4f} m/px to fit memory bounds: {canvas_w}x{canvas_h}")

        logger.info(f"Allocating mosaic canvas: {canvas_w}x{canvas_h} pixels at GSD {target_gsd_m*100:.2f} cm/px")

        # Accumulator arrays (float32 for seamless blending)
        accum_rgb = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        accum_weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        if use_homography and blend_mode in (BlendMode.MULTIBAND, BlendMode.FEATHER):
            final_rgb, final_alpha = cls._composite_best_frame(
                images, transformer, min_x, max_y, canvas_w, canvas_h,
                target_gsd_m, max_range_m, progress_callback)
            return final_rgb, final_alpha, cls._spatial_meta(
                crs, min_x, min_y, max_x, max_y, canvas_w, canvas_h,
                target_gsd_m, total_images)

        # Process each image
        for idx, img_info in enumerate(images):
            if progress_callback:
                pct = 10.0 + (idx / total_images) * 80.0
                progress_callback(pct, f"Projecting & blending frame {idx+1}/{total_images} ({img_info.filename})...")

            # Load image
            img_bgr = cv2.imread(img_info.file_path)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            # Projected UTM Center
            cx, cy = transformer.transform(img_info.longitude, img_info.latitude)

            # --- Oblique-capable path: rectify onto the ground plane ----------------
            if use_homography and img_info.pose_is_known:
                K = intrinsics_from_35mm(
                    img_info.focal_35mm or img_info.focal_length_mm, w, h)
                R = rotation_world_to_camera(
                    img_info.yaw_deg, img_info.pitch_deg, img_info.roll_deg)
                M = homography_canvas_from_image(
                    K, R, img_info.agl_m, cx, cy, min_x, max_y, target_gsd_m)
                if M is not None:
                    warped_rgb = cv2.warpPerspective(
                        img_rgb, M, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                    binary_mask = cv2.warpPerspective(
                        np.full((h, w), 255, np.uint8), M, (canvas_w, canvas_h),
                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0)

                    weight = nadir_weight_map(
                        (canvas_h, canvas_w), cx, cy, img_info.agl_m,
                        min_x, max_y, target_gsd_m,
                        max_range_m or (2.5 * img_info.agl_m))
                    weight = weight * (binary_mask > 0)

                    if blend_mode in (BlendMode.MULTIBAND, BlendMode.FEATHER):
                        # Taper the seam so adjacent frames cross-fade.
                        dist = cv2.distanceTransform(
                            (weight > 0).astype(np.uint8) * 255, cv2.DIST_L2, 5)
                        feather = np.minimum(dist / max(1.0, 0.15 / target_gsd_m), 1.0)
                        weight = weight * feather.astype(np.float32)

                    weight_3d = np.repeat(weight[:, :, np.newaxis], 3, axis=2)
                    accum_rgb += warped_rgb.astype(np.float32) * weight_3d
                    accum_weight += weight
                    continue
                logger.warning("Degenerate ground homography for %s; using similarity",
                               img_info.filename)

            # --- Original nadir similarity path ------------------------------------
            # Ground dimensions & nominal GSD
            gw, gh, nominal_gsd = CoordinateService.calculate_ground_footprint(img_info)
            scale = nominal_gsd / target_gsd_m

            # Canvas center in pixel coordinates
            pcx = (cx - min_x) / target_gsd_m
            pcy = (max_y - cy) / target_gsd_m

            # Rotation (Yaw)
            # In image coordinates, Y is down (inverted relative to UTM North)
            yaw_rad = math.radians(img_info.yaw_deg)
            cos_a = math.cos(yaw_rad)
            sin_a = math.sin(yaw_rad)

            # Transformation matrix: Scale, Rotation, Translation
            # Maps point (x, y) in source image to canvas (px, py)
            alpha = scale * cos_a
            beta = scale * sin_a

            src_center_x = w / 2.0
            src_center_y = h / 2.0

            tx = pcx - (alpha * src_center_x - beta * src_center_y)
            ty = pcy - (beta * src_center_x + alpha * src_center_y)

            M = np.array([
                [alpha, -beta, tx],
                [beta,  alpha, ty]
            ], dtype=np.float32)

            # Warp image to canvas coordinates
            warped_rgb = cv2.warpAffine(
                img_rgb,
                M,
                (canvas_w, canvas_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

            # Create binary mask of warped image
            binary_mask = cv2.warpAffine(
                np.ones((h, w), dtype=np.uint8) * 255,
                M,
                (canvas_w, canvas_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

            if blend_mode in (BlendMode.MULTIBAND, BlendMode.FEATHER):
                # Distance transform for smooth feathering
                dist = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
                max_dist = dist.max()
                if max_dist > 0:
                    weight = dist / max_dist
                else:
                    weight = (binary_mask > 0).astype(np.float32)
            else:
                weight = (binary_mask > 0).astype(np.float32)

            # Accumulate weighted RGB
            weight_3d = np.repeat(weight[:, :, np.newaxis], 3, axis=2)
            accum_rgb += warped_rgb.astype(np.float32) * weight_3d
            accum_weight += weight

        if progress_callback:
            progress_callback(92.0, "Normalizing blended orthomosaic and generating transparency mask...")

        # Normalize accumulated RGB by total weights
        valid_mask = accum_weight > 1e-4
        final_rgb = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        for c in range(3):
            chan = np.zeros((canvas_h, canvas_w), dtype=np.float32)
            chan[valid_mask] = accum_rgb[:, :, c][valid_mask] / accum_weight[valid_mask]
            final_rgb[:, :, c] = np.clip(chan, 0, 255).astype(np.uint8)

        final_alpha = (valid_mask * 255).astype(np.uint8)

        # Compute inverse bounds in WGS-84
        inv_transformer = CoordinateService.get_transformer(crs, "EPSG:4326")
        min_lon, min_lat = inv_transformer.transform(min_x, min_y)
        max_lon, max_lat = inv_transformer.transform(max_x, max_y)

        spatial_meta = {
            "crs": crs,
            "min_x": min_x,
            "max_y": max_y,
            "max_x": max_x,
            "min_y": min_y,
            "width_px": canvas_w,
            "height_px": canvas_h,
            "gsd_m": target_gsd_m,
            "gsd_cm": round(target_gsd_m * 100.0, 2),
            "bounds_utm": [round(min_x, 2), round(min_y, 2), round(max_x, 2), round(max_y, 2)],
            "bounds_wgs84": [round(min_lon, 6), round(min_lat, 6), round(max_lon, 6), round(max_lat, 6)],
            "total_images_processed": total_images
        }

        logger.info(f"Orthomosaic generation completed: {canvas_w}x{canvas_h} px, GSD {spatial_meta['gsd_cm']} cm/px")
        return final_rgb, final_alpha, spatial_meta
