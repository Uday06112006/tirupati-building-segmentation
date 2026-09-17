"""Image registration for non-georeferenced before/after orthophotos."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def raster_info(path: Path):
    import rasterio

    with rasterio.open(path) as source:
        return {
            "width": source.width,
            "height": source.height,
            "crs": str(source.crs) if source.crs else None,
            "transform": tuple(source.transform) if source.transform else None,
        }


def read_preview(path: Path, max_size=1800):
    import rasterio

    with rasterio.open(path) as source:
        scale = min(1.0, max_size / max(source.width, source.height))
        image = source.read(
            out_shape=(min(3, source.count), max(1, int(source.height * scale)), max(1, int(source.width * scale))),
            resampling=1,
        )
    image = np.transpose(image, (1, 2, 0))
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] > 3:
        image = image[:, :, :3]
    image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0).astype(np.float32)
    if image.max(initial=0) > 255:
        image *= 255.0 / max(np.percentile(image, 99.5), 1.0)
    return np.clip(image, 0, 255).astype(np.uint8), scale


def _translation_fallback(before, after):
    import cv2

    before_gray = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY).astype(np.float32)
    after_gray = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY).astype(np.float32)
    resized = cv2.resize(after_gray, (before_gray.shape[1], before_gray.shape[0]))
    shift, response = cv2.phaseCorrelate(before_gray, resized)
    matrix = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]], dtype=np.float64)
    return matrix, float(response), "translation"


def register_images(before_path: Path, after_path: Path, method="auto"):
    """Return a matrix mapping AFTER preview coordinates into BEFORE preview coordinates."""
    import cv2

    before_info = raster_info(before_path)
    after_info = raster_info(after_path)
    if before_info["crs"] and after_info["crs"] and before_info["transform"] and after_info["transform"] and before_info["crs"] == after_info["crs"]:
        raise NotImplementedError("Georeferenced registration is not yet enabled; use pixel registration only for these inputs.")
    before, before_scale = read_preview(before_path)
    after, after_scale = read_preview(after_path)
    before_gray = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
    detector = cv2.SIFT_create(nfeatures=6000) if hasattr(cv2, "SIFT_create") else cv2.ORB_create(nfeatures=6000)
    key_before, desc_before = detector.detectAndCompute(before_gray, None)
    key_after, desc_after = detector.detectAndCompute(after_gray, None)
    matrix = None; inlier_ratio = 0.0; used_method = method
    if method in {"auto", "sift", "orb", "homography", "affine"} and desc_before is not None and desc_after is not None:
        norm = cv2.NORM_L2 if hasattr(cv2, "SIFT_create") else cv2.NORM_HAMMING
        matches = cv2.BFMatcher(norm).knnMatch(desc_after, desc_before, k=2)
        good = [first for first, second in matches if first.distance < 0.75 * second.distance]
        if len(good) >= 8:
            source = np.float32([key_after[item.queryIdx].pt for item in good])
            destination = np.float32([key_before[item.trainIdx].pt for item in good])
            if method in {"affine"}:
                matrix, inliers = cv2.estimateAffinePartial2D(source, destination, method=cv2.RANSAC, ransacReprojThreshold=4.0)
            else:
                matrix, inliers = cv2.findHomography(source, destination, cv2.RANSAC, 5.0)
            if matrix is not None and inliers is not None:
                inlier_ratio = float(inliers.mean())
                used_method = "homography" if matrix.shape == (3, 3) else "affine"
    if matrix is None or inlier_ratio < 0.15:
        matrix, response, used_method = _translation_fallback(before, after)
        inlier_ratio = max(inlier_ratio, min(1.0, max(0.0, response)))
    preview_matrix = np.asarray(matrix, dtype=np.float64)
    if preview_matrix.shape == (2, 3):
        preview_matrix = np.vstack([preview_matrix, [0.0, 0.0, 1.0]])
    # Convert preview coordinates to full raster coordinates.
    to_before = np.diag([1.0 / before_scale, 1.0 / before_scale, 1.0])
    from_after = np.diag([after_scale, after_scale, 1.0])
    full_matrix = to_before @ preview_matrix @ from_after
    return {
        "matrix": full_matrix.tolist(),
        "preview_matrix": preview_matrix.tolist(),
        "method": used_method,
        "inlier_ratio": inlier_ratio,
        "before": before_info,
        "after": after_info,
        "before_preview": before,
        "after_preview": after,
        "quality_ok": bool(inlier_ratio >= 0.15),
    }