"""One-to-one before/after instance matching in BEFORE pixel coordinates."""

from __future__ import annotations

import math

import cv2
import numpy as np


def transform_detection(detection, matrix, output_shape):
    """Warp one sparse AFTER detection into the BEFORE image coordinate space."""
    mask = detection["mask"].astype(np.uint8)
    x1, y1, x2, y2 = detection["box"]
    corners = np.float32([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]])
    transformed = cv2.perspectiveTransform(corners, np.asarray(matrix, dtype=np.float64))[0]
    ox, oy = np.floor(transformed.min(axis=0)).astype(int)
    ex, ey = np.ceil(transformed.max(axis=0)).astype(int) + 1
    width, height = max(1, ex - ox), max(1, ey - oy)
    input_to_global = np.array([[1.0, 0.0, x1], [0.0, 1.0, y1], [0.0, 0.0, 1.0]])
    global_to_output = np.array([[1.0, 0.0, -ox], [0.0, 1.0, -oy], [0.0, 0.0, 1.0]])
    local_matrix = global_to_output @ np.asarray(matrix) @ input_to_global
    warped = cv2.warpPerspective(mask, local_matrix, (width, height), flags=cv2.INTER_NEAREST) > 0
    ys, xs = np.where(warped)
    if not len(xs):
        return None
    global_box = [int(ox + xs.min()), int(oy + ys.min()), int(ox + xs.max() + 1), int(oy + ys.max() + 1)]
    image_height, image_width = output_shape
    clipped = [max(0, global_box[0]), max(0, global_box[1]), min(image_width, global_box[2]), min(image_height, global_box[3])]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    crop_x1, crop_y1 = clipped[0] - ox, clipped[1] - oy
    crop_x2, crop_y2 = clipped[2] - ox, clipped[3] - oy
    result = dict(detection)
    result["mask"] = warped[crop_y1:crop_y2, crop_x1:crop_x2]
    result["box"] = clipped
    result["coordinate_system"] = "before_image_pixels"
    result["original_box"] = detection["box"]
    result["original_instance_id"] = detection["instance_id"]
    result["image_shape"] = list(output_shape)
    return result


def _intersection_metrics(first, second):
    xa, ya = max(first["box"][0], second["box"][0]), max(first["box"][1], second["box"][1])
    xb, yb = min(first["box"][2], second["box"][2]), min(first["box"][3], second["box"][3])
    if xb <= xa or yb <= ya:
        return 0.0, 0.0
    first_mask = first["mask"][ya - first["box"][1]:yb - first["box"][1], xa - first["box"][0]:xb - first["box"][0]]
    second_mask = second["mask"][ya - second["box"][1]:yb - second["box"][1], xa - second["box"][0]:xb - second["box"][0]]
    intersection = np.logical_and(first_mask, second_mask).sum()
    union = first["mask"].sum() + second["mask"].sum() - intersection
    box_intersection = (xb - xa) * (yb - ya)
    box_area = _box_area(first["box"]) + _box_area(second["box"]) - box_intersection
    return float(intersection / union) if union else 0.0, float(box_intersection / box_area) if box_area else 0.0


def _box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _centroid(detection):
    ys, xs = np.where(detection["mask"])
    if not len(xs):
        return np.array([(detection["box"][0] + detection["box"][2]) / 2, (detection["box"][1] + detection["box"][3]) / 2])
    return np.array([xs.mean() + detection["box"][0], ys.mean() + detection["box"][1]])


def match_instances(before, after, mask_iou_threshold=0.1, centroid_distance=250.0):
    candidates = []
    for before_index, first in enumerate(before["instances"]):
        for after_index, second in enumerate(after["instances"]):
            mask_iou, box_iou = _intersection_metrics(first, second)
            distance = float(np.linalg.norm(_centroid(first) - _centroid(second)))
            if (mask_iou >= mask_iou_threshold or box_iou >= 0.2) and distance <= centroid_distance:
                candidates.append((mask_iou + 0.25 * box_iou - distance / max(centroid_distance, 1), before_index, after_index, mask_iou, box_iou, distance))
    matches = []
    used_before, used_after = set(), set()
    for _, before_index, after_index, mask_iou, box_iou, distance in sorted(candidates, reverse=True):
        if before_index in used_before or after_index in used_after:
            continue
        used_before.add(before_index); used_after.add(after_index)
        matches.append({"before_index": before_index, "after_index": after_index, "mask_iou": mask_iou, "box_iou": box_iou, "centroid_distance": distance})
    return {"matches": matches, "unmatched_before": [index for index in range(len(before["instances"])) if index not in used_before], "unmatched_after": [index for index in range(len(after["instances"])) if index not in used_after]}