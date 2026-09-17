"""Tiled Mask R-CNN inference with sparse full-image instance masks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from maskrcnn_pipeline.inference import suppress_duplicates
from maskrcnn_pipeline.utils import iter_tiles, read_window


def _box_iou(first, second):
    xa, ya = max(first[0], second[0]), max(first[1], second[1])
    xb, yb = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, xb - xa) * max(0, yb - ya)
    area_first = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_second = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def _merge_candidates(candidates, mask_iou_threshold, box_iou_threshold):
    kept = []
    for candidate in sorted(candidates, key=lambda value: value["score"], reverse=True):
        duplicate = False
        for existing in kept:
            overlap = _box_iou(candidate["box"], existing["box"])
            if overlap < box_iou_threshold:
                continue
            xa, ya = max(candidate["box"][0], existing["box"][0]), max(candidate["box"][1], existing["box"][1])
            xb, yb = min(candidate["box"][2], existing["box"][2]), min(candidate["box"][3], existing["box"][3])
            first = candidate["mask"][int(ya - candidate["box"][1]):int(yb - candidate["box"][1]), int(xa - candidate["box"][0]):int(xb - candidate["box"][0])]
            second = existing["mask"][int(ya - existing["box"][1]):int(yb - existing["box"][1]), int(xa - existing["box"][0]):int(xb - existing["box"][0])]
            intersection = np.logical_and(first, second).sum()
            union = np.logical_or(first, second).sum()
            if union and intersection / union >= mask_iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


@torch.inference_mode()
def detect_instances(model, image_path: Path, device, tile_size=512, stride=384, score_threshold=0.5, mask_threshold=0.5, merge_mask_iou=0.5, merge_box_iou=0.7):
    import rasterio

    candidates = []
    with rasterio.open(image_path) as source:
        width, height = source.width, source.height
        for tile_x, tile_y, tile_width, tile_height in iter_tiles(width, height, tile_size, stride):
            tile = read_window(source, tile_x, tile_y, tile_width, tile_height)
            tensor = torch.from_numpy(tile.transpose(2, 0, 1)).float().div(255.0).to(device)
            output = model([tensor])[0]
            for local_mask, score, local_box in zip(output["masks"][:, 0].cpu().numpy(), output["scores"].cpu().numpy(), output["boxes"].cpu().numpy()):
                if float(score) < score_threshold:
                    continue
                mask = local_mask >= mask_threshold
                ys, xs = np.where(mask)
                if not len(xs):
                    continue
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
                candidates.append({
                    "mask": mask[y1:y2, x1:x2].copy(),
                    "box": [tile_x + x1, tile_y + y1, tile_x + x2, tile_y + y2],
                    "model_box": [float(local_box[0] + tile_x), float(local_box[1] + tile_y), float(local_box[2] + tile_x), float(local_box[3] + tile_y)],
                    "score": float(score),
                    "source_image": image_path.name,
                    "tile": [tile_x, tile_y, tile_width, tile_height],
                })
    detections = _merge_candidates(candidates, merge_mask_iou, merge_box_iou)
    for instance_id, detection in enumerate(detections, 1):
        detection["instance_id"] = instance_id
    return {"image": image_path.name, "width": width, "height": height, "coordinate_system": "image_pixels", "instances": detections}