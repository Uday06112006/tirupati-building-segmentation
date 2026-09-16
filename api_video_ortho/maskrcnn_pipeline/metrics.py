"""Lightweight instance metrics independent of pycocotools."""

from __future__ import annotations

import numpy as np


def mask_iou(first, second):
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else 0.0


def box_iou(first, second):
    xa, ya, xb, yb = max(first[0], second[0]), max(first[1], second[1]), min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match_instances(prediction, target, iou_threshold=0.5):
    predictions = prediction.get("masks", np.zeros((0, 1, 1), dtype=bool)) > 0.5
    targets = target.get("masks", np.zeros((0, 1, 1), dtype=bool)) > 0
    candidates = sorted(((mask_iou(pred, truth), p, t) for p, pred in enumerate(predictions) for t, truth in enumerate(targets)), reverse=True)
    used_p, used_t, matches = set(), set(), []
    for overlap, p, t in candidates:
        if overlap < iou_threshold or p in used_p or t in used_t:
            continue
        used_p.add(p); used_t.add(t); matches.append(overlap)
    return len(matches), len(predictions) - len(matches), len(targets) - len(matches), matches


def summarize(records, iou_threshold=0.5):
    tp = fp = fn = 0
    overlaps = []
    for prediction, target in records:
        current_tp, current_fp, current_fn, current_iou = match_instances(prediction, target, iou_threshold)
        tp += current_tp; fp += current_fp; fn += current_fn; overlaps.extend(current_iou)
    return {"precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0, "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0, "matched_mask_iou": float(np.mean(overlaps)) if overlaps else 0.0, "tp": tp, "fp": fp, "fn": fn}


def average_precision_at_iou(records, threshold):
    detections, positives = [], 0
    for prediction, target in records:
        truths = target.get("masks", np.zeros((0, 1, 1), dtype=bool)) > 0
        positives += len(truths)
        for index, mask in enumerate(prediction.get("masks", np.zeros((0, 1, 1), dtype=bool)) > 0.5):
            detections.append((float(prediction.get("scores", np.ones(len(prediction["masks"])))[index]), mask, truths))
    detections.sort(key=lambda item: item[0], reverse=True)
    used, tp, fp, area, previous_recall = set(), 0, 0, 0.0, 0.0
    for _, mask, truths in detections:
        available = [(mask_iou(mask, truth), index) for index, truth in enumerate(truths) if index not in used]
        best = max(available, default=(0.0, -1))
        if best[0] >= threshold:
            used.add(best[1]); tp += 1
        else:
            fp += 1
        recall = tp / positives if positives else 0.0
        area += (tp / (tp + fp)) * max(0.0, recall - previous_recall)
        previous_recall = recall
    return area


def average_precision_50_95(records):
    return float(np.mean([average_precision_at_iou(records, threshold) for threshold in np.arange(0.5, 1.0, 0.05)]))