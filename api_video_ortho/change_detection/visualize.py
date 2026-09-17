"""QA and change visualization writers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


COLORS = {"UNCHANGED": (180, 180, 180), "NEW_BUILDING": (40, 210, 60), "REMOVED_BUILDING": (230, 50, 50), "CHANGED_BUILDING": (255, 150, 30)}


def _draw_instances(image, instances, colors=None):
    canvas = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    for index, instance in enumerate(instances):
        color = colors[index] if colors else (70, 180, 240)
        x1, y1, x2, y2 = instance["box"]
        mask = instance["mask"]
        overlay = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        overlay[y1:y2, x1:x2, :3] = color
        overlay[y1:y2, x1:x2, 3] = mask.astype(np.uint8) * 75
        canvas = Image.alpha_composite(canvas.convert("RGBA"), Image.fromarray(overlay, "RGBA"))
        draw = ImageDraw.Draw(canvas, "RGBA")
        draw.rectangle((x1, y1, x2, y2), outline=color + (255,), width=2)
        draw.text((x1 + 2, y1 + 2), str(instance["instance_id"]), fill=(255, 255, 255, 255), stroke_width=1, stroke_fill=color + (255,))
    return np.asarray(canvas.convert("RGB"))


def save_detection_visualization(image, detections, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_draw_instances(image, detections["instances"])).save(path)


def save_alignment_qa(before_preview, after_preview, preview_matrix, path: Path):
    height, width = before_preview.shape[:2]
    aligned = cv2.warpPerspective(after_preview, np.asarray(preview_matrix), (width, height))
    after_resized = cv2.resize(after_preview, (width, height))
    blend = cv2.addWeighted(before_preview, 0.5, aligned, 0.5, 0)
    canvas = np.concatenate([before_preview, after_resized, aligned, blend], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


def save_matched_visualization(before_image, before, after, changes, path: Path):
    colors = []
    for instance in before["instances"]:
        change = next((item for item in changes if item["before_id"] == instance["instance_id"]), None)
        colors.append(COLORS.get(change["classification"] if change else "UNCHANGED"))
    canvas = Image.fromarray(_draw_instances(before_image, before["instances"], colors)).convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    for instance in after["instances"]:
        x1, y1, x2, y2 = instance["box"]
        draw.rectangle((x1, y1, x2, y2), outline=(30, 220, 220, 220), width=2)
    Image.fromarray(np.asarray(canvas.convert("RGB"))).save(path)


def save_change_visualization(before_image, before, after, changes, path: Path):
    colors = []
    for instance in before["instances"]:
        change = next((item for item in changes if item["before_id"] == instance["instance_id"]), None)
        colors.append(COLORS.get(change["classification"] if change else "UNCHANGED"))
    canvas = Image.fromarray(_draw_instances(before_image, before["instances"], colors)).convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    for instance in after["instances"]:
        change = next((item for item in changes if item["after_id"] == instance["instance_id"]), None)
        if change and change["classification"] == "NEW_BUILDING":
            x1, y1, x2, y2 = instance["box"]
            draw.rectangle((x1, y1, x2, y2), outline=COLORS["NEW_BUILDING"] + (255,), width=3)
    legend_y = 20
    for name, color in COLORS.items():
        draw.rectangle((20, legend_y, 40, legend_y + 20), fill=color + (255,))
        draw.text((48, legend_y + 2), name, fill=(255, 255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0, 255))
        legend_y += 26
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(path)