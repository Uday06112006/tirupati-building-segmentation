"""Shared raster, geometry, tiling, and visualization helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.affinity import affine_transform
from shapely.geometry import box, shape
from shapely.ops import unary_union


ORTHOPHOTO_TO_GEOJSON = {
    "DJI_0510": "DJI_0510.geojson", "DJI_0513": "Annotations_13_correct.geojson",
    "DJI_0517": "DJI_0517.geojson", "DJI_0520": "DJI_0520_annotations.geojson",
    "DJI_0522": "Annotations_22.geojson", "DJI_0523": "Annotations_23.geojson",
    "DJI_0527": "Annotations_27.geojson", "DJI_0529": "Annotations_29.geojson",
    "DJI_0530": "DJI_0530.geojson", "DJI_0531": "DJI_0531.geojson",
    "DJI_0532": "DJI_0532.geojson", "DJI_0533": "DJI_0533.geojson",
    "DJI_0534": "DJI_0534.geojson", "DJI_0535": "DJI_0535.geojson",
}

SPLITS = {
    "train": {"DJI_0510", "DJI_0513", "DJI_0517", "DJI_0520", "DJI_0522", "DJI_0523", "DJI_0527", "DJI_0529"},
    "val": {"DJI_0530", "DJI_0531", "DJI_0532"},
    "test": {"DJI_0533", "DJI_0534", "DJI_0535"},
}


def orthophoto_id(path: Path) -> str:
    name = path.stem
    for suffix in ("_2sec_orthophoto", "_1fps_orthophoto", "_orthophoto"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def find_annotation(annotation_dir: Path, image_path: Path) -> Path:
    image_id = orthophoto_id(image_path)
    expected = ORTHOPHOTO_TO_GEOJSON.get(image_id)
    candidates = [annotation_dir / expected] if expected else []
    candidates += [annotation_dir / f"{image_id}.geojson", annotation_dir / f"{image_path.stem}.geojson"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No GeoJSON found for {image_path.name} in {annotation_dir}")


def image_space_geometry(geometry):
    """Convert the repository's (x, -row) coordinates to image coordinates."""
    return affine_transform(geometry, [1, 0, 0, -1, 0, 0])


def connected_components(geometries):
    groups = []
    for geometry in geometries:
        matching = [index for index, group in enumerate(groups) if any(geometry.intersects(part) for part in group)]
        if not matching:
            groups.append([geometry])
            continue
        first = matching[0]
        groups[first].append(geometry)
        for index in reversed(matching[1:]):
            groups[first].extend(groups.pop(index))
    return [unary_union(group) for group in groups]


def load_instances(annotation_path: Path) -> list:
    """Load one Shapely geometry per physical polygon component."""
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    instances = []
    for feature in data.get("features", []):
        raw = feature.get("geometry")
        if not raw or raw.get("type") not in {"Polygon", "MultiPolygon"}:
            continue
        try:
            geometry = image_space_geometry(shape(raw))
        except Exception:
            continue
        parts = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
        for part in connected_components(parts):
            if not part.is_valid:
                part = part.buffer(0)
            repaired = list(part.geoms) if part.geom_type == "MultiPolygon" else [part]
            instances.extend(component for component in repaired if not component.is_empty and component.area > 0)
    return instances


def read_window(src, x: int, y: int, width: int, height: int) -> np.ndarray:
    from rasterio.windows import Window
    array = np.transpose(src.read(window=Window(x, y, width, height), boundless=True, fill_value=0), (1, 2, 0))
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] > 3:
        array = array[:, :, :3]
    if np.issubdtype(array.dtype, np.integer) and array.dtype.itemsize > 1:
        array = array.astype(np.float32)
        high = np.nanpercentile(array, 99.5) if np.isfinite(array).any() else 1.0
        array = array / max(high, 1.0) * 255.0
    elif array.dtype.kind == "f" and array.size and np.nanmax(array) <= 1.0:
        array = array * 255.0
    return np.clip(np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0), 0, 255).astype(np.uint8)


def iter_tiles(width: int, height: int, tile_size: int, stride: int):
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive")
    xs = list(range(0, max(width - tile_size, 0) + 1, stride))
    ys = list(range(0, max(height - tile_size, 0) + 1, stride))
    if not xs or xs[-1] != max(width - tile_size, 0):
        xs.append(max(width - tile_size, 0))
    if not ys or ys[-1] != max(height - tile_size, 0):
        ys.append(max(height - tile_size, 0))
    for y in sorted(set(ys)):
        for x in sorted(set(xs)):
            yield x, y, min(tile_size, width - x), min(tile_size, height - y)


def draw_instances(image: np.ndarray, masks: np.ndarray, boxes: np.ndarray, scores=None) -> np.ndarray:
    canvas = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    rng = random.Random(17)
    for index, mask in enumerate(masks):
        color = tuple(rng.randint(50, 230) for _ in range(3))
        ys, xs = np.where(mask > 0)
        if len(xs):
            draw.polygon(list(zip(xs.tolist(), ys.tolist())), fill=color + (70,))
        if len(boxes) > index:
            x1, y1, x2, y2 = boxes[index]
            draw.rectangle((x1, y1, x2, y2), outline=color + (255,), width=2)
            label = str(index + 1) if scores is None else f"{index + 1}:{scores[index]:.2f}"
            draw.text((x1 + 2, y1 + 2), label, fill=(255, 255, 255, 255), stroke_width=1, stroke_fill=color + (255,))
    return np.asarray(canvas)


def save_qa(image: np.ndarray, masks: np.ndarray, boxes: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(draw_instances(image, masks, boxes)).save(path)