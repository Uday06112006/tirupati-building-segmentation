"""Tile-based PyTorch Dataset for building instance segmentation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from rasterio.features import rasterize
from shapely.affinity import translate
from shapely.geometry import box
from torch.utils.data import Dataset

from .utils import SPLITS, find_annotation, iter_tiles, load_instances, read_window


@dataclass(frozen=True)
class TileRecord:
    image_path: Path
    annotation_path: Path
    x: int
    y: int
    width: int
    height: int
    split: str


def discover_images(orthophoto_dir: Path) -> list[Path]:
    return sorted(p for p in orthophoto_dir.rglob("*.tif") if "orthophoto" in p.name.lower() and "preview" not in p.name.lower())


def build_records(orthophoto_dir: Path, annotation_dir: Path, tile_size: int, stride: int, split: str) -> list[TileRecord]:
    records = []
    for image_path in discover_images(orthophoto_dir):
        image_id = image_path.stem.replace("_2sec_orthophoto", "").replace("_1fps_orthophoto", "")
        if image_id not in SPLITS[split]:
            continue
        annotation_path = find_annotation(annotation_dir, image_path)
        import rasterio
        with rasterio.open(image_path) as src:
            tiles = iter_tiles(src.width, src.height, tile_size, stride)
        records.extend(TileRecord(image_path, annotation_path, *tile, split) for tile in tiles)
    if not records:
        raise RuntimeError(f"No {split} orthophoto tiles found. Check --orthophoto-dir and --annotation-dir.")
    return records


class BuildingTileDataset(Dataset):
    def __init__(self, records, augment=False, min_mask_pixels=4):
        self.records = list(records)
        self.augment = augment
        self.min_mask_pixels = min_mask_pixels
        self._instances = {}

    def __len__(self):
        return len(self.records)

    def _get_instances(self, path):
        key = str(path)
        if key not in self._instances:
            self._instances[key] = load_instances(path)
        return self._instances[key]

    def __getitem__(self, index):
        record = self.records[index]
        import rasterio
        with rasterio.open(record.image_path) as src:
            image = read_window(src, record.x, record.y, record.width, record.height)
        tile_box = box(record.x, record.y, record.x + record.width, record.y + record.height)
        masks, boxes = [], []
        for geometry in self._get_instances(record.annotation_path):
            clipped = geometry.intersection(tile_box)
            if clipped.is_empty:
                continue
            local = translate(clipped, xoff=-record.x, yoff=-record.y)
            mask = rasterize([(local, 1)], out_shape=(record.height, record.width), fill=0, dtype="uint8")
            ys, xs = np.where(mask > 0)
            if len(xs) < self.min_mask_pixels:
                continue
            x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            if x2 <= x1 or y2 <= y1:
                continue
            masks.append(mask.astype(bool))
            boxes.append([x1, y1, x2, y2])
        mask_array = np.asarray(masks, dtype=bool) if masks else np.zeros((0, record.height, record.width), dtype=bool)
        box_array = np.asarray(boxes, dtype=np.float32).reshape((-1, 4))
        image, mask_array, box_array = self._augment(image, mask_array, box_array)
        target = {
            "boxes": torch.as_tensor(box_array, dtype=torch.float32),
            "labels": torch.ones((len(mask_array),), dtype=torch.int64),
            "masks": torch.as_tensor(mask_array, dtype=torch.uint8),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": torch.as_tensor((box_array[:, 2] - box_array[:, 0]) * (box_array[:, 3] - box_array[:, 1]), dtype=torch.float32),
            "iscrowd": torch.zeros((len(mask_array),), dtype=torch.int64),
        }
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0, target

    def _augment(self, image, masks, boxes):
        if not self.augment:
            return image, masks, boxes
        if random.random() < 0.5:
            image, masks = image[:, ::-1].copy(), masks[:, :, ::-1].copy()
        if random.random() < 0.5:
            image, masks = image[::-1].copy(), masks[:, ::-1, :].copy()
        if random.random() < 0.25:
            image, masks = np.rot90(image, 1).copy(), np.rot90(masks, 1, axes=(1, 2)).copy()
        if len(masks):
            boxes = np.asarray([[np.where(mask)[1].min(), np.where(mask)[0].min(), np.where(mask)[1].max() + 1, np.where(mask)[0].max() + 1] for mask in masks], dtype=np.float32)
        return image, masks, boxes


def collate_fn(batch):
    return tuple(zip(*batch))


def tile_statistics(records, dataset):
    counts = [len(dataset[index][1]["labels"]) for index in range(len(records))]
    positive = sum(value > 0 for value in counts)
    return {"tiles": len(counts), "tiles_with_instances": positive, "empty_tiles": len(counts) - positive, "total_instances": sum(counts), "min_instances": min(counts, default=0), "max_instances": max(counts, default=0), "mean_instances": float(np.mean(counts)) if counts else 0.0}