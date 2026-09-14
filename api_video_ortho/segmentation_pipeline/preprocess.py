#!/usr/bin/env python3
"""Separate building segmentation preprocessing for the orthophoto inventory.

The script reads the 14 orthophoto TIFF files in outputs/ and the 14 prompt-supplied
GeoJSON annotations in Dataset/Annotations/. It converts the GeoJSON polygon coordinate
pairs into a binary mask whose pixels are image coordinates. It never edits the
existing generator in tools/ and writes all new artifacts into the Dataset tree only.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape, mapping
from shapely.affinity import affine_transform
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ANNOT_DIR = ROOT / 'Dataset' / 'Annotations'
OUTPUT_DIR = ROOT / 'outputs'
DATASET_DIR = ROOT / 'Dataset'
IMAGE_DIR = DATASET_DIR / 'images'
MASK_DIR = DATASET_DIR / 'masks'
QA_DIR = DATASET_DIR / 'qa'


def load_config_file(path: str | None):
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = (ROOT / file_path)
    if not file_path.exists():
        raise FileNotFoundError(f'Config file not found: {file_path}')
    with open(file_path, 'r', encoding='utf-8') as fp:
        return json.load(fp)


def resolve_path(raw, base_root: Path):
    if raw is None:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = base_root / p
    return p.expanduser().resolve()


def apply_runtime_paths(args):
    global ROOT, ANNOT_DIR, OUTPUT_DIR, DATASET_DIR, IMAGE_DIR, MASK_DIR, QA_DIR
    cfg = load_config_file(args.config)
    base_root = resolve_path(cfg.get('root_dir', str(ROOT)), ROOT)
    if args.root_dir:
        base_root = resolve_path(args.root_dir, ROOT)
    ROOT = base_root
    ANNOT_DIR = resolve_path(args.annotation_dir or cfg.get('annotation_dir', 'Dataset/Annotations'), ROOT)
    OUTPUT_DIR = resolve_path(args.orthophoto_dir or cfg.get('orthophoto_dir', 'outputs'), ROOT)
    DATASET_DIR = resolve_path(args.dataset_dir or cfg.get('dataset_dir', 'Dataset'), ROOT)
    IMAGE_DIR = resolve_path(args.image_dir or cfg.get('image_dir', 'Dataset/images'), ROOT)
    MASK_DIR = resolve_path(args.mask_dir or cfg.get('mask_dir', 'Dataset/masks'), ROOT)
    QA_DIR = resolve_path(args.qa_dir or cfg.get('qa_dir', 'Dataset/qa'), ROOT)

    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)

# Authoritative orthophoto -> GeoJSON mapping derived from the file inventory.
ORTHOPHOTO_TO_GEOJSON = {
    'DJI_0510_2sec_orthophoto.tif': 'DJI_0510.geojson',
    'DJI_0513_2sec_orthophoto.tif': 'Annotations_13_correct.geojson',
    'DJI_0517_2sec_orthophoto.tif': 'DJI_0517.geojson',
    'DJI_0520_2sec_orthophoto.tif': 'DJI_0520_annotations.geojson',
    'DJI_0522_2sec_orthophoto.tif': 'Annotations_22.geojson',
    'DJI_0523_2sec_orthophoto.tif': 'Annotations_23.geojson',
    'DJI_0527_2sec_orthophoto.tif': 'Annotations_27.geojson',
    'DJI_0529_2sec_orthophoto.tif': 'Annotations_29.geojson',
    'DJI_0530_2sec_orthophoto.tif': 'DJI_0530.geojson',
    'DJI_0531_2sec_orthophoto.tif': 'DJI_0531.geojson',
    'DJI_0532_2sec_orthophoto.tif': 'DJI_0532.geojson',
    'DJI_0533_2sec_orthophoto.tif': 'DJI_0533.geojson',
    'DJI_0534_2sec_orthophoto.tif': 'DJI_0534.geojson',
    'DJI_0535_2sec_orthophoto.tif': 'DJI_0535.geojson',
}


def list_orthophoto_paths() -> List[Path]:
    """Return the 14 generated orthophoto TIFFs that match the GeoJSON inventory."""
    candidates = []
    for img in sorted(OUTPUT_DIR.rglob('*.tif')):
        # do not return preview or test variants
        if 'orthophoto' in img.name.lower() and 'preview' not in img.name.lower():
            if img.name in ORTHOPHOTO_TO_GEOJSON:
                candidates.append(img)
    return candidates


def load_geojson(path: Path):
    with open(path, 'r', encoding='utf-8') as fp:
        return json.load(fp)


def geojson_to_mask(img_path: Path, geojson_path: Path, mask_path: Path) -> np.ndarray:
    """Convert GeoJSON polygon annotations into a binary building mask.

    Orthophoto TIFFs are image-plane rasters with no geotransform. The GeoJSON
    files write polygon coordinates in a CRS84 mixed local system whose Y values
    are inverted relative to image-space rows, i.e. the intended image row is
    -y. We therefore normalize every polygon coordinate pair from (x, y) into
    (x, -y) before rasterizing. This preserves the spatial relationship in the
    image frame and produces a binary 0/1 mask.
    """
    with rasterio.open(img_path) as src:
        height = src.height
        width = src.width
        transform = src.transform

    data = load_geojson(geojson_path)
    features = []
    for feature in data.get('features', []):
        geom = feature.get('geometry', None)
        if geom is None:
            continue
        if geom.get('type') not in ('Polygon', 'MultiPolygon'):
            continue
        try:
            s = shape(geom)
        except Exception:
            continue
        # If the geometry is a Polygon or MultiPolygon, flip y coordinate sign
        # to reach image pixel space. The GeoJSON payload stores y downward as
        # negative world coordinate values; in the raster image, rows grow down.
        # In other words: coordinate (x, -y) is image-space.
        s = affine_transform(s, [1, 0, 0, -1, 0, 0])
        if not s.is_valid:
            continue
        features.append((s, 1))

    if not features:
        mask = np.zeros((height, width), dtype=np.uint8)
    else:
        mask = rasterize(
            shapes=features,
            out_shape=(height, width),
            fill=0,
            dtype='uint8',
            transform=transform,
            all_touched=False,
        )

    # Save mask in PNG. Binary label 0/1; mask value 255 for labels. PNG should
    # be read as an image later. Some consumers expect 0 and 255.
    Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path)
    return mask


def write_image_copy(img_path: Path, dst: Path):
    """Copy an orthophoto TIFF into the image dataset directory as a PNG."""
    with rasterio.open(img_path) as src:
        arr = src.read()
        # rasterio returns channels first. convert into HWC.
        arr = np.transpose(arr, (1, 2, 0))
        # clip and convert
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        # if grayscale, stack to RGB
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        Image.fromarray(arr).save(dst)


def write_overlay(img_path: Path, mask_path: Path, out_path: Path):
    """Show orthophoto + mask overlay for QA inspection."""
    img = Image.open(img_path).convert('RGB')
    img_arr = np.array(img)
    mask_arr = np.array(Image.open(mask_path).convert('L'))
    overlay = img_arr.copy()
    idx = mask_arr > 0
    overlay[idx] = np.clip(overlay[idx] * 0.7 + np.array([0, 200, 0]) * 0.3, 0, 255)
    Image.fromarray(overlay.astype(np.uint8)).save(out_path)


def preprocess():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)

    # Create a mapping JSON artifact for later reproducibility.
    map_path = DATASET_DIR / 'segmentation_geojson_map.json'
    with open(map_path, 'w', encoding='utf-8') as fp:
        json.dump(ORTHOPHOTO_TO_GEOJSON, fp, indent=2)

    # Preprocess the exact 14 known orthophotos and make sure a file exists.
    for img_name, geojson_name in ORTHOPHOTO_TO_GEOJSON.items():
        candidates = list(OUTPUT_DIR.rglob(img_name))
        if not candidates:
            raise FileNotFoundError(f'No {img_name} found in outputs/ tree')
        img_path = candidates[0]
        geojson_path = ANNOT_DIR / geojson_name
        if not geojson_path.exists():
            raise FileNotFoundError(f'Missing GeoJSON annotation {geojson_path}')

        # Save orthophoto image copy under Dataset/images.
        image_dst = IMAGE_DIR / img_name.replace('.tif', '.png')
        write_image_copy(img_path, image_dst)

        # Save binary mask.
        mask_dst = MASK_DIR / img_name.replace('.tif', '_mask.png')
        _ = geojson_to_mask(img_path, geojson_path, mask_dst)

        # Save overlay for QA.
        overlay_dst = QA_DIR / img_name.replace('.tif', '_overlay.png')
        write_overlay(image_dst, mask_dst, overlay_dst)

    print(f'Wrote {len(ORTHOPHOTO_TO_GEOJSON)} binary masks and QA overlays.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image-plane orthophoto-to-GeoJSON segmentation mask generator.')
    parser.add_argument('--config', default=None, help='Optional JSON config file for path portability.')
    parser.add_argument('--root-dir', default=None, help='Repository root override.')
    parser.add_argument('--orthophoto-dir', default=None, help='Directory containing orthophoto TIFF inputs.')
    parser.add_argument('--dataset-dir', default=None, help='Dataset root containing annotations, masks, images, and QA outputs.')
    parser.add_argument('--annotation-dir', default=None, help='Directory containing GeoJSON annotation files.')
    parser.add_argument('--image-dir', default=None, help='Directory for orthophoto image copies.')
    parser.add_argument('--mask-dir', default=None, help='Directory for generated binary masks.')
    parser.add_argument('--qa-dir', default=None, help='Directory for QA overlays.')
    args = parser.parse_args()

    apply_runtime_paths(args)
    print('Segmentation inputs are orthophoto TIFFs and GeoJSON building masks; no drone video input is required.')
    preprocess()
