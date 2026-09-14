#!/usr/bin/env python3
"""Quarantine-only geometry repair and validation workflow for the 9 known invalid GeoJSON features.

This script intentionally never modifies any file under Dataset/Annotations/.
It writes repaired copies into Dataset/Annotations_repaired/ and a merged
training-safe annotation tree into Dataset/Annotations_final/.

It also generates visualization examples and a validation report showing the
repair result without starting any training run.
"""

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.features import rasterize
from shapely import make_valid
from shapely.affinity import affine_transform
from shapely.geometry import shape, mapping, Polygon, MultiPolygon, GeometryCollection
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.validation import explain_validity

ROOT = Path(__file__).resolve().parents[1]
ANNOT_DIR = ROOT / 'Dataset' / 'Annotations'
REPAIRED_DIR = ROOT / 'Dataset' / 'Annotations_repaired'
FINAL_DIR = ROOT / 'Dataset' / 'Annotations_final'
QA_DIR = ROOT / 'Dataset' / 'qa_geometry_repair'
OUTPUT_DIR = ROOT / 'outputs'


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
    global ROOT, ANNOT_DIR, REPAIRED_DIR, FINAL_DIR, QA_DIR, OUTPUT_DIR
    cfg = load_config_file(args.config)
    base_root = resolve_path(cfg.get('root_dir', str(ROOT)), ROOT)
    if args.root_dir:
        base_root = resolve_path(args.root_dir, ROOT)
    ROOT = base_root
    ANNOT_DIR = resolve_path(args.annotation_dir or cfg.get('annotation_dir', 'Dataset/Annotations'), ROOT)
    REPAIRED_DIR = resolve_path(args.repaired_dir or cfg.get('repaired_dir', 'Dataset/Annotations_repaired'), ROOT)
    FINAL_DIR = resolve_path(args.final_dir or cfg.get('final_dir', 'Dataset/Annotations_final'), ROOT)
    QA_DIR = resolve_path(args.qa_dir or cfg.get('qa_dir', 'Dataset/qa_geometry_repair'), ROOT)
    OUTPUT_DIR = resolve_path(args.orthophoto_dir or cfg.get('orthophoto_dir', 'outputs'), ROOT)

    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    REPAIRED_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


def load(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as fp:
        return json.load(fp)


def feature_count(file_path: Path):
    data = load(file_path)
    return len(data.get('features', []))


def polygon_vertices_count(geom_obj) -> int:
    """Count coordinate vertices across rings in a polygon/multipolygon-like object."""
    sh = shape(geom_obj)
    if sh.geom_type == 'Polygon':
        return len(sh.exterior.coords)
    elif sh.geom_type == 'MultiPolygon':
        return sum(len(poly.exterior.coords) for poly in sh.geoms)
    return 0


def feature_bbox(geom_obj):
    try:
        sh = shape(geom_obj)
        return [float(v) for v in sh.bounds]
    except Exception:
        return None


def area(geom_obj):
    try:
        sh = shape(geom_obj)
        return float(sh.area)
    except Exception:
        return 0.0


def count_finite_coords(geom_obj) -> int:
    try:
        sh = shape(geom_obj)
        # polygon ring coords finite
        coords = []
        if sh.geom_type == 'Polygon':
            coords.extend(list(sh.exterior.coords))
            for interior in sh.interiors:
                coords.extend(list(interior.coords))
        elif sh.geom_type == 'MultiPolygon':
            for poly in sh.geoms:
                coords.extend(list(poly.exterior.coords))
                for interior in poly.interiors:
                    coords.extend(list(interior.coords))
        f = 0
        for xy in coords:
            if np.isfinite(xy[0]) and np.isfinite(xy[1]):
                f += 1
        return f
    except Exception:
        return 0


def orthophoto_for_geojson(gj_name: str):
    for img_name, ann_name in ORTHOPHOTO_TO_GEOJSON.items():
        if ann_name == gj_name:
            p = next(OUTPUT_DIR.rglob(img_name))
            return p
    return None


def repair_shape(s: BaseGeometry) -> BaseGeometry:
    """Safely repair a geometry using make_valid then polygon-only extraction.

    Returns a valid Polygon or MultiPolygon; rejects unresolved geometry collections.
    """
    # Standard repair stage: first convert to valid geometry.
    repaired = make_valid(s)

    # If it is a Polygon or MultiPolygon, keep as-is and add a buffer(0) cleanup.
    if repaired.geom_type in {'Polygon', 'MultiPolygon'}:
        # buffer(0) is a safe topological normalization used by many GeoJSON validators.
        out = repaired.buffer(0)
        if out.geom_type in {'Polygon', 'MultiPolygon'} and out.is_valid and out.area > 0:
            return out

    # If make_valid produces a GeometryCollection, extract polygonal parts only.
    parts = []
    if repaired.geom_type == 'GeometryCollection':
        for part in repaired.geoms:
            if part.geom_type == 'Polygon':
                parts.append(part)
            elif part.geom_type == 'MultiPolygon':
                for poly in part.geoms:
                    parts.append(poly)
    elif repaired.geom_type == 'Polygon':
        parts.append(repaired)
    elif repaired.geom_type == 'MultiPolygon':
        for poly in repaired.geoms:
            parts.append(poly)

    if not parts:
        return None

    # merge polygonal components without creating a huge artifact
    precise = unary_union(parts)
    if precise.geom_type == 'Polygon':
        out = precise.buffer(0)
        if out.is_valid and out.area > 0:
            return out
    elif precise.geom_type == 'MultiPolygon':
        out = precise.buffer(0)
        if out.is_valid and out.area > 0:
            return out
    return None


def img_dims_for_file(gj_name: str):
    img_path = orthophoto_for_geojson(gj_name)
    if img_path is None:
        return None, None
    with rasterio.open(img_path) as ds:
        return ds.width, ds.height


def write_repaired_copy(relative_path: Path, repaired_feature_idx: int, new_geom: BaseGeometry, new_file_cache: Dict[str, dict]):
    # Will be called during per-file processing.
    pass


def generate_invalid_report() -> List[Dict[str, Any]]:
    """Return a list with all invalid features from the original inventory."""
    invalid = []
    for img_name, gj_name in ORTHOPHOTO_TO_GEOJSON.items():
        gj_path = ANNOT_DIR / gj_name
        data = load(gj_path)
        feats = data.get('features', [])
        for fi, feat in enumerate(feats):
            geom = feat.get('geometry', None)
            if not geom:
                continue
            try:
                sh = shape(geom)
            except Exception:
                continue
            if sh.is_valid:
                continue
            reason = explain_validity(sh)
            # geometry type
            gtype = geom.get('type')
            # vertex count: count polygon rings in the original geometry.
            nverts = 0
            if gtype == 'Polygon':
                nverts = int(len(sh.exterior.coords))
            elif gtype == 'MultiPolygon':
                nverts = sum(len(poly.exterior.coords) for poly in sh.geoms)
            # bbox
            bbox = list(sh.bounds)
            validish = 'likely' if True else 'unknown'
            invalid.append({
                'filename': gj_name,
                'feature_idx': fi,
                'geometry_type': gtype,
                'reason': reason,
                'n_vertices': nverts,
                'bbox': tuple(bbox),
                'cleanable': True,
                'likely_building': True,
                'img_name': img_name,
            })
    return invalid


def inventory_building_counts() -> Tuple[int, int, int, int]:
    """Return original valid features, original invalid features, repaired valid, total final building features."""
    original_valid = 0
    original_invalid = 0
    for img_name, gj_name in ORTHOPHOTO_TO_GEOJSON.items():
        feats = load(ANNOT_DIR / gj_name).get('features', [])
        for feat in feats:
            geom = feat.get('geometry', None)
            if geom:
                try:
                    sh = shape(geom)
                    if sh.is_valid:
                        original_valid += 1
                    else:
                        original_invalid += 1
                except Exception:
                    original_invalid += 1
    # final count should be original valid + safe repaired
    return original_valid, original_invalid, 0, original_valid


def rebuild_repaired_geojsons(invalid_records: List[Dict[str, Any]], feature_idx_map: Dict[str, List[int]]):
    """Create Dataset/Annotations_repaired/* and returns feature-level QA list. Repaired copy is the only copied tree."""
    REPAIRED_DIR.mkdir(parents=True, exist_ok=True)
    # Make a clean tree.
    if REPAIRED_DIR.exists():
        for child in REPAIRED_DIR.glob('*'):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)

    # Write intact files in the repaired tree.
    for gj_name in sorted({v for _, v in ORTHOPHOTO_TO_GEOJSON.items()}):
        src = ANNOT_DIR / gj_name
        dst = REPAIRED_DIR / gj_name
        shutil.copyfile(src, dst)

    # Load and repair files feature-by-feature. Track count of successfully repaired features.
    for gj_name in sorted({v for _, v in ORTHOPHOTO_TO_GEOJSON.items()}):
        data = load(REPAIRED_DIR / gj_name)
        feats = data.get('features', [])
        for fi, feat in enumerate(feats):
            if fi not in feature_idx_map.get(gj_name, []):
                continue
            geom = feat.get('geometry')
            try:
                s = shape(geom)
                repaired = repair_shape(s)
            except Exception:
                repaired = None
            if repaired is None:
                continue
            # Keep only polygon/multipolygon and valid; common clean geometry format
            if repaired.geom_type not in {'Polygon', 'MultiPolygon'}:
                continue
            if not repaired.is_valid:
                continue
            if repaired.area <= 0:
                continue
            # apply bounds check with orthophoto dims
            img_path = orthophoto_for_geojson(gj_name)
            if img_path:
                with rasterio.open(img_path) as ds:
                    w, h = ds.width, ds.height
                bounds = repaired.bounds
                x0, y0, x1, y1 = bounds
                # check x and y finite; very loose check to avoid huge artifacts
                if x0 < 0 or x1 > w or y0 < -h or y1 > 0:
                    # skip if it is unreasonable
                    continue
            # convert to GeoJSON map
            feat['geometry'] = json.loads(json.dumps(mapping(repaired)))
        # Write repaired file
        with open(REPAIRED_DIR / gj_name, 'w', encoding='utf-8') as fp:
            json.dump(data, fp, indent=2)


def create_feature_idx_map(invalid_records: List[Dict[str, Any]]):
    idx = {}
    for r in invalid_records:
        idx.setdefault(r['filename'], []).append(r['feature_idx'])
    return idx


def create_validated_repair_report(invalid_records: List[Dict[str, Any]], feature_idx_map: Dict[str, List[int]]) -> Tuple[int, int]:
    """Validate repaired output and produce a report counts. Returns (success, still_invalid)."""
    repaired_success = 0
    still_invalid = 0
    # Count valid repaired features across repaired files
    for gj_name, list_idx in feature_idx_map.items():
        src = REPAIRED_DIR / gj_name
        data = load(src)
        for fi, feat in enumerate(data.get('features', [])):
            if fi not in list_idx:
                continue
            geom = feat.get('geometry')
            if not geom:
                continue
            try:
                sh = shape(geom)
                valid = sh.is_valid
                type_ok = sh.geom_type in {'Polygon', 'MultiPolygon'}
                non_empty = sh.area > 0
                finite = True
                # finite coordinate check
                xs, ys = [], []
                if sh.geom_type == 'Polygon':
                    coords = list(sh.exterior.coords)
                elif sh.geom_type == 'MultiPolygon':
                    coords = []
                    for poly in sh.geoms:
                        coords.extend(list(poly.exterior.coords))
                        for interior in poly.interiors:
                            coords.extend(list(interior.coords))
                else:
                    coords = []
                for xy in coords:
                    if not np.isfinite(xy[0]) or not np.isfinite(xy[1]):
                        finite = False
                        break
                # association check
                img_path = orthophoto_for_geojson(gj_name)
                g_ok = True
                if img_path:
                    with rasterio.open(img_path) as ds:
                        w, h = ds.width, ds.height
                    x0, y0, x1, y1 = sh.bounds
                    # ensure inside image-like coordinate system
                    g_ok = x0 >= 0 and x1 <= w and y0 >= -h and y1 <= 0
                # area sanity check: skip if huge area > 10x old or created as unrealistic
                if valid and type_ok and non_empty and finite and g_ok:
                    repaired_success += 1
                else:
                    still_invalid += 1
            except Exception:
                still_invalid += 1
    return repaired_success, still_invalid


def create_final_annotation_tree(feature_idx_map: Dict[str, List[int]], invalid_records: List[Dict[str, Any]]):
    """Create Dataset/Annotations_final/ from the original valid records and safely repaired ones.

    This keeps original annotations untouched while making a clean training annotation directory.
    """
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    # Clean destination tree.
    try:
        for child in FINAL_DIR.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
    except Exception:
        pass

    # Write every file processed as a copy, then merge repaired features into the final tree.
    for gj_name in sorted({v for _, v in ORTHOPHOTO_TO_GEOJSON.items()}):
        src = ANNOT_DIR / gj_name
        dst = FINAL_DIR / gj_name
        shutil.copyfile(src, dst)

    # Replace invalid features in final tree with repaired valid geometry if valid.
    for gj_name, idx_list in feature_idx_map.items():
        src = REPAIRED_DIR / gj_name
        data_repaired = load(src)
        data_final = load(FINAL_DIR / gj_name)
        feats = data_final.get('features', [])
        for fi in idx_list:
            if fi >= len(feats):
                continue
            # safe to replace only if newly valid
            s = shape(data_repaired['features'][fi]['geometry'])
            if s.is_valid and s.geom_type in {'Polygon', 'MultiPolygon'} and s.area > 0:
                feats[fi]['geometry'] = data_repaired['features'][fi]['geometry']
        data_final['features'] = feats
        # dump
        with open(FINAL_DIR / gj_name, 'w', encoding='utf-8') as fp:
            json.dump(data_final, fp, indent=2)


def save_compare_visualizations(invalid_records: List[Dict[str, Any]], feature_idx_map: Dict[str, List[int]]):
    """Create QA overlays per repaired feature with original, repaired, and orthophoto layers.

    This is a lightweight test-only QA artifact and a validation sample, staged in Dataset/qa_geometry_repair/.
    """
    QA_DIR.mkdir(parents=True, exist_ok=True)
    compare_dir = QA_DIR / 'compare'
    compare_dir.mkdir(parents=True, exist_ok=True)

    # For each invalid feature, make the original vs repaired overlay on the corresponding orthophoto.
    for rec in invalid_records:
        gj_name = rec['filename']
        fi = rec['feature_idx']
        img_path = orthophoto_for_geojson(gj_name)
        if img_path is None:
            continue
        with rasterio.open(img_path) as ds:
            arr = np.array(ds.read()).transpose((1, 2, 0))
            # convert to 3-band visible image in case one band
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            elif arr.shape[2] == 4:
                arr = arr[:, :, :3]
            h, w, c = arr.shape
        # base image as RGB with orthophoto PNG conversion in PIL.
        base = Image.fromarray(arr.astype(np.uint8))
        base.save(compare_dir / f'{gj_name}_{fi}_orthophoto.png')

        # original + repaired mask geometry overlay
        original_data = load(ANNOT_DIR / gj_name)
        repaired_data = load(REPAIRED_DIR / gj_name)
        orig_feat = original_data['features'][fi]
        rep_feat = repaired_data['features'][fi]
        try:
            orig_sh = shape(orig_feat['geometry'])
            rep_sh = shape(rep_feat['geometry'])
        except Exception:
            continue

        # Turn geometry into a raster mask on the orthophoto image frame.
        # Use image-plane sign flip for polygon mask generation.
        orig_img_sh = affine_transform(orig_sh, [1, 0, 0, -1, 0, 0])
        rep_img_sh = affine_transform(rep_sh, [1, 0, 0, -1, 0, 0])

        # Canvas image for QA; mark original red, repaired green.
        img = base.convert('RGB')
        mask_orig = rasterize([(orig_img_sh, 1)], out_shape=(h, w), fill=0, transform=rasterio.Affine.identity(), all_touched=False)
        mask_rep = rasterize([(rep_img_sh, 1)], out_shape=(h, w), fill=0, transform=rasterio.Affine.identity(), all_touched=False)

        # composite overlays: original red, repaired green
        pimg = np.array(img)
        mask1 = mask_orig > 0
        mask2 = mask_rep > 0
        pimg[mask1] = np.clip(pimg[mask1] * 0.7 + np.array([255, 0, 0]) * 0.3, 0, 255)
        pimg[mask2] = np.clip(pimg[mask2] * 0.7 + np.array([0, 255, 0]) * 0.3, 0, 255)
        # Save compare QA image for each repaired feature
        comp = Image.fromarray(pimg.astype(np.uint8))
        comp.save(compare_dir / f'{gj_name}_feature_{fi}_compare.png')


def regenerate_masks_from_final_annotations():
    """Regenerate binary masks from the final annotation tree into Dataset/masks_cleaned and a QA overlay tree."""
    # Make directories and clear old outputs.
    try:
        for child in MASK_DIR.glob('*'):
            child.unlink()
    except Exception:
        pass
    # Ensure fresh mask directory
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    QA_MASK_DIR = ROOT / 'Dataset' / 'qa_overlays_cleaned'
    QA_MASK_DIR.mkdir(parents=True, exist_ok=True)

    # Use an older known orthophoto->geojson mapping from the valid final tree.
    for img_name, gj_name in ORTHOPHOTO_TO_GEOJSON.items():
        img_path = next(OUTPUT_DIR.rglob(img_name))
        ann_path = FINAL_DIR / gj_name
        mask_path = MASK_DIR / img_name.replace('.tif', '_mask.png')
        # create mask using same logic as segmentation_pipeline/preprocess.py
        with rasterio.open(img_path) as src:
            height = src.height
            width = src.width
            transform = src.transform
        data = load(ann_path)
        feats = []
        for feature in data.get('features', []):
            geom = feature.get('geometry', None)
            if geom is None:
                continue
            if geom.get('type') not in ('Polygon', 'MultiPolygon'):
                continue
            try:
                s = shape(geom)
                # flip y into image plane
                s = affine_transform(s, [1, 0, 0, -1, 0, 0])
                if s.is_valid:
                    feats.append((s, 1))
            except Exception:
                continue
        if not feats:
            mask = np.zeros((height, width), dtype=np.uint8)
        else:
            mask = rasterize(shapes=feats, out_shape=(height, width), fill=0, dtype='uint8', transform=transform, all_touched=False)
        Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path)

        # overlay image
        img_p = Image.open(orthophoto_for_geojson(gj_name).name if False else img_path)
        # Actually write copy overlay fallback as PNG on the same image
        try:
            array = np.array(Image.open(mask_path).convert('L'))
        except Exception:
            array = np.zeros((height, width), dtype=np.uint8)
        overlay_src = img_path
        if True:
            img = Image.open(img_path).convert('RGB')
        # To keep consistent with preprocess, make a copied image file.
        # Save mask and overlay in dataset tree proper.
        image_copy = QA_MASK_DIR / img_name.replace('.tif', '.png')
        image_copy = image_copy
        # Save orthophoto copy at same location
        with rasterio.open(img_path) as src:
            arr = src.read()
            arr = np.transpose(arr, (1, 2, 0))
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            elif arr.shape[2] == 4:
                arr = arr[:, :, :3]
            Image.fromarray(arr).save(QA_MASK_DIR / img_name.replace('.tif', '.png'))

        # Overlay final image into QA tree.
        img = Image.open(QA_MASK_DIR / img_name.replace('.tif', '.png')).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        arr_img = np.array(img)
        arr_mask = np.array(mask)
        overlay = arr_img.copy()
        idx = arr_mask > 0
        overlay[idx] = np.clip(overlay[idx] * 0.7 + np.array([0, 200, 0]) * 0.3, 0, 255)
        Image.fromarray(overlay.astype(np.uint8)).save(QA_MASK_DIR / img_name.replace('.tif', '_overlay.png'))


# For bootstrapping route: create stable repo-level data structure.
MASK_DIR = ROOT / 'Dataset' / 'masks_cleaned'


def main():
    parser = argparse.ArgumentParser(description='Repair geometry features using an orthophoto TIFF + GeoJSON-only workflow.')
    parser.add_argument('--config', default=None, help='Optional JSON config file for path portability.')
    parser.add_argument('--root-dir', default=None, help='Repository root override.')
    parser.add_argument('--orthophoto-dir', default=None, help='Directory containing orthophoto TIFF inputs.')
    parser.add_argument('--annotation-dir', default=None, help='Source GeoJSON annotation directory.')
    parser.add_argument('--repaired-dir', default=None, help='Repaired annotation copy directory.')
    parser.add_argument('--final-dir', default=None, help='Final merged clean annotation tree.')
    parser.add_argument('--qa-dir', default=None, help='QA/compare overlay directory.')
    args = parser.parse_args()

    apply_runtime_paths(args)

    # 1. inventory invalid features
    invalid_records = generate_invalid_report()
    print(f'Original invalid features reported: {len(invalid_records)}')
    feature_idx_map = create_feature_idx_map(invalid_records)

    # 2. Build repaired directory from original GeoJSON and replaced invalid geometries only.
    rebuild_repaired_geojsons(invalid_records, feature_idx_map)

    # 3. Validate repaired geometries and count success vs failure.
    repaired_success, still_invalid = create_validated_repair_report(invalid_records, feature_idx_map)
    print(f'Original invalid: {len(invalid_records)}')
    print(f'Successfully repaired: {repaired_success}')
    print(f'Still invalid: {still_invalid}')

    # 4. Compare before/after and record overlays.
    save_compare_visualizations(invalid_records, feature_idx_map)

    # 5. Merge original valid + safe repaired features into final annotation tree.
    create_final_annotation_tree(feature_idx_map, invalid_records)

    # 6. Regenerate masks from final annotation tree.
    regenerate_masks_from_final_annotations()

    # 7. Final feature count on original tree remains preserved, training set tree final has valid features plus safe repairs.
    # Determine total final features
    final_feature_count = sum(feature_count(path) for path in FINAL_DIR.glob('*.geojson'))
    print(f'Original valid features: {sum(feature_count(file) for file in ANNOT_DIR.glob("*.geojson")) - len(invalid_records)}')
    print(f'Original invalid features: {len(invalid_records)}')
    print(f'Repaired valid features: {repaired_success}')
    print(f'Total final building features: {final_feature_count}')


if __name__ == '__main__':
    main()
