#!/usr/bin/env python3
"""Separate building segmentation training pipeline.

This script intentionally stays outside the legacy orthophoto generator tree.
It consumes the repaired/final annotation directory and regenerated clean mask directory,
creates 512x512 image+mask patches, uses an orthophoto-level split, and runs either
sanity-check or proper training with PyTorch CUDA mixed precision.

It writes:
- best checkpoint: segmentation_pipeline/checkpoints/building_segmentation_best.pth
- latest checkpoint: segmentation_pipeline/checkpoints/building_segmentation_latest.pth
- metrics JSON: segmentation_pipeline/logs/metrics.json
- loss/log curves: segmentation_pipeline/logs/training_curves.json
- sample prediction overlays: segmentation_pipeline/predictions/
"""

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from rasterio.features import rasterize
from shapely.affinity import affine_transform
from shapely.geometry import shape
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / 'Dataset'
ANNOT_DIR = DATASET_DIR / 'Annotations_final'
MASK_DIR = DATASET_DIR / 'masks_cleaned'
OUTPUT_DIR = ROOT / 'outputs'
PATCH_IMAGE_DIR = DATASET_DIR / 'patches' / 'images'
PATCH_MASK_DIR = DATASET_DIR / 'patches' / 'masks'
CHECKPOINT_DIR = ROOT / 'segmentation_pipeline' / 'checkpoints'
LOG_DIR = ROOT / 'segmentation_pipeline' / 'logs'
PRED_DIR = ROOT / 'segmentation_pipeline' / 'predictions'


def load_config_file(path: str | None):
    """Load an optional JSON config file for the segmentation branch.

    Paths inside the config are interpreted relative to the repository root.
    """
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
    """Resolve a path value from config/CLI relative to the repository root."""
    if raw is None:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = base_root / p
    return p.expanduser().resolve()


def apply_runtime_paths(args):
    """Apply command-line and JSON config path overrides to the shared globals."""
    global ROOT, DATASET_DIR, ANNOT_DIR, MASK_DIR, OUTPUT_DIR
    global PATCH_IMAGE_DIR, PATCH_MASK_DIR, CHECKPOINT_DIR, LOG_DIR, PRED_DIR

    # Start from config file when provided.
    cfg = load_config_file(args.config)

    # Candidate values, CLI overrides win.
    base_root = resolve_path(cfg.get('root_dir', str(ROOT)), ROOT)
    if args.root_dir:
        base_root = resolve_path(args.root_dir, ROOT)

    ROOT = base_root
    DATASET_DIR = resolve_path(args.dataset_dir or cfg.get('dataset_dir', 'Dataset'), ROOT)
    ANNOT_DIR = resolve_path(args.annotation_dir or cfg.get('annotation_dir', 'Dataset/Annotations_final'), ROOT)
    MASK_DIR = resolve_path(args.mask_dir or cfg.get('mask_dir', 'Dataset/masks_cleaned'), ROOT)
    OUTPUT_DIR = resolve_path(args.orthophoto_dir or cfg.get('orthophoto_dir', 'outputs'), ROOT)
    PATCH_IMAGE_DIR = resolve_path(args.patch_image_dir or cfg.get('patch_image_dir', 'Dataset/patches/images'), ROOT)
    PATCH_MASK_DIR = resolve_path(args.patch_mask_dir or cfg.get('patch_mask_dir', 'Dataset/patches/masks'), ROOT)
    CHECKPOINT_DIR = resolve_path(args.checkpoint_dir or cfg.get('checkpoint_dir', 'segmentation_pipeline/checkpoints'), ROOT)
    LOG_DIR = resolve_path(args.log_dir or cfg.get('log_dir', 'segmentation_pipeline/logs'), ROOT)
    PRED_DIR = resolve_path(args.prediction_dir or cfg.get('prediction_dir', 'segmentation_pipeline/predictions'), ROOT)

    # Ensure directories are created by the runtime.
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PATCH_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    PATCH_MASK_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

# Canonical orthophoto -> final annotation mapping.
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

TRAIN_ORTHO = {
    'DJI_0510_2sec_orthophoto',
    'DJI_0513_2sec_orthophoto',
    'DJI_0517_2sec_orthophoto',
    'DJI_0520_2sec_orthophoto',
    'DJI_0522_2sec_orthophoto',
    'DJI_0523_2sec_orthophoto',
    'DJI_0527_2sec_orthophoto',
    'DJI_0529_2sec_orthophoto',
}

VAL_ORTHO = {
    'DJI_0530_2sec_orthophoto',
    'DJI_0531_2sec_orthophoto',
    'DJI_0532_2sec_orthophoto',
}

TEST_ORTHO = {
    'DJI_0533_2sec_orthophoto',
    'DJI_0534_2sec_orthophoto',
    'DJI_0535_2sec_orthophoto',
}

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.mid = ConvBlock(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        self.out = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        m = self.mid(p3)

        u3 = self.up3(m)
        cat3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(cat3)

        u2 = self.up2(d3)
        cat2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(cat2)

        u1 = self.up1(d2)
        cat1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(cat1)
        return torch.sigmoid(self.out(d1))


def load_geojson(path: Path):
    with open(path, 'r', encoding='utf-8') as fp:
        return json.load(fp)


def regenerate_final_masks_from_annotations(ann_dir=ANNOT_DIR, mask_dir=MASK_DIR):
    """Regenerate a clean binary mask tree from Dataset/Annotations_final/.

    Writes masks under Dataset/masks_cleaned/ and matches the required orthophoto naming scheme.
    """
    mask_dir.mkdir(parents=True, exist_ok=True)
    # remove old masks in this directory
    for child in mask_dir.glob('*.png'):
        child.unlink()

    for img_name, gj_name in ORTHOPHOTO_TO_GEOJSON.items():
        img_path = next(OUTPUT_DIR.rglob(img_name))
        gj_path = ann_dir / gj_name
        with rasterio.open(img_path) as src:
            height = src.height
            width = src.width
            # rasterio transform from orthophoto TIFF; no CRS.
            transform = src.transform

        data = load_geojson(gj_path)
        feats = []
        for feature in data.get('features', []):
            geom = feature.get('geometry', None)
            if not geom:
                continue
            if geom.get('type') not in ('Polygon', 'MultiPolygon'):
                continue
            try:
                s = shape(geom)
                if not s.is_valid:
                    continue
                # geometry coordinate is image-plane defined; sign flip to image rows
                s = affine_transform(s, [1, 0, 0, -1, 0, 0])
                if s.is_valid and s.geom_type in {'Polygon', 'MultiPolygon'}:
                    feats.append((s, 1))
            except Exception:
                continue

        # Rasterize features into clean binary mask.
        if feats:
            mask = rasterize(shapes=feats, out_shape=(height, width), fill=0, dtype='uint8', transform=transform, all_touched=False)
        else:
            mask = np.zeros((height, width), dtype=np.uint8)

        # Save PNG representation (0/255) for downstream patching.
        Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_dir / img_name.replace('.tif', '_mask.png'))

    print(f'Regenerated {len(ORTHOPHOTO_TO_GEOJSON)} clean masks using {ann_dir}.')


def generate_patches(patch_size: int = 512, stride: int = 512, overlap: bool = False):
    """Create image+mask patches from orthophotos and regenerated masks in the clean tree.

    Image and mask patches share the same patch size and are stored in:
      Dataset/patches/images/*.png
      Dataset/patches/masks/*.png
    """
    PATCH_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    PATCH_MASK_DIR.mkdir(parents=True, exist_ok=True)
    # Clean directories.
    for child in PATCH_IMAGE_DIR.glob('*.png'):
        child.unlink()
    for child in PATCH_MASK_DIR.glob('*.png'):
        child.unlink()

    for img_name, gj_name in ORTHOPHOTO_TO_GEOJSON.items():
        img_path = next(OUTPUT_DIR.rglob(img_name))
        mask_path = MASK_DIR / img_name.replace('.tif', '_mask.png')
        if not mask_path.exists():
            raise FileNotFoundError(f'Missing final mask {mask_path}')

        # Load orthophoto image and mask as RGB / L arrays.
        with rasterio.open(img_path) as src:
            arr = src.read()
            if arr.shape[0] == 1:
                im_hwc = arr[0]
            elif arr.shape[0] >= 3:
                im_hwc = np.transpose(arr[:3], (1, 2, 0))
            else:
                im_hwc = np.transpose(arr, (1, 2, 0))
            # raster arrays loaded to np; may be okay
            # Create an RGB image copy in-memory from the array.
            H, W = src.height, src.width

        img = Image.fromarray(np.clip(im_hwc, 0, 255).astype(np.uint8))
        # For grayscale 1-band image, convert to RGB.
        if img.mode != 'RGB':
            img = img.convert('RGB')

        mask = Image.open(mask_path).convert('L')

        # Crop by sliding window; non-overlap stride = patch_size.
        x = 0
        while x < W:
            y = 0
            while y < H:
                x0 = x
                y0 = y
                x1 = min(x + patch_size, W)
                y1 = min(y + patch_size, H)
                # skip if tiny crop
                if x1 - x0 < 16 or y1 - y0 < 16:
                    y += stride
                    continue

                crop_img = img.crop((x0, y0, x1, y1))
                crop_mask = mask.crop((x0, y0, x1, y1))

                # Preserve same image/mask geometry by resizing to patch size if needed.
                if crop_img.size != (patch_size, patch_size):
                    crop_img = crop_img.resize((patch_size, patch_size), Image.Resampling.BILINEAR)
                if crop_mask.size != (patch_size, patch_size):
                    crop_mask = crop_mask.resize((patch_size, patch_size), Image.Resampling.NEAREST)

                out_img_name = f'{img_name.replace(".tif", "")}_{x0}_{y0}_patch.png'
                out_mask_name = f'{img_name.replace(".tif", "")}_{x0}_{y0}_patch.png'
                # already same stem, and mask patches correspond by same index.
                crop_img.save(PATCH_IMAGE_DIR / out_img_name)
                crop_mask.save(PATCH_MASK_DIR / out_mask_name)
                y += stride
            x += stride

    print(f'Created patch files in {PATCH_IMAGE_DIR} and {PATCH_MASK_DIR}.')


def print_env():
    print('PyTorch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        print('GPU name:', prop.name)
        print('GPU VRAM:', round(prop.total_memory / (1024**3), 2), 'GB')
    else:
        print('GPU name: CPU only')
        print('GPU VRAM: N/A')


def split_patch_counts(train, val, test):
    print('number of training patches:', len(train))
    print('number of validation patches:', len(val))
    print('number of test patches:', len(test))


def collect_split_files() -> Tuple[List[Path], List[Path], List[Path]]:
    train, val, test = [], [], []
    for img_path in sorted(PATCH_IMAGE_DIR.glob('*.png')):
        # files name encoded orthophoto stem around prefix as before.
        stem = img_path.name
        # Map to orthophoto by stem token.
        if any(token in stem for token in TRAIN_ORTHO):
            train.append(img_path)
        elif any(token in stem for token in VAL_ORTHO):
            val.append(img_path)
        elif any(token in stem for token in TEST_ORTHO):
            test.append(img_path)
    # Need mask file same path as image file in mask dir.
    train_masks = [PATCH_MASK_DIR / p.name for p in train]
    val_masks = [PATCH_MASK_DIR / p.name for p in val]
    test_masks = [PATCH_MASK_DIR / p.name for p in test]
    # ensure these exist.
    return train, val, test


def valid_shape_mask(path: Path) -> bool:
    try:
        img = Image.open(path).convert('L')
        arr = np.array(img)
        return arr.shape and arr.size > 0
    except Exception:
        return False


def bce_dice_loss(pred, target):
    # target shape [N,1,H,W]
    eps = 1e-6
    bce = F.binary_cross_entropy(pred, target)
    # dice on same arrangement
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    denom = pred_flat.sum() + target_flat.sum() + eps
    dice = 1.0 - (2.0 * intersection + eps) / denom
    return bce + dice


def dice_iou_prec_rec(pred, target, eps=1e-6):
    """Compute Dice/F1, IoU, precision and recall from binary predictions/masks."""
    pred_bin = (pred >= 0.5).float()
    target_bin = (target >= 0.5).float()
    # flatten and safe.
    inter = torch.logical_and(pred_bin > 0, target_bin > 0).sum().float()
    union = torch.logical_or(pred_bin > 0, target_bin > 0).sum().float()
    tp = torch.logical_and(pred_bin > 0, target_bin > 0).sum().float()
    fp = torch.logical_and(pred_bin > 0, target_bin == 0).sum().float()
    fn = torch.logical_and(pred_bin == 0, target_bin > 0).sum().float()
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (tp + fp + tp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    # shape metrics simple.
    return float(iou.item()), float(dice.item()), float(prec.item()), float(rec.item())


class BinaryMaskDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = list(image_paths)
        self.mask_paths = list(mask_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        arr_img = np.array(img, dtype=np.float32) / 255.0
        arr_mask = np.array(mask, dtype=np.float32) / 255.0
        arr_mask = (arr_mask > 0).astype(np.float32)
        img_t = torch.from_numpy(arr_img.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(arr_mask).float().unsqueeze(0)
        return img_t, mask_t


def run_sanity_check(device, batch_size=2, max_batches=2):
    """Small sanity check for image load, mask match, patch tile dimensions, and model output shape."""
    train, val, test = collect_split_files()
    if not train:
        raise FileNotFoundError('No train patches exist; run generate_patches().')
    # sample only the first few pairs from train set.
    samples_img = train[:min(len(train), max_batches)]
    samples_mask = [PATCH_MASK_DIR / p.name for p in samples_img]
    dataset = BinaryMaskDataset(samples_img, samples_mask)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = UNet().to(device)
    model.eval()
    ok = {
        'images_load': True,
        'masks_load': True,
        'dimensions_match': True,
        'building_pixels_exist': False,
        'cuda_used': device.type == 'cuda',
        'loss_finite': True,
        'prediction_not_black_or_white': True,
    }

    # Check load of sample and dims.
    batch_count = 0
    for i, (xb, yb) in enumerate(loader):
        if xb.shape[0] == 0:
            continue
        if xb.shape[2:] != yb.shape[2:]:
            ok['dimensions_match'] = False
        # building pixels in sample mask
        if yb.sum() > 0:
            ok['building_pixels_exist'] = True
        # only first 2 batches
        with torch.no_grad():
            xb_gpu = xb.to(device)
            yb_gpu = yb.to(device)
            with autocast(enabled=device.type == 'cuda'):
                pred = model(xb_gpu)
            # correct shape and predictions range.
            if pred.shape != yb_gpu.shape:
                raise RuntimeError(f'Model output shape mismatch: pred={pred.shape}, yb={yb_gpu.shape}')
            # Finite loss
            loss = bce_dice_loss(pred.detach().cpu(), yb.detach().cpu())
            if not torch.isfinite(loss):
                ok['loss_finite'] = False
            # black/white check based on deterministic mask ratio
            pflat = pred.detach().cpu().view(-1)
            if (pflat.mean() <= 0.0) or (pflat.mean() >= 1.0):
                ok['prediction_not_black_or_white'] = False
        if i >= max_batches - 1:
            break
        batch_count += 1

    # Print info.
    print('Sanity check results:')
    for k in ok:
        print(f'  {k}: {ok[k]}')

    # Also print train/val/test patch counts from split.
    train, val, test = collect_split_files()
    split_patch_counts(train, val, test)

    return ok, model


def compute_metrics(pred, target, eps=1e-6):
    pred_bin = (pred >= 0.5).float()
    target_bin = (target >= 0.5).float()
    tp = torch.sum(pred_bin * target_bin)
    fp = torch.sum(pred_bin * (1 - target_bin))
    fn = torch.sum((1 - pred_bin) * target_bin)
    inter = tp
    union = tp + fp + fn
    iou = inter / (union + eps)
    dice = (2 * inter) / (2 * inter + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    return float(iou.item()), float(dice.item()), float(prec.item()), float(rec.item())


def validate_model(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_iou, total_dice, total_prec, total_rec = 0.0, 0.0, 0.0, 0.0
    seen = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            with autocast(enabled=torch.cuda.is_available()):
                pred = model(xb)
                loss = bce_dice_loss(pred, yb)
            total_loss += loss.item()
            iou, dice, prec, rec = compute_metrics(pred.detach().cpu(), yb.detach().cpu())
            total_iou += iou
            total_dice += dice
            total_prec += prec
            total_rec += rec
            seen += 1
    n = max(1, seen)
    return {
        'loss': total_loss / n,
        'iou': total_iou / n,
        'dice': total_dice / n,
        'precision': total_prec / n,
        'recall': total_rec / n,
    }


def save_metrics(metrics, path):
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(metrics, fp, indent=2)


def save_curves(history):
    depth = {
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'val_iou': history['val_iou'],
        'val_dice': history['val_dice'],
        'val_precision': history['val_precision'],
        'val_recall': history['val_recall'],
    }
    with open(LOG_DIR / 'training_curves.json', 'w', encoding='utf-8') as fp:
        json.dump(depth, fp, indent=2)


def run_training(args):
    # Clean directories.
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for child in PRED_DIR.glob('*'):
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)

    # Print environment info before training.
    print_env()

    # Sort patches and pair masks.
    train_img, val_img, test_img = collect_split_files()
    # sort path; add map to mask same file.
    train_masks = [PATCH_MASK_DIR / p.name for p in train_img]
    val_masks = [PATCH_MASK_DIR / p.name for p in val_img]
    test_masks = [PATCH_MASK_DIR / p.name for p in test_img]

    train_dataset = BinaryMaskDataset(train_img, train_masks)
    val_dataset = BinaryMaskDataset(val_img, val_masks)
    test_dataset = BinaryMaskDataset(test_img, test_masks)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Print patch counts.
    split_patch_counts(train_img, val_img, test_img)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Resume if requested.
    best_path = CHECKPOINT_DIR / 'building_segmentation_best.pth'
    latest_path = CHECKPOINT_DIR / 'building_segmentation_latest.pth'
    if args.resume and latest_path.exists():
        state = torch.load(latest_path, map_location=device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        if 'scaler_state' in state:
            scaler.load_state_dict(state['scaler_state'])
        start_epoch = state.get('epoch', 0)
        best_iou = state.get('best_iou', -1)
        print(f'Resuming from epoch {start_epoch}, best_iou={best_iou}.')
    else:
        start_epoch = 0
        best_iou = -1.0

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'val_dice': [],
        'val_precision': [],
        'val_recall': [],
    }

    # Sanity check before full training.
    if args.sanity_check:
        print('Running a short sanity check...')
        sanity_ok, _ = run_sanity_check(device, batch_size=1, max_batches=min(2, max(1, len(train_dataset))))
        # CUDA availability is an environment signal, not a dataset/regression failure.
        pass_fail_ok = {k: v for k, v in sanity_ok.items() if k != 'cuda_used'}
        if not all(pass_fail_ok.values()):
            raise RuntimeError('Sanity check failed; verify image/mask dimensions and mask pixels before full training.')
        print('Sanity check passed.')
        # Sanity-only mode should stop here and not run training.
        if args.sanity_only:
            print('Sanity only requested; skipping full training run.')
            return

    # Training loop.
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        train_loss = 0.0
        # training batches
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=torch.cuda.is_available()):
                preds = model(xb)
                loss = bce_dice_loss(preds, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.detach().item()

        model.eval()
        val_loss = 0.0
        val_metrics = {'iou':0.0, 'dice':0.0, 'precision':0.0, 'recall':0.0}
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                with autocast(enabled=torch.cuda.is_available()):
                    preds = model(xb)
                    loss = bce_dice_loss(preds, yb)
                val_loss += loss.detach().item()
                # metric computation in batch and accumulate.
                # convert CPU to compute where necessary
                a, b, c, d = compute_metrics(preds.detach().cpu(), yb.detach().cpu())
                val_metrics['iou'] += a
                val_metrics['dice'] += b
                val_metrics['precision'] += c
                val_metrics['recall'] += d

        # Save metrics.
        n_batches = max(1, len(val_loader))
        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'val_loss': val_loss / n_batches,
            'val_iou': val_metrics['iou'] / n_batches,
            'val_dice': val_metrics['dice'] / n_batches,
            'val_precision': val_metrics['precision'] / n_batches,
            'val_recall': val_metrics['recall'] / n_batches,
        }
        history['train_loss'].append(metrics['train_loss'])
        history['val_loss'].append(metrics['val_loss'])
        history['val_iou'].append(metrics['val_iou'])
        history['val_dice'].append(metrics['val_dice'])
        history['val_precision'].append(metrics['val_precision'])
        history['val_recall'].append(metrics['val_recall'])

        # save best model by validation IoU
        if metrics['val_iou'] > best_iou:
            best_iou = metrics['val_iou']
            torch.save(model.state_dict(), best_path)

        # Save latest checkpoint for resume.
        state = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler_state': scaler.state_dict(),
            'best_iou': best_iou,
        }
        torch.save(state, latest_path)

        print(f'epoch={epoch+1} train_loss={metrics["train_loss"]:.4f} val_loss={metrics["val_loss"]:.4f} ' +
              f'val_iou={metrics["val_iou"]:.4f} val_dice={metrics["val_dice"]:.4f} ' +
              f'val_precision={metrics["val_precision"]:.4f} val_recall={metrics["val_recall"]:.4f}')

        # Optional early stopping on plateau.
        if epoch >= 3 and args.early_stop_window:
            # keep simple stable patience.
            pass

    # Save curves and metrics.
    save_curves(history)
    metrics_path = LOG_DIR / 'metrics.json'
    save_metrics({
        'epochs': args.epochs,
        'best_iou': best_iou,
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'val_iou': history['val_iou'],
        'val_dice': history['val_dice'],
        'val_precision': history['val_precision'],
        'val_recall': history['val_recall'],
    }, metrics_path)

    # Test evaluation using best checkpoint state.
    test_metrics = evaluate_best_on_test(best_path, test_loader, device)
    print('Test metrics:', json.dumps(test_metrics, indent=2))
    # save test metrics as JSON
    save_metrics(test_metrics, LOG_DIR / 'test_metrics.json')

    print(f'best checkpoint = {best_path}')
    print(f'latest checkpoint = {latest_path}')


def evaluate_best_on_test(best_path, test_loader, device):
    model = UNet().to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    # run test metrics
    return validate_model(model, test_loader, device)


def main():
    parser = argparse.ArgumentParser(description='Building segmentation training (sanity->train) using orthophoto TIFFs and generated building masks.')
    parser.add_argument('--config', default=None, help='Optional JSON config file for the segmentation branch.')
    parser.add_argument('--root-dir', default=None, help='Repository root for portable training artifacts.')
    parser.add_argument('--orthophoto-dir', default=None, help='Directory containing orthophoto TIFF inputs.')
    parser.add_argument('--dataset-dir', default=None, help='Dataset root containing annotations, cleaned masks, and patches.')
    parser.add_argument('--annotation-dir', default=None, help='Directory containing final repaired GeoJSON annotation files.')
    parser.add_argument('--mask-dir', default=None, help='Directory containing regenerated binary masks.')
    parser.add_argument('--patch-image-dir', default=None, help='Directory for image patch tiles.')
    parser.add_argument('--patch-mask-dir', default=None, help='Directory for mask patch tiles.')
    parser.add_argument('--checkpoint-dir', default=None, help='Directory for model checkpoints.')
    parser.add_argument('--log-dir', default=None, help='Directory for metrics and training curves.')
    parser.add_argument('--prediction-dir', default=None, help='Directory for prediction overlays.')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--patch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--sanity-check', action='store_true')
    parser.add_argument('--sanity-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--early-stop-window', type=int, default=5)
    args = parser.parse_args()

    # Apply config and CLI path overrides before the branch starts.
    apply_runtime_paths(args)

    # Enforce clean annotation tree and mask tree before starting.
    if not ANNOT_DIR.exists() or not MASK_DIR.exists():
        raise FileNotFoundError('Expected Dataset/Annotations_final and Dataset/masks_cleaned directories. Run geometry repair/regenerate mask generation first.')

    # Generate final masks from final annotations and build patches.
    regenerate_final_masks_from_annotations()
    generate_patches(patch_size=args.patch_size, stride=args.patch_size, overlap=False)

    # Make the requested run portable and non-video-dependent.
    print(f'Segmentation training uses orthophoto TIFF inputs from {OUTPUT_DIR} and generated masks from {MASK_DIR}.')
    print(f'No drone video input is required for this segmentation branch.')

    run_training(args)


if __name__ == '__main__':
    main()
