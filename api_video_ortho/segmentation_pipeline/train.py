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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

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

# ImageNet normalization stats, required when using an ImageNet-pretrained
# encoder (smp.Unet with encoder_weights='imagenet'). The scratch UNet keeps
# using raw [0,1] inputs, matching its original training distribution.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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
        return self.out(d1)


def build_model(args):
    """Build either the original scratch UNet or an ImageNet-pretrained
    encoder U-Net (via segmentation-models-pytorch), based on args.

    Both paths return a model whose forward(x) yields raw logits of shape
    (N, 1, H, W), matching what bce_dice_loss / compute_metrics expect
    everywhere else in this file. Nothing downstream of the model needs to
    know which one is active.
    """
    if args.scratch_unet:
        return UNet()

    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            'segmentation-models-pytorch is required for --encoder/pretrained '
            'training. Install it with: pip install segmentation-models-pytorch '
            '(or pass --scratch-unet to use the original from-scratch UNet).'
        ) from exc

    encoder_weights = None if args.encoder_weights.lower() == 'none' else args.encoder_weights
    return smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,  # keep raw logits; loss/metrics apply sigmoid themselves
    )


def load_geojson(path: Path):
    with open(path, 'r', encoding='utf-8') as fp:
        return json.load(fp)


def regenerate_final_masks_from_annotations(ann_dir=None, mask_dir=None):
    ann_dir = ann_dir or ANNOT_DIR
    mask_dir = mask_dir or MASK_DIR
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


def focal_loss_from_logits(logits, target, gamma=2.0):
    """Compute focal loss from logits using stable BCE-with-logits values."""
    logits = logits.float()
    target = target.float()
    focal_bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    pt = torch.exp(-focal_bce)
    return ((1.0 - pt).pow(gamma) * focal_bce).mean()


def bce_dice_loss(logits, target, pos_weight=None, return_components=False):
    """Return a float32 BCE + per-image Dice loss, with focal kept separate."""
    eps = 1e-6
    logits = logits.float()
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight
    )
    probs = torch.sigmoid(logits)
    reduce_dims = tuple(range(1, probs.ndim))
    intersection = (probs * target).sum(dim=reduce_dims)
    pred_sum = probs.sum(dim=reduce_dims)
    target_sum = target.sum(dim=reduce_dims)
    denominator = pred_sum + target_sum + eps
    dice_score = (2.0 * intersection + eps) / denominator
    dice_loss = (1.0 - dice_score).mean()
    total = 0.5 * bce + 0.5 * dice_loss
    if return_components:
        return total, bce.detach(), dice_loss.detach()
    return total


def ensure_finite(tensor, name, epoch, batch_index):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(
            f'Non-finite {name} detected at epoch {epoch}, batch {batch_index}.'
        )


def _print_nonfinite_gradient_report(name, gradient, epoch, batch_index, loss_info=None):
    has_nan = torch.isnan(gradient).any().item()
    has_inf = torch.isinf(gradient).any().item()
    details = loss_info or {}
    print(
        'Non-finite gradient detected:\n'
        f'  epoch: {epoch}\n'
        f'  batch: {batch_index}\n'
        f'  parameter: {name}\n'
        f'  has_nan: {has_nan}\n'
        f'  has_inf: {has_inf}\n'
        f'  gradient_min: {gradient.min().item() if not has_nan and not has_inf else "unavailable"}\n'
        f'  gradient_max: {gradient.max().item() if not has_nan and not has_inf else "unavailable"}\n'
        f'  loss: {details.get("loss", "unavailable")}\n'
        f'  bce: {details.get("bce", "unavailable")}\n'
        f'  dice: {details.get("dice", "unavailable")}\n'
        f'  logits_min: {details.get("logits_min", "unavailable")}\n'
        f'  logits_max: {details.get("logits_max", "unavailable")}\n'
        f'  prob_min: {details.get("prob_min", "unavailable")}\n'
        f'  prob_max: {details.get("prob_max", "unavailable")}\n'
        f'  target_min: {details.get("target_min", "unavailable")}\n'
        f'  target_max: {details.get("target_max", "unavailable")}\n'
        f'  target_positive_pixels: {details.get("target_positive_pixels", "unavailable")}\n'
        f'  target_positive_fraction: {details.get("target_positive_fraction", "unavailable")}'
    )


def ensure_finite_gradients(model, epoch, batch_index, loss_info=None):
    """Raise on any non-finite gradient. Use only when NOT using GradScaler
    (i.e. the --no-amp / true fp32 path), where a non-finite gradient reflects
    genuine numerical instability rather than an expected transient fp16
    overflow that GradScaler is designed to detect and recover from on its own.
    """
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            gradient = parameter.grad.detach().float()
            _print_nonfinite_gradient_report(name, gradient, epoch, batch_index, loss_info)
            raise RuntimeError(
                f'Non-finite gradient detected for {name} at epoch {epoch}, batch {batch_index}.'
            )


def check_finite_gradients_amp(model, epoch, batch_index, loss_info=None):
    """Non-raising gradient check for the AMP/GradScaler path.

    A transient non-finite gradient after scaler.unscale_() is normal, expected
    GradScaler behaviour (the loss-scale factor is occasionally too high; fp16
    overflows during backward before unscaling). GradScaler.step()/update()
    already detect this internally: they skip the optimizer step for this
    batch and shrink the scale factor for future batches. This function only
    reports what happened for visibility; it must NOT raise, or it turns a
    self-correcting AMP event into a hard crash.

    Returns True if all gradients are finite, False if an overflow was
    detected (in which case the caller should skip gradient clipping and let
    scaler.step()/scaler.update() handle the skip + scale backoff).
    """
    all_finite = True
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            all_finite = False
            gradient = parameter.grad.detach().float()
            _print_nonfinite_gradient_report(name, gradient, epoch, batch_index, loss_info)
            break
    return all_finite


def ensure_finite_parameters(model, epoch, batch_index):
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(
                f'Non-finite parameter detected for {name} at epoch {epoch}, batch {batch_index}.'
            )


def validate_batch(xb, yb, epoch, batch_index):
    ensure_finite(xb, 'input batch', epoch, batch_index)
    ensure_finite(yb, 'target batch', epoch, batch_index)
    if xb.numel() and batch_index == 0:
        print(f'First batch input range: {xb.min().item():.6f} to {xb.max().item():.6f}')
    if yb.numel() and (yb.min().item() < 0.0 or yb.max().item() > 1.0):
        raise RuntimeError(
            f'Target values outside [0, 1] detected at epoch {epoch}, batch {batch_index}: '
            f'min={yb.min().item()}, max={yb.max().item()}'
        )
    if yb.numel() and not torch.all((yb == 0.0) | (yb == 1.0)):
        raise RuntimeError(
            f'Target contains non-binary values at epoch {epoch}, batch {batch_index}.'
        )


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
    def __init__(self, image_paths, mask_paths, normalize_imagenet=False):
        self.image_paths = list(image_paths)
        self.mask_paths = list(mask_paths)
        self.normalize_imagenet = normalize_imagenet

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        arr_img = np.array(img, dtype=np.float32) / 255.0
        if self.normalize_imagenet:
            arr_img = (arr_img - IMAGENET_MEAN) / IMAGENET_STD
        arr_mask = np.array(mask, dtype=np.float32) / 255.0
        arr_mask = (arr_mask > 0).astype(np.float32)
        img_t = torch.from_numpy(arr_img.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(arr_mask).float().unsqueeze(0)
        return img_t, mask_t


def run_sanity_check(args, device, batch_size=2, max_batches=2, amp_enabled=True):
    """Small sanity check for image load, mask match, patch tile dimensions, and model output shape."""
    train, val, test = collect_split_files()
    if not train:
        raise FileNotFoundError('No train patches exist; run generate_patches().')
    # sample only the first few pairs from train set.
    samples_img = train[:min(len(train), max_batches)]
    samples_mask = [PATCH_MASK_DIR / p.name for p in samples_img]
    dataset = BinaryMaskDataset(samples_img, samples_mask, normalize_imagenet=not args.scratch_unet)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = build_model(args).to(device)
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
        validate_batch(xb, yb, 0, i)
        if xb.shape[2:] != yb.shape[2:]:
            ok['dimensions_match'] = False
        # building pixels in sample mask
        if yb.sum() > 0:
            ok['building_pixels_exist'] = True
        # only first 2 batches
        with torch.no_grad():
            xb_gpu = xb.to(device)
            yb_gpu = yb.to(device)
            with autocast(enabled=amp_enabled and device.type == 'cuda'):
                pred = model(xb_gpu)
            # correct shape and predictions range.
            if pred.shape != yb_gpu.shape:
                raise RuntimeError(f'Model output shape mismatch: pred={pred.shape}, yb={yb_gpu.shape}')
            ensure_finite(pred, 'model output', 0, i)
            pred_probs = torch.sigmoid(pred)
            # Finite loss
            loss = bce_dice_loss(pred.detach().cpu(), yb.detach().cpu())
            ensure_finite(loss, 'loss', 0, i)
            if not torch.isfinite(loss):
                ok['loss_finite'] = False
            # black/white check based on deterministic mask ratio
            pflat = pred_probs.detach().cpu().view(-1)
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


def compute_metrics(pred, target, eps=1e-6, threshold=0.5):
    pred_bin = (pred >= threshold).float()
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


def pixel_percentages(pred, target, threshold=0.5):
    pred_pct = float((pred >= threshold).float().mean().item() * 100.0)
    target_pct = float((target >= 0.5).float().mean().item() * 100.0)
    return pred_pct, target_pct


def validate_model(model, loader, device, pos_weight=None, amp_enabled=True):
    model.eval()
    total_loss = 0.0
    total_iou, total_dice, total_prec, total_rec = 0.0, 0.0, 0.0, 0.0
    total_pred_pct, total_target_pct = 0.0, 0.0
    threshold_values = (0.30, 0.40, 0.50, 0.60, 0.70)
    threshold_ious = {threshold: 0.0 for threshold in threshold_values}
    seen = 0
    with torch.no_grad():
        for batch_index, (xb, yb) in enumerate(loader):
            validate_batch(xb, yb, 0, batch_index)
            xb = xb.to(device)
            yb = yb.to(device)
            with autocast(enabled=amp_enabled and device.type == 'cuda'):
                logits = model(xb)
                ensure_finite(logits, 'model output', 0, batch_index)
            loss, bce_component, dice_component = bce_dice_loss(
                logits, yb, pos_weight, return_components=True
            )
            ensure_finite(loss, 'loss', 0, batch_index)
            ensure_finite(bce_component, 'BCE loss', 0, batch_index)
            ensure_finite(dice_component, 'Dice loss', 0, batch_index)
            total_loss += loss.item()
            probs = torch.sigmoid(logits.float())
            probs_cpu = probs.detach().cpu()
            target_cpu = yb.detach().cpu()
            iou, dice, prec, rec = compute_metrics(probs_cpu, target_cpu)
            pred_pct, target_pct = pixel_percentages(probs_cpu, target_cpu)
            total_iou += iou
            total_dice += dice
            total_prec += prec
            total_rec += rec
            total_pred_pct += pred_pct
            total_target_pct += target_pct
            for threshold in threshold_values:
                threshold_ious[threshold] += compute_metrics(
                    probs_cpu, target_cpu, threshold=threshold
                )[0]
            seen += 1
    n = max(1, seen)
    threshold_results = {threshold: value / n for threshold, value in threshold_ious.items()}
    print('Validation threshold sweep:')
    for threshold, iou in threshold_results.items():
        print(f'  {threshold:.2f} -> IoU {iou:.4f}')
    print(
        f'Predicted building pixels: {total_pred_pct / n:.2f}% | '
        f'Actual building pixels: {total_target_pct / n:.2f}%'
    )
    return {
        'loss': total_loss / n,
        'iou': total_iou / n,
        'dice': total_dice / n,
        'precision': total_prec / n,
        'recall': total_rec / n,
        'predicted_building_pct': total_pred_pct / n,
        'actual_building_pct': total_target_pct / n,
        'threshold_iou': {f'{threshold:.2f}': value for threshold, value in threshold_results.items()},
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

    train_dataset = BinaryMaskDataset(train_img, train_masks, normalize_imagenet=not args.scratch_unet)
    val_dataset = BinaryMaskDataset(val_img, val_masks, normalize_imagenet=not args.scratch_unet)
    test_dataset = BinaryMaskDataset(test_img, test_masks, normalize_imagenet=not args.scratch_unet)

    building_fractions = []
    positive_pixels = 0
    negative_pixels = 0
    for mask_path in train_masks:
        with Image.open(mask_path) as mask_image:
            mask = np.asarray(mask_image.convert('L'))
        positive = int(np.count_nonzero(mask > 0))
        total = int(mask.size)
        positive_pixels += positive
        negative_pixels += total - positive
        building_fractions.append(positive / max(1, total))

    empty_count = sum(fraction == 0.0 for fraction in building_fractions)
    low_count = sum(0.0 < fraction <= 0.01 for fraction in building_fractions)
    medium_count = sum(0.01 < fraction <= 0.10 for fraction in building_fractions)
    high_count = sum(fraction > 0.10 for fraction in building_fractions)
    sample_weights = []
    for fraction in building_fractions:
        if fraction == 0.0:
            sample_weights.append(0.5)
        elif fraction <= 0.01:
            sample_weights.append(1.5)
        elif fraction <= 0.10:
            sample_weights.append(3.0)
        else:
            sample_weights.append(4.0)

    if building_fractions:
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_dataset),
            replacement=True,
        )
        print(
            'Training patch sampling:\n'
            f'  total patches: {len(building_fractions)}\n'
            f'  empty patches: {empty_count}\n'
            f'  low-building patches (0-1%): {low_count}\n'
            f'  medium-building patches (1-10%): {medium_count}\n'
            f'  high-building patches (>10%): {high_count}\n'
            f'  building fraction min/max/mean: {min(building_fractions):.6f}/'
            f'{max(building_fractions):.6f}/{np.mean(building_fractions):.6f}\n'
            '  sampling strategy: replacement with fraction-based weights '
            '(empty=0.5, low=1.5, medium=3.0, high=4.0)'
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    else:
        print('Training patch sampling: weighted sampling disabled because no patches were found.')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Print patch counts.
    split_patch_counts(train_img, val_img, test_img)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args).to(device)
    if args.scratch_unet:
        print('Model: scratch UNet (32/64/128/256 channels, no pretrained weights)')
    else:
        print(f'Model: smp.Unet(encoder_name={args.encoder!r}, encoder_weights={args.encoder_weights!r})')
    amp_enabled = torch.cuda.is_available() and not args.no_amp
    scaler = GradScaler() if amp_enabled else None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f'Learning rate: {args.lr:.6g}')
    pixel_pos_weight = negative_pixels / max(1, positive_pixels)
    pos_weight = min(5.0, max(1.0, pixel_pos_weight))
    pos_weight = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    print(f'BCE positive pixel weight: {pos_weight.item():.4f} (raw ratio: {pixel_pos_weight:.4f})')

    current_model_tag = 'scratch_unet' if args.scratch_unet else f'smp_unet_{args.encoder}'

    # Resume if requested.
    best_path = CHECKPOINT_DIR / 'building_segmentation_best.pth'
    latest_path = CHECKPOINT_DIR / 'building_segmentation_latest.pth'
    if args.resume and latest_path.exists():
        state = torch.load(latest_path, map_location=device)
        saved_model_tag = state.get('model_type')
        if saved_model_tag is not None and saved_model_tag != current_model_tag:
            raise RuntimeError(
                f'--resume checkpoint was trained with model_type={saved_model_tag!r} but '
                f'this run is configured for model_type={current_model_tag!r}. Architectures '
                'are not interchangeable (different layer names/shapes). Use a fresh '
                '--checkpoint-dir for this architecture, or drop --resume.'
            )
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        if scaler is not None and 'scaler_state' in state:
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
        sanity_ok, _ = run_sanity_check(
            args,
            device,
            batch_size=1,
            max_batches=min(2, max(1, len(train_dataset))),
            amp_enabled=amp_enabled,
        )
        # CUDA availability is an environment signal, not a dataset/regression failure.
        pass_fail_ok = {k: v for k, v in sanity_ok.items() if k != 'cuda_used'}
        if not all(pass_fail_ok.values()):
            raise RuntimeError('Sanity check failed; verify image/mask dimensions and mask pixels before full training.')
        print('Sanity check passed.')
        # Sanity-only mode should stop here and not run training.
        if args.sanity_only:
            print('Sanity only requested; skipping full training run.')
            return

    amp_overflow_count = 0

    # Training loop.
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        train_loss = 0.0
        train_bce = 0.0
        train_dice = 0.0
        epoch_amp_overflows = 0
        # training batches
        for batch_index, (xb, yb) in enumerate(train_loader):
            validate_batch(xb, yb, epoch + 1, batch_index)
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled):
                logits = model(xb)
                ensure_finite(logits, 'model output', epoch + 1, batch_index)
            loss, bce_component, dice_component = bce_dice_loss(
                logits, yb, pos_weight, return_components=True
            )
            ensure_finite(loss, 'loss', epoch + 1, batch_index)
            ensure_finite(bce_component, 'BCE loss', epoch + 1, batch_index)
            ensure_finite(dice_component, 'Dice loss', epoch + 1, batch_index)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            loss_info = {
                'loss': loss.detach().item(),
                'bce': bce_component.item(),
                'dice': dice_component.item(),
                'logits_min': logits.detach().float().min().item(),
                'logits_max': logits.detach().float().max().item(),
                'prob_min': torch.sigmoid(logits.detach().float()).min().item(),
                'prob_max': torch.sigmoid(logits.detach().float()).max().item(),
                'target_min': yb.detach().float().min().item(),
                'target_max': yb.detach().float().max().item(),
                'target_positive_pixels': int((yb.detach() > 0.5).sum().item()),
                'target_positive_fraction': (yb.detach() > 0.5).float().mean().item(),
            }
            if scaler is not None:
                # AMP path: a transient non-finite gradient here is expected,
                # self-correcting GradScaler behaviour (see
                # check_finite_gradients_amp docstring). Report it, skip
                # clipping for this batch, and let scaler.step()/update()
                # skip the optimizer step and back off the scale factor.
                grads_finite = check_finite_gradients_amp(model, epoch + 1, batch_index, loss_info)
                if grads_finite:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                else:
                    amp_overflow_count += 1
                    epoch_amp_overflows += 1
                    print(
                        f'AMP overflow #{amp_overflow_count}: skipping optimizer step for '
                        f'epoch {epoch + 1}, batch {batch_index}; GradScaler will shrink '
                        'its scale factor automatically.'
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                # No-amp / true fp32 path: there is no loss scaling, so a
                # non-finite gradient here reflects genuine numerical
                # instability, not an AMP artifact. Fail fast as before.
                ensure_finite_gradients(model, epoch + 1, batch_index, loss_info)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            ensure_finite_parameters(model, epoch + 1, batch_index)
            train_loss += loss.detach().item()
            train_bce += bce_component.item()
            train_dice += dice_component.item()

        model.eval()
        val_loss = 0.0
        val_metrics = {'iou':0.0, 'dice':0.0, 'precision':0.0, 'recall':0.0}
        threshold_values = (0.30, 0.40, 0.50, 0.60, 0.70)
        threshold_ious = {threshold: 0.0 for threshold in threshold_values}
        with torch.no_grad():
            for batch_index, (xb, yb) in enumerate(val_loader):
                validate_batch(xb, yb, epoch + 1, batch_index)
                xb = xb.to(device)
                yb = yb.to(device)
                with autocast(enabled=amp_enabled):
                    logits = model(xb)
                    ensure_finite(logits, 'model output', epoch + 1, batch_index)
                loss = bce_dice_loss(logits, yb, pos_weight)
                ensure_finite(loss, 'loss', epoch + 1, batch_index)
                val_loss += loss.detach().item()
                # metric computation in batch and accumulate.
                # convert CPU to compute where necessary
                probs = torch.sigmoid(logits.float())
                probs_cpu = probs.detach().cpu()
                target_cpu = yb.detach().cpu()
                a, b, c, d = compute_metrics(probs_cpu, target_cpu)
                val_metrics['iou'] += a
                val_metrics['dice'] += b
                val_metrics['precision'] += c
                val_metrics['recall'] += d
                pred_pct, target_pct = pixel_percentages(probs_cpu, target_cpu)
                val_metrics.setdefault('predicted_building_pct', 0.0)
                val_metrics.setdefault('actual_building_pct', 0.0)
                val_metrics['predicted_building_pct'] += pred_pct
                val_metrics['actual_building_pct'] += target_pct
                for threshold in threshold_values:
                    threshold_ious[threshold] += compute_metrics(
                        probs_cpu, target_cpu, threshold=threshold
                    )[0]

        # Save metrics.
        n_batches = max(1, len(val_loader))
        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'train_bce': train_bce / len(train_loader),
            'train_dice': train_dice / len(train_loader),
            'val_loss': val_loss / n_batches,
            'val_iou': val_metrics['iou'] / n_batches,
            'val_dice': val_metrics['dice'] / n_batches,
            'val_precision': val_metrics['precision'] / n_batches,
            'val_recall': val_metrics['recall'] / n_batches,
            'predicted_building_pct': val_metrics.get('predicted_building_pct', 0.0) / n_batches,
            'actual_building_pct': val_metrics.get('actual_building_pct', 0.0) / n_batches,
            'threshold_iou': {
                f'{threshold:.2f}': value / n_batches
                for threshold, value in threshold_ious.items()
            },
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
            'best_iou': best_iou,
            'model_type': current_model_tag,
        }
        if scaler is not None:
            state['scaler_state'] = scaler.state_dict()
        torch.save(state, latest_path)

        print('epoch={} train_loss={:.4f} bce={:.4f} dice={:.4f} val_loss={:.4f} val_iou={:.4f} val_dice={:.4f} val_precision={:.4f} val_recall={:.4f} pred_building={:.2f}% actual_building={:.2f}%'.format(epoch + 1, metrics['train_loss'], metrics['train_bce'], metrics['train_dice'], metrics['val_loss'], metrics['val_iou'], metrics['val_dice'], metrics['val_precision'], metrics['val_recall'], metrics['predicted_building_pct'], metrics['actual_building_pct']))
        print('Validation threshold sweep:')
        for threshold, iou in metrics['threshold_iou'].items():
            print(f'  {threshold} -> IoU {iou:.4f}')
        if scaler is not None:
            print(f'AMP overflow batches this epoch: {epoch_amp_overflows} / {len(train_loader)} '
                  f'(current scale factor: {scaler.get_scale():.1f})')

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
    test_metrics = evaluate_best_on_test(args, best_path, test_loader, device, pos_weight, amp_enabled)
    test_metrics['model_type'] = 'scratch_unet' if args.scratch_unet else f'smp_unet_{args.encoder}'
    print('Test metrics:', json.dumps(test_metrics, indent=2))
    # save test metrics as JSON
    save_metrics(test_metrics, LOG_DIR / 'test_metrics.json')

    print(f'best checkpoint = {best_path}')
    print(f'latest checkpoint = {latest_path}')


def evaluate_best_on_test(args, best_path, test_loader, device, pos_weight=None, amp_enabled=True):
    model = build_model(args).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    # run test metrics
    return validate_model(model, test_loader, device, pos_weight, amp_enabled)


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
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--scratch-unet', action='store_true',
                         help='Use the original from-scratch UNet (32/64/128/256 channels, no '
                              'pretrained weights) instead of an ImageNet-pretrained encoder. '
                              'Use this to A/B against the pretrained-encoder default.')
    parser.add_argument('--encoder', default='resnet34',
                         help='Encoder backbone for the pretrained U-Net (segmentation-models-pytorch '
                              'name, e.g. resnet34, resnet50, efficientnet-b0). Ignored with --scratch-unet.')
    parser.add_argument('--encoder-weights', default='imagenet',
                         help="Pretrained weights for the encoder, e.g. 'imagenet' or 'none' for "
                              'random init with the same architecture. Ignored with --scratch-unet.')
    parser.add_argument('--sanity-check', action='store_true')
    parser.add_argument('--sanity-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-amp', action='store_true', help='Disable CUDA AMP for numerical-stability testing.')
    parser.add_argument('--early-stop-window', type=int, default=5)
    args = parser.parse_args()

    # Apply config and CLI path overrides before the branch starts.
    apply_runtime_paths(args)

    # Enforce clean annotation tree and mask tree before starting.
    if not ANNOT_DIR.exists() or not MASK_DIR.exists():
        raise FileNotFoundError('Expected Dataset/Annotations_final and Dataset/masks_cleaned directories. Run geometry repair/regenerate mask generation first.')

    # Generate final masks from final annotations and build patches.
    regenerate_final_masks_from_annotations(ANNOT_DIR, MASK_DIR)
    generate_patches(patch_size=args.patch_size, stride=args.patch_size, overlap=False)

    # Make the requested run portable and non-video-dependent.
    print(f'Segmentation training uses orthophoto TIFF inputs from {OUTPUT_DIR} and generated masks from {MASK_DIR}.')
    print(f'No drone video input is required for this segmentation branch.')

    run_training(args)


if __name__ == '__main__':
    main()