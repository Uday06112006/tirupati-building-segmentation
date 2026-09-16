"""Memory-aware full-orthophoto Mask R-CNN inference and output writer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .dataset import discover_images
from .metrics import box_iou, mask_iou
from .model import build_model, load_checkpoint
from .utils import find_annotation, iter_tiles, orthophoto_id, read_window, draw_instances


def suppress_duplicates(predictions, mask_iou_threshold=0.5, box_iou_threshold=0.7):
    kept = []
    for candidate in sorted(predictions, key=lambda item: item["score"], reverse=True):
        duplicate = any(mask_iou(candidate["mask"], existing["mask"]) >= mask_iou_threshold or box_iou(candidate["box"], existing["box"]) >= box_iou_threshold for existing in kept)
        if not duplicate:
            kept.append(candidate)
    return kept


@torch.no_grad()
def predict_image(model, image_path, device, tile_size, stride, score_threshold):
    import rasterio
    predictions = []
    with rasterio.open(image_path) as src:
        width, height = src.width, src.height
        for x, y, tile_width, tile_height in iter_tiles(width, height, tile_size, stride):
            tile = read_window(src, x, y, tile_width, tile_height)
            tensor = torch.from_numpy(tile.transpose(2, 0, 1)).float().div(255).to(device)
            output = model([tensor])[0]
            for mask, score, box_tensor in zip(output["masks"][:, 0].cpu().numpy(), output["scores"].cpu().numpy(), output["boxes"].cpu().numpy()):
                if score < score_threshold:
                    continue
                local_mask = mask > 0.5
                ys, xs = np.where(local_mask)
                if not len(xs):
                    continue
                full_mask = np.zeros((height, width), dtype=bool)
                full_mask[y:y + tile_height, x:x + tile_width] = local_mask
                predictions.append({"mask": full_mask, "score": float(score), "box": [float(box_tensor[0] + x), float(box_tensor[1] + y), float(box_tensor[2] + x), float(box_tensor[3] + y)]})
        original = read_window(src, 0, 0, width, height)
    return original, suppress_duplicates(predictions)


def make_parser():
    parser = argparse.ArgumentParser(description="Run tiled Mask R-CNN inference on orthophoto TIFFs")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--orthophoto-dir", default=None)
    parser.add_argument("--annotation-dir", default=None, help="Accepted for CLI symmetry; annotations are not needed for prediction")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="maskrcnn_outputs/inference")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--image", default=None, help="Process one TIFF instead of the test split")
    return parser


def resolve_path(value, root):
    path = Path(value)
    return path if path.is_absolute() else root / path


def main(args=None):
    args = make_parser().parse_args(args)
    root = Path(args.root_dir).expanduser().resolve()
    image_dir = resolve_path(args.orthophoto_dir or "Dataset/Orthophotos", root)
    output_dir = resolve_path(args.output_dir, root); output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(pretrained=False).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    images = [Path(args.image)] if args.image else [path for path in discover_images(image_dir) if orthophoto_id(path) in {"DJI_0533", "DJI_0534", "DJI_0535"}]
    if not images:
        raise RuntimeError("No test orthophotos found")
    for image_path in images:
        original, predictions = predict_image(model, image_path, device, args.tile_size, args.stride, args.score_threshold)
        image_output = output_dir / orthophoto_id(image_path); mask_dir = image_output / "masks"; mask_dir.mkdir(parents=True, exist_ok=True)
        combined = np.zeros(original.shape[:2], dtype=np.uint8)
        metadata = []
        masks, boxes, scores = [], [], []
        for index, prediction in enumerate(predictions, 1):
            mask = prediction["mask"]
            combined[mask] = 255
            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_dir / f"instance_{index:04d}.png")
            ys, xs = np.where(mask)
            metadata.append({"instance_id": index, "score": prediction["score"], "bounding_box": prediction["box"], "area": int(mask.sum())})
            masks.append(mask); boxes.append(prediction["box"]); scores.append(prediction["score"])
        Image.fromarray(original).save(image_output / "original.png")
        Image.fromarray(combined).save(image_output / "building_mask.png")
        overlay = draw_instances(original, np.asarray(masks, dtype=bool), np.asarray(boxes, dtype=np.float32).reshape((-1, 4)), scores)
        Image.fromarray(overlay).save(image_output / "prediction_visualization.png")
        (image_output / "instances.json").write_text(json.dumps({"image": image_path.name, "coordinate_system": "image_pixels_only", "instances": metadata}, indent=2), encoding="utf-8")
        print(f"{image_path.name}: {len(metadata)} instances -> {image_output}")


if __name__ == "__main__":
    main()