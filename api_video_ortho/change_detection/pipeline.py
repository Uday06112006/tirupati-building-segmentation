"""Run before/after building change detection, F1 by default."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from maskrcnn_pipeline.model import build_model, load_checkpoint

from .change_classifier import classify_changes
from .inference import detect_instances
from .matching import match_instances, transform_detection
from .register import register_images
from .visualize import save_alignment_qa, save_change_visualization, save_detection_visualization, save_matched_visualization


def resolve_path(value, root):
    path = Path(value)
    return path if path.is_absolute() else root / path


def json_detection(detection, mask_path):
    return {key: value for key, value in detection.items() if key != "mask"} | {"mask_path": str(mask_path)}


def save_detections(payload, output_dir):
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in payload.items() if key != "instances"}
    serializable["instances"] = []
    for instance in payload["instances"]:
        mask_path = mask_dir / f"instance_{instance['instance_id']:04d}.png"
        Image.fromarray((instance["mask"] * 255).astype(np.uint8)).save(mask_path)
        serializable["instances"].append(json_detection(instance, mask_path.relative_to(output_dir)))
    return serializable


def run_pair(pair, root, args):
    before_path = resolve_path(Path("outputs") / pair["before"], root)
    after_path = resolve_path(Path("outputs") / pair["after"], root)
    for path in (before_path, after_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing pair input: {path}")
    output_dir = resolve_path(args.output_dir, root) / pair["id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(pretrained=False).to(device)
    load_checkpoint(model, resolve_path(args.checkpoint, root), device)
    model.eval()
    registration = register_images(before_path, after_path, args.registration_method)
    (output_dir / "registration.json").write_text(json.dumps({key: value for key, value in registration.items() if not isinstance(value, np.ndarray)}, indent=2), encoding="utf-8")
    save_alignment_qa(registration["before_preview"], registration["after_preview"], registration["preview_matrix"], output_dir / "alignment.png")
    if not registration["quality_ok"] and not args.allow_poor_registration:
        raise RuntimeError(f"Registration quality is poor for {pair['id']} (inlier ratio {registration['inlier_ratio']:.3f}); inspect alignment.png or use --allow-poor-registration.")
    before = detect_instances(model, before_path, device, args.tile_size, args.stride, args.score_threshold, args.mask_threshold, args.merge_mask_iou, args.merge_box_iou)
    after = detect_instances(model, after_path, device, args.tile_size, args.stride, args.score_threshold, args.mask_threshold, args.merge_mask_iou, args.merge_box_iou)
    after_aligned = dict(after)
    after_aligned["instances"] = [transformed for instance in after["instances"] if (transformed := transform_detection(instance, registration["matrix"], (before["height"], before["width"]))) is not None]
    before_json = save_detections(before, output_dir / "before")
    after_json = save_detections(after, output_dir / "after")
    aligned_json = save_detections(after_aligned, output_dir / "after_aligned")
    matching = match_instances(before, after_aligned, args.mask_iou_threshold, args.centroid_distance)
    changes = classify_changes(before, after_aligned, matching, args.area_change_threshold, args.changed_mask_iou_threshold)
    (output_dir / "before_detections.json").write_text(json.dumps(before_json, indent=2), encoding="utf-8")
    (output_dir / "after_detections.json").write_text(json.dumps(after_json, indent=2), encoding="utf-8")
    (output_dir / "after_aligned_detections.json").write_text(json.dumps(aligned_json, indent=2), encoding="utf-8")
    (output_dir / "matches.json").write_text(json.dumps(matching, indent=2), encoding="utf-8")
    (output_dir / "changes.json").write_text(json.dumps(changes, indent=2), encoding="utf-8")
    with __import__("rasterio").open(before_path) as source:
        from maskrcnn_pipeline.utils import read_window
        before_image = read_window(source, 0, 0, source.width, source.height)
    with __import__("rasterio").open(after_path) as source:
        from maskrcnn_pipeline.utils import read_window
        after_image = read_window(source, 0, 0, source.width, source.height)
    save_detection_visualization(before_image, before, output_dir / "before_detections.png")
    save_detection_visualization(after_image, after, output_dir / "after_detections.png")
    save_matched_visualization(before_image, before, after_aligned, changes, output_dir / "matched_buildings.png")
    save_change_visualization(before_image, before, after_aligned, changes, output_dir / "change_detection.png")
    summary = {name: sum(item["classification"] == name for item in changes) for name in ("UNCHANGED", "NEW_BUILDING", "REMOVED_BUILDING", "CHANGED_BUILDING")}
    print(json.dumps({"pair": pair["id"], "before_dimensions": [before["width"], before["height"]], "after_dimensions": [after["width"], after["height"]], "before_detections": len(before["instances"]), "after_detections": len(after["instances"]), "matched_buildings": len(matching["matches"]), **{key.lower(): value for key, value in summary.items()}}, indent=2))


def make_parser():
    parser = argparse.ArgumentParser(description="Detect building changes using a trained Mask R-CNN")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--pair-id", default="F1")
    parser.add_argument("--all-pairs", action="store_true")
    parser.add_argument("--pairs-file", default="change_detection/pairs.json")
    parser.add_argument("--checkpoint", default="maskrcnn_outputs/checkpoints/best.pt")
    parser.add_argument("--output-dir", default="maskrcnn_outputs/change_detection")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--registration-method", choices=("auto", "affine", "homography"), default="auto")
    parser.add_argument("--mask-iou-threshold", type=float, default=0.1)
    parser.add_argument("--centroid-distance", type=float, default=250.0)
    parser.add_argument("--area-change-threshold", type=float, default=0.30)
    parser.add_argument("--changed-mask-iou-threshold", type=float, default=0.50)
    parser.add_argument("--merge-mask-iou", type=float, default=0.5)
    parser.add_argument("--merge-box-iou", type=float, default=0.7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-poor-registration", action="store_true")
    return parser


def main(args=None):
    arguments = make_parser().parse_args(args)
    root = Path(arguments.root_dir).expanduser().resolve()
    pairs = json.loads(resolve_path(arguments.pairs_file, root).read_text(encoding="utf-8"))
    selected = pairs if arguments.all_pairs else [pair for pair in pairs if pair["id"] == arguments.pair_id]
    if not selected:
        raise ValueError(f"Pair {arguments.pair_id} is not present in {arguments.pairs_file}")
    for pair in selected:
        run_pair(pair, root, arguments)


if __name__ == "__main__":
    main()