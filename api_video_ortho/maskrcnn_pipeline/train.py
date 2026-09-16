"""Train Mask R-CNN on tiled orthophotos.

Run from the repository root with ``python -m maskrcnn_pipeline.train``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import BuildingTileDataset, build_records, collate_fn, tile_statistics
from .metrics import average_precision_50_95, average_precision_at_iou, summarize
from .model import build_model, load_checkpoint
from .utils import save_qa


def train_one_epoch(model, loader, optimizer, device, scaler, amp_enabled):
    model.train()
    total = 0.0
    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled):
            losses = model(images, targets)
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {losses}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach())
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, score_threshold=0.5):
    model.eval()
    records = []
    for images, targets in loader:
        outputs = model([image.to(device) for image in images])
        for output, target in zip(outputs, targets):
            keep = output["scores"].detach().cpu() >= score_threshold
            prediction = {
                "masks": output["masks"].detach().cpu().numpy()[:, 0][keep.numpy()],
                "scores": output["scores"].detach().cpu().numpy()[keep.numpy()],
            }
            truth = {"masks": target["masks"].numpy()}
            records.append((prediction, truth))
    metrics = summarize(records, 0.5)
    metrics["AP50"] = average_precision_at_iou(records, 0.5)
    metrics["AP75"] = average_precision_at_iou(records, 0.75)
    metrics["AP50_95"] = average_precision_50_95(records)
    return metrics


def write_qa_images(dataset, output_dir, count=6):
    rng = random.Random(17)
    indices = rng.sample(range(len(dataset)), min(count, len(dataset)))
    for number, index in enumerate(indices, 1):
        image, target = dataset[index]
        image_array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        save_qa(image_array, target["masks"].numpy(), target["boxes"].numpy(), output_dir / f"tile_{number:02d}.png")


def make_parser():
    parser = argparse.ArgumentParser(description="Train a tiled Mask R-CNN building instance segmenter")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--orthophoto-dir", default=None)
    parser.add_argument("--annotation-dir", default=None)
    parser.add_argument("--output-dir", default="maskrcnn_outputs")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--negative-ratio", type=float, default=1.0, help="Maximum sampled empty tiles per positive tile")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--sanity-only", action="store_true")
    return parser


def resolve_path(value, root):
    path = Path(value)
    return path if path.is_absolute() else root / path


def main(args=None):
    args = make_parser().parse_args(args)
    root = Path(args.root_dir).expanduser().resolve()
    orthophoto_dir = resolve_path(args.orthophoto_dir or "Dataset/Orthophotos", root)
    annotation_dir = resolve_path(args.annotation_dir or "Dataset/Annotations_final", root)
    output_dir = resolve_path(args.output_dir, root)
    checkpoint_dir = resolve_path(args.checkpoint_dir, root) if args.checkpoint_dir else output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True); checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_records = build_records(orthophoto_dir, annotation_dir, args.tile_size, args.stride, "train")
    val_records = build_records(orthophoto_dir, annotation_dir, args.tile_size, args.stride, "val")
    test_records = build_records(orthophoto_dir, annotation_dir, args.tile_size, args.stride, "test")
    train_set = BuildingTileDataset(train_records, augment=True)
    val_set = BuildingTileDataset(val_records, augment=False)
    test_set = BuildingTileDataset(test_records, augment=False)
    print("train:", tile_statistics(train_records, train_set))
    print("validation:", tile_statistics(val_records, val_set))
    print("test:", tile_statistics(test_records, test_set))
    write_qa_images(val_set, output_dir / "qa")
    if args.sanity_only:
        return
    train_counts = [len(train_set[index][1]["labels"]) for index in range(len(train_set))]
    positive = max(sum(value > 0 for value in train_counts), 1)
    empty_weight = positive * args.negative_ratio / max(sum(value == 0 for value in train_counts), 1)
    weights = [1.0 if value > 0 else empty_weight for value in train_counts]
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and not args.no_amp
    model = build_model(pretrained=not args.no_pretrained and not args.resume).to(device)
    optimizer = torch.optim.SGD([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scaler = GradScaler(enabled=amp_enabled)
    start_epoch = 0; best_ap = -1.0
    if args.resume:
        state = load_checkpoint(model, args.resume, device, optimizer, scheduler)
        start_epoch = int(state.get("epoch", -1)) + 1; best_ap = float(state.get("validation_metric", -1.0))
    history = []
    for epoch in range(start_epoch, args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device, scaler, amp_enabled)
        metrics = evaluate(model, val_loader, device, args.score_threshold)
        scheduler.step()
        record = {"epoch": epoch, "loss": loss, **metrics}
        history.append(record)
        print(json.dumps(record))
        state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "validation_metric": metrics["AP50"], "args": vars(args)}
        torch.save(state, checkpoint_dir / "latest.pt")
        if metrics["AP50"] > best_ap:
            best_ap = metrics["AP50"]; torch.save(state, checkpoint_dir / "best.pt")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()