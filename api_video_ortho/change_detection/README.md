# Before/after building change detection

This module is separate from `maskrcnn_pipeline/`. It loads the trained checkpoint and never retrains or changes it. The authoritative survey pairing is in [pairs.json](pairs.json), based on the supplied F1-F7 correspondence.

The current TIFFs are read from `outputs/`, not moved or renamed. The repaired annotation folders are not required for inference because the trained Mask R-CNN supplies detections.

## F1 quality-control run

From the repository root on the GPU machine:

```powershell
python -m change_detection.pipeline `
  --root-dir "F:\Model Traning\tirupati-building-segmentation" `
  --pair-id F1 `
  --checkpoint "maskrcnn_outputs\checkpoints\best.pt" `
  --output-dir "maskrcnn_outputs\change_detection" `
  --score-threshold 0.5
```

The command processes only F1 unless `--all-pairs` is explicitly supplied. Inspect `maskrcnn_outputs/change_detection/F1/alignment.png`, `change_detection.png`, and `matched_buildings.png` before processing the remaining surveys.

## Registration

The source F1 TIFFs have different dimensions and no CRS or affine geotransform. The pipeline therefore registers the AFTER preview to the BEFORE image using SIFT (or ORB when SIFT is unavailable), RANSAC homography/affine estimation, and a phase-correlation translation fallback. It writes the transform and inlier quality to `registration.json`. A poor registration stops the pair by default after saving `alignment.png`; `--allow-poor-registration` is an explicit override and should only be used after visual inspection.

All coordinates remain image pixels. No CRS, GPS coordinates, or geographic GeoJSON is invented.

## Detection and matching

Inference uses 512-pixel tiles with stride 384, configurable score and mask thresholds, and `torch.inference_mode()`. Tile predictions are merged with mask and box overlap tests. Masks are stored as crop-local arrays plus full-image boxes, avoiding one full-resolution boolean array per building. AFTER detections are warped into BEFORE pixel coordinates, then matched one-to-one using mask IoU, box IoU, and centroid distance.

The default classification thresholds are 30% absolute area change or mask IoU below 0.50 for `CHANGED_BUILDING`. Unmatched instances are `NEW_BUILDING` or `REMOVED_BUILDING`; small registration or mask differences should be reviewed through the QA images.

## Outputs per pair

Each pair directory contains `before_detections.json`, `after_detections.json`, `after_aligned_detections.json`, `matches.json`, `changes.json`, `registration.json`, `before_detections.png`, `after_detections.png`, `alignment.png`, `matched_buildings.png`, and `change_detection.png`. Per-instance crop masks are stored under `before/masks`, `after/masks`, and `after_aligned/masks`.

Legend: gray is unchanged, green is new, red is removed, and orange is changed.

After F1 is visually and logically verified, process all supplied pairs with:

```powershell
python -m change_detection.pipeline --root-dir "F:\Model Traning\tirupati-building-segmentation" --all-pairs --checkpoint "maskrcnn_outputs\checkpoints\best.pt"
```