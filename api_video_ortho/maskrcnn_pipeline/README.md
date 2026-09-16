# Mask R-CNN building instance segmentation

This is a separate pipeline from `segmentation_pipeline/`; it does not modify the existing U-Net workflow.

## GPU installation

Use a fresh environment on the Windows GPU machine. Install a PyTorch/torchvision pair matching the machine's CUDA wheel, then install:

```powershell
pip install torch torchvision
pip install numpy pillow shapely rasterio
# Optional COCO evaluator:
pip install pycocotools
```

The code does not invent a CRS. Coordinates and boxes in outputs are pixel coordinates because the source TIFFs are not georeferenced.

## Data and instances

The fixed split is train `DJI_0510 DJI_0513 DJI_0517 DJI_0520 DJI_0522 DJI_0523 DJI_0527 DJI_0529`, validation `DJI_0530 DJI_0531 DJI_0532`, and test `DJI_0533 DJI_0534 DJI_0535`. The GeoJSON-to-image convention follows the existing preprocessing: `(x, y)` becomes `(x, -y)`.

Each Polygon is one instance. A MultiPolygon is expanded into one instance for every disconnected Polygon component; touching or overlapping components are grouped back into one connected footprint. Invalid shapes are repaired in memory only. Each instance is intersected with each tile, rasterized independently, and given its own mask, box, label, area, and `iscrowd=0`. Fragments with fewer than four raster pixels are ignored.

Tiles default to 512 pixels with stride 384. Edge tiles are included. Empty tiles remain available, but the training `WeightedRandomSampler` caps them at `--negative-ratio` per positive tile.

## Training

From the directory containing `maskrcnn_pipeline/`:

```powershell
python -m maskrcnn_pipeline.train --root-dir "F:\Model Traning\tirupati-building-segmentation" --orthophoto-dir "F:\Model Traning\tirupati-building-segmentation\Dataset\Orthophotos" --annotation-dir "F:\Model Traning\tirupati-building-segmentation\Dataset\Annotations_final" --output-dir "maskrcnn_outputs" --epochs 1 --batch-size 2
```

Use `--sanity-only` to build statistics and QA tiles without creating a model. Use `--no-amp` for AMP debugging and `--resume maskrcnn_outputs/checkpoints/latest.pt` to continue. The output includes `qa/`, `history.json`, `checkpoints/latest.pt`, and `checkpoints/best.pt`.

## Inference

```powershell
python -m maskrcnn_pipeline.inference --root-dir "F:\Model Traning\tirupati-building-segmentation" --orthophoto-dir "F:\Model Traning\tirupati-building-segmentation\Dataset\Orthophotos" --checkpoint "maskrcnn_outputs\checkpoints\best.pt" --output-dir "maskrcnn_outputs\inference" --score-threshold 0.5
```

Inference processes tiles sequentially, maps masks and boxes back to full-image pixel coordinates, and suppresses duplicate detections from overlapping tiles using mask IoU or box IoU. Each test image gets `original.png`, `building_mask.png`, `prediction_visualization.png`, `instances.json`, and one PNG per instance under `masks/`.

Validation reports greedy matched-instance precision, recall, Dice, matched mask IoU, AP50, AP75, and the mean AP across IoU thresholds 0.50 through 0.95. This is a scored detection approximation rather than the full COCO evaluator; install and add `pycocotools` if exact COCO protocol reporting is required. No model quality is implied before training and evaluation.