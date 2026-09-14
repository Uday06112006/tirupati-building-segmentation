# Separate Building Segmentation Pipeline

This folder is intentionally isolated from the existing orthophoto generator in
`tools/generate_orthophoto.py`, `tools/generate_orthophoto_test.py`,
`tools/extract_geotagged.py`, and `tools/derive_attitude.py`.

The original GeoJSON annotation files remain untouched in
`Dataset/Annotations/`. This folder creates binary masks and patches in a
separate training tree:

- `Dataset/images/` for orthophoto image copies
- `Dataset/masks/` for binary masks
- `Dataset/patches/images/` and `Dataset/patches/masks/` for tile-level samples

The segmentation training branch consumes orthophoto TIFFs and cleaned
building masks/patches only. It does not require the original drone videos.
All dataset paths can be supplied through the CLI (`--config`,
`--orthophoto-dir`, `--dataset-dir`, `--annotation-dir`, `--mask-dir`,
`--patch-image-dir`, `--patch-mask-dir`, `--checkpoint-dir`, `--log-dir`,
`--prediction-dir`) or the JSON config example in
`segmentation_pipeline/config.example.json`.

The mapping of orthophoto TIFFs to GeoJSON files is stored in the
preprocessing script and should be validated with `preprocess.py` before training.
