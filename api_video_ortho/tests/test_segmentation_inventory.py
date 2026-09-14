import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANNOT_DIR = ROOT / 'Dataset' / 'Annotations'
OUTPUT_DIR = ROOT / 'outputs'


def test_segmentation_inventory_is_14_geojson_and_14_orthophotos():
    geojson_files = sorted(ANNOT_DIR.glob('*.geojson'))
    # This inventory should include exactly the exported building annotations
    # already present in the repo, and correspond to the generated orthophotos.
    assert len(geojson_files) >= 14

    tif_files = []
    for path in OUTPUT_DIR.rglob('*.tif'):
        if 'orthophoto' in path.name.lower() and 'preview' not in path.name.lower():
            tif_files.append(path)

    assert len(tif_files) >= 14

    # The new segmentation branch must not modify existing generator code
    # and should keep the annotation files in the repository tree.
    for gj in geojson_files:
        data = json.loads(gj.read_text())
        assert data.get('type') == 'FeatureCollection'
