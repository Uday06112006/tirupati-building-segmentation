import os
import json
import pytest
from app.models.schemas import FrameMetadata
from app.services.geojson_builder import GeoJSONBuilder

def test_geojson_builder(tmp_path):
    frames = [
        FrameMetadata(
            frame_index=1,
            frame_id=1,
            timestamp_str="2024-11-03 11:35:42",
            timestamp_sec=0.0,
            latitude=13.398962,
            longitude=79.563553,
            altitude=214.707,
            iso=100,
            shutter="1/640.0",
            fnum=2.8,
            focal_len=28.0,
            filename="frame_000001.jpg"
        ),
        FrameMetadata(
            frame_index=2,
            frame_id=2,
            timestamp_str="2024-11-03 11:35:43",
            timestamp_sec=1.0,
            latitude=13.398970,
            longitude=79.563560,
            altitude=214.650,
            iso=100,
            shutter="1/640.0",
            fnum=2.8,
            focal_len=28.0,
            filename="frame_000002.jpg"
        )
    ]
    
    geojson = GeoJSONBuilder.build_geojson(frames, include_line_string=True)
    
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 3  # 2 points + 1 LineString
    
    # Check Point feature
    p1 = geojson["features"][0]
    assert p1["type"] == "Feature"
    assert p1["geometry"]["type"] == "Point"
    assert p1["geometry"]["coordinates"] == [79.563553, 13.398962, 214.707]
    assert p1["properties"]["filename"] == "frame_000001.jpg"
    
    # Check LineString trajectory
    traj = geojson["features"][2]
    assert traj["geometry"]["type"] == "LineString"
    assert len(traj["geometry"]["coordinates"]) == 2
    assert traj["properties"]["name"] == "Drone Flight Trajectory"
    
    # Test saving
    out_file = str(tmp_path / "test.geojson")
    GeoJSONBuilder.save_geojson(geojson, out_file)
    assert os.path.exists(out_file)
    
    with open(out_file, "r") as f:
        loaded = json.load(f)
    assert loaded["type"] == "FeatureCollection"
