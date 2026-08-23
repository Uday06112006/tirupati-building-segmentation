import os
import json
import pytest
import piexif
import numpy as np
from PIL import Image

from app.services.exif_writer import EXIFWriter

def test_decimal_to_dms_rational():
    lat = 13.398962
    dms = EXIFWriter.decimal_to_dms_rational(lat)
    
    # Deg = 13
    assert dms[0] == (13, 1)
    # Min = int((13.398962 - 13) * 60) = int(23.93772) = 23
    assert dms[1] == (23, 1)
    # Sec = (23.93772 - 23) * 60 = 56.2632
    assert dms[2][1] == 1000000
    sec_val = dms[2][0] / dms[2][1]
    assert abs(sec_val - 56.2632) < 0.01

def test_parse_shutter():
    assert EXIFWriter.parse_shutter_to_rational("1/640.0") == (1000, 640000)
    assert EXIFWriter.parse_shutter_to_rational("1/500") == (1000, 500000)

def test_create_and_read_exif(tmp_path):
    exif_bytes = EXIFWriter.create_exif_bytes(
        latitude=13.398962,
        longitude=79.563553,
        altitude=214.707,
        timestamp_str="2024-11-03 11:35:42.416841",
        iso=100,
        shutter="1/640.0",
        fnum=2.8,
        focal_len=28.0,
        yaw=145.5,
        pitch=-10.2,
        roll=0.5,
        frame_index=1
    )
    
    assert isinstance(exif_bytes, bytes)
    assert len(exif_bytes) > 0
    
    # Save a test image
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    img_path = str(tmp_path / "test_frame.jpg")
    EXIFWriter.save_image_with_exif(img, img_path, exif_bytes, quality=100)
    
    # Reload and verify EXIF
    loaded = piexif.load(img_path)
    assert piexif.GPSIFD.GPSLatitude in loaded["GPS"]
    assert loaded["GPS"][piexif.GPSIFD.GPSLatitudeRef] == b"N"
    assert loaded["GPS"][piexif.GPSIFD.GPSLongitudeRef] == b"E"
    assert loaded["Exif"][piexif.ExifIFD.ISOSpeedRatings] == 100
    
    # Verify UserComment JSON
    user_comment_bytes = loaded["Exif"][piexif.ExifIFD.UserComment]
    # Remove prefix b'ASCII\x00\x00\x00'
    json_str = user_comment_bytes[8:].decode("utf-8")
    data = json.loads(json_str)
    assert data["latitude"] == 13.398962
    assert data["frame_index"] == 1
    assert data["altitude_m"] == 214.707
