import os
import json
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from PIL import Image
import piexif

logger = logging.getLogger(__name__)

@dataclass
class ImageGeoInfo:
    file_path: str
    filename: str
    width: int
    height: int
    latitude: float
    longitude: float
    altitude_m: float
    focal_length_mm: float
    sensor_width_mm: float = 6.4  # Default 1/2.3" or 1/2" sensor width (e.g. DJI Mavic/Mini)
    sensor_height_mm: float = 4.8
    yaw_deg: float = 0.0
    pitch_deg: float = -90.0  # Default nadir
    roll_deg: float = 0.0
    timestamp_str: Optional[str] = None
    extra_meta: Optional[Dict[str, Any]] = None

class EXIFReader:
    """
    Extracts WGS-84 GPS coordinates and camera parameters from image EXIF metadata.
    """

    @staticmethod
    def _dms_to_decimal(dms_tuple, ref_str: str) -> Optional[float]:
        """Convert EXIF rational DMS tuple to decimal degrees."""
        if not dms_tuple or len(dms_tuple) < 3:
            return None
        try:
            d = dms_tuple[0][0] / dms_tuple[0][1]
            m = dms_tuple[1][0] / dms_tuple[1][1]
            s = dms_tuple[2][0] / dms_tuple[2][1]
            dec = d + (m / 60.0) + (s / 3600.0)
            if ref_str and ref_str.upper() in ("S", "W"):
                dec = -dec
            return dec
        except Exception as e:
            logger.warning(f"Error converting DMS {dms_tuple}: {e}")
            return None

    @classmethod
    def read_image_geoinfo(cls, image_path: str) -> Optional[ImageGeoInfo]:
        """Parse geotagged image and return ImageGeoInfo."""
        if not os.path.exists(image_path):
            return None

        try:
            with Image.open(image_path) as img:
                width, height = img.size
                exif_bytes = img.info.get("exif")
                
            if not exif_bytes:
                return None
                
            exif_data = piexif.load(exif_bytes)
            gps = exif_data.get("GPS", {})
            exif_ifd = exif_data.get("Exif", {})
            
            # 1. Parse GPS Latitude and Longitude
            lat_dms = gps.get(piexif.GPSIFD.GPSLatitude)
            lat_ref = gps.get(piexif.GPSIFD.GPSLatitudeRef, b"N")
            if isinstance(lat_ref, bytes): lat_ref = lat_ref.decode("utf-8", errors="ignore")
            
            lon_dms = gps.get(piexif.GPSIFD.GPSLongitude)
            lon_ref = gps.get(piexif.GPSIFD.GPSLongitudeRef, b"E")
            if isinstance(lon_ref, bytes): lon_ref = lon_ref.decode("utf-8", errors="ignore")

            latitude = cls._dms_to_decimal(lat_dms, lat_ref)
            longitude = cls._dms_to_decimal(lon_dms, lon_ref)

            if latitude is None or longitude is None:
                return None

            # 2. Parse Altitude
            altitude_m = 100.0  # default
            alt_rational = gps.get(piexif.GPSIFD.GPSAltitude)
            if alt_rational and alt_rational[1] != 0:
                altitude_m = float(alt_rational[0]) / float(alt_rational[1])

            # 3. Parse Focal Length
            focal_length_mm = 24.0
            fl_rational = exif_ifd.get(piexif.ExifIFD.FocalLength)
            if fl_rational and fl_rational[1] != 0:
                focal_length_mm = float(fl_rational[0]) / float(fl_rational[1])

            # 4. Parse Heading / Yaw
            yaw_deg = 0.0
            img_dir_rational = gps.get(piexif.GPSIFD.GPSImgDirection)
            if img_dir_rational and img_dir_rational[1] != 0:
                yaw_deg = float(img_dir_rational[0]) / float(img_dir_rational[1])

            # 5. Parse UserComment JSON if present (telemetry payload from our extractor!)
            user_comment_bytes = exif_ifd.get(piexif.ExifIFD.UserComment, b"")
            extra_meta = {}
            if user_comment_bytes.startswith(b"ASCII\x00\x00\x00"):
                try:
                    payload_str = user_comment_bytes[8:].decode("utf-8", errors="ignore")
                    extra_meta = json.loads(payload_str)
                    if "yaw" in extra_meta and extra_meta["yaw"] is not None:
                        yaw_deg = float(extra_meta["yaw"])
                except Exception:
                    pass

            dt_orig = exif_ifd.get(piexif.ExifIFD.DateTimeOriginal, b"")
            if isinstance(dt_orig, bytes):
                dt_orig = dt_orig.decode("utf-8", errors="ignore")

            return ImageGeoInfo(
                file_path=image_path,
                filename=os.path.basename(image_path),
                width=width,
                height=height,
                latitude=latitude,
                longitude=longitude,
                altitude_m=altitude_m,
                focal_length_mm=focal_length_mm,
                yaw_deg=yaw_deg,
                timestamp_str=dt_orig or None,
                extra_meta=extra_meta
            )
        except Exception as e:
            logger.warning(f"Failed to read EXIF from {image_path}: {e}")
            return None

    @classmethod
    def scan_directory(cls, folder_path: str, max_images: Optional[int] = None) -> List[ImageGeoInfo]:
        """Scan a directory for all geotagged images."""
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        files = sorted(
            [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_exts]
        )
        
        geo_list: List[ImageGeoInfo] = []
        for f in files:
            info = cls.read_image_geoinfo(f)
            if info:
                geo_list.append(info)
            if max_images and len(geo_list) >= max_images:
                break
                
        logger.info(f"Found {len(geo_list)} valid geotagged images in {folder_path}")
        return geo_list
