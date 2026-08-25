import json
import logging
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from PIL import Image
import piexif
import piexif.helper

logger = logging.getLogger(__name__)

class EXIFWriter:
    """
    Service for embedding standard photogrammetric GPS, camera and telemetry EXIF metadata
    into extracted image frames.
    """
    
    @staticmethod
    def decimal_to_dms_rational(deg_float: float) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """Convert decimal degrees to EXIF DMS rational tuples ((d, 1), (m, 1), (s_num, s_den))."""
        deg_float = abs(deg_float)
        d = int(deg_float)
        m_float = (deg_float - d) * 60.0
        m = int(m_float)
        s_float = (m_float - m) * 60.0
        # 6 decimal places of precision for seconds gives sub-millimeter precision
        s_num = int(round(s_float * 1000000))
        s_den = 1000000
        return ((d, 1), (m, 1), (s_num, s_den))

    @staticmethod
    def parse_shutter_to_rational(shutter_str: Optional[str]) -> Optional[Tuple[int, int]]:
        """Convert shutter string (e.g., '1/640.0', '1/500') to rational tuple."""
        if not shutter_str:
            return None
        try:
            shutter_str = shutter_str.strip()
            if "/" in shutter_str:
                parts = shutter_str.split("/")
                num = float(parts[0])
                den = float(parts[1])
                return (int(round(num * 1000)), int(round(den * 1000)))
            else:
                val = float(shutter_str)
                if val > 0:
                    return (1, int(round(1.0 / val)))
        except Exception as e:
            logger.warning(f"Failed to parse shutter '{shutter_str}': {e}")
        return None

    @classmethod
    def create_exif_bytes(
        cls,
        latitude: Optional[float],
        longitude: Optional[float],
        altitude: Optional[float],
        timestamp_str: Optional[str] = None,
        iso: Optional[int] = None,
        shutter: Optional[str] = None,
        fnum: Optional[float] = None,
        focal_len: Optional[float] = None,
        yaw: Optional[float] = None,
        pitch: Optional[float] = None,
        roll: Optional[float] = None,
        frame_index: Optional[int] = None,
        extra_telemetry: Optional[Dict[str, Any]] = None,
        make: str = "DJI",
        model: str = "Drone Camera"
    ) -> bytes:
        """Construct full EXIF dictionary and return serialized EXIF bytes."""
        
        # Parse timestamp or default to now
        dt_obj = None
        if timestamp_str:
            for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
                try:
                    # Clean trailing parts if needed
                    clean_ts = timestamp_str.replace(",", ".")
                    dt_obj = datetime.strptime(clean_ts.split(".")[0], "%Y-%m-%d %H:%M:%S")
                    break
                except Exception:
                    pass
        if not dt_obj:
            dt_obj = datetime.utcnow()
            
        datetime_formatted = dt_obj.strftime("%Y:%m:%d %H:%M:%S")
        gps_date_str = dt_obj.strftime("%Y:%m:%d")
        gps_time_tuple = (
            (dt_obj.hour, 1),
            (dt_obj.minute, 1),
            (dt_obj.second, 1)
        )
        
        # 0th IFD
        zeroth_ifd = {
            piexif.ImageIFD.Make: make,
            piexif.ImageIFD.Model: model,
            piexif.ImageIFD.Software: "OrthoGenerator-Microservice/1.0",
            piexif.ImageIFD.DateTime: datetime_formatted,
            piexif.ImageIFD.XResolution: (72, 1),
            piexif.ImageIFD.YResolution: (72, 1),
            piexif.ImageIFD.ResolutionUnit: 2,
        }
        
        # Description
        desc_parts = []
        if frame_index is not None:
            desc_parts.append(f"Frame: {frame_index}")
        if latitude is not None and longitude is not None:
            desc_parts.append(f"GPS: ({latitude:.6f}, {longitude:.6f})")
        if altitude is not None:
            desc_parts.append(f"Alt: {altitude:.2f}m")
        if desc_parts:
            zeroth_ifd[piexif.ImageIFD.ImageDescription] = " | ".join(desc_parts)

        # Exif IFD
        exif_ifd = {
            piexif.ExifIFD.DateTimeOriginal: datetime_formatted,
            piexif.ExifIFD.DateTimeDigitized: datetime_formatted,
            piexif.ExifIFD.ExifVersion: b"0231",
        }
        
        if iso is not None:
            exif_ifd[piexif.ExifIFD.ISOSpeedRatings] = int(iso)
            
        if fnum is not None:
            exif_ifd[piexif.ExifIFD.FNumber] = (int(round(fnum * 100)), 100)
            
        shutter_rational = cls.parse_shutter_to_rational(shutter)
        if shutter_rational:
            exif_ifd[piexif.ExifIFD.ExposureTime] = shutter_rational
            
        if focal_len is not None:
            exif_ifd[piexif.ExifIFD.FocalLength] = (int(round(focal_len * 100)), 100)
            # 35mm equivalent focal length
            exif_ifd[piexif.ExifIFD.FocalLengthIn35mmFilm] = int(round(focal_len))

        # Full telemetry json payload in UserComment
        telemetry_dict = {
            "frame_index": frame_index,
            "timestamp": timestamp_str or datetime_formatted,
            "latitude": latitude,
            "longitude": longitude,
            "altitude_m": altitude,
            "iso": iso,
            "shutter": shutter,
            "fnum": fnum,
            "focal_len": focal_len,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
        }
        if extra_telemetry:
            telemetry_dict.update(extra_telemetry)
            
        user_comment_str = json.dumps(telemetry_dict, ensure_ascii=False)
        exif_ifd[piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(user_comment_str)

        # GPS IFD
        gps_ifd = {
            piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0),
            piexif.GPSIFD.GPSMapDatum: "WGS-84",
            piexif.GPSIFD.GPSDateStamp: gps_date_str,
            piexif.GPSIFD.GPSTimeStamp: gps_time_tuple,
        }
        
        if latitude is not None:
            lat_ref = "N" if latitude >= 0 else "S"
            gps_ifd[piexif.GPSIFD.GPSLatitudeRef] = lat_ref
            gps_ifd[piexif.GPSIFD.GPSLatitude] = cls.decimal_to_dms_rational(latitude)
            
        if longitude is not None:
            lon_ref = "E" if longitude >= 0 else "W"
            gps_ifd[piexif.GPSIFD.GPSLongitudeRef] = lon_ref
            gps_ifd[piexif.GPSIFD.GPSLongitude] = cls.decimal_to_dms_rational(longitude)
            
        if altitude is not None:
            alt_ref = 0 if altitude >= 0 else 1
            gps_ifd[piexif.GPSIFD.GPSAltitudeRef] = alt_ref
            gps_ifd[piexif.GPSIFD.GPSAltitude] = (int(round(abs(altitude) * 1000)), 1000)
            
        if yaw is not None:
            # GPS Image Direction (True North)
            gps_ifd[piexif.GPSIFD.GPSImgDirectionRef] = "T"
            # Normalize yaw to [0, 360)
            norm_yaw = yaw % 360.0
            gps_ifd[piexif.GPSIFD.GPSImgDirection] = (int(round(norm_yaw * 100)), 100)

        exif_dict = {
            "0th": zeroth_ifd,
            "Exif": exif_ifd,
            "GPS": gps_ifd,
            "1st": {},
            "thumbnail": None
        }
        
        return piexif.dump(exif_dict)

    @classmethod
    def save_image_with_exif(
        cls,
        image_pil: Image.Image,
        output_path: str,
        exif_bytes: bytes,
        quality: int = 100,
        xmp_bytes: Optional[bytes] = None
    ) -> str:
        """
        Save a PIL Image as high-quality JPEG with embedded EXIF metadata.

        ``xmp_bytes`` adds an APP1 XMP segment alongside EXIF. Gimbal pitch and roll
        have no standard EXIF representation, so photogrammetry engines read camera
        orientation from XMP -- see :mod:`app.services.xmp_writer`.
        """
        save_kwargs = dict(
            format="JPEG",
            quality=quality,
            subsampling=0,  # 4:4:4 chroma subsampling for best quality
            exif=exif_bytes,
        )
        if xmp_bytes:
            save_kwargs["xmp"] = xmp_bytes
        image_pil.save(output_path, **save_kwargs)
        return output_path
