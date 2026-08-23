import math
import logging
from typing import Tuple, List
import pyproj
from .exif_reader import ImageGeoInfo

logger = logging.getLogger(__name__)

class CoordinateService:
    """
    Handles coordinate reference systems, UTM zone auto-detection,
    and spatial ground footprint projection calculations.
    """

    @staticmethod
    def get_utm_epsg_for_lon_lat(lon: float, lat: float) -> str:
        """Calculate EPSG code for the appropriate UTM Zone."""
        zone = int(math.floor((lon + 180.0) / 6.0)) + 1
        if lat >= 0:
            return f"EPSG:{32600 + zone}"  # Northern Hemisphere
        else:
            return f"EPSG:{32700 + zone}"  # Southern Hemisphere

    @classmethod
    def get_transformer(cls, source_crs: str, target_crs: str) -> pyproj.Transformer:
        """Get PyProj transformer."""
        return pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)

    @classmethod
    def calculate_ground_footprint(cls, info: ImageGeoInfo) -> Tuple[float, float, float]:
        """
        Calculate ground footprint dimensions (width_m, height_m) and nominal GSD (meters/pixel).
        """
        # Nominal GSD formula
        # altitude in meters, sensor dimensions in mm, focal length in mm
        alt = max(5.0, info.altitude_m)
        fl = max(5.0, info.focal_length_mm)
        sw = info.sensor_width_mm
        sh = info.sensor_height_mm
        
        ground_w = (alt * sw) / fl
        ground_h = (alt * sh) / fl
        
        # Nominal GSD in meters per pixel
        gsd = ground_w / max(1, info.width)
        return ground_w, ground_h, gsd

    @classmethod
    def compute_projected_bounds(
        cls,
        images: List[ImageGeoInfo],
        target_crs: str
    ) -> Tuple[float, float, float, float, str]:
        """
        Compute total bounding box (min_x, min_y, max_x, max_y) in target CRS.
        """
        if not images:
            raise ValueError("No images provided to compute bounds.")

        # Determine CRS if AUTO
        if target_crs.upper() in ("AUTO", "AUTO_UTM"):
            median_lon = sum(img.longitude for img in images) / len(images)
            median_lat = sum(img.latitude for img in images) / len(images)
            target_crs = cls.get_utm_epsg_for_lon_lat(median_lon, median_lat)

        transformer = cls.get_transformer("EPSG:4326", target_crs)

        all_corners_x = []
        all_corners_y = []

        for img in images:
            cx, cy = transformer.transform(img.longitude, img.latitude)
            gw, gh, _ = cls.calculate_ground_footprint(img)
            
            # Account for rotation (yaw heading)
            yaw_rad = math.radians(img.yaw_deg)
            cos_y = abs(math.cos(yaw_rad))
            sin_y = abs(math.sin(yaw_rad))
            
            half_w = (gw * cos_y + gh * sin_y) / 2.0
            half_h = (gw * sin_y + gh * cos_y) / 2.0
            
            all_corners_x.extend([cx - half_w, cx + half_w])
            all_corners_y.extend([cy - half_h, cy + half_h])

        min_x = min(all_corners_x)
        max_x = max(all_corners_x)
        min_y = min(all_corners_y)
        max_y = max(all_corners_y)

        logger.info(f"Target CRS: {target_crs} | Spatial Extent: X=[{min_x:.2f}, {max_x:.2f}], Y=[{min_y:.2f}, {max_y:.2f}]")
        return min_x, min_y, max_x, max_y, target_crs
