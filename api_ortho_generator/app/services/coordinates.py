import math
import logging
from typing import Tuple, List, Optional
import pyproj
from .exif_reader import ImageGeoInfo
from .ground_projection import (
    intrinsics_from_35mm, rotation_world_to_camera, ground_offsets_for_corners,
)

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
        Nadir-equivalent ground footprint (width_m, height_m) and nominal GSD (m/px).

        Two things matter here and are easy to get wrong:

        * **Height must be above ground, not above sea level.** EXIF ``GPSAltitude`` is
          normally MSL; using it directly overstates the footprint by the terrain
          elevation (here 214 m MSL vs 55 m AGL, a 3.9x error).
        * **A 35 mm-equivalent focal length implies a 36 mm sensor.** Pairing DJI's
          35 mm-equivalent figure with a physical 6.4 mm sensor width gives a 13 degree
          field of view where the truth is 65 degrees.

        For an oblique frame this returns the nadir-equivalent extent; the real
        footprint is a trapezoid and is computed by
        :func:`ground_projection.ground_offsets_for_corners`.
        """
        height_m = info.agl_m if info.agl_m and info.agl_m > 0 else info.altitude_m
        alt = max(5.0, height_m)

        if info.focal_35mm and info.focal_35mm > 0:
            sw, sh = 36.0, 36.0 * (info.height / max(1, info.width))
            fl = max(1.0, info.focal_35mm)
        else:
            sw, sh = info.sensor_width_mm, info.sensor_height_mm
            fl = max(5.0, info.focal_length_mm)

        ground_w = (alt * sw) / fl
        ground_h = (alt * sh) / fl

        # Nominal GSD in meters per pixel
        gsd = ground_w / max(1, info.width)
        return ground_w, ground_h, gsd

    @classmethod
    def compute_projected_bounds(
        cls,
        images: List[ImageGeoInfo],
        target_crs: str,
        max_range_m: Optional[float] = None
    ) -> Tuple[float, float, float, float, str]:
        """
        Compute total bounding box (min_x, min_y, max_x, max_y) in target CRS.

        ``max_range_m`` caps how far from each camera the footprint is allowed to
        reach; without it a steeply oblique frame projects towards the horizon and the
        canvas balloons. Defaults to 4x the flying height.
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

            quad = None
            if img.pose_is_known and not img.is_nadir:
                # Oblique frame: the real footprint is a trapezoid, so project the
                # actual image corners onto the ground rather than assuming a rectangle.
                K = intrinsics_from_35mm(img.focal_35mm or img.focal_length_mm,
                                         img.width, img.height)
                R = rotation_world_to_camera(img.yaw_deg, img.pitch_deg, img.roll_deg)
                quad = ground_offsets_for_corners(
                    K, R, img.agl_m, img.width, img.height,
                    max_range_m=max_range_m if max_range_m else 4.0 * img.agl_m)

            if quad is not None:
                all_corners_x.extend((cx + quad[:, 0]).tolist())
                all_corners_y.extend((cy + quad[:, 1]).tolist())
                continue

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
