import json
import math
import logging
from typing import List, Dict, Any, Optional
from ..models.schemas import FrameMetadata, GeoJSONFeatureCollection

logger = logging.getLogger(__name__)

class GeoJSONBuilder:
    """
    Service for generating RFC 7946 compliant GeoJSON from extracted frame telemetry.
    """
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, alt1: float, lat2: float, lon2: float, alt2: float) -> float:
        """Calculate 3D distance between two GPS coordinates in meters."""
        R = 6371000.0  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = math.sin(delta_phi / 2.0) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        surface_dist = R * c
        
        # Add altitude component
        delta_alt = alt2 - alt1
        return math.sqrt(surface_dist ** 2 + delta_alt ** 2)

    @classmethod
    def build_geojson(
        cls,
        frames: List[FrameMetadata],
        include_line_string: bool = True,
        job_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Construct full GeoJSON FeatureCollection dictionary."""
        features = []
        coordinates_3d = []
        total_distance = 0.0
        
        min_lat = float("inf")
        max_lat = float("-inf")
        min_lon = float("inf")
        max_lon = float("-inf")
        min_alt = float("inf")
        max_alt = float("-inf")

        for i, frame in enumerate(frames):
            lat = frame.latitude
            lon = frame.longitude
            alt = frame.altitude
            
            min_lat = min(min_lat, lat)
            max_lat = max(max_lat, lat)
            min_lon = min(min_lon, lon)
            max_lon = max(max_lon, lon)
            min_alt = min(min_alt, alt)
            max_alt = max(max_alt, alt)

            coord_pt = [lon, lat, round(alt, 3)]
            coordinates_3d.append(coord_pt)

            if i > 0:
                prev = frames[i - 1]
                total_distance += cls.haversine_distance(
                    prev.latitude, prev.longitude, prev.altitude,
                    lat, lon, alt
                )

            properties = {
                "frame_index": frame.frame_index,
                "frame_id": frame.frame_id,
                "filename": frame.filename,
                "timestamp": frame.timestamp_str,
                "timestamp_sec": frame.timestamp_sec,
                "latitude": lat,
                "longitude": lon,
                "altitude_m": round(alt, 3),
                "iso": frame.iso,
                "shutter": frame.shutter,
                "f_number": frame.fnum,
                "focal_length_mm": frame.focal_len,
                "ev": frame.ev,
                "ct": frame.ct,
                "color_mode": frame.color_md,
                "yaw": frame.yaw,
                "pitch": frame.pitch,
                "roll": frame.roll
            }
            
            # Remove None values
            properties = {k: v for k, v in properties.items() if v is not None}

            feature = {
                "type": "Feature",
                "id": frame.frame_index,
                "geometry": {
                    "type": "Point",
                    "coordinates": coord_pt
                },
                "properties": properties
            }
            features.append(feature)

        # Include flight path trajectory LineString
        if include_line_string and len(coordinates_3d) >= 2:
            trajectory_feature = {
                "type": "Feature",
                "id": "flight_trajectory",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates_3d
                },
                "properties": {
                    "name": "Drone Flight Trajectory",
                    "total_points": len(coordinates_3d),
                    "total_distance_meters": round(total_distance, 2),
                    "start_time": frames[0].timestamp_str if frames else None,
                    "end_time": frames[-1].timestamp_str if frames else None
                }
            }
            features.append(trajectory_feature)

        bbox = None
        if frames:
            bbox = [
                round(min_lon, 6), round(min_lat, 6), round(min_alt, 2),
                round(max_lon, 6), round(max_lat, 6), round(max_alt, 2)
            ]

        geojson_dict = {
            "type": "FeatureCollection",
            "bbox": bbox,
            "properties": {
                "total_frames": len(frames),
                "total_distance_meters": round(total_distance, 2),
                "bbox": bbox,
                "job_info": job_info or {}
            },
            "features": features
        }
        
        return geojson_dict

    @classmethod
    def save_geojson(cls, geojson_dict: Dict[str, Any], output_path: str) -> str:
        """Write GeoJSON to file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson_dict, f, indent=2)
        logger.info(f"Saved GeoJSON to {output_path}")
        return output_path
