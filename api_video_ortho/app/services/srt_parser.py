import re
import csv
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
from bisect import bisect_left

logger = logging.getLogger(__name__)

@dataclass
class TelemetryRecord:
    index: int
    start_sec: float
    end_sec: float
    mid_sec: float
    frame_cnt: Optional[int] = None
    diff_time_ms: Optional[int] = None
    timestamp_str: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None
    iso: Optional[int] = None
    shutter: Optional[str] = None
    fnum: Optional[float] = None
    focal_len: Optional[float] = None
    ev: Optional[float] = None
    ct: Optional[int] = None
    color_md: Optional[str] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    raw_text: Optional[str] = None

class SRTTelemetryParser:
    """
    Parser for drone subtitle (.srt) and telemetry files (.csv).
    Handles standard DJI drone telemetry subtitles and custom subtitle formats.
    """
    
    # Regex patterns
    SRT_TIMESTAMP_RE = re.compile(
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
    )
    FRAME_CNT_RE = re.compile(r"(?:FrameCnt|Frame|frame)\s*[:=]\s*(\d+)", re.IGNORECASE)
    DIFF_TIME_RE = re.compile(r"DiffTime\s*[:=]\s*(\d+)\s*ms", re.IGNORECASE)
    DATETIME_RE = re.compile(r"(\d{4}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}:\d{2}(?:[,\.]\d+)?)")
    
    # Tag patterns e.g. [key : value] or key: value
    TAG_RE = re.compile(r"\[\s*([\w_]+)\s*[:=]\s*([^\]]+)\s*\]")
    
    def __init__(self):
        self.records: List[TelemetryRecord] = []
        self._start_times: List[float] = []

    def parse_file(self, file_path: str) -> List[TelemetryRecord]:
        """Parse an SRT or CSV file and return a list of TelemetryRecords."""
        file_path_lower = file_path.lower()
        if file_path_lower.endswith(".csv"):
            return self.parse_csv(file_path)
        else:
            return self.parse_srt(file_path)

    def parse_srt(self, file_path: str) -> List[TelemetryRecord]:
        """Parse standard SRT file containing subtitle telemetry."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return self.parse_srt_text(content)

    def parse_srt_text(self, srt_text: str) -> List[TelemetryRecord]:
        """Parse SRT text blocks."""
        self.records = []
        # Split by double newline or block patterns
        blocks = re.split(r"\n\s*\n", srt_text.strip())
        
        for block_idx, block in enumerate(blocks):
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue
            
            # Find timestamp line
            time_match = None
            time_line_idx = -1
            for i, line in enumerate(lines):
                m = self.SRT_TIMESTAMP_RE.search(line)
                if m:
                    time_match = m
                    time_line_idx = i
                    break
            
            if not time_match:
                continue
            
            # Calculate start and end seconds
            sh, sm, ss, sms = map(int, time_match.group(1, 2, 3, 4))
            eh, em, es, ems = map(int, time_match.group(5, 6, 7, 8))
            start_sec = sh * 3600 + sm * 60 + ss + sms / 1000.0
            end_sec = eh * 3600 + em * 60 + es + ems / 1000.0
            mid_sec = (start_sec + end_sec) / 2.0
            
            # Combine remaining lines as payload
            payload_lines = lines[time_line_idx + 1:]
            raw_text = " ".join(payload_lines)
            
            # Clean HTML tags like <font size="...">
            clean_text = re.sub(r"<[^>]+>", " ", raw_text)
            
            record = self._parse_telemetry_text(
                index=block_idx + 1,
                start_sec=start_sec,
                end_sec=end_sec,
                mid_sec=mid_sec,
                text=clean_text
            )
            self.records.append(record)
            
        self._start_times = [r.start_sec for r in self.records]
        logger.info(f"Parsed {len(self.records)} telemetry records from SRT text.")
        return self.records

    def parse_csv(self, file_path: str) -> List[TelemetryRecord]:
        """Parse CSV telemetry file."""
        self.records = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                start_sec = float(row.get("Start", idx))
                end_sec = float(row.get("End", start_sec + 0.04))
                text = row.get("Text", "")
                
                # Check if columns are directly in CSV
                lat = float(row["latitude"]) if "latitude" in row else None
                lon = float(row["longtitude"]) if "longtitude" in row else (float(row["longitude"]) if "longitude" in row else None)
                alt = float(row["altitude"]) if "altitude" in row else None
                
                if lat is not None and lon is not None:
                    record = TelemetryRecord(
                        index=idx + 1,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        mid_sec=(start_sec + end_sec) / 2.0,
                        latitude=lat,
                        longitude=lon,
                        altitude=alt,
                        raw_text=text
                    )
                else:
                    record = self._parse_telemetry_text(
                        index=idx + 1,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        mid_sec=(start_sec + end_sec) / 2.0,
                        text=text
                    )
                self.records.append(record)
                
        self._start_times = [r.start_sec for r in self.records]
        return self.records

    def _parse_telemetry_text(self, index: int, start_sec: float, end_sec: float, mid_sec: float, text: str) -> TelemetryRecord:
        """Extract individual telemetry variables from text."""
        record = TelemetryRecord(
            index=index,
            start_sec=start_sec,
            end_sec=end_sec,
            mid_sec=mid_sec,
            raw_text=text
        )
        
        # FrameCnt
        fc_m = self.FRAME_CNT_RE.search(text)
        if fc_m:
            record.frame_cnt = int(fc_m.group(1))
            
        # DiffTime
        dt_m = self.DIFF_TIME_RE.search(text)
        if dt_m:
            record.diff_time_ms = int(dt_m.group(1))
            
        # DateTime string
        dt_str_m = self.DATETIME_RE.search(text)
        if dt_str_m:
            record.timestamp_str = dt_str_m.group(1).replace(",", ".")

        # Extract bracketed tags [key : value]
        tags = {}
        for k, v in self.TAG_RE.findall(text):
            tags[k.strip().lower()] = v.strip()
            
        # Also handle key: value without brackets if needed
        # Latitude
        if "latitude" in tags:
            try: record.latitude = float(tags["latitude"])
            except ValueError: pass
        elif "lat" in tags:
            try: record.latitude = float(tags["lat"])
            except ValueError: pass
        else:
            m = re.search(r"(?:latitude|lat)\s*[:=]\s*([-+]?\d+\.?\d*)", text, re.I)
            if m: record.latitude = float(m.group(1))

        # Longitude (handling DJI typo 'longtitude')
        if "longtitude" in tags:
            try: record.longitude = float(tags["longtitude"])
            except ValueError: pass
        elif "longitude" in tags:
            try: record.longitude = float(tags["longitude"])
            except ValueError: pass
        elif "lon" in tags or "lng" in tags:
            try: record.longitude = float(tags.get("lon") or tags.get("lng"))
            except ValueError: pass
        else:
            m = re.search(r"(?:longtitude|longitude|lon|lng)\s*[:=]\s*([-+]?\d+\.?\d*)", text, re.I)
            if m: record.longitude = float(m.group(1))

        # Altitude
        if "altitude" in tags:
            try: record.altitude = float(tags["altitude"])
            except ValueError: pass
        elif "alt" in tags:
            try: record.altitude = float(tags["alt"])
            except ValueError: pass
        elif "rel_alt" in tags:
            try: record.altitude = float(tags["rel_alt"])
            except ValueError: pass
        else:
            m = re.search(r"(?:altitude|alt|rel_alt)\s*[:=]\s*([-+]?\d+\.?\d*)", text, re.I)
            if m: record.altitude = float(m.group(1))

        # ISO
        if "iso" in tags:
            try: record.iso = int(tags["iso"])
            except ValueError: pass
        else:
            m = re.search(r"\biso\s*[:=]\s*(\d+)", text, re.I)
            if m: record.iso = int(m.group(1))

        # Shutter
        if "shutter" in tags:
            record.shutter = tags["shutter"]
        else:
            m = re.search(r"\bshutter\s*[:=]\s*([0-9/.]+)", text, re.I)
            if m: record.shutter = m.group(1)

        # F-Number / Aperture
        if "fnum" in tags:
            try:
                v = float(tags["fnum"])
                record.fnum = v / 100.0 if v >= 100 else v
            except ValueError: pass
        else:
            m = re.search(r"\bfnum\s*[:=]\s*(\d+\.?\d*)", text, re.I)
            if m:
                v = float(m.group(1))
                record.fnum = v / 100.0 if v >= 100 else v

        # Focal Length
        if "focal_len" in tags:
            try:
                v = float(tags["focal_len"])
                record.focal_len = v / 10.0 if v >= 100 else v
            except ValueError: pass
        else:
            m = re.search(r"\bfocal_len\s*[:=]\s*(\d+\.?\d*)", text, re.I)
            if m:
                v = float(m.group(1))
                record.focal_len = v / 10.0 if v >= 100 else v

        # EV
        if "ev" in tags:
            try: record.ev = float(tags["ev"])
            except ValueError: pass
        else:
            m = re.search(r"\bev\s*[:=]\s*([-+]?\d+\.?\d*)", text, re.I)
            if m: record.ev = float(m.group(1))

        # CT
        if "ct" in tags:
            try: record.ct = int(tags["ct"])
            except ValueError: pass
        else:
            m = re.search(r"\bct\s*[:=]\s*(\d+)", text, re.I)
            if m: record.ct = int(m.group(1))

        # Color Mode
        if "color_md" in tags:
            record.color_md = tags["color_md"]
        else:
            m = re.search(r"\bcolor_md\s*[:=]\s*(\w+)", text, re.I)
            if m: record.color_md = m.group(1)

        # Angles: Yaw, Pitch, Roll
        yaw_tag = tags.get("gb_yaw") or tags.get("yaw")
        if yaw_tag:
            try: record.yaw = float(yaw_tag)
            except ValueError: pass
        else:
            m = re.search(r"(?:gb_yaw|yaw)\s*[:=]\s*([-+]?\d+\.?\d*)", text, re.I)
            if m: record.yaw = float(m.group(1))

        pitch_tag = tags.get("gb_pitch") or tags.get("pitch")
        if pitch_tag:
            try: record.pitch = float(pitch_tag)
            except ValueError: pass
        else:
            m = re.search(r"(?:gb_pitch|pitch)\s*[:=]\s*([-+]?\d+\.?\d*)", text, re.I)
            if m: record.pitch = float(m.group(1))

        roll_tag = tags.get("gb_roll") or tags.get("roll")
        if roll_tag:
            try: record.roll = float(roll_tag)
            except ValueError: pass
        else:
            m = re.search(r"(?:gb_roll|roll)\s*[:=]\s*([-+]?\d+\.?\d*)", text, re.I)
            if m: record.roll = float(m.group(1))

        return record

    def get_telemetry_for_timestamp(self, timestamp_sec: float) -> Optional[TelemetryRecord]:
        """Find the closest telemetry record for a given timestamp in seconds."""
        if not self.records:
            return None
        
        # Binary search for closest start time
        idx = bisect_left(self._start_times, timestamp_sec)
        
        if idx == 0:
            return self.records[0]
        if idx >= len(self.records):
            return self.records[-1]
            
        prev_record = self.records[idx - 1]
        next_record = self.records[idx]
        
        # Check if timestamp falls inside previous interval [start, end]
        if prev_record.start_sec <= timestamp_sec <= prev_record.end_sec:
            return prev_record
        if next_record.start_sec <= timestamp_sec <= next_record.end_sec:
            return next_record
            
        # Otherwise pick nearest mid_sec
        diff_prev = abs(prev_record.mid_sec - timestamp_sec)
        diff_next = abs(next_record.mid_sec - timestamp_sec)
        return prev_record if diff_prev <= diff_next else next_record

    def get_telemetry_for_frame(self, frame_number: int) -> Optional[TelemetryRecord]:
        """Get record by frame number (1-based index)."""
        if 0 < frame_number <= len(self.records):
            # First check direct index match
            candidate = self.records[frame_number - 1]
            if candidate.frame_cnt == frame_number:
                return candidate
        
        # Fallback search across all records
        for r in self.records:
            if r.frame_cnt == frame_number:
                return r
                
        # If frame_cnt wasn't in subtitle, fallback to index
        if 0 < frame_number <= len(self.records):
            return self.records[frame_number - 1]
            
        return None
