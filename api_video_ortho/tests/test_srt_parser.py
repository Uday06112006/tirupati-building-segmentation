import pytest
from app.services.srt_parser import SRTTelemetryParser

SAMPLE_SRT = """1
00:00:00,000 --> 00:00:00,041
<font size="36">FrameCnt : 1, DiffTime : 41ms
2024-11-03 11:35:42,416,841
[iso : 100] [shutter : 1/640.0] [fnum : 280] [ev : 0.7] [ct : 5378] [color_md : default] [focal_len : 280] [latitude : 13.398962] [longtitude : 79.563553] [altitude: 214.707001] </font>

2
00:00:00,041 --> 00:00:00,083
<font size="36">FrameCnt : 2, DiffTime : 42ms
2024-11-03 11:35:42,458,559
[iso : 100] [shutter : 1/640.0] [fnum : 280] [ev : 0.7] [ct : 5387] [color_md : default] [focal_len : 280] [latitude : 13.398962] [longtitude : 79.563553] [altitude: 214.707001] </font>

3
00:00:00,083 --> 00:00:00,125
<font size="36">FrameCnt : 3, DiffTime : 42ms
2024-11-03 11:35:42,500,280
[iso : 100] [shutter : 1/640.0] [fnum : 280] [ev : 0.7] [ct : 5387] [color_md : default] [focal_len : 280] [latitude : 13.398959] [longtitude : 79.563554] [altitude: 214.656998] </font>
"""

def test_parse_srt_text():
    parser = SRTTelemetryParser()
    records = parser.parse_srt_text(SAMPLE_SRT)
    
    assert len(records) == 3
    
    r1 = records[0]
    assert r1.frame_cnt == 1
    assert r1.start_sec == 0.0
    assert abs(r1.end_sec - 0.041) < 1e-4
    assert r1.iso == 100
    assert r1.shutter == "1/640.0"
    assert r1.fnum == 2.8
    assert r1.focal_len == 28.0
    assert r1.latitude == 13.398962
    assert r1.longitude == 79.563553
    assert r1.altitude == 214.707001
    assert r1.ct == 5378
    assert r1.ev == 0.7

def test_lookup_by_timestamp_and_frame():
    parser = SRTTelemetryParser()
    parser.parse_srt_text(SAMPLE_SRT)
    
    # Lookup exact frame
    r = parser.get_telemetry_for_frame(2)
    assert r is not None
    assert r.frame_cnt == 2
    
    # Lookup by timestamp
    r_time = parser.get_telemetry_for_timestamp(0.09)
    assert r_time is not None
    assert r_time.frame_cnt == 3
