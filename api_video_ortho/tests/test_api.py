import io
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models.schemas import ExtractionRequest

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "opencv_version" in data
    assert "supported_video_formats" in data

def test_root_redirect():
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (307, 308, 302, 301)
    assert response.headers["location"] == "/docs"

def test_list_jobs_empty():
    response = client.get("/api/v1/jobs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_extraction_request_zero_sanitization():
    # Verify that passing 0 or 0.0 for optional bounds defaults to full video / no limit
    req = ExtractionRequest(
        frame_step=1,
        frame_interval_sec=0.0,
        end_time_sec=0.0,
        max_frames=0,
        fps=0.0,
        start_time_sec=0.0
    )
    assert req.frame_interval_sec is None
    assert req.end_time_sec is None
    assert req.max_frames is None
    assert req.fps is None
    assert req.start_time_sec == 0.0
    assert req.frame_step == 1

def test_upload_endpoint_zero_inputs(tmp_path):
    # Mock video and srt bytes
    video_content = b"fake-video-bytes"
    srt_content = b"1\n00:00:00,000 --> 00:00:01,000\nFrameCnt: 1\n[lat: 13.0] [lon: 79.0]\n"
    
    files = {
        "video": ("test.mov", io.BytesIO(video_content), "video/quicktime"),
        "srt": ("test.srt", io.BytesIO(srt_content), "text/plain")
    }
    data = {
        "frame_step": 1,
        "frame_interval_sec": 0.0,
        "end_time_sec": 0.0,
        "max_frames": 0,
        "fps": 0.0,
        "async_mode": True
    }
    
    response = client.post("/api/v1/jobs/extract-upload", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] in ("QUEUED", "PROCESSING", "COMPLETED", "FAILED")
    assert "job_id" in res_data
