import io
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

import server


def png_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (32, 24), "white").save(stream, format="PNG")
    return stream.getvalue()


def test_unsupported_upload_is_rejected_before_inference():
    with TestClient(server.app) as client:
        response = client.post("/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_corrupt_image_is_rejected_before_inference():
    with TestClient(server.app) as client:
        response = client.post("/analyze", files={"file": ("bad.png", b"broken", "image/png")})
    assert response.status_code == 400


def test_analysis_download_matches_result_and_upload_is_removed(monkeypatch):
    seen_paths = []

    def analyze(path, **kwargs):
        seen_paths.append(Path(path))
        assert Path(path).read_bytes() == png_bytes()
        return SimpleNamespace(
            result={"model": "fixture", "frames_processed": 1, "detections": []},
            preview=None,
        )

    monkeypatch.setattr(server, "analyze_media", analyze, raising=False)
    with TestClient(server.app) as client:
        response = client.post("/analyze", files={"file": ("image.png", png_bytes(), "image/png")})
        assert response.status_code == 200
        payload = response.json()
        downloaded = client.get("/download/" + payload.pop("session_id"))
        assert downloaded.status_code == 200
        assert json.loads(downloaded.content) == payload
    assert seen_paths and not seen_paths[0].exists()


def test_inference_failure_removes_upload_and_returns_service_error(monkeypatch):
    seen_paths = []

    def unavailable(path, **kwargs):
        seen_paths.append(Path(path))
        raise RuntimeError("Model could not be loaded")

    monkeypatch.setattr(server, "analyze_media", unavailable, raising=False)
    with TestClient(server.app) as client:
        response = client.post("/analyze", files={"file": ("image.png", png_bytes(), "image/png")})
    assert response.status_code == 503
    assert seen_paths and not seen_paths[0].exists()


def test_invalid_frame_limit_is_rejected():
    with TestClient(server.app) as client:
        response = client.post(
            "/analyze",
            files={"file": ("image.png", png_bytes(), "image/png")},
            data={"max_frames": 0},
        )
    assert response.status_code == 422


def test_declared_oversized_body_is_rejected_at_transport(monkeypatch):
    monkeypatch.setattr(server, "MAX_REQUEST_BYTES", 100, raising=False)
    with TestClient(server.app) as client:
        response = client.post("/analyze", content=b"x" * 101)
    assert response.status_code == 413


def test_streamed_oversized_body_is_rejected_without_content_length(monkeypatch):
    monkeypatch.setattr(server, "MAX_REQUEST_BYTES", 100, raising=False)
    with TestClient(server.app) as client:
        response = client.post("/analyze", content=iter([b"x" * 80, b"y" * 80]))
    assert response.status_code == 413


def test_legacy_gemma_image_response_fields_remain_available(monkeypatch):
    monkeypatch.setenv("VLLM_ENDPOINT", "http://fixture.invalid")
    import analysis

    monkeypatch.setattr(
        analysis,
        "_ask_gemma",
        lambda image, prompt: '[{"box_2d":[1,2,10,20],"label":"cat","confidence":0.9}]',
    )
    with TestClient(server.app) as client:
        response = client.post(
            "/process-image-gemma", files={"image": ("image.png", png_bytes(), "image/png")}
        )
    assert response.status_code == 200
    result = response.json()
    assert result["detections"] == [
        {"bbox": {"y1": 1, "x1": 2, "y2": 10, "x2": 20}, "label": "cat", "confidence": 0.9}
    ]
    assert '"box_2d"' in result["raw_response"]


def test_concurrent_analysis_is_rejected_before_a_second_decode(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    entered, release = threading.Event(), threading.Event()

    def slow_analysis(path, **kwargs):
        entered.set()
        release.wait(5)
        return SimpleNamespace(result={"model": "fixture", "detections": []}, preview=None)

    monkeypatch.setattr(server, "analyze_media", slow_analysis)
    with TestClient(server.app) as client, ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            client.post, "/analyze", files={"file": ("image.png", png_bytes(), "image/png")}
        )
        assert entered.wait(3)
        try:
            response = client.post(
                "/analyze", files={"file": ("image.png", png_bytes(), "image/png")}
            )
            assert response.status_code == 429
        finally:
            release.set()
            assert future.result(timeout=5).status_code == 200
