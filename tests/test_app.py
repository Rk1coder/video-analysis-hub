import json
from types import SimpleNamespace

from PIL import Image

import app


def test_modern_gradio_string_path_returns_useful_unknown_model_error(tmp_path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (32, 24), "white").save(path)
    result = app.run_analysis(str(path), "unknown", "")
    assert "error" in json.loads(result[0])
    assert result[1] is None


def test_missing_upload_returns_structured_error_and_clears_download():
    result = app.run_analysis(None, "yolos", "")
    assert "error" in json.loads(result[0])
    assert result[1:] == (None, None)


def test_successful_string_filepath_has_matching_preview_and_download(tmp_path, monkeypatch):
    path = tmp_path / "sample.png"
    Image.new("RGB", (32, 24), "white").save(path)
    expected = {"model": "fixture", "counts": {"cat": 2}}
    preview = Image.new("RGB", (32, 24), "green")

    def analyze(received, **kwargs):
        assert received == str(path)
        return SimpleNamespace(result=expected, preview=preview)

    monkeypatch.setattr(app, "analyze_media", analyze)
    text, rendered, download = app.run_analysis(str(path), "yolos", "cat")
    assert json.loads(text) == expected
    assert rendered is preview
    from pathlib import Path

    assert json.loads(Path(download).read_text()) == expected
