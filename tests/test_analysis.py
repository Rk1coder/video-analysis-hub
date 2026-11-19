import importlib.util
import json

import cv2
import numpy as np
import pytest
from PIL import Image


def runtime():
    if importlib.util.find_spec("analysis") is None:
        pytest.fail("The shared image/video analysis runtime is missing")
    import analysis

    return analysis


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for n in range(10):
        writer.write(np.full((48, 64, 3), n * 20, dtype=np.uint8))
    writer.release()
    return path


def test_video_sampling_is_bounded_and_covers_end_of_video(video):
    frames = runtime().sample_media(video, max_frames=3)
    assert len(frames) == 3
    assert frames[0].timestamp_sec == 0.0
    assert frames[-1].timestamp_sec == pytest.approx(0.9)
    assert frames[0].image.size == (64, 48)
    assert frames[-1].image.getpixel((0, 0))[0] > 150


@pytest.mark.parametrize("limit", [0, -1, 17, 1.5])
def test_invalid_sampling_limit_is_rejected(video, limit):
    with pytest.raises(ValueError):
        runtime().sample_media(video, max_frames=limit)


def test_corrupt_video_produces_actionable_error(tmp_path):
    path = tmp_path / "bad.mp4"
    path.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="video"):
        runtime().sample_media(path)


def test_frame_counts_and_annotated_preview_come_from_detector(video, monkeypatch):
    module = runtime()

    def predict(image, threshold, labels):
        return [
            {
                "label": "car",
                "confidence": 0.9,
                "bbox": {"x1": 5, "y1": 5, "x2": 40, "y2": 30},
            }
        ]

    monkeypatch.setattr(module, "predict_objects", predict)
    output = module.analyze_media(video, max_frames=3)
    assert output.result["frames_processed"] == 3
    assert len(output.result["detections"]) == 3
    assert output.result["counts"] == {"car": 3}
    assert output.result["detections"][-1]["timestamp_sec"] == pytest.approx(0.9)
    assert output.preview is not None
    assert output.preview.getpixel((5, 5)) != (0, 0, 0)
    json.dumps(output.result)


def test_invalid_threshold_fails_before_inference(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (32, 24), "white").save(image)
    with pytest.raises(ValueError, match="threshold"):
        runtime().analyze_media(image, threshold=float("nan"))


def test_default_mode_rejects_unknown_class_filter(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (32, 24), "white").save(image)
    with pytest.raises(ValueError, match="filter"):
        runtime().analyze_media(image, prompt="describe everything in detail")


def test_oversized_video_dimensions_fail_before_decoding(tmp_path, monkeypatch):
    path = tmp_path / "huge.avi"
    path.write_bytes(b"fixture")

    class Capture:
        released = False

        def isOpened(self):
            return True

        def get(self, key):
            return {
                cv2.CAP_PROP_FPS: 10,
                cv2.CAP_PROP_FRAME_COUNT: 10,
                cv2.CAP_PROP_FRAME_WIDTH: 50000,
                cv2.CAP_PROP_FRAME_HEIGHT: 50000,
            }[key]

        def read(self):
            pytest.fail("Oversized video must not be decoded")

        def set(self, *args):
            pass

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    with pytest.raises(ValueError, match="resolution"):
        runtime().sample_media(path)
    assert capture.released


def test_truncated_jpeg_is_a_validation_error(tmp_path):
    path = tmp_path / "truncated.jpg"
    Image.new("RGB", (64, 48), "white").save(path)
    path.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(ValueError, match="image"):
        runtime().sample_media(path)
