"""Bounded CPU image/video inference shared by Gradio and FastAPI."""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

MODEL_ID = "hustvl/yolos-tiny"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_MEDIA_PIXELS = 20_000_000
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
COCO_LABELS = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
}
_INFERENCE_LOCK = threading.Lock()


@dataclass
class Sample:
    image: Image.Image
    timestamp_sec: float | None


@dataclass
class AnalysisOutput:
    result: dict
    preview: Image.Image | None


def available_models() -> list[str]:
    return ["yolos"] + (["gemma3"] if os.getenv("VLLM_ENDPOINT") else [])


def validate_media(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise ValueError("Please upload an existing image or video file.")
    if path.suffix.lower() not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        raise ValueError("Unsupported file type. Use JPG, PNG, WebP, MP4, AVI, MOV, MKV or WebM.")
    size = path.stat().st_size
    if size == 0:
        raise ValueError("The uploaded file is empty.")
    if size > MAX_UPLOAD_BYTES:
        raise ValueError("The upload exceeds the 20 MB limit.")
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        try:
            with Image.open(path) as image:
                if image.width * image.height > MAX_MEDIA_PIXELS:
                    raise ValueError("Images must contain at most 20 million pixels.")
                image.verify()
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise ValueError(
                "The image could not be decoded. Please upload a valid image."
            ) from exc
    return path


def sample_media(
    path: str | Path, max_frames: int = 8, frame_interval: float | None = None
) -> list[Sample]:
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or not 1 <= max_frames <= 16:
        raise ValueError("max_frames must be an integer between 1 and 16.")
    if frame_interval is not None and (not math.isfinite(frame_interval) or frame_interval <= 0):
        raise ValueError("frame_interval must be positive and finite.")
    path = validate_media(path)
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((1280, 1280))
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise ValueError(
                "The image could not be decoded. Please upload a valid image."
            ) from exc
        return [Sample(image, None)]
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("The video could not be opened. Try an MP4 or AVI file.")
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if (
            not math.isfinite(width)
            or not math.isfinite(height)
            or width <= 0
            or height <= 0
            or width * height > MAX_MEDIA_PIXELS
        ):
            raise ValueError(
                "The video resolution must contain at most 20 million pixels per frame."
            )
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count < 1:
            raise ValueError("The video has no readable frame timing information.")
        count = int(count)
        if frame_interval is None:
            indices = np.linspace(0, count - 1, min(count, max_frames), dtype=int)
        else:
            step = max(1, round(fps * frame_interval))
            indices = range(0, min(count, step * max_frames), step)
        samples = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                continue
            if frame.shape[0] * frame.shape[1] > MAX_MEDIA_PIXELS:
                raise ValueError(
                    "The decoded video resolution exceeds 20 million pixels per frame."
                )
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image.thumbnail((1280, 1280))
            samples.append(Sample(image, round(int(index) / fps, 3)))
        if not samples:
            raise ValueError("The video contains no decodable frames.")
        return samples
    finally:
        capture.release()


@lru_cache(maxsize=1)
def _detector():
    import torch
    from transformers import YolosForObjectDetection, YolosImageProcessor

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    processor = YolosImageProcessor.from_pretrained(
        MODEL_ID, size={"shortest_edge": 480, "longest_edge": 640}
    )
    model = YolosForObjectDetection.from_pretrained(MODEL_ID, use_safetensors=True)
    model.to("cpu").eval()
    return processor, model


def predict_objects(image: Image.Image, threshold: float, labels: set[str]) -> list[dict]:
    import torch

    try:
        with _INFERENCE_LOCK, torch.inference_mode():
            processor, model = _detector()
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            result = processor.post_process_object_detection(
                outputs, threshold=threshold, target_sizes=[(image.height, image.width)]
            )[0]
    except (OSError, ImportError) as exc:
        raise RuntimeError(
            "The detector could not load. The first run needs internet access to "
            "download hustvl/yolos-tiny; install requirements.txt and retry."
        ) from exc
    detections = []
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        label = model.config.id2label[int(label_id)]
        if labels and label not in labels:
            continue
        x1, y1, x2, y2 = box.tolist()
        detections.append(
            {
                "label": label,
                "confidence": round(float(score), 4),
                "bbox": {
                    "x1": round(max(0.0, min(image.width, x1)), 2),
                    "y1": round(max(0.0, min(image.height, y1)), 2),
                    "x2": round(max(0.0, min(image.width, x2)), 2),
                    "y2": round(max(0.0, min(image.height, y2)), 2),
                },
            }
        )
    return detections


def _ask_gemma(image: Image.Image, prompt: str) -> str:
    endpoint = os.getenv("VLLM_ENDPOINT", "").rstrip("/")
    if not endpoint:
        raise RuntimeError(
            "Set VLLM_ENDPOINT to a running vision-language model server to use Gemma."
        )
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    try:
        response = httpx.post(
            endpoint + "/chat/completions",
            timeout=120,
            json={
                "model": os.getenv("GEMMA_MODEL", "google/gemma-3-4b-it"),
                "max_tokens": 512,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/jpeg;base64," + encoded},
                            },
                            {
                                "type": "text",
                                "text": f"Image dimensions: {image.width}x{image.height} pixels.\nCoordinates must be in pixel space.\n"
                                + (prompt or "Describe the visible objects and scene."),
                            },
                        ],
                    }
                ],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "The configured Gemma service is unavailable or returned an invalid response."
        ) from exc


def _gemma_detections(raw_text: str) -> list[dict]:
    """Preserve the legacy box_2d response contract without trusting malformed boxes."""
    detections = []
    for candidate in re.findall(r"\{[^{}]+\}", raw_text):
        try:
            item = json.loads(candidate)
            box = item["box_2d"]
            if not isinstance(box, list) or len(box) != 4:
                continue
            if not all(isinstance(n, (int, float)) and math.isfinite(n) for n in box):
                continue
            y1, x1, y2, x2 = box
            if y2 < y1 or x2 < x1:
                continue
            score = float(item.get("confidence", 1.0))
            if not math.isfinite(score) or not 0 <= score <= 1:
                continue
            detections.append(
                {
                    "bbox": {"y1": y1, "x1": x1, "y2": y2, "x2": x2},
                    "label": item.get("label", "object"),
                    "confidence": score,
                }
            )
        except (ValueError, KeyError, TypeError):
            continue
    return detections


def _preview(samples: list[Sample], per_frame: list[list[dict]]) -> Image.Image:
    tiles = []
    for sample, detections in zip(samples, per_frame):
        image = sample.image.copy()
        draw = ImageDraw.Draw(image)
        for detection in detections:
            box = detection["bbox"]
            draw.rectangle((box["x1"], box["y1"], box["x2"], box["y2"]), outline="#22c55e", width=3)
            draw.text(
                (box["x1"] + 3, box["y1"] + 3),
                f"{detection['label']} {detection['confidence']:.2f}",
                fill="#22c55e",
            )
        image.thumbnail((480, 360))
        tiles.append(image)
    if len(tiles) == 1:
        return tiles[0]
    width = max(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles) + 24
    columns = min(2, len(tiles))
    sheet = Image.new("RGB", (width * columns, height * math.ceil(len(tiles) / columns)), "#111827")
    draw = ImageDraw.Draw(sheet)
    for n, (tile, sample) in enumerate(zip(tiles, samples)):
        x, y = (n % columns) * width, (n // columns) * height
        sheet.paste(tile, (x, y))
        draw.text((x + 4, y + height - 20), f"{sample.timestamp_sec:.2f} s", fill="white")
    return sheet


def analyze_media(
    path: str | Path,
    model_key: str = "yolos",
    prompt: str = "",
    threshold: float = 0.5,
    max_frames: int = 8,
    frame_interval: float | None = None,
) -> AnalysisOutput:
    if model_key not in {"yolos", "gemma3"}:
        raise ValueError("Unknown model. Choose yolos, or configure Gemma with VLLM_ENDPOINT.")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite number between 0 and 1.")
    labels = {value.strip().lower() for value in prompt.split(",") if value.strip()}
    if model_key == "yolos" and labels - COCO_LABELS:
        raise ValueError(
            "The class filter must contain English COCO labels separated by commas, such as car, person, cat. Leave it blank for all classes."
        )
    started = time.perf_counter()
    samples = sample_media(path, max_frames=max_frames, frame_interval=frame_interval)
    if model_key == "gemma3":
        responses = [
            {"timestamp_sec": s.timestamp_sec, "response": _ask_gemma(s.image, prompt)}
            for s in samples
        ]
        detections = []
        for response in responses:
            for detection in _gemma_detections(response["response"]):
                if response["timestamp_sec"] is not None:
                    detection["timestamp_sec"] = response["timestamp_sec"]
                detections.append(detection)
        result = {
            "model": os.getenv("GEMMA_MODEL", "google/gemma-3-4b-it"),
            "frames_processed": len(samples),
            "responses": responses,
            "detections": detections,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        if samples[0].timestamp_sec is None:
            result["raw_response"] = responses[0]["response"]
        return AnalysisOutput(result, _preview(samples, [[] for _ in samples]))
    per_frame = [predict_objects(sample.image, threshold, labels) for sample in samples]
    detections = []
    for index, (sample, items) in enumerate(zip(samples, per_frame)):
        for detection in items:
            detections.append(
                {
                    **detection,
                    "frame_index": index,
                    "timestamp_sec": sample.timestamp_sec,
                }
            )
    result = {
        "model": MODEL_ID,
        "device": "cpu",
        "frames_processed": len(samples),
        "detections": detections,
        "counts": dict(Counter(d["label"] for d in detections)),
        "frame_sizes": [
            {
                "width": s.image.width,
                "height": s.image.height,
                "timestamp_sec": s.timestamp_sec,
            }
            for s in samples
        ],
        "threshold": threshold,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "counting_note": "Video counts are detection occurrences across sampled frames, not unique tracked objects.",
    }
    return AnalysisOutput(result, _preview(samples, per_frame))
