"""Gemma3 / Gemma4 vision analyzer — routes requests to a vLLM endpoint."""
import base64
import os
from pathlib import Path
from typing import Any

import cv2
import httpx

from models.base_analyzer import BaseAnalyzer
from utils.video_utils import extract_frames


VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://localhost:8000")
MODEL_NAME = os.getenv("GEMMA_MODEL", "google/gemma-3-4b-it")
FRAME_INTERVAL_SEC = float(os.getenv("FRAME_INTERVAL_SEC", "2.0"))


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


class Gemma3Analyzer(BaseAnalyzer):
    """
    Vision QA / object detection via Gemma3 or Gemma4.
    Uses vLLM OpenAI-compatible /v1/chat/completions endpoint.
    Coordinates returned in pixel space (embedded in prompt).
    """

    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.endpoint = f"{VLLM_ENDPOINT}/v1/chat/completions"
        self.model_name = MODEL_NAME

    def load_model(self) -> None:
        # Model lives in vLLM container — nothing to load locally.
        pass

    # ------------------------------------------------------------------
    def _call_vllm(self, image_b64: str, media_type: str, prompt: str,
                   width: int, height: int) -> dict:
        full_prompt = (
            f"Image dimensions: {width}x{height} pixels.\n"
            f"Coordinates must be in pixel space (0–{width} x, 0–{height} y).\n\n"
            + prompt
        )
        payload = {
            "model": self.model_name,
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_b64}"
                            },
                        },
                        {"type": "text", "text": full_prompt},
                    ],
                }
            ],
        }
        resp = httpx.post(self.endpoint, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()

    def _parse_detections(self, raw_text: str) -> list[dict]:
        """
        Parse Gemma box_2d format:
            {"box_2d": [y1, x1, y2, x2], "label": "...", "confidence": 0.9}
        Handles both list and single-object responses.
        """
        import json, re
        detections = []
        # Try to find JSON objects / arrays in the response
        candidates = re.findall(r'\{[^{}]+\}', raw_text)
        for c in candidates:
            try:
                obj = json.loads(c)
                if "box_2d" in obj:
                    coords = obj["box_2d"]  # [y1, x1, y2, x2]
                    detections.append({
                        "bbox": {
                            "y1": coords[0], "x1": coords[1],
                            "y2": coords[2], "x2": coords[3],
                        },
                        "label": obj.get("label", "object"),
                        "confidence": obj.get("confidence", 1.0),
                    })
            except (json.JSONDecodeError, IndexError, KeyError):
                continue
        return detections

    # ------------------------------------------------------------------
    def analyze_image(self, image_path: str, prompt: str, **kwargs) -> dict[str, Any]:
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        ext = Path(image_path).suffix.lower()
        media_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                      ".png": "image/png", ".webp": "image/webp"}.get(ext, "image/jpeg")
        b64 = _encode_image(image_path)
        resp = self._call_vllm(b64, media_type, prompt, w, h)
        raw = resp["choices"][0]["message"]["content"]
        return {
            "model": self.model_name,
            "detections": self._parse_detections(raw),
            "raw_response": raw,
        }

    def analyze_video(self, video_path: str, prompt: str,
                      frame_interval: float | None = None, **kwargs) -> dict[str, Any]:
        interval = frame_interval or FRAME_INTERVAL_SEC
        frames = extract_frames(video_path, interval_sec=interval)
        all_detections = []
        for frame_path, timestamp in frames:
            result = self.analyze_image(frame_path, prompt)
            for det in result["detections"]:
                det["timestamp_sec"] = timestamp
            all_detections.extend(result["detections"])
        return {
            "model": self.model_name,
            "frames_processed": len(frames),
            "detections": all_detections,
        }
