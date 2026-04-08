"""Gemma2 vision analyzer — transformers direct inference (no vLLM)."""
import os
from typing import Any

from models.base_analyzer import BaseAnalyzer
from utils.video_utils import extract_frames

GEMMA2_CHECKPOINT = os.getenv("GEMMA2_CHECKPOINT", "google/gemma-2-2b-it")


class Gemma2Analyzer(BaseAnalyzer):
    """Gemma2 multimodal QA via HuggingFace transformers."""

    def load_model(self) -> None:
        from transformers import AutoProcessor, AutoModelForImageTextToText  # type: ignore
        self.processor = AutoProcessor.from_pretrained(GEMMA2_CHECKPOINT)
        self.model = AutoModelForImageTextToText.from_pretrained(
            GEMMA2_CHECKPOINT, device_map="auto"
        )

    def _infer(self, image, prompt: str) -> str:
        import torch
        inputs = self.processor(text=prompt, images=image,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=512)
        return self.processor.decode(ids[0], skip_special_tokens=True)

    def analyze_image(self, image_path: str, prompt: str, **kwargs) -> dict[str, Any]:
        if self.model is None:
            self.load_model()
        from PIL import Image  # type: ignore
        img = Image.open(image_path).convert("RGB")
        return {"model": GEMMA2_CHECKPOINT, "response": self._infer(img, prompt)}

    def analyze_video(self, video_path: str, prompt: str, **kwargs) -> dict[str, Any]:
        if self.model is None:
            self.load_model()
        frames = extract_frames(video_path, interval_sec=2.0)
        responses = []
        for frame_path, ts in frames:
            r = self.analyze_image(frame_path, prompt)
            responses.append({"timestamp_sec": ts, "response": r["response"]})
        return {"model": GEMMA2_CHECKPOINT, "frames_processed": len(frames),
                "responses": responses}
