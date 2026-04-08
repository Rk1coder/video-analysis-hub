"""InternVideo2 action recognition analyzer."""
import os
from typing import Any

from models.base_analyzer import BaseAnalyzer

INTERNVIDEO_CHECKPOINT = os.getenv(
    "INTERNVIDEO_CHECKPOINT", "OpenGVLab/InternVideo2-Stage2_1B-224p-f4"
)


class InternVideoAnalyzer(BaseAnalyzer):
    """Action recognition via InternVideo2."""

    def load_model(self) -> None:
        from transformers import AutoProcessor, AutoModel  # type: ignore
        self.processor = AutoProcessor.from_pretrained(INTERNVIDEO_CHECKPOINT)
        self.model = AutoModel.from_pretrained(
            INTERNVIDEO_CHECKPOINT, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def analyze_image(self, image_path: str, prompt: str, **kwargs) -> dict[str, Any]:
        """Single-frame classification (fallback)."""
        from PIL import Image  # type: ignore
        if self.model is None:
            self.load_model()
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        import torch
        with torch.no_grad():
            logits = self.model(**inputs).logits
        top5 = logits.softmax(-1).topk(5)
        return {
            "model": "InternVideo2",
            "actions": [
                {"label": str(i.item()), "score": round(s.item(), 4)}
                for i, s in zip(top5.indices[0], top5.values[0])
            ],
        }

    def analyze_video(self, video_path: str, prompt: str, **kwargs) -> dict[str, Any]:
        """Clip-level action recognition."""
        if self.model is None:
            self.load_model()
        import decord, torch  # type: ignore
        vr = decord.VideoReader(video_path)
        num_frames = min(len(vr), 32)
        indices = list(range(0, len(vr), max(1, len(vr) // num_frames)))[:num_frames]
        frames = vr.get_batch(indices).asnumpy()
        inputs = self.processor(videos=list(frames), return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        top5 = logits.softmax(-1).topk(5)
        return {
            "model": "InternVideo2",
            "frames_processed": num_frames,
            "actions": [
                {"label": str(i.item()), "score": round(s.item(), 4)}
                for i, s in zip(top5.indices[0], top5.values[0])
            ],
        }
