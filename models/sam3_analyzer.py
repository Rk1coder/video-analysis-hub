"""
SAM3 analyzer — facebook/sam3 via HuggingFace Transformers.

Üç mod:
  - image      : Sam3Model  (text veya bbox prompt → tüm eşleşen instance'lar)
  - image_pvs  : Sam3TrackerModel  (nokta/bbox → tek instance, SAM2 drop-in)
  - video      : Sam3TrackerVideoModel  (bbox/nokta → video propagation)

HF token gerektirir: model gated (Meta lisansı).
"""
from __future__ import annotations

import os
from typing import Any

import torch

from models.base_analyzer import BaseAnalyzer
from utils.video_utils import extract_frames

HF_CHECKPOINT = os.getenv("SAM3_CHECKPOINT", "facebook/sam3")
HF_TOKEN = os.getenv("HF_TOKEN", None)
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Image — Promptable Concept Segmentation (text veya bbox)
# ─────────────────────────────────────────────────────────────────────────────
class Sam3ImageAnalyzer(BaseAnalyzer):
    """
    Sam3Model: text veya bbox prompt → görüntüdeki TÜM eşleşen objelerin maskesi.
    """

    def load_model(self) -> None:
        from transformers import Sam3Processor, Sam3Model  # type: ignore
        self.processor = Sam3Processor.from_pretrained(HF_CHECKPOINT, token=HF_TOKEN)
        self.model = Sam3Model.from_pretrained(HF_CHECKPOINT, token=HF_TOKEN).to(self.device)
        self.model.eval()

    def analyze_image(
        self,
        image_path: str,
        prompt: str,
        input_boxes: list | None = None,
        input_boxes_labels: list | None = None,
        threshold: float = 0.5,
        **kwargs,
    ) -> dict[str, Any]:
        if self.model is None:
            self.load_model()

        from PIL import Image  # type: ignore
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            text=prompt if prompt else None,
            input_boxes=[input_boxes] if input_boxes else None,
            input_boxes_labels=[input_boxes_labels] if input_boxes_labels else None,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        return {
            "model": "SAM3",
            "mode": "image_pcs",
            "prompt": prompt,
            "num_objects": len(results["masks"]),
            "boxes": results["boxes"].tolist(),      # xyxy, pixel coords
            "scores": results["scores"].tolist(),
            "masks_shape": list(results["masks"].shape),
        }

    def analyze_video(self, video_path: str, prompt: str, **kwargs) -> dict[str, Any]:
        frames = extract_frames(video_path, interval_sec=kwargs.get("interval_sec", 1.0))
        all_results = []
        for frame_path, ts in frames:
            r = self.analyze_image(frame_path, prompt, **kwargs)
            r["timestamp_sec"] = ts
            all_results.append(r)
        return {
            "model": "SAM3",
            "mode": "image_pcs_per_frame",
            "frames_processed": len(frames),
            "frame_results": all_results,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Image — Promptable Visual Segmentation (nokta / bbox → tek instance)
# ─────────────────────────────────────────────────────────────────────────────
class Sam3TrackerImageAnalyzer(BaseAnalyzer):
    """
    Sam3TrackerModel: SAM2 drop-in replacement.
    Nokta veya bbox prompt → belirli bir instance'ın maskesi.
    """

    def load_model(self) -> None:
        from transformers import Sam3TrackerProcessor, Sam3TrackerModel  # type: ignore
        self.processor = Sam3TrackerProcessor.from_pretrained(HF_CHECKPOINT, token=HF_TOKEN)
        self.model = Sam3TrackerModel.from_pretrained(HF_CHECKPOINT, token=HF_TOKEN).to(self.device)
        self.model.eval()

    def analyze_image(
        self,
        image_path: str,
        prompt: str = "",
        input_points: list | None = None,
        input_labels: list | None = None,
        input_boxes: list | None = None,
        multimask_output: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        if self.model is None:
            self.load_model()

        from PIL import Image  # type: ignore
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=multimask_output)

        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
        )[0]

        return {
            "model": "SAM3Tracker",
            "mode": "image_pvs",
            "num_objects": masks.shape[0],
            "masks_shape": list(masks.shape),
        }

    def analyze_video(self, video_path: str, prompt: str, **kwargs) -> dict[str, Any]:
        frames = extract_frames(video_path, interval_sec=kwargs.get("interval_sec", 1.0))
        results = []
        for fp, ts in frames:
            r = self.analyze_image(fp, prompt, **kwargs)
            r["timestamp_sec"] = ts
            results.append(r)
        return {"model": "SAM3Tracker", "frames_processed": len(frames), "results": results}


# ─────────────────────────────────────────────────────────────────────────────
# Video — Promptable Visual Segmentation (video propagation)
# ─────────────────────────────────────────────────────────────────────────────
class Sam3VideoTracker(BaseAnalyzer):
    """
    Sam3TrackerVideoModel: SAM2 Video drop-in replacement.
    İlk frame'e nokta/bbox ver → tüm video boyunca mask propagation.
    """

    def load_model(self) -> None:
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor  # type: ignore
        self.processor = Sam3TrackerVideoProcessor.from_pretrained(HF_CHECKPOINT, token=HF_TOKEN)
        self.model = Sam3TrackerVideoModel.from_pretrained(
            HF_CHECKPOINT, token=HF_TOKEN,
        ).to(self.device, dtype=torch.bfloat16)
        self.model.eval()

    def analyze_image(self, image_path: str, prompt: str, **kwargs) -> dict[str, Any]:
        raise NotImplementedError("Sam3VideoTracker yalnızca video analizi destekler.")

    def analyze_video(
        self,
        video_path: str,
        prompt: str = "",
        ann_frame_idx: int = 0,
        obj_ids: int | list[int] = 1,
        input_points: list | None = None,
        input_labels: list | None = None,
        input_boxes: list | None = None,
        max_frames: int = 200,
        **kwargs,
    ) -> dict[str, Any]:
        if self.model is None:
            self.load_model()

        from transformers.video_utils import load_video  # type: ignore
        video_frames, _ = load_video(video_path)

        inference_session = self.processor.init_video_session(
            video=video_frames,
            inference_device=self.device,
            dtype=torch.bfloat16,
        )

        self.processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=obj_ids,
            input_points=input_points or [[[[256, 256]]]],
            input_labels=input_labels or [[[1]]],
        )

        video_segments: dict[int, Any] = {}
        for output in self.model.propagate_in_video_iterator(
            inference_session, max_frame_num_to_track=max_frames
        ):
            masks = self.processor.post_process_masks(
                [output.pred_masks],
                original_sizes=[[inference_session.video_height,
                                  inference_session.video_width]],
                binarize=False,
            )[0]
            video_segments[output.frame_idx] = {
                "masks_shape": list(masks.shape),
                "obj_ids": inference_session.obj_ids,
            }

        return {
            "model": "SAM3TrackerVideo",
            "frames_tracked": len(video_segments),
            "video_segments": video_segments,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Default export
# ─────────────────────────────────────────────────────────────────────────────
class Sam3Analyzer(Sam3VideoTracker):
    """
    Varsayılan SAM3 analyzer.
    - Video → Sam3TrackerVideoModel (propagation)
    - Image → Sam3ImageAnalyzer'a delege eder (text/bbox prompt, tüm instance'lar)
    """

    def __init__(self, device: str = _DEVICE):
        super().__init__(device)
        self._image_analyzer: Sam3ImageAnalyzer | None = None

    def analyze_image(self, image_path: str, prompt: str, **kwargs) -> dict[str, Any]:
        if self._image_analyzer is None:
            self._image_analyzer = Sam3ImageAnalyzer(self.device)
        return self._image_analyzer.analyze_image(image_path, prompt, **kwargs)
