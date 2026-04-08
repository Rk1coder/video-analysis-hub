"""Base analyzer interface — all model modules must subclass this."""
from abc import ABC, abstractmethod
from typing import Any


class BaseAnalyzer(ABC):
    """Abstract base for all video/image analyzers."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights into memory."""
        ...

    @abstractmethod
    def analyze_video(self, video_path: str, prompt: str, **kwargs) -> dict[str, Any]:
        """
        Analyze a video file.

        Returns:
            {
                "detections": [...],
                "frames_processed": int,
                "model": str,
            }
        """
        ...

    @abstractmethod
    def analyze_image(self, image_path: str, prompt: str, **kwargs) -> dict[str, Any]:
        """Analyze a single image."""
        ...

    def unload_model(self) -> None:
        """Free GPU memory (optional override)."""
        import torch
        self.model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device})"
