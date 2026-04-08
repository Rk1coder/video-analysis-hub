"""Video / image utility helpers."""
import os
import tempfile
from pathlib import Path
from typing import NamedTuple

import cv2


class Frame(NamedTuple):
    path: str
    timestamp_sec: float


def extract_frames(
    video_path: str,
    interval_sec: float = 1.0,
    max_frames: int = 120,
    output_dir: str | None = None,
    return_dir: bool = False,
) -> list[Frame] | dict:
    """
    Extract frames from a video at a given interval.

    Args:
        video_path: Path to input video.
        interval_sec: Seconds between extracted frames.
        max_frames: Hard cap on total frames.
        output_dir: Directory to save PNGs (temp dir if None).
        return_dir: If True, return {"dir": ..., "frames": [...]} instead.

    Returns:
        List of (path, timestamp_sec) named tuples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(fps * interval_sec))

    save_dir = output_dir or tempfile.mkdtemp(prefix="vah_frames_")
    os.makedirs(save_dir, exist_ok=True)

    frames: list[Frame] = []
    frame_idx = 0
    saved = 0

    while saved < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, img = cap.read()
        if not ret:
            break
        ts = frame_idx / fps
        out_path = os.path.join(save_dir, f"frame_{saved:05d}.jpg")
        cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        frames.append(Frame(path=out_path, timestamp_sec=round(ts, 3)))
        frame_idx += step
        saved += 1

    cap.release()

    if return_dir:
        return {"dir": save_dir, "frames": frames}
    return frames


def write_video_h264(frames_dir: str, output_path: str, fps: float = 25.0) -> str:
    """
    Encode a directory of JPEG frames into an H.264 MP4.
    Falls back to mp4v if H.264 is not available.
    """
    frame_files = sorted(Path(frames_dir).glob("*.jpg"))
    if not frame_files:
        raise ValueError(f"No frames found in {frames_dir}")

    sample = cv2.imread(str(frame_files[0]))
    h, w = sample.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for f in frame_files:
        out.write(cv2.imread(str(f)))
    out.release()
    return output_path


def image_media_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")
