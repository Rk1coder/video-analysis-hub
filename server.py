"""
server.py — FastAPI backend for Docker / local GPU deployment.
Endpoints mirror the Mil-VIS pipeline conventions.
"""
import os
import uuid
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from models.gemma3_analyzer import Gemma3Analyzer
from models.sam3_analyzer import Sam3Analyzer
from models.internvideo import InternVideoAnalyzer
from utils.video_utils import image_media_type

app = FastAPI(title="Video Analysis Hub API", version="1.0.0")

# Session storage  {session_id: output_path}
_sessions: dict[str, str] = {}

# Lazy-loaded analyzers
_gemma3: Gemma3Analyzer | None = None
_sam3: Sam3Analyzer | None = None
_intern: InternVideoAnalyzer | None = None


def get_gemma3() -> Gemma3Analyzer:
    global _gemma3
    if _gemma3 is None:
        _gemma3 = Gemma3Analyzer()
    return _gemma3


# ── Health ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ── Video analysis (Gemma3) ──────────────────────────────────────────────────
@app.post("/process-video-gemma")
async def process_video_gemma(
    video: Annotated[UploadFile, File()],
    prompt: Annotated[str, Form()] = "Detect all objects.",
    frame_interval: Annotated[float, Form()] = 2.0,
):
    session_id = str(uuid.uuid4())
    suffix = Path(video.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        result = get_gemma3().analyze_video(
            tmp_path, prompt, frame_interval=frame_interval
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

    _sessions[session_id] = result
    return JSONResponse({"session_id": session_id, **result})


# ── Image analysis (Gemma3) ──────────────────────────────────────────────────
@app.post("/process-image-gemma")
async def process_image_gemma(
    image: Annotated[UploadFile, File()],
    prompt: Annotated[str, Form()] = "Detect all objects.",
):
    session_id = str(uuid.uuid4())
    suffix = Path(image.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        result = get_gemma3().analyze_image(tmp_path, prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

    _sessions[session_id] = result
    return JSONResponse({"session_id": session_id, **result})


# ── Download result ──────────────────────────────────────────────────────────
@app.get("/download/{session_id}")
def download_result(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    import json, tempfile
    data = _sessions[session_id]
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
    json.dump(data, out, ensure_ascii=False, indent=2)
    out.close()
    return FileResponse(
        out.name,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=result_{session_id}.json"},
    )
