"""FastAPI service using the same inference runtime as the Space."""

import json
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from analysis import MAX_UPLOAD_BYTES, analyze_media, available_models, validate_media

app = FastAPI(title="Video Analysis Hub API", version="1.1.0")
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 64 * 1024
_REQUEST_SLOTS = threading.BoundedSemaphore(1)


class AnalysisRequestLimits:
    """Bound uploads before multipart spooling, and reject excess concurrent work."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        paths = {"/analyze", "/process-image-gemma", "/process-video-gemma"}
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in paths:
            return await self.app(scope, receive, send)
        if not _REQUEST_SLOTS.acquire(blocking=False):
            return await JSONResponse(
                {"detail": "An analysis is already running. Please retry shortly."},
                status_code=429,
                headers={"Retry-After": "1"},
            )(scope, receive, send)
        try:
            headers = dict(scope.get("headers", []))
            try:
                length = int(headers.get(b"content-length", b"0"))
            except ValueError:
                return await JSONResponse({"detail": "Invalid Content-Length."}, status_code=400)(
                    scope, receive, send
                )
            if length > MAX_REQUEST_BYTES:
                return await JSONResponse(
                    {"detail": "Request exceeds the 20 MB upload limit."}, status_code=413
                )(scope, receive, send)
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunk = message.get("body", b"")
                if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                    return await JSONResponse(
                        {"detail": "Request exceeds the 20 MB upload limit."}, status_code=413
                    )(scope, receive, send)
                body.extend(chunk)
                if not message.get("more_body", False):
                    break
            replayed = False

            async def replay():
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            await self.app(scope, replay, send)
        finally:
            _REQUEST_SLOTS.release()


app.add_middleware(AnalysisRequestLimits)
_sessions = OrderedDict()
_SESSION_LOCK = threading.Lock()


def _remember(result):
    session_id = uuid.uuid4().hex
    with _SESSION_LOCK:
        _sessions[session_id] = (time.monotonic(), result)
        while len(_sessions) > 128:
            _sessions.popitem(last=False)
    return session_id


def _process(upload, model_key, prompt, threshold, max_frames, frame_interval=None):
    try:
        suffix = Path(upload.filename or "").suffix.lower()
        with tempfile.TemporaryDirectory(prefix="vah_upload_") as directory:
            path = Path(directory) / ("input" + suffix)
            size = 0
            with path.open("wb") as target:
                while chunk := upload.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "The upload exceeds the 20 MB limit.")
                    target.write(chunk)
            validate_media(path)
            output = analyze_media(
                path,
                model_key=model_key,
                prompt=prompt,
                threshold=threshold,
                max_frames=max_frames,
                frame_interval=frame_interval,
            )
            return {"session_id": _remember(output.result), **output.result}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except (RuntimeError, OSError) as exc:
        raise HTTPException(503, str(exc)) from exc
    finally:
        upload.file.close()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "default_model": "yolos",
        "available_models": available_models(),
    }


@app.post("/analyze")
def analyze(
    file: Annotated[UploadFile, File()],
    model_key: Annotated[str, Form()] = "yolos",
    prompt: Annotated[str, Form(max_length=2000)] = "",
    threshold: Annotated[float, Form(ge=0, le=1, allow_inf_nan=False)] = 0.5,
    max_frames: Annotated[int, Form(ge=1, le=16)] = 8,
):
    return _process(file, model_key, prompt, threshold, max_frames)


@app.post("/process-image-gemma")
def process_image_gemma(
    image: Annotated[UploadFile, File()],
    prompt: Annotated[str, Form(max_length=2000)] = "Detect all objects.",
):
    return _process(image, "gemma3", prompt, 0.5, 1)


@app.post("/process-video-gemma")
def process_video_gemma(
    video: Annotated[UploadFile, File()],
    prompt: Annotated[str, Form(max_length=2000)] = "Detect all objects.",
    frame_interval: Annotated[float, Form(gt=0, allow_inf_nan=False)] = 2.0,
):
    return _process(video, "gemma3", prompt, 0.5, 8, frame_interval)


@app.get("/download/{session_id}")
def download_result(session_id: str):
    with _SESSION_LOCK:
        entry = _sessions.get(session_id)
        if entry is None or time.monotonic() - entry[0] > 3600:
            _sessions.pop(session_id, None)
            raise HTTPException(404, "Result not found or expired. Please run the analysis again.")
        text = json.dumps(entry[1], ensure_ascii=False, indent=2)
    return Response(
        text,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="result_{session_id}.json"'},
    )
