---
title: Video Analysis Hub
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.11.0
python_version: "3.12"
app_file: app.py
pinned: true
license: mit
models:
- hustvl/yolos-tiny
short_description: CPU object detection, annotated previews and JSON reports.
---

# Video Analysis Hub

Upload an image or video, detect objects, inspect the annotated frames and download a JSON report. The default path runs a real, open [YOLOS Tiny detector](https://huggingface.co/hustvl/yolos-tiny) on CPU, without an API key, a separate inference service or a GPU.

[Open Hugging Face app](https://huggingface.co/spaces/rabitrabi/video-analysis-hub) · [GitHub source](https://github.com/Rk1coder/video-analysis-hub) · [Feedback](https://huggingface.co/spaces/rabitrabi/video-analysis-hub/discussions)

## Try it

1. Upload a JPG, PNG, WebP, MP4, AVI, MOV, MKV or WebM file, up to 20 MB.
2. Leave the class filter empty to detect all COCO classes, or enter English labels such as `car, person`.
3. Adjust confidence and the video frame limit, then click **Analiz Et**.
4. Review the image/contact sheet and download the JSON report.

The first inference downloads the model weights. Videos sample up to 16 frames across the clip (8 by default). Results include labels, confidence, bounding boxes, timestamps and elapsed time. Images and sampled frames are resized to fit 1280×1280; returned coordinates refer to the reported `frame_sizes`.

**Video counts are detection occurrences across sampled frames, not unique tracked objects.** Brief events between sampled frames may be missed. COCO classes do not cover arbitrary specialist objects, and this small detector is a working baseline rather than a research benchmark.

## Run locally

Use Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

On Windows, activate with `.venv\Scripts\Activate.ps1` instead. Open [localhost:7860](http://localhost:7860).

Start the API in another terminal:

```bash
uvicorn server:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health
curl -F file=@/path/to/your/image.jpg -F prompt=cat http://127.0.0.1:8000/analyze
```

The response contains `session_id`. Download that result from `GET /download/{session_id}`. Results are retained in memory for up to one hour, with a 128-result cap; restarting the API clears them. Temporary API uploads are removed on success and failure. [Interactive API docs](http://localhost:8000/docs).

## Docker

```bash
docker compose up --build
```

Default services run on CPU: UI on 7860 and API on 8000. Model downloads use a shared cache.

## Optional Gemma service

If you already operate a compatible vision-language model server, set `VLLM_ENDPOINT` and optionally `GEMMA_MODEL` before starting the app. The UI then offers `gemma3`; the prompt becomes a question for that server.

The optional GPU Compose profile requires an NVIDIA GPU and access to the selected model:

```bash
VLLM_ENDPOINT=http://vllm:8000 docker compose --profile gpu up --build
```

The legacy `/process-image-gemma` and `/process-video-gemma` routes remain available. Missing services return actionable errors. Gemma inference and the historical SAM3/InternVideo modules under `models/` need their own validated hardware/model setup; they are not advertised as verified CPU capabilities.

## Verify and contribute

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
ruff check analysis.py app.py server.py tests
```

The tests cover file handling, invalid inputs, bounded frame sampling, preview generation, API errors, upload cleanup and matching JSON downloads. They isolate only external inference; real model smoke-test evidence is reported separately.

GitHub Actions runs these checks before syncing the runtime to Hugging Face. Set the repository secret `HF_TOKEN` with write access to the Space to enable deployment. The sync uploads files through the Hub API and preserves Space history.

## Related work

- [TopoEKF: tracking, adaptive EKF and trajectory anomaly detection](https://github.com/Rk1coder/TopoEKF)
- [SAM3 Auto Dataset Factory: assisted annotation, review and YOLO export](https://github.com/Rk1coder/sam3-auto-dataset-factory)
- [Rabia on GitHub](https://github.com/Rk1coder) · [Rabia on Hugging Face](https://huggingface.co/rabitrabi)

Example image: [COCO sample used in the Hugging Face documentation](https://huggingface.co/datasets/huggingface/documentation-images/blob/main/coco_sample.png).
Application code: MIT. YOLOS Tiny weights: Apache 2.0; see the linked model card.

