# 🎯 Video Analysis Hub

> GPU-accelerated video & image analysis platform — powered by SAM3, Gemma 4, and InternVideo2.

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/YOUR_USERNAME/video-analysis-hub)
[![Docker](https://img.shields.io/badge/Docker-GPU%20Ready-2496ED?logo=docker)](https://hub.docker.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Architecture

```
video-analysis-hub/              ← GitHub (single source)
│
├── models/                      ← Model scriptleri (her biri bağımsız modül)
│   ├── base_analyzer.py         ← Abstract base class
│   ├── sam3_analyzer.py         ← SAM3 segmentation
│   ├── gemma2_analyzer.py       ← Gemma2 vision
│   ├── gemma3_analyzer.py       ← Gemma3 / Gemma4 vision (vLLM)
│   └── internvideo.py           ← InternVideo2 action recognition
│
├── utils/
│   └── video_utils.py           ← Frame extraction, codec helpers
│
├── app.py                       ← Gradio UI (HF Space entry point)
├── server.py                    ← FastAPI backend (Docker mode)
├── requirements.txt             ← HF Space dependencies
├── requirements-local.txt       ← Docker/lokal dependencies (CUDA)
├── docker-compose.yml
├── Dockerfile
├── examples/                    ← Demo videolar / görseller
└── .github/workflows/
    └── sync.yml                 ← GitHub → HF Space auto-sync
```

**İki deployment modu:**

| | HuggingFace Space | Docker Lokal |
|---|---|---|
| GPU | ZeroGPU (paylaşımlı) | Kendi GPU'n (tam kontrol) |
| Kurulum | Sıfır kurulum | `docker compose up` |
| Model yükleme | HF Hub'dan otomatik | Lokal `/models` mount |
| API | — | FastAPI `/process-*` endpoint'leri |

---

## Hızlı Başlangıç

### 🤗 HuggingFace Space (tarayıcıdan dene)

[space linkine git] → video/görsel yükle → analiz et.

### 🐳 Docker (lokal GPU)

```bash
git clone https://github.com/Rk1coder/video-analysis-hub.git
cd video-analysis-hub

# GPU destekli tam kurulum
docker compose up --build

# Arayüz: http://localhost:7860
# API:     http://localhost:8000
```

### 🐍 Manuel kurulum

```bash
pip install -r requirements-local.txt

# Gradio UI
python app.py

# veya FastAPI backend
uvicorn server:app --host 0.0.0.0 --port 8000
```

---

## Modeller

| Model | Görev | VRAM |
|---|---|---|
| **SAM3** | Video segmentation | ~8 GB |
| **Gemma3 / Gemma4** | Vision QA, detection (vLLM) | ~12 GB |
| **InternVideo2** | Action recognition | ~10 GB |

---

## Konfigürasyon

`docker-compose.yml` içinde ortam değişkenleri:

```yaml
ACTIVE_MODEL: gemma3        # sam3 | gemma2 | gemma3 | internvideo
VLLM_ENDPOINT: http://vllm:8000
HF_TOKEN: ""                # private model için
```

---

## Lisans

MIT — bkz. [LICENSE](LICENSE)
