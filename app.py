"""Gradio entry point shared by local runs and Hugging Face Spaces."""

import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
import gradio as gr

from analysis import MAX_UPLOAD_BYTES, analyze_media, available_models

_REPORTS = Path(tempfile.mkdtemp(prefix="vah_reports_"))
_LOGGER = logging.getLogger(__name__)


def run_analysis(file_obj, model_key="yolos", prompt="", threshold=0.5, max_frames=8):
    try:
        if file_obj is None:
            raise ValueError("Lütfen bir görüntü veya video yükleyin.")
        path = file_obj if isinstance(file_obj, (str, Path)) else file_obj.name
        output = analyze_media(
            path,
            model_key=model_key,
            prompt=prompt,
            threshold=float(threshold),
            max_frames=int(max_frames),
        )
        text = json.dumps(output.result, ensure_ascii=False, indent=2)
        previous = sorted(_REPORTS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for index, old in enumerate(previous):
            if index >= 127 or time.time() - old.stat().st_mtime > 3600:
                old.unlink(missing_ok=True)
        report = _REPORTS / (uuid.uuid4().hex + ".json")
        report.write_text(text, encoding="utf-8")
        return text, output.preview, str(report)
    except (ValueError, OSError, RuntimeError) as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), None, None
    except Exception:
        _LOGGER.exception("Analysis failed")
        return (
            json.dumps(
                {"error": "Analiz tamamlanamadı. Başka bir dosyayla tekrar deneyin."},
                ensure_ascii=False,
            ),
            None,
            None,
        )


with gr.Blocks(title="Video Analysis Hub", delete_cache=(3600, 3600)) as demo:
    gr.Markdown("# 🎯 Video Analysis Hub")
    gr.Markdown(
        "Görüntü ve videolarda nesne tespiti: **gerçek model çıktıları, işaretli önizleme ve indirilebilir JSON**. "
        "Varsayılan model CPU'da çalışır; ilk analizde model dosyaları indirilir."
    )
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Görüntü veya video · en fazla 20 MB",
                type="filepath",
                file_types=[
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                    ".mp4",
                    ".avi",
                    ".mov",
                    ".mkv",
                    ".webm",
                ],
            )
            model_select = gr.Dropdown(choices=available_models(), value="yolos", label="Model")
            prompt_input = gr.Textbox(
                label="Sınıf filtresi / Gemma sorusu",
                placeholder="CPU modeli: car, person, cat — tüm sınıflar için boş bırakın.",
                lines=2,
            )
            threshold_input = gr.Slider(0.1, 0.95, value=0.5, step=0.05, label="Güven eşiği")
            max_frames_input = gr.Slider(
                1, 16, value=8, step=1, label="Videodan analiz edilecek en fazla kare"
            )
            run_btn = gr.Button("▶ Analiz Et", variant="primary")
        with gr.Column(scale=1):
            preview = gr.Image(label="Tespitler · videolarda örneklenen kareler", type="pil")
            json_output = gr.Code(label="Sonuç (JSON)", language="json")
            download = gr.File(label="JSON raporunu indir", interactive=False)
    gr.Markdown(
        "Video sayımları örneklenen karelerdeki tespitlerdir; benzersiz nesne veya takip sayısı değildir. "
        "[Kaynak kod](https://github.com/Rk1coder/video-analysis-hub) · "
        "[Geri bildirim](https://huggingface.co/spaces/rabitrabi/video-analysis-hub/discussions)"
    )
    example = Path(__file__).parent / "examples" / "cats.jpg"
    if example.is_file():
        gr.Examples(
            examples=[[str(example), "yolos", "cat", 0.5, 1]],
            inputs=[
                file_input,
                model_select,
                prompt_input,
                threshold_input,
                max_frames_input,
            ],
            label="Hazır örnek · kediler",
        )
    run_btn.click(
        fn=run_analysis,
        inputs=[
            file_input,
            model_select,
            prompt_input,
            threshold_input,
            max_frames_input,
        ],
        outputs=[json_output, preview, download],
        api_name="analyze",
        concurrency_limit=1,
    )

if __name__ == "__main__":
    demo.queue(max_size=8, default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        max_file_size=MAX_UPLOAD_BYTES,
        theme=gr.themes.Monochrome(),
    )
