"""
app.py — Gradio frontend for Video Analysis Hub.
Entry point for HuggingFace Space deployment.
"""
import os
import tempfile
from pathlib import Path

import gradio as gr

# ── Model registry ──────────────────────────────────────────────────────────
MODELS = {
    "gemma3": None,
    "gemma2": None,
    "sam3": None,
    "internvideo": None,
}

def _get_analyzer(model_key: str):
    if MODELS[model_key] is None:
        if model_key == "gemma3":
            from models.gemma3_analyzer import Gemma3Analyzer
            MODELS[model_key] = Gemma3Analyzer()
        elif model_key == "gemma2":
            from models.gemma2_analyzer import Gemma2Analyzer
            MODELS[model_key] = Gemma2Analyzer()
        elif model_key == "sam3":
            from models.sam3_analyzer import Sam3Analyzer
            MODELS[model_key] = Sam3Analyzer()
        elif model_key == "internvideo":
            from models.internvideo import InternVideoAnalyzer
            MODELS[model_key] = InternVideoAnalyzer()
    return MODELS[model_key]


# ── Core inference ───────────────────────────────────────────────────────────
def run_analysis(file_obj, model_key: str, prompt: str):
    if file_obj is None:
        return "❌ Lütfen bir dosya yükleyin.", None

    path = file_obj.name
    ext = Path(path).suffix.lower()
    is_video = ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    analyzer = _get_analyzer(model_key)

    try:
        if is_video:
            result = analyzer.analyze_video(path, prompt)
        else:
            result = analyzer.analyze_image(path, prompt)
    except Exception as e:
        return f"❌ Hata: {e}", None

    import json
    return json.dumps(result, ensure_ascii=False, indent=2), path


# ── Gradio UI ────────────────────────────────────────────────────────────────
DEFAULT_PROMPT = (
    "Detect all objects, vehicles, and persons. "
    "Return each as JSON with box_2d [y1,x1,y2,x2] in pixel coordinates, "
    "label, and confidence."
)

with gr.Blocks(title="Video Analysis Hub", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🎯 Video Analysis Hub")
    gr.Markdown(
        "Upload a **video** (MP4/AVI/MOV) or **image** (JPG/PNG/WEBP) "
        "and analyze with SAM3, Gemma3/4, or InternVideo2."
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Video veya Görsel",
                file_types=[".mp4", ".avi", ".mov", ".mkv",
                            ".jpg", ".jpeg", ".png", ".webp"],
            )
            model_select = gr.Dropdown(
                choices=["gemma3", "gemma2", "sam3", "internvideo"],
                value="gemma3",
                label="Model",
            )
            prompt_input = gr.Textbox(
                value=DEFAULT_PROMPT,
                label="Prompt",
                lines=4,
            )
            run_btn = gr.Button("▶ Analiz Et", variant="primary")

        with gr.Column(scale=1):
            json_output = gr.Code(label="Sonuç (JSON)", language="json")
            preview = gr.Video(label="Önizleme", visible=False)

    run_btn.click(
        fn=run_analysis,
        inputs=[file_input, model_select, prompt_input],
        outputs=[json_output, preview],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
