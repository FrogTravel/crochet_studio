"""Shared Streamlit UI helpers.

Keeps ``app.py`` (public) and ``pages/1_Admin_Demo.py`` (admin) in
lock-step: model loader, upload/generate input block, sidebar controls,
and session-state bootstrapping all live here.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2 as cv
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from .config import DEFAULT_WEIGHTS


# ── Gemini wrapper (optional dependency) ───────────────────────────────────
try:
    from .generation import (
        DEFAULT_MODEL as GEMINI_DEFAULT_MODEL,
        DEFAULT_PROMPT as GEMINI_DEFAULT_PROMPT,
        generate_image,
    )
    GEMINI_AVAILABLE = True
except Exception:  # pragma: no cover — import-time defensive
    GEMINI_AVAILABLE = False
    generate_image = None  # type: ignore[assignment]
    GEMINI_DEFAULT_PROMPT = (
        "A high-resolution, technical crochet stitch diagram (scheme) on a "
        "clean white background with professional black line art."
    )
    GEMINI_DEFAULT_MODEL = "gemini-3.1-flash-image-preview"


# ── Model cache ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_crochet_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


# ── Image decoding helpers ─────────────────────────────────────────────────
def decode_uploaded_image(uploaded_file) -> np.ndarray | None:
    """Decode a Streamlit ``UploadedFile`` to a BGR numpy image."""
    if uploaded_file is None:
        return None
    data = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    return cv.imdecode(data, cv.IMREAD_COLOR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv.cvtColor(bgr, cv.COLOR_BGR2RGB))


# ── Session state helpers ──────────────────────────────────────────────────
_SESSION_KEYS: tuple[str, ...] = ("input_image", "input_pil", "input_source")


def ensure_input_state(keys: tuple[str, ...] = _SESSION_KEYS) -> None:
    for k in keys:
        if k not in st.session_state:
            st.session_state[k] = None


def store_image(bgr: np.ndarray, source: dict) -> None:
    st.session_state.input_image = bgr
    st.session_state.input_pil = bgr_to_pil(bgr)
    st.session_state.input_source = source


# ── Sidebar controls ───────────────────────────────────────────────────────
def sidebar_inference_controls(
    defaults: dict | None = None,
    weights_default: str = DEFAULT_WEIGHTS,
) -> dict:
    """Render the standard inference sidebar and return the settings."""
    d = defaults or {}
    with st.sidebar:
        st.header("Inference settings")
        conf = st.slider("Confidence threshold", 0.05, 0.95,
                         d.get("conf", 0.25), 0.05)
        iou = st.slider("NMS IoU threshold", 0.1, 0.9,
                        d.get("iou", 0.5), 0.05)
        target_stitch_px = st.slider("Target stitch size (px at 640)", 40, 200,
                                     d.get("target_stitch_px", 100), 10)
        tile_size = st.selectbox("YOLO tile size", [480, 640, 832, 1024],
                                 index=d.get("tile_index", 1))
        overlap = st.slider("Tile overlap", 0.0, 0.6,
                            d.get("overlap", 0.2), 0.05)
        st.divider()
        st.header("Model")
        weights_path = st.text_input("Weights path", weights_default)
    return {
        "conf":             conf,
        "iou":              iou,
        "target_stitch_px": target_stitch_px,
        "tile_size":        tile_size,
        "overlap":          overlap,
        "weights_path":     weights_path,
    }


# ── Input source block ─────────────────────────────────────────────────────
def input_source_block(
    mode_key: str = "input_mode",
    gemini_default_prompt: str | None = None,
) -> None:
    """Render the 'Generate with Gemini / Upload an image' block.

    Writes the resulting BGR image into ``st.session_state.input_image``
    (plus ``input_pil`` and ``input_source``) when the user provides one.
    """
    prompt_default = gemini_default_prompt or GEMINI_DEFAULT_PROMPT

    input_mode = st.radio(
        "Input source",
        ("Generate with Gemini", "Upload an image"),
        horizontal=True,
        key=mode_key,
    )

    if input_mode == "Generate with Gemini":
        if not GEMINI_AVAILABLE:
            st.warning(
                "The `google-genai` package isn't importable — the upload tab still works. "
                "Install it with `pip install google-genai` to enable generation."
            )
        prompt = st.text_area(
            "Prompt", value=prompt_default, height=150,
            help="Describe the crochet diagram you want Gemini to produce.",
        )
        col_a, col_b = st.columns([3, 1])
        with col_a:
            gmodel = st.text_input("Gemini model", GEMINI_DEFAULT_MODEL)
        with col_b:
            api_key = st.text_input(
                "API key (optional)", type="password",
                help="Falls back to GEMINI_API_KEY / GOOGLE_API_KEY env var.",
            )
        if st.button("🎨 Generate image",
                     disabled=not GEMINI_AVAILABLE, type="primary"):
            _do_generate(prompt, gmodel, api_key)
    else:
        uploaded = st.file_uploader(
            "Upload a crochet image", type=["jpg", "jpeg", "png", "bmp"],
        )
        if uploaded is not None:
            bgr = decode_uploaded_image(uploaded)
            if bgr is None:
                st.error("Could not decode the uploaded image.")
            else:
                store_image(bgr, {
                    "kind":     "upload",
                    "filename": uploaded.name,
                    "bytes":    int(uploaded.size) if hasattr(uploaded, "size") else None,
                })


def _do_generate(prompt: str, model_name: str, api_key: str) -> None:
    """Shared Gemini generation with progress UI."""
    if generate_image is None:
        st.error("Gemini generation is not available.")
        return
    with st.status("Generating image with Gemini…", expanded=True) as status:
        try:
            status.update(label="Submitting prompt to Gemini…", state="running")
            tmp_path = Path("free_output.png")
            t0 = time.time()
            generate_image(
                prompt=prompt,
                output_path=tmp_path,
                model=model_name,
                api_key=api_key or None,
            )
            elapsed = time.time() - t0
            pil = Image.open(tmp_path).convert("RGB")
            bgr = cv.cvtColor(np.array(pil), cv.COLOR_RGB2BGR)
            store_image(bgr, {
                "kind":    "gemini",
                "model":   model_name,
                "prompt":  prompt,
                "path":    str(tmp_path),
                "seconds": round(elapsed, 2),
            })
            status.update(
                label=f"✅ Image generated in {elapsed:.1f}s",
                state="complete",
            )
        except Exception as exc:
            status.update(label=f"❌ Generation failed: {exc}", state="error")
