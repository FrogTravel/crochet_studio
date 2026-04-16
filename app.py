"""Public Streamlit interface for the crochet scheme generator.

Minimal, user-facing flow:

1. Generate with Gemini from a text prompt, or upload an image.
2. Click "Analyze stitches" — adaptive tiled YOLOv8n OBB inference.
3. Get a black-line reconstructed scheme at the input's aspect ratio
   and a JSON export of all detections.

All the heavy lifting lives in :mod:`crochet.streamlit_ui` and
:mod:`crochet.detection` so the admin page (``pages/1_Admin_Demo.py``)
can reuse it.
"""

from __future__ import annotations

import io
import json
import time

import streamlit as st

from crochet.detection import predict_adaptive_with_progress
from crochet.json_export import class_counts, detections_to_json
from crochet.rendering import render_scheme_image
from crochet.streamlit_ui import (
    ensure_input_state,
    input_source_block,
    load_crochet_model,
    sidebar_inference_controls,
)


st.set_page_config(layout="wide", page_title="Crochet Scheme Generator")
st.title("🧶 Crochet Symbol to Scheme")
st.caption(
    "Generate a crochet diagram with Gemini or upload your own, then run "
    "adaptive tiled YOLOv8n OBB inference."
)

ensure_input_state()

settings = sidebar_inference_controls()

try:
    model = load_crochet_model(settings["weights_path"])
except Exception as exc:
    st.error(f"Could not load YOLO weights at `{settings['weights_path']}`: {exc}")
    st.stop()

input_source_block()

image_bgr = st.session_state.input_image

col_left, col_right = st.columns(2)
with col_left:
    st.subheader("Input image")
    if image_bgr is None:
        st.info("Provide an image above (generate or upload) to begin.")
    else:
        st.image(st.session_state.input_pil, use_container_width=True)
        h, w = image_bgr.shape[:2]
        st.caption(f"{w} × {h} px")

run_clicked = st.button(
    "🔎 Analyze stitches",
    type="primary",
    disabled=image_bgr is None,
)

if run_clicked and image_bgr is not None:
    tile_bar = st.progress(0.0, text="Preparing inference…")
    log_box = st.empty()
    log_lines: list[str] = []

    def _on_phase(msg: str) -> None:
        log_lines.append(f"• {msg}")
        log_box.markdown("\n".join(log_lines))

    def _on_tile(idx: int, total: int, x1: int, y1: int, n: int) -> None:
        frac = idx / max(total, 1)
        tile_bar.progress(
            frac,
            text=f"Tile {idx}/{total} @ ({x1},{y1}) — {n} detections",
        )

    with st.status("Running YOLO OBB inference…", expanded=True) as status:
        try:
            t0 = time.time()
            status.update(label="Estimating stitch size…", state="running")
            detections, info = predict_adaptive_with_progress(
                model,
                image_bgr,
                target_stitch_px=settings["target_stitch_px"],
                tile_size=settings["tile_size"],
                overlap=settings["overlap"],
                conf=settings["conf"],
                iou_threshold=settings["iou"],
                on_phase=_on_phase,
                on_tile=_on_tile,
            )
            infer_seconds = time.time() - t0
            info["seconds"] = round(infer_seconds, 2)

            tile_bar.progress(1.0, text="Rendering reconstruction…")
            status.update(label="Rendering reconstruction…", state="running")
            scheme_img = render_scheme_image(image_bgr, detections)

            status.update(label="Building JSON export…", state="running")
            payload = detections_to_json(
                image_bgr, detections, info, st.session_state.input_source or {},
            )

            status.update(
                label=f"✅ {len(detections)} stitches detected in {infer_seconds:.1f}s",
                state="complete",
            )
        except Exception as exc:
            status.update(label=f"❌ Inference failed: {exc}", state="error")
            st.exception(exc)
            st.stop()

    with col_right:
        st.subheader("Digital scheme")
        st.image(scheme_img, use_container_width=True)

        scheme_buf = io.BytesIO()
        scheme_img.save(scheme_buf, format="PNG")
        st.download_button(
            "⬇️ Download scheme PNG",
            data=scheme_buf.getvalue(),
            file_name="crochet_scheme.png",
            mime="image/png",
        )

    st.subheader("Detections JSON")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stitches", len(detections))
    m2.metric("Tiles", info.get("tiles", "—"))
    m3.metric("Effective tile px", info.get("effective_tile_px", "—"))
    est = info.get("estimated_stitch_height_px")
    m4.metric("Est. stitch px", f"{est:.1f}" if isinstance(est, (int, float)) else "—")

    counts = class_counts(detections)
    if counts:
        st.write("**Per-class counts**")
        st.json(counts)

    json_bytes = json.dumps(payload, indent=2).encode("utf-8")
    st.download_button(
        "⬇️ Download detections JSON",
        data=json_bytes,
        file_name="crochet_scheme.json",
        mime="application/json",
    )
    with st.expander("Preview JSON"):
        st.json(payload)
