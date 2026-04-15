"""Streamlit UI for the crochet pipeline.

Two input modes:
  1. Generate an image with Gemini from a text prompt.
  2. Upload your own photo / scan.

Both paths run the same adaptive tiled YOLOv8n OBB inference
(same logic as ``Google_colab_YOLO_OBB_pipeline.ipynb``) and render a
reconstruction figure + JSON export.
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import cv2 as cv
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from util.tiler import (
    Detection,
    _infer_tile,
    _nms,
    _tile_starts,
    estimate_stitch_size,
)

# Gemini generation is optional — the app still works without the API key.
try:
    from pipeline.generate_image import (
        DEFAULT_MODEL as GEMINI_DEFAULT_MODEL,
        DEFAULT_PROMPT as GEMINI_DEFAULT_PROMPT,
        generate_image,
    )
    GEMINI_AVAILABLE = True
except Exception:  # pragma: no cover — import-time defensive
    GEMINI_AVAILABLE = False
    GEMINI_DEFAULT_PROMPT = (
        "A high-resolution, technical crochet stitch diagram (scheme) on a "
        "clean white background with professional black line art."
    )
    GEMINI_DEFAULT_MODEL = "gemini-3.1-flash-image-preview"


# ── 1. Configuration ─────────────────────────────────────────────────────────
WEIGHTS_PATH = "runs/obb/obb_train23/weights/best.pt"

CLASS_CONFIG = {
    "chain":          {"abbr": "ch",  "color": "#0000FF"},
    "double":         {"abbr": "dc",  "color": "#FF00FF"},
    "double treble":  {"abbr": "dtr", "color": "#00AA00"},
    "enseble_chain":  {"abbr": "ec",  "color": "#FF8000"},
    "fan":            {"abbr": "fa",  "color": "#FF0000"},
    "half_double":    {"abbr": "hd",  "color": "#00FFFF"},
    "noise":          {"abbr": "no",  "color": "#808080"},
    "single":         {"abbr": "sc",  "color": "#FFD700"},
    "treble":         {"abbr": "tr",  "color": "#00FF00"},
}
DEFAULT_COLOR = "#808080"


# ── 2. Model cache ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_crochet_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


# ── 3. SVG icon drawing (same as notebook) ──────────────────────────────────
def draw_svg_icon(ax, label, x, y, w, h, angle_deg, color):
    t = transforms.Affine2D().rotate_deg_around(x, y, angle_deg) + ax.transData

    if label == "chain":
        ax.add_patch(patches.Ellipse(
            (x, y), w * 0.8, h * 0.4,
            fill=False, color=color, linewidth=2, transform=t,
        ))
    elif label in ("half_double", "double", "double treble", "treble", "fan"):
        ax.plot([x, x], [y - h / 2, y + h / 2], color=color, lw=2, transform=t)
        ax.plot([x - w / 3, x + w / 3], [y - h / 2, y - h / 2], color=color, lw=2, transform=t)
        if label == "double":
            ax.plot([x - w / 4, x + w / 4], [y - h / 8, y + h / 8], color=color, lw=1.5, transform=t)
        elif label == "treble":
            ax.plot([x - w / 4, x + w / 4], [y - h / 4, y - h / 12], color=color, lw=1.5, transform=t)
            ax.plot([x - w / 4, x + w / 4], [y + h / 12, y + h / 4], color=color, lw=1.5, transform=t)
        elif label == "double treble":
            ax.plot([x - w / 4, x + w / 4], [y - h / 3, y - h / 6], color=color, lw=1.5, transform=t)
            ax.plot([x - w / 4, x + w / 4], [y - h / 12, y + h / 12], color=color, lw=1.5, transform=t)
            ax.plot([x - w / 4, x + w / 4], [y + h / 6, y + h / 3], color=color, lw=1.5, transform=t)
        elif label == "fan":
            ax.plot([x, x - w / 3], [y + h / 2, y - h / 2], color=color, lw=1.5, transform=t, alpha=0.6)
            ax.plot([x, x + w / 3], [y + h / 2, y - h / 2], color=color, lw=1.5, transform=t, alpha=0.6)
    elif label == "enseble_chain":
        n_ovals = max(2, int(h / (w * 0.5)))
        oval_h = h / n_ovals
        for ci in range(n_ovals):
            cy = y - h / 2 + oval_h * (ci + 0.5)
            ax.add_patch(patches.Ellipse(
                (x, cy), w * 0.6, oval_h * 0.7,
                fill=False, color=color, linewidth=1.5, transform=t,
            ))
    elif label == "noise":
        ax.add_patch(patches.Circle(
            (x, y), min(w, h) * 0.3,
            fill=False, color=color, linewidth=1.5, transform=t,
        ))
    elif label == "single":
        ax.plot([x - w / 3, x + w / 3], [y - h / 3, y + h / 3], color=color, lw=2, transform=t)
        ax.plot([x - w / 3, x + w / 3], [y + h / 3, y - h / 3], color=color, lw=2, transform=t)
    else:
        ax.plot([x - w / 3, x + w / 3], [y - h / 3, y + h / 3], color=color, lw=2, transform=t)
        ax.plot([x - w / 3, x + w / 3], [y + h / 3, y - h / 3], color=color, lw=2, transform=t)


# ── 4. Adaptive tiling with progress callbacks ──────────────────────────────
def predict_adaptive_with_progress(
    model,
    image: np.ndarray,
    target_stitch_px: int = 100,
    tile_size: int = 640,
    overlap: float = 0.2,
    conf: float = 0.25,
    iou_threshold: float = 0.5,
    on_phase=None,       # callable(phase_label: str)
    on_tile=None,        # callable(idx: int, total: int, x1: int, y1: int, n_tile_dets: int)
) -> tuple[list[Detection], dict]:
    """Drop-in replacement for ``util.tiler.predict_adaptive`` that fires
    progress callbacks so we can reflect the work in the UI.

    Returns ``(detections, info)`` where ``info`` has diagnostic fields
    like the estimated stitch size and effective tile size.
    """
    info: dict = {}
    h, w = image.shape[:2]

    if on_phase:
        on_phase("Estimating stitch size (max of short axes)…")
    estimated_h = estimate_stitch_size(model, image, tile_size=tile_size,
                                       conf=min(conf, 0.15))
    info["estimated_stitch_height_px"] = estimated_h

    if estimated_h is None or estimated_h <= 0:
        effective_tile = tile_size
        info["note"] = "No detections on size-estimation pass — using fixed tiling."
    else:
        effective_tile = int(tile_size * estimated_h / target_stitch_px)
        effective_tile = max(effective_tile, 64)

    info["effective_tile_px"] = effective_tile

    # Single-shot path
    if effective_tile >= max(h, w):
        if on_phase:
            on_phase(f"Single-shot inference (effective tile {effective_tile}px ≥ image)")
        dets = _infer_tile(model, image, 0, 0, conf)
        if on_tile:
            on_tile(1, 1, 0, 0, len(dets))
        info["tiles"] = 1
        return dets, info

    # Tiled path
    stride = max(1, int(effective_tile * (1 - overlap)))
    xs = _tile_starts(w, effective_tile, stride)
    ys = _tile_starts(h, effective_tile, stride)
    total = len(xs) * len(ys)
    info["tiles"] = total

    if on_phase:
        on_phase(
            f"Tiling: {total} tiles at {effective_tile}px "
            f"(stitch≈{estimated_h:.1f}px → target {target_stitch_px}px)"
        )

    all_dets: list[Detection] = []
    idx = 0
    for y1 in ys:
        for x1 in xs:
            idx += 1
            tile = image[y1:min(y1 + effective_tile, h),
                         x1:min(x1 + effective_tile, w)]
            tile_dets = _infer_tile(model, tile, x1, y1, conf)
            all_dets.extend(tile_dets)
            if on_tile:
                on_tile(idx, total, x1, y1, len(tile_dets))

    if on_phase:
        on_phase(f"Running NMS on {len(all_dets)} raw detections…")
    merged = _nms(all_dets, iou_threshold)
    info["raw_detections"] = len(all_dets)
    info["merged_detections"] = len(merged)
    return merged, info


# ── 5. Reconstruction figure ────────────────────────────────────────────────
def render_reconstruction(image: np.ndarray, detections: list[Detection]) -> Image.Image:
    h, w = image.shape[:2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    ax1.set_title("YOLO OBB Detections (adaptive tiling)")

    ax2.set_facecolor("white")
    ax2.set_xlim(0, w)
    ax2.set_ylim(h, 0)
    ax2.set_title("Reconstructed SVG Scheme")

    for det in detections:
        corners = det.corners
        config = CLASS_CONFIG.get(det.cls_name, {"abbr": "??", "color": DEFAULT_COLOR})
        color, abbr = config["color"], config["abbr"]

        closed = np.vstack([corners, corners[0]])
        ax1.plot(closed[:, 0], closed[:, 1], color=color, linewidth=2)
        ax1.text(
            corners[0, 0], corners[0, 1] - 5, abbr,
            color="white", fontsize=8, fontweight="bold",
            bbox=dict(facecolor=color, edgecolor="none", alpha=0.7),
        )

        center = corners.mean(axis=0)
        width = float(np.linalg.norm(corners[0] - corners[1]))
        height = float(np.linalg.norm(corners[0] - corners[3]))
        angle = float(np.degrees(np.arctan2(
            corners[1, 1] - corners[0, 1],
            corners[1, 0] - corners[0, 0],
        )))
        draw_svg_icon(ax2, det.cls_name, center[0], center[1], width, height, angle, color)

    ax1.axis("off")
    ax2.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# ── 6. JSON export ──────────────────────────────────────────────────────────
def detections_to_json(image: np.ndarray, detections: list[Detection],
                       info: dict, source: dict) -> dict:
    h, w = image.shape[:2]
    records = []
    for det in detections:
        corners = det.corners
        center = corners.mean(axis=0)
        width = float(np.linalg.norm(corners[0] - corners[1]))
        height = float(np.linalg.norm(corners[0] - corners[3]))
        angle = float(np.degrees(np.arctan2(
            corners[1, 1] - corners[0, 1],
            corners[1, 0] - corners[0, 0],
        )))
        records.append({
            "class": det.cls_name,
            "abbr": CLASS_CONFIG.get(det.cls_name, {}).get("abbr", "??"),
            "confidence": round(det.confidence, 4),
            "corners": [[float(x), float(y)] for x, y in corners.tolist()],
            "center": [float(center[0]), float(center[1])],
            "width": width,
            "height": height,
            "angle_deg": angle,
        })
    return {
        "source": source,
        "image_size": {"width": int(w), "height": int(h)},
        "num_detections": len(records),
        "inference": info,
        "detections": records,
    }


# ── 7. Streamlit UI ─────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Crochet Scheme Generator")
st.title("🧶 Crochet Symbol to Scheme")
st.caption(
    "Generate a crochet diagram with Gemini or upload your own, then run "
    "adaptive tiled YOLOv8n OBB inference."
)

# Session state used to keep the input image across button clicks.
if "input_image" not in st.session_state:
    st.session_state.input_image = None            # np.ndarray BGR
    st.session_state.input_source = None           # dict describing where it came from
    st.session_state.input_pil = None              # PIL for display

# --- Sidebar: settings ---
with st.sidebar:
    st.header("Inference settings")
    conf_threshold = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    iou_threshold = st.slider("NMS IoU threshold", 0.1, 0.9, 0.5, 0.05)
    target_stitch_px = st.slider("Target stitch size (px at 640)", 40, 200, 100, 10)
    tile_size = st.selectbox("YOLO tile size", [480, 640, 832, 1024], index=1)
    overlap = st.slider("Tile overlap", 0.0, 0.6, 0.2, 0.05)

    st.divider()
    st.header("Model")
    weights_path = st.text_input("Weights path", WEIGHTS_PATH)

# Load the model (cached) and surface errors clearly.
try:
    model = load_crochet_model(weights_path)
except Exception as exc:
    st.error(f"Could not load YOLO weights at `{weights_path}`: {exc}")
    st.stop()

# --- Input mode ---
input_mode = st.radio(
    "Input source",
    ("Generate with Gemini", "Upload an image"),
    horizontal=True,
)

if input_mode == "Generate with Gemini":
    if not GEMINI_AVAILABLE:
        st.warning(
            "The `google-genai` package isn't importable — the upload tab still works. "
            "Install it with `pip install google-genai` to enable generation."
        )

    prompt = st.text_area(
        "Prompt",
        value=GEMINI_DEFAULT_PROMPT,
        height=150,
        help="Describe the crochet diagram you want Gemini to produce.",
    )
    col_a, col_b = st.columns([3, 1])
    with col_a:
        gemini_model = st.text_input("Gemini model", GEMINI_DEFAULT_MODEL)
    with col_b:
        api_key = st.text_input(
            "API key (optional)",
            type="password",
            help="Falls back to GEMINI_API_KEY / GOOGLE_API_KEY env var.",
        )

    if st.button("🎨 Generate image", disabled=not GEMINI_AVAILABLE, type="primary"):
        with st.status("Generating image with Gemini…", expanded=True) as status:
            try:
                status.update(label="Submitting prompt to Gemini…", state="running")
                tmp_path = Path("free_output.png")
                t0 = time.time()
                generate_image(
                    prompt=prompt,
                    output_path=tmp_path,
                    model=gemini_model,
                    api_key=api_key or None,
                )
                elapsed = time.time() - t0

                pil = Image.open(tmp_path).convert("RGB")
                arr = np.array(pil)
                bgr = cv.cvtColor(arr, cv.COLOR_RGB2BGR)

                st.session_state.input_image = bgr
                st.session_state.input_pil = pil
                st.session_state.input_source = {
                    "kind": "gemini",
                    "model": gemini_model,
                    "prompt": prompt,
                    "path": str(tmp_path),
                    "seconds": round(elapsed, 2),
                }
                status.update(
                    label=f"✅ Image generated in {elapsed:.1f}s",
                    state="complete",
                )
            except Exception as exc:
                status.update(label=f"❌ Generation failed: {exc}", state="error")
                st.stop()

else:
    uploaded = st.file_uploader("Upload a crochet image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded is not None:
        data = np.frombuffer(uploaded.read(), dtype=np.uint8)
        bgr = cv.imdecode(data, cv.IMREAD_COLOR)
        if bgr is None:
            st.error("Could not decode the uploaded image.")
        else:
            st.session_state.input_image = bgr
            st.session_state.input_pil = Image.fromarray(cv.cvtColor(bgr, cv.COLOR_BGR2RGB))
            st.session_state.input_source = {
                "kind": "upload",
                "filename": uploaded.name,
                "bytes": int(uploaded.size) if hasattr(uploaded, "size") else None,
            }

# --- Main view: input + analyze ---
image_bgr: np.ndarray | None = st.session_state.input_image

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

# --- Inference pass ---
if run_clicked and image_bgr is not None:
    total_box = st.empty()
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
                target_stitch_px=target_stitch_px,
                tile_size=tile_size,
                overlap=overlap,
                conf=conf_threshold,
                iou_threshold=iou_threshold,
                on_phase=_on_phase,
                on_tile=_on_tile,
            )
            infer_seconds = time.time() - t0
            info["seconds"] = round(infer_seconds, 2)

            tile_bar.progress(1.0, text="Rendering reconstruction…")
            status.update(label="Rendering reconstruction…", state="running")
            recon_img = render_reconstruction(image_bgr, detections)

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
        st.image(recon_img, use_container_width=True)

        png_buf = io.BytesIO()
        recon_img.save(png_buf, format="PNG")
        st.download_button(
            "⬇️ Download scheme PNG",
            data=png_buf.getvalue(),
            file_name="crochet_scheme.png",
            mime="image/png",
        )

    st.subheader("Detections JSON")
    counts: dict[str, int] = {}
    for d in detections:
        counts[d.cls_name] = counts.get(d.cls_name, 0) + 1

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stitches", len(detections))
    m2.metric("Tiles", info.get("tiles", "—"))
    m3.metric("Effective tile px", info.get("effective_tile_px", "—"))
    est = info.get("estimated_stitch_height_px")
    m4.metric("Est. stitch px", f"{est:.1f}" if isinstance(est, (int, float)) else "—")

    if counts:
        st.write("**Per-class counts**")
        st.json(dict(sorted(counts.items(), key=lambda kv: -kv[1])))

    json_bytes = json.dumps(payload, indent=2).encode("utf-8")
    st.download_button(
        "⬇️ Download detections JSON",
        data=json_bytes,
        file_name="crochet_scheme.json",
        mime="application/json",
    )
    with st.expander("Preview JSON"):
        st.json(payload)
