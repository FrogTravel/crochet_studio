"""Shared UI helpers used by both the public Streamlit app (``app.py``)
and the admin demo page (``pages/1_Admin_Demo.py``).

Keeps the two pages in lock-step: model loader, class config, SVG icon
drawing, adaptive tiled inference with progress callbacks, scheme
rendering, and JSON serialization all live here.
"""

from __future__ import annotations

import io
import math
import time
from pathlib import Path
from typing import Any, Callable

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


# ── Configuration ───────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = "runs/obb/obb_train23/weights/best.pt"

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
SCHEME_COLOR = "#000000"
ICON_SHRINK = 0.85


# ── Gemini wrapper (optional dependency) ────────────────────────────────────
try:
    from pipeline.generate_image import (
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


# ── Model cache ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_crochet_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


# ── Image decoding helpers ──────────────────────────────────────────────────
def decode_uploaded_image(uploaded_file) -> np.ndarray | None:
    """Decode a Streamlit ``UploadedFile`` to a BGR numpy image."""
    if uploaded_file is None:
        return None
    data = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    return cv.imdecode(data, cv.IMREAD_COLOR)


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv.cvtColor(bgr, cv.COLOR_BGR2RGB))


# ── SVG icon drawing ────────────────────────────────────────────────────────
def draw_svg_icon(ax, label, x, y, w, h, angle_deg, color, lw: float = 2.0) -> None:
    """Draw one crochet symbol at pose ``(x, y, angle_deg)``.

    ``lw`` controls the main stroke thickness; secondary accents are
    drawn at 0.75 × lw to preserve visual hierarchy.
    """
    t = transforms.Affine2D().rotate_deg_around(x, y, angle_deg) + ax.transData
    lw_sub = lw * 0.75

    if label == "chain":
        ax.add_patch(patches.Ellipse(
            (x, y), w * 0.8, h * 0.4,
            fill=False, color=color, linewidth=lw, transform=t,
        ))
    elif label in ("half_double", "double", "double treble", "treble", "fan"):
        ax.plot([x, x], [y - h / 2, y + h / 2], color=color, lw=lw, transform=t)
        ax.plot([x - w / 3, x + w / 3], [y - h / 2, y - h / 2], color=color, lw=lw, transform=t)
        if label == "double":
            ax.plot([x - w / 4, x + w / 4], [y - h / 8, y + h / 8], color=color, lw=lw_sub, transform=t)
        elif label == "treble":
            ax.plot([x - w / 4, x + w / 4], [y - h / 4, y - h / 12], color=color, lw=lw_sub, transform=t)
            ax.plot([x - w / 4, x + w / 4], [y + h / 12, y + h / 4], color=color, lw=lw_sub, transform=t)
        elif label == "double treble":
            ax.plot([x - w / 4, x + w / 4], [y - h / 3, y - h / 6], color=color, lw=lw_sub, transform=t)
            ax.plot([x - w / 4, x + w / 4], [y - h / 12, y + h / 12], color=color, lw=lw_sub, transform=t)
            ax.plot([x - w / 4, x + w / 4], [y + h / 6, y + h / 3], color=color, lw=lw_sub, transform=t)
        elif label == "fan":
            ax.plot([x, x - w / 3], [y + h / 2, y - h / 2], color=color, lw=lw_sub, transform=t, alpha=0.6)
            ax.plot([x, x + w / 3], [y + h / 2, y - h / 2], color=color, lw=lw_sub, transform=t, alpha=0.6)
    elif label == "enseble_chain":
        n_ovals = max(2, int(h / (w * 0.5)))
        oval_h = h / n_ovals
        for ci in range(n_ovals):
            cy = y - h / 2 + oval_h * (ci + 0.5)
            ax.add_patch(patches.Ellipse(
                (x, cy), w * 0.6, oval_h * 0.7,
                fill=False, color=color, linewidth=lw_sub, transform=t,
            ))
    elif label == "noise":
        # Rendered invisibly — noise boxes aren't meaningful stitches.
        return
    elif label == "single":
        ax.plot([x - w / 3, x + w / 3], [y - h / 3, y + h / 3], color=color, lw=lw, transform=t)
        ax.plot([x - w / 3, x + w / 3], [y + h / 3, y - h / 3], color=color, lw=lw, transform=t)
    else:
        ax.plot([x - w / 3, x + w / 3], [y - h / 3, y + h / 3], color=color, lw=lw, transform=t)
        ax.plot([x - w / 3, x + w / 3], [y + h / 3, y - h / 3], color=color, lw=lw, transform=t)


def linewidth_for_count(n: int) -> float:
    """Map detection count to drawing linewidth (points).

    Few stitches → thick lines (up to ~4pt).
    Many stitches → thin lines (down to ~0.4pt).
    """
    if n <= 1:
        return 4.0
    lw = 10.0 / math.sqrt(n)
    return max(0.4, min(4.0, lw))


# ── Rendering helpers ───────────────────────────────────────────────────────
# Minimum output height (in pixels) we always want to render at, regardless
# of how small the input is — keeps thumbnails readable.
MIN_RENDER_HEIGHT_PX = 1200
# Cap so we don't allocate huge figures for enormous inputs.
MAX_RENDER_HEIGHT_PX = 4000


def fig_to_pil(fig, pad_inches: float = 0.05, dpi: int | None = None) -> Image.Image:
    buf = io.BytesIO()
    kwargs = {"format": "png", "bbox_inches": "tight",
              "pad_inches": pad_inches, "facecolor": "white"}
    if dpi is not None:
        kwargs["dpi"] = dpi
    fig.savefig(buf, **kwargs)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def _figsize_and_dpi(image: np.ndarray, panel_h_in: float = 8.0,
                     max_h_in: float = 16.0) -> tuple[tuple[float, float], int]:
    """Pick a matplotlib figsize and dpi so the saved PNG has at least as
    many pixels as the source image (clamped by MAX_RENDER_HEIGHT_PX).

    Returns ``((w_inches, h_inches), dpi)``.
    """
    h, w = image.shape[:2]
    aspect = w / max(h, 1)
    # Grow the figure for very tall inputs so lines don't become hair-thin.
    h_in = min(max(panel_h_in, h / 200.0), max_h_in)
    w_in = max(h_in * aspect, 1.0)
    target_h_px = max(MIN_RENDER_HEIGHT_PX, min(h, MAX_RENDER_HEIGHT_PX))
    dpi = int(max(100, round(target_h_px / h_in)))
    return (w_in, h_in), dpi


def draw_scheme(ax, image: np.ndarray, detections: list[Detection],
                shrink: float = ICON_SHRINK,
                color: str = SCHEME_COLOR,
                lw: float | None = None) -> None:
    """Draw the reconstructed scheme into ``ax`` in input-image pixel coords."""
    h, w = image.shape[:2]
    ax.set_facecolor("white")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")  # keep pixel aspect so icons don't distort

    effective_lw = lw if lw is not None else linewidth_for_count(len(detections))

    for det in detections:
        corners = det.corners
        center = corners.mean(axis=0)
        width = float(np.linalg.norm(corners[0] - corners[1])) * shrink
        height = float(np.linalg.norm(corners[0] - corners[3])) * shrink
        angle = float(np.degrees(np.arctan2(
            corners[1, 1] - corners[0, 1],
            corners[1, 0] - corners[0, 0],
        )))
        draw_svg_icon(ax, det.cls_name, center[0], center[1], width, height,
                      angle, color, lw=effective_lw)
    ax.axis("off")


def _auto_detection_linewidth(n: int) -> float:
    """Box stroke width that scales down as the detection count grows."""
    if n <= 1:
        return 2.0
    lw = 8.0 / math.sqrt(n)
    return max(0.25, min(2.0, lw))


def draw_detection_overlay(ax, image: np.ndarray, detections: list[Detection],
                           show_labels: bool | str = "auto",
                           lw: float | None = None,
                           label_threshold: int = 40,
                           label_fontsize: float | None = None,
                           alpha: float = 0.85) -> None:
    """Overlay coloured OBB polygons on the input image.

    - ``lw=None`` auto-scales with detection count (many detections →
      thin strokes, so the underlying image is still visible).
    - ``show_labels="auto"`` shows per-box labels only when there are
      ≤ ``label_threshold`` detections. Pass ``True`` / ``False`` to
      force the behaviour.
    """
    h, w = image.shape[:2]
    n = len(detections)

    ax.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")

    effective_lw = lw if lw is not None else _auto_detection_linewidth(n)

    if show_labels == "auto":
        draw_labels = n <= label_threshold
    else:
        draw_labels = bool(show_labels)

    if label_fontsize is None:
        label_fontsize = max(5.0, min(8.0, 40.0 / max(math.sqrt(max(n, 1)), 1.0)))

    for det in detections:
        corners = det.corners
        config = CLASS_CONFIG.get(det.cls_name, {"abbr": "??", "color": DEFAULT_COLOR})
        color, abbr = config["color"], config["abbr"]
        closed = np.vstack([corners, corners[0]])
        ax.plot(closed[:, 0], closed[:, 1], color=color,
                linewidth=effective_lw, alpha=alpha)
        if draw_labels:
            ax.text(
                corners[0, 0], corners[0, 1] - 5, abbr,
                color="white", fontsize=label_fontsize, fontweight="bold",
                bbox=dict(facecolor=color, edgecolor="none", alpha=0.75, pad=1),
            )
    ax.axis("off")


def render_scheme_image(image: np.ndarray,
                        detections: list[Detection]) -> Image.Image:
    """Standalone scheme PNG at the input's exact aspect ratio.

    The output is rendered at a dpi high enough that the saved PNG has
    at least as many pixels as the source image (clamped by
    MAX_RENDER_HEIGHT_PX) — so large inputs don't lose fidelity.
    """
    figsize, dpi = _figsize_and_dpi(image)
    fig, ax = plt.subplots(figsize=figsize)
    draw_scheme(ax, image, detections)
    plt.tight_layout(pad=0)
    return fig_to_pil(fig, pad_inches=0.0, dpi=dpi)


def render_detection_image(image: np.ndarray,
                           detections: list[Detection],
                           show_labels: bool | str = "auto",
                           lw: float | None = None,
                           label_threshold: int = 40) -> Image.Image:
    """Input image with OBB boxes drawn on top (for the admin view).

    Line thickness and label visibility auto-scale with detection count
    so dense images stay readable.
    """
    figsize, dpi = _figsize_and_dpi(image)
    fig, ax = plt.subplots(figsize=figsize)
    draw_detection_overlay(ax, image, detections,
                           show_labels=show_labels, lw=lw,
                           label_threshold=label_threshold)
    plt.tight_layout(pad=0)
    return fig_to_pil(fig, pad_inches=0.0, dpi=dpi)


def render_tile_grid_image(image: np.ndarray,
                           tile_info: list[tuple[int, int, int, int]]) -> Image.Image:
    """Draw the tile-grid overlay. ``tile_info`` is ``[(x1, y1, tw, th), …]``."""
    h, w = image.shape[:2]
    figsize, dpi = _figsize_and_dpi(image)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    # Label font scales down as the grid gets dense.
    n = max(len(tile_info), 1)
    label_fs = max(6.0, min(11.0, 40.0 / max(math.sqrt(n), 1.0)))
    for i, (x1, y1, tw, th) in enumerate(tile_info):
        ax.add_patch(patches.Rectangle(
            (x1, y1), tw, th,
            linewidth=1.2, edgecolor="#ffa500",
            facecolor="#ffa500", alpha=0.10,
        ))
        ax.add_patch(patches.Rectangle(
            (x1, y1), tw, th,
            linewidth=1.2, edgecolor="#ffa500", facecolor="none",
        ))
        ax.text(
            x1 + tw / 2, y1 + th / 2, str(i),
            color="#ffa500", fontsize=label_fs, ha="center", va="center",
            fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1),
        )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig_to_pil(fig, pad_inches=0.0, dpi=dpi)


# ── Adaptive inference with progress + optional capture ─────────────────────
def predict_adaptive_with_progress(
    model,
    image: np.ndarray,
    target_stitch_px: int = 100,
    tile_size: int = 640,
    overlap: float = 0.2,
    conf: float = 0.25,
    iou_threshold: float = 0.5,
    on_phase: Callable[[str], None] | None = None,
    on_tile: Callable[[int, int, int, int, int], None] | None = None,
    capture_tiles: bool = False,
    capture_raw: bool = False,
) -> tuple[list[Detection], dict]:
    """Adaptive tiled inference, with streaming callbacks and optional capture.

    Callbacks:
      * ``on_phase(label)`` — phase transitions ("Estimating…", "Tiling…", "NMS…").
      * ``on_tile(idx, total, x1, y1, n)`` — fires once per tile.

    Capture flags (useful for admin/demo visualisations):
      * ``capture_tiles=True`` populates ``info["tile_records"]`` with
        ``[(x1, y1, tile_bgr, tile_detections), …]`` per tile.
      * ``capture_raw=True`` populates ``info["raw_det_list"]`` with all
        detections before NMS (so the admin view can diff before/after).

    Returns ``(detections, info)``.
    """
    info: dict[str, Any] = {}
    h, w = image.shape[:2]

    if on_phase:
        on_phase("Estimating stitch size (max of short axes)…")
    t0 = time.time()
    estimated_h = estimate_stitch_size(model, image, tile_size=tile_size,
                                       conf=min(conf, 0.15))
    info["estimate_seconds"] = round(time.time() - t0, 3)
    info["estimated_stitch_height_px"] = estimated_h

    if estimated_h is None or estimated_h <= 0:
        effective_tile = tile_size
        info["note"] = "No detections on size-estimation pass — falling back to fixed tiling."
    else:
        effective_tile = int(tile_size * estimated_h / target_stitch_px)
        effective_tile = max(effective_tile, 64)

    info["effective_tile_px"] = effective_tile
    info["target_stitch_px"] = target_stitch_px
    info["tile_size"] = tile_size
    info["overlap"] = overlap

    tile_records: list[tuple[int, int, np.ndarray, list[Detection]]] = []

    # Single-shot branch
    if effective_tile >= max(h, w):
        if on_phase:
            on_phase(
                f"Single-shot inference — effective tile {effective_tile}px ≥ image "
                f"{w}×{h}."
            )
        t1 = time.time()
        dets = _infer_tile(model, image, 0, 0, conf)
        info["infer_seconds"] = round(time.time() - t1, 3)
        if on_tile:
            on_tile(1, 1, 0, 0, len(dets))
        info["tiles"] = 1
        info["tile_geometry"] = [(0, 0, w, h)]
        if capture_tiles:
            tile_records.append((0, 0, image.copy(), list(dets)))
            info["tile_records"] = tile_records
        info["raw_detections"] = len(dets)
        info["merged_detections"] = len(dets)
        info["single_shot"] = True
        return dets, info

    # Tiled branch
    stride = max(1, int(effective_tile * (1 - overlap)))
    xs = _tile_starts(w, effective_tile, stride)
    ys = _tile_starts(h, effective_tile, stride)
    total = len(xs) * len(ys)
    info["tiles"] = total
    info["stride_px"] = stride
    info["single_shot"] = False

    if on_phase:
        on_phase(
            f"Tiling: {total} tiles at {effective_tile}px "
            f"(stitch≈{estimated_h:.1f}px → target {target_stitch_px}px)."
        )

    tile_geometry: list[tuple[int, int, int, int]] = []
    all_dets: list[Detection] = []
    idx = 0
    t_tiles = time.time()
    for y1 in ys:
        for x1 in xs:
            idx += 1
            x2 = min(x1 + effective_tile, w)
            y2 = min(y1 + effective_tile, h)
            tile = image[y1:y2, x1:x2]
            tile_dets = _infer_tile(model, tile, x1, y1, conf)
            all_dets.extend(tile_dets)
            tile_geometry.append((x1, y1, x2 - x1, y2 - y1))
            if capture_tiles:
                tile_records.append((x1, y1, tile.copy(), list(tile_dets)))
            if on_tile:
                on_tile(idx, total, x1, y1, len(tile_dets))
    info["infer_seconds"] = round(time.time() - t_tiles, 3)
    info["tile_geometry"] = tile_geometry
    if capture_tiles:
        info["tile_records"] = tile_records
    if capture_raw:
        # Clone list so later NMS doesn't mutate it.
        info["raw_det_list"] = list(all_dets)

    if on_phase:
        on_phase(f"Running NMS on {len(all_dets)} raw detections (IoU≥{iou_threshold})…")
    t_nms = time.time()
    merged = _nms(all_dets, iou_threshold)
    info["nms_seconds"] = round(time.time() - t_nms, 3)
    info["raw_detections"] = len(all_dets)
    info["merged_detections"] = len(merged)
    info["suppressed_detections"] = len(all_dets) - len(merged)
    return merged, info


# ── JSON export ─────────────────────────────────────────────────────────────
def detections_to_json(image: np.ndarray, detections: list[Detection],
                       info: dict, source: dict) -> dict:
    h, w = image.shape[:2]
    records: list[dict[str, Any]] = []
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

    # Strip heavy capture-only fields that don't belong in JSON exports.
    clean_info = {k: v for k, v in info.items()
                  if k not in ("tile_records", "raw_det_list", "tile_geometry")}

    return {
        "source": source,
        "image_size": {"width": int(w), "height": int(h)},
        "num_detections": len(records),
        "inference": clean_info,
        "detections": records,
    }


def class_counts(detections: list[Detection]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in detections:
        counts[d.cls_name] = counts.get(d.cls_name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# ── Session-state helpers ───────────────────────────────────────────────────
def ensure_input_state(keys: tuple[str, ...] = ("input_image", "input_pil", "input_source")) -> None:
    for k in keys:
        if k not in st.session_state:
            st.session_state[k] = None


def store_image(bgr: np.ndarray, source: dict) -> None:
    st.session_state.input_image = bgr
    st.session_state.input_pil = bgr_to_pil(bgr)
    st.session_state.input_source = source


# ── Streamlit sidebar / input blocks (reusable across pages) ────────────────
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
        "conf": conf,
        "iou": iou,
        "target_stitch_px": target_stitch_px,
        "tile_size": tile_size,
        "overlap": overlap,
        "weights_path": weights_path,
    }


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
                    "kind": "upload",
                    "filename": uploaded.name,
                    "bytes": int(uploaded.size) if hasattr(uploaded, "size") else None,
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
                "kind": "gemini",
                "model": model_name,
                "prompt": prompt,
                "path": str(tmp_path),
                "seconds": round(elapsed, 2),
            })
            status.update(
                label=f"✅ Image generated in {elapsed:.1f}s",
                state="complete",
            )
        except Exception as exc:
            status.update(label=f"❌ Generation failed: {exc}", state="error")


__all__ = [
    # configuration
    "DEFAULT_WEIGHTS", "CLASS_CONFIG", "DEFAULT_COLOR", "SCHEME_COLOR", "ICON_SHRINK",
    "GEMINI_AVAILABLE", "GEMINI_DEFAULT_PROMPT", "GEMINI_DEFAULT_MODEL",
    # detection type
    "Detection",
    # model
    "load_crochet_model",
    # drawing
    "draw_svg_icon", "linewidth_for_count",
    "draw_scheme", "draw_detection_overlay",
    "render_scheme_image", "render_detection_image", "render_tile_grid_image",
    "fig_to_pil",
    # inference
    "predict_adaptive_with_progress",
    # helpers
    "decode_uploaded_image", "bgr_to_pil",
    "detections_to_json", "class_counts",
    "ensure_input_state", "store_image",
    "sidebar_inference_controls", "input_source_block",
]
