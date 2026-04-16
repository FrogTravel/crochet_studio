"""Admin / presentation page — walks through every pipeline stage.

Every box below maps to one step of the inference pipeline, shown in
order. Useful for internal debugging and for demoing the system end to
end to stakeholders.

Run the multi-page app from the repo root::

    streamlit run app.py

Then pick "Admin Demo" in the sidebar.
"""

from __future__ import annotations

import io
import json
import time

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from crochet.config import CLASS_CONFIG, DEFAULT_COLOR
from crochet.detection import predict_adaptive_with_progress
from crochet.json_export import class_counts, detections_to_json
from crochet.rendering import (
    fig_to_pil,
    linewidth_for_count,
    render_detection_image,
    render_scheme_image,
    render_tile_grid_image,
)
from crochet.streamlit_ui import (
    ensure_input_state,
    input_source_block,
    load_crochet_model,
    sidebar_inference_controls,
)


st.set_page_config(layout="wide", page_title="Admin Demo — Crochet Pipeline")
st.title("🛠️ Admin Demo — Pipeline Walkthrough")
st.caption(
    "Every stage is shown explicitly: input → size estimation → tile plan "
    "→ per-tile inference → NMS → reconstruction → JSON."
)

ensure_input_state()
settings = sidebar_inference_controls(defaults={"conf": 0.20})

with st.sidebar:
    st.divider()
    st.header("Demo options")
    show_tile_gallery = st.checkbox("Show per-tile gallery", value=True)
    max_tiles_in_gallery = st.slider("Max tiles in gallery", 1, 48, 16)

try:
    model = load_crochet_model(settings["weights_path"])
except Exception as exc:
    st.error(f"Could not load YOLO weights at `{settings['weights_path']}`: {exc}")
    st.stop()

st.markdown("## Step 1 — Input image")
st.markdown(
    "The pipeline accepts either a Gemini-generated crochet diagram or a "
    "user-uploaded photo. Both produce the same BGR numpy array downstream."
)
input_source_block(mode_key="admin_input_mode")

image_bgr = st.session_state.input_image
if image_bgr is None:
    st.info("Provide an image above to walk through the pipeline.")
    st.stop()

h, w = image_bgr.shape[:2]
col_img, col_meta = st.columns([3, 1])
with col_img:
    st.image(st.session_state.input_pil, use_container_width=True)
with col_meta:
    st.metric("Width (px)", w)
    st.metric("Height (px)", h)
    st.metric("Aspect", f"{w / max(h, 1):.3f}")
    st.write("**Source**")
    st.json(st.session_state.input_source or {})

if not st.button("🚀 Run full pipeline", type="primary"):
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Size-estimation pass
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("## Step 2 — Size-estimation pass (downsampled)")
st.markdown(
    "We downsample the image so its longest side matches the YOLO tile size, "
    "run a low-confidence pass, and take the **median of each OBB's long "
    "axis** (the stem length for T-shaped stitches). This answers *how tall "
    "is a typical stitch in the image?* — if that already lands near the "
    "target, tiling is skipped entirely."
)

tile_size = settings["tile_size"]
scale = tile_size / max(h, w)
if scale < 1.0:
    small_bgr = cv.resize(image_bgr, (int(w * scale), int(h * scale)))
else:
    small_bgr = image_bgr
    scale = 1.0

with st.spinner("Running low-confidence estimation pass…"):
    t0 = time.time()
    estim_res = model.predict(small_bgr, conf=min(settings["conf"], 0.15),
                              verbose=False)[0]
    estim_seconds = time.time() - t0

long_axes: list[float] = []
if estim_res.obb is not None:
    for box in estim_res.obb:
        corners = box.xyxyxyxy.cpu().numpy().reshape(4, 2)
        side_a = float(np.linalg.norm(corners[0] - corners[1]))
        side_b = float(np.linalg.norm(corners[1] - corners[2]))
        long_axes.append(max(side_a, side_b))

median_long_downsampled = float(np.median(long_axes)) if long_axes else 0.0
max_long_downsampled = max(long_axes) if long_axes else 0.0
estimated_h_px = median_long_downsampled / scale if scale > 0 else None

col_a, col_b = st.columns([2, 1])
with col_a:
    fig, ax = plt.subplots(
        figsize=(8, 8 * (small_bgr.shape[0] / max(small_bgr.shape[1], 1))),
    )
    ax.imshow(cv.cvtColor(small_bgr, cv.COLOR_BGR2RGB))
    if estim_res.obb is not None:
        for box in estim_res.obb:
            corners = box.xyxyxyxy.cpu().numpy().reshape(4, 2)
            cls_id = int(box.cls[0])
            cls_name = estim_res.names[cls_id]
            cfg = CLASS_CONFIG.get(cls_name, {"color": DEFAULT_COLOR})
            closed = np.vstack([corners, corners[0]])
            ax.plot(closed[:, 0], closed[:, 1], color=cfg["color"], lw=1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout(pad=0)
    st.image(
        fig_to_pil(fig, pad_inches=0.0), use_container_width=True,
        caption=f"Downsampled {small_bgr.shape[1]}×{small_bgr.shape[0]} "
                f"(scale {scale:.3f}) — {len(long_axes)} low-conf detections",
    )

with col_b:
    st.metric("Estimate seconds", f"{estim_seconds:.2f}")
    st.metric("Low-conf detections", len(long_axes))
    st.metric(
        "Median long-axis (downsampled)",
        f"{median_long_downsampled:.1f}px" if long_axes else "—",
    )
    st.metric(
        "→ estimated stitch height",
        f"{estimated_h_px:.1f}px" if estimated_h_px else "—",
    )

if long_axes:
    st.markdown("**Distribution of long-axis lengths (downsampled pixels)**")
    fig_h, ax_h = plt.subplots(figsize=(8, 2.5))
    ax_h.hist(long_axes, bins=20, color="#4c72b0", edgecolor="white")
    ax_h.axvline(median_long_downsampled, color="#55a868", linewidth=2,
                 label=f"median = {median_long_downsampled:.1f}px (used)")
    if len(long_axes) > 1:
        ax_h.axvline(max_long_downsampled, color="#dd8452", linewidth=2,
                     linestyle="--",
                     label=f"max = {max_long_downsampled:.1f}px")
    ax_h.set_xlabel("Long-axis length (px, downsampled)")
    ax_h.set_ylabel("Count")
    ax_h.legend()
    st.pyplot(fig_h)
    plt.close(fig_h)


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 — Tile plan
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("## Step 3 — Tile plan")
st.markdown(
    "From the estimated stitch size we compute the **effective tile size** "
    "in *original-image pixels* so that stitches appear at "
    f"`target_stitch_px = {settings['target_stitch_px']}` inside each YOLO "
    "input. If the effective tile already covers the whole image, we do a "
    "single-shot forward pass. Otherwise we lay out a grid with the "
    f"configured overlap ({settings['overlap']:.0%})."
)

if estimated_h_px and estimated_h_px > 0:
    effective_tile = max(
        int(settings["tile_size"] * estimated_h_px / settings["target_stitch_px"]),
        64,
    )
else:
    effective_tile = settings["tile_size"]

single_shot = effective_tile >= max(h, w)

m1, m2, m3, m4 = st.columns(4)
m1.metric("tile_size (YOLO input)", settings["tile_size"])
m2.metric("target_stitch_px", settings["target_stitch_px"])
m3.metric("effective tile (px in src)", f"{effective_tile}")
m4.metric("Mode", "single-shot" if single_shot else "tiled")

st.code(
    f"effective_tile = {settings['tile_size']} × "
    f"{estimated_h_px or 0:.1f} / {settings['target_stitch_px']} = "
    f"{effective_tile}  "
    f"{'≥' if single_shot else '<'} max(h, w) = {max(h, w)}",
    language="text",
)


# ═══════════════════════════════════════════════════════════════════════════
# Step 4 — Per-tile inference (runs the real pipeline with capture)
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("## Step 4 — Per-tile inference")
st.markdown(
    "Each tile is forwarded through the model at `tile_size` resolution. "
    "Detections are translated back into original-image coordinates; we keep "
    "them per-tile here so you can inspect any individual crop."
)

phase_log = st.empty()
tile_bar = st.progress(0.0, text="Preparing inference…")
phase_lines: list[str] = []


def _on_phase(msg: str) -> None:
    phase_lines.append(f"• {msg}")
    phase_log.markdown("\n".join(phase_lines))


def _on_tile(idx: int, total: int, x1: int, y1: int, n: int) -> None:
    frac = idx / max(total, 1)
    tile_bar.progress(frac, text=f"Tile {idx}/{total} @ ({x1},{y1}) — {n} dets")


with st.status("Running adaptive inference…", expanded=True) as status:
    t0 = time.time()
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
        capture_tiles=True,
        capture_raw=True,
    )
    total_seconds = time.time() - t0
    info["seconds"] = round(total_seconds, 2)
    tile_bar.progress(1.0, text="Inference complete")
    status.update(
        label=f"✅ {len(detections)} stitches after NMS in {total_seconds:.1f}s",
        state="complete",
    )

# Tile grid overlay
if info.get("tile_geometry"):
    st.markdown("**Tile grid on the input**")
    grid_img = render_tile_grid_image(image_bgr, info["tile_geometry"])
    st.image(grid_img, use_container_width=True)

# Per-tile gallery
tile_records = info.get("tile_records") or []
if tile_records and show_tile_gallery:
    shown = tile_records[: int(max_tiles_in_gallery)]
    hidden = len(tile_records) - len(shown)
    st.markdown(
        f"**Individual tiles ({len(shown)} of {len(tile_records)} shown"
        + (f"; {hidden} hidden — raise 'Max tiles in gallery' to see more" if hidden else "")
        + ")**"
    )
    cols_per_row = 4
    for row_start in range(0, len(shown), cols_per_row):
        row = shown[row_start:row_start + cols_per_row]
        cols = st.columns(len(row))
        for col, (x1, y1, tile_bgr, tile_dets) in zip(cols, row):
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(cv.cvtColor(tile_bgr, cv.COLOR_BGR2RGB))
            for det in tile_dets:
                corners = det.corners.copy()
                corners[:, 0] -= x1
                corners[:, 1] -= y1
                cfg = CLASS_CONFIG.get(
                    det.cls_name, {"abbr": "??", "color": DEFAULT_COLOR},
                )
                closed = np.vstack([corners, corners[0]])
                ax.plot(closed[:, 0], closed[:, 1], color=cfg["color"], lw=1.5)
            ax.set_aspect("equal")
            ax.axis("off")
            plt.tight_layout(pad=0)
            col.image(fig_to_pil(fig, pad_inches=0.0),
                      caption=f"origin ({x1}, {y1}) — {len(tile_dets)} dets",
                      use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# Step 5 — NMS merge
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("## Step 5 — Non-maximum suppression")
st.markdown(
    "Detections from overlapping tiles are deduplicated class-by-class. We "
    "keep the highest-confidence box whenever two boxes of the same class "
    f"have IoU ≥ `{settings['iou']}`."
)

raw_list = info.get("raw_det_list", [])
m1, m2, m3 = st.columns(3)
m1.metric("Raw detections (pre-NMS)", info.get("raw_detections", len(raw_list)))
m2.metric("Kept after NMS", info.get("merged_detections", len(detections)))
m3.metric("Suppressed", info.get("suppressed_detections", "—"))

if raw_list:
    col_pre, col_post = st.columns(2)
    with col_pre:
        st.markdown("**Before NMS** (includes duplicates at tile seams)")
        st.image(render_detection_image(image_bgr, raw_list),
                 use_container_width=True)
    with col_post:
        st.markdown("**After NMS**")
        st.image(render_detection_image(image_bgr, detections),
                 use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# Step 6 — Reconstruction
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("## Step 6 — Scheme reconstruction")
st.markdown(
    f"Each detection is redrawn as its matching crochet symbol in pure "
    f"black. Icon size follows the detection's own OBB; line thickness is "
    f"**`linewidth_for_count({len(detections)}) = "
    f"{linewidth_for_count(len(detections)):.2f}`** points (thicker for "
    "sparse images, thinner for dense ones). The output preserves the "
    "input's exact aspect ratio."
)

scheme_img = render_scheme_image(image_bgr, detections)
st.image(scheme_img, use_container_width=True)

buf = io.BytesIO()
scheme_img.save(buf, format="PNG")
st.download_button(
    "⬇️ Download scheme PNG",
    data=buf.getvalue(),
    file_name="crochet_scheme.png",
    mime="image/png",
)


# ═══════════════════════════════════════════════════════════════════════════
# Step 7 — Metrics + JSON export
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("## Step 7 — Metrics & JSON export")

counts = class_counts(detections)
col_metrics, col_counts = st.columns([2, 1])
with col_metrics:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stitches", len(detections))
    m2.metric("Tiles", info.get("tiles", "—"))
    m3.metric("Effective tile px", info.get("effective_tile_px", "—"))
    est = info.get("estimated_stitch_height_px")
    m4.metric("Est. stitch px", f"{est:.1f}" if isinstance(est, (int, float)) else "—")

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Estimate (s)", info.get("estimate_seconds", "—"))
    t2.metric("Inference (s)", info.get("infer_seconds", "—"))
    t3.metric("NMS (s)", info.get("nms_seconds", "—"))
    t4.metric("Total (s)", info.get("seconds", "—"))

with col_counts:
    st.markdown("**Per-class counts**")
    st.json(counts or {})

payload = detections_to_json(
    image_bgr, detections, info, st.session_state.input_source or {},
)
json_bytes = json.dumps(payload, indent=2).encode("utf-8")
st.download_button(
    "⬇️ Download detections JSON",
    data=json_bytes,
    file_name="crochet_scheme.json",
    mime="application/json",
)
with st.expander("Preview JSON payload"):
    st.json(payload)
