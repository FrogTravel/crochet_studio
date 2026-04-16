"""Tiled YOLO OBB inference with adaptive tile sizing.

The pipeline is:

1. :func:`estimate_stitch_size` — downsample, run a low-confidence pass,
   take the median long-axis length to get a robust stitch-height
   estimate in original-image pixels.
2. :func:`predict_adaptive` — pick an effective tile size so that
   stitches land at ``target_stitch_px`` inside each YOLO input, then
   call :func:`predict_tiled`.
3. :func:`predict_tiled` — slice the image into overlapping tiles, run
   inference on each, translate detections back into original coordinates,
   and merge with class-aware NMS.

:func:`predict_adaptive_with_progress` is the Streamlit-facing variant:
it emits phase/tile callbacks and can optionally capture raw pre-NMS
detections and per-tile crops for the admin view.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import cv2 as cv
import numpy as np

from .config import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    DEFAULT_OVERLAP,
    DEFAULT_TARGET_STITCH_PX,
    DEFAULT_TILE_SIZE,
)


# ── Detection container ────────────────────────────────────────────────────
@dataclass
class Detection:
    """A single YOLO OBB detection in original image coordinates.

    ``corners`` is a ``(4, 2)`` float32 array of polygon vertices.
    """

    corners: np.ndarray
    cls_id: int
    cls_name: str
    confidence: float


# ── NMS + tiling helpers ───────────────────────────────────────────────────
def _aabb(corners: np.ndarray) -> tuple[float, float, float, float]:
    """Axis-aligned bounding rect of an OBB polygon."""
    x1, y1 = corners.min(axis=0)
    x2, y2 = corners.max(axis=0)
    return float(x1), float(y1), float(x2), float(y2)


def _rect_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Approximate IoU using each OBB's axis-aligned bounding rect.

    Fast and sufficient for deduplicating detections at tile seams.
    """
    ax1, ay1, ax2, ay2 = _aabb(a)
    bx1, by1, bx2, by2 = _aabb(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def nms(detections: list[Detection], iou_threshold: float = DEFAULT_IOU) -> list[Detection]:
    """Greedy class-aware NMS.

    Keeps the highest-confidence detection; suppresses same-class boxes
    whose IoU (approximated via AABBs) meets ``iou_threshold``.
    """
    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [
            d for d in detections
            if d.cls_id != best.cls_id
            or _rect_iou(best.corners, d.corners) < iou_threshold
        ]
    return kept


def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """Return top-left offsets for tiles along one axis.

    The last tile is anchored to the far edge so the image is fully covered.
    """
    starts = list(range(0, length - tile_size, stride))
    if not starts or starts[-1] + tile_size < length:
        starts.append(max(0, length - tile_size))
    return starts


def infer_tile(
    model,
    tile: np.ndarray,
    x_offset: int,
    y_offset: int,
    conf: float,
) -> list[Detection]:
    """Run YOLO on one tile and translate detections back to source coords."""
    res = model.predict(tile, conf=conf, verbose=False)[0]
    if res.obb is None:
        return []

    detections: list[Detection] = []
    for box in res.obb:
        corners = box.xyxyxyxy.cpu().numpy().reshape(4, 2).astype(np.float32)
        corners[:, 0] += x_offset
        corners[:, 1] += y_offset
        cls_id = int(box.cls[0])
        detections.append(Detection(
            corners=corners,
            cls_id=cls_id,
            cls_name=res.names[cls_id],
            confidence=float(box.conf[0]),
        ))
    return detections


# ── Core inference ─────────────────────────────────────────────────────────
def predict_tiled(
    model,
    image: np.ndarray,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: float = DEFAULT_OVERLAP,
    conf: float = DEFAULT_CONF,
    iou_threshold: float = DEFAULT_IOU,
) -> list[Detection]:
    """Run YOLO OBB inference on overlapping tiles and merge the results.

    Falls back to a single-shot forward pass when the image already fits
    in one tile.
    """
    h, w = image.shape[:2]
    if h <= tile_size and w <= tile_size:
        return infer_tile(model, image, 0, 0, conf)

    stride = max(1, int(tile_size * (1 - overlap)))
    all_dets: list[Detection] = []
    for y1 in tile_starts(h, tile_size, stride):
        for x1 in tile_starts(w, tile_size, stride):
            tile = image[y1:min(y1 + tile_size, h), x1:min(x1 + tile_size, w)]
            all_dets.extend(infer_tile(model, tile, x1, y1, conf))
    return nms(all_dets, iou_threshold)


def estimate_stitch_size(
    model,
    image: np.ndarray,
    tile_size: int = DEFAULT_TILE_SIZE,
    conf: float = 0.15,
) -> float | None:
    """Estimate typical stitch height (in original-image px) with one pass.

    Downsamples so the longest side equals ``tile_size``, runs a
    low-confidence pass, then returns the **median** of each OBB's long
    axis (the stem length for T-shaped stitches). The median — rather
    than the max — keeps the estimate stable on dense small-stitch images
    where a handful of low-confidence boxes can be spuriously large.

    Returns ``None`` when no detections are found on the estimation pass.
    """
    h, w = image.shape[:2]
    scale = tile_size / max(h, w)
    if scale < 1.0:
        small = cv.resize(image, (int(w * scale), int(h * scale)))
    else:
        small, scale = image, 1.0

    res = model.predict(small, conf=conf, verbose=False)[0]
    if res.obb is None or len(res.obb) == 0:
        return None

    sizes: list[float] = []
    for box in res.obb:
        corners = box.xyxyxyxy.cpu().numpy().reshape(4, 2)
        side_a = float(np.linalg.norm(corners[0] - corners[1]))
        side_b = float(np.linalg.norm(corners[1] - corners[2]))
        sizes.append(max(side_a, side_b))
    return float(np.median(sizes)) / scale


def predict_adaptive(
    model,
    image: np.ndarray,
    target_stitch_px: int = DEFAULT_TARGET_STITCH_PX,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: float = DEFAULT_OVERLAP,
    conf: float = DEFAULT_CONF,
    iou_threshold: float = DEFAULT_IOU,
) -> list[Detection]:
    """Estimate stitch size, then tile so stitches land at ~``target_stitch_px``.

    Large stitches → effective tile ≥ image → single-shot inference.
    Small stitches → many small tiles, each upscaled to ``tile_size`` by YOLO.
    """
    h, w = image.shape[:2]
    estimated_h = estimate_stitch_size(model, image, tile_size=tile_size,
                                       conf=min(conf, 0.15))

    if estimated_h is None or estimated_h <= 0:
        return predict_tiled(model, image, tile_size=tile_size, overlap=overlap,
                             conf=conf, iou_threshold=iou_threshold)

    effective_tile = max(int(tile_size * estimated_h / target_stitch_px), 64)
    if effective_tile >= max(h, w):
        return infer_tile(model, image, 0, 0, conf)

    return predict_tiled(model, image, tile_size=effective_tile, overlap=overlap,
                         conf=conf, iou_threshold=iou_threshold)


# ── Progress-reporting variant (used by Streamlit UI) ──────────────────────
PhaseCallback = Callable[[str], None]
TileCallback = Callable[[int, int, int, int, int], None]  # (idx, total, x1, y1, n)


def predict_adaptive_with_progress(
    model,
    image: np.ndarray,
    target_stitch_px: int = DEFAULT_TARGET_STITCH_PX,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: float = DEFAULT_OVERLAP,
    conf: float = DEFAULT_CONF,
    iou_threshold: float = DEFAULT_IOU,
    on_phase: PhaseCallback | None = None,
    on_tile: TileCallback | None = None,
    capture_tiles: bool = False,
    capture_raw: bool = False,
) -> tuple[list[Detection], dict[str, Any]]:
    """Adaptive tiled inference with streaming callbacks and optional capture.

    Callbacks:
      * ``on_phase(label)`` — phase transitions (estimation / tiling / NMS).
      * ``on_tile(idx, total, x1, y1, n_dets)`` — fires once per tile.

    Capture flags (useful for admin/demo visualisations):
      * ``capture_tiles=True`` → ``info["tile_records"]`` holds
        ``[(x1, y1, tile_bgr, tile_detections), …]`` per tile.
      * ``capture_raw=True`` → ``info["raw_det_list"]`` holds the full
        pre-NMS detection list.

    Returns ``(detections, info)``.
    """
    info: dict[str, Any] = {}
    h, w = image.shape[:2]

    if on_phase:
        on_phase("Estimating stitch size (median of long axes)…")
    t0 = time.time()
    estimated_h = estimate_stitch_size(model, image, tile_size=tile_size,
                                       conf=min(conf, 0.15))
    info["estimate_seconds"] = round(time.time() - t0, 3)
    info["estimated_stitch_height_px"] = estimated_h

    if estimated_h is None or estimated_h <= 0:
        effective_tile = tile_size
        info["note"] = "No detections on size-estimation pass — falling back to fixed tiling."
    else:
        effective_tile = max(int(tile_size * estimated_h / target_stitch_px), 64)

    info.update({
        "effective_tile_px": effective_tile,
        "target_stitch_px":  target_stitch_px,
        "tile_size":         tile_size,
        "overlap":           overlap,
    })

    tile_records: list[tuple[int, int, np.ndarray, list[Detection]]] = []

    # Single-shot branch
    if effective_tile >= max(h, w):
        if on_phase:
            on_phase(
                f"Single-shot inference — effective tile {effective_tile}px ≥ image {w}×{h}."
            )
        t1 = time.time()
        dets = infer_tile(model, image, 0, 0, conf)
        info["infer_seconds"] = round(time.time() - t1, 3)
        if on_tile:
            on_tile(1, 1, 0, 0, len(dets))
        info["tiles"] = 1
        info["tile_geometry"] = [(0, 0, w, h)]
        info["raw_detections"] = len(dets)
        info["merged_detections"] = len(dets)
        info["single_shot"] = True
        if capture_tiles:
            info["tile_records"] = [(0, 0, image.copy(), list(dets))]
        return dets, info

    # Tiled branch
    stride = max(1, int(effective_tile * (1 - overlap)))
    xs = tile_starts(w, effective_tile, stride)
    ys = tile_starts(h, effective_tile, stride)
    total = len(xs) * len(ys)
    info["tiles"] = total
    info["stride_px"] = stride
    info["single_shot"] = False

    if on_phase and estimated_h is not None:
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
            tile_dets = infer_tile(model, tile, x1, y1, conf)
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
        info["raw_det_list"] = list(all_dets)

    if on_phase:
        on_phase(f"Running NMS on {len(all_dets)} raw detections (IoU≥{iou_threshold})…")
    t_nms = time.time()
    merged = nms(all_dets, iou_threshold)
    info["nms_seconds"] = round(time.time() - t_nms, 3)
    info["raw_detections"] = len(all_dets)
    info["merged_detections"] = len(merged)
    info["suppressed_detections"] = len(all_dets) - len(merged)
    return merged, info
