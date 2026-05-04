"""Step 3 — adaptive tiled inference with YOLOv8n-OBB.

Real-world photos rarely match the training receptive scale. The
functions in this module first estimate stitch size with a low-confidence
pass on a downsampled image, then choose a tile size that puts stitches
at the *training* scale inside each YOLO call. Detections from each
tile are translated back into original-image coordinates and reconciled
with class-aware non-maximum suppression.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class Detection:
    """A single oriented-bounding-box detection in original-image coords.

    Attributes:
        corners: ``(4, 2)`` float array of polygon vertices, in
            *original* image pixels (not tile-local).
        cls_id: Integer class identifier matching :data:`config.ID_TO_NAME`.
        cls_name: Human-readable class name.
        confidence: Detector score in ``[0, 1]``.
    """

    corners: np.ndarray
    cls_id: int
    cls_name: str
    confidence: float


@dataclass(frozen=True)
class TileInfo:
    """One tile produced by the adaptive splitter, kept for visualisation.

    Attributes:
        tile: BGR tile cropped from the original image.
        x_offset: x-coordinate of the tile's top-left in the original image.
        y_offset: y-coordinate of the tile's top-left in the original image.
        detections: Raw detections that came out of THIS tile, already
            translated to original-image coordinates. These are *pre-NMS*,
            so neighbouring tiles may report duplicate detections in the
            seam regions — that is exactly what the reassembly view shows.
        effective_size: The tile side length (in original-image pixels)
            chosen by the adaptive sizer.
    """

    tile: np.ndarray
    x_offset: int
    y_offset: int
    detections: list[Detection]
    effective_size: int


# ── Model loading ─────────────────────────────────────────────────────────
@lru_cache(maxsize=4)
def load_model(weights_path: str | Path) -> Any:
    """Instantiate a cached ``ultralytics.YOLO`` model.

    The cache means repeated calls during a Streamlit session or a
    notebook run do not re-load the checkpoint from disk.
    """
    from ultralytics import YOLO

    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    return YOLO(str(weights_path))


# ── NMS primitives ────────────────────────────────────────────────────────
def rect_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Axis-aligned IoU between two corner polygons.

    Used as a fast surrogate for full OBB IoU during NMS.
    """
    ax1, ay1 = a.min(axis=0)
    ax2, ay2 = a.max(axis=0)
    bx1, by1 = b.min(axis=0)
    bx2, by2 = b.max(axis=0)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


def nms(detections: Iterable[Detection],
        iou_threshold: float = 0.5) -> list[Detection]:
    """Greedy class-aware NMS."""
    pool = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    while pool:
        best = pool.pop(0)
        kept.append(best)
        pool = [d for d in pool
                if d.cls_id != best.cls_id
                or rect_iou(best.corners, d.corners) < iou_threshold]
    return kept


# ── Tiling ────────────────────────────────────────────────────────────────
def tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """Compute tile-start offsets that fully cover a 1-D length."""
    if tile_size >= length:
        return [0]
    starts = list(range(0, length - tile_size, stride))
    if not starts or starts[-1] + tile_size < length:
        starts.append(max(0, length - tile_size))
    return starts


def _infer_tile(model: Any, tile: np.ndarray,
                x_offset: int, y_offset: int,
                conf: float) -> list[Detection]:
    """Run YOLO on a single tile and translate detections to image coords."""
    res = model.predict(tile, conf=conf, verbose=False)[0]
    out: list[Detection] = []
    if res.obb is None:
        return out
    for box in res.obb:
        corners = box.xyxyxyxy.cpu().numpy().reshape(4, 2).astype(np.float32)
        corners[:, 0] += x_offset
        corners[:, 1] += y_offset
        cls_id = int(box.cls[0])
        out.append(Detection(corners=corners,
                             cls_id=cls_id,
                             cls_name=res.names[cls_id],
                             confidence=float(box.conf[0])))
    return out


def predict_tiled(model: Any, image: np.ndarray,
                  tile_size: int = 640, overlap: float = 0.2,
                  conf: float = 0.25,
                  iou_threshold: float = 0.5) -> list[Detection]:
    """Tile-and-infer with class-aware NMS over the union of tiles."""
    h, w = image.shape[:2]
    if h <= tile_size and w <= tile_size:
        return _infer_tile(model, image, 0, 0, conf)
    stride = max(1, int(tile_size * (1 - overlap)))
    out: list[Detection] = []
    for y1 in tile_starts(h, tile_size, stride):
        for x1 in tile_starts(w, tile_size, stride):
            tile = image[y1:min(y1 + tile_size, h), x1:min(x1 + tile_size, w)]
            out.extend(_infer_tile(model, tile, x1, y1, conf))
    return nms(out, iou_threshold)


# ── Adaptive tiling ───────────────────────────────────────────────────────
def estimate_stitch_size(model: Any, image: np.ndarray,
                         tile_size: int = 640,
                         conf: float = 0.15) -> float | None:
    """Estimate median stitch short-axis length on a downsampled image."""
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
        c = box.xyxyxyxy.cpu().numpy().reshape(4, 2)
        sizes.append(float(min(np.linalg.norm(c[0] - c[1]),
                               np.linalg.norm(c[1] - c[2]))))
    return float(np.median(sizes)) / scale


def predict_adaptive(model: Any, image: np.ndarray,
                     target_stitch_px: int = 100,
                     tile_size: int = 640,
                     overlap: float = 0.2,
                     conf: float = 0.25,
                     iou_threshold: float = 0.5) -> list[Detection]:
    """Estimate stitch size and tile so stitches appear at training scale."""
    h, w = image.shape[:2]
    est = estimate_stitch_size(model, image, tile_size=tile_size,
                               conf=min(conf, 0.15))
    if est is None or est <= 0:
        return predict_tiled(model, image, tile_size=tile_size,
                             overlap=overlap, conf=conf,
                             iou_threshold=iou_threshold)

    eff = max(64, int(tile_size * est / target_stitch_px))
    if eff >= max(h, w):
        return _infer_tile(model, image, 0, 0, conf)
    return predict_tiled(model, image, tile_size=eff, overlap=overlap,
                         conf=conf, iou_threshold=iou_threshold)


# ── Visualisation-aware variants ──────────────────────────────────────────
def predict_tiled_with_info(model: Any, image: np.ndarray,
                            tile_size: int = 640, overlap: float = 0.2,
                            conf: float = 0.25,
                            iou_threshold: float = 0.5
                            ) -> tuple[list[Detection], list[TileInfo]]:
    """Same as :func:`predict_tiled` but also returns per-tile information.

    Returns:
        tuple[list[Detection], list[TileInfo]]:
            ``(final_detections_post_nms, tiles_info)``. ``tiles_info``
            preserves the *raw* per-tile detections so callers can show
            duplicates in seam regions before NMS removes them.
    """
    h, w = image.shape[:2]
    if h <= tile_size and w <= tile_size:
        dets = _infer_tile(model, image, 0, 0, conf)
        return dets, [TileInfo(image, 0, 0, dets, tile_size)]

    stride = max(1, int(tile_size * (1 - overlap)))
    tiles_info: list[TileInfo] = []
    all_dets: list[Detection] = []
    for y1 in tile_starts(h, tile_size, stride):
        for x1 in tile_starts(w, tile_size, stride):
            tile = image[y1:min(y1 + tile_size, h), x1:min(x1 + tile_size, w)]
            tile_dets = _infer_tile(model, tile, x1, y1, conf)
            tiles_info.append(TileInfo(tile, x1, y1, tile_dets, tile_size))
            all_dets.extend(tile_dets)
    return nms(all_dets, iou_threshold), tiles_info


def predict_adaptive_with_tiles(model: Any, image: np.ndarray,
                                target_stitch_px: int = 100,
                                tile_size: int = 640,
                                overlap: float = 0.2,
                                conf: float = 0.25,
                                iou_threshold: float = 0.5
                                ) -> tuple[list[Detection], list[TileInfo]]:
    """Adaptive prediction that also returns per-tile info for visualisation.

    Same algorithm as :func:`predict_adaptive`. Use this variant when you
    need to render the tile grid, the per-tile detections, or the
    pre-vs-post-NMS reassembly view in the Streamlit app or notebook.

    Returns:
        tuple[list[Detection], list[TileInfo]]:
            ``(final_detections, tiles_info)``.
    """
    h, w = image.shape[:2]
    est = estimate_stitch_size(model, image, tile_size=tile_size,
                               conf=min(conf, 0.15))
    if est is None or est <= 0:
        return predict_tiled_with_info(model, image, tile_size=tile_size,
                                       overlap=overlap, conf=conf,
                                       iou_threshold=iou_threshold)

    eff = max(64, int(tile_size * est / target_stitch_px))
    if eff >= max(h, w):
        dets = _infer_tile(model, image, 0, 0, conf)
        return dets, [TileInfo(image, 0, 0, dets, eff)]
    return predict_tiled_with_info(model, image, tile_size=eff, overlap=overlap,
                                   conf=conf, iou_threshold=iou_threshold)
