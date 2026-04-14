"""
tiler.py

Slices a large image into overlapping tiles, runs YOLO OBB inference on each,
and merges the results back into original image coordinate space with NMS.

Includes adaptive tiling that estimates stitch size first, then tiles so
stitches always appear at a consistent scale inside each YOLO input tile.
"""

from __future__ import annotations
from dataclasses import dataclass

import cv2 as cv
import numpy as np


@dataclass
class Detection:
    """A single YOLO OBB detection in original image coordinates."""
    corners: np.ndarray   # shape (4, 2), float32, in original image px
    cls_id: int
    cls_name: str
    confidence: float


def _bbox(corners: np.ndarray) -> tuple[float, float, float, float]:
    """Axis-aligned bounding rect of an OBB polygon."""
    x1, y1 = corners.min(axis=0)
    x2, y2 = corners.max(axis=0)
    return x1, y1, x2, y2


def _rect_iou(a: np.ndarray, b: np.ndarray) -> float:
    """
    Approximate IoU between two OBB polygons using their axis-aligned bounding rects.
    Fast and sufficient for deduplicating detections that straddle tile borders.
    """
    ax1, ay1, ax2, ay2 = _bbox(a)
    bx1, by1, bx2, by2 = _bbox(b)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """
    Greedy NMS: keeps the highest-confidence detection and suppresses
    same-class overlapping detections below the IoU threshold.
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


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """
    Return the list of top-left pixel offsets for tiles along one axis.
    The last tile is anchored to the far edge so the image is fully covered.
    """
    starts = list(range(0, length - tile_size, stride))
    # Always include a tile anchored to the far edge
    if not starts or starts[-1] + tile_size < length:
        starts.append(max(0, length - tile_size))
    return starts


def _infer_tile(
    model,
    tile: np.ndarray,
    x_offset: int,
    y_offset: int,
    conf: float,
) -> list[Detection]:
    """Run inference on a single tile and translate corners to original coords."""
    results = model.predict(tile, conf=conf, verbose=False)
    res = results[0]

    detections: list[Detection] = []
    if len(res.obb) == 0:
        return detections

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


def predict_tiled(
    model,
    image: np.ndarray,
    tile_size: int = 640,
    overlap: float = 0.2,
    conf: float = 0.25,
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """
    Run YOLO OBB inference on overlapping tiles and return merged detections.

    For images smaller than tile_size, falls back to a single full-image inference.

    Args:
        model:          Loaded YOLO model.
        image:          BGR image as a numpy array.
        tile_size:      Square tile side length in pixels.
        overlap:        Fractional overlap between adjacent tiles (0.0-1.0).
        conf:           YOLO confidence threshold.
        iou_threshold:  IoU threshold for NMS deduplication at tile boundaries.

    Returns:
        List of Detection objects with coordinates in original image space.
    """
    h, w = image.shape[:2]

    # If the image fits in one tile, skip the tiling overhead
    if h <= tile_size and w <= tile_size:
        return _infer_tile(model, image, x_offset=0, y_offset=0, conf=conf)

    stride = max(1, int(tile_size * (1 - overlap)))
    all_detections: list[Detection] = []

    y_starts = _tile_starts(h, tile_size, stride)
    x_starts = _tile_starts(w, tile_size, stride)

    for y1 in y_starts:
        for x1 in x_starts:
            y2 = min(y1 + tile_size, h)
            x2 = min(x1 + tile_size, w)
            tile = image[y1:y2, x1:x2]
            all_detections.extend(_infer_tile(model, tile, x1, y1, conf))

    return _nms(all_detections, iou_threshold)


# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive Tiling
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_stitch_size(
    model,
    image: np.ndarray,
    tile_size: int = 640,
    conf: float = 0.15,
) -> float | None:
    """
    Quick low-confidence pass on a downsampled image to estimate median stitch height.
    The short side of each OBB box is used (robust to elongated symbols like chains).
    Returns stitch height in original image pixels, or None if nothing is detected.
    """
    h, w = image.shape[:2]
    scale = tile_size / max(h, w)
    if scale < 1.0:
        small = cv.resize(image, (int(w * scale), int(h * scale)))
    else:
        small, scale = image, 1.0

    results = model.predict(small, conf=conf, verbose=False)
    res = results[0]

    if len(res.obb) == 0:
        return None

    sizes = []
    for box in res.obb:
        corners = box.xyxyxyxy.cpu().numpy().reshape(4, 2)
        side_a = float(np.linalg.norm(corners[0] - corners[1]))
        side_b = float(np.linalg.norm(corners[1] - corners[2]))
        sizes.append(min(side_a, side_b))  # short axis ~ stitch height

    return float(np.median(sizes)) / scale


def predict_adaptive(
    model,
    image: np.ndarray,
    target_stitch_px: int = 50,
    tile_size: int = 640,
    overlap: float = 0.2,
    conf: float = 0.25,
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """
    Estimate stitch size first, then tile so stitches always appear at
    ~target_stitch_px inside each tile fed to YOLO.

    Big stitches  -> effective_tile >= image -> single-shot inference, no overhead.
    Small stitches -> small effective_tile  -> many tiles, each upscaled to tile_size
                                              by YOLO, so stitches appear at the
                                              same scale as during training.

    Args:
        model:            Loaded YOLO model.
        image:            BGR image as a numpy array.
        target_stitch_px: Expected stitch short-side at training resolution.
        tile_size:        YOLO input size (usually 640).
        overlap:          Fractional overlap between tiles.
        conf:             YOLO confidence threshold.
        iou_threshold:    IoU threshold for NMS.

    Returns:
        List of Detection objects with coordinates in original image space.
    """
    h, w = image.shape[:2]

    estimated_h = estimate_stitch_size(model, image, tile_size=tile_size,
                                       conf=min(conf, 0.15))

    if estimated_h is None or estimated_h <= 0:
        # Fallback to fixed tiling
        return predict_tiled(model, image, tile_size=tile_size, overlap=overlap,
                             conf=conf, iou_threshold=iou_threshold)

    # Original-image pixels that should fill one tile so stitches appear
    # at target_stitch_px
    effective_tile = int(tile_size * estimated_h / target_stitch_px)
    effective_tile = max(effective_tile, 64)

    if effective_tile >= max(h, w):
        # Stitches large enough — single-shot inference
        return _infer_tile(model, image, 0, 0, conf)

    return predict_tiled(model, image, tile_size=effective_tile, overlap=overlap,
                         conf=conf, iou_threshold=iou_threshold)
