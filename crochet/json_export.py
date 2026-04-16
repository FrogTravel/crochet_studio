"""Serialise detection results to plain JSON for downstream consumers.

Keeps the Streamlit app, the CLI pipeline, and the MCP server in lock-step
on the JSON schema they produce.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import CLASS_CONFIG
from .detection import Detection


# Fields captured only for the admin view; never shipped in JSON exports.
_CAPTURE_ONLY_FIELDS = frozenset({"tile_records", "raw_det_list", "tile_geometry"})


def class_counts(detections: list[Detection]) -> dict[str, int]:
    """Return per-class counts, ordered by descending frequency."""
    counts: dict[str, int] = {}
    for d in detections:
        counts[d.cls_name] = counts.get(d.cls_name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _detection_record(det: Detection) -> dict[str, Any]:
    corners = det.corners
    center = corners.mean(axis=0)
    width = float(np.linalg.norm(corners[0] - corners[1]))
    height = float(np.linalg.norm(corners[0] - corners[3]))
    angle = float(np.degrees(np.arctan2(
        corners[1, 1] - corners[0, 1],
        corners[1, 0] - corners[0, 0],
    )))
    return {
        "class":      det.cls_name,
        "abbr":       CLASS_CONFIG.get(det.cls_name, {}).get("abbr", "??"),
        "confidence": round(det.confidence, 4),
        "corners":    [[float(x), float(y)] for x, y in corners.tolist()],
        "center":     [float(center[0]), float(center[1])],
        "width":      width,
        "height":     height,
        "angle_deg":  angle,
    }


def detections_to_json(
    image: np.ndarray,
    detections: list[Detection],
    info: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    """Build the JSON payload emitted by the Streamlit apps and CLI."""
    h, w = image.shape[:2]
    clean_info = {k: v for k, v in info.items() if k not in _CAPTURE_ONLY_FIELDS}
    return {
        "source":          source,
        "image_size":      {"width": int(w), "height": int(h)},
        "num_detections":  len(detections),
        "inference":       clean_info,
        "detections":      [_detection_record(d) for d in detections],
    }
