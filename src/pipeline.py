"""End-to-end pipeline orchestration.

These functions string the per-stage modules together so callers — the
CLI in ``main.py``, the Streamlit app in ``app.py``, the notebook —
share a single happy path.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import cv2 as cv

from .config import (
    DEFAULT_CONF, DEFAULT_IOU, DEFAULT_OVERLAP,
    DEFAULT_TARGET_STITCH_PX, DEFAULT_TILE_SIZE, DEFAULT_WEIGHTS,
)
from .generation import generate_image
from .inference import Detection, load_model, predict_adaptive
from .rendering import render_two_panel


def detections_to_json(detections: Iterable[Detection],
                       image_path: str | None = None,
                       image_width: int | None = None,
                       image_height: int | None = None) -> dict[str, Any]:
    """Serialise detections into a JSON-ready dictionary."""
    items: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for det in detections:
        items.append({
            "class":      det.cls_name,
            "class_id":   int(det.cls_id),
            "confidence": float(det.confidence),
            "corners":    det.corners.tolist(),
        })
        counts[det.cls_name] += 1
    return {
        "image_path":   str(image_path) if image_path else None,
        "image_width":  image_width,
        "image_height": image_height,
        "n_detections": len(items),
        "class_counts": dict(counts),
        "detections":   items,
    }


def classify_stitches(image_path: str | Path,
                      weights_path: str | Path = DEFAULT_WEIGHTS,
                      *,
                      conf: float = DEFAULT_CONF,
                      iou_threshold: float = DEFAULT_IOU,
                      target_stitch_px: int = DEFAULT_TARGET_STITCH_PX,
                      tile_size: int = DEFAULT_TILE_SIZE,
                      overlap: float = DEFAULT_OVERLAP,
                      output_figure: str | Path | None = None
                      ) -> dict[str, Any]:
    """Run adaptive tiled inference on an existing image."""
    image_path = Path(image_path)
    image = cv.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = image.shape[:2]

    model = load_model(weights_path)
    detections = predict_adaptive(
        model, image,
        target_stitch_px=target_stitch_px,
        tile_size=tile_size,
        overlap=overlap,
        conf=conf,
        iou_threshold=iou_threshold,
    )

    payload = detections_to_json(detections, str(image_path), w, h)

    if output_figure is not None:
        output_figure = Path(output_figure)
        output_figure.parent.mkdir(parents=True, exist_ok=True)
        render_two_panel(image, detections, out_path=output_figure)
        payload["figure_path"] = str(output_figure)

    return payload


def run_pipeline(prompt: str,
                 image_out: str | Path,
                 figure_out: str | Path,
                 weights_path: str | Path = DEFAULT_WEIGHTS,
                 *,
                 gemini_model: str = "gemini-3.1-flash-image-preview",
                 conf: float = DEFAULT_CONF,
                 iou_threshold: float = DEFAULT_IOU,
                 target_stitch_px: int = DEFAULT_TARGET_STITCH_PX,
                 tile_size: int = DEFAULT_TILE_SIZE,
                 overlap: float = DEFAULT_OVERLAP,
                 skip_generation: bool = False) -> dict[str, Any]:
    """Generate a chart with Gemini, then classify and render it."""
    image_out = Path(image_out)
    if not skip_generation:
        generate_image(prompt=prompt, output_path=image_out, model=gemini_model)
    payload = classify_stitches(
        image_out,
        weights_path=weights_path,
        conf=conf,
        iou_threshold=iou_threshold,
        target_stitch_px=target_stitch_px,
        tile_size=tile_size,
        overlap=overlap,
        output_figure=figure_out,
    )
    payload["prompt"] = prompt
    return payload
