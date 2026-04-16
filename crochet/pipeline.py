"""End-to-end generate-then-classify pipeline.

Generates a crochet scheme image with Gemini (optional), runs adaptive
tiled YOLOv8n OBB inference on it, and saves a side-by-side figure
(raw detections + reconstructed scheme) plus a tile-grid overlay.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from .config import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    DEFAULT_OVERLAP,
    DEFAULT_TARGET_STITCH_PX,
    DEFAULT_TILE_SIZE,
    DEFAULT_WEIGHTS,
)
from .detection import Detection, predict_adaptive, tile_starts
from .generation import DEFAULT_MODEL, DEFAULT_PROMPT, generate_image
from .json_export import _detection_record
from .rendering.figures import (
    _figsize_and_dpi,
    draw_detection_overlay,
    draw_scheme,
    fig_to_pil,
    render_scheme_image,
    render_tile_grid_image,
)


def _derive_output_paths(
    image_path: Path,
    output_figure: str | Path | None,
    tile_figure: str | Path | None,
) -> tuple[Path, Path]:
    out_fig = Path(output_figure) if output_figure else image_path.with_name(
        f"{image_path.stem}_predictions.png"
    )
    tile_fig = Path(tile_figure) if tile_figure else image_path.with_name(
        f"{image_path.stem}_tiles.png"
    )
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    tile_fig.parent.mkdir(parents=True, exist_ok=True)
    return out_fig, tile_fig


def _render_side_by_side(
    image: np.ndarray,
    detections: list[Detection],
    output_path: Path,
) -> None:
    h, w = image.shape[:2]
    aspect = w / max(h, 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10 * aspect + 2, 10))
    draw_detection_overlay(ax1, image, detections)
    ax1.set_title("YOLO OBB detections (adaptive tiling)")
    draw_scheme(ax2, image, detections)
    ax2.set_title("Reconstructed scheme")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _derive_tile_geometry(
    image: np.ndarray,
    tile_size: int,
    overlap: float,
) -> list[tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    if tile_size >= max(h, w):
        return [(0, 0, w, h)]
    stride = max(1, int(tile_size * (1 - overlap)))
    geometry: list[tuple[int, int, int, int]] = []
    for y1 in tile_starts(h, tile_size, stride):
        for x1 in tile_starts(w, tile_size, stride):
            x2 = min(x1 + tile_size, w)
            y2 = min(y1 + tile_size, h)
            geometry.append((x1, y1, x2 - x1, y2 - y1))
    return geometry


def classify_stitches(
    image_path: str | Path,
    weights_path: str | Path = DEFAULT_WEIGHTS,
    conf: float = DEFAULT_CONF,
    iou_threshold: float = DEFAULT_IOU,
    target_stitch_px: int = DEFAULT_TARGET_STITCH_PX,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: float = DEFAULT_OVERLAP,
    output_figure: str | Path | None = None,
    tile_figure: str | Path | None = None,
    visualize_tiles: bool = True,
) -> dict[str, Any]:
    """Run adaptive tiled OBB inference and save the result figures."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"YOLO weights not found: {weights_path}. Run training or update "
            "weights_path."
        )

    output_figure_path, tile_figure_path = _derive_output_paths(
        image_path, output_figure, tile_figure,
    )

    # Lazy import so --help works without ultralytics installed.
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    image = cv.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    detections = predict_adaptive(
        model, image,
        target_stitch_px=target_stitch_px,
        tile_size=tile_size,
        overlap=overlap,
        conf=conf,
        iou_threshold=iou_threshold,
    )

    _render_side_by_side(image, detections, output_figure_path)

    # Standalone reconstructed scheme (clean icons only, no original image).
    scheme_path = output_figure_path.parent / "scheme.png"
    scheme_img = render_scheme_image(image, detections)
    scheme_img.save(scheme_path, dpi=(150, 150))

    if visualize_tiles:
        geometry = _derive_tile_geometry(image, tile_size, overlap)
        render_tile_grid_image(image, geometry).save(tile_figure_path)

    return {
        "figure_path":       output_figure_path,
        "scheme_path":       scheme_path,
        "tile_figure_path":  tile_figure_path if visualize_tiles else None,
        "num_detections":    len(detections),
        "detections":        [_detection_record(d) for d in detections],
    }


def run_pipeline(
    prompt: str = DEFAULT_PROMPT,
    image_out: str | Path = "free_output.png",
    figure_out: str | Path | None = None,
    tile_figure: str | Path | None = None,
    weights_path: str | Path = DEFAULT_WEIGHTS,
    gemini_model: str = DEFAULT_MODEL,
    conf: float = DEFAULT_CONF,
    iou_threshold: float = DEFAULT_IOU,
    target_stitch_px: int = DEFAULT_TARGET_STITCH_PX,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: float = DEFAULT_OVERLAP,
    visualize_tiles: bool = True,
    api_key: str | None = None,
    skip_generation: bool = False,
) -> dict[str, Any]:
    """Run the full generate-then-classify pipeline.

    When ``skip_generation`` is ``True`` the existing file at ``image_out``
    is reused (handy for iterating on inference without burning API quota).
    """
    image_out = Path(image_out)

    if skip_generation:
        if not image_out.exists():
            raise FileNotFoundError(
                f"skip_generation=True but {image_out} does not exist."
            )
    else:
        generate_image(
            prompt=prompt, output_path=image_out,
            model=gemini_model, api_key=api_key,
        )

    return classify_stitches(
        image_path=image_out,
        weights_path=weights_path,
        conf=conf, iou_threshold=iou_threshold,
        target_stitch_px=target_stitch_px,
        tile_size=tile_size, overlap=overlap,
        output_figure=figure_out, tile_figure=tile_figure,
        visualize_tiles=visualize_tiles,
    )
