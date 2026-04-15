"""Adaptive tiled YOLOv8n OBB stitch classification.

Ports the inference logic from ``Google_colab_YOLO_OBB_pipeline.ipynb``:

* estimate median stitch height on a downsampled pass,
* pick an effective tile size so stitches land at ~``target_stitch_px``
  inside each YOLO input tile,
* run the model tile by tile, offset detections back to the original
  image coordinates, and merge results with class-aware NMS,
* render a side-by-side figure (raw OBB detections + reconstructed SVG
  scheme), and optionally save the tile grid visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_WEIGHTS = "runs/obb/obb_train23/weights/best.pt"

CLASS_CONFIG: dict[str, dict[str, str]] = {
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


# ── 1. Detection container ──────────────────────────────────────────────────
@dataclass
class Detection:
    corners: np.ndarray   # (4, 2) in original image coords
    cls_id: int
    cls_name: str
    confidence: float


# ── 2. NMS + tiling helpers ─────────────────────────────────────────────────
def _rect_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1 = a.min(axis=0); ax2, ay2 = a.max(axis=0)
    bx1, by1 = b.min(axis=0); bx2, by2 = b.max(axis=0)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _nms(detections: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
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
    starts = list(range(0, length - tile_size, stride))
    if not starts or starts[-1] + tile_size < length:
        starts.append(max(0, length - tile_size))
    return starts


def _infer_tile(model, tile: np.ndarray, x_offset: int, y_offset: int, conf: float) -> list[Detection]:
    res = model.predict(tile, conf=conf, verbose=False)[0]
    dets: list[Detection] = []
    if res.obb is None:
        return dets
    for box in res.obb:
        corners = box.xyxyxyxy.cpu().numpy().reshape(4, 2).astype(np.float32)
        corners[:, 0] += x_offset
        corners[:, 1] += y_offset
        cls_id = int(box.cls[0])
        dets.append(Detection(
            corners=corners,
            cls_id=cls_id,
            cls_name=res.names[cls_id],
            confidence=float(box.conf[0]),
        ))
    return dets


# ── 3. Tile visualization (optional) ────────────────────────────────────────
def _visualize_tiles(image: np.ndarray, tiles_info: list, tile_size: int,
                     save_path: Path | None = None, show: bool = False) -> None:
    """Two-panel visualisation: tile grid on original image + individual tiles."""
    import cv2 as cv
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    n = len(tiles_info)

    fig_map, ax_map = plt.subplots(1, 1, figsize=(12, 8))
    ax_map.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    for i, (tile, x1, y1) in enumerate(tiles_info):
        th, tw = tile.shape[:2]
        ax_map.add_patch(patches.Rectangle(
            (x1, y1), tw, th,
            linewidth=1.5, edgecolor="yellow", facecolor="yellow", alpha=0.15,
        ))
        ax_map.add_patch(patches.Rectangle(
            (x1, y1), tw, th,
            linewidth=1.5, edgecolor="yellow", facecolor="none",
        ))
        ax_map.text(
            x1 + tw / 2, y1 + th / 2, str(i),
            color="yellow", fontsize=9, ha="center", va="center", fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1),
        )
    ax_map.set_title(f"Tile grid — {n} tiles  (effective tile size ≈ {tile_size} px)", fontsize=13)
    ax_map.axis("off")
    plt.tight_layout()
    if save_path is not None:
        fig_map.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[classify_stitches] Saved tile grid to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig_map)


# ── 4. Tiled inference ──────────────────────────────────────────────────────
def predict_tiled(model, image: np.ndarray, tile_size: int = 640, overlap: float = 0.2,
                  conf: float = 0.25, iou_threshold: float = 0.5,
                  visualize: bool = False, tile_figure_path: Path | None = None,
                  show: bool = False) -> list[Detection]:
    h, w = image.shape[:2]
    if h <= tile_size and w <= tile_size:
        if visualize:
            _visualize_tiles(image, [(image, 0, 0)], tile_size,
                             save_path=tile_figure_path, show=show)
        return _infer_tile(model, image, 0, 0, conf)

    stride = max(1, int(tile_size * (1 - overlap)))
    all_dets: list[Detection] = []
    tiles_info: list = []
    for y1 in _tile_starts(h, tile_size, stride):
        for x1 in _tile_starts(w, tile_size, stride):
            tile = image[y1:min(y1 + tile_size, h), x1:min(x1 + tile_size, w)]
            tiles_info.append((tile, x1, y1))
            all_dets.extend(_infer_tile(model, tile, x1, y1, conf))

    if visualize:
        _visualize_tiles(image, tiles_info, tile_size,
                         save_path=tile_figure_path, show=show)
    return _nms(all_dets, iou_threshold)


def estimate_stitch_size(model, image: np.ndarray, tile_size: int = 640,
                         conf: float = 0.15) -> float | None:
    """Quick low-confidence pass on a downsampled image to estimate stitch size.

    Uses the *maximum* OBB short axis across detections rather than the
    median. When an image mixes big stitches (fans, trebles) with many
    small fillers (chains, '+' marks), the median gets pulled toward the
    small symbols and causes over-tiling that chops the big stitches
    across tiles. Using the max asks "how big is the biggest stitch
    here?" — if that already fits well, we skip tiling entirely.

    Returns the estimated stitch short-side size in original-image pixels,
    or ``None`` if nothing was detected.
    """
    import cv2 as cv

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
        side_a = np.linalg.norm(corners[0] - corners[1])
        side_b = np.linalg.norm(corners[1] - corners[2])
        sizes.append(float(min(side_a, side_b)))  # short axis ≈ stitch height

    return float(np.max(sizes)) / scale


def predict_adaptive(model, image: np.ndarray, target_stitch_px: int = 100,
                     tile_size: int = 640, overlap: float = 0.2, conf: float = 0.25,
                     iou_threshold: float = 0.5, visualize: bool = False,
                     tile_figure_path: Path | None = None,
                     show: bool = False) -> list[Detection]:
    """Estimate stitch size first, then tile so stitches land at ~``target_stitch_px``.

    Large stitches → effective tile ≥ image → single-shot inference.
    Small stitches → many small tiles, each upscaled to ``tile_size`` by YOLO.
    """
    h, w = image.shape[:2]
    estimated_h = estimate_stitch_size(model, image, tile_size=tile_size,
                                       conf=min(conf, 0.15))

    if estimated_h is None or estimated_h <= 0:
        print("  [adaptive] No detections on size-estimation pass — using fixed tiling.")
        return predict_tiled(
            model, image, tile_size=tile_size, overlap=overlap, conf=conf,
            iou_threshold=iou_threshold, visualize=visualize,
            tile_figure_path=tile_figure_path, show=show,
        )

    effective_tile = int(tile_size * estimated_h / target_stitch_px)
    effective_tile = max(effective_tile, 64)

    print(
        f"  [adaptive] estimated stitch height: {estimated_h:.1f} px  →  "
        f"effective tile: {effective_tile} px  (target: {target_stitch_px} px at {tile_size})"
    )

    if effective_tile >= max(h, w):
        print("  [adaptive] Stitches large enough — single-shot inference.")
        if visualize:
            _visualize_tiles(image, [(image, 0, 0)], effective_tile,
                             save_path=tile_figure_path, show=show)
        return _infer_tile(model, image, 0, 0, conf)

    return predict_tiled(
        model, image, tile_size=effective_tile, overlap=overlap, conf=conf,
        iou_threshold=iou_threshold, visualize=visualize,
        tile_figure_path=tile_figure_path, show=show,
    )


# ── 5. SVG icon drawing ─────────────────────────────────────────────────────
def draw_svg_icon(ax, label: str, x: float, y: float, w: float, h: float,
                  angle_deg: float, color: str) -> None:
    import matplotlib.patches as patches
    import matplotlib.transforms as transforms

    t = transforms.Affine2D().rotate_deg_around(x, y, angle_deg) + ax.transData

    if label == "chain":
        ax.add_patch(patches.Ellipse(
            (x, y), w * 0.8, h * 0.4,
            fill=False, color=color, linewidth=2, transform=t,
        ))
    elif label in ("half_double", "double", "double treble", "treble", "fan"):
        ax.plot([x, x], [y - h / 2, y + h / 2], color=color, lw=2, transform=t)
        ax.plot([x - w / 3, x + w / 3], [y - h / 2, y - h / 2],
                color=color, lw=2, transform=t)
        if label == "double":
            ax.plot([x - w / 4, x + w / 4], [y - h / 8, y + h / 8],
                    color=color, lw=1.5, transform=t)
        elif label == "treble":
            ax.plot([x - w / 4, x + w / 4], [y - h / 4, y - h / 12],
                    color=color, lw=1.5, transform=t)
            ax.plot([x - w / 4, x + w / 4], [y + h / 12, y + h / 4],
                    color=color, lw=1.5, transform=t)
        elif label == "double treble":
            ax.plot([x - w / 4, x + w / 4], [y - h / 3, y - h / 6],
                    color=color, lw=1.5, transform=t)
            ax.plot([x - w / 4, x + w / 4], [y - h / 12, y + h / 12],
                    color=color, lw=1.5, transform=t)
            ax.plot([x - w / 4, x + w / 4], [y + h / 6, y + h / 3],
                    color=color, lw=1.5, transform=t)
        elif label == "fan":
            ax.plot([x, x - w / 3], [y + h / 2, y - h / 2],
                    color=color, lw=1.5, transform=t, alpha=0.6)
            ax.plot([x, x + w / 3], [y + h / 2, y - h / 2],
                    color=color, lw=1.5, transform=t, alpha=0.6)
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
        ax.plot([x - w / 3, x + w / 3], [y - h / 3, y + h / 3],
                color=color, lw=2, transform=t)
        ax.plot([x - w / 3, x + w / 3], [y + h / 3, y - h / 3],
                color=color, lw=2, transform=t)
    else:
        ax.plot([x - w / 3, x + w / 3], [y - h / 3, y + h / 3],
                color=color, lw=2, transform=t)
        ax.plot([x - w / 3, x + w / 3], [y + h / 3, y - h / 3],
                color=color, lw=2, transform=t)


# ── 6. Top-level entry point ────────────────────────────────────────────────
def classify_stitches(
    image_path: str | Path,
    weights_path: str | Path = DEFAULT_WEIGHTS,
    conf: float = 0.25,
    iou_threshold: float = 0.5,
    target_stitch_px: int = 100,
    tile_size: int = 640,
    overlap: float = 0.2,
    output_figure: str | Path | None = None,
    tile_figure: str | Path | None = None,
    visualize_tiles: bool = True,
    show: bool = False,
) -> dict[str, Any]:
    """Run adaptive tiled OBB inference and render detection + scheme figure.

    Parameters
    ----------
    image_path:
        Path to the input PNG (e.g. one produced by ``generate_image``).
    weights_path:
        Path to the trained ``.pt`` weights. Defaults to
        ``runs/obb/obb_train23/weights/best.pt``.
    conf, iou_threshold:
        Detection confidence threshold and NMS IoU threshold.
    target_stitch_px:
        Expected stitch short-side size at training resolution. Controls
        how aggressively the image is tiled.
    tile_size, overlap:
        YOLO input size and fractional overlap between neighbouring tiles.
    output_figure:
        Where to save the 2-panel detection/scheme figure. Defaults to
        ``<image>_predictions.png`` next to the input.
    tile_figure:
        Where to save the tile-grid overlay (only meaningful when
        ``visualize_tiles`` is True). Defaults to ``<image>_tiles.png``.
    visualize_tiles:
        Whether to render the tile grid figure alongside detection.
    show:
        If True, also call ``plt.show()``.

    Returns
    -------
    dict
        Summary with paths of saved figures, number of detections, and a
        list of per-detection records.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"YOLO weights not found: {weights_path}. Run training or point "
            "weights_path at the correct checkpoint."
        )

    if output_figure is None:
        output_figure = image_path.with_name(f"{image_path.stem}_predictions.png")
    output_figure = Path(output_figure)
    output_figure.parent.mkdir(parents=True, exist_ok=True)

    if tile_figure is None:
        tile_figure = image_path.with_name(f"{image_path.stem}_tiles.png")
    tile_figure = Path(tile_figure)
    tile_figure.parent.mkdir(parents=True, exist_ok=True)

    # Lazy imports so --help works without deps installed.
    import cv2 as cv
    import matplotlib.pyplot as plt
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
        visualize=visualize_tiles,
        tile_figure_path=tile_figure if visualize_tiles else None,
        show=show,
    )
    print(f"[classify_stitches] Detected {len(detections)} stitches")

    # Side-by-side figure: raw detections + reconstructed scheme
    h, w = image.shape[:2]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    ax1.set_title("YOLO OBB Detections (adaptive tiling)")

    ax2.set_facecolor("white")
    ax2.set_xlim(0, w)
    ax2.set_ylim(h, 0)
    ax2.set_title("Reconstructed SVG Scheme")

    records: list[dict[str, Any]] = []
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

        records.append({
            "class": det.cls_name,
            "abbr": abbr,
            "confidence": det.confidence,
            "center": (float(center[0]), float(center[1])),
            "width": width,
            "height": height,
            "angle_deg": angle,
        })

    ax1.axis("off")
    ax2.axis("off")
    plt.tight_layout()
    fig.savefig(output_figure, dpi=150, bbox_inches="tight")
    print(f"[classify_stitches] Saved figure to {output_figure}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "figure_path": output_figure,
        "tile_figure_path": tile_figure if visualize_tiles else None,
        "num_detections": len(detections),
        "detections": records,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Adaptive tiled YOLO OBB stitch classification")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to .pt weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--target-stitch-px", type=int, default=100,
                        help="Target stitch short-side size at training resolution")
    parser.add_argument("--tile-size", type=int, default=640, help="YOLO input tile size")
    parser.add_argument("--overlap", type=float, default=0.2, help="Tile overlap fraction")
    parser.add_argument("--output", default=None, help="Path to save the detection figure")
    parser.add_argument("--tile-figure", default=None, help="Path to save tile-grid figure")
    parser.add_argument("--no-tile-viz", action="store_true", help="Skip tile-grid visualization")
    parser.add_argument("--show", action="store_true", help="Show figures interactively")
    args = parser.parse_args()

    summary = classify_stitches(
        image_path=args.image,
        weights_path=args.weights,
        conf=args.conf,
        iou_threshold=args.iou,
        target_stitch_px=args.target_stitch_px,
        tile_size=args.tile_size,
        overlap=args.overlap,
        output_figure=args.output,
        tile_figure=args.tile_figure,
        visualize_tiles=not args.no_tile_viz,
        show=args.show,
    )
    print(f"[classify_stitches] {summary['num_detections']} detections")
