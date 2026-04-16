"""Image-level renderers.

Three outputs are produced for the UI:

* :func:`render_scheme_image` — reconstructed black-line scheme at the
  input's exact aspect ratio.
* :func:`render_detection_image` — input image with OBB overlays.
* :func:`render_tile_grid_image` — tile geometry overlay for the admin
  walkthrough.

Helpers :func:`draw_scheme` and :func:`draw_detection_overlay` expose the
underlying axis-painting routines so callers can compose custom figures.
"""

from __future__ import annotations

import io
import math
from typing import Iterable

import cv2 as cv
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ..config import (
    CLASS_CONFIG,
    DEFAULT_COLOR,
    ICON_SHRINK,
    MAX_RENDER_HEIGHT_PX,
    MIN_RENDER_HEIGHT_PX,
    SCHEME_COLOR,
)
from ..detection import Detection
from .icons import draw_svg_icon


# Icons whose natural orientation puts the long axis HORIZONTAL (e.g. the
# wide oval of a chain). All other icons draw their long axis VERTICALLY.
_HORIZONTAL_LONG_AXIS_LABELS = {"chain"}


# ── Line-width heuristics ──────────────────────────────────────────────────
def linewidth_for_count(n: int) -> float:
    """Scheme-glyph linewidth that scales down as detection count grows."""
    if n <= 1:
        return 4.0
    return max(0.4, min(4.0, 10.0 / math.sqrt(n)))


def _detection_linewidth(n: int) -> float:
    """Detection-box stroke width that scales down as detection count grows."""
    if n <= 1:
        return 2.0
    return max(0.25, min(2.0, 8.0 / math.sqrt(n)))


# ── Figure-size / DPI helpers ──────────────────────────────────────────────
def _figsize_and_dpi(
    image: np.ndarray,
    panel_h_in: float = 8.0,
    max_h_in: float = 16.0,
) -> tuple[tuple[float, float], int]:
    """Pick a matplotlib figsize + dpi so the PNG preserves the source pixel count."""
    h, w = image.shape[:2]
    aspect = w / max(h, 1)
    # Grow the figure for very tall inputs so lines don't become hair-thin.
    h_in = min(max(panel_h_in, h / 200.0), max_h_in)
    w_in = max(h_in * aspect, 1.0)
    target_h_px = max(MIN_RENDER_HEIGHT_PX, min(h, MAX_RENDER_HEIGHT_PX))
    dpi = int(max(100, round(target_h_px / h_in)))
    return (w_in, h_in), dpi


def fig_to_pil(fig, pad_inches: float = 0.05, dpi: int | None = None) -> Image.Image:
    """Save a matplotlib figure to PNG bytes and return a PIL image."""
    buf = io.BytesIO()
    kwargs: dict[str, object] = {
        "format":      "png",
        "bbox_inches": "tight",
        "pad_inches":  pad_inches,
        "facecolor":   "white",
    }
    if dpi is not None:
        kwargs["dpi"] = dpi
    fig.savefig(buf, **kwargs)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# ── Axis painters ──────────────────────────────────────────────────────────
def _orient_icon_axes(corners: np.ndarray) -> tuple[float, float, float]:
    """Return ``(short_len, long_len, long_angle_deg)`` for one OBB.

    Picks whichever of the two edges from corner 0 is longer as the long
    axis — YOLO's OBB corner ordering isn't guaranteed, so we can't just
    assume ``corners[0]→corners[1]`` is the bar (short) side.
    """
    side01 = corners[1] - corners[0]
    side03 = corners[3] - corners[0]
    len01 = float(np.linalg.norm(side01))
    len03 = float(np.linalg.norm(side03))
    if len03 >= len01:
        long_vec, long_len, short_len = side03, len03, len01
    else:
        long_vec, long_len, short_len = side01, len01, len03
    long_angle = float(math.degrees(math.atan2(long_vec[1], long_vec[0])))
    return short_len, long_len, long_angle


def draw_scheme(
    ax,
    image: np.ndarray,
    detections: list[Detection],
    shrink: float = ICON_SHRINK,
    color: str = SCHEME_COLOR,
    lw: float | None = None,
) -> None:
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
        short_len, long_len, long_angle = _orient_icon_axes(corners)

        if det.cls_name in _HORIZONTAL_LONG_AXIS_LABELS:
            # Chain ovals are naturally horizontal: long axis = oval major.
            w_icon = long_len * shrink
            h_icon = short_len * shrink
            angle_icon = long_angle
        else:
            # Trebles/doubles/etc.: stem (icon's +Y) follows the long axis.
            # +90° puts the bar on the side ``long_vec`` points toward.
            w_icon = short_len * shrink
            h_icon = long_len * shrink
            angle_icon = long_angle + 90.0

        draw_svg_icon(ax, det.cls_name, center[0], center[1],
                      w_icon, h_icon, angle_icon, color, lw=effective_lw)
    ax.axis("off")


def draw_detection_overlay(
    ax,
    image: np.ndarray,
    detections: list[Detection],
    show_labels: bool | str = "auto",
    lw: float | None = None,
    label_threshold: int = 40,
    label_fontsize: float | None = None,
    alpha: float = 0.85,
) -> None:
    """Overlay coloured OBB polygons on the input image.

    - ``lw=None`` auto-scales with detection count (many detections →
      thin strokes so the underlying image stays visible).
    - ``show_labels="auto"`` draws per-box labels only when there are
      ``≤ label_threshold`` detections. Pass ``True``/``False`` to force.
    """
    h, w = image.shape[:2]
    n = len(detections)

    ax.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")

    effective_lw = lw if lw is not None else _detection_linewidth(n)

    if show_labels == "auto":
        draw_labels = n <= label_threshold
    else:
        draw_labels = bool(show_labels)

    if label_fontsize is None:
        label_fontsize = max(5.0, min(8.0, 40.0 / max(math.sqrt(max(n, 1)), 1.0)))

    for det in detections:
        corners = det.corners
        cfg = CLASS_CONFIG.get(det.cls_name, {"abbr": "??", "color": DEFAULT_COLOR})
        color, abbr = cfg["color"], cfg["abbr"]
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


# ── High-level image renderers ─────────────────────────────────────────────
def render_scheme_image(image: np.ndarray, detections: list[Detection]) -> Image.Image:
    """Standalone scheme PNG at the input's exact aspect ratio."""
    figsize, dpi = _figsize_and_dpi(image)
    fig, ax = plt.subplots(figsize=figsize)
    draw_scheme(ax, image, detections)
    plt.tight_layout(pad=0)
    return fig_to_pil(fig, pad_inches=0.0, dpi=dpi)


def render_detection_image(
    image: np.ndarray,
    detections: list[Detection],
    show_labels: bool | str = "auto",
    lw: float | None = None,
    label_threshold: int = 40,
) -> Image.Image:
    """Input image with OBB boxes drawn on top (for the admin view)."""
    figsize, dpi = _figsize_and_dpi(image)
    fig, ax = plt.subplots(figsize=figsize)
    draw_detection_overlay(ax, image, detections,
                           show_labels=show_labels, lw=lw,
                           label_threshold=label_threshold)
    plt.tight_layout(pad=0)
    return fig_to_pil(fig, pad_inches=0.0, dpi=dpi)


def render_tile_grid_image(
    image: np.ndarray,
    tile_info: Iterable[tuple[int, int, int, int]],
) -> Image.Image:
    """Draw the tile-grid overlay. ``tile_info`` is ``[(x1, y1, tw, th), …]``."""
    tile_info = list(tile_info)
    h, w = image.shape[:2]
    figsize, dpi = _figsize_and_dpi(image)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

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
