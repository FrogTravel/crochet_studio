"""Visualisation helpers — overlays and reconstructed scheme.

The two-panel figure (left: detections on the original image, right:
clean reconstructed scheme) is the canonical demo output. Individual
glyphs are drawn procedurally with matplotlib so the scheme remains
crisp at any output resolution.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Iterable

import cv2 as cv
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
from PIL import Image

from .config import (
    CLASS_CONFIG, DEFAULT_COLOR, SCHEME_COLOR, TALL_STITCH_BARS,
)
from .inference import Detection, TileInfo


def compute_stroke_width(detections: Iterable[Detection],
                         *,
                         default: float = 1.6,
                         min_lw: float = 0.4,
                         max_lw: float = 2.6,
                         scale: float = 0.04) -> float:
    """Pick a matplotlib stroke width proportional to the median glyph.

    The same image can hold anywhere from a handful of large stitches
    (a single fan) to thousands of small ones (a tiled photograph).
    A fixed ``linewidth=2`` looks crisp for the former and chunky-bordering-
    on-illegible for the latter. This helper takes the *median* OBB short
    axis (in original-image pixels) as a proxy for "how big does a stitch
    look" and scales the stroke width linearly, with hard floor and ceiling
    so the line never disappears or overwhelms a glyph.

    Args:
        detections: Iterable of :class:`Detection` objects produced for
            the figure being drawn.
        default: Fallback when ``detections`` is empty.
        min_lw: Lower clamp, in matplotlib points.
        max_lw: Upper clamp, in matplotlib points.
        scale: Fraction of the median short axis used as the unclamped
            stroke width. The default ``0.04`` maps a 50-pixel median
            glyph to roughly 2pt, which matches the historical default.

    Returns:
        float: Stroke width in matplotlib points.
    """
    sizes: list[float] = []
    for det in detections:
        sa = float(np.linalg.norm(det.corners[0] - det.corners[1]))
        sb = float(np.linalg.norm(det.corners[1] - det.corners[2]))
        sizes.append(min(sa, sb))
    if not sizes:
        return default
    median = float(np.median(sizes))
    return float(max(min_lw, min(max_lw, median * scale)))


def draw_svg_icon(ax: Any, label: str,
                  x: float, y: float,
                  w: float, h: float,
                  angle_deg: float,
                  color: str,
                  linewidth: float = 2.0) -> None:
    """Draw a single stitch glyph on a matplotlib axis.

    Args:
        ax: A matplotlib ``Axes``.
        label: Stitch class name.
        x: Centre x in axis coordinates.
        y: Centre y in axis coordinates.
        w: Glyph width.
        h: Glyph height.
        angle_deg: Rotation in degrees, applied around the centre.
        color: Hex stroke colour.
        linewidth: Main stroke width in matplotlib points. Inner detail
            strokes (slashes, fan rays) are drawn at 75% of this value.
            Compute a sensible value once per figure with
            :func:`compute_stroke_width`.
    """
    t = transforms.Affine2D().rotate_deg_around(x, y, angle_deg) + ax.transData
    svg_type = CLASS_CONFIG.get(label, {}).get("svg_type", "cross")
    lw_main = linewidth
    lw_thin = max(0.3, linewidth * 0.75)

    if svg_type == "chain":
        ax.add_patch(patches.Ellipse((x, y), w * 0.8, h * 0.4,
                                     fill=False, color=color,
                                     linewidth=lw_main, transform=t))
    elif svg_type == "tall_stitch" or svg_type == "fan":
        ax.plot([x, x], [y - h / 2, y + h / 2],
                color=color, lw=lw_main, transform=t)
        ax.plot([x - w / 3, x + w / 3], [y - h / 2, y - h / 2],
                color=color, lw=lw_main, transform=t)
        n_bars = TALL_STITCH_BARS.get(label, 0)
        if svg_type == "fan":
            ax.plot([x, x - w / 3], [y + h / 2, y - h / 2],
                    color=color, lw=lw_thin, transform=t, alpha=0.6)
            ax.plot([x, x + w / 3], [y + h / 2, y - h / 2],
                    color=color, lw=lw_thin, transform=t, alpha=0.6)
        elif n_bars == 1:
            ax.plot([x - w / 4, x + w / 4], [y - h / 8, y + h / 8],
                    color=color, lw=lw_thin, transform=t)
        elif n_bars == 2:
            ax.plot([x - w / 4, x + w / 4], [y - h / 4, y - h / 12],
                    color=color, lw=lw_thin, transform=t)
            ax.plot([x - w / 4, x + w / 4], [y + h / 12, y + h / 4],
                    color=color, lw=lw_thin, transform=t)
        elif n_bars == 3:
            for y0, y1 in [(-h / 3, -h / 6), (-h / 12, h / 12), (h / 6, h / 3)]:
                ax.plot([x - w / 4, x + w / 4], [y + y0, y + y1],
                        color=color, lw=lw_thin, transform=t)
    elif svg_type == "enseble_chain":
        n = max(2, int(h / max(w * 0.5, 1)))
        oh = h / n
        for i in range(n):
            cy = y - h / 2 + oh * (i + 0.5)
            ax.add_patch(patches.Ellipse((x, cy), w * 0.6, oh * 0.7,
                                         fill=False, color=color,
                                         linewidth=lw_thin, transform=t))
    elif svg_type == "noise":
        ax.add_patch(patches.Circle((x, y), min(w, h) * 0.3,
                                    fill=False, color=color,
                                    linewidth=lw_thin, transform=t))
    else:  # "cross" (single) and any unknown
        ax.plot([x - w / 3, x + w / 3], [y - h / 3, y + h / 3],
                color=color, lw=lw_main, transform=t)
        ax.plot([x - w / 3, x + w / 3], [y + h / 3, y - h / 3],
                color=color, lw=lw_main, transform=t)


def render_two_panel(image: np.ndarray,
                     detections: Iterable[Detection],
                     out_path: str | Path | None = None,
                     dpi: int = 150) -> Any:
    """Render the canonical two-panel demo figure."""
    detections = list(detections)
    h, w = image.shape[:2]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # One stroke width for the whole figure, derived from the median
    # glyph short axis. The overlay polygons (left panel) get a slightly
    # heavier stroke than the scheme glyphs (right panel) so the boxes
    # stay visible on top of the photograph.
    glyph_lw = compute_stroke_width(detections)
    overlay_lw = max(0.6, glyph_lw * 1.1)

    ax1.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    ax1.set_title("YOLO-OBB detections (adaptive tiling)")
    ax2.set_facecolor("white")
    ax2.set_xlim(0, w)
    ax2.set_ylim(h, 0)
    ax2.set_title("Reconstructed scheme")

    for det in detections:
        cfg = CLASS_CONFIG.get(det.cls_name, {"abbr": "??", "color": DEFAULT_COLOR})
        color = cfg["color"]
        abbr = cfg["abbr"]
        closed = np.vstack([det.corners, det.corners[0]])
        ax1.plot(closed[:, 0], closed[:, 1], color=color, linewidth=overlay_lw)
        ax1.text(det.corners[0, 0], det.corners[0, 1] - 5, abbr,
                 color="white", fontsize=8, fontweight="bold",
                 bbox=dict(facecolor=color, edgecolor="none", alpha=0.7))
        centre = det.corners.mean(axis=0)
        width  = float(np.linalg.norm(det.corners[0] - det.corners[1]))
        height = float(np.linalg.norm(det.corners[0] - det.corners[3]))
        angle  = float(np.degrees(np.arctan2(det.corners[1, 1] - det.corners[0, 1],
                                             det.corners[1, 0] - det.corners[0, 0])))
        draw_svg_icon(ax2, det.cls_name, centre[0], centre[1],
                      width, height, angle, color, linewidth=glyph_lw)

    ax1.axis("off")
    ax2.axis("off")
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    return fig


def render_scheme(detections: Iterable[Detection],
                  width: int, height: int,
                  out_path: str | Path | None = None) -> Image.Image:
    """Render the right-hand "clean scheme" panel only."""
    detections = list(detections)
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.set_facecolor("white")
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    glyph_lw = compute_stroke_width(detections)
    for det in detections:
        centre = det.corners.mean(axis=0)
        w_ = float(np.linalg.norm(det.corners[0] - det.corners[1]))
        h_ = float(np.linalg.norm(det.corners[0] - det.corners[3]))
        angle = float(np.degrees(np.arctan2(det.corners[1, 1] - det.corners[0, 1],
                                            det.corners[1, 0] - det.corners[0, 0])))
        draw_svg_icon(ax, det.cls_name, centre[0], centre[1],
                      w_, h_, angle, SCHEME_COLOR, linewidth=glyph_lw)
    ax.axis("off")
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    if out_path is not None:
        img.save(str(out_path))
    return img


def detections_to_svg(detections: Iterable[Detection],
                      width: int, height: int,
                      *,
                      stroke: str = "#000000",
                      stroke_width: float = 1.5,
                      out_path: str | Path | None = None) -> str:
    """Convert detections into a self-contained SVG document.

    The output uses each detection's *rotated bounding rectangle* — a
    quick deliverable that is good enough for embedding in tutorials.
    """
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    ]
    for det in detections:
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in det.corners)
        parts.append(
            f'<polygon points="{pts}" fill="none" '
            f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )
    parts.append("</svg>")
    svg = "\n".join(parts)
    if out_path is not None:
        Path(out_path).write_text(svg)
    return svg


# ── Tiling visualisations ─────────────────────────────────────────────────
def _tile_color(idx: int, n: int) -> tuple[float, float, float]:
    """Internal: pick a distinct RGB triple for tile ``idx`` of ``n``."""
    cmap = plt.get_cmap("tab20" if n > 10 else "tab10")
    return cmap(idx % cmap.N)[:3]


def render_tile_grid(image: np.ndarray,
                     tiles_info: Iterable[TileInfo],
                     out_path: str | Path | None = None,
                     dpi: int = 120) -> Any:
    """Render the original image with numbered tile rectangles overlaid.

    Each tile is shown as a translucent yellow patch with a yellow border
    and its index number at the centre. The figure title carries the
    effective tile size chosen by the adaptive sizer so users can see
    how the algorithm reacted to the input.

    Args:
        image: BGR original image.
        tiles_info: Iterable of :class:`TileInfo` from
            :func:`predict_adaptive_with_tiles`.
        out_path: If provided, save the figure to this path.
        dpi: Resolution when ``out_path`` is provided.

    Returns:
        Any: The matplotlib ``Figure`` instance.
    """
    tiles_info = list(tiles_info)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

    eff = tiles_info[0].effective_size if tiles_info else 0
    for i, info in enumerate(tiles_info):
        th, tw = info.tile.shape[:2]
        # Translucent fill
        ax.add_patch(patches.Rectangle(
            (info.x_offset, info.y_offset), tw, th,
            linewidth=1.5, edgecolor="gold", facecolor="gold", alpha=0.18,
        ))
        # Solid border on top
        ax.add_patch(patches.Rectangle(
            (info.x_offset, info.y_offset), tw, th,
            linewidth=1.5, edgecolor="gold", facecolor="none",
        ))
        ax.text(
            info.x_offset + tw / 2, info.y_offset + th / 2, str(i),
            color="gold", fontsize=11, ha="center", va="center",
            fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2),
        )

    ax.set_title(
        f"Tile grid — {len(tiles_info)} tile(s)  |  effective tile = {eff} px",
        fontsize=13,
    )
    ax.axis("off")
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    return fig


def render_tiles_panel(tiles_info: Iterable[TileInfo],
                       max_tiles: int = 12,
                       out_path: str | Path | None = None,
                       dpi: int = 120) -> Any:
    """Render a grid of individual tiles, each annotated with its detections.

    Tile-local detection counts are drawn as bounding-box outlines (not
    full SVG glyphs) so dense tiles stay readable. If there are more
    than ``max_tiles`` tiles, the rendering is truncated and the title
    notes how many were skipped.

    Args:
        tiles_info: Iterable of :class:`TileInfo`.
        max_tiles: Upper bound on the number of tiles drawn.
        out_path: If provided, save the figure to this path.
        dpi: Resolution when ``out_path`` is provided.

    Returns:
        Any: The matplotlib ``Figure`` instance.
    """
    tiles_info = list(tiles_info)
    total = len(tiles_info)
    shown = tiles_info[:max_tiles]
    n = len(shown)

    cols = min(4, max(1, n))
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for i, info in enumerate(shown):
        ax = axes[i]
        ax.imshow(cv.cvtColor(info.tile, cv.COLOR_BGR2RGB))
        th, tw = info.tile.shape[:2]
        for det in info.detections:
            # Translate to tile-local coords for drawing on this axis.
            local = det.corners.copy()
            local[:, 0] -= info.x_offset
            local[:, 1] -= info.y_offset
            cfg = CLASS_CONFIG.get(det.cls_name, {"color": DEFAULT_COLOR})
            closed = np.vstack([local, local[0]])
            ax.plot(closed[:, 0], closed[:, 1],
                    color=cfg["color"], linewidth=1.4)
        ax.set_title(
            f"Tile {i}  ({info.x_offset}, {info.y_offset})  "
            f"{len(info.detections)} det",
            fontsize=8,
        )
        ax.set_xlim(0, tw)
        ax.set_ylim(th, 0)
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    suptitle = f"Per-tile detections  ({n} of {total} tiles shown)"
    if total > max_tiles:
        suptitle += f"  —  {total - max_tiles} hidden"
    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    return fig


def render_reassembly(image: np.ndarray,
                      tiles_info: Iterable[TileInfo],
                      final_detections: Iterable[Detection],
                      out_path: str | Path | None = None,
                      dpi: int = 120) -> Any:
    """Render a side-by-side "before vs after NMS" view.

    Left panel:  every tile's *raw* detections plotted on the full image,
                 colour-coded by tile index. Duplicates produced by tiles
                 that overlap in their seams show up as same-spot overlays.
    Right panel: the final detections after class-aware NMS. The total
                 count drops compared to the left, and the duplicates are
                 gone — that drop is the "glue" step.

    Args:
        image: BGR original image.
        tiles_info: Iterable of :class:`TileInfo`.
        final_detections: Detections after NMS — typically the first
            element of the tuple returned by
            :func:`predict_adaptive_with_tiles`.
        out_path: If provided, save the figure to this path.
        dpi: Resolution when ``out_path`` is provided.

    Returns:
        Any: The matplotlib ``Figure`` instance.
    """
    tiles_info = list(tiles_info)
    final_detections = list(final_detections)
    n_pre = sum(len(t.detections) for t in tiles_info)
    n_post = len(final_detections)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(20, 10))
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    axL.imshow(rgb)
    axR.imshow(rgb)

    # Adapt overlay weight to median glyph size so dense, small-stitch
    # photos get readable hairlines and sparse, large-stitch ones get
    # solid outlines.
    overlay_lw = compute_stroke_width(final_detections,
                                      default=1.4, min_lw=0.4, max_lw=2.0,
                                      scale=0.035)

    # Left: per-tile raw detections, colour-coded by tile index.
    for i, info in enumerate(tiles_info):
        color = _tile_color(i, len(tiles_info))
        # Outline the tile faintly so the colour mapping is legible.
        th, tw = info.tile.shape[:2]
        axL.add_patch(patches.Rectangle(
            (info.x_offset, info.y_offset), tw, th,
            linewidth=0.8, edgecolor=color, facecolor="none", alpha=0.4,
        ))
        for det in info.detections:
            closed = np.vstack([det.corners, det.corners[0]])
            axL.plot(closed[:, 0], closed[:, 1],
                     color=color, linewidth=overlay_lw)

    axL.set_title(
        f"Before NMS — {n_pre} raw detections, colour-coded by tile",
        fontsize=12,
    )
    axL.axis("off")

    # Right: final NMS-filtered detections, using each class's brand colour.
    for det in final_detections:
        cfg = CLASS_CONFIG.get(det.cls_name, {"color": DEFAULT_COLOR})
        closed = np.vstack([det.corners, det.corners[0]])
        axR.plot(closed[:, 0], closed[:, 1],
                 color=cfg["color"], linewidth=overlay_lw * 1.15)

    removed = max(0, n_pre - n_post)
    axR.set_title(
        f"After NMS — {n_post} final detections  "
        f"({removed} duplicate(s) removed)",
        fontsize=12,
    )
    axR.axis("off")

    fig.suptitle(
        "Reassembly: how overlapping tiles get glued back together",
        fontsize=14,
    )
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    return fig
