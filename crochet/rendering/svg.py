"""Standalone SVG glyph generators for embeddable scheme reconstructions.

Used by the MCP server to ship an SVG rendering of a detected scheme
alongside the detection data, so clients can drop it straight into
HTML/Markdown output without re-rasterising anything.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..config import CLASS_CONFIG, TALL_STITCH_BARS
from ..detection import Detection


# ── SVG primitives ─────────────────────────────────────────────────────────
def _transform(cx: float, cy: float, angle_deg: float) -> str:
    if abs(angle_deg) < 0.5:
        return ""
    return f' transform="rotate({angle_deg:.1f},{cx:.1f},{cy:.1f})"'


def svg_chain(cx, cy, w, h, angle, color):
    rx = max(w * 0.35, 3)
    ry = max(h * 0.2, 2)
    t = _transform(cx, cy, angle)
    return (f'<g{t}><ellipse cx="{cx:.1f}" cy="{cy:.1f}" '
            f'rx="{rx:.1f}" ry="{ry:.1f}" fill="none" '
            f'stroke="{color}" stroke-width="1.5"/></g>')


def svg_tall_stitch(cx, cy, w, h, angle, color, n_bars: int = 1):
    half_h = h / 2
    bar_w = w * 0.3
    slash_w = w * 0.2
    t = _transform(cx, cy, angle)
    parts = [
        f'<g{t}>',
        f'<line x1="{cx:.1f}" y1="{cy - half_h:.1f}" x2="{cx:.1f}" y2="{cy + half_h:.1f}" stroke="{color}" stroke-width="1.8"/>',
        f'<line x1="{cx - bar_w:.1f}" y1="{cy - half_h:.1f}" x2="{cx + bar_w:.1f}" y2="{cy - half_h:.1f}" stroke="{color}" stroke-width="1.8"/>',
    ]
    if n_bars >= 1:
        parts.append(
            f'<line x1="{cx - slash_w:.1f}" y1="{cy + h * 0.06:.1f}" '
            f'x2="{cx + slash_w:.1f}" y2="{cy - h * 0.06:.1f}" stroke="{color}" stroke-width="1.3"/>'
        )
    if n_bars >= 2:
        off = h * 0.15
        parts.append(
            f'<line x1="{cx - slash_w:.1f}" y1="{cy + off:.1f}" '
            f'x2="{cx + slash_w:.1f}" y2="{cy + off - h * 0.12:.1f}" stroke="{color}" stroke-width="1.3"/>'
        )
        parts.append(
            f'<line x1="{cx - slash_w:.1f}" y1="{cy - off + h * 0.12:.1f}" '
            f'x2="{cx + slash_w:.1f}" y2="{cy - off:.1f}" stroke="{color}" stroke-width="1.3"/>'
        )
    if n_bars >= 3:
        parts.append(
            f'<line x1="{cx - slash_w:.1f}" y1="{cy + h * 0.25:.1f}" '
            f'x2="{cx + slash_w:.1f}" y2="{cy + h * 0.13:.1f}" stroke="{color}" stroke-width="1.3"/>'
        )
    parts.append('</g>')
    return "\n".join(parts)


def svg_cross(cx, cy, w, h, angle, color):
    dx = w * 0.25
    dy = h * 0.25
    t = _transform(cx, cy, angle)
    return (
        f'<g{t}>'
        f'<line x1="{cx - dx:.1f}" y1="{cy - dy:.1f}" x2="{cx + dx:.1f}" y2="{cy + dy:.1f}" stroke="{color}" stroke-width="1.8"/>'
        f'<line x1="{cx - dx:.1f}" y1="{cy + dy:.1f}" x2="{cx + dx:.1f}" y2="{cy - dy:.1f}" stroke="{color}" stroke-width="1.8"/>'
        f'</g>'
    )


def svg_fan(cx, cy, w, h, angle, color):
    half_h = h / 2
    spread = w * 0.4
    n_spokes = 5
    t = _transform(cx, cy, angle)
    parts = [f'<g{t}>']
    base_y = cy + half_h
    for i in range(n_spokes):
        frac = i / (n_spokes - 1)
        tip_x = cx - spread + 2 * spread * frac
        tip_y = cy - half_h
        parts.append(
            f'<line x1="{cx:.1f}" y1="{base_y:.1f}" '
            f'x2="{tip_x:.1f}" y2="{tip_y:.1f}" stroke="{color}" stroke-width="1.8"/>'
        )
    arc_left = cx - spread
    arc_right = cx + spread
    arc_y = cy - half_h
    cp_y = arc_y - h * 0.12
    parts.append(
        f'<path d="M{arc_left:.1f},{arc_y:.1f} Q{cx:.1f},{cp_y:.1f} '
        f'{arc_right:.1f},{arc_y:.1f}" fill="none" stroke="{color}" stroke-width="1.2"/>'
    )
    parts.append('</g>')
    return "\n".join(parts)


def svg_enseble_chain(cx, cy, w, h, angle, color):
    n_ovals = max(2, round(h / max(w * 0.6, 8)))
    oval_h = h / n_ovals
    rx = max(w * 0.25, 3)
    ry = max(oval_h * 0.35, 2)
    t = _transform(cx, cy, angle)
    parts = [f'<g{t}>']
    for i in range(n_ovals):
        oy = cy - h / 2 + oval_h * (i + 0.5)
        parts.append(
            f'<ellipse cx="{cx:.1f}" cy="{oy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" '
            f'fill="none" stroke="{color}" stroke-width="1.2"/>'
        )
    parts.append('</g>')
    return "\n".join(parts)


def svg_noise(cx, cy, w, h, angle, color):
    r = max(min(w, h) * 0.2, 2)
    return (
        f'<g><circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" '
        f'fill="none" stroke="{color}" stroke-width="1" opacity="0.5"/></g>'
    )


SVG_DRAWERS: dict[str, Callable[..., str]] = {
    "chain":         svg_chain,
    "tall_stitch":   svg_tall_stitch,
    "cross":         svg_cross,
    "fan":           svg_fan,
    "enseble_chain": svg_enseble_chain,
    "noise":         svg_noise,
}


# ── Geometry helpers ───────────────────────────────────────────────────────
def detection_geometry(det: Detection) -> dict[str, float]:
    """Return centre, width, height, and angle for an OBB detection."""
    corners = det.corners
    center = corners.mean(axis=0)
    w = float(np.linalg.norm(corners[0] - corners[1]))
    h = float(np.linalg.norm(corners[0] - corners[3]))
    angle = float(np.degrees(np.arctan2(
        corners[1, 1] - corners[0, 1],
        corners[1, 0] - corners[0, 0],
    )))
    return {"cx": float(center[0]), "cy": float(center[1]),
            "w": w, "h": h, "angle": angle}


# ── Document assembly ──────────────────────────────────────────────────────
def detections_to_svg(
    detections: list[Detection],
    img_w: int,
    img_h: int,
    include_noise: bool = False,
) -> str:
    """Render a complete SVG document showing the reconstructed scheme."""
    parts = [
        f'<svg width="{img_w}" height="{img_h}" viewBox="0 0 {img_w} {img_h}" '
        'xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{img_w}" height="{img_h}" fill="#faf8f4"/>',
    ]
    for det in detections:
        cfg = CLASS_CONFIG.get(det.cls_name)
        if cfg is None:
            continue
        svg_type = cfg["svg_type"]
        if svg_type == "noise" and not include_noise:
            continue
        drawer = SVG_DRAWERS.get(svg_type)
        if drawer is None:
            continue

        g = detection_geometry(det)
        kwargs: dict[str, object] = {
            "cx": g["cx"], "cy": g["cy"], "w": g["w"], "h": g["h"],
            "angle": g["angle"], "color": cfg["color"],
        }
        if svg_type == "tall_stitch":
            kwargs["n_bars"] = TALL_STITCH_BARS.get(det.cls_name, 1)
        parts.append(drawer(**kwargs))
    parts.append('</svg>')
    return "\n".join(parts)
