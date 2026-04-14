"""
CrochetDesigner MCP Tool — Crochet chart image → structured detection data + SVG.

Returns JSON-formatted detection results AND an embeddable SVG reconstruction
so that Claude can compose beautiful HTML crochet tutorials with accurate diagrams.

Setup:
  Add to Claude Desktop config (~/Library/Application Support/Claude/claude_desktop_config.json):
  {
    "mcpServers": {
      "CrochetDesigner": {
        "command": "python",
        "args": ["/ABSOLUTE/PATH/TO/crochet_tool.py"]
      }
    }
  }
"""

from __future__ import annotations

import json
import math
import os
import sys

import cv2 as cv
import numpy as np

from mcp.server.fastmcp import FastMCP

# Ensure local imports work
sys.path.insert(0, os.path.dirname(__file__))
from util.tiler import predict_adaptive, Detection

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PATH = os.path.join(os.path.dirname(__file__), "runs", "obb", "train6", "weights", "best.pt")

# 9-class system matching data/project_yolo_obb/data.yaml (ground truth)
CLASS_CONFIG = {
    "chain":          {"id": 0, "abbr": "ch", "color": "#4a6fa5", "svg_type": "chain"},
    "double":         {"id": 1, "abbr": "dc", "color": "#8b5e83", "svg_type": "tall_stitch"},
    "double treble":  {"id": 2, "abbr": "dtr","color": "#3a9a3a", "svg_type": "tall_stitch"},
    "enseble_chain":  {"id": 3, "abbr": "ec", "color": "#b07a3a", "svg_type": "enseble_chain"},
    "fan":            {"id": 4, "abbr": "fa", "color": "#c45a4a", "svg_type": "fan"},
    "half_double":    {"id": 5, "abbr": "hd", "color": "#5a8a7a", "svg_type": "tall_stitch"},
    "noise":          {"id": 6, "abbr": "no", "color": "#999999", "svg_type": "noise"},
    "single":         {"id": 7, "abbr": "sc", "color": "#c4943a", "svg_type": "cross"},
    "treble":         {"id": 8, "abbr": "tr", "color": "#5aaa5a", "svg_type": "tall_stitch"},
}

# ═══════════════════════════════════════════════════════════════════════════════
# SVG Symbol Generators
# ═══════════════════════════════════════════════════════════════════════════════
# Each returns an SVG group (<g>) string that can be embedded in an <svg> element.
# All symbols are drawn relative to center (cx, cy) with given width/height/angle.

def _svg_transform(cx: float, cy: float, angle_deg: float) -> str:
    """Build SVG transform attribute for rotation around center."""
    if abs(angle_deg) < 0.5:
        return ""
    return f' transform="rotate({angle_deg:.1f},{cx:.1f},{cy:.1f})"'


def svg_chain(cx: float, cy: float, w: float, h: float, angle: float, color: str) -> str:
    """Chain stitch: small oval/ellipse."""
    rx = max(w * 0.35, 3)
    ry = max(h * 0.2, 2)
    t = _svg_transform(cx, cy, angle)
    return (
        f'<g{t}>'
        f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" '
        f'fill="none" stroke="{color}" stroke-width="1.5"/>'
        f'</g>'
    )


def svg_tall_stitch(cx: float, cy: float, w: float, h: float, angle: float,
                    color: str, n_bars: int = 1) -> str:
    """Tall stitches: vertical stem + T-bar + diagonal slash(es).
    n_bars: 1=double, 0=half_double, 2=treble."""
    half_h = h / 2
    bar_w = w * 0.3
    t = _svg_transform(cx, cy, angle)
    parts = [f'<g{t}>']
    # Main vertical stem
    parts.append(
        f'<line x1="{cx:.1f}" y1="{cy - half_h:.1f}" '
        f'x2="{cx:.1f}" y2="{cy + half_h:.1f}" '
        f'stroke="{color}" stroke-width="1.8"/>'
    )
    # Top T-bar
    parts.append(
        f'<line x1="{cx - bar_w:.1f}" y1="{cy - half_h:.1f}" '
        f'x2="{cx + bar_w:.1f}" y2="{cy - half_h:.1f}" '
        f'stroke="{color}" stroke-width="1.8"/>'
    )
    # Diagonal slash bars
    if n_bars >= 1:
        slash_w = w * 0.2
        parts.append(
            f'<line x1="{cx - slash_w:.1f}" y1="{cy + h * 0.06:.1f}" '
            f'x2="{cx + slash_w:.1f}" y2="{cy - h * 0.06:.1f}" '
            f'stroke="{color}" stroke-width="1.3"/>'
        )
    if n_bars >= 2:
        slash_w = w * 0.2
        offset = h * 0.15
        parts.append(
            f'<line x1="{cx - slash_w:.1f}" y1="{cy + offset:.1f}" '
            f'x2="{cx + slash_w:.1f}" y2="{cy + offset - h * 0.12:.1f}" '
            f'stroke="{color}" stroke-width="1.3"/>'
        )
        parts.append(
            f'<line x1="{cx - slash_w:.1f}" y1="{cy - offset + h * 0.12:.1f}" '
            f'x2="{cx + slash_w:.1f}" y2="{cy - offset:.1f}" '
            f'stroke="{color}" stroke-width="1.3"/>'
        )
    parts.append('</g>')
    return '\n'.join(parts)


def svg_cross(cx: float, cy: float, w: float, h: float, angle: float, color: str) -> str:
    """Single crochet / X marker."""
    dx = w * 0.25
    dy = h * 0.25
    t = _svg_transform(cx, cy, angle)
    return (
        f'<g{t}>'
        f'<line x1="{cx - dx:.1f}" y1="{cy - dy:.1f}" '
        f'x2="{cx + dx:.1f}" y2="{cy + dy:.1f}" '
        f'stroke="{color}" stroke-width="1.8"/>'
        f'<line x1="{cx - dx:.1f}" y1="{cy + dy:.1f}" '
        f'x2="{cx + dx:.1f}" y2="{cy - dy:.1f}" '
        f'stroke="{color}" stroke-width="1.8"/>'
        f'</g>'
    )


def svg_fan(cx: float, cy: float, w: float, h: float, angle: float, color: str) -> str:
    """Fan/shell stitch: multiple lines radiating from base + arc at top."""
    half_h = h / 2
    spread = w * 0.4
    n_spokes = 5
    t = _svg_transform(cx, cy, angle)
    parts = [f'<g{t}>']
    base_y = cy + half_h
    for i in range(n_spokes):
        frac = i / (n_spokes - 1) if n_spokes > 1 else 0.5
        tip_x = cx - spread + 2 * spread * frac
        tip_y = cy - half_h
        parts.append(
            f'<line x1="{cx:.1f}" y1="{base_y:.1f}" '
            f'x2="{tip_x:.1f}" y2="{tip_y:.1f}" '
            f'stroke="{color}" stroke-width="1.8"/>'
        )
        # Small dc cross-bar on each spoke
        mx = (cx + tip_x) / 2
        my = (base_y + tip_y) / 2
        bar_len = 2.5
        dx = tip_x - cx
        dy = tip_y - base_y
        spoke_len = math.sqrt(dx * dx + dy * dy) or 1
        nx = -dy / spoke_len * bar_len
        ny = dx / spoke_len * bar_len
        parts.append(
            f'<line x1="{mx - nx:.1f}" y1="{my - ny:.1f}" '
            f'x2="{mx + nx:.1f}" y2="{my + ny:.1f}" '
            f'stroke="{color}" stroke-width="1.2"/>'
        )
    # Arc at top
    arc_left_x = cx - spread
    arc_right_x = cx + spread
    arc_y = cy - half_h
    cp_y = arc_y - h * 0.12
    parts.append(
        f'<path d="M{arc_left_x:.1f},{arc_y:.1f} '
        f'Q{cx:.1f},{cp_y:.1f} {arc_right_x:.1f},{arc_y:.1f}" '
        f'fill="none" stroke="{color}" stroke-width="1.2"/>'
    )
    parts.append('</g>')
    return '\n'.join(parts)


def svg_enseble_chain(cx: float, cy: float, w: float, h: float,
                      angle: float, color: str) -> str:
    """Ensemble chain: vertical column of small ovals."""
    n_ovals = max(2, round(h / max(w * 0.6, 8)))
    oval_h = h / n_ovals
    rx = max(w * 0.25, 3)
    ry = max(oval_h * 0.35, 2)
    t = _svg_transform(cx, cy, angle)
    parts = [f'<g{t}>']
    for i in range(n_ovals):
        oy = cy - h / 2 + oval_h * (i + 0.5)
        parts.append(
            f'<ellipse cx="{cx:.1f}" cy="{oy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" '
            f'fill="none" stroke="{color}" stroke-width="1.2"/>'
        )
    parts.append('</g>')
    return '\n'.join(parts)


def svg_noise(cx: float, cy: float, w: float, h: float, angle: float, color: str) -> str:
    """Noise/annotation: small circle marker."""
    r = max(min(w, h) * 0.2, 2)
    t = _svg_transform(cx, cy, angle)
    return (
        f'<g{t}>'
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" '
        f'fill="none" stroke="{color}" stroke-width="1" opacity="0.5"/>'
        f'</g>'
    )


# Registry mapping svg_type → drawing function
SVG_DRAWERS = {
    "chain":          svg_chain,
    "tall_stitch":    svg_tall_stitch,
    "cross":          svg_cross,
    "fan":            svg_fan,
    "enseble_chain":  svg_enseble_chain,
    "noise":          svg_noise,
}

def _n_bars_for_class(cls_name: str) -> int:
    """Number of diagonal bars for tall stitches."""
    return {"double": 1, "half_double": 0, "treble": 2, "double treble": 3}.get(cls_name, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Row Detection & Grouping
# ═══════════════════════════════════════════════════════════════════════════════

def _group_into_rows(detections: list[Detection], img_h: int) -> list[list[Detection]]:
    """Group detections into rows by Y coordinate clustering."""
    if not detections:
        return []
    # Sort by center Y
    dets_with_cy = [(d, d.corners.mean(axis=0)[1]) for d in detections]
    dets_with_cy.sort(key=lambda x: x[1])

    # Estimate median stitch height for row gap threshold
    heights = []
    for d, _ in dets_with_cy:
        side_a = float(np.linalg.norm(d.corners[0] - d.corners[1]))
        side_b = float(np.linalg.norm(d.corners[1] - d.corners[2]))
        heights.append(max(side_a, side_b))
    median_h = sorted(heights)[len(heights) // 2] if heights else 30
    row_gap = median_h * 0.5

    rows: list[list[Detection]] = []
    current_row: list[Detection] = [dets_with_cy[0][0]]
    current_cy = dets_with_cy[0][1]

    for det, cy in dets_with_cy[1:]:
        if abs(cy - current_cy) > row_gap:
            rows.append(current_row)
            current_row = [det]
            current_cy = cy
        else:
            current_row.append(det)
            # Running average of row center
            current_cy = current_cy * 0.8 + cy * 0.2

    if current_row:
        rows.append(current_row)

    # Sort each row left-to-right
    for row in rows:
        row.sort(key=lambda d: d.corners.mean(axis=0)[0])

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Detection → SVG Conversion
# ═══════════════════════════════════════════════════════════════════════════════

def _det_geometry(det: Detection) -> dict:
    """Extract center, width, height, angle from a detection."""
    corners = det.corners
    center = corners.mean(axis=0)
    w = float(np.linalg.norm(corners[0] - corners[1]))
    h = float(np.linalg.norm(corners[0] - corners[3]))
    angle = float(np.degrees(np.arctan2(
        corners[1, 1] - corners[0, 1],
        corners[1, 0] - corners[0, 0]
    )))
    return {"cx": float(center[0]), "cy": float(center[1]),
            "w": w, "h": h, "angle": angle}


def detections_to_svg(detections: list[Detection], img_w: int, img_h: int,
                      include_noise: bool = False) -> str:
    """Convert detection list to a complete SVG string."""
    parts = [
        f'<svg width="{img_w}" height="{img_h}" viewBox="0 0 {img_w} {img_h}" '
        f'xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{img_w}" height="{img_h}" fill="#faf8f4"/>',
    ]

    for det in detections:
        cfg = CLASS_CONFIG.get(det.cls_name)
        if cfg is None:
            continue
        if cfg["svg_type"] == "noise" and not include_noise:
            continue

        geom = _det_geometry(det)
        drawer = SVG_DRAWERS.get(cfg["svg_type"])
        if drawer is None:
            continue

        kwargs = {
            "cx": geom["cx"], "cy": geom["cy"],
            "w": geom["w"], "h": geom["h"],
            "angle": geom["angle"], "color": cfg["color"],
        }
        if cfg["svg_type"] == "tall_stitch":
            kwargs["n_bars"] = _n_bars_for_class(det.cls_name)

        parts.append(drawer(**kwargs))

    parts.append('</svg>')
    return '\n'.join(parts)


def detections_to_json(detections: list[Detection], img_w: int, img_h: int) -> list[dict]:
    """Convert detections to a JSON-serializable list for Claude to use."""
    results = []
    rows = _group_into_rows(detections, img_h)

    for row_idx, row in enumerate(rows):
        for det in row:
            cfg = CLASS_CONFIG.get(det.cls_name, {})
            geom = _det_geometry(det)
            results.append({
                "class": det.cls_name,
                "abbreviation": cfg.get("abbr", "??"),
                "confidence": round(det.confidence, 3),
                "row": row_idx,
                "center_x": round(geom["cx"], 1),
                "center_y": round(geom["cy"], 1),
                "width": round(geom["w"], 1),
                "height": round(geom["h"], 1),
                "angle_deg": round(geom["angle"], 1),
                "corners": [[round(float(c[0]), 1), round(float(c[1]), 1)]
                            for c in det.corners],
            })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Summary Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def _build_summary(detections: list[Detection], img_w: int, img_h: int,
                   rows: list[list[Detection]]) -> dict:
    """Build a summary of the chart analysis."""
    from collections import Counter
    class_counts = Counter(d.cls_name for d in detections)

    return {
        "total_stitches": len(detections),
        "image_size": {"width": img_w, "height": img_h},
        "num_rows": len(rows),
        "stitches_per_row": [len(r) for r in rows],
        "class_counts": dict(class_counts),
        "classes_found": sorted(class_counts.keys()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MCP Server
# ═══════════════════════════════════════════════════════════════════════════════

mcp = FastMCP("CrochetDesigner")

# Lazy model loading (only load when first tool call happens)
_model = None

def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(MODEL_PATH)
    return _model


@mcp.tool()
def analyze_crochet_chart(image_path: str) -> str:
    """
    Analyzes a crochet chart image and returns a digital reconstruction.
    Args:
        image_path: The local path to the crochet chart image.
    """
    model = _get_model()
    image = cv.imread(image_path)
    if image is None:
        return json.dumps({"error": f"Could not read image: {image_path}"})

    img_h, img_w = image.shape[:2]

    # Run adaptive tiled inference
    detections = predict_adaptive(
        model, image,
        target_stitch_px=50,
        tile_size=640,
        overlap=0.25,
        conf=0.25,
        iou_threshold=0.45,
    )

    # Group into rows
    rows = _group_into_rows(detections, img_h)

    # Build all outputs
    summary = _build_summary(detections, img_w, img_h, rows)
    det_json = detections_to_json(detections, img_w, img_h)
    svg_str = detections_to_svg(detections, img_w, img_h, include_noise=False)

    # Build text representation of rows for Claude's understanding
    row_descriptions = []
    for i, row in enumerate(rows):
        stitch_seq = []
        for det in row:
            cfg = CLASS_CONFIG.get(det.cls_name, {})
            abbr = cfg.get("abbr", "??")
            stitch_seq.append(abbr)
        row_descriptions.append(f"Row {i + 1}: {', '.join(stitch_seq)}")

    result = {
        "summary": summary,
        "row_descriptions": row_descriptions,
        "detections": det_json,
        "svg": svg_str,
        "legend": {name: {"abbreviation": cfg["abbr"], "color": cfg["color"]}
                   for name, cfg in CLASS_CONFIG.items()
                   if name in summary.get("classes_found", [])},
        "notes": (
            "The SVG above is a direct reconstruction from YOLO OBB detections. "
            "You can embed it in HTML, scale it, or use the detection data to "
            "redraw the chart in any style. Each detection includes row number, "
            "position, rotation angle, and bounding box corners. "
            "The 'row_descriptions' field gives a shorthand stitch sequence per row. "
            "Noise annotations (row numbers, arrows) are excluded from the SVG "
            "but included in detections if you need them."
        ),
    }

    return json.dumps(result, indent=2)


if __name__ == "__main__":
    mcp.run()
