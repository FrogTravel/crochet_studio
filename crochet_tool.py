"""
CrochetDesigner MCP Tool — Crochet chart image → structured detection data + SVG.

Returns:
  - An annotated overlay image (ImageContent) so Claude can see the detections.
  - A compact JSON summary (TextContent) with row-grouped detections, per-row
    stitch sequences, a legend, and an embeddable SVG reconstruction.

Claude can use this output to:
  - Confirm the recognition quality at a glance (visual)
  - Compose HTML tutorials with accurate SVG diagrams (data + svg)
  - Build stitch-by-stitch instructions (row_descriptions)

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

# ── Silence ultralytics BEFORE importing anything that imports it ─────────────
# Any print to stdout corrupts MCP's JSON-RPC protocol, so we force quiet mode.
import os
os.environ["YOLO_VERBOSE"] = "False"
os.environ["ULTRALYTICS_LOG_LEVEL"] = "WARNING"

import base64
import io
import json
import logging
import math
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr

import cv2 as cv
import numpy as np

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, ImageContent

# Suppress ultralytics logger at the Python level too
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Ensure local imports work regardless of launching CWD
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from util.tiler import predict_adaptive, Detection

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PATH = os.path.join(_SCRIPT_DIR, "runs", "obb", "obb_train23", "weights", "best.pt")

# 9-class system matching data/project_yolo_obb/data.yaml
CLASS_CONFIG = {
    "chain":          {"id": 0, "abbr": "ch",  "color": "#4a6fa5", "svg_type": "chain"},
    "double":         {"id": 1, "abbr": "dc",  "color": "#8b5e83", "svg_type": "tall_stitch"},
    "double treble":  {"id": 2, "abbr": "dtr", "color": "#3a9a3a", "svg_type": "tall_stitch"},
    "enseble_chain":  {"id": 3, "abbr": "ec",  "color": "#b07a3a", "svg_type": "enseble_chain"},
    "fan":            {"id": 4, "abbr": "fa",  "color": "#c45a4a", "svg_type": "fan"},
    "half_double":    {"id": 5, "abbr": "hd",  "color": "#5a8a7a", "svg_type": "tall_stitch"},
    "noise":          {"id": 6, "abbr": "no",  "color": "#999999", "svg_type": "noise"},
    "single":         {"id": 7, "abbr": "sc",  "color": "#c4943a", "svg_type": "cross"},
    "treble":         {"id": 8, "abbr": "tr",  "color": "#5aaa5a", "svg_type": "tall_stitch"},
}

# Inference params
TARGET_STITCH_PX = 100
TILE_SIZE        = 640
TILE_OVERLAP     = 0.25
CONF_THRESHOLD   = 0.25
NMS_IOU          = 0.45

# Output size caps — keep the text payload comfortably under Claude's
# tool-response limit (~1 MB). For dense charts we skip the per-stitch
# detection list and the full SVG; the overlay image + row-level summary
# are always enough for Claude to compose a tutorial.
MAX_DETECTIONS_FULL = 400     # include compact detection list up to this many
MAX_SVG_CHARS       = 200_000 # include SVG only if it's smaller than this

# Overlay image max dimension (keeps payload reasonable)
OVERLAY_MAX_DIM = 1600
OVERLAY_JPEG_QUALITY = 80


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """'#4a6fa5' → (165, 111, 74) in BGR."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _eprint(*args, **kwargs):
    """Print to stderr only — never pollute stdout (MCP uses it for JSON-RPC)."""
    print(*args, file=sys.stderr, **kwargs)


def _resolve_image_path(image_path: str) -> str:
    """Expand ~, make absolute, and check existence.
    Returns absolute path or raises FileNotFoundError."""
    p = os.path.expanduser(image_path)
    if not os.path.isabs(p):
        # Try CWD first, then script directory, then common locations
        candidates = [
            os.path.abspath(p),
            os.path.join(_SCRIPT_DIR, p),
            os.path.join(_SCRIPT_DIR, "data", "raw", p),
            os.path.join(_SCRIPT_DIR, "data", "raw", "big", p),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        raise FileNotFoundError(
            f"Could not find image '{image_path}'. Tried: {candidates}"
        )
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Image does not exist: {p}")
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# SVG Symbol Generators (for embeddable reconstructions)
# ═══════════════════════════════════════════════════════════════════════════════

def _svg_transform(cx: float, cy: float, angle_deg: float) -> str:
    if abs(angle_deg) < 0.5:
        return ""
    return f' transform="rotate({angle_deg:.1f},{cx:.1f},{cy:.1f})"'


def svg_chain(cx, cy, w, h, angle, color):
    rx = max(w * 0.35, 3)
    ry = max(h * 0.2, 2)
    t = _svg_transform(cx, cy, angle)
    return (f'<g{t}><ellipse cx="{cx:.1f}" cy="{cy:.1f}" '
            f'rx="{rx:.1f}" ry="{ry:.1f}" fill="none" '
            f'stroke="{color}" stroke-width="1.5"/></g>')


def svg_tall_stitch(cx, cy, w, h, angle, color, n_bars=1):
    half_h = h / 2
    bar_w = w * 0.3
    t = _svg_transform(cx, cy, angle)
    parts = [f'<g{t}>',
             f'<line x1="{cx:.1f}" y1="{cy-half_h:.1f}" x2="{cx:.1f}" y2="{cy+half_h:.1f}" stroke="{color}" stroke-width="1.8"/>',
             f'<line x1="{cx-bar_w:.1f}" y1="{cy-half_h:.1f}" x2="{cx+bar_w:.1f}" y2="{cy-half_h:.1f}" stroke="{color}" stroke-width="1.8"/>']
    slash_w = w * 0.2
    if n_bars >= 1:
        parts.append(f'<line x1="{cx-slash_w:.1f}" y1="{cy+h*0.06:.1f}" x2="{cx+slash_w:.1f}" y2="{cy-h*0.06:.1f}" stroke="{color}" stroke-width="1.3"/>')
    if n_bars >= 2:
        off = h * 0.15
        parts.append(f'<line x1="{cx-slash_w:.1f}" y1="{cy+off:.1f}" x2="{cx+slash_w:.1f}" y2="{cy+off-h*0.12:.1f}" stroke="{color}" stroke-width="1.3"/>')
        parts.append(f'<line x1="{cx-slash_w:.1f}" y1="{cy-off+h*0.12:.1f}" x2="{cx+slash_w:.1f}" y2="{cy-off:.1f}" stroke="{color}" stroke-width="1.3"/>')
    if n_bars >= 3:
        parts.append(f'<line x1="{cx-slash_w:.1f}" y1="{cy+h*0.25:.1f}" x2="{cx+slash_w:.1f}" y2="{cy+h*0.13:.1f}" stroke="{color}" stroke-width="1.3"/>')
    parts.append('</g>')
    return '\n'.join(parts)


def svg_cross(cx, cy, w, h, angle, color):
    dx = w * 0.25
    dy = h * 0.25
    t = _svg_transform(cx, cy, angle)
    return (f'<g{t}>'
            f'<line x1="{cx-dx:.1f}" y1="{cy-dy:.1f}" x2="{cx+dx:.1f}" y2="{cy+dy:.1f}" stroke="{color}" stroke-width="1.8"/>'
            f'<line x1="{cx-dx:.1f}" y1="{cy+dy:.1f}" x2="{cx+dx:.1f}" y2="{cy-dy:.1f}" stroke="{color}" stroke-width="1.8"/>'
            f'</g>')


def svg_fan(cx, cy, w, h, angle, color):
    half_h = h / 2
    spread = w * 0.4
    n_spokes = 5
    t = _svg_transform(cx, cy, angle)
    parts = [f'<g{t}>']
    base_y = cy + half_h
    for i in range(n_spokes):
        frac = i / (n_spokes - 1)
        tip_x = cx - spread + 2 * spread * frac
        tip_y = cy - half_h
        parts.append(f'<line x1="{cx:.1f}" y1="{base_y:.1f}" x2="{tip_x:.1f}" y2="{tip_y:.1f}" stroke="{color}" stroke-width="1.8"/>')
    arc_left_x = cx - spread
    arc_right_x = cx + spread
    arc_y = cy - half_h
    cp_y = arc_y - h * 0.12
    parts.append(f'<path d="M{arc_left_x:.1f},{arc_y:.1f} Q{cx:.1f},{cp_y:.1f} {arc_right_x:.1f},{arc_y:.1f}" fill="none" stroke="{color}" stroke-width="1.2"/>')
    parts.append('</g>')
    return '\n'.join(parts)


def svg_enseble_chain(cx, cy, w, h, angle, color):
    n_ovals = max(2, round(h / max(w * 0.6, 8)))
    oval_h = h / n_ovals
    rx = max(w * 0.25, 3)
    ry = max(oval_h * 0.35, 2)
    t = _svg_transform(cx, cy, angle)
    parts = [f'<g{t}>']
    for i in range(n_ovals):
        oy = cy - h / 2 + oval_h * (i + 0.5)
        parts.append(f'<ellipse cx="{cx:.1f}" cy="{oy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" fill="none" stroke="{color}" stroke-width="1.2"/>')
    parts.append('</g>')
    return '\n'.join(parts)


def svg_noise(cx, cy, w, h, angle, color):
    r = max(min(w, h) * 0.2, 2)
    return f'<g><circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="none" stroke="{color}" stroke-width="1" opacity="0.5"/></g>'


SVG_DRAWERS = {
    "chain":          svg_chain,
    "tall_stitch":    svg_tall_stitch,
    "cross":          svg_cross,
    "fan":            svg_fan,
    "enseble_chain":  svg_enseble_chain,
    "noise":          svg_noise,
}


def _n_bars_for_class(cls_name: str) -> int:
    return {"half_double": 0, "double": 1, "treble": 2, "double treble": 3}.get(cls_name, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Geometry & row grouping
# ═══════════════════════════════════════════════════════════════════════════════

def _det_geometry(det: Detection) -> dict:
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


def _group_into_rows(detections, img_h):
    if not detections:
        return []
    dets_with_cy = [(d, d.corners.mean(axis=0)[1]) for d in detections]
    dets_with_cy.sort(key=lambda x: x[1])
    heights = []
    for d, _ in dets_with_cy:
        a = float(np.linalg.norm(d.corners[0] - d.corners[1]))
        b = float(np.linalg.norm(d.corners[1] - d.corners[2]))
        heights.append(max(a, b))
    median_h = sorted(heights)[len(heights) // 2] if heights else 30
    row_gap = median_h * 0.5

    rows, current_row, current_cy = [], [dets_with_cy[0][0]], dets_with_cy[0][1]
    for det, cy in dets_with_cy[1:]:
        if abs(cy - current_cy) > row_gap:
            rows.append(current_row)
            current_row = [det]
            current_cy = cy
        else:
            current_row.append(det)
            current_cy = current_cy * 0.8 + cy * 0.2
    if current_row:
        rows.append(current_row)
    for row in rows:
        row.sort(key=lambda d: d.corners.mean(axis=0)[0])
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Overlay (annotated image)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_overlay(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw detection OBBs + labels on a copy of the image."""
    out = image.copy()
    for det in detections:
        cfg = CLASS_CONFIG.get(det.cls_name)
        if cfg is None:
            continue
        color = _hex_to_bgr(cfg["color"])
        pts = det.corners.astype(np.int32).reshape((-1, 1, 2))
        cv.polylines(out, [pts], isClosed=True, color=color, thickness=2)

        # Small text label at the top-left corner of the OBB
        tl = tuple(det.corners[0].astype(int))
        cv.putText(out, cfg["abbr"], (tl[0], max(tl[1] - 4, 10)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv.LINE_AA)
    return out


def _encode_overlay(overlay: np.ndarray) -> str:
    """Resize if too large, JPEG-encode, return base64 string."""
    h, w = overlay.shape[:2]
    max_side = max(h, w)
    if max_side > OVERLAY_MAX_DIM:
        scale = OVERLAY_MAX_DIM / max_side
        overlay = cv.resize(overlay, (int(w * scale), int(h * scale)),
                            interpolation=cv.INTER_AREA)
    ok, buf = cv.imencode(".jpg", overlay,
                          [cv.IMWRITE_JPEG_QUALITY, OVERLAY_JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ═══════════════════════════════════════════════════════════════════════════════
# Compact detection output
# ═══════════════════════════════════════════════════════════════════════════════

def _compact_detections(rows, skip_noise=True):
    """Compact per-detection records (no verbose corners)."""
    out = []
    for row_idx, row in enumerate(rows):
        for det in row:
            if skip_noise and det.cls_name == "noise":
                continue
            cfg = CLASS_CONFIG.get(det.cls_name, {})
            g = _det_geometry(det)
            out.append({
                "cls":  det.cls_name,
                "ab":   cfg.get("abbr", "??"),
                "row":  row_idx,
                "cx":   round(g["cx"], 1),
                "cy":   round(g["cy"], 1),
                "w":    round(g["w"], 1),
                "h":    round(g["h"], 1),
                "ang":  round(g["angle"], 1),
                "conf": round(float(det.confidence), 2),
            })
    return out


def _compress_row(row, skip_noise=True):
    """Run-length encode a row of stitches into a compact string like
    '3ch, 12dc, 2sc, dc, 4ch'. Much shorter than listing every stitch."""
    tokens = []
    prev, run = None, 0
    for det in row:
        if skip_noise and det.cls_name == "noise":
            continue
        abbr = CLASS_CONFIG.get(det.cls_name, {}).get("abbr", "?")
        if abbr == prev:
            run += 1
        else:
            if prev is not None:
                tokens.append(f"{run}{prev}" if run > 1 else prev)
            prev, run = abbr, 1
    if prev is not None:
        tokens.append(f"{run}{prev}" if run > 1 else prev)
    return ", ".join(tokens)


def _detections_to_svg(detections, img_w, img_h, include_noise=False):
    parts = [
        f'<svg width="{img_w}" height="{img_h}" viewBox="0 0 {img_w} {img_h}" xmlns="http://www.w3.org/2000/svg">',
        f'<rect width="{img_w}" height="{img_h}" fill="#faf8f4"/>',
    ]
    for det in detections:
        cfg = CLASS_CONFIG.get(det.cls_name)
        if cfg is None:
            continue
        if cfg["svg_type"] == "noise" and not include_noise:
            continue
        g = _det_geometry(det)
        drawer = SVG_DRAWERS.get(cfg["svg_type"])
        if drawer is None:
            continue
        kwargs = dict(cx=g["cx"], cy=g["cy"], w=g["w"], h=g["h"],
                      angle=g["angle"], color=cfg["color"])
        if cfg["svg_type"] == "tall_stitch":
            kwargs["n_bars"] = _n_bars_for_class(det.cls_name)
        parts.append(drawer(**kwargs))
    parts.append('</svg>')
    return '\n'.join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# MCP Server
# ═══════════════════════════════════════════════════════════════════════════════

mcp = FastMCP("CrochetDesigner")

_model = None


def _get_model():
    """Lazy model load. Silences ultralytics output during load."""
    global _model
    if _model is None:
        _eprint(f"[CrochetDesigner] Loading model: {MODEL_PATH}")
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(
                f"Model weights not found at {MODEL_PATH}. "
                f"Train the model first or update MODEL_PATH in crochet_tool.py."
            )
        from ultralytics import YOLO
        # Redirect both stdout and stderr during load to avoid polluting MCP JSON-RPC
        with redirect_stdout(sys.stderr), redirect_stderr(sys.stderr):
            _model = YOLO(MODEL_PATH)
        _eprint(f"[CrochetDesigner] Model loaded.")
    return _model


def _load_image(image_path: str, image_base64: str) -> tuple[np.ndarray, str]:
    """Load an image from either a filesystem path or base64-encoded bytes.

    Claude's sandbox (`/mnt/user-data/uploads/...`) is NOT on the same filesystem
    as the machine running this MCP server, so `image_path` only works when the
    user points at a file on the server's own disk. For uploaded chat images,
    Claude should read the file in its sandbox, base64-encode the content, and
    pass it here as `image_base64`.

    Returns (image_bgr, source_description).
    """
    if image_base64:
        b64 = image_base64.strip()
        # Strip optional data-URI prefix like "data:image/png;base64,..."
        if b64.startswith("data:"):
            _, _, b64 = b64.partition(",")
        try:
            raw = base64.b64decode(b64, validate=False)
        except Exception as e:
            raise ValueError(f"base64 decode failed: {e}")
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv.imdecode(arr, cv.IMREAD_COLOR)
        if img is None:
            raise ValueError(
                "Could not decode base64 image. Expected PNG/JPEG/etc bytes."
            )
        return img, f"<base64 {len(raw)} bytes>"

    if image_path:
        resolved = _resolve_image_path(image_path)
        img = cv.imread(resolved)
        if img is None:
            raise ValueError(f"Could not decode image: {resolved}")
        return img, resolved

    raise ValueError(
        "Must provide either `image_path` (file on the server's filesystem) "
        "or `image_base64` (base64-encoded image content). If you uploaded "
        "the image in this chat, read its bytes and pass them as image_base64."
    )


@mcp.tool()
def analyze_crochet_chart(image_base64: str = "", image_path: str = "") -> list:
    """Analyze a crochet chart image and return an annotated overlay plus
    structured detection data (class per stitch, row grouping, SVG, etc).

    ── HOW TO PASS THE IMAGE ──────────────────────────────────────────────────
    PREFERRED — use `image_base64` when the user uploaded/attached an image in
    the chat (i.e. the image is NOT on the MCP server's filesystem). Read the
    file's raw bytes and base64-encode them before calling. A `data:image/...
    ;base64,` URI prefix is accepted but not required. This is the path that
    works for uploaded chat images; do NOT try to "copy the file first" or
    pass a /mnt/user-data/... path — those paths don't exist on the server.

    Use `image_path` ONLY when the user explicitly references a file that
    already lives on the same machine as this MCP server (e.g. a file under
    the project's own `data/raw/` directory).

    Example of the correct call for an uploaded chat image, from Python:
        import base64
        with open("/path/claude/can/read", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        analyze_crochet_chart(image_base64=b64)

    ── RETURNS ────────────────────────────────────────────────────────────────
    A list with:
      - ImageContent: JPEG overlay (original chart + OBB boxes + class labels).
      - TextContent: JSON with summary, row_descriptions, detections, legend, svg.

    Args:
        image_base64: Base64-encoded PNG/JPEG bytes of the chart image. Use
            THIS for anything the user uploaded into the chat.
        image_path: Absolute or relative path to the image on the server's
            filesystem. Only for files the server can actually read.
    """
    try:
        # Load image from either source
        image, source = _load_image(image_path, image_base64)
        _eprint(f"[CrochetDesigner] analyzing: {source}")
        img_h, img_w = image.shape[:2]

        # Model inference (silence stdout to protect JSON-RPC)
        model = _get_model()
        with redirect_stdout(sys.stderr):
            detections = predict_adaptive(
                model, image,
                target_stitch_px=TARGET_STITCH_PX,
                tile_size=TILE_SIZE,
                overlap=TILE_OVERLAP,
                conf=CONF_THRESHOLD,
                iou_threshold=NMS_IOU,
            )

        _eprint(f"[CrochetDesigner] {len(detections)} detections found.")

        # Group & summarize
        rows = _group_into_rows(detections, img_h)
        from collections import Counter
        class_counts = Counter(d.cls_name for d in detections)
        # Counts for stitches only (excluding noise annotations)
        stitch_counts = Counter(d.cls_name for d in detections
                                if d.cls_name != "noise")

        summary = {
            "total_detections": len(detections),
            "stitch_count":     sum(stitch_counts.values()),
            "noise_count":      class_counts.get("noise", 0),
            "image_size":       {"width": img_w, "height": img_h},
            "num_rows":         len(rows),
            "class_counts":     dict(stitch_counts),
        }

        # Row descriptions — run-length compressed, noise excluded.
        # For a dense chart this can shrink a 150-stitch row from
        # "ch, ch, ch, dc, dc, dc, sc, sc, ch, ..." to "3ch, 3dc, 2sc, ch, ...".
        row_descriptions = []
        for i, row in enumerate(rows):
            compressed = _compress_row(row, skip_noise=True)
            if compressed:
                row_descriptions.append(f"Row {i + 1}: {compressed}")

        legend = {name: {"abbr": cfg["abbr"], "color": cfg["color"]}
                  for name, cfg in CLASS_CONFIG.items()
                  if name in stitch_counts}

        # Make annotated overlay for visual feedback (always include)
        overlay = _make_overlay(image, detections)
        overlay_b64 = _encode_overlay(overlay)

        # Payload (build up gradually, skip large sections for dense charts)
        text_payload = {
            "summary":          summary,
            "row_descriptions": row_descriptions,
            "legend":           legend,
        }

        # Per-stitch detection list — only for small/medium charts
        n_stitches = summary["stitch_count"]
        if n_stitches <= MAX_DETECTIONS_FULL:
            text_payload["detections"] = _compact_detections(rows, skip_noise=True)
        else:
            text_payload["detections_note"] = (
                f"Per-stitch list omitted for a dense chart ({n_stitches} stitches). "
                f"Use row_descriptions + overlay image instead. "
                f"Re-run with CONF_THRESHOLD raised in crochet_tool.py if the model "
                f"is over-detecting."
            )

        # SVG reconstruction — only include if it fits comfortably
        svg_str = _detections_to_svg(detections, img_w, img_h, include_noise=False)
        if len(svg_str) <= MAX_SVG_CHARS:
            text_payload["svg"] = svg_str
        else:
            text_payload["svg_note"] = (
                f"SVG omitted ({len(svg_str)} chars > {MAX_SVG_CHARS} limit). "
                f"The overlay image + row_descriptions are sufficient for writing "
                f"the tutorial; for a diagram, redraw rows stylistically rather "
                f"than mirroring every stitch 1:1."
            )

        text_payload["notes"] = (
            "Overlay image = original chart with detected OBBs + class labels. "
            "`row_descriptions` use run-length notation (e.g. '3ch, 5dc' = "
            "3 chains then 5 double crochets). Rows run bottom-to-top in the "
            "detected order. Noise annotations are filtered out of all lists. "
            "Legend maps class abbreviations to hex colors."
        )
        text_json = json.dumps(text_payload, indent=1, ensure_ascii=False)
        _eprint(f"[CrochetDesigner] text payload: {len(text_json)} chars, "
                f"image: {len(overlay_b64)} b64 chars")

        return [
            ImageContent(type="image", data=overlay_b64, mimeType="image/jpeg"),
            TextContent(type="text", text=text_json),
        ]

    except FileNotFoundError as e:
        return [TextContent(type="text",
                            text=json.dumps({"error": str(e)}))]
    except Exception as e:
        tb = traceback.format_exc()
        _eprint(f"[CrochetDesigner] ERROR: {e}\n{tb}")
        return [TextContent(type="text",
                            text=json.dumps({"error": f"{type(e).__name__}: {e}"}))]


if __name__ == "__main__":
    mcp.run()
