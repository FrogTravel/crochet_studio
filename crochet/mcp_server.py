"""CrochetDesigner MCP tool — image → detections + SVG reconstruction.

Returns:
  - ImageContent: annotated overlay so Claude can see the detections.
  - TextContent:  JSON with summary, row-grouped sequences, legend, SVG.

Claude can use the output to confirm recognition quality, compose HTML
tutorials with accurate SVG diagrams, and build stitch-by-stitch
instructions.

Setup:
  Add to Claude Desktop config
  (``~/Library/Application Support/Claude/claude_desktop_config.json``)::

      {
        "mcpServers": {
          "CrochetDesigner": {
            "command": "python",
            "args": ["-m", "crochet.mcp_server"]
          }
        }
      }
"""

from __future__ import annotations

# Silence ultralytics BEFORE importing anything that imports it.
# Any stdout write corrupts MCP's JSON-RPC protocol.
import os
os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("ULTRALYTICS_LOG_LEVEL", "WARNING")

import base64
import json
import logging
import sys
import traceback
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout

import cv2 as cv
import numpy as np
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

from .config import (
    CLASS_CONFIG,
    DEFAULT_CONF,
    DEFAULT_TARGET_STITCH_PX,
    DEFAULT_TILE_SIZE,
    DEFAULT_WEIGHTS,
    project_root,
)
from .detection import Detection, predict_adaptive
from .rendering.svg import detections_to_svg

logging.getLogger("ultralytics").setLevel(logging.ERROR)


# ── Configuration ──────────────────────────────────────────────────────────
MODEL_PATH: str = str(project_root() / DEFAULT_WEIGHTS)

TILE_OVERLAP = 0.25
NMS_IOU = 0.45

# Output size caps — keep the text payload comfortably under the
# tool-response limit (~1 MB). For dense charts we skip the per-stitch
# detection list and the full SVG; the overlay image + row-level summary
# are always enough for Claude to compose a tutorial.
MAX_DETECTIONS_FULL = 400
MAX_SVG_CHARS = 200_000

# Overlay image caps (keeps payload reasonable).
OVERLAY_MAX_DIM = 1600
OVERLAY_JPEG_QUALITY = 80


# ── Logging helpers ────────────────────────────────────────────────────────
def _eprint(*args, **kwargs) -> None:
    """Print to stderr only — never pollute stdout (MCP uses it for JSON-RPC)."""
    print(*args, file=sys.stderr, **kwargs)


# ── Colour / path helpers ──────────────────────────────────────────────────
def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """``'#4a6fa5'`` → ``(165, 111, 74)`` (OpenCV BGR tuple)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return b, g, r


def _resolve_image_path(image_path: str) -> str:
    """Expand ``~``, resolve relative → absolute, and assert existence."""
    p = os.path.expanduser(image_path)
    if os.path.isabs(p):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Image does not exist: {p}")
        return p

    root = str(project_root())
    candidates = [
        os.path.abspath(p),
        os.path.join(root, p),
        os.path.join(root, "data", "raw", p),
        os.path.join(root, "data", "raw", "big", p),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f"Could not find image '{image_path}'. Tried: {candidates}"
    )


def _load_image(image_path: str, image_base64: str) -> tuple[np.ndarray, str]:
    """Load an image from either a filesystem path or base64-encoded bytes."""
    if image_base64:
        b64 = image_base64.strip()
        if b64.startswith("data:"):
            _, _, b64 = b64.partition(",")
        try:
            raw = base64.b64decode(b64, validate=False)
        except Exception as exc:
            raise ValueError(f"base64 decode failed: {exc}") from exc
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv.imdecode(arr, cv.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode base64 image. Expected PNG/JPEG/etc bytes.")
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


# ── Row grouping ───────────────────────────────────────────────────────────
def _group_into_rows(detections: list[Detection]) -> list[list[Detection]]:
    if not detections:
        return []
    dets_with_cy = [(d, d.corners.mean(axis=0)[1]) for d in detections]
    dets_with_cy.sort(key=lambda x: x[1])

    heights: list[float] = []
    for d, _ in dets_with_cy:
        a = float(np.linalg.norm(d.corners[0] - d.corners[1]))
        b = float(np.linalg.norm(d.corners[1] - d.corners[2]))
        heights.append(max(a, b))
    median_h = sorted(heights)[len(heights) // 2] if heights else 30.0
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
            current_cy = current_cy * 0.8 + cy * 0.2
    if current_row:
        rows.append(current_row)
    for row in rows:
        row.sort(key=lambda d: d.corners.mean(axis=0)[0])
    return rows


# ── Overlay + encoding ─────────────────────────────────────────────────────
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
        tl = tuple(det.corners[0].astype(int))
        cv.putText(out, cfg["abbr"], (tl[0], max(tl[1] - 4, 10)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv.LINE_AA)
    return out


def _encode_overlay(overlay: np.ndarray) -> str:
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


# ── Compact payload helpers ────────────────────────────────────────────────
def _det_geometry(det: Detection) -> dict[str, float]:
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


def _compact_detections(
    rows: list[list[Detection]],
    skip_noise: bool = True,
) -> list[dict]:
    out: list[dict] = []
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


def _compress_row(row: list[Detection], skip_noise: bool = True) -> str:
    """Run-length encode a row: ``'3ch, 12dc, 2sc, dc, 4ch'``."""
    tokens: list[str] = []
    prev: str | None = None
    run = 0
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


# ── MCP server ─────────────────────────────────────────────────────────────
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
                "Train the model first or update DEFAULT_WEIGHTS in crochet/config.py."
            )
        from ultralytics import YOLO
        # Redirect both stdout and stderr during load to avoid polluting JSON-RPC.
        with redirect_stdout(sys.stderr), redirect_stderr(sys.stderr):
            _model = YOLO(MODEL_PATH)
        _eprint("[CrochetDesigner] Model loaded.")
    return _model


@mcp.tool()
def analyze_crochet_chart(image_base64: str = "", image_path: str = "") -> list:
    """Analyze a crochet chart image and return an annotated overlay plus
    structured detection data (class per stitch, row grouping, SVG, etc).

    ── HOW TO PASS THE IMAGE ──────────────────────────────────────────────
    PREFERRED — use ``image_base64`` when the user uploaded/attached an
    image in the chat (i.e. the image is NOT on the MCP server's
    filesystem). Read the file's raw bytes and base64-encode them before
    calling. A ``data:image/...;base64,`` URI prefix is accepted but not
    required. This is the path that works for uploaded chat images; do
    NOT try to "copy the file first" or pass a ``/mnt/user-data/...``
    path — those paths don't exist on the server.

    Use ``image_path`` ONLY when the user explicitly references a file
    that already lives on the same machine as this MCP server.

    Example call for an uploaded chat image, from Python::

        import base64
        with open("/path/claude/can/read", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        analyze_crochet_chart(image_base64=b64)

    ── RETURNS ────────────────────────────────────────────────────────────
    A list with:
      - ImageContent: JPEG overlay (original chart + OBB boxes + labels).
      - TextContent:  JSON with summary, row_descriptions, detections,
                      legend, svg.
    """
    try:
        image, source = _load_image(image_path, image_base64)
        _eprint(f"[CrochetDesigner] analyzing: {source}")
        img_h, img_w = image.shape[:2]

        model = _get_model()
        with redirect_stdout(sys.stderr):
            detections = predict_adaptive(
                model, image,
                target_stitch_px=DEFAULT_TARGET_STITCH_PX,
                tile_size=DEFAULT_TILE_SIZE,
                overlap=TILE_OVERLAP,
                conf=DEFAULT_CONF,
                iou_threshold=NMS_IOU,
            )
        _eprint(f"[CrochetDesigner] {len(detections)} detections found.")

        rows = _group_into_rows(detections)
        class_counts = Counter(d.cls_name for d in detections)
        stitch_counts = Counter(d.cls_name for d in detections if d.cls_name != "noise")

        summary = {
            "total_detections": len(detections),
            "stitch_count":     sum(stitch_counts.values()),
            "noise_count":      class_counts.get("noise", 0),
            "image_size":       {"width": img_w, "height": img_h},
            "num_rows":         len(rows),
            "class_counts":     dict(stitch_counts),
        }

        row_descriptions: list[str] = []
        for i, row in enumerate(rows):
            compressed = _compress_row(row, skip_noise=True)
            if compressed:
                row_descriptions.append(f"Row {i + 1}: {compressed}")

        legend = {
            name: {"abbr": cfg["abbr"], "color": cfg["color"]}
            for name, cfg in CLASS_CONFIG.items()
            if name in stitch_counts
        }

        overlay_b64 = _encode_overlay(_make_overlay(image, detections))

        text_payload: dict[str, object] = {
            "summary":          summary,
            "row_descriptions": row_descriptions,
            "legend":           legend,
        }

        n_stitches = summary["stitch_count"]
        if n_stitches <= MAX_DETECTIONS_FULL:
            text_payload["detections"] = _compact_detections(rows, skip_noise=True)
        else:
            text_payload["detections_note"] = (
                f"Per-stitch list omitted for a dense chart ({n_stitches} stitches). "
                "Use row_descriptions + overlay image instead."
            )

        svg_str = detections_to_svg(detections, img_w, img_h, include_noise=False)
        if len(svg_str) <= MAX_SVG_CHARS:
            text_payload["svg"] = svg_str
        else:
            text_payload["svg_note"] = (
                f"SVG omitted ({len(svg_str)} chars > {MAX_SVG_CHARS} limit). "
                "The overlay image + row_descriptions are sufficient for writing "
                "the tutorial; for a diagram, redraw rows stylistically rather "
                "than mirroring every stitch 1:1."
            )

        text_payload["notes"] = (
            "Overlay image = original chart with detected OBBs + class labels. "
            "`row_descriptions` use run-length notation (e.g. '3ch, 5dc' = 3 "
            "chains then 5 double crochets). Rows run bottom-to-top in the "
            "detected order. Noise annotations are filtered out of all lists. "
            "Legend maps class abbreviations to hex colors."
        )
        text_json = json.dumps(text_payload, indent=1, ensure_ascii=False)
        _eprint(
            f"[CrochetDesigner] text payload: {len(text_json)} chars, "
            f"image: {len(overlay_b64)} b64 chars"
        )

        return [
            ImageContent(type="image", data=overlay_b64, mimeType="image/jpeg"),
            TextContent(type="text", text=text_json),
        ]

    except FileNotFoundError as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]
    except Exception as exc:
        tb = traceback.format_exc()
        _eprint(f"[CrochetDesigner] ERROR: {exc}\n{tb}")
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"{type(exc).__name__}: {exc}"}),
        )]


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
