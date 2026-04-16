"""Central configuration for the crochet pipeline.

One source of truth for class metadata, default paths, and inference
parameters. All other modules import from here rather than redefining
constants locally.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict


# ── Paths ──────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS: str = "/Users/elevchenko/Documents/DataScience/Crochet/runs/obb/obb_train23/weights/best.pt"


# ── Class metadata ─────────────────────────────────────────────────────────
class ClassInfo(TypedDict):
    """Metadata for a single stitch class.

    ``abbr``     — short human-readable abbreviation (``"dc"``, ``"ch"``).
    ``color``    — hex color used for overlays and UI badges.
    ``svg_type`` — drawing routine used by ``crochet.rendering.svg`` to
                   emit an embeddable SVG glyph for this class.
    """

    abbr: str
    color: str
    svg_type: str


CLASS_CONFIG: dict[str, ClassInfo] = {
    "chain":         {"abbr": "ch",  "color": "#0000FF", "svg_type": "chain"},
    "double":        {"abbr": "dc",  "color": "#FF00FF", "svg_type": "tall_stitch"},
    "double treble": {"abbr": "dtr", "color": "#00AA00", "svg_type": "tall_stitch"},
    "enseble_chain": {"abbr": "ec",  "color": "#FF8000", "svg_type": "enseble_chain"},
    "fan":           {"abbr": "fa",  "color": "#FF0000", "svg_type": "fan"},
    "half_double":   {"abbr": "hd",  "color": "#00FFFF", "svg_type": "tall_stitch"},
    "noise":         {"abbr": "no",  "color": "#808080", "svg_type": "noise"},
    "single":        {"abbr": "sc",  "color": "#FFD700", "svg_type": "cross"},
    "treble":        {"abbr": "tr",  "color": "#00FF00", "svg_type": "tall_stitch"},
}

DEFAULT_COLOR: str = "#808080"
SCHEME_COLOR: str = "#000000"


# ── Inference defaults ─────────────────────────────────────────────────────
DEFAULT_CONF: float = 0.25
DEFAULT_IOU: float = 0.5
DEFAULT_TILE_SIZE: int = 640
DEFAULT_OVERLAP: float = 0.2
DEFAULT_TARGET_STITCH_PX: int = 100


# ── Rendering defaults ─────────────────────────────────────────────────────
ICON_SHRINK: float = 0.85
MIN_RENDER_HEIGHT_PX: int = 1200
MAX_RENDER_HEIGHT_PX: int = 4000


# ── Stitch-bar counts for SVG ``tall_stitch`` glyphs ───────────────────────
TALL_STITCH_BARS: dict[str, int] = {
    "half_double":   0,
    "double":        1,
    "treble":        2,
    "double treble": 3,
}


def project_root() -> Path:
    """Return the project root (the folder containing this package)."""
    return Path(__file__).resolve().parent.parent
