"""Single source of truth for class metadata and default hyperparameters.

Every module that needs to know about classes, colours, or default
inference / training parameters imports them from here. Changing a
constant here propagates through training, inference, the renderer,
and the Streamlit UI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, TypedDict


# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
NOTEBOOKS_DIR: Final[Path] = PROJECT_ROOT / "notebooks"
DEFAULT_WEIGHTS: Final[Path] = PROJECT_ROOT / "models" / "run23_yolo_obb_v8n_best.pt"


# ── Class metadata ─────────────────────────────────────────────────────────
class ClassInfo(TypedDict):
    """Metadata describing a single stitch class.

    Attributes:
        abbr: Short human-readable abbreviation (e.g. ``"dc"``).
        color: Hex colour string used for overlays and UI badges.
        svg_type: Drawing routine name dispatched in :mod:`rendering`.
    """

    abbr: str
    color: str
    svg_type: str


CLASS_CONFIG: Final[dict[str, ClassInfo]] = {
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
"""Per-class metadata, ordered identically to ``data.yaml``."""

CLASS_MAP: Final[dict[str, int]] = {name: i for i, name in enumerate(CLASS_CONFIG)}
"""Forward mapping ``class name -> integer id``."""

ID_TO_NAME: Final[dict[int, str]] = {i: name for name, i in CLASS_MAP.items()}
"""Reverse mapping ``integer id -> class name``."""

NUM_CLASSES: Final[int] = len(CLASS_CONFIG)
DEFAULT_COLOR: Final[str] = "#808080"
SCHEME_COLOR: Final[str] = "#000000"


# ── Inference defaults ─────────────────────────────────────────────────────
DEFAULT_CONF: Final[float] = 0.25
DEFAULT_IOU: Final[float] = 0.5
DEFAULT_TILE_SIZE: Final[int] = 640
DEFAULT_OVERLAP: Final[float] = 0.2
DEFAULT_TARGET_STITCH_PX: Final[int] = 100


# ── Training defaults ──────────────────────────────────────────────────────
DEFAULT_BASE_WEIGHTS: Final[str] = "run23_yolo_obb_v8n_best.pt"
DEFAULT_EPOCHS_LOCAL: Final[int] = 10
DEFAULT_EPOCHS_REMOTE: Final[int] = 150
DEFAULT_BATCH_LOCAL: Final[int] = 4
DEFAULT_BATCH_REMOTE: Final[int] = 16
DEFAULT_IMGSZ: Final[int] = 640
DEFAULT_DEGREES: Final[float] = 180.0


# ── Synthetic-data palettes ────────────────────────────────────────────────
PALETTES: Final[dict[str, dict[str, tuple[int, int, int]]]] = {
    "green":  {"fg": (40, 120, 60),  "bg": (245, 248, 240)},
    "purple": {"fg": (90, 50, 130),  "bg": (248, 244, 252)},
    "pink":   {"fg": (160, 50, 80),  "bg": (252, 245, 248)},
    "black":  {"fg": (30, 30, 30),   "bg": (255, 255, 255)},
    "blue":   {"fg": (50, 70, 150),  "bg": (245, 248, 255)},
}
"""Foreground / background colour pairs used by the synthetic generator."""


# ── Stitch-bar counts for SVG ``tall_stitch`` glyphs ───────────────────────
TALL_STITCH_BARS: Final[dict[str, int]] = {
    "half_double":   0,
    "double":        1,
    "treble":        2,
    "double treble": 3,
}
"""Number of diagonal bars drawn on each ``tall_stitch``-style glyph."""
