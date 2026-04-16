"""Crochet stitch detection and scheme reconstruction package.

Public API
----------
``Detection``                     — OBB detection dataclass.
``predict_adaptive``              — adaptive tiled YOLO OBB inference.
``predict_adaptive_with_progress``— same, with Streamlit-friendly callbacks.
``classify_stitches``             — run inference + save result figures.
``run_pipeline``                  — generate image with Gemini, then classify.
``detections_to_json``            — JSON serialisation for detection lists.
``detections_to_svg``             — SVG reconstruction (used by MCP server).
``generate_image``                — Google Gemini image wrapper.
``CLASS_CONFIG``, ``DEFAULT_WEIGHTS`` — central configuration.
"""

from .config import (
    CLASS_CONFIG,
    DEFAULT_COLOR,
    DEFAULT_CONF,
    DEFAULT_IOU,
    DEFAULT_OVERLAP,
    DEFAULT_TARGET_STITCH_PX,
    DEFAULT_TILE_SIZE,
    DEFAULT_WEIGHTS,
    SCHEME_COLOR,
)
from .detection import (
    Detection,
    estimate_stitch_size,
    nms,
    predict_adaptive,
    predict_adaptive_with_progress,
    predict_tiled,
)
from .generation import DEFAULT_MODEL, DEFAULT_PROMPT, generate_image
from .json_export import class_counts, detections_to_json
from .pipeline import classify_stitches, run_pipeline
from .rendering.svg import detections_to_svg

__all__ = [
    # Config
    "CLASS_CONFIG", "DEFAULT_WEIGHTS", "DEFAULT_COLOR", "SCHEME_COLOR",
    "DEFAULT_CONF", "DEFAULT_IOU", "DEFAULT_OVERLAP",
    "DEFAULT_TARGET_STITCH_PX", "DEFAULT_TILE_SIZE",
    # Detection
    "Detection", "estimate_stitch_size", "nms",
    "predict_adaptive", "predict_adaptive_with_progress", "predict_tiled",
    # Generation
    "generate_image", "DEFAULT_MODEL", "DEFAULT_PROMPT",
    # Pipeline
    "classify_stitches", "run_pipeline",
    # Export
    "class_counts", "detections_to_json", "detections_to_svg",
]
