"""Crochet stitch generation + classification pipeline."""

from .generate_image import generate_image, DEFAULT_PROMPT, DEFAULT_MODEL
from .classify_stitches import (
    classify_stitches,
    predict_tiled,
    predict_adaptive,
    estimate_stitch_size,
    draw_svg_icon,
    Detection,
    DEFAULT_WEIGHTS,
    CLASS_CONFIG,
)
from .run_pipeline import run_pipeline

__all__ = [
    "generate_image",
    "classify_stitches",
    "predict_tiled",
    "predict_adaptive",
    "estimate_stitch_size",
    "draw_svg_icon",
    "Detection",
    "run_pipeline",
    "DEFAULT_PROMPT",
    "DEFAULT_MODEL",
    "DEFAULT_WEIGHTS",
    "CLASS_CONFIG",
]
