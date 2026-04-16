"""CLI: generate a crochet diagram with Gemini, then classify stitches.

Examples
--------
From the repo root (must contain ``runs/obb/obb_train23``)::

    export GEMINI_API_KEY=...
    python scripts/run_pipeline.py \\
        --image-out free_output.png \\
        --figure-out free_output_predictions.png

Re-run inference on an existing image (no Gemini call)::

    python scripts/run_pipeline.py --skip-generation --image-out free_output.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crochet.config import (  # noqa: E402
    DEFAULT_CONF,
    DEFAULT_IOU,
    DEFAULT_OVERLAP,
    DEFAULT_TARGET_STITCH_PX,
    DEFAULT_TILE_SIZE,
    DEFAULT_WEIGHTS,
)
from crochet.generation import DEFAULT_MODEL, DEFAULT_PROMPT  # noqa: E402
from crochet.pipeline import run_pipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="Gemini prompt")
    p.add_argument("--image-out", default="free_output.png",
                   help="Where to save the generated image")
    p.add_argument("--figure-out", default=None,
                   help="Where to save the detection figure (default: next to image)")
    p.add_argument("--tile-figure", default=None,
                   help="Where to save the tile-grid figure (default: next to image)")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to YOLO OBB weights")
    p.add_argument("--gemini-model", default=DEFAULT_MODEL, help="Gemini image model name")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF, help="YOLO confidence threshold")
    p.add_argument("--iou", type=float, default=DEFAULT_IOU, help="NMS IoU threshold")
    p.add_argument("--target-stitch-px", type=int, default=DEFAULT_TARGET_STITCH_PX,
                   help="Target stitch short-side size at training resolution")
    p.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE, help="YOLO input tile size")
    p.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP, help="Tile overlap fraction")
    p.add_argument("--no-tile-viz", action="store_true", help="Skip tile-grid visualization")
    p.add_argument("--api-key", default=None,
                   help="Gemini API key (falls back to GEMINI_API_KEY env var)")
    p.add_argument("--skip-generation", action="store_true",
                   help="Reuse an existing image at --image-out instead of calling Gemini")
    return p


def main() -> None:
    args = build_parser().parse_args()
    summary = run_pipeline(
        prompt=args.prompt,
        image_out=args.image_out,
        figure_out=args.figure_out,
        tile_figure=args.tile_figure,
        weights_path=args.weights,
        gemini_model=args.gemini_model,
        conf=args.conf,
        iou_threshold=args.iou,
        target_stitch_px=args.target_stitch_px,
        tile_size=args.tile_size,
        overlap=args.overlap,
        visualize_tiles=not args.no_tile_viz,
        api_key=args.api_key,
        skip_generation=args.skip_generation,
    )
    print(f"[run_pipeline] Done. {summary['num_detections']} stitches detected.")


if __name__ == "__main__":
    main()
