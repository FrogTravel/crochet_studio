"""End-to-end pipeline: generate a crochet diagram with Gemini, then
classify the stitches on it with the YOLOv8n OBB best weights, using
the adaptive tiled inference logic from ``Google_colab_YOLO_OBB_pipeline.ipynb``.

Usage
-----
From the repo root (the folder that contains ``runs/obb/obb_train23``):

    export GEMINI_API_KEY=...
    python -m pipeline.run_pipeline \\
        --image-out free_output.png \\
        --figure-out free_output_predictions.png

Re-run inference on an existing image:

    python -m pipeline.run_pipeline --skip-generation --image-out free_output.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .generate_image import DEFAULT_MODEL, DEFAULT_PROMPT, generate_image
from .classify_stitches import DEFAULT_WEIGHTS, classify_stitches


def run_pipeline(
    prompt: str = DEFAULT_PROMPT,
    image_out: str | Path = "free_output.png",
    figure_out: str | Path | None = None,
    tile_figure: str | Path | None = None,
    weights_path: str | Path = DEFAULT_WEIGHTS,
    gemini_model: str = DEFAULT_MODEL,
    conf: float = 0.25,
    iou_threshold: float = 0.5,
    target_stitch_px: int = 100,
    tile_size: int = 640,
    overlap: float = 0.2,
    visualize_tiles: bool = True,
    api_key: str | None = None,
    show: bool = False,
    skip_generation: bool = False,
) -> dict:
    """Run the full generate-then-classify pipeline with adaptive tiling.

    If ``skip_generation`` is True the existing file at ``image_out`` is
    reused (handy for iterating on inference without burning Gemini quota).
    """
    image_out = Path(image_out)

    if skip_generation:
        if not image_out.exists():
            raise FileNotFoundError(
                f"--skip-generation was set but {image_out} does not exist."
            )
        print(f"[run_pipeline] Skipping generation, using {image_out}")
    else:
        generate_image(
            prompt=prompt,
            output_path=image_out,
            model=gemini_model,
            api_key=api_key,
        )

    summary = classify_stitches(
        image_path=image_out,
        weights_path=weights_path,
        conf=conf,
        iou_threshold=iou_threshold,
        target_stitch_px=target_stitch_px,
        tile_size=tile_size,
        overlap=overlap,
        output_figure=figure_out,
        tile_figure=tile_figure,
        visualize_tiles=visualize_tiles,
        show=show,
    )

    print(f"[run_pipeline] Done. {summary['num_detections']} stitches detected.")
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="Gemini generation prompt")
    p.add_argument("--image-out", default="free_output.png", help="Where to save the generated image")
    p.add_argument("--figure-out", default=None,
                   help="Where to save the detection figure (default: next to the image)")
    p.add_argument("--tile-figure", default=None,
                   help="Where to save the tile-grid figure (default: next to the image)")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to YOLO OBB weights")
    p.add_argument("--gemini-model", default=DEFAULT_MODEL, help="Gemini image model name")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    p.add_argument("--target-stitch-px", type=int, default=100,
                   help="Target stitch short-side size at training resolution")
    p.add_argument("--tile-size", type=int, default=640, help="YOLO input tile size")
    p.add_argument("--overlap", type=float, default=0.2, help="Tile overlap fraction")
    p.add_argument("--no-tile-viz", action="store_true", help="Skip tile-grid visualization")
    p.add_argument("--api-key", default=None,
                   help="Gemini API key (falls back to GEMINI_API_KEY env var)")
    p.add_argument("--show", action="store_true", help="Show figures interactively")
    p.add_argument("--skip-generation", action="store_true",
                   help="Reuse an existing image at --image-out instead of calling Gemini")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_pipeline(
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
        show=args.show,
        skip_generation=args.skip_generation,
    )


if __name__ == "__main__":
    main()
