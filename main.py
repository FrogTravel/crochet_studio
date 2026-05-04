"""Command-line entry point for Crochet Studio.

Subcommands map directly to pipeline stages so the same module functions
that power the Streamlit app and the notebook can be driven from a
shell or a Makefile.

Example:
    $ python main.py generate --output-dir data/synthetic
    $ python main.py train --data data/synthetic/data.yaml
    $ python main.py evaluate --weights runs/obb/.../best.pt \
                              --data data/synthetic/data.yaml
    $ python main.py infer --image data/raw/easy/0.png \
                           --weights runs/obb/.../best.pt
    $ python main.py tile --image-dir data/raw/big \
                          --weights runs/obb/.../best.pt \
                          --out-root data/big_tiles
    $ python main.py pipeline --prompt "doily chart" \
                              --image-out out/free_output.png \
                              --figure-out out/free_output_predictions.png \
                              --weights runs/obb/.../best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import (
    DEFAULT_CONF, DEFAULT_IOU, DEFAULT_OVERLAP, DEFAULT_TARGET_STITCH_PX,
    DEFAULT_TILE_SIZE, DEFAULT_WEIGHTS, ID_TO_NAME,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with one subparser per stage."""
    parser = argparse.ArgumentParser(prog="crochet",
                                     description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate", help="Generate the synthetic dataset")
    p_gen.add_argument("--output-dir", default="data/synthetic")
    p_gen.add_argument("--n-train", type=int, default=300)
    p_gen.add_argument("--n-val",   type=int, default=60)
    p_gen.add_argument("--img-size", type=int, default=640)
    p_gen.add_argument("--real-dir", default=None,
                       help="Optional dir with real images/+labels/ to mix in.")
    p_gen.set_defaults(func=cmd_generate)

    p_tr = sub.add_parser("train", help="Train YOLOv8n-OBB")
    p_tr.add_argument("--data", required=True, help="Path to data.yaml")
    p_tr.add_argument("--base-weights", default="yolov8n-obb.pt")
    p_tr.add_argument("--epochs", type=int, default=10)
    p_tr.add_argument("--batch", type=int, default=4)
    p_tr.add_argument("--imgsz", type=int, default=640)
    p_tr.add_argument("--remote", action="store_true",
                      help="Use the GPU recipe (Colab/CUDA).")
    p_tr.set_defaults(func=cmd_train)

    p_ev = sub.add_parser("evaluate", help="Run validation evaluation")
    p_ev.add_argument("--weights", required=True)
    p_ev.add_argument("--data",    required=True, help="Path to data.yaml")
    p_ev.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p_ev.add_argument("--iou",  type=float, default=DEFAULT_IOU)
    p_ev.add_argument("--plot", action="store_true",
                      help="Render confusion-matrix and bar-chart figures.")
    p_ev.set_defaults(func=cmd_evaluate)

    p_in = sub.add_parser("infer", help="Run adaptive tiled inference")
    p_in.add_argument("--image", required=True)
    p_in.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p_in.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p_in.add_argument("--iou",  type=float, default=DEFAULT_IOU)
    p_in.add_argument("--target-stitch-px", type=int, default=DEFAULT_TARGET_STITCH_PX)
    p_in.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    p_in.add_argument("--overlap",  type=float, default=DEFAULT_OVERLAP)
    p_in.add_argument("--figure", default=None,
                      help="Optional path to save the two-panel figure.")
    p_in.add_argument("--json",   default=None,
                      help="Optional path to save detections as JSON.")
    p_in.set_defaults(func=cmd_infer)

    p_tl = sub.add_parser("tile", help="Tile photos for Label Studio")
    p_tl.add_argument("--image-dir", required=True)
    p_tl.add_argument("--weights", required=True)
    p_tl.add_argument("--out-root", default="data/big_tiles")
    p_tl.add_argument("--url-prefix",
                      default="/data/local-files/?d=big_tiles/images/")
    p_tl.set_defaults(func=cmd_tile)

    p_pl = sub.add_parser("pipeline",
                          help="Generate with Gemini, classify, render")
    p_pl.add_argument("--prompt", required=True)
    p_pl.add_argument("--image-out",  required=True)
    p_pl.add_argument("--figure-out", required=True)
    p_pl.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p_pl.add_argument("--skip-generation", action="store_true",
                      help="Reuse the file at --image-out instead of calling Gemini.")
    p_pl.set_defaults(func=cmd_pipeline)

    return parser


def cmd_generate(args: argparse.Namespace) -> None:
    """Subcommand: generate the synthetic dataset."""
    from src.data_generation import generate_dataset, mix_real_data

    yaml_path = generate_dataset(
        n_train=args.n_train,
        n_val=args.n_val,
        output_dir=args.output_dir,
        img_size=args.img_size,
    )
    if args.real_dir:
        n_tr, n_va = mix_real_data(args.real_dir, args.output_dir)
        print(f"Mixed in {n_tr} real train + {n_va} real val images.")
    print(f"Done. data.yaml -> {yaml_path}")


def cmd_train(args: argparse.Namespace) -> None:
    """Subcommand: train YOLOv8n-OBB."""
    from src.training import train_local, train_remote

    fn = train_remote if args.remote else train_local
    best = fn(
        data_yaml=args.data,
        base_weights=args.base_weights,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
    )
    print(f"Training complete. Best weights -> {best}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Subcommand: run validation evaluation."""
    from src.training import evaluate, plot_evaluation

    result = evaluate(args.weights, args.data,
                      conf=args.conf, iou_threshold=args.iou)
    print(f"{'Class':<16}{'P':>10}{'R':>10}{'F1':>10}")
    for name, p, r, f in zip(result.class_names, result.precision,
                             result.recall, result.f1):
        print(f"{name:<16}{p:>10.3f}{r:>10.3f}{f:>10.3f}")
    print(f"{'macro':<16}"
          f"{result.precision.mean():>10.3f}"
          f"{result.recall.mean():>10.3f}"
          f"{result.f1.mean():>10.3f}")
    if args.plot:
        plot_evaluation(result)


def cmd_infer(args: argparse.Namespace) -> None:
    """Subcommand: run adaptive tiled inference on a single image."""
    from src.pipeline import classify_stitches

    payload = classify_stitches(
        args.image,
        weights_path=args.weights,
        conf=args.conf,
        iou_threshold=args.iou,
        target_stitch_px=args.target_stitch_px,
        tile_size=args.tile_size,
        overlap=args.overlap,
        output_figure=args.figure,
    )
    print(f"Detected {payload['n_detections']} stitches "
          f"({payload['class_counts']})")
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"JSON -> {args.json}")
    if args.figure:
        print(f"Figure -> {args.figure}")


def cmd_tile(args: argparse.Namespace) -> None:
    """Subcommand: tile every image in a folder for Label Studio."""
    from src.inference import load_model
    from src.label_studio import build_tasks_json, tile_and_save

    model = load_model(args.weights)
    image_dir = Path(args.image_dir)
    out_root = Path(args.out_root)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = sorted([p for p in image_dir.iterdir()
                     if p.suffix.lower() in exts])
    print(f"Found {len(images)} images in {image_dir}")
    grand_t = grand_d = 0
    for img_path in images:
        n_t, n_d = tile_and_save(img_path, model, out_root)
        print(f"  {img_path.name}: {n_t} tiles, {n_d} detections")
        grand_t += n_t
        grand_d += n_d

    class_names = [ID_TO_NAME[i] for i in range(len(ID_TO_NAME))]
    tasks_path = build_tasks_json(out_root, class_names, args.url_prefix)
    print(f"Total: {grand_t} tiles, {grand_d} detections")
    print(f"tasks.json -> {tasks_path}")


def cmd_pipeline(args: argparse.Namespace) -> None:
    """Subcommand: generate with Gemini, classify, render."""
    from src.pipeline import run_pipeline

    payload = run_pipeline(
        prompt=args.prompt,
        image_out=args.image_out,
        figure_out=args.figure_out,
        weights_path=args.weights,
        skip_generation=args.skip_generation,
    )
    print(f"Detected {payload['n_detections']} stitches "
          f"({payload['class_counts']})")
    print(f"Image  -> {args.image_out}")
    print(f"Figure -> {args.figure_out}")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected subcommand."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
