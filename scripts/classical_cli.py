"""CLI for the classical CV pipeline (granny-square scheme parser + HOG/SVM).

Commands
--------
parse     Parse a scheme image → save symbol crops to a directory
label     Interactively label saved crops with keyboard shortcuts
train     Train the SVM classifier on a labelled dataset
predict   Parse a scheme and classify each detected symbol
evaluate  Evaluate a saved model on the labelled dataset

Examples
--------
    python scripts/classical_cli.py parse data/raw/easy/2.png \\
        --out data/segments/unlabeled
    python scripts/classical_cli.py label \\
        --crops data/segments/unlabeled --out data/segments/labeled
    python scripts/classical_cli.py train \\
        --data data/segments/labeled --model model.joblib
    python scripts/classical_cli.py predict data/raw/easy/3.png \\
        --model model.joblib
    python scripts/classical_cli.py evaluate \\
        --data data/segments/labeled --model model.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2 as cv  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from crochet.classical import (  # noqa: E402
    Labeler,
    ParserConfig,
    SchemeParser,
    SymbolClassifier,
    dataset_summary,
    load_labeled_dataset,
    split_dataset,
)


# ── Sub-commands ───────────────────────────────────────────────────────────
def cmd_parse(args: argparse.Namespace) -> None:
    parser = SchemeParser(ParserConfig(
        morph_kernel_size=args.kernel,
        padding=args.padding,
        min_area=args.min_area,
        max_area=args.max_area,
    ))
    symbols = parser.parse(args.image)
    print(f"Detected {len(symbols)} symbols in {args.image}")
    parser.save_crops(symbols, args.out)
    if args.visualize:
        _visualize_crops([s.image for s in symbols], title="Detected symbols")


def cmd_label(args: argparse.Namespace) -> None:
    Labeler(
        crops_dir=args.crops,
        output_dir=args.out,
        raw_dir=args.raw,
    ).run()


def cmd_train(args: argparse.Namespace) -> None:
    images, labels = load_labeled_dataset(args.data)
    dataset_summary(labels)

    if len(set(labels)) < 2:
        print("Error: need at least 2 classes to train. Label more images first.")
        sys.exit(1)

    clf = SymbolClassifier()
    X_train, X_test, y_train, y_test = split_dataset(images, labels)

    clf.train(X_train, y_train)
    clf.cross_validate(images, labels)
    clf.evaluate(X_test, y_test, plot=True)

    if args.show_errors:
        clf.show_misclassified(X_test, y_test)

    clf.save(args.model)


def cmd_predict(args: argparse.Namespace) -> None:
    parser = SchemeParser()
    symbols = parser.parse(args.image)
    print(f"Detected {len(symbols)} symbols")

    clf = SymbolClassifier.load(args.model)

    results: list[tuple] = []
    for sym in symbols:
        label = clf.predict(sym.image)
        proba = clf.predict_proba(sym.image)
        confidence = proba[label]
        results.append((sym, label, confidence))
        x, y, _, _ = sym.bbox
        print(f"  ({x:4d},{y:4d})  {label:<15}  conf={confidence:.2f}")

    if args.visualize:
        img = cv.imread(str(args.image))
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        for sym, label, conf in results:
            x, y, w, h = sym.bbox
            cv.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv.putText(img_rgb, f"{label} {conf:.2f}", (x, y - 6),
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
        plt.figure(figsize=(10, 10))
        plt.imshow(img_rgb)
        plt.title(f"{len(symbols)} symbols detected")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


def cmd_evaluate(args: argparse.Namespace) -> None:
    images, labels = load_labeled_dataset(args.data)
    dataset_summary(labels)

    clf = SymbolClassifier.load(args.model)
    clf.evaluate(images, labels, plot=True)
    clf.show_misclassified(images, labels)


# ── Helpers ────────────────────────────────────────────────────────────────
def _visualize_crops(images: list, title: str = "Crops", cols: int = 8) -> None:
    rows = max(1, (len(images) + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, img in zip(axes, images):
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# ── Argument parser ────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="classical_cli.py",
        description="Crochet scheme symbol recognition pipeline (classical CV).",
    )
    sub = root.add_subparsers(dest="command", required=True)

    p = sub.add_parser("parse", help="Extract symbol crops from a scheme image")
    p.add_argument("image", help="Path to scheme image")
    p.add_argument("--out", default="data/segments/unlabeled")
    p.add_argument("--kernel", type=int, default=3)
    p.add_argument("--padding", type=int, default=4)
    p.add_argument("--min-area", type=float, default=100, dest="min_area")
    p.add_argument("--max-area", type=float, default=8000, dest="max_area")
    p.add_argument("--visualize", action="store_true")

    p = sub.add_parser("label", help="Interactively label symbol crops")
    p.add_argument("--crops", default="data/segments/unlabeled")
    p.add_argument("--out", default="data/segments/labeled")
    p.add_argument("--raw", default="data/raw/easy",
                   help="Raw images folder for context thumbnails (optional)")

    p = sub.add_parser("train", help="Train the SVM classifier")
    p.add_argument("--data", default="data/segments/labeled")
    p.add_argument("--model", default="model.joblib")
    p.add_argument("--show-errors", action="store_true", dest="show_errors")

    p = sub.add_parser("predict", help="Parse + classify symbols in a scheme image")
    p.add_argument("image", help="Path to scheme image")
    p.add_argument("--model", default="model.joblib")
    p.add_argument("--visualize", action="store_true")

    p = sub.add_parser("evaluate", help="Evaluate saved model on labelled dataset")
    p.add_argument("--data", default="data/segments/labeled")
    p.add_argument("--model", default="model.joblib")

    return root


def main() -> None:
    args = build_parser().parse_args()
    commands = {
        "parse":    cmd_parse,
        "label":    cmd_label,
        "train":    cmd_train,
        "predict":  cmd_predict,
        "evaluate": cmd_evaluate,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
