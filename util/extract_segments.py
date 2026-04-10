"""
extract_segments.py

Batch-extract symbol crops from every scheme image in a folder and save
them to an output directory for labeling.

The pipeline mirrors preprocess.ipynb exactly:
  grayscale → triangle mask → Otsu threshold → invert →
  morphological closing → find contours → area filter →
  remove_contained → save crops

No shape filter is applied so ALL symbol types (treble, chain, slip stitch,
double crochet, etc.) end up in the output folder ready for labeling.

Usage:
    python extract_segments.py                          # uses defaults
    python extract_segments.py --src data/raw/easy --out data/segments/unlabeled
"""

import argparse
import cv2 as cv
import numpy as np
from pathlib import Path


# ── Pipeline helpers (same logic as preprocess.ipynb) ────────────────────────

def preprocess(img_path: Path):
    """
    Load one scheme image and return (gray, closed_binary).
    closed_binary has white symbols on black background, ready for findContours.
    """
    img = cv.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Top-triangle mask — granny squares have 4-fold symmetry
    mask = np.zeros_like(gray)
    triangle = np.array([[0, 0], [w, 0], [w // 2, h // 2]], dtype=np.int32)
    cv.fillPoly(mask, [triangle], 255)
    masked = cv.bitwise_and(gray, gray, mask=mask)

    # Otsu threshold then invert so symbols are white on black
    _, thresh = cv.threshold(masked, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    thresh_inv = cv.bitwise_not(thresh)

    # Morphological closing to bridge small gaps in circle outlines
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    closed = cv.morphologyEx(thresh_inv, cv.MORPH_CLOSE, kernel)

    return gray, closed


def find_clean_contours(closed, min_area=30, max_area=50000):
    """
    Find contours, apply area filter, then remove wrapper contours that
    fully contain smaller ones (prevents counting a group AND its members).
    """
    contours, _ = cv.findContours(closed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Area filter
    contours = [c for c in contours if min_area < cv.contourArea(c) < max_area]

    # Remove wrappers
    contours = _remove_contained(contours)

    return contours


def _remove_contained(contours):
    boxes = [cv.boundingRect(c) for c in contours]
    to_remove = set()
    for i, (xi, yi, wi, hi) in enumerate(boxes):
        for j, (xj, yj, wj, hj) in enumerate(boxes):
            if i == j:
                continue
            if xi <= xj and yi <= yj and xi + wi >= xj + wj and yi + hi >= yj + hj:
                to_remove.add(i)
                break
    return [c for i, c in enumerate(contours) if i not in to_remove]


def save_crops(gray, contours, out_dir: Path, prefix: str, padding: int = 4):
    """
    Crop each contour from the grayscale image and save as a PNG.
    Files are named  <prefix>_<index:04d>.png
    Returns the number of crops saved.
    """
    h_img, w_img = gray.shape
    saved = 0
    for i, c in enumerate(contours):
        x, y, w, h = cv.boundingRect(c)
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, w_img)
        y2 = min(y + h + padding, h_img)
        crop = gray[y1:y2, x1:x2]
        fname = out_dir / f"{prefix}_{i:04d}.png"
        cv.imwrite(str(fname), crop)
        saved += 1
    return saved


# ── Main ─────────────────────────────────────────────────────────────────────

def run(src_dir: Path, out_dir: Path, padding: int = 4,
        min_area: int = 30, max_area: int = 50000):

    out_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(src_dir.glob("*.png")) + sorted(src_dir.glob("*.jpg"))
    if not image_files:
        print(f"No images found in {src_dir}")
        return

    total = 0
    for img_path in image_files:
        try:
            gray, closed = preprocess(img_path)
            contours = find_clean_contours(closed, min_area, max_area)
            prefix = img_path.stem
            n = save_crops(gray, contours, out_dir, prefix, padding)
            print(f"  {img_path.name}: {n} segments")
            total += n
        except Exception as e:
            print(f"  {img_path.name}: ERROR — {e}")

    print(f"\nDone. {total} segments saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-extract crochet symbol crops")
    parser.add_argument("--src", default="data/raw/easy",
                        help="Folder with raw scheme images (default: data/raw/easy)")
    parser.add_argument("--out", default="data/segments/unlabeled",
                        help="Output folder for crops (default: data/segments/unlabeled)")
    parser.add_argument("--padding", type=int, default=4,
                        help="Padding in px around each crop (default: 4)")
    parser.add_argument("--min-area", type=int, default=30, dest="min_area")
    parser.add_argument("--max-area", type=int, default=50000, dest="max_area")
    args = parser.parse_args()

    run(Path(args.src), Path(args.out), args.padding, args.min_area, args.max_area)
