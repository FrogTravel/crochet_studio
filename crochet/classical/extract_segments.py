"""Batch-extract symbol crops from every scheme image in a folder.

Mirrors the notebook preprocessing pipeline:
grayscale → triangle mask → Otsu threshold → invert → morphological
closing → find contours → area filter → remove containing contours →
save crops. No shape filter is applied, so every symbol type ends up in
the output folder ready for labelling.
"""

from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np


def preprocess(img_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one scheme image and return ``(gray, closed_binary)``."""
    img = cv.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Top-triangle mask — granny squares have 4-fold symmetry.
    mask = np.zeros_like(gray)
    triangle = np.array([[0, 0], [w, 0], [w // 2, h // 2]], dtype=np.int32)
    cv.fillPoly(mask, [triangle], 255)
    masked = cv.bitwise_and(gray, gray, mask=mask)

    # Otsu threshold + invert so symbols are white on black.
    _, thresh = cv.threshold(masked, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    thresh_inv = cv.bitwise_not(thresh)

    # Morphological closing to bridge small gaps in circle outlines.
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    closed = cv.morphologyEx(thresh_inv, cv.MORPH_CLOSE, kernel)
    return gray, closed


def find_clean_contours(
    closed: np.ndarray,
    min_area: float = 30,
    max_area: float = 50_000,
) -> list:
    """Find contours, apply area filter, then remove wrapping contours."""
    contours, _ = cv.findContours(closed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if min_area < cv.contourArea(c) < max_area]
    return _remove_contained(contours)


def _remove_contained(contours: list) -> list:
    """Drop contours that fully contain a smaller contour from the same image."""
    boxes = [cv.boundingRect(c) for c in contours]
    to_remove: set[int] = set()
    for i, (xi, yi, wi, hi) in enumerate(boxes):
        for j, (xj, yj, wj, hj) in enumerate(boxes):
            if i == j:
                continue
            if xi <= xj and yi <= yj and xi + wi >= xj + wj and yi + hi >= yj + hj:
                to_remove.add(i)
                break
    return [c for i, c in enumerate(contours) if i not in to_remove]


def save_crops(
    gray: np.ndarray,
    contours: list,
    out_dir: Path,
    prefix: str,
    padding: int = 4,
) -> int:
    """Crop each contour and save it as a PNG; return the number saved."""
    h_img, w_img = gray.shape
    saved = 0
    for i, c in enumerate(contours):
        x, y, w, h = cv.boundingRect(c)
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, w_img)
        y2 = min(y + h + padding, h_img)
        crop = gray[y1:y2, x1:x2]
        cv.imwrite(str(out_dir / f"{prefix}_{i:04d}.png"), crop)
        saved += 1
    return saved


def run(
    src_dir: Path,
    out_dir: Path,
    padding: int = 4,
    min_area: float = 30,
    max_area: float = 50_000,
) -> int:
    """Extract segments from every image in ``src_dir``; return total saved."""
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(src_dir.glob("*.png")) + sorted(src_dir.glob("*.jpg"))
    if not image_files:
        print(f"No images found in {src_dir}")
        return 0

    total = 0
    for img_path in image_files:
        try:
            gray, closed = preprocess(img_path)
            contours = find_clean_contours(closed, min_area, max_area)
            n = save_crops(gray, contours, out_dir, img_path.stem, padding)
            print(f"  {img_path.name}: {n} segments")
            total += n
        except Exception as exc:  # pragma: no cover — file I/O
            print(f"  {img_path.name}: ERROR — {exc}")

    print(f"\nDone. {total} segments saved to {out_dir}/")
    return total
