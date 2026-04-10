"""
dataset.py

Utilities for loading and splitting labeled symbol datasets.
Expects an ImageFolder-style directory layout:

    data/segments/labeled/
        treble/
            symbol_0001.png
            ...
        slip_stitch/
            symbol_0002.png
            ...
        chain/
            ...
"""

from __future__ import annotations
from pathlib import Path

import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split


def load_labeled_dataset(
    data_dir: str | Path,
    img_size: tuple[int, int] = (64, 64),
    extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> tuple[list[np.ndarray], list[str]]:
    """
    Load all labeled images from an ImageFolder-style directory.

    Args:
        data_dir:   root directory containing one sub-folder per class
        img_size:   (width, height) to resize images to; None to keep original
        extensions: file extensions to include

    Returns:
        images: list of grayscale uint8 numpy arrays
        labels: list of class name strings (matches images by index)
    """
    data_dir = Path(data_dir)
    images, labels = [], []
    skipped = []

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        files = [f for f in class_dir.iterdir() if f.suffix.lower() in extensions]
        if not files:
            skipped.append(class_dir.name)
            continue
        for fpath in sorted(files):
            img = cv.imread(str(fpath), cv.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  Warning: could not read {fpath}, skipping.")
                continue
            if img_size is not None:
                img = cv.resize(img, img_size)
            images.append(img)
            labels.append(class_dir.name)

    if skipped:
        print(f"Skipped empty classes: {skipped}")

    return images, labels


def dataset_summary(labels: list[str]) -> None:
    """Print per-class sample counts."""
    from collections import Counter
    counts = Counter(labels)
    total = sum(counts.values())
    print(f"{'Class':<20} {'Count':>6}  {'Share':>7}")
    print("-" * 37)
    for cls, n in sorted(counts.items()):
        print(f"{cls:<20} {n:>6}  {n/total:>6.1%}")
    print(f"{'TOTAL':<20} {total:>6}")


def split_dataset(
    images: list[np.ndarray],
    labels: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list, list, list, list]:
    """
    Stratified train/test split.

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(
        images, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
