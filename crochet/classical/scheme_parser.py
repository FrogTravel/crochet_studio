"""Granny-square crochet scheme parser.

Returns cropped symbol images from a scheme. Only the top triangle is
processed because granny squares have 4-fold symmetry — all unique
symbols are already present there.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np


@dataclass
class Symbol:
    """A single detected symbol crop with its position in the original image."""

    image: np.ndarray                   # grayscale crop, already padded
    bbox: tuple[int, int, int, int]     # (x, y, w, h) in original image coords
    area: float
    circularity: float
    aspect_ratio: float


@dataclass
class ParserConfig:
    """Tunable parameters for :class:`SchemeParser`."""

    morph_kernel_size: int = 3      # morphological closing kernel size
    padding: int = 4                # px of padding around each crop
    min_area: float = 100           # contour area lower bound (filters noise)
    max_area: float = 8000          # contour area upper bound (filters image border)
    min_circularity: float = 0.55   # roundness threshold (circles ≈ 1.0)
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0


class SchemeParser:
    """Parse a granny-square crochet scheme into symbol crops.

    Usage::

        parser = SchemeParser()
        symbols = parser.parse("data/raw/easy/2.png")
    """

    def __init__(self, config: ParserConfig | None = None) -> None:
        self.config = config or ParserConfig()

    # ── Public API ────────────────────────────────────────────────────────
    def parse(self, image_path: str | Path) -> list[Symbol]:
        """Load a scheme image and return detected :class:`Symbol` objects."""
        img = self._load(image_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        masked = self._apply_triangle_mask(gray)
        binary = self._threshold(masked)
        closed = self._close_gaps(binary)
        contours, _ = cv.findContours(closed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        return self._extract_symbols(gray, list(contours))

    def save_crops(
        self,
        symbols: list[Symbol],
        output_dir: str | Path,
        prefix: str = "symbol",
    ) -> None:
        """Save symbol crops to ``output_dir`` as numbered PNGs."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, sym in enumerate(symbols):
            cv.imwrite(str(output_dir / f"{prefix}_{i:04d}.png"), sym.image)
        print(f"Saved {len(symbols)} crops to {output_dir}/")

    # ── Pipeline steps ────────────────────────────────────────────────────
    @staticmethod
    def _load(image_path: str | Path) -> np.ndarray:
        img = cv.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return img

    @staticmethod
    def _apply_triangle_mask(gray: np.ndarray) -> np.ndarray:
        """Mask out everything except the top triangle of the image."""
        h, w = gray.shape
        mask = np.zeros_like(gray, dtype=np.uint8)
        triangle = np.array([[0, 0], [w, 0], [w // 2, h // 2]], dtype=np.int32)
        cv.fillPoly(mask, [triangle], 255)
        return cv.bitwise_and(gray, gray, mask=mask)

    @staticmethod
    def _threshold(masked: np.ndarray) -> np.ndarray:
        """Otsu threshold then invert so symbols are white on black."""
        _, thresh = cv.threshold(masked, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        return cv.bitwise_not(thresh)

    def _close_gaps(self, binary: np.ndarray) -> np.ndarray:
        """Morphological closing to bridge small gaps in drawn outlines."""
        kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE,
            (self.config.morph_kernel_size, self.config.morph_kernel_size),
        )
        return cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)

    def _extract_symbols(self, gray: np.ndarray, contours: list) -> list[Symbol]:
        cfg = self.config
        symbols: list[Symbol] = []

        for c in contours:
            ar, circularity, area = self._shape_features(c)

            if not (cfg.min_aspect_ratio < ar < cfg.max_aspect_ratio):
                continue
            if circularity < cfg.min_circularity:
                continue
            if not (cfg.min_area < area < cfg.max_area):
                continue

            x, y, w, h = cv.boundingRect(c)
            x1 = max(x - cfg.padding, 0)
            y1 = max(y - cfg.padding, 0)
            x2 = min(x + w + cfg.padding, gray.shape[1])
            y2 = min(y + h + cfg.padding, gray.shape[0])

            symbols.append(Symbol(
                image=gray[y1:y2, x1:x2],
                bbox=(x, y, w, h),
                area=area,
                circularity=circularity,
                aspect_ratio=ar,
            ))
        return symbols

    # ── Helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _shape_features(contour) -> tuple[float, float, float]:
        """Return ``(aspect_ratio, circularity, area)`` for one contour."""
        _, _, w, h = cv.boundingRect(contour)
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        aspect_ratio = (w / h) if h > 0 else 0.0
        circularity = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0
        return aspect_ratio, circularity, area
