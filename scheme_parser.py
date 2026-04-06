"""
scheme_parser.py

Parses a granny square crochet scheme image and returns cropped symbol images.
Only the top triangle of the scheme is processed (it contains all unique symbols
due to the 4-fold symmetry of granny squares).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import cv2 as cv
import numpy as np


@dataclass
class Symbol:
    """A single detected symbol crop with its position in the original image."""
    image: np.ndarray          # grayscale crop, already padded
    bbox: tuple[int, int, int, int]  # (x, y, w, h) in original image coords
    area: float
    circularity: float
    aspect_ratio: float


@dataclass
class ParserConfig:
    """Tunable parameters for the scheme parser."""
    morph_kernel_size: int = 3      # morphological closing kernel (bridges gaps in circles)
    padding: int = 4                # px padding around each crop
    min_area: float = 100           # contour area lower bound (filters noise)
    max_area: float = 8000          # contour area upper bound (filters image border)
    min_circularity: float = 0.55   # roundness threshold (circles ≈ 1.0)
    min_aspect_ratio: float = 0.5   # width/height lower bound
    max_aspect_ratio: float = 2.0   # width/height upper bound


class SchemeParser:
    """
    Parses a granny square crochet scheme image and returns detected symbol crops.

    Usage:
        parser = SchemeParser()
        symbols = parser.parse("data/raw/easy/2.png")
        for sym in symbols:
            cv.imshow("symbol", sym.image)
            cv.waitKey(0)
    """

    def __init__(self, config: ParserConfig | None = None):
        self.config = config or ParserConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, image_path: str | Path) -> list[Symbol]:
        """
        Load a scheme image and return a list of detected Symbol objects.

        Args:
            image_path: path to the input scheme image (.png / .jpg)

        Returns:
            List of Symbol objects, one per detected symbol in the top triangle.
        """
        img = self._load(image_path)
        gray = self._to_grayscale(img)
        masked = self._apply_triangle_mask(gray)
        binary = self._threshold(masked)
        closed = self._close_gaps(binary)
        contours = self._find_contours(closed)
        symbols = self._extract_symbols(gray, contours)
        return symbols

    # ------------------------------------------------------------------
    # Pipeline steps (private)
    # ------------------------------------------------------------------

    def _load(self, image_path: str | Path) -> np.ndarray:
        img = cv.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return img

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    def _apply_triangle_mask(self, gray: np.ndarray) -> np.ndarray:
        """
        Mask out everything except the top triangle of the image.
        Granny squares have 4-fold symmetry, so the top triangle holds
        all unique symbols — no need to process the full image.
        """
        h, w = gray.shape
        mask = np.zeros_like(gray, dtype=np.uint8)
        triangle = np.array([[0, 0], [w, 0], [w // 2, h // 2]], dtype=np.int32)
        cv.fillPoly(mask, [triangle], 255)
        return cv.bitwise_and(gray, gray, mask=mask)

    def _threshold(self, masked: np.ndarray) -> np.ndarray:
        """
        Binarise with Otsu then invert so symbols are white on black.
        Inversion is required for correct morphological closing: we want
        to close gaps in white symbol outlines, not in the background.
        """
        _, thresh = cv.threshold(masked, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        return cv.bitwise_not(thresh)

    def _close_gaps(self, binary: np.ndarray) -> np.ndarray:
        """
        Morphological closing to bridge small gaps in drawn circle outlines.
        Crochet schemes are often scanned or printed with minor discontinuities.
        """
        kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE,
            (self.config.morph_kernel_size, self.config.morph_kernel_size)
        )
        return cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)

    def _find_contours(self, binary: np.ndarray) -> list:
        contours, _ = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        return contours

    def _extract_symbols(
        self, gray: np.ndarray, contours: list
    ) -> list[Symbol]:
        """
        Filter contours by shape and crop symbol patches from the grayscale image.
        """
        cfg = self.config
        symbols = []

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

            crop = gray[y1:y2, x1:x2]
            symbols.append(Symbol(
                image=crop,
                bbox=(x, y, w, h),
                area=area,
                circularity=circularity,
                aspect_ratio=ar,
            ))

        return symbols

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _shape_features(contour) -> tuple[float, float, float]:
        """Returns (aspect_ratio, circularity, area) for a contour."""
        x, y, w, h = cv.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        circularity = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0
        return aspect_ratio, circularity, area

    def save_crops(
        self,
        symbols: list[Symbol],
        output_dir: str | Path,
        prefix: str = "symbol",
    ) -> None:
        """Save all symbol crops to a directory as numbered PNG files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, sym in enumerate(symbols):
            cv.imwrite(str(output_dir / f"{prefix}_{i:04d}.png"), sym.image)
        print(f"Saved {len(symbols)} crops to {output_dir}/")
