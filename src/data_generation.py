"""Step 1 — synthetic crochet-chart data generation.

Produces a YOLO-OBB-ready dataset of synthetic chart images and labels.
The pipeline is:

1. *Drawer* — emit a transparent BGRA glyph for a given stitch class.
2. *Compositor* — paste a glyph onto a canvas with rotation, recording
   the four rotated corners as a YOLO-OBB label.
3. *Layout generator* — stack glyphs into a realistic chart pattern.
4. *Augmentation* — apply photographic noise (brightness, blur, JPEG).
5. *Writer* — save PNG + ``.txt`` pairs and emit ``data.yaml``.

The reference walkthrough lives in ``notebooks/full_pipeline_YOLO_OBB.ipynb``,
Step 1.
"""

from __future__ import annotations

import math
import random
import shutil
from glob import glob
from pathlib import Path
from typing import Callable, Literal, Sequence

import cv2 as cv
import numpy as np
import yaml

from .config import CLASS_MAP, ID_TO_NAME, NUM_CLASSES, PALETTES


# ── Drawers ────────────────────────────────────────────────────────────────
def draw_chain(size: int = 40, thickness: int = 2,
               color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Draw a chain stitch (open ellipse) as a BGRA image."""
    w, h = size, int(size * 0.55)
    img = np.zeros((h + 4, w + 4, 4), dtype=np.uint8)
    cv.ellipse(img, ((w + 4) // 2, (h + 4) // 2), (w // 2, h // 2),
               0, 0, 360, (*color, 255), thickness)
    return img


def draw_single(size: int = 40, thickness: int = 2,
                color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Draw a single-crochet stitch (``+`` cross) as a BGRA image."""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    m = size // 2
    cv.line(img, (m, 2), (m, size - 3), (*color, 255), thickness)
    cv.line(img, (m - size // 4, m), (m + size // 4, m), (*color, 255), thickness)
    return img


def _draw_tall(size: int, n_slashes: int, thickness: int,
               color: tuple[int, int, int]) -> np.ndarray:
    """Internal: T-stem with N evenly-spaced slashes (double / treble / dtr)."""
    w, h = int(size * 0.35), size
    img = np.zeros((h, w, 4), dtype=np.uint8)
    mx = w // 2
    cv.line(img, (mx, 2), (mx, h - 3), (*color, 255), thickness)
    cv.line(img, (mx - w // 4, 2), (mx + w // 4, 2), (*color, 255), thickness)
    if n_slashes == 1:
        offsets = [0]
    elif n_slashes >= 2:
        offsets = np.linspace(-h // 6, h // 6, n_slashes).astype(int).tolist()
    else:
        offsets = []
    for off in offsets:
        cy = h // 2 + off
        cv.line(img, (mx - w // 3, cy + h // 8),
                (mx + w // 3, cy - h // 8), (*color, 255), thickness)
    return img


def draw_double(size: int = 80, thickness: int = 2,
                color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Draw a double-crochet stitch (T-stem with one diagonal slash)."""
    return _draw_tall(size, 1, thickness, color)


def draw_half_double(size: int = 60, thickness: int = 2,
                     color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Draw a half-double crochet (T-stem with one horizontal bar)."""
    w, h = int(size * 0.4), size
    img = np.zeros((h, w, 4), dtype=np.uint8)
    mx = w // 2
    cv.line(img, (mx, 2), (mx, h - 3), (*color, 255), thickness)
    y_bar = h // 3
    cv.line(img, (mx - w // 3, y_bar), (mx + w // 3, y_bar), (*color, 255), thickness)
    cv.line(img, (mx - w // 4, 2), (mx + w // 4, 2), (*color, 255), thickness)
    return img


def draw_treble(size: int = 100, thickness: int = 2,
                color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Draw a treble crochet (T-stem with two diagonal slashes)."""
    return _draw_tall(size, 2, thickness, color)


def draw_double_treble(size: int = 120, thickness: int = 2,
                       color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Draw a double-treble crochet (T-stem with three slashes)."""
    return _draw_tall(size, 3, thickness, color)


def draw_fan(n_spokes: int = 5, spoke_len: int = 70, thickness: int = 2,
             color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Draw a fan / shell stitch (radiating spokes from a base point)."""
    spread = math.radians(90)
    w, h = int(spoke_len * 1.6), int(spoke_len * 1.2)
    img = np.zeros((h, w, 4), dtype=np.uint8)
    base = (w // 2, h - 4)
    for i in range(n_spokes):
        angle = math.pi / 2 + spread / 2 - (spread * i / max(n_spokes - 1, 1))
        ex = int(base[0] + spoke_len * math.cos(angle))
        ey = int(base[1] - spoke_len * math.sin(angle))
        cv.line(img, base, (ex, ey), (*color, 255), thickness)
    return img


def draw_ensemble_chain(n_chains: int = 5, chain_size: int = 20,
                        thickness: int = 2,
                        color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Draw a vertical bundle of chain ovals."""
    ch_h = int(chain_size * 0.6)
    gap = 2
    total_h = n_chains * (ch_h + gap) + 4
    w = chain_size + 8
    img = np.zeros((total_h, w, 4), dtype=np.uint8)
    cx = w // 2
    for i in range(n_chains):
        cy = 2 + ch_h // 2 + i * (ch_h + gap)
        cv.ellipse(img, (cx, cy), (chain_size // 2, ch_h // 2),
                   0, 0, 360, (*color, 255), thickness)
    return img


def draw_noise(kind: Literal["circle", "number", "arrow"] = "circle",
               size: int = 30,
               thickness: int = 2,
               color: tuple[int, int, int] = (0, 0, 0),
               number: int = 1,
               direction: Literal["left", "right"] = "right") -> np.ndarray:
    """Draw a noise annotation (row counter, arrow, or plain circle)."""
    if kind == "circle":
        img = np.zeros((size, size, 4), dtype=np.uint8)
        cv.circle(img, (size // 2, size // 2), size // 3, (*color, 255), thickness)
        return img

    if kind == "number":
        img = np.zeros((size, size, 4), dtype=np.uint8)
        r = size // 2 - 2
        cv.circle(img, (size // 2, size // 2), r, (*color, 255), 1)
        font = cv.FONT_HERSHEY_SIMPLEX
        sc = size / 60
        (tw, th), _ = cv.getTextSize(str(number), font, sc, 1)
        cv.putText(img, str(number),
                   (size // 2 - tw // 2, size // 2 + th // 2),
                   font, sc, (*color, 255), 1, cv.LINE_AA)
        return img

    # arrow
    w, h = size, max(int(size * 0.4), 8)
    img = np.zeros((h, w, 4), dtype=np.uint8)
    y = h // 2
    if direction == "right":
        cv.line(img, (2, y), (w - 6, y), (*color, 255), thickness)
        cv.line(img, (w - 10, y - 5), (w - 4, y), (*color, 255), thickness)
        cv.line(img, (w - 10, y + 5), (w - 4, y), (*color, 255), thickness)
    else:
        cv.line(img, (6, y), (w - 2, y), (*color, 255), thickness)
        cv.line(img, (10, y - 5), (4, y), (*color, 255), thickness)
        cv.line(img, (10, y + 5), (4, y), (*color, 255), thickness)
    return img


# ── Templates + symbol picker ─────────────────────────────────────────────
def load_templates(template_dir: str | Path) -> dict[str, list[np.ndarray]]:
    """Load PNG templates organised by class subfolder."""
    templates: dict[str, list[np.ndarray]] = {cls: [] for cls in CLASS_MAP}
    template_dir = Path(template_dir)
    if not template_dir.is_dir():
        return templates
    for cls_name in CLASS_MAP:
        cls_dir = template_dir / cls_name
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.glob("*.png")):
            img = cv.imread(str(p), cv.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.ndim == 2:
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                img = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
            templates[cls_name].append(img)
    return templates


# Drawer registry — used by ``get_symbol`` to pick a procedural drawer.
_PROCEDURAL_DRAWERS: dict[str, Sequence[Callable[..., np.ndarray]]] = {
    "chain":         [draw_chain],
    "double":        [draw_double],
    "double treble": [draw_double_treble],
    "single":        [draw_single],
    "half_double":   [draw_half_double],
    "treble":        [draw_treble],
    "fan":           [draw_fan],
    "enseble_chain": [draw_ensemble_chain],
    "noise":         [draw_noise],
}


def get_symbol(cls_name: str, target_h: int,
               templates: dict[str, list[np.ndarray]],
               color: tuple[int, int, int] = (0, 0, 0),
               thickness: int = 2) -> np.ndarray:
    """Return a glyph for ``cls_name`` of approximate height ``target_h``."""
    use_template = random.random() < 0.5 and len(templates.get(cls_name, [])) > 0
    if use_template:
        tmpl = random.choice(templates[cls_name]).copy()
    else:
        drawers = _PROCEDURAL_DRAWERS.get(cls_name, [])
        if not drawers:
            tmpl = np.zeros((20, 20, 4), dtype=np.uint8)
            cv.circle(tmpl, (10, 10), 5, (*color, 255), -1)
        else:
            drawer = random.choice(drawers)
            kwargs: dict = {"color": color, "thickness": thickness}
            if cls_name == "chain":
                kwargs["size"] = random.randint(30, 50)
            elif cls_name == "double":
                kwargs["size"] = random.randint(60, 100)
            elif cls_name == "single":
                kwargs["size"] = random.randint(25, 45)
            elif cls_name == "half_double":
                kwargs["size"] = random.randint(45, 70)
            elif cls_name == "treble":
                kwargs["size"] = random.randint(80, 120)
            elif cls_name == "double treble":
                kwargs["size"] = random.randint(100, 140)
            elif cls_name == "fan":
                kwargs = {"n_spokes": random.randint(3, 7),
                          "spoke_len": random.randint(50, 80),
                          "thickness": thickness, "color": color}
            elif cls_name == "enseble_chain":
                kwargs = {"n_chains": random.randint(3, 8),
                          "chain_size": random.randint(15, 25),
                          "thickness": thickness, "color": color}
            elif cls_name == "noise":
                kind = random.choice(["circle", "number", "arrow"])
                kwargs = {"kind": kind, "color": color, "thickness": thickness,
                          "size": random.randint(18, 40)}
                if kind == "number":
                    kwargs["number"] = random.randint(1, 30)
                elif kind == "arrow":
                    kwargs["direction"] = random.choice(["left", "right"])
            tmpl = drawer(**kwargs)

    if tmpl.shape[0] < 1 or tmpl.shape[1] < 1:
        return np.zeros((target_h, max(target_h // 2, 10), 4), dtype=np.uint8)
    scale = target_h / tmpl.shape[0]
    return cv.resize(tmpl, (max(int(tmpl.shape[1] * scale), 1), target_h),
                     interpolation=cv.INTER_AREA)


# ── Compositing ───────────────────────────────────────────────────────────
def paste_symbol(canvas: np.ndarray, symbol: np.ndarray,
                 cx: int, cy: int,
                 angle_deg: float = 0.0) -> np.ndarray | None:
    """Rotate and alpha-blend a glyph onto a BGR canvas.

    Returns the four rotated corner points in canvas pixels, or ``None``
    if the glyph would lie entirely outside the canvas.
    """
    h, w = symbol.shape[:2]
    M = cv.getRotationMatrix2D((w / 2, h / 2), -angle_deg, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    rotated = cv.warpAffine(symbol, M, (new_w, new_h),
                            flags=cv.INTER_LINEAR, borderValue=(0, 0, 0, 0))

    x1 = int(cx - new_w / 2)
    y1 = int(cy - new_h / 2)
    ch, cw = canvas.shape[:2]
    sx, sy = max(0, -x1), max(0, -y1)
    ex, ey = min(new_w, cw - x1), min(new_h, ch - y1)
    if sx >= ex or sy >= ey:
        return None

    roi = canvas[y1 + sy:y1 + ey, x1 + sx:x1 + ex]
    patch = rotated[sy:ey, sx:ex]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    for c in range(3):
        roi[:, :, c] = (alpha[:, :, 0] * patch[:, :, c]
                        + (1 - alpha[:, :, 0]) * roi[:, :, c]).astype(np.uint8)

    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    pts = np.hstack([corners, np.ones((4, 1), dtype=np.float32)])
    transformed = (M @ pts.T).T
    transformed[:, 0] += x1
    transformed[:, 1] += y1
    return transformed


def obb_to_yolo(corners: np.ndarray, img_w: int, img_h: int) -> list[float]:
    """Convert four pixel corners into a normalised YOLO-OBB label row."""
    out: list[float] = []
    for px, py in corners:
        out.extend([float(np.clip(px / img_w, 0, 1)),
                    float(np.clip(py / img_h, 0, 1))])
    return out


def _add_grid_lines(canvas: np.ndarray, spacing: int = 30,
                    color: tuple[int, int, int] = (200, 200, 200),
                    thickness: int = 1) -> None:
    """Internal: overlay faint grid lines on a canvas."""
    h, w = canvas.shape[:2]
    for x in range(0, w, spacing):
        cv.line(canvas, (x, 0), (x, h), color, thickness)
    for y in range(0, h, spacing):
        cv.line(canvas, (0, y), (w, y), color, thickness)


def _add_aging_texture(canvas: np.ndarray, intensity: float = 0.03) -> np.ndarray:
    """Internal: add subtle Gaussian noise to a canvas."""
    noise = np.random.normal(0, intensity * 255, canvas.shape).astype(np.float32)
    return np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _add_edge_noise(canvas: np.ndarray, labels: list, img_size: int,
                    fg: tuple[int, int, int], margin: int) -> None:
    """Internal: sprinkle row numbers / arrows / circles along the edges."""
    for i in range(random.randint(0, 5)):
        ny = random.randint(margin, img_size - margin)
        side = random.choice(["left", "right"])
        nx = (random.randint(2, max(margin - 5, 3)) if side == "left"
              else random.randint(img_size - margin + 5, img_size - 5))
        kind = random.choice(["number", "arrow", "circle"])
        if kind == "number":
            sym = draw_noise("number", number=i + 1,
                             size=random.randint(18, 28), color=fg)
        elif kind == "arrow":
            sym = draw_noise("arrow", size=random.randint(25, 40), color=fg,
                             direction="right" if side == "left" else "left")
        else:
            sym = draw_noise("circle", size=random.randint(12, 22), color=fg)
        c = paste_symbol(canvas, sym, nx, ny, random.gauss(0, 3))
        if c is not None:
            labels.append((CLASS_MAP["noise"], *obb_to_yolo(c, img_size, img_size)))


# ── Layout generators ─────────────────────────────────────────────────────
LabelTuple = tuple[int, float, float, float, float, float, float, float, float]


def generate_dense_rows(img_size: int = 640
                        ) -> tuple[np.ndarray, list[LabelTuple]]:
    """Produce a dense-rows chart, the most common real-world layout."""
    palette = random.choice(list(PALETTES.values()))
    bg = np.full((img_size, img_size, 3), palette["bg"], dtype=np.uint8)
    fg = palette["fg"]
    if random.random() < 0.6:
        _add_grid_lines(bg, spacing=random.randint(20, 40),
                        color=tuple(int(c * 0.9) for c in palette["bg"]))
    templates = _CACHED_TEMPLATES.get_or_load()
    labels: list[LabelTuple] = []
    stitch_h = random.randint(25, 50)
    row_gap = stitch_h + random.randint(5, 15)
    margin = random.randint(20, 50)
    row_classes = random.choices(
        ["chain", "double", "single", "half_double", "treble"],
        weights=[5, 4, 3, 2, 1], k=random.randint(3, 8))
    y, row_num = margin, 0
    while y + stitch_h < img_size - margin:
        cls = row_classes[row_num % len(row_classes)]
        x = margin + random.randint(-5, 5)
        stitch_w = int(stitch_h * random.uniform(0.3, 0.8))
        gap_x = stitch_w + random.randint(2, 8)
        while x + stitch_w < img_size - margin:
            sym = get_symbol(cls, stitch_h, templates, color=fg)
            c = paste_symbol(bg, sym, x + stitch_w // 2, y + stitch_h // 2,
                             random.gauss(0, 3))
            if c is not None:
                labels.append((CLASS_MAP[cls], *obb_to_yolo(c, img_size, img_size)))
            x += gap_x
        y += row_gap
        row_num += 1
    _add_edge_noise(bg, labels, img_size, fg, margin)
    return _add_aging_texture(bg), labels


def generate_v_pattern_grid(img_size: int = 640
                            ) -> tuple[np.ndarray, list[LabelTuple]]:
    """Produce a V-stitch grid (paired rotated double crochets)."""
    palette = random.choice(list(PALETTES.values()))
    bg = np.full((img_size, img_size, 3), palette["bg"], dtype=np.uint8)
    fg = palette["fg"]
    templates = _CACHED_TEMPLATES.get_or_load()
    labels: list[LabelTuple] = []
    stitch_h = random.randint(35, 55)
    v_spread = random.randint(15, 30)
    row_gap = stitch_h + random.randint(8, 20)
    col_gap = random.randint(25, 45)
    margin = random.randint(25, 50)
    y, row_num = margin + stitch_h // 2, 0
    while y + stitch_h // 2 < img_size - margin:
        x = margin + ((col_gap // 2) if row_num % 2 == 1 else 0)
        while x + col_gap < img_size - margin:
            for ang, dx in [(v_spread, -col_gap // 6), (-v_spread, col_gap // 6)]:
                sym = get_symbol("double", stitch_h, templates, color=fg)
                c = paste_symbol(bg, sym, x + dx, y, ang + random.gauss(0, 3))
                if c is not None:
                    labels.append((CLASS_MAP["double"],
                                   *obb_to_yolo(c, img_size, img_size)))
            if random.random() < 0.7:
                ch = get_symbol("chain", int(stitch_h * 0.35), templates, color=fg)
                c = paste_symbol(bg, ch, x, y + stitch_h // 3, random.gauss(0, 5))
                if c is not None:
                    labels.append((CLASS_MAP["chain"],
                                   *obb_to_yolo(c, img_size, img_size)))
            x += col_gap
        y += row_gap
        row_num += 1
    _add_edge_noise(bg, labels, img_size, fg, margin)
    return _add_aging_texture(bg), labels


def generate_triangular_chart(img_size: int = 640
                              ) -> tuple[np.ndarray, list[LabelTuple]]:
    """Produce a triangular / expanding chart."""
    palette = random.choice(list(PALETTES.values()))
    bg = np.full((img_size, img_size, 3), palette["bg"], dtype=np.uint8)
    fg = palette["fg"]
    templates = _CACHED_TEMPLATES.get_or_load()
    labels: list[LabelTuple] = []
    stitch_h = random.randint(28, 45)
    row_gap = stitch_h + random.randint(5, 12)
    margin = random.randint(20, 40)
    n_rows = random.randint(6, 14)
    min_n, max_n = random.randint(2, 5), random.randint(15, 30)
    v_spread = random.randint(12, 25)
    for row_i in range(n_rows):
        t = row_i / max(n_rows - 1, 1)
        n = int(min_n + t * (max_n - min_n))
        y = img_size - margin - row_i * row_gap
        if y - stitch_h // 2 < margin:
            break
        row_w = n * (stitch_h * 0.6)
        x_start = (img_size - row_w) / 2
        gap = row_w / max(n, 1)
        for j in range(n):
            x = x_start + j * gap + gap / 2
            if random.random() < 0.7:
                for sgn, dx in [(1, -4), (-1, 4)]:
                    sym = get_symbol("double", stitch_h, templates, color=fg)
                    c = paste_symbol(bg, sym, int(x + dx), y,
                                     sgn * v_spread + random.gauss(0, 2))
                    if c is not None:
                        labels.append((CLASS_MAP["double"],
                                       *obb_to_yolo(c, img_size, img_size)))
            else:
                cls = random.choices(
                    ["double", "single", "chain", "half_double", "treble"],
                    weights=[4, 2, 3, 2, 1])[0]
                sym = get_symbol(cls, stitch_h, templates, color=fg)
                c = paste_symbol(bg, sym, int(x), y, random.gauss(0, 5))
                if c is not None:
                    labels.append((CLASS_MAP[cls],
                                   *obb_to_yolo(c, img_size, img_size)))
    _add_edge_noise(bg, labels, img_size, fg, margin)
    return _add_aging_texture(bg), labels


GeneratorFn = Callable[[int], tuple[np.ndarray, list[LabelTuple]]]
GENERATORS: Sequence[tuple[GeneratorFn, float]] = [
    (generate_v_pattern_grid,   0.40),
    (generate_triangular_chart, 0.30),
    (generate_dense_rows,       0.30),
]
"""``(generator, weight)`` pairs sampled by :func:`generate_dataset`."""


# ── Augmentation ──────────────────────────────────────────────────────────
def augment_image(img: np.ndarray) -> np.ndarray:
    """Apply random photographic augmentations to a BGR image."""
    out = img.copy()
    if random.random() < 0.5:
        alpha = random.uniform(0.85, 1.15)
        beta = random.randint(-15, 15)
        out = np.clip(alpha * out.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    if random.random() < 0.3:
        k = random.choice([3, 5])
        out = cv.GaussianBlur(out, (k, k), 0)
    if random.random() < 0.3:
        quality = random.randint(50, 90)
        _, enc = cv.imencode(".jpg", out, [cv.IMWRITE_JPEG_QUALITY, quality])
        out = cv.imdecode(enc, cv.IMREAD_COLOR)
    if random.random() < 0.2:
        h, w = out.shape[:2]
        M = cv.getRotationMatrix2D((w / 2, h / 2), random.uniform(-3, 3), 1.0)
        out = cv.warpAffine(out, M, (w, h), borderValue=(255, 255, 255))
    if random.random() < 0.2:
        scale = random.uniform(0.85, 1.0)
        h, w = out.shape[:2]
        nh, nw = int(h * scale), int(w * scale)
        y1 = random.randint(0, h - nh)
        x1 = random.randint(0, w - nw)
        out = cv.resize(out[y1:y1 + nh, x1:x1 + nw], (w, h))
    return out


# ── Template cache + dataset writer ───────────────────────────────────────
class _TemplateCache:
    """Lazy module-level cache of ``data/raw/templates`` so generators
    don't reload every call."""

    def __init__(self) -> None:
        self._loaded: dict[str, list[np.ndarray]] | None = None

    def get_or_load(self,
                    template_dir: str | Path = "data/raw/templates"
                    ) -> dict[str, list[np.ndarray]]:
        if self._loaded is None:
            self._loaded = load_templates(template_dir)
        return self._loaded

    def reset(self) -> None:
        self._loaded = None


_CACHED_TEMPLATES = _TemplateCache()


def generate_dataset(n_train: int = 300,
                     n_val: int = 60,
                     output_dir: str | Path = "data/synthetic",
                     img_size: int = 640) -> Path:
    """Generate a full YOLO-OBB dataset and write ``data.yaml``."""
    output_dir = Path(output_dir)
    train_img = output_dir / "train" / "images"
    train_lbl = output_dir / "train" / "labels"
    val_img   = output_dir / "val"   / "images"
    val_lbl   = output_dir / "val"   / "labels"
    for d in (train_img, train_lbl, val_img, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    gen_funcs, gen_weights = zip(*GENERATORS)

    def _produce(n: int, img_dir: Path, lbl_dir: Path, prefix: str) -> int:
        written = 0
        for i in range(n):
            gen = random.choices(gen_funcs, weights=gen_weights, k=1)[0]
            try:
                canvas, labels = gen(img_size)
            except Exception as exc:  # pragma: no cover — defensive
                print(f"  [warn] {gen.__name__}: {exc}")
                continue
            canvas = augment_image(canvas)
            valid: list[LabelTuple] = []
            for lbl in labels:
                xs = lbl[1::2]
                ys = lbl[2::2]
                if (max(xs) - min(xs)) > 0.005 and (max(ys) - min(ys)) > 0.005:
                    valid.append(lbl)
            if not valid:
                continue
            stem = f"{prefix}_{i:04d}"
            cv.imwrite(str(img_dir / f"{stem}.png"), canvas)
            with open(lbl_dir / f"{stem}.txt", "w") as f:
                for lbl in valid:
                    f.write(f"{int(lbl[0])} "
                            + " ".join(f"{c:.6f}" for c in lbl[1:]) + "\n")
            written += 1
            if (i + 1) % 50 == 0:
                print(f"  generated {i + 1}/{n}")
        return written

    print(f"Generating {n_train} train + {n_val} val images -> {output_dir}")
    _produce(n_train, train_img, train_lbl, "syn_train")
    _produce(n_val,   val_img,   val_lbl,   "syn_val")

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump({
            "path":  str(output_dir.resolve()),
            "train": "train/images",
            "val":   "val/images",
            "names": ID_TO_NAME,
        }, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {yaml_path}")
    return yaml_path


def mix_real_data(real_dir: str | Path,
                  output_dir: str | Path = "data/synthetic",
                  val_fraction: float = 0.25) -> tuple[int, int]:
    """Copy hand-labelled real images into the synthetic dataset."""
    real_dir = Path(real_dir)
    output_dir = Path(output_dir)
    train_img = output_dir / "train" / "images"
    train_lbl = output_dir / "train" / "labels"
    val_img   = output_dir / "val"   / "images"
    val_lbl   = output_dir / "val"   / "labels"
    for d in (train_img, train_lbl, val_img, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in (real_dir / "images").glob("*")
                     if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not images:
        return 0, 0

    n_val = max(1, int(len(images) * val_fraction))
    val_indices = set(random.sample(range(len(images)), n_val))
    n_train_copied = n_val_copied = 0
    for idx, img_path in enumerate(images):
        lbl_path = real_dir / "labels" / (img_path.stem + ".txt")
        if not lbl_path.is_file():
            continue
        if idx in val_indices:
            shutil.copy2(img_path, val_img / f"real_{img_path.name}")
            shutil.copy2(lbl_path, val_lbl / f"real_{img_path.stem}.txt")
            n_val_copied += 1
        else:
            shutil.copy2(img_path, train_img / f"real_{img_path.name}")
            shutil.copy2(lbl_path, train_lbl / f"real_{img_path.stem}.txt")
            n_train_copied += 1
    return n_train_copied, n_val_copied
