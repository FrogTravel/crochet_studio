"""Dominant-color palette extraction for the Create Instruction UI.

Uses Pillow's built-in median-cut quantiser so we avoid pulling in scikit-
learn or OpenCV for what is fundamentally a histogram-of-colors task.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass
class PaletteColor:
    hex: str
    rgb: tuple[int, int, int]
    weight: float         # fraction of pixels represented, in [0, 1]
    name: str             # human-readable name (e.g. "soft cream")

    def as_dict(self) -> dict:
        return {
            "hex":    self.hex,
            "rgb":    list(self.rgb),
            "weight": round(self.weight, 3),
            "name":   self.name,
        }


def _nearest_color_name(r: int, g: int, b: int) -> str:
    """Return a crochet-friendly color name for an RGB tuple.

    Deliberately small vocabulary - we want names a crafter can match to a
    real yarn label (Red Heart / Lion Brand style), not a 600-entry swatch
    book.
    """
    anchors: list[tuple[str, tuple[int, int, int]]] = [
        ("ivory",         (245, 240, 225)),
        ("cream",         (235, 220, 195)),
        ("soft white",    (250, 250, 248)),
        ("warm beige",    (210, 180, 140)),
        ("taupe",         (139, 121, 94)),
        ("chocolate",     (95,  65,  40)),
        ("caramel",       (175, 110, 60)),
        ("rust",          (180, 80,  45)),
        ("terracotta",    (205, 120, 95)),
        ("blush pink",    (240, 190, 190)),
        ("rose",          (210, 95,  120)),
        ("dusty pink",    (210, 160, 170)),
        ("coral",         (245, 130, 115)),
        ("mustard",       (220, 170, 50)),
        ("sunflower",     (245, 205, 90)),
        ("sage",          (165, 180, 140)),
        ("olive",         (120, 130, 70)),
        ("forest",        (55,  95,  70)),
        ("mint",          (170, 220, 195)),
        ("teal",          (55,  125, 135)),
        ("sky",           (150, 195, 225)),
        ("denim",         (70,  110, 160)),
        ("navy",          (30,  45,  80)),
        ("lavender",      (195, 175, 220)),
        ("plum",          (110, 70,  120)),
        ("charcoal",      (60,  60,  65)),
        ("black",         (20,  20,  20)),
        ("silver",        (190, 190, 195)),
    ]
    best_name = "neutral"
    best_d = 10 ** 9
    for name, (ar, ag, ab) in anchors:
        d = (ar - r) ** 2 + (ag - g) ** 2 + (ab - b) ** 2
        if d < best_d:
            best_d = d
            best_name = name
    return best_name


def extract_palette(image_path: str | Path, n_colors: int = 5) -> list[PaletteColor]:
    """Return ``n_colors`` dominant PaletteColors, sorted by weight desc.

    Returns an empty list if Pillow is not available or the image can't be
    read (caller should handle that case gracefully).
    """
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return []

    path = Path(image_path)
    if not path.is_file():
        return []

    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return []

    # Downsample so quantisation is fast regardless of input resolution.
    img.thumbnail((256, 256))
    try:
        # Median-cut is the most visually pleasing for photographs.
        q = img.quantize(colors=max(1, n_colors), method=Image.Quantize.MEDIANCUT)
    except Exception:
        q = img.quantize(colors=max(1, n_colors))

    pal = q.getpalette() or []
    hist = q.getcolors() or []
    total = sum(count for count, _ in hist) or 1

    colors: list[PaletteColor] = []
    for count, idx in sorted(hist, key=lambda c: -c[0]):
        start = idx * 3
        if start + 3 > len(pal):
            continue
        r, g, b = pal[start], pal[start + 1], pal[start + 2]
        colors.append(PaletteColor(
            hex=f"#{r:02X}{g:02X}{b:02X}",
            rgb=(r, g, b),
            weight=count / total,
            name=_nearest_color_name(r, g, b),
        ))
        if len(colors) >= n_colors:
            break
    return colors


def palette_to_markdown(palette: list[PaletteColor]) -> str:
    """Render a palette as a tidy markdown table suitable for instructions."""
    if not palette:
        return ""
    lines = [
        "| Swatch | Hex | Suggested yarn color | Share |",
        "|:---:|:---:|:---|:---:|",
    ]
    for c in palette:
        lines.append(
            f"| `■` {c.hex} | `{c.hex}` | {c.name.title()} | {c.weight:.0%} |"
        )
    return "\n".join(lines)
