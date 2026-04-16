"""Matplotlib glyph drawing for individual crochet stitches.

Each label maps to one of a handful of drawing routines (chain ovals,
T-stem stitches with varying numbers of bars, X single crochet, ensemble
chains, etc.). Every glyph is drawn inside an OBB at the pose
``(x, y, angle_deg)``.
"""

from __future__ import annotations

import matplotlib.patches as patches
import matplotlib.transforms as transforms


_TALL_STITCH_LABELS = {"half_double", "double", "double treble", "treble", "fan"}


def draw_svg_icon(
    ax,
    label: str,
    x: float,
    y: float,
    w: float,
    h: float,
    angle_deg: float,
    color: str,
    lw: float = 2.0,
) -> None:
    """Draw one crochet symbol at pose ``(x, y, angle_deg)``.

    ``lw`` controls the main stroke thickness; secondary accents are
    drawn at 0.75 × ``lw`` so the visual hierarchy is preserved.
    """
    t = transforms.Affine2D().rotate_deg_around(x, y, angle_deg) + ax.transData
    lw_sub = lw * 0.75

    if label == "chain":
        ax.add_patch(patches.Ellipse(
            (x, y), w * 0.8, h * 0.4,
            fill=False, color=color, linewidth=lw, transform=t,
        ))
        return

    if label in _TALL_STITCH_LABELS:
        # stem + bar cap
        ax.plot([x, x], [y - h / 2, y + h / 2], color=color, lw=lw, transform=t)
        ax.plot([x - w / 3, x + w / 3], [y - h / 2, y - h / 2], color=color, lw=lw, transform=t)
        if label == "double":
            ax.plot([x - w / 4, x + w / 4], [y - h / 8, y + h / 8], color=color, lw=lw_sub, transform=t)
        elif label == "treble":
            ax.plot([x - w / 4, x + w / 4], [y - h / 4, y - h / 12], color=color, lw=lw_sub, transform=t)
            ax.plot([x - w / 4, x + w / 4], [y + h / 12, y + h / 4], color=color, lw=lw_sub, transform=t)
        elif label == "double treble":
            ax.plot([x - w / 4, x + w / 4], [y - h / 3, y - h / 6], color=color, lw=lw_sub, transform=t)
            ax.plot([x - w / 4, x + w / 4], [y - h / 12, y + h / 12], color=color, lw=lw_sub, transform=t)
            ax.plot([x - w / 4, x + w / 4], [y + h / 6, y + h / 3], color=color, lw=lw_sub, transform=t)
        elif label == "fan":
            ax.plot([x, x - w / 3], [y + h / 2, y - h / 2], color=color, lw=lw_sub, transform=t, alpha=0.6)
            ax.plot([x, x + w / 3], [y + h / 2, y - h / 2], color=color, lw=lw_sub, transform=t, alpha=0.6)
        return

    if label == "enseble_chain":
        n_ovals = max(2, int(h / (w * 0.5)))
        oval_h = h / n_ovals
        for ci in range(n_ovals):
            cy = y - h / 2 + oval_h * (ci + 0.5)
            ax.add_patch(patches.Ellipse(
                (x, cy), w * 0.6, oval_h * 0.7,
                fill=False, color=color, linewidth=lw_sub, transform=t,
            ))
        return

    if label == "noise":
        # Rendered invisibly — noise boxes aren't meaningful stitches.
        return

    # Default (single crochet and unknown labels) → X
    ax.plot([x - w / 3, x + w / 3], [y - h / 3, y + h / 3], color=color, lw=lw, transform=t)
    ax.plot([x - w / 3, x + w / 3], [y + h / 3, y - h / 3], color=color, lw=lw, transform=t)
