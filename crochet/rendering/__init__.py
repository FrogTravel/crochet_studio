"""Matplotlib and SVG rendering helpers.

``icons``   — per-stitch matplotlib glyph drawing.
``figures`` — image-level renderers (scheme, detection overlay, tile grid).
``svg``     — standalone SVG glyph generators used by the MCP server.
"""

from .figures import (
    draw_detection_overlay,
    draw_scheme,
    fig_to_pil,
    linewidth_for_count,
    render_detection_image,
    render_scheme_image,
    render_tile_grid_image,
)
from .icons import draw_svg_icon

__all__ = [
    "draw_svg_icon",
    "draw_scheme",
    "draw_detection_overlay",
    "fig_to_pil",
    "linewidth_for_count",
    "render_scheme_image",
    "render_detection_image",
    "render_tile_grid_image",
]
