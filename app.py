"""Streamlit entry point for the public Crochet Studio app.

User flow:

1. Generate a chart with Gemini from a text prompt, *or* upload an image.
2. Click "Analyze stitches" — adaptive tiled YOLOv8n-OBB inference.
3. Get a black-line reconstructed scheme plus a JSON download.

Run from the project root::

    streamlit run app.py

The actual logic lives in :mod:`src.inference`, :mod:`src.rendering`,
:mod:`src.generation` and :mod:`src.pipeline`; this file only wires
those modules into Streamlit widgets.
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import streamlit as st
from PIL import Image

from src.config import (
    DEFAULT_CONF, DEFAULT_IOU, DEFAULT_OVERLAP, DEFAULT_TARGET_STITCH_PX,
    DEFAULT_TILE_SIZE, DEFAULT_WEIGHTS,
)
from src.generation import DEFAULT_PROMPT, generate_image
from src.inference import load_model, predict_adaptive_with_tiles
from src.pipeline import detections_to_json
from src.rendering import (
    render_reassembly, render_scheme, render_tile_grid, render_tiles_panel,
    render_two_panel,
)


EXAMPLES_DIR: Path = Path(__file__).resolve().parent / "data" / "examples"
"""Folder scanned by :func:`input_block` for the "Example image" tab.

Drop ``.png`` / ``.jpg`` files in here and they appear as a thumbnail
gallery — no other registration step needed.
"""

EXAMPLE_GALLERY_COLS: int = 3
"""How many thumbnails per row in the example-image gallery."""

EXAMPLE_THUMBNAIL_SIZE: int = 400
"""Side length, in pixels, of each square example thumbnail.

Thumbnails are rendered with ``use_container_width=True``, so the actual
on-screen size scales with the column width — but because every
thumbnail has the same 1:1 aspect ratio, they all end up the same
height regardless of the source image's shape.
"""


def configure_page() -> None:
    """Apply ``st.set_page_config`` and render the page title."""
    st.set_page_config(layout="wide", page_title="Crochet Scheme Generator")
    st.title("Crochet Symbol to Scheme")
    st.caption(
        "Generate a crochet diagram with Gemini or upload your own, "
        "then run adaptive tiled YOLOv8n-OBB inference."
    )


def sidebar_controls() -> dict[str, Any]:
    """Render the sidebar of inference hyperparameters."""
    with st.sidebar:
        st.header("Inference settings")
        weights_path = st.text_input("YOLO weights", value=DEFAULT_WEIGHTS)
        conf = st.slider("Confidence",        0.05, 0.9,  DEFAULT_CONF, 0.05)
        iou  = st.slider("NMS IoU threshold", 0.1,  0.9,  DEFAULT_IOU,  0.05)
        target_stitch = st.slider("Target stitch px (in tile)",
                                  40, 200, DEFAULT_TARGET_STITCH_PX, 5)
        tile_size = st.select_slider("Tile size",
                                     options=[384, 512, 640, 768, 1024],
                                     value=DEFAULT_TILE_SIZE)
        overlap = st.slider("Tile overlap", 0.05, 0.5, DEFAULT_OVERLAP, 0.05)
    return {
        "weights_path":     weights_path,
        "conf":             conf,
        "iou_threshold":    iou,
        "target_stitch_px": target_stitch,
        "tile_size":        tile_size,
        "overlap":          overlap,
    }


def _set_input_image(image_bytes: bytes, source: str) -> None:
    """Stash the current input image in session state.

    Anything routed through here becomes the source of truth for the
    rest of the page on subsequent reruns. Setting a new input also
    invalidates any analysis result computed against the previous one.

    Args:
        image_bytes: Raw image bytes (PNG / JPG).
        source: ``"gemini"`` or ``"upload"`` — used for telemetry and
            to remember which tab last produced the active input.
    """
    st.session_state["input_image_bytes"] = image_bytes
    st.session_state["input_source"] = source
    st.session_state.pop("analysis_result", None)


def _on_upload_change() -> None:
    """File-uploader callback: keep session state in sync with the widget.

    Streamlit fires ``on_change`` only when the *user* actually selects
    or clears a file, which is exactly when we want to update the input.
    Plain ``if upload is not None`` inside the render function would also
    fire on unrelated reruns and could clobber a freshly-generated image.
    """
    upload = st.session_state.get("upload_widget")
    if upload is None:
        # User cleared the file — drop the upload-sourced state, but
        # leave gemini-sourced state alone.
        if st.session_state.get("input_source") == "upload":
            st.session_state.pop("input_image_bytes", None)
            st.session_state.pop("input_source", None)
            st.session_state.pop("analysis_result", None)
        return
    _set_input_image(upload.getvalue(), "upload")


def _list_example_images() -> list[Path]:
    """Return image files under :data:`EXAMPLES_DIR`, sorted by name.

    Both ``.png`` and ``.jpg`` / ``.jpeg`` are picked up. The function is
    side-effect-free so it is safe to call on every rerun; if the folder
    is missing it returns an empty list.

    Returns:
        list[Path]: Discovered example-image paths, alphabetised.
    """
    if not EXAMPLES_DIR.is_dir():
        return []
    return sorted(
        p for p in EXAMPLES_DIR.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


@st.cache_data(show_spinner=False)
def _square_thumbnail(path_str: str, mtime: float,
                      size: int = EXAMPLE_THUMBNAIL_SIZE) -> bytes:
    """Center-crop an example image to a square and return PNG bytes.

    Crochet charts come in wildly different aspect ratios — squares,
    tall portraits, wide landscapes. Without normalisation, the gallery
    row would have a ragged bottom edge. This helper center-crops each
    image to a 1:1 square at a fixed pixel size so every thumbnail
    renders at exactly the same height once Streamlit scales it to the
    column width.

    Cached on ``(path_str, mtime)`` so the crop is computed once per
    file and reused on every rerun. Including ``mtime`` invalidates the
    cache automatically if a file is replaced on disk.

    Args:
        path_str: Absolute path of the source image, as a string.
        mtime: ``Path.stat().st_mtime`` — only used as a cache key.
        size: Side length of the resulting square, in pixels.

    Returns:
        bytes: PNG-encoded thumbnail.
    """
    del mtime  # only used for cache invalidation
    img = Image.open(path_str).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _render_examples_tab() -> None:
    """Render the example-image gallery inside the active Streamlit tab.

    Each thumbnail gets a "Use this" button that routes the file's bytes
    through :func:`_set_input_image` so the rest of the page treats it
    exactly like an uploaded image. Empty folder yields a friendly hint
    instead of an error. Thumbnails are forced to a 1:1 aspect ratio via
    :func:`_square_thumbnail` so the row never looks ragged.
    """
    examples = _list_example_images()
    if not examples:
        st.info(
            f"No example images found in `{EXAMPLES_DIR.relative_to(Path.cwd())}`. "
            "Drop a few `.png` or `.jpg` files there to populate this gallery."
        )
        return

    st.caption(
        "Pick one of the bundled crochet charts to skip generation and "
        "uploading."
    )
    for row_start in range(0, len(examples), EXAMPLE_GALLERY_COLS):
        row = examples[row_start:row_start + EXAMPLE_GALLERY_COLS]
        cols = st.columns(EXAMPLE_GALLERY_COLS)
        for col, path in zip(cols, row):
            with col:
                thumb = _square_thumbnail(str(path), path.stat().st_mtime)
                st.image(thumb, use_container_width=True)
                if st.button(f"Use {path.name}", key=f"example_{path.name}"):
                    _set_input_image(path.read_bytes(), "example")


def input_block() -> tuple[bytes | None, str | None]:
    """Render the input widget with three sources: examples, Gemini, upload.

    The widget never returns "transient" bytes: the active input lives
    in ``st.session_state`` so it survives the rerun caused by clicking
    "Analyze stitches". Without this persistence the generated image
    would appear to vanish on click, since ``st.button("Generate")``
    only returns ``True`` on the single rerun caused by *its* click.

    Returns:
        tuple[bytes | None, str | None]: ``(image_bytes, source)`` where
        ``source`` is ``"gemini"``, ``"upload"``, ``"example"``, or ``None``.
    """
    tab_gen, tab_up, tab_examples = st.tabs([
        "Generate with Gemini",
        "Upload image",
        "Example image",
    ])

    with tab_gen:
        # The API key is *user-supplied* on purpose: the deployed app
        # must not call Gemini on the developer's billing account, and
        # storing keys server-side via .env would do exactly that. The
        # widget persists the value in session state for the duration
        # of the session via its ``key=`` parameter — no on-disk write.
        api_key = st.text_input(
            "Your Google Gemini API key",
            type="password",
            key="gemini_api_key",
            placeholder="Paste your Google AI Studio API key",
            help=(
                "Get one for free at https://aistudio.google.com/apikey. "
                "The key stays in this browser session — it is never "
                "logged or stored on the server."
            ),
        )
        prompt = st.text_area("Prompt", value=DEFAULT_PROMPT, height=80)
        if st.button("Generate", key="generate"):
            if not api_key:
                st.error(
                    "Please paste your Gemini API key above before "
                    "clicking Generate."
                )
            else:
                try:
                    with tempfile.NamedTemporaryFile(suffix=".png",
                                                      delete=False) as tmp:
                        out_path = Path(tmp.name)
                    generate_image(
                        prompt=prompt,
                        output_path=out_path,
                        api_key=api_key,
                    )
                    _set_input_image(out_path.read_bytes(), "gemini")
                    st.success("Image generated.")
                except Exception as exc:
                    st.error(f"Gemini error: {exc}")

    with tab_up:
        st.file_uploader(
            "Upload PNG / JPG",
            type=["png", "jpg", "jpeg"],
            key="upload_widget",
            on_change=_on_upload_change,
        )

    with tab_examples:
        _render_examples_tab()

    image_bytes: bytes | None = st.session_state.get("input_image_bytes")
    source: str | None = st.session_state.get("input_source")

    if image_bytes:
        st.image(image_bytes,
                 caption=f"Input image ({source})",
                 use_container_width=True)
    return image_bytes, source


def render_tiling_inspector(image: np.ndarray,
                             tiles_info: list,
                             final_detections: list) -> None:
    """Render an "Inspect tiling" expander with three explanatory tabs.

    The tabs show how the adaptive tiler split the image, what each tile
    saw on its own, and how the per-tile detections are reconciled into
    a single coherent set via class-aware NMS.

    Args:
        image: BGR original image.
        tiles_info: List of :class:`src.inference.TileInfo` returned by
            :func:`src.inference.predict_adaptive_with_tiles`.
        final_detections: Final post-NMS detection list.
    """
    n_pre = sum(len(t.detections) for t in tiles_info)
    n_post = len(final_detections)
    eff = tiles_info[0].effective_size if tiles_info else 0

    with st.expander(
        f"Inspect tiling — {len(tiles_info)} tile(s), "
        f"{n_pre} raw -> {n_post} after NMS  |  "
        f"effective tile = {eff} px",
        expanded=True,
    ):
        st.caption(
            "The detector was trained at a fixed receptive scale, so the "
            "adaptive tiler estimates stitch size first, then chooses a tile "
            "size that puts stitches at that scale inside every YOLO call. "
            "Each tile is inferred independently; overlapping tiles produce "
            "duplicate detections in the seams, which class-aware NMS then "
            "removes."
        )
        tab_grid, tab_tiles, tab_glue = st.tabs([
            "Tile grid",
            "Per-tile detections",
            "Reassembly (before / after NMS)",
        ])
        with tab_grid:
            st.markdown(
                "How the original image is split. Numbers index each tile "
                "in row-major order; adjacent tiles share an overlap region "
                "controlled by the **Tile overlap** slider."
            )
            st.pyplot(render_tile_grid(image, tiles_info))
        with tab_tiles:
            st.markdown(
                "What every individual tile fed to YOLO actually contains, "
                "with that tile's raw detections drawn on top."
            )
            st.pyplot(render_tiles_panel(tiles_info, max_tiles=12))
        with tab_glue:
            st.markdown(
                "**Left** — every tile's raw detections plotted on the full "
                "image, colour-coded by tile index. Duplicates show up as "
                "two same-spot polygons in different tile colours, in the "
                "seam where two tiles overlap.  \n"
                "**Right** — the same image after class-aware NMS: each "
                "stitch is kept exactly once, drawn in its class colour."
            )
            st.pyplot(render_reassembly(image, tiles_info, final_detections))


def compute_inference(image_bytes: bytes,
                      settings: dict[str, Any]) -> dict[str, Any] | None:
    """Decode an image, run adaptive tiled inference, and return raw data.

    The result is meant to be stashed in ``st.session_state`` and
    re-rendered on every rerun by :func:`render_analysis`. Splitting
    "compute" from "render" is what keeps the visualisations alive when
    the user opens tabs or expanders, since those interactions trigger
    Streamlit reruns where ``st.button(...)`` would otherwise return
    ``False`` and the entire output would disappear.

    Args:
        image_bytes: Raw image bytes from the input block.
        settings: Inference settings from :func:`sidebar_controls`.

    Returns:
        dict[str, Any] | None: ``{"image", "detections", "tiles_info"}``
        on success, or ``None`` if decoding / model loading failed
        (errors are surfaced via ``st.error``).
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv.imdecode(nparr, cv.IMREAD_COLOR)
    if image is None:
        st.error("Could not decode image.")
        return None

    try:
        model = load_model(settings["weights_path"])
    except FileNotFoundError as exc:
        st.error(f"Could not load YOLO weights: {exc}")
        return None

    with st.spinner("Running adaptive tiled inference..."):
        detections, tiles_info = predict_adaptive_with_tiles(
            model, image,
            target_stitch_px=settings["target_stitch_px"],
            tile_size=settings["tile_size"],
            overlap=settings["overlap"],
            conf=settings["conf"],
            iou_threshold=settings["iou_threshold"],
        )
    return {
        "image":       image,
        "detections":  detections,
        "tiles_info":  tiles_info,
    }


def render_analysis(result: dict[str, Any]) -> None:
    """Render every panel from a previously-computed inference result.

    This function is called on *every* rerun once a result is stashed in
    ``st.session_state``, so the user can open tabs and toggle the
    expander freely without losing the figures.

    Args:
        result: Output of :func:`compute_inference`.
    """
    image       = result["image"]
    detections  = result["detections"]
    tiles_info  = result["tiles_info"]
    h, w = image.shape[:2]

    payload = detections_to_json(detections, None, w, h)
    st.success(f"Detected {payload['n_detections']} stitches.")
    st.json(payload["class_counts"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Detections + reconstructed scheme")
        st.pyplot(render_two_panel(image, detections))
    with col2:
        st.subheader("Clean scheme")
        st.image(render_scheme(detections, width=w, height=h),
                 use_container_width=True)

    # Make the tiling visualisation unmissable: divider + header above it.
    st.divider()
    st.header("How adaptive tiling worked")
    render_tiling_inspector(image, tiles_info, detections)

    st.download_button(
        "Download detections.json",
        data=json.dumps(payload, indent=2),
        file_name="detections.json",
        mime="application/json",
    )


def main() -> None:
    """Render the full single-page Streamlit application."""
    configure_page()
    settings = sidebar_controls()
    image_bytes, _ = input_block()
    if image_bytes is None:
        st.info("Generate a chart with Gemini or upload an image to begin.")
        return

    # ``st.button`` returns True only on the rerun caused by the click.
    # We therefore *compute* on click and *render* on every subsequent
    # rerun from session state — that's what keeps the tile-inspection
    # tabs alive when the user clicks around.
    if st.button("Analyze stitches", type="primary"):
        result = compute_inference(image_bytes, settings)
        if result is not None:
            st.session_state["analysis_result"] = result

    if "analysis_result" in st.session_state:
        render_analysis(st.session_state["analysis_result"])


if __name__ == "__main__":
    main()
else:
    # ``streamlit run app.py`` imports the module — render immediately.
    main()
