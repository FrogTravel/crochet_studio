"""Create Instruction - end-user Streamlit page.

A polished, standalone-feeling page that turns a prompt and/or reference
image into a professional crochet instruction. It streams progress as the
pipeline runs: the generated image appears the moment Gemini returns it, a
dominant color palette is shown next to the image, a progress bar tracks
each phase, and the final instruction is rendered as formatted markdown.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


# ---------------------------------------------------------------------------
# Bootstrapping
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_dotenv_once() -> Path | None:
    if st.session_state.get("_env_loaded"):
        return st.session_state.get("_env_path")
    for candidate in (REPO_ROOT / "notebooks" / ".env", REPO_ROOT / ".env"):
        if candidate.is_file():
            for raw in candidate.read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
            st.session_state._env_loaded = True
            st.session_state._env_path = candidate
            return candidate
    st.session_state._env_loaded = True
    st.session_state._env_path = None
    return None


_env_path = _load_dotenv_once()

from crochet.palette import PaletteColor  # noqa: E402
from crochet.routing import capabilities, run_router  # noqa: E402


# ---------------------------------------------------------------------------
# Page chrome + custom CSS for a distinctly "separate" look
# ---------------------------------------------------------------------------

st.set_page_config(
    layout="wide",
    page_title="Create Crochet Instruction",
    page_icon="🧶",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
  /* Hide the default Streamlit page navigation so this page feels separate */
  [data-testid="stSidebarNav"] { display: none; }

  /* Hero header */
  .ci-hero {
    background: linear-gradient(135deg, #fdf6ec 0%, #f3e6d4 40%, #e6d3b3 100%);
    border: 1px solid #e6d3b3;
    border-radius: 16px;
    padding: 2rem 2.25rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 6px 24px rgba(120, 90, 60, 0.08);
  }
  .ci-hero h1 {
    font-family: Georgia, "Times New Roman", serif;
    font-size: 2.2rem;
    margin: 0 0 0.5rem 0;
    color: #3b2a15;
    letter-spacing: 0.01em;
  }
  .ci-hero p {
    margin: 0;
    color: #5a4326;
    font-size: 1.05rem;
    max-width: 680px;
    line-height: 1.5;
  }

  /* Panel cards */
  .ci-card {
    background: #ffffff;
    border: 1px solid #ece3d4;
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    box-shadow: 0 2px 10px rgba(60, 40, 20, 0.04);
    margin-bottom: 1rem;
  }
  .ci-card h3 {
    margin-top: 0;
    color: #3b2a15;
    font-family: Georgia, serif;
    font-size: 1.2rem;
    border-bottom: 1px solid #f0e7d7;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
  }

  /* Palette swatches */
  .ci-swatch-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 0.5rem;
  }
  .ci-swatch {
    flex: 1 1 110px;
    min-width: 110px;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #e6dcc9;
    box-shadow: 0 1px 4px rgba(60, 40, 20, 0.06);
    background: #fff;
  }
  .ci-swatch .swatch-color {
    height: 72px;
    width: 100%;
  }
  .ci-swatch .swatch-meta {
    padding: 6px 10px 8px 10px;
    font-size: 0.8rem;
    color: #3b2a15;
    line-height: 1.25;
  }
  .ci-swatch .swatch-name {
    font-weight: 600;
    text-transform: capitalize;
  }
  .ci-swatch .swatch-hex {
    font-family: "SF Mono", Consolas, monospace;
    color: #6b5a3d;
  }

  /* Compact capability chips */
  .ci-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
  .ci-chip {
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    border: 1px solid #e0d4bd;
    background: #fbf6ec;
    color: #5a4326;
  }
  .ci-chip.ok   { background: #eaf5ea; color: #2d6b2d; border-color: #b4dcb4; }
  .ci-chip.miss { background: #fbe9e9; color: #883030; border-color: #e6b5b5; }

  /* Instruction markdown wrapper */
  .ci-instruction {
    background: #fffdf8;
    border: 1px solid #ece3d4;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    box-shadow: 0 2px 10px rgba(60, 40, 20, 0.05);
  }
  .ci-instruction h1 {
    font-family: Georgia, serif;
    color: #3b2a15;
    border-bottom: 2px solid #d9c59e;
    padding-bottom: 0.4rem;
  }
  .ci-instruction h2 {
    font-family: Georgia, serif;
    color: #5a4326;
    margin-top: 1.4rem;
  }
  .ci-instruction table {
    border-collapse: collapse;
    margin: 0.6rem 0;
  }
  .ci-instruction th, .ci-instruction td {
    border: 1px solid #ece3d4;
    padding: 4px 10px;
  }
  .ci-instruction th {
    background: #faf3e3;
  }
</style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown(
    """
<div class="ci-hero">
  <h1>Create your crochet instruction</h1>
  <p>Describe the piece you want to make, upload a reference photo, or do both.
     The router will pick the right pipeline - a flat scheme or a 3D amigurumi -
     and deliver a polished, professional instruction complete with a yarn
     color palette sampled from your image.</p>
</div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Capability chips (compact, inline - no sidebar)
# ---------------------------------------------------------------------------

caps = capabilities()


def _chip(label: str, ok: bool) -> str:
    cls = "ok" if ok else "miss"
    mark = "✓" if ok else "×"
    return f'<span class="ci-chip {cls}">{mark} {label}</span>'


chips_html = (
    "<div class='ci-chips'>"
    + _chip("ModernBERT", caps["transformers"])
    + _chip("Gemini",     caps["genai"] and caps["gemini_key"])
    + _chip("YOLO OBB",   caps["ultralytics"] and caps["yolo_weights"])
    + _chip("Hunyuan3D",  caps["gradio_client"])
    + _chip("trimesh",    caps["trimesh"])
    + "</div>"
)
st.markdown(chips_html, unsafe_allow_html=True)
if _env_path:
    st.caption(f".env loaded from `{_env_path.relative_to(REPO_ROOT)}`")


# ---------------------------------------------------------------------------
# Input form (prompt + image + button)
# ---------------------------------------------------------------------------

st.markdown("<div class='ci-card'>", unsafe_allow_html=True)
st.markdown("### Your design brief")

col_prompt, col_upload = st.columns([3, 2])
with col_prompt:
    prompt_text = st.text_area(
        "Describe what you want to crochet",
        placeholder=(
            "e.g. 'A cute teddy bear amigurumi, around 15 cm tall, with "
            "pastel colors' - or 'A lace doily chart with an 8-pointed star'."
        ),
        height=140,
        key="ci_prompt",
    )

with col_upload:
    uploaded_file = st.file_uploader(
        "Reference image (optional)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
        key="ci_uploader",
        help="Upload a photo of a finished piece you want to replicate, or a "
             "scheme you want analysed. Leave blank to have Gemini generate "
             "one from your prompt.",
    )
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Your reference", use_container_width=True)

run_clicked = st.button(
    "✨  Create instruction",
    type="primary",
    disabled=(not prompt_text.strip() and uploaded_file is None),
    use_container_width=True,
)
st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Live-progress rendering helpers
# ---------------------------------------------------------------------------

def _render_palette(slot, palette) -> None:
    """Render a swatch row for a list of PaletteColor / dict entries."""
    if not palette:
        slot.empty()
        return
    html_parts = ["<div class='ci-swatch-row'>"]
    for p in palette:
        if isinstance(p, PaletteColor):
            hex_ = p.hex; name = p.name; weight = p.weight
        else:
            hex_ = p.get("hex", "#000000")
            name = p.get("name", "neutral")
            weight = float(p.get("weight", 0))
        html_parts.append(
            "<div class='ci-swatch'>"
            f"  <div class='swatch-color' style='background:{hex_};'></div>"
            "  <div class='swatch-meta'>"
            f"    <div class='swatch-name'>{name}</div>"
            f"    <div class='swatch-hex'>{hex_} - {weight:.0%}</div>"
            "  </div>"
            "</div>"
        )
    html_parts.append("</div>")
    slot.markdown("".join(html_parts), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if run_clicked:
    # Persist the upload to a temp file for the router.
    image_path: Path | None = None
    if uploaded_file is not None:
        (REPO_ROOT / "notebooks_output").mkdir(exist_ok=True)
        suffix = Path(uploaded_file.name).suffix or ".png"
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, dir=str(REPO_ROOT / "notebooks_output"),
        )
        tmp.write(uploaded_file.getvalue())
        tmp.close()
        image_path = Path(tmp.name)

    work_dir = REPO_ROOT / "notebooks_output" / "create_instruction"

    # --- progress + live artefact slots ----------------------------------

    st.markdown("<div class='ci-card'>", unsafe_allow_html=True)
    st.markdown("### Generation progress")
    status_text = st.empty()
    progress_bar = st.progress(0.0)

    live_cols = st.columns([3, 2])
    with live_cols[0]:
        st.caption("Generated / uploaded image")
        image_slot = st.empty()
    with live_cols[1]:
        st.caption("Color palette")
        palette_slot = st.empty()

    # Separate row for 3D projection views (shown only when views arrive)
    views_header_slot = st.empty()
    views_cols_slot = st.empty()

    with st.expander("Show detailed log", expanded=False):
        log_slot = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

    log_lines: list[str] = []
    shown_images: set[str] = set()
    view_images: list[tuple[str, str]] = []

    def on_log(msg: str) -> None:
        log_lines.append(f"• {msg}")
        log_slot.code("\n".join(log_lines[-200:]))

    def on_phase(name: str, pct: float) -> None:
        status_text.markdown(f"**{name}**  &nbsp; _{int(pct*100)}%_")
        progress_bar.progress(min(max(pct, 0.0), 1.0))

    def on_image(path: Path, label: str) -> None:
        key = str(path)
        if key in shown_images or not Path(path).is_file():
            return
        shown_images.add(key)
        lbl = (label or "").lower()
        if lbl.startswith("3d view:"):
            # Show projection views in their own row, don't replace the generated image
            view_images.append((str(path), label or path.name))
            views_header_slot.caption("3D projection views")
            cols = views_cols_slot.columns(len(view_images))
            for i, (vpath, vcap) in enumerate(view_images):
                cols[i].image(vpath, caption=vcap, use_container_width=True)
        else:
            # Keep the Hunyuan3D / Gemini generated image in its own slot
            image_slot.image(str(path), caption=label or path.name, use_container_width=True)

    def on_palette(colors) -> None:
        _render_palette(palette_slot, colors)

    t0 = time.time()
    try:
        result = run_router(
            prompt=prompt_text,
            image_path=image_path,
            work_dir=work_dir,
            on_log=on_log,
            on_phase=on_phase,
            on_image=on_image,
            on_palette=on_palette,
        )
    except Exception as exc:
        status_text.error(f"Router failed: {exc}")
        st.exception(exc)
        st.stop()

    elapsed = time.time() - t0
    status_text.success(f"✓ Finished in {elapsed:.1f}s - routed to "
                        f"**{result['path'].upper()}** pipeline")
    progress_bar.progress(1.0)

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------

    # Summary card
    st.markdown("<div class='ci-card'>", unsafe_allow_html=True)
    st.markdown("### Pattern summary")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Pipeline", result["path"].upper())
    sc2.metric("Label", result["label"])
    top_score = max(result.get("scores", {}).values() or [0.0])
    sc3.metric("Confidence", f"{top_score:.0%}")
    if result["path"] == "2d":
        sc4.metric("Stitches", result.get("num_detections", 0))
    else:
        sc4.metric("Rounds", len(result.get("amigurumi_rows", [])))
    with st.expander("Classifier scores"):
        st.json(result.get("scores", {}))
    if result.get("note"):
        st.info(result["note"])
    st.markdown("</div>", unsafe_allow_html=True)

    # Final palette (in case on_palette missed it)
    final_palette = result.get("palette") or []
    if final_palette:
        st.markdown("<div class='ci-card'>", unsafe_allow_html=True)
        st.markdown("### Color palette")
        pal_slot = st.empty()
        _render_palette(pal_slot, final_palette)
        st.markdown("</div>", unsafe_allow_html=True)

    # Instruction - rendered as styled HTML (matches the reference book
    # layout) with a plain-markdown fallback / download for portability.
    instruction_md   = result.get("instruction")      or "_(no instruction produced)_"
    instruction_html = result.get("instruction_html") or ""

    st.markdown("### Your crochet tutorial")
    st.caption(
        "Rendered below the way it would appear in a printed pattern book. "
        "Scroll inside the frame to read the full document, or download a "
        "self-contained copy."
    )

    if instruction_html:
        components.html(instruction_html, height=1600, scrolling=True)
    else:
        st.markdown("<div class='ci-instruction'>", unsafe_allow_html=True)
        st.markdown(instruction_md)
        st.markdown("</div>", unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "⬇ Download tutorial (.html)",
            data=(instruction_html or f"<pre>{instruction_md}</pre>").encode("utf-8"),
            file_name="crochet_tutorial.html",
            mime="text/html",
            use_container_width=True,
            help="Self-contained styled HTML - open in any browser or print "
                 "to PDF for a pattern-book look.",
        )
    with d2:
        st.download_button(
            "⬇ Download instruction (.md)",
            data=instruction_md.encode("utf-8"),
            file_name="crochet_instruction.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with d3:
        st.download_button(
            "⬇ Download instruction (.txt)",
            data=instruction_md.encode("utf-8"),
            file_name="crochet_instruction.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with st.expander("View plain-markdown version"):
        st.markdown(instruction_md)

    # Path-specific artefacts
    if result["path"] == "2d":
        st.markdown("<div class='ci-card'>", unsafe_allow_html=True)
        st.markdown("### Scheme artefacts")
        cols = st.columns(2)
        img_path = result.get("image_path")
        fig_path = result.get("figure_path")
        with cols[0]:
            st.caption("Source image")
            if img_path and Path(img_path).is_file():
                st.image(img_path, use_container_width=True)
            else:
                st.info("No source image available.")
        with cols[1]:
            st.caption("YOLO OBB predictions")
            if fig_path and Path(fig_path).is_file():
                st.image(fig_path, use_container_width=True)
            else:
                st.info("No prediction figure available.")
        counts = result.get("counts", {})
        if counts:
            st.write("**Per-class stitch counts**")
            st.json(counts)
        if fig_path and Path(fig_path).is_file():
            with open(fig_path, "rb") as f:
                st.download_button(
                    "⬇ Download annotated scheme (.png)",
                    data=f.read(),
                    file_name="scheme_predictions.png",
                    mime="image/png",
                )
        st.markdown("</div>", unsafe_allow_html=True)

    else:  # 3d
        st.markdown("<div class='ci-card'>", unsafe_allow_html=True)
        st.markdown("### 3D artefacts")
        ref_path = result.get("reference_image_path")
        if ref_path and Path(ref_path).is_file():
            st.caption("Reference image")
            st.image(ref_path, width=360)
        views = [Path(p) for p in result.get("views", []) if Path(p).is_file()]
        if views:
            raw_src = result.get("mesh_source", "?")
            src_label = (
                raw_src.replace("hunyuan3d:", "") if raw_src.startswith("hunyuan3d:") else
                "Cached GLB" if raw_src == "fallback" else
                "Demo shape" if raw_src == "demo" else raw_src
            )
            st.caption(f"Multi-view previews ({src_label})")
            cols = st.columns(len(views))
            for col, vp in zip(cols, views):
                with col:
                    st.image(str(vp), caption=vp.stem, use_container_width=True)
        va = result.get("view_analysis", [])
        if va:
            st.write("**Per-view YOLO detections**")
            st.json(va)
        rows = result.get("amigurumi_rows", [])
        if rows:
            with st.expander("Raw row list (JSON)"):
                st.json(rows)
        glb_path = result.get("glb_path")
        if glb_path and Path(glb_path).is_file():
            with open(glb_path, "rb") as f:
                st.download_button(
                    "⬇ Download mesh (.glb)",
                    data=f.read(),
                    file_name="mesh.glb",
                    mime="model/gltf-binary",
                )
        st.markdown("</div>", unsafe_allow_html=True)
