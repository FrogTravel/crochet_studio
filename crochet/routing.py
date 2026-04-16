"""End-to-end router that turns (prompt, image) into a crochet instruction.

This module is the reusable Python API behind the notebook
``notebooks/complex_scheme_generation.ipynb`` *and* the Streamlit page
``pages/2_Create_Instruction.py``.

Pipeline:

1. **Classify** the prompt with ModernBERT zero-shot into ``2d_scheme`` or
   ``3d_object``. If only an image is given we fall back to a rule based on
   the image's aspect ratio and the optional caption.
2. **Route** to one of two paths:

   * ``2d`` — get/generate a flat diagram image, run YOLOv8n OBB,
     reconstruct a clean scheme, produce per-class counts and a
     professional natural language stitch instruction.
   * ``3d`` — get/generate a reference image, call Hunyuan3D image-to-3D,
     render multi-view screenshots, slice the mesh horizontally, and emit
     an amigurumi row-by-row pattern.

3. Return a single dict with every artefact (paths, counts, text,
   classifier scores, color palette) so the UI can render whatever makes
   sense.

The router also accepts **streaming callbacks** so that Streamlit (or any
other progressive UI) can display intermediate artefacts as soon as they
are produced:

* ``on_phase(name, pct)`` - called at the start of each named phase.
* ``on_image(path, label)`` - called when a new image file is ready to
  display (uploaded, Gemini-generated, multi-view render, ...).
* ``on_palette(colors)`` - called when a dominant-color palette has been
  extracted from the source image.
* ``on_log(msg)`` - called with per-step diagnostic messages.

All callbacks are optional; the router degrades gracefully when any
dependency is missing.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from .config import (
    CLASS_CONFIG,
    DEFAULT_CONF,
    DEFAULT_IOU,
    DEFAULT_OVERLAP,
    DEFAULT_TARGET_STITCH_PX,
    DEFAULT_TILE_SIZE,
    DEFAULT_WEIGHTS,
)
from .html_instruction import build_2d_html, build_3d_html
from .palette import PaletteColor, extract_palette, palette_to_markdown


# ---------------------------------------------------------------------------
# Streaming callback plumbing
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Callbacks:
    """Bundle of UI callbacks used to stream progress into a view."""
    on_phase:   Callable[[str, float], None] | None = None
    on_image:   Callable[[Path, str], None] | None = None
    on_palette: Callable[[list[PaletteColor]], None] | None = None
    on_log:     Callable[[str], None] | None = None

    def phase(self, name: str, pct: float) -> None:
        if self.on_phase: self.on_phase(name, pct)
        if self.on_log: self.on_log(f"[{int(pct*100):3d}%] {name}")

    def image(self, path: Path, label: str = "") -> None:
        if self.on_image and path and Path(path).is_file():
            self.on_image(Path(path), label)

    def palette(self, colors: list[PaletteColor]) -> None:
        if self.on_palette and colors:
            self.on_palette(colors)

    def log(self, msg: str) -> None:
        if self.on_log: self.on_log(msg)


# ---------------------------------------------------------------------------
# Feature probing
# ---------------------------------------------------------------------------

def _has(mod: str) -> bool:
    try:
        importlib.import_module(mod); return True
    except Exception:
        return False


def capabilities() -> dict[str, bool]:
    """Report which optional dependencies are available right now."""
    return {
        "transformers":  _has("transformers"),
        "langchain":     _has("langchain_core"),
        "trimesh":       _has("trimesh"),
        "genai":         _has("google.genai"),
        "gradio_client": _has("gradio_client"),
        "ultralytics":   _has("ultralytics"),
        "pillow":        _has("PIL"),
        "gemini_key":    bool(
            os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        ),
        "yolo_weights":  Path(DEFAULT_WEIGHTS).is_file(),
    }


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

ZSC_LABELS = {
    "2d_scheme": (
        "A flat crochet chart or motif drawn on paper - a symbol diagram "
        "readable as a single image (granny square, doily, lace motif, stitch chart)."
    ),
    "3d_object": (
        "A physical three-dimensional crocheted object that cannot be represented "
        "by a single flat image - a toy, amigurumi, hat, bag, sweater, sock, shoe, "
        "or any wearable garment."
    ),
}


def _classify_rule_based(prompt: str) -> tuple[str, dict[str, float]]:
    p = prompt.lower()
    td = [
        "hat", "beanie", "sock", "shoe", "boot", "sweater", "cardigan",
        "scarf", "bag", "purse", "doll", "toy", "amigurumi", "plush",
        "dress", "skirt", "mitten", "glove", "slipper", "blanket",
    ]
    flat = [
        "chart", "motif", "diagram", "scheme", "granny square",
        "doily", "lace", "mandala", "coaster", "placemat",
    ]
    s3d = sum(1 for w in td if w in p)
    s2d = sum(1 for w in flat if w in p)
    if s3d == 0 and s2d == 0:
        scores = {"2d_scheme": 0.55, "3d_object": 0.45}
    elif s3d >= s2d:
        conf = 0.55 + 0.1 * min(s3d, 4)
        scores = {"3d_object": conf, "2d_scheme": 1 - conf}
    else:
        conf = 0.55 + 0.1 * min(s2d, 4)
        scores = {"2d_scheme": conf, "3d_object": 1 - conf}
    return max(scores, key=scores.get), scores


_zsc_pipe = None


def classify_prompt(prompt: str) -> tuple[str, dict[str, float]]:
    """Return ``(label, {label: score})`` for ``prompt``."""
    global _zsc_pipe
    if not prompt.strip():
        return "2d_scheme", {"2d_scheme": 0.5, "3d_object": 0.5}
    if not _has("transformers"):
        return _classify_rule_based(prompt)
    if _zsc_pipe is None:
        from transformers import pipeline
        _zsc_pipe = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/ModernBERT-large-zeroshot-v2.0",
        )
    labels = list(ZSC_LABELS.keys())
    descriptions = [ZSC_LABELS[l] for l in labels]
    out = _zsc_pipe(
        prompt,
        candidate_labels=descriptions,
        hypothesis_template="This text is asking for {}.",
        multi_label=False,
    )
    desc_to_label = {ZSC_LABELS[l]: l for l in labels}
    scores = {desc_to_label[d]: float(s) for d, s in zip(out["labels"], out["scores"])}
    return max(scores, key=scores.get), scores


def _classify_image_shape(image_path: Path) -> tuple[str, dict[str, float]]:
    try:
        import cv2 as cv  # type: ignore
        img = cv.imread(str(image_path))
        if img is None:
            raise RuntimeError("could not read image")
        h, w = img.shape[:2]
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        white_frac = float((gray > 230).mean())
        ratio = max(h, w) / max(1, min(h, w))
        if white_frac > 0.55 and ratio < 1.25:
            return "2d_scheme", {"2d_scheme": 0.75, "3d_object": 0.25}
        return "3d_object", {"3d_object": 0.7, "2d_scheme": 0.3}
    except Exception:
        return "3d_object", {"3d_object": 0.55, "2d_scheme": 0.45}


# ---------------------------------------------------------------------------
# Professional instruction formatting
# ---------------------------------------------------------------------------

_CLASS_FRIENDLY: dict[str, str] = {}
for _cid, _meta in CLASS_CONFIG.items():
    if isinstance(_meta, dict):
        _name = _meta.get("name", str(_cid))
        _label = _meta.get("label", _name)
        _CLASS_FRIENDLY[_name] = _label


def _difficulty_from_counts(total: int) -> str:
    if total < 30:   return "Beginner"
    if total < 120:  return "Easy"
    if total < 300:  return "Intermediate"
    return "Advanced"


def _difficulty_from_rows(rows: int) -> str:
    if rows < 10:   return "Beginner"
    if rows < 20:   return "Easy"
    if rows < 40:   return "Intermediate"
    return "Advanced"


def _estimate_size_from_rows(rows: list["AmiRow"], stitch_height_cm: float = 0.6) -> str:
    if not rows:
        return "unknown"
    height_cm = max(1, len(rows)) * stitch_height_cm
    max_sts = max((r.stitches for r in rows), default=6)
    # Rough: circumference = stitches / stitches_per_cm, with sts/cm = 2.
    max_diam_cm = max_sts / 2.0 / 3.1416 * 2
    return f"approx. {height_cm:.0f} cm tall x {max_diam_cm:.0f} cm wide"


def format_2d_instruction(
    prompt: str,
    counts: dict[str, int],
    total: int,
    palette: list[PaletteColor] | None = None,
) -> str:
    """Professionally formatted markdown instruction for the 2D path."""
    title = (prompt.strip() or "Custom Crochet Scheme").rstrip(".!?")
    title = title[0].upper() + title[1:] if title else title
    palette = palette or []
    out: list[str] = []

    out.append(f"# {title}")
    out.append("")
    out.append(f"_A detected-from-chart crochet pattern, auto-generated from "
               f"your scheme image._")
    out.append("")
    out.append("## Overview")
    out.append("")
    if total == 0:
        out.append("No stitch symbols were detected on the uploaded chart. "
                   "Double-check that the scheme is on a light background, "
                   "is in focus, and uses standard crochet symbols.")
    else:
        unique = len(counts)
        out.append(
            f"- **Total stitches:** {total}\n"
            f"- **Distinct symbol types:** {unique}\n"
            f"- **Difficulty:** {_difficulty_from_counts(total)}\n"
            f"- **Pattern style:** Worked from a symbol chart"
        )
    out.append("")

    out.append("## Materials")
    out.append("")
    out.append("- **Yarn:** DK or worsted weight cotton/acrylic, approx. 100 g.")
    out.append("- **Hook:** size matching the yarn band (3.5 - 5.0 mm).")
    out.append("- **Notions:** tapestry needle, scissors, stitch markers.")
    out.append("")

    out.append("## Gauge")
    out.append("")
    out.append("Work a 10 x 10 cm swatch in double crochet with the suggested "
               "yarn and hook. Block the swatch and measure; adjust the hook "
               "size up or down to match your preferred drape.")
    out.append("")

    out.append("## Abbreviations")
    out.append("")
    out.append(
        "| Abbreviation | Stitch |\n"
        "|:---:|:---|\n"
        "| ch | chain |\n"
        "| sl st | slip stitch |\n"
        "| sc | single crochet |\n"
        "| hdc | half double crochet |\n"
        "| dc | double crochet |\n"
        "| tr | treble crochet |\n"
        "| sk | skip |"
    )
    out.append("")

    if palette:
        out.append("## Suggested Color Palette")
        out.append("")
        out.append("Colors sampled from your reference image - use them as "
                   "yarn-color inspiration:")
        out.append("")
        out.append(palette_to_markdown(palette))
        out.append("")

    if counts:
        out.append("## Stitch Inventory")
        out.append("")
        out.append("| Symbol | Count |")
        out.append("|:---|:---:|")
        for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            pretty = _CLASS_FRIENDLY.get(name, name.replace("_", " ").title())
            out.append(f"| {pretty} | {n} |")
        out.append("")

    out.append("## Instructions")
    out.append("")
    out.append("1. **Start at the centre** of the chart. Make a magic ring "
               "(or chain 4-6 and join with a slip stitch into a ring if the "
               "chart opens flat).")
    out.append("2. **Work outward**, one concentric round at a time, reading "
               "the chart counter-clockwise (clockwise for left-handed).")
    out.append("3. **Follow the symbols** - each symbol on the chart maps to "
               "a stitch in the Abbreviations table above. Work each stitch "
               "into the stitch directly beneath the symbol.")
    out.append("4. **Close each round** with a slip stitch into the first "
               "stitch of the round, unless the chart clearly spirals.")
    out.append("5. **Turn work** only if the chart shows an explicit turn "
               "chain; otherwise continue in the same direction.")
    out.append("")

    out.append("## Finishing")
    out.append("")
    out.append("After the final round, fasten off leaving a 15 cm tail. "
               "Weave in all ends on the wrong side, wet-block the finished "
               "piece to its full size, and let it dry flat.")
    out.append("")

    out.append("## Notes")
    out.append("")
    out.append("- The detected stitch counts come from computer-vision "
               "analysis of your chart - minor miscounts are possible on "
               "busy motifs. Cross-check with the chart image before starting.")
    out.append("- If you'd like a specific gauge or finished size, swap the "
               "suggested hook for one that gives that gauge over a 10 cm "
               "swatch.")
    return "\n".join(out)


def _format_3d_instruction(
    prompt: str,
    rows: list["AmiRow"],
    palette: list[PaletteColor] | None = None,
) -> str:
    """Professionally formatted markdown instruction for the 3D path."""
    title = (prompt.strip() or "Custom Amigurumi").rstrip(".!?")
    title = title[0].upper() + title[1:] if title else title
    palette = palette or []
    out: list[str] = []

    out.append(f"# {title}")
    out.append("")
    out.append(f"_An amigurumi pattern generated from a 3D model of your "
               f"prompt. Worked in continuous rounds of single crochet._")
    out.append("")

    out.append("## Overview")
    out.append("")
    out.append(
        f"- **Pattern type:** Amigurumi in continuous rounds\n"
        f"- **Finished size:** {_estimate_size_from_rows(rows)}\n"
        f"- **Total rounds:** {len(rows)}\n"
        f"- **Difficulty:** {_difficulty_from_rows(len(rows))}"
    )
    out.append("")

    out.append("## Materials")
    out.append("")
    out.append("- **Yarn:** worsted-weight cotton or acrylic, 50-100 g per color.")
    out.append("- **Hook:** 2.5 - 3.5 mm (use a hook one size smaller than "
               "the yarn label recommends, to keep stitches tight).")
    out.append("- **Fiberfill stuffing** for a firm but not lumpy finish.")
    out.append("- **Notions:** stitch marker, tapestry needle, scissors, "
               "safety eyes / embroidery floss if the design needs features.")
    out.append("")

    out.append("## Gauge")
    out.append("")
    out.append("With the yarn and hook above, the slicer assumes roughly "
               "**2 stitches per cm** and a row height of **0.6 cm**. If "
               "your personal gauge differs, scale the finished piece by the "
               "ratio between your gauge and this one.")
    out.append("")

    out.append("## Abbreviations")
    out.append("")
    out.append(
        "| Abbreviation | Stitch |\n"
        "|:---:|:---|\n"
        "| sc | single crochet |\n"
        "| inc | 2 sc in the same stitch (increase) |\n"
        "| dec | invisible decrease over 2 stitches |\n"
        "| MR | magic ring |\n"
        "| st(s) | stitch(es) |\n"
        "| R | round |\n"
        "| BLO | back loops only |\n"
        "| FO | fasten off |"
    )
    out.append("")

    if palette:
        out.append("## Suggested Color Palette")
        out.append("")
        out.append("Extracted from your reference image - use these as yarn-"
                   "color suggestions. The largest share is usually the "
                   "main body color; smaller shares work well for details.")
        out.append("")
        out.append(palette_to_markdown(palette))
        out.append("")

    out.append("## Pattern")
    out.append("")
    out.append("Work in continuous rounds (do **not** join or turn) unless "
               "a round specifies otherwise. Move the stitch marker up "
               "every round to keep track of where each round starts.")
    out.append("")
    if rows:
        # Annotate the first row so the reader knows how to start the tube.
        first = rows[0]
        out.append(f"- **{first.text}** _(start by making a magic ring, "
                   "then work the stitches into it and pull the tail closed.)_")
        for r in rows[1:]:
            out.append(f"- **{r.text}**")
    else:
        out.append("_(no rows generated)_")
    out.append("")

    out.append("## Assembly & Finishing")
    out.append("")
    out.append("1. **Stuff firmly** as the piece narrows; use small amounts "
               "at a time so lumps don't form.")
    out.append("2. **Close** the final round by FO, leaving a long tail. "
               "Weave the tail through the front loops of the remaining "
               "stitches and pull tight to close the opening.")
    out.append("3. **Weave in all ends** on the inside of the piece.")
    out.append("4. **Block lightly** by steaming over a wet cloth to relax "
               "the stitches.")
    out.append("")

    out.append("## Notes")
    out.append("")
    out.append("- The pattern is auto-generated by slicing a 3D mesh into "
               "horizontal rounds. Increases/decreases are distributed "
               "evenly, which produces a smooth shape but may not match the "
               "detail of a hand-designed pattern.")
    out.append("- For multi-part shapes (limbs, ears, horns), work each "
               "piece separately and sew on during assembly.")
    out.append("- Swap yarn colors on any round to add stripes without "
               "changing the stitch count.")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 2D prompt enhancement
# ---------------------------------------------------------------------------

_2D_SCHEME_TEMPLATE: str = (
    "A high-resolution, technical crochet stitch diagram (scheme) on a clean "
    "white background with generous white margins on all sides. "
    "The ENTIRE pattern must fit comfortably within the frame — nothing "
    "cropped or cut off at the edges. "
    "The image features professional black line art showing "
    "intricate crochet symbols: chains (ovals), double crochets (T-shapes "
    "with slashes), and slip stitches (dots). The layout is a perfectly "
    "symmetrical {motif_description}, showing clear stitch intersections and "
    "structural details. Minimalist aesthetic, sharp vector-like lines, no "
    "text, no hands, no 3D yarn — only the 2D schematic symbols. "
    "Professional craft book style, 8k resolution, top-down flat lay view, "
    "square 1:1 aspect ratio image, "
    "centered composition with at least 10% padding around the pattern."
)

# Short prompts that are clearly just a motif name get the full treatment.
# Anything already longer than this word count is assumed to be a complete
# prompt the user crafted themselves, so we leave it alone.
_MAX_SHORT_WORDS = 12


def _enhance_2d_prompt(raw_prompt: str) -> str:
    """Expand a terse motif description into a full Gemini image prompt.

    If the user already wrote a detailed prompt (> ~12 words), return it
    unchanged.  Otherwise, slot the user's text into the professional
    crochet-scheme template so Gemini produces a clean diagram rather than
    a vague artistic interpretation.

    >>> _enhance_2d_prompt("mandala pattern")  # short → enhanced
    'A high-resolution, technical crochet stitch diagram ...'
    >>> _enhance_2d_prompt("A full prompt I wrote myself with many words and details")
    'A full prompt I wrote myself with many words and details'
    """
    stripped = raw_prompt.strip()
    if not stripped:
        return _2D_SCHEME_TEMPLATE.format(motif_description="circular mandala pattern")
    if len(stripped.split()) > _MAX_SHORT_WORDS:
        return stripped  # already detailed enough
    # Slot the user's short description in as the motif shape/layout.
    return _2D_SCHEME_TEMPLATE.format(motif_description=stripped)


# ---------------------------------------------------------------------------
# 2D path
# ---------------------------------------------------------------------------

def _ensure_image_for_2d(
    prompt: str,
    uploaded_image: Path | None,
    out_dir: Path,
    cb: Callbacks,
) -> tuple[Path | None, str | None]:
    if uploaded_image is not None:
        cb.log(f"using uploaded image: {uploaded_image.name}")
        return uploaded_image, None

    caps = capabilities()
    if not (caps["genai"] and caps["gemini_key"]):
        return None, (
            "No image was uploaded and Gemini is unavailable (missing "
            "`google-genai` or `GEMINI_API_KEY`)."
        )
    try:
        from .generation import generate_image
        out_path = out_dir / "generated_2d.png"
        enhanced = _enhance_2d_prompt(prompt)
        cb.phase("Generating scheme image with Gemini", 0.20)
        cb.log(f"enhanced prompt → {enhanced[:120]}…" if len(enhanced) > 120 else f"enhanced prompt → {enhanced}")
        generated = generate_image(prompt=enhanced, output_path=out_path)
        return Path(generated), None
    except Exception as exc:
        return None, f"Gemini image generation failed: {exc}"


def _run_2d_path(
    prompt: str,
    uploaded_image: Path | None,
    out_dir: Path,
    cb: Callbacks,
) -> dict[str, Any]:
    caps = capabilities()
    result: dict[str, Any] = {
        "path": "2d",
        "image_path": None,
        "figure_path": None,
        "tile_figure_path": None,
        "num_detections": 0,
        "counts": {},
        "instruction": "",
        "instruction_html": "",
        "detections": [],
        "palette": [],
    }
    out_dir.mkdir(parents=True, exist_ok=True)

    cb.phase("Preparing 2D input", 0.10)
    image_path, note = _ensure_image_for_2d(prompt, uploaded_image, out_dir, cb)
    if note:
        result["note"] = note
    if image_path is None:
        result["instruction"] = format_2d_instruction(prompt, {}, 0, palette=[])
        result["instruction_html"] = build_2d_html(
            prompt=prompt, counts={}, total=0, palette=[], note=note,
        )
        return result

    result["image_path"] = str(image_path)
    cb.image(image_path, label="Source scheme")

    cb.phase("Extracting color palette", 0.35)
    palette = extract_palette(image_path, n_colors=5)
    result["palette"] = [c.as_dict() for c in palette]
    cb.palette(palette)

    if not (caps["ultralytics"] and caps["yolo_weights"]):
        result["note"] = (
            (result.get("note", "") + " | ")
            + "YOLO skipped (ultralytics or weights missing)."
        ).lstrip(" |")
        result["instruction"] = format_2d_instruction(prompt, {}, 0, palette=palette)
        result["instruction_html"] = build_2d_html(
            prompt=prompt, counts={}, total=0, palette=palette,
            cover_image_path=image_path, note=result.get("note"),
        )
        return result

    cb.phase("Running YOLOv8n OBB inference on the scheme", 0.60)
    from .pipeline import classify_stitches

    figure_path = out_dir / "predictions.png"
    summary = classify_stitches(
        image_path=image_path,
        weights_path=str(DEFAULT_WEIGHTS),
        conf=DEFAULT_CONF,
        iou_threshold=DEFAULT_IOU,
        target_stitch_px=DEFAULT_TARGET_STITCH_PX,
        tile_size=DEFAULT_TILE_SIZE,
        overlap=DEFAULT_OVERLAP,
        output_figure=figure_path,
        tile_figure=out_dir / "tiles.png",
        visualize_tiles=False,
    )
    dets = summary.get("detections", [])
    counts = Counter(
        d.get("class_name") or d.get("name") or str(d.get("class_id"))
        for d in dets
    )
    counts_d = dict(counts)
    result["figure_path"] = str(summary.get("figure_path")) if summary.get("figure_path") else None
    result["scheme_path"] = str(summary.get("scheme_path")) if summary.get("scheme_path") else None
    result["num_detections"] = summary.get("num_detections", 0)
    result["counts"] = counts_d
    result["detections"] = dets
    if result["figure_path"]:
        cb.image(Path(result["figure_path"]), label="YOLO OBB predictions")
    if result["scheme_path"]:
        cb.image(Path(result["scheme_path"]), label="Reconstructed pattern")

    cb.phase("Formatting crochet instruction", 0.90)
    result["instruction"] = format_2d_instruction(
        prompt, counts_d, int(result["num_detections"]), palette=palette,
    )
    result["instruction_html"] = build_2d_html(
        prompt=prompt,
        counts=counts_d,
        total=int(result["num_detections"]),
        palette=palette,
        cover_image_path=image_path,
        yolo_figure_path=result.get("figure_path"),
        scheme_image_path=result.get("scheme_path"),
        note=result.get("note"),
    )
    cb.phase("Done", 1.0)
    return result


# ---------------------------------------------------------------------------
# 3D path
# ---------------------------------------------------------------------------

_HUNYUAN_CLIENT = None


def _hunyuan_client():
    global _HUNYUAN_CLIENT
    if _HUNYUAN_CLIENT is None:
        from gradio_client import Client
        _HUNYUAN_CLIENT = Client("tencent/Hunyuan3D-2")
    return _HUNYUAN_CLIENT


def _hunyuan_endpoints(client) -> list[tuple[str, int, list]]:
    try:
        info = client.view_api(return_format="dict", print_info=False)
    except TypeError:
        info = client.view_api(return_format="dict")
    named = info.get("named_endpoints", {}) if isinstance(info, dict) else {}
    out: list[tuple[str, int, list]] = []
    for name, meta in named.items():
        params = meta.get("parameters", []) if isinstance(meta, dict) else []
        out.append((name, len(params), params))
    return out


def _fill_hunyuan_args(params: list, prompt: str, image_path: str | None):
    import gradio_client
    args = []
    for p in params:
        label = (p.get("label") or p.get("parameter_name") or "").lower()
        ptype = ""
        if isinstance(p.get("python_type"), dict):
            ptype = p["python_type"].get("type", "")
        if any(k in label for k in ("image", "img", "input_image", "upload")):
            args.append(gradio_client.handle_file(image_path) if image_path else None)
        elif any(k in label for k in ("caption", "prompt", "text", "description")):
            args.append(prompt)
        elif "seed" in label:
            args.append(1234)
        elif "randomize" in label:
            args.append(True)
        elif "rembg" in label or "remove" in label or "background" in label:
            args.append(True)
        elif "step" in label:
            args.append(30)
        elif "guidance" in label or "cfg" in label:
            args.append(5.0)
        elif "resolution" in label or "octree" in label:
            args.append(256)
        elif "chunk" in label:
            args.append(8000)
        elif ptype == "bool":
            args.append(False)
        elif ptype in ("int", "float"):
            args.append(0)
        else:
            args.append(None)
    return args


def _find_mesh_path(obj):
    MESH_EXT = (".glb", ".obj", ".ply", ".stl")
    if isinstance(obj, str) and obj.lower().endswith(MESH_EXT):
        return obj
    if isinstance(obj, dict):
        for key in ("path", "name", "url", "file"):
            v = obj.get(key)
            if isinstance(v, str) and v.lower().endswith(MESH_EXT):
                return v
        for v in obj.values():
            got = _find_mesh_path(v)
            if got:
                return got
    if isinstance(obj, (list, tuple)):
        for v in obj:
            got = _find_mesh_path(v)
            if got:
                return got
    return None


# Cached .glb from a previous successful Hunyuan3D run.  Used as fallback
# when the remote GPU quota is exhausted or gradio_client is unavailable.
_FALLBACK_GLB = Path(
    os.environ.get(
        "CROCHET_FALLBACK_GLB",
        Path(__file__).resolve().parent.parent
        / "notebooks_output"
        / "create_instruction"
        / "3d"
        / "generated.glb",
    )
)


def _demo_mesh_for_prompt(prompt: str):
    import trimesh

    # Prefer a real mesh from a previous run
    if _FALLBACK_GLB.is_file():
        try:
            return trimesh.load(str(_FALLBACK_GLB), force="mesh")
        except Exception:
            pass  # fall through to primitive shapes

    p = (prompt or "").lower()
    if any(w in p for w in ("hat", "beanie", "cap")):
        return trimesh.creation.annulus(r_min=9.0, r_max=10.0, height=12.0, sections=64)
    if any(w in p for w in ("bag", "basket", "bowl")):
        return trimesh.creation.cylinder(radius=10.0, height=18.0, sections=64)
    if "scarf" in p or "blanket" in p:
        return trimesh.creation.box(extents=(50.0, 15.0, 0.5))
    return trimesh.creation.icosphere(subdivisions=4, radius=6.0)


def _ensure_image_for_3d(
    prompt: str,
    uploaded_image: Path | None,
    out_dir: Path,
    cb: Callbacks,
) -> Path | None:
    if uploaded_image is not None:
        cb.log(f"using uploaded image as 3D reference: {uploaded_image.name}")
        return uploaded_image
    caps = capabilities()
    if not (caps["genai"] and caps["gemini_key"]):
        return None
    try:
        from .generation import generate_image
        ref_prompt = (
            f"A single photograph of {prompt}. Full object in frame, centered, "
            "plain neutral background, soft studio lighting, no text, "
            "photorealistic, 3/4 angle showing depth."
        )
        cb.phase("Generating reference image with Gemini", 0.18)
        out_path = out_dir / "reference.png"
        return Path(generate_image(prompt=ref_prompt, output_path=out_path))
    except Exception as exc:
        cb.log(f"Gemini reference image failed: {exc}")
        return None


def _generate_mesh(
    prompt: str,
    uploaded_image: Path | None,
    out_dir: Path,
    cb: Callbacks,
) -> tuple[Any, str, Path | None]:
    import trimesh
    caps = capabilities()
    ref_image = _ensure_image_for_3d(prompt, uploaded_image, out_dir, cb)
    if ref_image is not None:
        cb.image(ref_image, label="Reference image for Hunyuan3D")
        palette = extract_palette(ref_image, n_colors=5)
        cb.palette(palette)

    if not caps["gradio_client"]:
        src = "fallback" if _FALLBACK_GLB.is_file() else "demo"
        cb.log(f"gradio_client missing — using {src} mesh")
        return _demo_mesh_for_prompt(prompt), src, ref_image

    cb.phase("Calling Hunyuan3D image-to-3D", 0.45)
    try:
        client = _hunyuan_client()
        endpoints = _hunyuan_endpoints(client)

        def _is_gen(name: str) -> bool:
            n = name.lower()
            return (
                "generation" in n or "shape" in n or "text_to" in n or "image_to" in n
            ) and ("lambda" not in n and "change" not in n and "export" not in n)

        gen_endpoints = [e for e in endpoints if _is_gen(e[0])]

        def _score(name: str) -> int:
            n = name.lower()
            if "generation_all" in n: return 0
            if "shape_generation" in n: return 1
            if "image_to" in n: return 2
            if "text_to" in n: return 3
            return 4

        gen_endpoints.sort(key=lambda e: _score(e[0]))

        image_str = str(ref_image) if ref_image else None
        last_error = None
        for api_name, _, params in gen_endpoints:
            args = _fill_hunyuan_args(params, prompt, image_str)
            cb.log(f"trying {api_name}...")
            try:
                res = client.predict(*args, api_name=api_name)
            except Exception as exc:
                last_error = f"{api_name} -> {type(exc).__name__}: {exc}"
                cb.log(f"  [skip] {last_error}")
                continue
            mesh_path = _find_mesh_path(res)
            if mesh_path:
                cb.log(f"  [ok] got mesh -> {mesh_path}")
                return trimesh.load(mesh_path, force="mesh"), f"hunyuan3d:{api_name}", ref_image
            cb.log(f"  [skip] {api_name} returned no mesh ({type(res).__name__})")
        raise RuntimeError(last_error or "no endpoint returned a mesh")
    except Exception as exc:
        src = "fallback" if _FALLBACK_GLB.is_file() else "demo"
        cb.log(f"Hunyuan3D failed ({exc}); using {src} mesh")
        return _demo_mesh_for_prompt(prompt), src, ref_image


_MAX_RENDER_FACES = 25_000  # target face count for preview renders
_VOXEL_PITCH = 0.05         # voxel size for remeshing (in mesh units)


def _simplify_for_render(mesh, target_faces: int, cb: Callbacks):
    """Return a lighter mesh for matplotlib preview rendering.

    Strategy (in priority order):
    1. Quadric decimation  – best quality, needs ``fast_simplification``.
    2. Voxel remesh         – good quality, needs ``scipy`` + ``scikit-image``.
    3. Return original      – last resort if the mesh is not too huge.
    """
    import trimesh

    if len(mesh.faces) <= target_faces:
        return mesh

    cb.log(
        f"mesh has {len(mesh.faces):,} faces; simplifying for preview renders"
    )

    # 1) Quadric decimation (best quality)
    try:
        simplified = mesh.simplify_quadric_decimation(target_faces)
        cb.log(f"  quadric decimation -> {len(simplified.faces):,} faces")
        return simplified
    except Exception:
        pass

    # 2) Voxel remesh via marching cubes (good quality, always-solid result)
    try:
        vox = mesh.voxelized(pitch=_VOXEL_PITCH)
        simplified = vox.marching_cubes
        cb.log(f"  voxel remesh (pitch={_VOXEL_PITCH}) -> {len(simplified.faces):,} faces")
        return simplified
    except Exception as exc:
        cb.log(f"  voxel remesh failed: {exc}")

    cb.log("  no simplification available; rendering full mesh (may be slow)")
    return mesh


def _multi_view_renders(mesh, out_dir: Path, cb: Callbacks, size_px: int = 512) -> list[Path]:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)

    render_mesh = _simplify_for_render(mesh, _MAX_RENDER_FACES, cb)

    views = (("front", (0, 0)), ("side", (0, 90)), ("top", (90, 0)))
    paths: list[Path] = []
    for name, (elev, azim) in views:
        fig = plt.figure(figsize=(size_px / 100, size_px / 100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(
            render_mesh.vertices[:, 0],
            render_mesh.vertices[:, 1],
            render_mesh.vertices[:, 2],
            triangles=render_mesh.faces,
            linewidth=0, color="#e8e0d0", shade=True,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_box_aspect((1, 1, 1))
        p = out_dir / f"{name}.png"
        fig.savefig(p, bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close(fig)
        paths.append(p)
        cb.image(p, label=f"3D view: {name}")
    return paths


# ---- amigurumi slicer ------------------------------------------------------

@dataclasses.dataclass
class AmiRow:
    idx: int
    stitches: int
    delta: int
    text: str


def _distribute_operations(prev_n: int, cur_n: int) -> str:
    delta = cur_n - prev_n
    if delta == 0:
        return f"{prev_n} sc = {cur_n}"
    if delta > 0:
        groups = delta
        if prev_n <= 0:
            return f"{cur_n} sc in magic ring = {cur_n}"
        g = prev_n // groups if groups else prev_n
        rem = prev_n - g * groups
        if rem == 0 and g >= 1:
            body = f"({g - 1} sc, inc)" if g > 1 else "inc"
            return f"{body} x{groups} = {cur_n}"
        return f"inc x{groups} spread evenly over {prev_n} st -> {cur_n}"
    dec = -delta
    if prev_n <= 0:
        return f"?{cur_n}"
    g = prev_n // dec if dec else prev_n
    rem = prev_n - g * dec
    if rem == 0 and g >= 2:
        body = f"({g - 2} sc, dec)" if g > 2 else "dec"
        return f"{body} x{dec} = {cur_n}"
    return f"dec x{dec} spread evenly over {prev_n} st -> {cur_n}"


def _mesh_to_amigurumi(
    mesh,
    stitches_per_cm: float = 2.0,
    stitch_height_cm: float = 0.6,
    magic_ring_start: int = 6,
    min_stitches: int = 6,
    max_stitches: int = 160,
) -> tuple[list[AmiRow], dict[str, Any]]:
    mesh = mesh.copy()
    extents = mesh.extents
    axis = int(np.argmax(extents))
    normal = np.zeros(3); normal[axis] = 1.0
    z_lo, z_hi = mesh.bounds[0, axis], mesh.bounds[1, axis]
    total_h_cm = float(z_hi - z_lo)
    heights = np.arange(z_lo + 1e-4, z_hi - 1e-4, stitch_height_cm)
    if len(heights) < 1:
        raise ValueError(
            f"Mesh height {total_h_cm:.2f} cm < stitch_height_cm {stitch_height_cm}"
        )
    origin = np.zeros(3); origin[axis] = z_lo
    relative = heights - z_lo
    sections = mesh.section_multiplane(
        plane_origin=origin, plane_normal=normal, heights=relative,
    )
    perimeters: list[float] = []
    for section in sections:
        if section is None:
            perimeters.append(0.0); continue
        try:
            polys = list(section.polygons_full) if section.polygons_full else []
            poly = max(polys, key=lambda p: p.length) if polys else None
            perim = float(poly.length) if poly is not None else float(section.length)
        except Exception:
            perim = float(section.length)
        perimeters.append(perim)
    rows: list[AmiRow] = [
        AmiRow(
            idx=1, stitches=magic_ring_start, delta=magic_ring_start,
            text=f"R1: {magic_ring_start} sc in magic ring = {magic_ring_start}",
        ),
    ]
    prev_n = magic_ring_start
    for i, perim in enumerate(perimeters, start=2):
        target = prev_n if perim <= 0 else max(
            min_stitches, min(max_stitches, int(round(perim * stitches_per_cm))),
        )
        rows.append(AmiRow(
            idx=i, stitches=target, delta=target - prev_n,
            text=f"R{i}: {_distribute_operations(prev_n, target)}",
        ))
        prev_n = target
    return rows, {
        "axis":              "xyz"[axis],
        "total_height_cm":   total_h_cm,
        "num_slices":        len(heights),
        "stitches_per_cm":   stitches_per_cm,
        "stitch_height_cm":  stitch_height_cm,
        "raw_perimeters_cm": perimeters,
    }


def _run_3d_path(
    prompt: str,
    uploaded_image: Path | None,
    out_dir: Path,
    cb: Callbacks,
) -> dict[str, Any]:
    caps = capabilities()
    result: dict[str, Any] = {
        "path": "3d",
        "reference_image_path": None,
        "mesh_source": None,
        "glb_path": None,
        "views": [],
        "view_analysis": [],
        "amigurumi_rows": [],
        "amigurumi_debug": {},
        "instruction": "",
        "instruction_html": "",
        "palette": [],
    }
    out_dir.mkdir(parents=True, exist_ok=True)

    if not caps["trimesh"]:
        result["note"] = (
            "3D path skipped: `trimesh` is not installed. Run "
            "`pip install 'trimesh[easy]' shapely` and try again."
        )
        result["instruction"] = (
            "# 3D pipeline unavailable\n\n"
            "The 3D pattern generator needs the `trimesh` library. "
            "Install it with `pip install 'trimesh[easy]' shapely` and "
            "try again to get a full amigurumi pattern."
        )
        result["instruction_html"] = build_3d_html(
            prompt=prompt, rows=[], palette=[],
            mesh_source="trimesh-missing",
        )
        return result

    cb.phase("Preparing 3D input", 0.10)
    mesh, source, ref_image = _generate_mesh(prompt, uploaded_image, out_dir, cb)
    result["mesh_source"] = source
    if ref_image is not None:
        result["reference_image_path"] = str(ref_image)
        palette = extract_palette(ref_image, n_colors=5)
        result["palette"] = [c.as_dict() for c in palette]
    else:
        palette = []

    glb_path = out_dir / "generated.glb"
    try:
        mesh.export(glb_path)
        result["glb_path"] = str(glb_path)
    except Exception as exc:
        cb.log(f"could not export .glb: {exc}")

    cb.phase("Rendering multi-view previews", 0.65)
    views = _multi_view_renders(mesh, out_dir / "views", cb)
    result["views"] = [str(v) for v in views]

    if caps["ultralytics"] and caps["yolo_weights"]:
        cb.phase("Analysing views with YOLO OBB", 0.78)
        try:
            from .pipeline import classify_stitches
            for p in views:
                summary = classify_stitches(
                    image_path=p,
                    weights_path=str(DEFAULT_WEIGHTS),
                    visualize_tiles=False,
                )
                result["view_analysis"].append({
                    "view": p.stem,
                    "num_detections": summary.get("num_detections", 0),
                    "figure_path": str(summary.get("figure_path", "")),
                })
        except Exception as exc:
            cb.log(f"view analysis failed: {exc}")

    cb.phase("Slicing mesh into amigurumi rows", 0.88)
    rows, debug = _mesh_to_amigurumi(mesh)
    result["amigurumi_rows"] = [dataclasses.asdict(r) for r in rows]
    result["amigurumi_debug"] = debug

    cb.phase("Formatting crochet instruction", 0.95)
    result["instruction"] = _format_3d_instruction(prompt, rows, palette=palette)
    result["instruction_html"] = build_3d_html(
        prompt=prompt,
        rows=rows,
        palette=palette,
        cover_image_path=ref_image,
        view_paths=views,
        mesh_source=source,
    )
    cb.phase("Done", 1.0)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RouteDecision:
    label: Literal["2d_scheme", "3d_object"]
    scores: dict[str, float]
    prompt: str
    has_image: bool


def route(prompt: str, image_path: Path | None) -> RouteDecision:
    prompt = (prompt or "").strip()
    if prompt:
        label, scores = classify_prompt(prompt)
    elif image_path is not None:
        label, scores = _classify_image_shape(image_path)
    else:
        label, scores = "2d_scheme", {"2d_scheme": 0.5, "3d_object": 0.5}
    return RouteDecision(
        label=label, scores=scores, prompt=prompt, has_image=image_path is not None,
    )


def run_router(
    prompt: str | None = None,
    image_path: str | Path | None = None,
    work_dir: str | Path | None = None,
    on_log: Callable[[str], None] | None = None,
    on_phase: Callable[[str, float], None] | None = None,
    on_image: Callable[[Path, str], None] | None = None,
    on_palette: Callable[[list[PaletteColor]], None] | None = None,
) -> dict[str, Any]:
    """Classify the input and run the matching path end-to-end.

    Callbacks (all optional) let a UI stream in intermediate artefacts:

    * ``on_phase(name, pct)`` - phase started (``pct`` in ``[0, 1]``).
    * ``on_image(path, label)`` - new image file ready to display.
    * ``on_palette(colors)`` - dominant color palette extracted.
    * ``on_log(msg)`` - per-step diagnostic messages.
    """
    cb = Callbacks(
        on_phase=on_phase, on_image=on_image,
        on_palette=on_palette, on_log=on_log,
    )
    prompt = (prompt or "").strip()
    if not prompt and image_path is None:
        raise ValueError("Provide at least a prompt or an image.")

    img_path: Path | None = Path(image_path) if image_path else None
    if img_path is not None and not img_path.is_file():
        raise FileNotFoundError(f"Uploaded image not found: {img_path}")

    base = Path(work_dir) if work_dir else Path.cwd() / "routing_out"
    base.mkdir(parents=True, exist_ok=True)

    cb.phase("Classifying intent", 0.05)
    decision = route(prompt, img_path)
    cb.log(f"route={decision.label} (scores={decision.scores})")

    if img_path is not None:
        cb.image(img_path, label="Uploaded image")

    if decision.label == "3d_object":
        result = _run_3d_path(prompt, img_path, base / "3d", cb)
    else:
        result = _run_2d_path(prompt, img_path, base / "2d", cb)

    result["prompt"] = prompt
    result["scores"] = decision.scores
    result["label"] = decision.label
    result["has_image"] = decision.has_image
    if img_path is not None:
        result.setdefault("uploaded_image_path", str(img_path))
    return result
