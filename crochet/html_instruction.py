"""Render router output as a professionally-styled HTML document.

The visual language matches a Keito-Dama-style pattern book: Cormorant
Garamond serif titles, Jost body type, a dark cover, numbered sections in
large serif numerals, cream-and-linen cards, dark pattern-row code blocks,
soft-rose notes, and a checklist-style finishing section.

Two public entry points:

* ``build_2d_html(ctx)`` - for the 2D / chart pipeline.
* ``build_3d_html(ctx)`` - for the 3D / amigurumi pipeline.

Both return a full standalone HTML string. The Streamlit page embeds this
in an ``st.components.v1.html`` iframe and offers the same string as a
``.html`` download so the user can open it in a browser or print to PDF.
"""

from __future__ import annotations

import base64
import colorsys
import html as _html
import mimetypes
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from .palette import PaletteColor


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

BASE_THEME: dict[str, str] = {
    "cream":      "#f5f0e8",
    "warm_white": "#faf8f4",
    "linen":      "#e8e0d0",
    "taupe":      "#b8a898",
    "deep":       "#3a3228",
    "accent":     "#8b7355",
    "soft_rose":  "#d4c4b0",
}


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hsv(hex_color: str) -> tuple[float, float, float]:
    r, g, b = _hex_to_rgb(hex_color)
    return colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)


def theme_from_palette(palette: Iterable[PaletteColor] | None) -> dict[str, str]:
    """Return a CSS variable dict derived from the user's palette.

    The base cream/taupe/deep aesthetic is kept; we only swap in the
    palette's most saturated mid-value hue as the ``soft_rose`` accent so
    the document feels tied to the image without turning chaotic.
    """
    theme = dict(BASE_THEME)
    if not palette:
        return theme
    best_hex: str | None = None
    best_sat = 0.0
    for p in palette:
        _, s, v = _rgb_to_hsv(p.hex)
        if 0.18 <= v <= 0.88 and s >= 0.15 and s > best_sat:
            best_sat = s
            best_hex = p.hex
    if best_hex:
        theme["soft_rose"] = best_hex
    return theme


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _esc(text: Any) -> str:
    return _html.escape("" if text is None else str(text))


def _data_uri(path: str | Path | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    mime = mimetypes.guess_type(str(p))[0] or "image/png"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


# ---------------------------------------------------------------------------
# CSS (shared across both pipelines)
# ---------------------------------------------------------------------------

def _stylesheet(theme: dict[str, str]) -> str:
    return f"""
:root {{
  --cream:      {theme['cream']};
  --warm-white: {theme['warm_white']};
  --linen:      {theme['linen']};
  --taupe:      {theme['taupe']};
  --deep:       {theme['deep']};
  --accent:     {theme['accent']};
  --soft-rose:  {theme['soft_rose']};
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

html, body {{ background: var(--warm-white); }}
body {{
  color: var(--deep);
  font-family: 'Jost', 'Helvetica Neue', Arial, sans-serif;
  font-weight: 300;
  line-height: 1.7;
}}

/* Cover */
.cover {{
  min-height: 520px;
  background: var(--deep);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 56px 40px;
  position: relative;
  overflow: hidden;
}}
.cover::before {{
  content: '';
  position: absolute;
  inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' stroke='%23ffffff0d' stroke-width='1'%3E%3Ccircle cx='30' cy='30' r='20'/%3E%3Ccircle cx='30' cy='30' r='10'/%3E%3Cline x1='10' y1='30' x2='50' y2='30'/%3E%3Cline x1='30' y1='10' x2='30' y2='50'/%3E%3C/g%3E%3C/svg%3E");
  opacity: 0.45;
}}
.cover-lace, .cover-lace-bottom {{
  position: absolute; left: 0; right: 0;
  width: 100%; height: 80px;
  opacity: 0.18;
  display: block;
}}
.cover-lace {{ top: 0; }}
.cover-lace-bottom {{ bottom: 0; transform: rotate(180deg); }}
.cover-illu {{
  position: relative; z-index: 1;
  margin-bottom: 22px;
  padding: 6px;
  max-width: 220px;
  max-height: 220px;
  opacity: 0.92;
  border-radius: 6px;
}}
.cover-illu.has-photo {{
  background: rgba(245, 240, 232, 0.08);
  border: 1px solid rgba(245, 240, 232, 0.18);
}}
.cover h1 {{
  font-family: 'Cormorant Garamond', 'Cormorant', 'Georgia', serif;
  font-size: clamp(2.4rem, 6vw, 4.4rem);
  font-weight: 300;
  color: var(--cream);
  letter-spacing: 0.04em;
  line-height: 1.12;
  position: relative; z-index: 1;
}}
.cover h1 em {{ font-style: italic; color: var(--soft-rose); }}
.cover-sub {{
  font-size: 0.82rem;
  color: var(--taupe);
  letter-spacing: 0.3em;
  text-transform: uppercase;
  margin-top: 22px;
  position: relative; z-index: 1;
}}
.cover-divider {{
  width: 60px;
  height: 1px;
  background: var(--taupe);
  margin: 26px auto;
  position: relative; z-index: 1;
}}
.cover-meta {{
  font-size: 0.78rem;
  color: var(--taupe);
  letter-spacing: 0.2em;
  text-transform: uppercase;
  position: relative; z-index: 1;
}}

/* Layout */
.container {{ max-width: 860px; margin: 0 auto; padding: 0 40px; }}

.section {{
  padding: 72px 0;
  border-bottom: 1px solid var(--linen);
}}
.section:last-child {{ border-bottom: none; }}
.section-number {{
  font-family: 'Cormorant Garamond', 'Georgia', serif;
  font-size: 5rem;
  font-weight: 300;
  color: var(--linen);
  line-height: 1;
  margin-bottom: -18px;
}}
.section-title {{
  font-family: 'Cormorant Garamond', 'Georgia', serif;
  font-size: 2.1rem;
  font-weight: 400;
  color: var(--deep);
  margin-bottom: 6px;
}}
.section-subtitle {{
  font-size: 0.78rem;
  letter-spacing: 0.25em;
  text-transform: uppercase;
  color: var(--taupe);
  margin-bottom: 36px;
}}
p {{ margin-bottom: 14px; font-size: 0.95rem; }}
strong {{ font-weight: 500; }}

/* Sub-heading inside a section */
.sub-heading {{
  font-family: 'Cormorant Garamond', 'Georgia', serif;
  font-size: 1.35rem;
  font-weight: 400;
  margin: 28px 0 10px 0;
  color: var(--deep);
}}

/* Materials grid */
.materials-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
  margin-top: 28px;
}}
.material-card {{
  background: var(--cream);
  padding: 22px 24px;
  border-left: 3px solid var(--accent);
}}
.material-card h4 {{
  font-family: 'Cormorant Garamond', 'Georgia', serif;
  font-size: 1.1rem;
  font-weight: 500;
  margin-bottom: 6px;
  color: var(--accent);
}}
.material-card p {{ font-size: 0.88rem; margin: 0; line-height: 1.6; }}

/* Stitch demo */
.stitch-demo {{
  background: var(--cream);
  padding: 32px 40px;
  margin: 28px 0;
  text-align: center;
}}
.stitch-demo svg {{ max-width: 100%; height: auto; }}
.stitch-caption {{
  font-size: 0.78rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--taupe);
  margin-top: 14px;
}}

/* Steps */
.step {{
  display: flex;
  gap: 28px;
  margin-bottom: 42px;
  align-items: flex-start;
}}
.step-num {{
  font-family: 'Cormorant Garamond', 'Georgia', serif;
  font-size: 3rem;
  font-weight: 300;
  color: var(--linen);
  line-height: 1;
  flex-shrink: 0;
  width: 60px;
  text-align: right;
}}
.step-content h3 {{
  font-family: 'Cormorant Garamond', 'Georgia', serif;
  font-size: 1.35rem;
  font-weight: 400;
  margin-bottom: 6px;
}}
.step-content p {{ font-size: 0.92rem; margin-bottom: 10px; }}

/* Pieces table */
.pieces-table {{
  width: 100%;
  border-collapse: collapse;
  margin: 22px 0;
  font-size: 0.88rem;
}}
.pieces-table th {{
  background: var(--deep);
  color: var(--cream);
  padding: 11px 14px;
  text-align: left;
  font-weight: 400;
  letter-spacing: 0.1em;
  font-size: 0.8rem;
}}
.pieces-table td {{
  padding: 11px 14px;
  border-bottom: 1px solid var(--linen);
}}
.pieces-table tr:nth-child(even) td {{ background: var(--cream); }}

/* Note box */
.note {{
  border: 1px solid var(--soft-rose);
  padding: 18px 22px;
  margin: 22px 0;
  font-size: 0.88rem;
  background: #fdf9f5;
}}
.note strong {{
  color: var(--accent);
  display: block;
  margin-bottom: 6px;
  font-size: 0.78rem;
  letter-spacing: 0.15em;
  text-transform: uppercase;
}}

/* Pattern row / code block */
.pattern-row {{
  background: var(--deep);
  color: var(--cream);
  padding: 18px 22px;
  margin: 14px 0;
  font-family: 'Courier New', monospace;
  font-size: 0.85rem;
  line-height: 1.85;
}}
.pattern-row .row-label {{
  color: var(--soft-rose);
  font-weight: bold;
}}

/* Assembly diagram */
.assembly-diagram {{
  background: var(--cream);
  padding: 32px;
  margin: 28px 0;
  text-align: center;
}}
.assembly-diagram svg {{ max-width: 100%; height: auto; }}

/* Checklist */
.checklist {{
  list-style: none;
  margin: 18px 0;
  padding: 0;
}}
.checklist li {{
  padding: 9px 0;
  border-bottom: 1px solid var(--linen);
  font-size: 0.92rem;
  display: flex;
  align-items: flex-start;
  gap: 12px;
}}
.checklist li::before {{
  content: '\\25C7';
  color: var(--taupe);
  flex-shrink: 0;
  margin-top: 2px;
}}

/* Yarn palette strip */
.yarn-palette {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 14px;
  margin-top: 26px;
}}
.yarn-card {{
  background: var(--cream);
  border-left: 3px solid var(--accent);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}}
.yarn-swatch {{ height: 90px; width: 100%; }}
.yarn-meta {{
  padding: 12px 16px 14px 16px;
  font-size: 0.84rem;
  line-height: 1.45;
}}
.yarn-meta .yarn-name {{
  font-family: 'Cormorant Garamond', 'Georgia', serif;
  font-size: 1.05rem;
  font-weight: 500;
  color: var(--deep);
  text-transform: capitalize;
  display: block;
  margin-bottom: 2px;
}}
.yarn-meta .yarn-hex {{
  font-family: 'Courier New', monospace;
  font-size: 0.78rem;
  color: var(--accent);
  letter-spacing: 0.05em;
}}
.yarn-meta .yarn-share {{
  font-size: 0.76rem;
  color: var(--taupe);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-top: 6px;
}}

/* Stitch inventory chips */
.stitch-chips {{
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 18px 0 8px 0;
}}
.stitch-chips .chip {{
  background: var(--cream);
  border-left: 3px solid var(--accent);
  padding: 8px 14px;
  font-size: 0.85rem;
  display: inline-flex;
  gap: 8px;
  align-items: baseline;
}}
.stitch-chips .chip .chip-count {{
  font-family: 'Cormorant Garamond', 'Georgia', serif;
  font-size: 1.1rem;
  color: var(--accent);
  font-weight: 500;
}}

/* Footer */
.foot {{
  background: var(--deep);
  padding: 44px 40px;
  text-align: center;
  color: var(--taupe);
}}
.foot-mark {{
  display: inline-block;
  margin-bottom: 12px;
  opacity: 0.5;
}}
.foot p {{
  font-size: 0.8rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  margin: 0;
}}
.foot p + p {{
  color: #5a5048;
  font-size: 0.72rem;
  letter-spacing: 0.06em;
  text-transform: none;
  margin-top: 6px;
}}

/* Reconstructed pattern hero image */
.pattern-figures {{
  margin: 36px 0;
}}
.pattern-fig-hero {{
  background: var(--cream);
  padding: 16px;
  margin-top: 18px;
  text-align: center;
  position: relative;
}}
.pattern-fig-hero img {{
  width: 100%;
  max-width: 100%;
  height: auto;
  display: block;
}}
.open-full-btn {{
  display: inline-block;
  margin-top: 14px;
  padding: 10px 28px;
  background: var(--deep);
  color: var(--cream);
  font-family: 'Jost', 'Helvetica Neue', Arial, sans-serif;
  font-size: 0.82rem;
  font-weight: 400;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  border: none;
  cursor: pointer;
  transition: background 0.2s, transform 0.15s;
}}
.open-full-btn:hover {{
  background: var(--accent);
  transform: translateY(-1px);
}}

/* Print */
@media print {{
  .cover {{ min-height: auto; padding: 60px 40px; page-break-after: always; }}
  .section {{ page-break-inside: avoid; }}
  .pattern-fig-hero img {{ max-height: 480px; object-fit: contain; }}
}}

@media (max-width: 640px) {{
  .container {{ padding: 0 20px; }}
  .materials-grid {{ grid-template-columns: 1fr; }}
  .step {{ flex-direction: column; gap: 8px; }}
  .step-num {{ width: auto; text-align: left; }}
  .pattern-fig-hero {{ padding: 8px; }}
}}
"""


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

COVER_LACE_SVG = """\
<svg class="{cls}" viewBox="0 0 1200 80" preserveAspectRatio="none"
     width="100%" height="100%"
     xmlns="http://www.w3.org/2000/svg">
  <g fill="white">
    <path d="M0,40 Q30,10 60,40 Q90,70 120,40 Q150,10 180,40 Q210,70 240,40
             Q270,10 300,40 Q330,70 360,40 Q390,10 420,40 Q450,70 480,40
             Q510,10 540,40 Q570,70 600,40 Q630,10 660,40 Q690,70 720,40
             Q750,10 780,40 Q810,70 840,40 Q870,10 900,40 Q930,70 960,40
             Q990,10 1020,40 Q1050,70 1080,40 Q1110,10 1140,40 Q1170,70 1200,40
             L1200,0 L0,0 Z"/>
  </g>
</svg>"""


DEFAULT_COVER_SILHOUETTE = """\
<svg width="180" height="180" viewBox="0 0 180 180" xmlns="http://www.w3.org/2000/svg">
  <g fill="none" stroke="#d4c4b0" stroke-width="1.4">
    <circle cx="90" cy="90" r="70"/>
    <circle cx="90" cy="90" r="50"/>
    <circle cx="90" cy="90" r="30"/>
    <line x1="20" y1="90" x2="160" y2="90"/>
    <line x1="90" y1="20" x2="90" y2="160"/>
    <line x1="36" y1="36" x2="144" y2="144"/>
    <line x1="144" y1="36" x2="36" y2="144"/>
  </g>
  <g fill="#b8a898" opacity="0.85">
    <circle cx="90" cy="90" r="6"/>
  </g>
</svg>"""


FAN_STITCH_SVG = """\
<svg width="560" height="180" viewBox="0 0 560 180" xmlns="http://www.w3.org/2000/svg">
  <rect width="560" height="180" fill="var(--cream, #f5f0e8)"/>
  <g fill="#3a3228">
    <circle cx="80"  cy="150" r="3"/><circle cx="100" cy="150" r="3"/>
    <circle cx="120" cy="150" r="3"/><circle cx="140" cy="150" r="3"/>
    <circle cx="160" cy="150" r="3"/><circle cx="180" cy="150" r="3"/>
    <circle cx="200" cy="150" r="3"/><circle cx="220" cy="150" r="3"/>
    <circle cx="240" cy="150" r="3"/><circle cx="260" cy="150" r="3"/>
    <circle cx="280" cy="150" r="3"/><circle cx="300" cy="150" r="3"/>
    <circle cx="320" cy="150" r="3"/><circle cx="340" cy="150" r="3"/>
    <circle cx="360" cy="150" r="3"/><circle cx="380" cy="150" r="3"/>
    <circle cx="400" cy="150" r="3"/><circle cx="420" cy="150" r="3"/>
    <circle cx="440" cy="150" r="3"/><circle cx="460" cy="150" r="3"/>
  </g>
  <g stroke="#3a3228" stroke-width="1.6" fill="none">
    <path d="M120,148 L100,110 M120,148 L110,106 M120,148 L120,104
             M120,148 L130,106 M120,148 L140,110"/>
    <path d="M100,110 Q120,100 140,110" stroke-width="1.1"/>
    <circle cx="160" cy="128" r="4"/>
    <path d="M200,148 L180,110 M200,148 L190,106 M200,148 L200,104
             M200,148 L210,106 M200,148 L220,110"/>
    <path d="M180,110 Q200,100 220,110" stroke-width="1.1"/>
    <circle cx="240" cy="128" r="4"/>
    <path d="M280,148 L260,110 M280,148 L270,106 M280,148 L280,104
             M280,148 L290,106 M280,148 L300,110"/>
    <path d="M260,110 Q280,100 300,110" stroke-width="1.1"/>
    <circle cx="320" cy="128" r="4"/>
    <path d="M360,148 L340,110 M360,148 L350,106 M360,148 L360,104
             M360,148 L370,106 M360,148 L380,110"/>
    <path d="M340,110 Q360,100 380,110" stroke-width="1.1"/>
    <circle cx="400" cy="128" r="4"/>
    <path d="M440,148 L420,110 M440,148 L430,106 M440,148 L440,104
             M440,148 L450,106 M440,148 L460,110"/>
    <path d="M420,110 Q440,100 460,110" stroke-width="1.1"/>
  </g>
  <g stroke="#8b7355" stroke-width="1.5" fill="none" opacity="0.8">
    <path d="M160,125 L140,80  M160,125 L150,76  M160,125 L160,75
             M160,125 L170,76  M160,125 L180,80"/>
    <path d="M140,80 Q160,72 180,80" stroke-width="1.1"/>
    <circle cx="200" cy="95" r="4"/>
    <path d="M240,125 L220,80  M240,125 L230,76  M240,125 L240,75
             M240,125 L250,76  M240,125 L260,80"/>
    <path d="M220,80 Q240,72 260,80" stroke-width="1.1"/>
    <circle cx="280" cy="95" r="4"/>
    <path d="M320,125 L300,80  M320,125 L310,76  M320,125 L320,75
             M320,125 L330,76  M320,125 L340,80"/>
    <path d="M300,80 Q320,72 340,80" stroke-width="1.1"/>
    <circle cx="360" cy="95" r="4"/>
    <path d="M400,125 L380,80  M400,125 L390,76  M400,125 L400,75
             M400,125 L410,76  M400,125 L420,80"/>
    <path d="M380,80 Q400,72 420,80" stroke-width="1.1"/>
  </g>
  <text x="20" y="20"  font-family="Jost" font-size="11" fill="#b8a898">Row 4</text>
  <text x="20" y="66"  font-family="Jost" font-size="11" fill="#b8a898">Row 3</text>
  <text x="20" y="112" font-family="Jost" font-size="11" fill="#b8a898">Row 2</text>
  <text x="20" y="158" font-family="Jost" font-size="11" fill="#b8a898">Row 1</text>
</svg>"""


SC_STITCH_SVG = """\
<svg width="560" height="150" viewBox="0 0 560 150" xmlns="http://www.w3.org/2000/svg">
  <rect width="560" height="150" fill="var(--cream, #f5f0e8)"/>
  <g fill="none" stroke="#3a3228" stroke-width="1.6">
    <path d="M60,100 q0,-26 22,-26 q22,0 22,26"/>
    <line x1="82" y1="74" x2="82" y2="116"/>
  </g>
  <text x="58" y="134" font-family="Jost" font-size="10" fill="#b8a898">sc</text>
  <g fill="none" stroke="#3a3228" stroke-width="1.6">
    <path d="M160,100 q0,-32 24,-32 q24,0 24,32"/>
    <line x1="184" y1="68" x2="184" y2="118"/>
    <line x1="174" y1="86" x2="194" y2="86"/>
  </g>
  <text x="158" y="134" font-family="Jost" font-size="10" fill="#b8a898">hdc</text>
  <g fill="none" stroke="#3a3228" stroke-width="1.6">
    <path d="M266,100 q0,-40 28,-40 q28,0 28,40"/>
    <line x1="294" y1="60" x2="294" y2="120"/>
    <line x1="284" y1="82" x2="304" y2="82"/>
  </g>
  <text x="264" y="134" font-family="Jost" font-size="10" fill="#b8a898">dc</text>
  <g fill="none" stroke="#8b7355" stroke-width="1.6">
    <path d="M370,100 q0,-28 18,-28 q18,0 18,28"/>
    <path d="M406,100 q0,-28 18,-28 q18,0 18,28"/>
    <line x1="388" y1="72" x2="388" y2="118"/>
    <line x1="424" y1="72" x2="424" y2="118"/>
    <path d="M360,100 q46,10 92,0" stroke="#8b7355"/>
  </g>
  <text x="380" y="134" font-family="Jost" font-size="10" fill="#b8a898">inc (2 sc in 1 st)</text>
  <g fill="none" stroke="#d4c4b0" stroke-width="1.6">
    <path d="M478,100 q0,-26 14,-26 q14,0 14,26"/>
    <path d="M506,100 q0,-26 14,-26 q14,0 14,26"/>
    <line x1="492" y1="74" x2="520" y2="74"/>
    <line x1="506" y1="74" x2="506" y2="118"/>
  </g>
  <text x="482" y="134" font-family="Jost" font-size="10" fill="#b8a898">dec (2 tog)</text>
</svg>"""


def _section(number: int, title: str, subtitle: str, content: str) -> str:
    return f"""\
<div class="section">
  <div class="container">
    <div class="section-number">{number:02d}</div>
    <h2 class="section-title">{_esc(title)}</h2>
    <p class="section-subtitle">{_esc(subtitle)}</p>
    {content}
  </div>
</div>"""


def _material_card(title: str, body_html: str) -> str:
    return (
        f"<div class='material-card'><h4>{_esc(title)}</h4>"
        f"<p>{body_html}</p></div>"
    )


def _materials_grid(cards: list[tuple[str, str]]) -> str:
    inner = "".join(_material_card(t, b) for t, b in cards)
    return f"<div class='materials-grid'>{inner}</div>"


def _note(title: str, body_html: str) -> str:
    return (
        f"<div class='note'><strong>{_esc(title)}</strong>{body_html}</div>"
    )


def _stitch_demo(svg: str, caption: str) -> str:
    return (
        f"<div class='stitch-demo'>{svg}"
        f"<p class='stitch-caption'>{_esc(caption)}</p></div>"
    )


def _pattern_row_block(lines: list[tuple[str, str]]) -> str:
    body = "<br>".join(
        f"<span class='row-label'>{_esc(label)}</span> {_esc(text)}"
        for label, text in lines
    )
    return f"<div class='pattern-row'>{body}</div>"


def _step(number: int, title: str, paragraphs: list[str]) -> str:
    body = "".join(f"<p>{_esc(p)}</p>" for p in paragraphs)
    return (
        f"<div class='step'><div class='step-num'>{number}</div>"
        f"<div class='step-content'><h3>{_esc(title)}</h3>{body}</div></div>"
    )


def _pieces_table(columns: list[str], rows: list[list[str]]) -> str:
    thead = "<tr>" + "".join(f"<th>{_esc(c)}</th>" for c in columns) + "</tr>"
    tbody = "".join(
        "<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in r) + "</tr>" for r in rows
    )
    return (
        f"<table class='pieces-table'>"
        f"<thead>{thead}</thead><tbody>{tbody}</tbody></table>"
    )


def _checklist(items: list[str]) -> str:
    lis = "".join(f"<li>{_esc(i)}</li>" for i in items)
    return f"<ul class='checklist'>{lis}</ul>"


def _yarn_palette(palette: list[PaletteColor]) -> str:
    if not palette:
        return "<p>No palette extracted - pick any yarn colors you love.</p>"
    cards = []
    for p in palette:
        cards.append(
            "<div class='yarn-card'>"
            f"<div class='yarn-swatch' style='background:{_esc(p.hex)};'></div>"
            "<div class='yarn-meta'>"
            f"<span class='yarn-name'>{_esc(p.name)}</span>"
            f"<span class='yarn-hex'>{_esc(p.hex)}</span>"
            f"<div class='yarn-share'>{p.weight:.0%} of image</div>"
            "</div></div>"
        )
    return f"<div class='yarn-palette'>{''.join(cards)}</div>"


def _pattern_figures(
    scheme_uri: str | None,
) -> str:
    """Render the reconstructed scheme as a large, prominent image."""
    if not scheme_uri:
        return ""
    return (
        "<div class='pattern-figures'>"
        "<h3 class='sub-heading'>Reconstructed Pattern</h3>"
        "<p>The stitch symbols detected on your chart, redrawn as clean icons "
        "at their original positions. Use this as a reading aid alongside "
        "the original chart.</p>"
        "<div class='pattern-fig-hero'>"
        f"<img id='scheme-img' src='{scheme_uri}' alt='Reconstructed crochet pattern'>"
        "<button class='open-full-btn' onclick=\""
        "var w=window.open('','_blank','');"
        "w.document.write("
        "'<!DOCTYPE html><html><head><title>Reconstructed Pattern</title>"
        "<style>*{margin:0;padding:0;background:#f5f0e8}"
        "img{display:block;max-width:100%;height:auto;margin:0 auto}</style>"
        "</head><body><img src=\\'' + document.getElementById(\\'scheme-img\\').src + '\\'></body></html>');"
        "w.document.close();"
        "\">Open Full Size</button>"
        "</div>"
        "</div>"
    )


def _stitch_chips(counts: dict[str, int]) -> str:
    if not counts:
        return ""
    chips = []
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        pretty = name.replace("_", " ").title()
        chips.append(
            "<span class='chip'>"
            f"<span class='chip-count'>{n}</span>"
            f"<span>{_esc(pretty)}</span>"
            "</span>"
        )
    return f"<div class='stitch-chips'>{''.join(chips)}</div>"


def _assembly_flowchart_svg(steps: list[str]) -> str:
    width = max(520, 120 * len(steps) + 20)
    boxes: list[str] = []
    arrows: list[str] = []
    for i, label in enumerate(steps):
        x = 20 + i * 120
        fill = "#3a3228" if i < len(steps) - 1 else "#8b7355"
        label_safe = _esc(label).replace("\n", " ")
        boxes.append(
            f"<rect x='{x}' y='80' width='100' height='54' rx='3' fill='{fill}'/>"
            f"<text x='{x + 50}' y='112' text-anchor='middle' "
            f"fill='#f5f0e8' font-family='Jost' font-size='11'>{label_safe}</text>"
        )
        if i < len(steps) - 1:
            arrows.append(
                f"<line x1='{x + 100}' y1='107' x2='{x + 120}' y2='107' "
                f"stroke='#b8a898' stroke-width='1.5' marker-end='url(#arr)'/>"
            )
    return f"""\
<div class="assembly-diagram">
  <svg width="{width}" height="220" viewBox="0 0 {width} 220"
       xmlns="http://www.w3.org/2000/svg">
    <rect width="{width}" height="220" fill="var(--cream, #f5f0e8)"/>
    <defs>
      <marker id="arr" markerWidth="6" markerHeight="6" refX="5" refY="3"
              orient="auto">
        <path d="M0,0 L6,3 L0,6" fill="#b8a898"/>
      </marker>
    </defs>
    {''.join(boxes)}
    {''.join(arrows)}
    <text x="20" y="180" font-family="Cormorant Garamond" font-size="12"
          font-style="italic" fill="#3a3228">Seam method:</text>
    <text x="20" y="200" font-family="Jost" font-size="11" fill="#3a3228">
      Slip-stitch joins are near-invisible; single-crochet joins leave a
      decorative ridge.
    </text>
  </svg>
</div>"""


# ---------------------------------------------------------------------------
# Document shell
# ---------------------------------------------------------------------------

@dataclass
class Document:
    title: str
    subtitle: str
    meta: str
    sections: list[str]
    palette: list[PaletteColor] = field(default_factory=list)
    cover_svg: str = DEFAULT_COVER_SILHOUETTE
    cover_image_uri: str | None = None
    footer_label: str = "Crochet Instruction"
    footer_note: str = "Auto-generated pattern - verify gauge before starting"


def render_document(doc: Document) -> str:
    theme = theme_from_palette(doc.palette)
    cover_inner: str
    if doc.cover_image_uri:
        cover_inner = (
            f"<img class='cover-illu has-photo' src='{doc.cover_image_uri}' "
            f"alt='Reference image'>"
        )
    else:
        cover_inner = f"<div class='cover-illu'>{doc.cover_svg}</div>"

    return f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<meta name='viewport' content='width=device-width, initial-scale=1.0'>
<title>{_esc(doc.title)} - Crochet Tutorial</title>
<link href='https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;1,300;1,400&family=Jost:wght@300;400;500&display=swap' rel='stylesheet'>
<style>{_stylesheet(theme)}</style>
</head>
<body>
<div class='cover'>
  {COVER_LACE_SVG.format(cls='cover-lace')}
  {cover_inner}
  <h1>{_esc(doc.title)}</h1>
  <div class='cover-divider'></div>
  <p class='cover-sub'>{_esc(doc.subtitle)}</p>
  <p class='cover-meta' style='margin-top:14px;'>{_esc(doc.meta)}</p>
  {COVER_LACE_SVG.format(cls='cover-lace-bottom')}
</div>

{''.join(doc.sections)}

<div class='foot'>
  <svg class='foot-mark' width='36' height='36' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'>
    <path d='M20,5 L20,35 M5,20 L35,20 M8,8 L32,32 M32,8 L8,32'
          stroke='white' stroke-width='1' fill='none'/>
    <circle cx='20' cy='20' r='12' stroke='white' stroke-width='1' fill='none'/>
  </svg>
  <p>{_esc(doc.footer_label)}</p>
  <p>{_esc(doc.footer_note)}</p>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# 2D pipeline: chart -> professional HTML
# ---------------------------------------------------------------------------

def _difficulty_from_counts(total: int) -> str:
    if total < 30:   return "Beginner"
    if total < 120:  return "Easy"
    if total < 300:  return "Intermediate"
    return "Advanced"


def build_2d_html(
    *,
    prompt: str,
    counts: dict[str, int],
    total: int,
    palette: list[PaletteColor] | None = None,
    cover_image_path: str | Path | None = None,
    yolo_figure_path: str | Path | None = None,
    scheme_image_path: str | Path | None = None,
    note: str | None = None,
) -> str:
    title = (prompt or "Crochet Chart").strip().rstrip(".!?") or "Crochet Chart"
    title = title[0].upper() + title[1:]
    palette = palette or []
    difficulty = _difficulty_from_counts(total)
    unique = len(counts)

    # Pre-encode reconstructed scheme as data URI for embedding.
    scheme_uri = _data_uri(scheme_image_path)

    # ---- 01 Materials & Gauge -------------------------------------------
    mat_cards = [
        ("Yarn",
         "DK or worsted-weight cotton or cotton blend, approx. "
         "<strong>100&nbsp;g</strong>. Choose a smooth yarn so each stitch "
         "symbol reads clearly in the finished piece."),
        ("Hook",
         "Size <strong>3.5&nbsp;mm - 5.0&nbsp;mm</strong> depending on your "
         "preferred drape. Start with the size on the yarn band and adjust "
         "down for a tighter, more graphic lace."),
        ("Notions",
         "Tapestry needle for weaving in ends, small sharp scissors, "
         "stitch markers for tracking the pattern repeat, blocking pins and "
         "a mat to open up the finished lace."),
        ("Gauge",
         "Work a <strong>10 x 10&nbsp;cm</strong> swatch of the most "
         "common stitch and block it. Measure; change hooks to hit the "
         "drape you want before casting on the full piece."),
    ]
    overview_para = (
        f"A hand-drawn crochet chart decoded with computer vision: "
        f"<strong>{total}</strong> stitch symbols across "
        f"<strong>{unique}</strong> distinct types. Difficulty is rated "
        f"<strong>{difficulty}</strong> for this size of motif. The pattern "
        f"reads outward from the centre - each symbol becomes one stitch in "
        f"the yarn, worked into the position beneath it on the chart."
        if total > 0 else
        "No stitch symbols were detected on the uploaded chart. Check that "
        "the photo is in focus, on a light background, and uses standard "
        "crochet symbols. The rest of this tutorial is a generic "
        "chart-reading reference."
    )
    note_html = ""
    if note:
        note_html = _note("Heads up", _esc(note))
    sec1 = _section(
        1, "Materials & Gauge", "What you will need before you begin",
        f"<p>{overview_para}</p>" + _materials_grid(mat_cards) + note_html
        + _note(
            "Skill Level",
            f"This chart is rated <strong>{_esc(difficulty)}</strong>. "
            "Comfortable reading symbols outward from a centre ring? "
            "You are ready.",
        ),
    )

    # ---- 02 Stitch Guide ------------------------------------------------
    abbrev_rows = [
        ("Foundation:", "Chain 4-6 and slip-stitch to a ring (or make a magic ring)."),
        ("Round 1:",    "Work the centre stitches into the ring as shown at the chart's centre."),
        ("Round 2+:",   "Follow each symbol clockwise around the chart; use the legend below."),
        ("Repeat:",     "Continue outward until every symbol is worked."),
    ]
    abbrev_legend = _materials_grid([
        ("ch", "chain stitch"),
        ("sl st", "slip stitch"),
        ("sc", "single crochet"),
        ("hdc", "half double crochet"),
        ("dc", "double crochet"),
        ("tr", "treble crochet"),
        ("sk", "skip a stitch"),
        ("*...*", "repeat the marked section"),
    ])
    sec2 = _section(
        2, "Stitch Guide", "Standard symbols used across the chart",
        "<p>Each symbol on your chart maps to one of the stitches below. "
        "Work in the direction indicated by the chart; for circular motifs "
        "that is typically counter-clockwise (clockwise for left-handed).</p>"
        + _stitch_demo(SC_STITCH_SVG, "Core stitch family - sc, hdc, dc, inc, dec")
        + _pattern_row_block(abbrev_rows)
        + "<h3 class='sub-heading'>Abbreviations</h3>"
        + abbrev_legend,
    )

    # ---- 03 Stitch inventory + how to read ------------------------------
    inventory_body = (
        "<p>The detector picked out these symbols from your chart. Counts "
        "are approximate - verify any surprising numbers against the chart "
        "before you begin.</p>" + _stitch_chips(counts)
        if counts else
        "<p>No stitch symbols were detected. Work through the chart manually "
        "and note down the counts you see.</p>"
    )
    steps_html = "".join([
        _step(1, "Start at the centre",
              ["Begin with a magic ring (or chain 4-6 and slip stitch to form "
               "a ring). Work the first round of stitches directly into the "
               "ring, then pull the tail tight to close it."]),
        _step(2, "Work outward one round at a time",
              ["Each concentric band on the chart is a round. Work every "
               "symbol, moving counter-clockwise (clockwise if you are "
               "left-handed).",
               "Close each round with a slip stitch into the first stitch "
               "of the round - unless the chart clearly spirals without a "
               "join."]),
        _step(3, "Follow the symbols",
              ["Each printed symbol represents one stitch. Work the stitch "
               "into the stitch directly beneath the symbol on the chart, "
               "using the abbreviations above.",
               "If the chart shows a cluster of symbols joined at the base, "
               "work them all into the same stitch; if they fan out at the "
               "top, they share the top loop."]),
        _step(4, "Watch your increases",
              ["Outer rounds usually contain more stitches than inner ones - "
               "the chart compensates with chain spaces and v-stitches. If "
               "your work is cupping, loosen your tension; if it is ruffling, "
               "tighten up or drop a hook size."]),
    ])
    figures_html = _pattern_figures(scheme_uri)
    sec3 = _section(
        3, "Reading the Chart", "How to convert symbols into stitches",
        inventory_body + figures_html + steps_html,
    )

    # ---- 04 Finishing ---------------------------------------------------
    finishing_checklist = [
        "Fasten off, leaving a 15 cm tail for weaving in.",
        "Weave in every end on the wrong side using a tapestry needle, "
        "splitting plies to lock the tail discreetly.",
        "Wet-block by soaking in cool water for 10 minutes, pressing out "
        "the water in a towel, and pinning the piece to its finished shape.",
        "Let dry flat, completely, before unpinning - this opens the lace "
        "and sets the stitches.",
        "Optional: starch the finished piece lightly if it needs to hold "
        "shape on a wall hanging or table mat.",
    ]
    sec4 = _section(
        4, "Finishing", "The difference between homemade and heirloom",
        "<p>Blocking makes your chart look like the original. Skip it and "
        "the stitches stay puckered; do it well and every fan opens to its "
        "full size.</p>" + _checklist(finishing_checklist)
        + _note(
            "Detection Disclaimer",
            "Stitch counts come from computer vision and can be off by a "
            "few percent on busy charts. Treat them as a checklist, not a "
            "ground truth - cross-check unusual totals against the chart "
            "image itself.",
        ),
    )

    # ---- 05 Tips --------------------------------------------------------
    tips = _materials_grid([
        ("Use Stitch Markers",
         "Drop a marker every 10 symbols so you can catch miscounts early "
         "rather than unravelling whole rounds."),
        ("Check Tension Daily",
         "Crochet tension tightens when you are tired. Start each session "
         "with a few practice stitches to recalibrate before you continue."),
        ("Photograph Each Round",
         "Snap the finished piece after every round. When you make a "
         "mistake you can rewind visually instead of counting backwards."),
        ("Block Aggressively",
         "Pin the damp piece larger than you think; the fibres relax as "
         "they dry. Under-blocking is the single most common reason home-"
         "made lace looks flat."),
    ])
    sec5 = _section(
        5, "Tips & Notes", "For the best possible finished piece",
        tips + _note(
            "Pattern Source",
            "Auto-generated by the Crochet Scheme Generator: ModernBERT "
            "routes your brief, Gemini (optionally) produces a reference "
            "image, and a YOLOv8n OBB model decodes the chart. Your eye is "
            "still the final authority.",
        ),
    )

    doc = Document(
        title=title,
        subtitle="Chart Pattern Tutorial",
        meta=(
            f"{total} Stitches  -  {unique} Symbols  -  {difficulty}"
            if total else "Chart Reading Reference"
        ),
        sections=[sec1, sec2, sec3, sec4, sec5],
        palette=palette,
        cover_image_uri=_data_uri(cover_image_path),
        footer_label="Crochet Chart Tutorial",
        footer_note=(
            f"Generated {date.today():%d %B %Y}  -  verify gauge before you begin"
        ),
    )
    return render_document(doc)


# ---------------------------------------------------------------------------
# 3D pipeline: amigurumi -> professional HTML
# ---------------------------------------------------------------------------

def _difficulty_from_rows(rows: int) -> str:
    if rows < 10:  return "Beginner"
    if rows < 20:  return "Easy"
    if rows < 40:  return "Intermediate"
    return "Advanced"


def _estimate_size(rows: list[Any], stitch_height_cm: float = 0.6) -> str:
    if not rows:
        return "unknown"
    try:
        max_sts = max(getattr(r, "stitches", r.get("stitches", 0)) for r in rows)
    except Exception:
        max_sts = 0
    height_cm = max(1, len(rows)) * stitch_height_cm
    diam_cm = max_sts / 2.0 / 3.1416 * 2 if max_sts else 0
    return f"{height_cm:.0f} cm tall  -  {diam_cm:.0f} cm wide"


def build_3d_html(
    *,
    prompt: str,
    rows: list[Any],
    palette: list[PaletteColor] | None = None,
    cover_image_path: str | Path | None = None,
    view_paths: list[str | Path] | None = None,
    mesh_source: str | None = None,
) -> str:
    title = (prompt or "Amigurumi").strip().rstrip(".!?") or "Amigurumi"
    title = title[0].upper() + title[1:]
    palette = palette or []
    n_rows = len(rows)
    difficulty = _difficulty_from_rows(n_rows)
    size = _estimate_size(rows)

    def _row_text(r: Any) -> str:
        return getattr(r, "text", r.get("text", "") if isinstance(r, dict) else str(r))

    # ---- 01 Materials --------------------------------------------------
    mat_cards = [
        ("Yarn",
         "Worsted-weight cotton or acrylic, approx. "
         "<strong>50 - 100&nbsp;g</strong> per main colour. Cotton holds "
         "definition for tight amigurumi; acrylic forgives uneven tension."),
        ("Hook",
         "<strong>2.5&nbsp;mm - 3.5&nbsp;mm</strong>. Use a hook one size "
         "smaller than the yarn band suggests - tight stitches keep the "
         "fibrefill from peeking through."),
        ("Filling & Eyes",
         "Polyester fibrefill for stuffing (packed firmly but not lumpy). "
         "Safety eyes, embroidery floss, or felt details for the face."),
        ("Notions",
         "A locking stitch marker (essential for continuous rounds), a "
         "tapestry needle for closing and weaving in ends, and small sharp "
         "scissors."),
    ]
    overview = (
        f"This pattern was generated by slicing a 3D model of "
        f"<em>{_esc(title.lower())}</em> into horizontal rounds at a gauge "
        f"of 2 stitches per cm and 0.6 cm per row. The finished piece will "
        f"be roughly <strong>{_esc(size)}</strong> at the suggested gauge. "
        f"Rounds are worked continuously - do not join or turn."
    )
    sec1 = _section(
        1, "Materials & Gauge", "Everything you will need",
        f"<p>{overview}</p>" + _materials_grid(mat_cards) + _note(
            "Skill Level",
            f"Rated <strong>{_esc(difficulty)}</strong> - "
            f"{n_rows} rounds total. Comfortable with magic ring, sc, inc, "
            "and invisible decrease? You are ready.",
        ),
    )

    # ---- 02 Stitch guide ----------------------------------------------
    abbrev_legend = _materials_grid([
        ("MR", "magic ring"),
        ("sc", "single crochet"),
        ("inc", "2 sc in the same stitch (increase)"),
        ("dec", "invisible decrease - 2 sc worked together"),
        ("st(s)", "stitch(es)"),
        ("R", "round"),
        ("BLO", "back loops only"),
        ("FO", "fasten off"),
    ])
    sec2 = _section(
        2, "Stitch Guide", "The building blocks of amigurumi",
        "<p>Every round on this piece is worked with the same small set of "
        "stitches. Master them and you can work the whole pattern on "
        "autopilot.</p>"
        + _stitch_demo(SC_STITCH_SVG, "Core amigurumi stitches - sc, hdc, dc, inc, dec")
        + "<h3 class='sub-heading'>Abbreviations</h3>"
        + abbrev_legend
        + _note(
            "Reading the Pattern",
            "Notation like <strong>(2 sc, inc) x6 = 24</strong> means: work "
            "2 single crochets, then an increase, and repeat that bracketed "
            "sequence six times. The ` = 24` is the stitch count at the end "
            "of the round.",
        ),
    )

    # ---- 03 Colour palette --------------------------------------------
    sec3 = _section(
        3, "Colour Palette", "Sampled from your reference image",
        "<p>The largest share below usually becomes the body colour. Smaller "
        "shares work well for contrast rounds, paws, belly patches, or "
        "embroidered details.</p>"
        + _yarn_palette(palette),
    )

    # ---- 04 Pattern rounds --------------------------------------------
    # Break rounds into chunks of ~10 so the dark pattern-row blocks don't get huge.
    pattern_blocks: list[str] = []
    for i in range(0, n_rows, 10):
        chunk = rows[i:i + 10]
        lines = []
        for r in chunk:
            text = _row_text(r)
            # Split "R7: ..." into a label + body so we can colour the label.
            if ":" in text:
                lbl, body = text.split(":", 1)
                lines.append((f"{lbl.strip()}:", body.strip()))
            else:
                lines.append(("R?:", text))
        pattern_blocks.append(_pattern_row_block(lines))
    rounds_html = "".join(pattern_blocks) or "<p><em>No rounds generated.</em></p>"
    intro_steps = _step(
        1, "Start the body",
        ["Begin with a magic ring and work round 1 into it. Pull the tail "
         "tight to close the ring - this becomes the bottom of the piece."],
    ) + _step(
        2, "Work in continuous rounds",
        ["Do not join or turn. Slip a stitch marker into the first stitch "
         "of each round and move it up as you go.",
         "Check the target stitch count at the end of every round. If the "
         "numbers drift, unravel the current round and rework it - mistakes "
         "compound quickly in amigurumi."],
    ) + _step(
        3, "Stuff as you go",
        ["Add fibrefill every 5 - 6 rounds, in small pinches. Never wait "
         "until the end; you cannot reach the bottom through a narrow "
         "closing round."],
    )
    sec4 = _section(
        4, "Pattern Rounds", "Round-by-round stitch counts",
        intro_steps
        + "<h3 class='sub-heading'>The Rounds</h3>"
        + rounds_html,
    )

    # ---- 05 Assembly / Finishing --------------------------------------
    assembly = _assembly_flowchart_svg([
        "Stuff Body",
        "Close Final Round",
        "Weave Ends",
        "Attach Details",
        "Light Block",
    ])
    finishing = [
        "Stuff firmly as the piece narrows - use small pinches so no lumps "
        "form along the surface.",
        "Fasten off with a long tail. Weave the tail through the front "
        "loops of the final round and pull tight to close the opening "
        "cleanly.",
        "Secure the closing tail on the inside of the piece, then weave in "
        "all remaining ends.",
        "Attach any separate parts (ears, arms, horns, tail) with a matching-"
        "colour yarn using a whip stitch around the base of each piece.",
        "Lightly steam-block to relax the stitches and even out the shape - "
        "hold the steam iron a few centimetres above the surface, never "
        "touching.",
    ]
    sec5 = _section(
        5, "Assembly & Finishing", "Turning the tube into a toy",
        "<p>The pattern produces one continuous tube. These finishing steps "
        "turn that tube into a recognisable figure.</p>"
        + assembly
        + _checklist(finishing),
    )

    # ---- 06 Tips -------------------------------------------------------
    tips = _materials_grid([
        ("Stitches, not Rows",
         "Count stitches at the end of every round. Getting that count "
         "right matters far more than how quickly you finish the round."),
        ("Small Hook, Tight Stitches",
         "If your stuffing shows through the fabric, drop a hook size. "
         "Amigurumi fabric should feel dense, almost stiff."),
        ("Auto-Generated vs Hand-Designed",
         "The slicer distributes increases and decreases evenly, which is "
         "smooth but generic. For features like an expressive muzzle or "
         "pointed ears, hand-design those pieces separately."),
        ("Mesh Source",
         f"Mesh produced via <strong>{_esc(mesh_source or 'demo')}</strong>. "
         "A demo mesh is a fallback shape - expect realistic geometry only "
         "when Hunyuan3D or your own model supplies the mesh."),
    ])
    sec6 = _section(
        6, "Tips & Notes", "To get the best result",
        tips + _note(
            "Gauge Reminder",
            "Every finished size on this page assumes 2 stitches per cm "
            "and 0.6 cm per row. If your personal gauge differs, scale "
            "the finished piece by the ratio between your gauge and "
            "this one.",
        ),
    )

    doc = Document(
        title=title,
        subtitle="Amigurumi Pattern Tutorial",
        meta=f"{n_rows} Rounds  -  {size}  -  {difficulty}",
        sections=[sec1, sec2, sec3, sec4, sec5, sec6],
        palette=palette,
        cover_image_uri=_data_uri(cover_image_path),
        footer_label="Crochet Amigurumi Tutorial",
        footer_note=(
            f"Generated {date.today():%d %B %Y}  -  pattern auto-sliced from a 3D mesh"
        ),
    )
    return render_document(doc)
