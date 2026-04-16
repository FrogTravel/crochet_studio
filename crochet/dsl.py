"""Convert a parsed crochet-scheme JSON into CrochetPARADE DSL.

DSL reference: https://www.crochetparade.org/Manual.html

Input JSON shape
----------------
Top-level key ``detections`` is a list of objects of the form::

    {
        "abbr"  : "sc" | "dc" | "ch" | "hdc" | "tr" | ...,
        "center": [x, y],
        "width" : <float, optional>,
        "height": <float, optional>,
        ...
    }

Processing pipeline
-------------------
1. **Row clustering** — stitches are grouped into rows by their y
   coordinate (robust to rotated stitches).
2. **Shell detection** — runs of 3+ close-by ``dc`` stitches collapse
   into a ``shell`` token (``DEF: shell = dc,4dc@[@]``).
3. **Skip inference** — the pitch is calibrated against the foundation
   chain row; gaps wider than one pitch become ``sk`` / ``Nsk`` tokens.
4. **Edge labels** — the leftmost / rightmost ``sc`` of each sc-edged
   row is labeled ``sc.B<k>`` / ``sc.A<k>``; a following ``dc``-edged
   row attaches to those labels via ``@A<k>`` / ``@B<k>``.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple


# ── Data classes ───────────────────────────────────────────────────────────
@dataclass
class Stitch:
    abbr: str
    x: float
    y: float
    width: float = 0.0
    height: float = 0.0

    @classmethod
    def from_detection(cls, det: dict) -> "Stitch":
        cx, cy = det["center"]
        return cls(
            abbr=det["abbr"],
            x=float(cx),
            y=float(cy),
            width=float(det.get("width", 0.0)),
            height=float(det.get("height", 0.0)),
        )


@dataclass
class Row:
    stitches: List[Stitch] = field(default_factory=list)

    @property
    def mean_y(self) -> float:
        return sum(s.y for s in self.stitches) / max(len(self.stitches), 1)


@dataclass
class Token:
    """A logical DSL element before skip-insertion and rendering."""

    kind: str      # "shell" | "dcs" | "stitch"
    count: int
    abbr: str
    x_left: float
    x_right: float
    x_anchor: float


# ── Row clustering ─────────────────────────────────────────────────────────
def tall_stitch_height(stitches: Sequence[Stitch]) -> float:
    """75th-percentile height (= typical upright ``dc`` / ``tr`` height)."""
    heights = sorted(s.height for s in stitches if s.height > 0)
    if not heights:
        return 100.0
    idx = min(len(heights) - 1, int(0.75 * len(heights)))
    return heights[idx]


def cluster_into_rows(
    stitches: Sequence[Stitch],
    row_tolerance: float | None = None,
) -> List[Row]:
    if not stitches:
        return []

    if row_tolerance is None:
        row_tolerance = 0.5 * tall_stitch_height(stitches)

    ordered = sorted(stitches, key=lambda s: s.y)
    rows: List[Row] = []
    current = Row(stitches=[ordered[0]])
    for st in ordered[1:]:
        if abs(st.y - current.mean_y) <= row_tolerance:
            current.stitches.append(st)
        else:
            rows.append(current)
            current = Row(stitches=[st])
    rows.append(current)

    return _absorb_stragglers(rows, min_size=3)


def _absorb_stragglers(rows: List[Row], min_size: int = 3) -> List[Row]:
    """Merge tiny rows (<min_size stitches) into their nearest neighbour."""
    if len(rows) <= 1:
        return rows
    changed = True
    while changed:
        changed = False
        for i, r in enumerate(rows):
            if len(r.stitches) >= min_size:
                continue
            candidates = [j for j in (i - 1, i + 1) if 0 <= j < len(rows)]
            if not candidates:
                continue
            j = min(candidates, key=lambda k: abs(rows[k].mean_y - r.mean_y))
            rows[j].stitches.extend(r.stitches)
            del rows[i]
            changed = True
            break
    return rows


# ── Token building ─────────────────────────────────────────────────────────
SHELL_SIZE = 5
SHELL_DC_GAP = 70.0


def _centroid(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def build_tokens(row_stitches: List[Stitch]) -> List[Token]:
    """Sort by x and collapse consecutive ``dc`` / ``ch`` runs."""
    row_sorted = sorted(row_stitches, key=lambda s: s.x)
    out: List[Token] = []
    i = 0
    while i < len(row_sorted):
        s = row_sorted[i]

        # dc run → shell / dcs / single
        if s.abbr == "dc":
            j = i + 1
            while (
                j < len(row_sorted)
                and row_sorted[j].abbr == "dc"
                and (row_sorted[j].x - row_sorted[j - 1].x) <= SHELL_DC_GAP
            ):
                j += 1
            run = row_sorted[i:j]
            xs = [r.x for r in run]
            count = len(run)
            kind = "shell" if count >= 3 else ("dcs" if count == 2 else "stitch")
            out.append(Token(
                kind=kind, count=count, abbr="dc",
                x_left=xs[0], x_right=xs[-1], x_anchor=_centroid(xs),
            ))
            i = j
            continue

        # ch run → single Nch token
        if s.abbr == "ch":
            j = i + 1
            while (
                j < len(row_sorted)
                and row_sorted[j].abbr == "ch"
                and (row_sorted[j].x - row_sorted[j - 1].x) <= SHELL_DC_GAP
            ):
                j += 1
            run = row_sorted[i:j]
            xs = [r.x for r in run]
            out.append(Token(
                kind="stitch", count=len(run), abbr="ch",
                x_left=xs[0], x_right=xs[-1], x_anchor=_centroid(xs),
            ))
            i = j
            continue

        # anything else: single stitch token
        out.append(Token(
            kind="stitch", count=1, abbr=s.abbr,
            x_left=s.x, x_right=s.x, x_anchor=s.x,
        ))
        i += 1
    return out


# ── Pitch calibration & skip insertion ─────────────────────────────────────
def calibrate_pitch(foundation: Sequence[Stitch], fallback: float = 80.0) -> float:
    """Median x-gap between neighbouring chains in the foundation row."""
    xs = sorted(s.x for s in foundation)
    diffs = [b - a for a, b in zip(xs, xs[1:]) if (b - a) > 5]
    if not diffs:
        return fallback
    return statistics.median(diffs)


def _sk(n: int) -> str:
    if n <= 0:
        return ""
    return "sk" if n == 1 else f"{n}sk"


# ── Row rendering ──────────────────────────────────────────────────────────
def render_foundation(row: Row) -> str:
    ch_count = sum(1 for s in row.stitches if s.abbr == "ch") or len(row.stitches)
    return f"{ch_count}ch, turn"


def _token_text(tok: Token) -> str:
    if tok.kind == "shell":
        return "shell" if tok.count == SHELL_SIZE else f"{tok.count}dc"
    if tok.kind == "dcs":
        return f"{tok.count}dc"
    if tok.count > 1:
        return f"{tok.count}{tok.abbr}"
    return tok.abbr


def render_row(
    row: Row,
    pitch: float,
    row_index: int,
    prev_sc_label_suffix: str | None,
    x_start: float,
    x_end: float,
) -> Tuple[str, str | None]:
    """Render one pattern row to DSL.

    Returns ``(dsl_line, new_sc_label_suffix)`` — the second element is
    the suffix introduced by this row if it was an sc-edged row, so the
    next row can reference it via ``@A<sfx>``/``@B<sfx>``.
    """
    tokens = build_tokens(row.stitches)
    if not tokens:
        return "", prev_sc_label_suffix

    # Peel off edge turning chains.
    # Rising turning chains sit between rows; the clusterer attaches them
    # to the nearer row. In DSL work-order they always appear *first*
    # (they start the new row after `turn`), regardless of which x they
    # occupied in the image.
    turning_ch = 0
    while tokens and tokens[0].abbr == "ch":
        turning_ch += tokens[0].count
        tokens.pop(0)
    while tokens and tokens[-1].abbr == "ch":
        turning_ch += tokens[-1].count
        tokens.pop()

    if not tokens:
        return f"{turning_ch}ch, turn", prev_sc_label_suffix

    first, last = tokens[0], tokens[-1]
    sc_edged = first.abbr == "sc" and last.abbr == "sc"
    dc_edged = first.abbr == "dc" and last.abbr == "dc"

    sc_label_suffix: str | None = None
    if sc_edged:
        k = (row_index - 1) // 2
        sc_label_suffix = "" if k == 0 else str(k)

    def decorated(tok: Token, position: str) -> str:
        base = _token_text(tok)
        if sc_edged and tok.abbr == "sc" and position in {"first", "last"}:
            tag = "B" if position == "first" else "A"
            return f"{base}.{tag}{sc_label_suffix}"
        if dc_edged and prev_sc_label_suffix is not None:
            # After turn, the new row's left == old row's right == .A.
            if position == "first" and tok.abbr in {"dc", "ch"}:
                return f"{base}@A{prev_sc_label_suffix}"
            if position == "last" and tok.abbr in {"dc", "ch"}:
                return f"{base}@B{prev_sc_label_suffix}"
        return base

    pieces: List[str] = []
    if turning_ch:
        # For dc-edged rows we want at least 3ch (to match dc height).
        n_ch = max(turning_ch, 3) if dc_edged else turning_ch
        pieces.append(f"{n_ch}ch" if n_ch > 1 else "ch")

    prev_right: float | None = None
    for idx, tok in enumerate(tokens):
        if prev_right is not None:
            gap = tok.x_anchor - prev_right
            units = gap / pitch
            # One pitch = one underlying chain of spacing between centres,
            # so the number of *skipped* stitches is one less than that.
            n_sk = max(0, int(round(units - 1)))
            if n_sk > 0:
                pieces.append(_sk(n_sk))
        position = "first" if idx == 0 else ("last" if idx == len(tokens) - 1 else "mid")
        pieces.append(decorated(tok, position))
        prev_right = tok.x_anchor

    return ", ".join(pieces) + ", turn", sc_label_suffix


# ── Assembly ───────────────────────────────────────────────────────────────
PREAMBLE = [
    "# --- Reusable stitch definitions -------------------------------",
    "DEF: shell = dc,4dc@[@]",
    "COLOR: rgb(60,90,160)",
    "BACKGROUND: white",
    "# --- Foundation chain ------------------------------------------",
]


def convert(detections: Iterable[dict], row_tolerance: float | None = None) -> str:
    """Convert a list of detection dicts into a CrochetPARADE DSL string."""
    stitches = [Stitch.from_detection(d) for d in detections]
    rows_top_down = cluster_into_rows(stitches, row_tolerance)
    if not rows_top_down:
        return "\n".join(PREAMBLE) + "\n"

    rows_bu = list(reversed(rows_top_down))  # foundation first

    foundation = rows_bu[0]
    pitch = calibrate_pitch(foundation.stitches)
    xs = sorted(s.x for s in foundation.stitches)
    x_start, x_end = xs[0], xs[-1]

    lines: List[str] = list(PREAMBLE)
    lines.append(render_foundation(foundation))

    prev_sc_suffix: str | None = None
    for i, row in enumerate(rows_bu[1:], start=1):
        dsl_line, new_suffix = render_row(
            row, pitch, i, prev_sc_suffix, x_start, x_end,
        )
        if dsl_line:
            lines.append(dsl_line)
        if new_suffix is not None:
            prev_sc_suffix = new_suffix

    # Strip trailing ", turn" on the final line (legal but tidy).
    if lines and lines[-1].endswith(", turn"):
        lines[-1] = lines[-1][: -len(", turn")]

    return "\n".join(lines) + "\n"
