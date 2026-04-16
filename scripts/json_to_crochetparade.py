"""CLI: convert a stitch-detection JSON into CrochetPARADE DSL.

Usage
-----
    python scripts/json_to_crochetparade.py data/samples/input.json -o out.cpd
    python scripts/json_to_crochetparade.py data/samples/input.json   # stdout
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crochet.dsl import convert  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="Path to the detector JSON file.")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Where to write the DSL (default: stdout).")
    p.add_argument("--row-tolerance", type=float, default=None,
                   help="Y-distance within which stitches are in the same row "
                        "(default: 0.5 × 75th-percentile stitch height).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with args.input.open() as f:
        data = json.load(f)
    detections = data.get("detections", data)
    dsl = convert(detections, row_tolerance=args.row_tolerance)
    if args.output is None:
        sys.stdout.write(dsl)
    else:
        args.output.write_text(dsl)
        print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
