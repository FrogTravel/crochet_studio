"""Interactive keyboard-driven labelling tool for symbol crops.

Default key map (matches ``data/project_yolo_obb/data.yaml`` — 9 classes)::

    c - chain                                                          [id 0]
    d - double crochet (t with one bar)                                [id 1]
    r - double treble crochet (t with three bars)                      [id 2]
    e - enseble_chain (several chains together)                        [id 3]
    f - fan (multiple stitches from one base)                          [id 4]
    h - half_double                                                    [id 5]
    n - noise                                                          [id 6]
    s - single crochet (t with no bars)                                [id 7]
    t - treble crochet (t with two bars)                               [id 8]
    Space / → → skip (leave in unlabeled, move to next)
    u       → undo last assignment
    q       → quit (saves progress)

Crop filenames follow the convention ``{image_stem}_{idx:04d}.png``.
Already-labelled crops (anywhere under ``output_dir/<class>/``) are
skipped on subsequent runs — labelling is incremental-safe.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2 as cv
import numpy as np


DEFAULT_SYMBOL_KEYS: dict[str, str] = {
    "c": "chain",          # id 0
    "d": "double",         # id 1
    "r": "double treble",  # id 2
    "e": "enseble_chain",  # id 3
    "f": "fan",            # id 4
    "h": "half_double",    # id 5
    "n": "noise",          # id 6
    "s": "single",         # id 7
    "t": "treble",         # id 8
}


class Labeler:
    """Keyboard-driven labelling tool.

    Usage::

        labeler = Labeler("data/segments/unlabeled", "data/segments/labeled")
        labeler.run()
    """

    WINDOW_NAME: str = "Labeler"
    DISPLAY_SIZE: tuple[int, int] = (320, 320)

    def __init__(
        self,
        crops_dir: str | Path,
        output_dir: str | Path,
        raw_dir: str | Path | None = None,
        symbol_keys: dict[str, str] | None = None,
        context_size: int = 200,
    ) -> None:
        self.crops_dir = Path(crops_dir)
        self.output_dir = Path(output_dir)
        self.raw_dir = Path(raw_dir) if raw_dir else None
        self.symbol_keys = symbol_keys or DEFAULT_SYMBOL_KEYS
        self.context_size = context_size

        for cls in self.symbol_keys.values():
            (self.output_dir / cls).mkdir(parents=True, exist_ok=True)

    # ── Public ────────────────────────────────────────────────────────────
    def run(self) -> None:
        files = self._pending_files()
        if not files:
            print(f"Nothing to label — no new crops in {self.crops_dir}")
            self._print_summary()
            return

        print(self._help_text(len(files)))
        history: list[tuple[Path, Path]] = []
        i = 0

        while i < len(files):
            fpath = files[i]
            img = cv.imread(str(fpath))
            if img is None:
                print(f"  Skipping unreadable file: {fpath.name}")
                i += 1
                continue

            source_stem, symbol_idx = self._parse_filename(fpath)
            display = self._build_display(img, source_stem, symbol_idx)

            title = (
                f"[{i + 1}/{len(files)}]  {source_stem} #{symbol_idx}"
                f"  |  {self._key_hint()}"
            )
            cv.imshow(self.WINDOW_NAME, display)
            cv.setWindowTitle(self.WINDOW_NAME, title)

            key = cv.waitKey(0) & 0xFF
            cv.destroyAllWindows()
            ch = chr(key) if key < 128 else ""

            if ch == "q":
                print(f"Quit. Labelled {i} / {len(files)} images this session.")
                break
            if ch == "u":
                if history:
                    src, dst = history.pop()
                    shutil.move(str(dst), str(src))
                    print(f"  Undo: {dst.name} → unlabeled")
                    i = max(0, i - 1)
                else:
                    print("  Nothing to undo.")
                continue
            if key in (ord(" "), 83, 0xFF & ord("\t")):  # Space / right-arrow / Tab
                print(f"  [{i + 1}] {fpath.name} → skipped")
                i += 1
                continue
            if ch in self.symbol_keys:
                cls = self.symbol_keys[ch]
                dst = self.output_dir / cls / fpath.name
                shutil.move(str(fpath), str(dst))
                print(f"  [{i + 1}] {fpath.name} → {cls}")
                history.append((fpath, dst))
                i += 1
                continue

            print(
                f"  Unknown key '{ch}' (code {key}). "
                f"Use: {self._key_hint()}  |  Space=skip  u=undo  q=quit"
            )

        cv.destroyAllWindows()
        print("\nLabelling session complete.")
        self._print_summary()

    # ── Private helpers ───────────────────────────────────────────────────
    def _pending_files(self) -> list[Path]:
        labeled_names: set[str] = {p.name for p in self.output_dir.glob("**/*.png")}
        all_files = sorted(
            f for f in self.crops_dir.iterdir()
            if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        return [f for f in all_files if f.name not in labeled_names]

    @staticmethod
    def _parse_filename(fpath: Path) -> tuple[str, str]:
        """Parse ``{image_stem}_{idx:04d}.png`` → ``(image_stem, idx_str)``."""
        stem = fpath.stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0], parts[1]
        return stem, "?"

    def _build_display(
        self,
        crop_bgr: np.ndarray,
        source_stem: str,
        symbol_idx: str,
    ) -> np.ndarray:
        """Left: enlarged crop. Right (optional): source-image context thumbnail."""
        crop_disp = cv.resize(crop_bgr, self.DISPLAY_SIZE,
                              interpolation=cv.INTER_NEAREST)

        context = self._load_context(source_stem)
        if context is not None:
            h = self.DISPLAY_SIZE[1]
            ctx_w = int(context.shape[1] * h / context.shape[0])
            context_disp = cv.resize(context, (ctx_w, h))
            sep = np.full((h, 4, 3), 180, dtype=np.uint8)
            display = np.hstack([crop_disp, sep, context_disp])
        else:
            display = crop_disp

        label_text = f"src: {source_stem}  #{symbol_idx}"
        cv.putText(display, label_text, (6, self.DISPLAY_SIZE[1] - 8),
                   cv.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 0), 1, cv.LINE_AA)
        return display

    def _load_context(self, source_stem: str) -> np.ndarray | None:
        if self.raw_dir is None:
            return None
        for ext in (".png", ".jpg", ".jpeg"):
            p = self.raw_dir / (source_stem + ext)
            if p.exists():
                img = cv.imread(str(p))
                if img is not None:
                    h, w = img.shape[:2]
                    scale = min(1.0, self.context_size * 2 / max(h, w))
                    return cv.resize(img, (int(w * scale), int(h * scale)))
        return None

    def _key_hint(self) -> str:
        return "  ".join(f"{k}={v}" for k, v in self.symbol_keys.items())

    def _help_text(self, total: int) -> str:
        return "\n".join([
            f"\nLabelling {total} new crop(s) from {self.crops_dir}",
            f"Keys : {self._key_hint()}",
            "      Space = skip  |  u = undo  |  q = quit",
            "-" * 56,
        ])

    def _print_summary(self) -> None:
        print("\nLabelled counts:")
        for cls in sorted(self.symbol_keys.values()):
            folder = self.output_dir / cls
            n = len(list(folder.glob("*.png"))) if folder.exists() else 0
            print(f"  {cls:<15}: {n}")
        remaining = len(list(self.crops_dir.glob("*.png")))
        if remaining:
            print(f"  {'(unlabeled)':<15}: {remaining}")
