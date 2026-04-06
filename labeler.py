"""
labeler.py

Interactive keyboard-driven labeling tool for symbol crops.
Press a key to assign a class; press 'u' to undo the last assignment.

Default key map:
    c → chain
    s → slip_stitch
    t → treble
    d → double
    n → noise
    u → undo last
    q → quit (saves progress)
"""

from __future__ import annotations
import os
import shutil
from pathlib import Path

import cv2 as cv


DEFAULT_SYMBOL_KEYS: dict[str, str] = {
    "c": "chain",
    "s": "slip_stitch",
    "t": "treble",
    "d": "double",
    "n": "noise",
}


class Labeler:
    """
    Keyboard-driven labeling tool.

    Usage:
        labeler = Labeler("data/segments/unlabeled", "data/segments/labeled")
        labeler.run()
    """

    WINDOW_NAME = "Labeler"
    DISPLAY_SIZE = (300, 300)   # upscale tiny crops for easier viewing

    def __init__(
        self,
        crops_dir: str | Path,
        output_dir: str | Path,
        symbol_keys: dict[str, str] | None = None,
    ):
        self.crops_dir = Path(crops_dir)
        self.output_dir = Path(output_dir)
        self.symbol_keys = symbol_keys or DEFAULT_SYMBOL_KEYS

        # Create output class folders up front
        for cls in self.symbol_keys.values():
            (self.output_dir / cls).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the interactive labeling session."""
        files = sorted(
            f for f in self.crops_dir.iterdir()
            if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not files:
            print(f"No images found in {self.crops_dir}")
            return

        print(self._help_text(len(files)))
        history: list[tuple[Path, Path]] = []  # (src, dst) for undo
        i = 0

        while i < len(files):
            fpath = files[i]
            img = cv.imread(str(fpath))
            if img is None:
                print(f"  Skipping unreadable file: {fpath.name}")
                i += 1
                continue

            display = cv.resize(img, self.DISPLAY_SIZE, interpolation=cv.INTER_NEAREST)
            title = f"[{i+1}/{len(files)}]  {fpath.name}  |  {self._key_hint()}"
            cv.imshow(title, display)
            cv.setWindowTitle(self.WINDOW_NAME, title)

            key = chr(cv.waitKey(0) & 0xFF)
            cv.destroyAllWindows()

            if key == "q":
                print(f"Quit. Labeled {i} / {len(files)} images.")
                break
            elif key == "u":
                if history:
                    src, dst = history.pop()
                    shutil.move(str(dst), str(src))
                    print(f"  Undid: moved {dst.name} back to unlabeled")
                    i = max(0, i - 1)
                else:
                    print("  Nothing to undo.")
            elif key in self.symbol_keys:
                cls = self.symbol_keys[key]
                dst = self.output_dir / cls / fpath.name
                shutil.move(str(fpath), str(dst))
                print(f"  [{i+1}] {fpath.name} → {cls}")
                history.append((fpath, dst))
                i += 1
            else:
                print(f"  Unknown key '{key}'. Use: {self._key_hint()} | u=undo | q=quit")

        print("Labeling session complete.")
        self._print_summary()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _key_hint(self) -> str:
        return "  ".join(f"{k}={v}" for k, v in self.symbol_keys.items())

    def _help_text(self, total: int) -> str:
        lines = [
            f"\nLabeling {total} images from {self.crops_dir}",
            f"Keys: {self._key_hint()}  |  u=undo  q=quit",
            "-" * 50,
        ]
        return "\n".join(lines)

    def _print_summary(self) -> None:
        print("\nFinal counts:")
        for cls in sorted(self.symbol_keys.values()):
            folder = self.output_dir / cls
            n = len(list(folder.glob("*.png"))) if folder.exists() else 0
            print(f"  {cls}: {n}")
        remaining = len(list(self.crops_dir.glob("*.png")))
        if remaining:
            print(f"  (unlabeled remaining: {remaining})")
