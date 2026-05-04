"""Tile + emit a Label Studio ``tasks.json`` import file."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, TypedDict

import cv2 as cv
import numpy as np

from .inference import estimate_stitch_size, tile_starts


class RotatedRect(TypedDict):
    """Label Studio ``RectangleLabels`` value with rotation."""

    x: float
    y: float
    width: float
    height: float
    rotation: float


def obb_corners_to_rotated_rect(corners_norm: list[tuple[float, float]],
                                img_w: int, img_h: int) -> RotatedRect:
    """Convert four normalised OBB corners into a Label Studio rotated rect.

    The anchor is the corner with smallest y (smallest x as tie-breaker)
    so the rectangle's "width" axis is the more horizontal of its sides.
    """
    pts = [(c[0] * img_w, c[1] * img_h) for c in corners_norm]
    anchor_idx, anchor = sorted(enumerate(pts),
                                key=lambda p: (p[1][1], p[1][0]))[0]
    next_pt = pts[(anchor_idx + 1) % 4]
    prev_pt = pts[(anchor_idx + 3) % 4]
    ang_next = math.atan2(next_pt[1] - anchor[1], next_pt[0] - anchor[0])
    ang_prev = math.atan2(prev_pt[1] - anchor[1], prev_pt[0] - anchor[0])

    if abs(ang_next) <= abs(ang_prev):
        wv = (next_pt[0] - anchor[0], next_pt[1] - anchor[1])
        hv = (prev_pt[0] - anchor[0], prev_pt[1] - anchor[1])
        rotation = math.degrees(ang_next)
    else:
        wv = (prev_pt[0] - anchor[0], prev_pt[1] - anchor[1])
        hv = (next_pt[0] - anchor[0], next_pt[1] - anchor[1])
        rotation = math.degrees(ang_prev)
    if rotation < 0:
        rotation += 360

    return {
        "x":        anchor[0] / img_w * 100,
        "y":        anchor[1] / img_h * 100,
        "width":    math.hypot(*wv) / img_w * 100,
        "height":   math.hypot(*hv) / img_h * 100,
        "rotation": rotation,
    }


def tile_and_save(image_path: str | Path,
                  model: Any,
                  out_root: str | Path,
                  *,
                  tile_size: int = 640,
                  target_stitch_px: int = 100,
                  overlap: float = 0.25,
                  conf: float = 0.20,
                  iou_nms: float = 0.45,
                  min_tile: int = 128) -> tuple[int, int]:
    """Adaptively tile a photo and save tiles + per-tile YOLO-OBB labels."""
    image_path = Path(image_path)
    out_root = Path(out_root)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)

    image = cv.imread(str(image_path))
    if image is None:
        return 0, 0
    h, w = image.shape[:2]
    stem = image_path.stem.replace(" ", "_")

    est = estimate_stitch_size(model, image, tile_size=tile_size, conf=0.15)
    if est is None or est <= 0:
        eff = min(tile_size, min(h, w))
    else:
        eff = max(min_tile, int(tile_size * est / target_stitch_px))
        eff = min(eff, min(h, w))

    if eff >= max(h, w):
        x_starts, y_starts = [0], [0]
    else:
        stride = max(1, int(eff * (1 - overlap)))
        x_starts = tile_starts(w, eff, stride)
        y_starts = tile_starts(h, eff, stride)

    n_tiles = n_dets = 0
    for y1 in y_starts:
        for x1 in x_starts:
            y2 = min(y1 + eff, h)
            x2 = min(x1 + eff, w)
            tile = image[y1:y2, x1:x2]
            th, tw = tile.shape[:2]
            res = model.predict(tile, conf=conf, iou=iou_nms, verbose=False)[0]
            lines: list[str] = []
            if res.obb is not None:
                for box in res.obb:
                    corn = box.xyxyxyxy.cpu().numpy().reshape(4, 2).astype(np.float32)
                    coords: list[float] = []
                    for cx, cy in corn:
                        coords.append(float(np.clip(cx / tw, 0, 1)))
                        coords.append(float(np.clip(cy / th, 0, 1)))
                    lines.append(f"{int(box.cls[0])} "
                                 + " ".join(f"{c:.6f}" for c in coords))

            tile_name = f"{stem}_tile_{n_tiles:03d}"
            cv.imwrite(str(out_root / "images" / f"{tile_name}.jpg"),
                       tile, [cv.IMWRITE_JPEG_QUALITY, 92])
            with open(out_root / "labels" / f"{tile_name}.txt", "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            n_tiles += 1
            n_dets += len(lines)
    return n_tiles, n_dets


def build_tasks_json(out_root: str | Path,
                     class_names: list[str],
                     url_prefix: str,
                     tasks_filename: str = "tasks.json") -> Path:
    """Build a Label Studio import file from saved tiles + labels."""
    out_root = Path(out_root)
    tasks: list[dict[str, Any]] = []
    for img_path in sorted((out_root / "images").glob("*.jpg")):
        img = cv.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        lbl_path = out_root / "labels" / (img_path.stem + ".txt")
        results: list[dict[str, Any]] = []
        if lbl_path.is_file():
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                cls_id = int(parts[0])
                if not (0 <= cls_id < len(class_names)):
                    continue
                coords = list(map(float, parts[1:9]))
                rect = obb_corners_to_rotated_rect(
                    [(coords[2 * i], coords[2 * i + 1]) for i in range(4)],
                    iw, ih,
                )
                results.append({
                    "original_width":  iw,
                    "original_height": ih,
                    "image_rotation":  0,
                    "value": {
                        **rect,
                        "rectanglelabels": [class_names[cls_id]],
                    },
                    "from_name": "label",
                    "to_name":   "image",
                    "type":      "rectanglelabels",
                })
        task = {"data": {"image": url_prefix + img_path.name}}
        if results:
            task["predictions"] = [{
                "model_version": "yolo-obb-auto",
                "result":        results,
            }]
        tasks.append(task)

    out_path = out_root / tasks_filename
    out_path.write_text(json.dumps(tasks, indent=2))
    return out_path
