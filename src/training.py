"""Step 2 — training and evaluation of the YOLOv8n-OBB detector.

Two recipes mirror the reference notebook:

* :func:`train_local`  — short MPS / CPU runs on a laptop.
* :func:`train_remote` — long Colab / CUDA runs.

Evaluation matches predictions to ground-truth OBB labels via greedy
IoU and reports a confusion matrix plus per-class precision, recall, F1.
"""

from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Literal

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import yaml


def train_local(data_yaml: str | Path,
                base_weights: str = "yolov8n-obb.pt",
                epochs: int = 10,
                batch: int = 4,
                imgsz: int = 640,
                degrees: float = 180.0,
                workers: int = 4,
                device: Literal["mps", "cpu"] = "mps") -> Path:
    """Train YOLOv8n-OBB on Apple Silicon (MPS) or CPU."""
    import torch
    from ultralytics import YOLO

    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    model = YOLO(base_weights)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        degrees=degrees,
        workers=workers,
        plots=True,
    )
    return Path(model.trainer.best)


def train_remote(data_yaml: str | Path,
                 base_weights: str = "yolov8n-obb.pt",
                 epochs: int = 150,
                 batch: int = 16,
                 imgsz: int = 640,
                 degrees: float = 180.0,
                 workers: int = 8,
                 device: int | str = 0,
                 project: str | Path | None = None,
                 name: str = "obb_train",
                 cache: bool = True) -> Path:
    """Train YOLOv8n-OBB on a CUDA GPU."""
    from ultralytics import YOLO

    model = YOLO(base_weights)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        degrees=degrees,
        workers=workers,
        cache=cache,
        project=str(project) if project is not None else None,
        name=name,
    )
    return Path(model.trainer.best)


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate metrics returned by :func:`evaluate`.

    Attributes:
        confusion_matrix: ``(N+1, N+1)`` integer matrix; the last
            row/column is "background / missed".
        precision: Per-class precision, length ``N``.
        recall: Per-class recall, length ``N``.
        f1: Per-class F1, length ``N``.
        class_names: Class names ordered to match the matrix axes.
    """

    confusion_matrix: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    class_names: list[str]


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Internal: AABB IoU between two corner polygons (used for matching)."""
    ax1, ay1 = a.min(0); ax2, ay2 = a.max(0)
    bx1, by1 = b.min(0); bx2, by2 = b.max(0)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


def _read_gt(label_path: Path, img_w: int, img_h: int
             ) -> list[tuple[int, np.ndarray]]:
    """Internal: read a YOLO-OBB ground-truth label file."""
    gt: list[tuple[int, np.ndarray]] = []
    if not label_path.is_file():
        return gt
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        cls_id = int(parts[0])
        corners = np.array(parts[1:9], dtype=float).reshape(4, 2)
        corners[:, 0] *= img_w
        corners[:, 1] *= img_h
        gt.append((cls_id, corners))
    return gt


def evaluate(weights_path: str | Path,
             data_yaml: str | Path,
             conf: float = 0.25,
             iou_threshold: float = 0.5) -> EvaluationResult:
    """Evaluate a trained checkpoint on the ``val`` split of a dataset."""
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    root = Path(cfg.get("path", Path(data_yaml).parent))
    val_imgs_dir = (Path(cfg["val"]) if Path(cfg["val"]).is_absolute()
                    else root / cfg["val"])
    val_lbls_dir = Path(str(val_imgs_dir).replace("/images", "/labels"))

    id2name = {int(k): v for k, v in model.names.items()}
    n_cls = len(id2name)
    BG = n_cls
    cm = np.zeros((n_cls + 1, n_cls + 1), dtype=int)

    img_files = sorted([p for p in val_imgs_dir.iterdir()
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    for img_path in img_files:
        img = cv.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        gt_list = _read_gt(val_lbls_dir / (img_path.stem + ".txt"), iw, ih)
        res = model.predict(img, conf=conf, verbose=False)[0]
        preds: list[tuple[int, np.ndarray, float]] = []
        if res.obb is not None:
            for box in res.obb:
                corners = box.xyxyxyxy.cpu().numpy().reshape(4, 2).astype(float)
                preds.append((int(box.cls[0]), corners, float(box.conf[0])))

        matched_gt: set[int] = set()
        for pi, (pcls, pcorn, _) in sorted(enumerate(preds),
                                           key=lambda x: -x[1][2]):
            best_iou, best_gi = 0.0, -1
            for gi, (_, gcorn) in enumerate(gt_list):
                if gi in matched_gt:
                    continue
                iou = _bbox_iou(pcorn, gcorn)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= iou_threshold:
                cm[gt_list[best_gi][0], pcls] += 1
                matched_gt.add(best_gi)
            else:
                cm[BG, pcls] += 1
        for gi, (gcls, _) in enumerate(gt_list):
            if gi not in matched_gt:
                cm[gcls, BG] += 1

    tp = np.array([cm[i, i] for i in range(n_cls)], dtype=float)
    fp = np.array([cm[:, i].sum() - cm[i, i] for i in range(n_cls)], dtype=float)
    fn = np.array([cm[i, :].sum() - cm[i, i] for i in range(n_cls)], dtype=float)
    prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    rec  = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f1   = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)

    return EvaluationResult(
        confusion_matrix=cm,
        precision=prec,
        recall=rec,
        f1=f1,
        class_names=[id2name[i] for i in range(n_cls)],
    )


def plot_evaluation(result: EvaluationResult,
                    show_normalized: bool = True) -> None:
    """Render the confusion matrix and per-class P/R/F1 bars."""
    cm = result.confusion_matrix
    n = len(result.class_names)
    tick = result.class_names + ["bg"]

    fig, axes = plt.subplots(1, 2 if show_normalized else 1,
                             figsize=(18 if show_normalized else 9, 7))
    panels = []
    if show_normalized:
        panels.append((axes[0], cm, "Confusion matrix (raw)", None))
        norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        panels.append((axes[1], norm, "Confusion matrix (recall-normalised)", 1.0))
    else:
        panels.append((axes, cm, "Confusion matrix (raw)", None))

    for ax, data, title, vmax in panels:
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(n + 1))
        ax.set_yticks(range(n + 1))
        ax.set_xticklabels(tick, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(tick, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground truth")
        ax.set_title(title)
        threshold = (data.max() or 1) * 0.5
        for r in range(n + 1):
            for c in range(n + 1):
                v = data[r, c]
                if v == 0 or (isinstance(v, float) and v < 0.005):
                    continue
                txt = f"{v:.2f}" if isinstance(v, float) else str(v)
                ax.text(c, r, txt, ha="center", va="center", fontsize=8,
                        color="white" if v > threshold else "black")
    plt.tight_layout()
    plt.show()

    # Per-class P/R/F1 bars
    fig2, ax3 = plt.subplots(figsize=(max(9, n * 1.6), 5))
    x = np.arange(n)
    width = 0.24
    for offset, vals, color, lbl in [
        (-1.5 * width, result.precision, "#4C72B0", "Precision"),
        (-0.5 * width, result.recall,    "#DD8452", "Recall"),
        ( 0.5 * width, result.f1,        "#55A868", "F1"),
    ]:
        ax3.bar(x + offset, vals, width, label=lbl, color=color)
    ax3.set_xticks(x)
    ax3.set_xticklabels(result.class_names, rotation=30, ha="right")
    ax3.set_ylim(0, 1.18)
    ax3.set_ylabel("Score")
    ax3.set_title(f"Per-class metrics  (macro F1 = {result.f1.mean():.3f})")
    ax3.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
