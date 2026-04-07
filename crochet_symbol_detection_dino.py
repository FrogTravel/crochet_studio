"""
Crochet symbol detection using DETR-family transformer detectors (HuggingFace).
Works with any version of torchvision.

Install (one line, won't break your existing PyTorch):
    pip install transformers

Model options — set MODEL_NAME below:
    "facebook/detr-resnet-50"                -- base DETR, ResNet-50 backbone
    "facebook/detr-resnet-101"               -- stronger, ResNet-101 backbone
    "microsoft/conditional-detr-resnet-50"   -- Conditional DETR: closer to DINO,
                                                faster convergence, better on small objects

For the exact DINO from the music recognition paper (Swin-L backbone, mmdetection):
    see the note at the bottom of this file.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForObjectDetection,
    AutoImageProcessor,
)
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ================================================================
#  CONFIG
# ================================================================
DATA_DIR        = "data/project2_img"
ANNOTATION_FILE = os.path.join(DATA_DIR, "result.json")
TEST_IMG_DIR    = "data/test_img"

# Model backbone — swap by changing MODEL_NAME only
MODEL_NAME      = "microsoft/conditional-detr-resnet-50"

BATCH_SIZE      = 1       # transformer detectors are memory-hungry; start at 1
EPOCHS          = 20
LR              = 1e-4    # transformer head / decoder
LR_BACKBONE     = 1e-5    # pretrained backbone needs a much smaller LR
WEIGHT_DECAY    = 1e-4
GRAD_CLIP_NORM  = 0.1     # essential for transformer training stability
SCORE_THRESHOLD = 0.0     # DETR outputs 300 queries — threshold filters out background ones
                          # raise to 0.7 if you still see too many overlapping boxes

DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "crochet_detr.pth"

CLASS_COLORS    = ["orangered", "royalblue", "limegreen", "gold", "violet"]


# ================================================================
#  DATASET
# ================================================================
class CrochetDataset(Dataset):
    """
    Reads Label Studio COCO-format exports.
    Returns (PIL image, COCO annotation dict) — the processor handles the rest.
    """
    def __init__(self, root, annotation_file):
        self.root = root
        with open(annotation_file) as f:
            coco = json.load(f)

        self.images     = coco["images"]
        self.categories = coco["categories"]

        self.ann_by_image = {}
        for ann in coco["annotations"]:
            self.ann_by_image.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        filename = os.path.basename(img_info["file_name"])
        img_path = os.path.join(self.root, "images", filename)
        img      = Image.open(img_path).convert("RGB")

        anns = self.ann_by_image.get(img_info["id"], [])

        # HuggingFace DETR uses 0-indexed category_ids (no background offset needed)
        annotation = {
            "image_id": img_info["id"],
            "annotations": [
                {
                    "id":          ann["id"],
                    "image_id":    ann["image_id"],
                    "category_id": ann["category_id"],   # stays 0-indexed
                    "bbox":        ann["bbox"],           # [x, y, w, h] pixels
                    "area":        ann["bbox"][2] * ann["bbox"][3],
                    "iscrowd":     0,
                }
                for ann in anns
            ],
        }

        return img, annotation


# ================================================================
#  COLLATE — processor converts images + COCO annotations into tensors
# ================================================================
def make_collate_fn(processor):
    def collate_fn(batch):
        images, annotations = zip(*batch)
        # processor resizes, normalises, pads, and converts COCO [x,y,w,h]
        # bboxes to normalised [cx, cy, w, h] automatically
        encoding = processor(
            images=list(images),
            annotations=list(annotations),
            return_tensors="pt",
        )
        return encoding
    return collate_fn


# ================================================================
#  MODEL
# ================================================================
def get_model(id2label, label2id):
    """
    Load a pretrained DETR-family model and replace the classification head
    for our custom classes.  `ignore_mismatched_sizes=True` handles the
    91 (COCO) → N (crochet) class swap automatically.
    """
    model = AutoModelForObjectDetection.from_pretrained(
        MODEL_NAME,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    return model


# ================================================================
#  OPTIMIZER  (lower LR for backbone, higher for transformer head)
# ================================================================
def build_optimizer(model):
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params,     "lr": LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )


# ================================================================
#  TRAINING
# ================================================================
def train():
    with open(ANNOTATION_FILE) as f:
        coco = json.load(f)

    # HuggingFace models use id2label / label2id dicts
    id2label = {cat["id"]: cat["name"] for cat in coco["categories"]}
    label2id = {cat["name"]: cat["id"] for cat in coco["categories"]}

    print(f"[data]   classes: {list(id2label.values())}")
    print(f"[model]  {MODEL_NAME}")
    print(f"[device] {DEVICE}")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    dataset   = CrochetDataset(DATA_DIR, ANNOTATION_FILE)
    loader    = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=make_collate_fn(processor),
    )

    model     = get_model(id2label, label2id)
    model.to(DEVICE)
    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        total_loss     = 0.0
        loss_breakdown = {}

        for batch in loader:
            batch   = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                       for k, v in batch.items()}

            # move label tensors to device (they're a list of dicts)
            if "labels" in batch:
                batch["labels"] = [
                    {k2: v2.to(DEVICE) for k2, v2 in lbl.items()}
                    for lbl in batch["labels"]
                ]

            outputs = model(**batch)
            loss    = outputs.loss

            for k, v in outputs.loss_dict.items():
                loss_breakdown[k] = loss_breakdown.get(k, 0.0) + v.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(loader)
        lr_head  = optimizer.param_groups[1]["lr"]
        bd       = "  ".join(
            f"{k}={v / len(loader):.3f}" for k, v in loss_breakdown.items()
        )

        print(f"Epoch {epoch + 1:3d}/{EPOCHS}  loss={avg_loss:.4f}  "
              f"lr={lr_head:.1e}  | {bd}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ↳ checkpoint saved")

    print(f"\nDone. Best loss {best_loss:.4f}  →  {MODEL_SAVE_PATH}")


# ================================================================
#  SHARED HELPER — load model + processor for inference
# ================================================================
def _load_for_inference(model_path):
    with open(ANNOTATION_FILE) as f:
        coco = json.load(f)
    id2label  = {cat["id"]: cat["name"] for cat in coco["categories"]}
    label2id  = {cat["name"]: cat["id"] for cat in coco["categories"]}
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model     = get_model(id2label, label2id)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, processor, id2label


def _predict(model, processor, img_pil):
    """Run inference on a single PIL image, return post-processed results."""
    encoding = processor(images=img_pil, return_tensors="pt")
    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = model(**encoding)
    # target_sizes is (H, W) of the original image
    target_sizes = torch.tensor([[img_pil.height, img_pil.width]]).to(DEVICE)
    return processor.post_process_object_detection(
        outputs, threshold=SCORE_THRESHOLD, target_sizes=target_sizes
    )[0]


# ================================================================
#  VISUALIZATION — training images (GT vs predictions side by side)
# ================================================================
def visualize_predictions(model_path=MODEL_SAVE_PATH, num_images=None):
    model, processor, id2label = _load_for_inference(model_path)
    dataset = CrochetDataset(DATA_DIR, ANNOTATION_FILE)

    n = len(dataset) if num_images is None else min(num_images, len(dataset))

    for i in range(n):
        img_pil, annotation = dataset[i]
        results = _predict(model, processor, img_pil)

        fig, axes = plt.subplots(1, 2, figsize=(18, 9))

        # --- left: ground truth ---
        axes[0].imshow(img_pil)
        axes[0].set_title("Ground Truth", fontsize=12)
        for ann in annotation["annotations"]:
            x, y, w, h = ann["bbox"]
            cat_id = ann["category_id"]
            c      = CLASS_COLORS[cat_id % len(CLASS_COLORS)]
            axes[0].add_patch(patches.Rectangle(
                (x, y), w, h, linewidth=1.5, edgecolor=c, facecolor="none"))
            axes[0].text(x, y - 3, id2label.get(cat_id, "?"),
                         color=c, fontsize=7, fontweight="bold")

        # --- right: predictions ---
        axes[1].imshow(img_pil)
        axes[1].set_title(f"Predictions  (threshold={SCORE_THRESHOLD})", fontsize=12)
        for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
            x1, y1, x2, y2 = box.cpu().tolist()
            cat_id = label.item()
            c      = CLASS_COLORS[cat_id % len(CLASS_COLORS)]
            axes[1].add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor=c, facecolor="none"))
            name = id2label.get(cat_id, f"cls{cat_id}")
            axes[1].text(x1, y1 - 3, f"{name} {score:.2f}",
                         color=c, fontsize=7, fontweight="bold")

        plt.tight_layout()
        plt.show()


# ================================================================
#  VISUALIZATION — unseen test images
# ================================================================
def visualize_test_predictions(model_path=MODEL_SAVE_PATH, num_images=3):
    model, processor, id2label = _load_for_inference(model_path)

    for img_name in sorted(os.listdir(TEST_IMG_DIR))[:num_images]:
        img_pil = Image.open(os.path.join(TEST_IMG_DIR, img_name)).convert("RGB")
        results = _predict(model, processor, img_pil)

        fig, ax = plt.subplots(1, figsize=(14, 10))
        ax.imshow(img_pil)
        ax.set_title(img_name, fontsize=12)

        n_shown = 0
        for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
            x1, y1, x2, y2 = box.cpu().tolist()
            cat_id = label.item()
            c      = CLASS_COLORS[cat_id % len(CLASS_COLORS)]
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=c, facecolor="none"))
            name = id2label.get(cat_id, f"cls{cat_id}")
            ax.text(x1, y1 - 3, f"{name} {score:.2f}",
                    color=c, fontsize=8, fontweight="bold")
            n_shown += 1

        ax.set_xlabel(f"{n_shown} detections above threshold={SCORE_THRESHOLD}")
        plt.tight_layout()
        plt.show()


# ================================================================
#  ENTRY POINT
# ================================================================
if __name__ == "__main__":
    train()
    visualize_predictions()
    visualize_test_predictions()


# ================================================================
#  UPGRADING TO THE EXACT DINO FROM THE MUSIC PAPER
# ================================================================
"""
The music paper uses DINO (IDEACVR/DINO) with Swin-L or FocalNet backbones,
trained via mmdetection.  The model in this script (Conditional DETR) uses the
same core idea — transformer decoder + Hungarian matching — but without
denoising pre-training (the DN-DETR component that makes DINO especially good
at many small overlapping objects).

When to upgrade:
  - You have 50+ labeled images
  - Conditional DETR's accuracy has plateaued

How to upgrade:
    pip install mmengine "mmcv>=2.0.0" mmdet

    Then use config:
      mmdet/configs/dino/dino-4scale_r50_8xb2-12e_coco.py   (ResNet-50)
      mmdet/configs/dino/dino-5scale_swin-l_8xb2-36e_coco.py (Swin-L, as in paper)

    Key config changes for your dataset:
      model.bbox_head.num_classes = 3
      dataset_type = 'CocoDataset'
      metainfo = dict(classes=('chain', 'double crochet', 'half double crochet group 3'))
      data_root = 'data/project1_2img/'

MODEL_NAME swap guide (no other code changes needed):
    "facebook/detr-resnet-50"               fast baseline, 32M params
    "facebook/detr-resnet-101"              stronger baseline, 51M params
    "microsoft/conditional-detr-resnet-50"  current default, faster convergence
    "microsoft/conditional-detr-resnet-50"  → upgrade to mmdet DINO when ready
"""
