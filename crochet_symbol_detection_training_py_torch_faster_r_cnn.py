import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import json
import os
from PIL import Image

"""
Train object detection model on COCO-format dataset (Label Studio export)
for crochet symbols (Option C).
"""

# ---------------- CONFIG ----------------
DATA_DIR = "data/project2_img"  # folder with images + annotations.json
ANNOTATION_FILE = os.path.join(DATA_DIR, "result.json")
BATCH_SIZE = 2
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------

class CrochetDataset(Dataset):
    def __init__(self, root, annotation_file):
        self.root = root
        with open(annotation_file) as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]

        # group annotations by image_id
        self.ann_by_image = {}
        for ann in self.annotations:
            self.ann_by_image.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        filename = os.path.basename(img_info["file_name"])
        img_path = os.path.join(self.root, "images", filename)

        img = Image.open(img_path).convert("RGB")

        anns = self.ann_by_image.get(img_info["id"], [])

        boxes = []
        labels = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"] + 1)  # shift: 0→1, 1→2, 2→3 (0 reserved for background)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        img = F.to_tensor(img)

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ---------------- MODEL ----------------

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # replace head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    return model


# ---------------- TRAIN ----------------

def train():
    dataset = CrochetDataset(DATA_DIR, ANNOTATION_FILE)

    # find number of classes
    with open(ANNOTATION_FILE) as f:
        coco = json.load(f)
    num_classes = len(coco["categories"]) + 1  # + background

    print(f"Found {num_classes-1} classes: {[cat['name'] for cat in coco['categories']]}")

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = get_model(num_classes)
    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for images, targets in data_loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "crochet_detector.pth")
    print("Model saved!")


# ---------------- VISUALIZATION ----------------
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_predictions(model_path="crochet_detector.pth", num_images=3):
    dataset = CrochetDataset(DATA_DIR, ANNOTATION_FILE)

    # load categories
    with open(ANNOTATION_FILE) as f:
        coco = json.load(f)
    id_to_name = {cat["id"] + 1: cat["name"] for cat in coco["categories"]}

    model = get_model(len(coco["categories"]) + 1)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    for i in range(min(num_images, len(dataset))):
        img, _ = dataset[i]
        img_tensor = img.to(DEVICE)

        with torch.no_grad():
            outputs = model([img_tensor])[0]

        fig, ax = plt.subplots(1)
        ax.imshow(img.permute(1, 2, 0))

        boxes = outputs["boxes"].cpu()
        labels = outputs["labels"].cpu()
        scores = outputs["scores"].cpu()

        for box, label, score in zip(boxes, labels, scores):
            if score < 0.5:
                continue

            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)

            class_name = id_to_name.get(label.item(), "unknown")
            ax.text(x1, y1, f"{class_name} {score:.2f}", color='red')

        plt.show()


TEST_IMG_DIR = "data/test_img"

def visualize_test_predictions(model_path="crochet_detector.pth", num_images=3):
    # load categories
    with open(ANNOTATION_FILE) as f:
        coco = json.load(f)

    model = get_model(len(coco["categories"]) + 1)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    for img_name in os.listdir(TEST_IMG_DIR)[:num_images]:
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        img = Image.open(img_path).convert("RGB")
        img_tensor = F.to_tensor(img).to(DEVICE)

        with torch.no_grad():
            outputs = model([img_tensor])[0]

        fig, ax = plt.subplots(1)
        ax.imshow(img)

        boxes = outputs["boxes"].cpu()
        labels = outputs["labels"].cpu()
        scores = outputs["scores"].cpu()

        print(f"Labels: {labels}")

        classes = ['bobble', 'chain_arc', 'dc_line', 'fan_group']
        colors = ['red', 'blue', 'green', 'orange']
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=colors[label.item() % len(colors)], facecolor='none'
            )
            ax.add_patch(rect)

            ax.text(x1, y1, f"Class {classes[label.item() % len(classes)]} {score:.2f}", color=colors[label.item() % len(colors)])

        plt.show()


if __name__ == "__main__":
    #train()
    visualize_predictions()
    visualize_test_predictions()
