import os
import json
import random
from PIL import Image, ImageDraw

"""
This script helps you create a simple object detection dataset
for crochet granny square patterns.

It assumes:
- You have images of patterns
- You manually (or semi-manually) define bounding boxes

It outputs COCO-style annotations.
"""

# ---------------- CONFIG ----------------
IMAGE_DIR = "data/raw/easy"  # folder with your images
OUTPUT_JSON = "annotations.json"
CATEGORIES = [
    {"id": 1, "name": "granny_square"},
    {"id": 2, "name": "flower_center"},
    {"id": 3, "name": "border"}
]

# ----------------------------------------


def load_images(image_dir):
    images = []
    for i, filename in enumerate(os.listdir(image_dir)):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(image_dir, filename)
            with Image.open(path) as img:
                width, height = img.size
            images.append({
                "id": i,
                "file_name": filename,
                "width": width,
                "height": height
            })
    return images


def manual_annotation_tool(image_path):
    """
    VERY simple annotation helper.
    You will input bounding boxes manually.
    """
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    print(f"Annotating: {image_path}")
    print("Enter boxes as: x_min y_min width height category_id")
    print("Type 'done' when finished")

    annotations = []

    while True:
        user_input = input("> ")
        if user_input.lower() == "done":
            break

        try:
            x, y, w, h, cat = map(int, user_input.split())
            draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
            annotations.append({
                "bbox": [x, y, w, h],
                "category_id": cat
            })
        except Exception as e:
            print("Invalid input. Try again.")

    img.show()
    return annotations


def create_dataset():
    images = load_images(IMAGE_DIR)
    annotations = []

    ann_id = 0

    for img_info in images:
        image_path = os.path.join(IMAGE_DIR, img_info["file_name"])

        anns = manual_annotation_tool(image_path)

        for ann in anns:
            annotations.append({
                "id": ann_id,
                "image_id": img_info["id"],
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann["bbox"][2] * ann["bbox"][3],
                "iscrowd": 0
            })
            ann_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(coco_format, f, indent=2)

    print(f"Dataset saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    create_dataset()
