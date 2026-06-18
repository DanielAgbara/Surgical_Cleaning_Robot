#!/usr/bin/env python3

"""
prepare_tray_dataset.py

Purpose:
    Convert LabelMe polygon annotations into COCO instance segmentation format
    for Detectron2 Mask R-CNN fine-tuning.

Expected input folder:

    Fine_Tuning/
    └── data/
        └── dataset/
            └── tray_raw/
                └── images/
                    ├── tray_00000.png
                    ├── tray_00000.json
                    ├── tray_00001.png
                    ├── tray_00001.json
                    └── ...

Expected LabelMe label:

    tray

Output folder:

    Fine_Tuning/
    └── data/
        └── dataset/
            └── tray_coco/
                ├── train/
                │   ├── images/
                │   └── annotations.json
                └── val/
                    ├── images/
                    └── annotations.json
"""

import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# Paths
# --------------------------------------------------

# Folder where this script is located.
# Example: Fine_Tuning/
ROOT = Path(__file__).resolve().parent

# Folder containing your raw images and LabelMe JSON files.
# Example:
#   tray_00000.png
#   tray_00000.json
RAW_DIR = ROOT / "data" / "dataset" / "tray_raw" / "images"

# Output folder for Detectron2-ready COCO dataset.
OUT_DIR = ROOT / "data" / "dataset" / "tray_coco"

# Output train/validation image folders.
TRAIN_IMAGE_DIR = OUT_DIR / "train" / "images"
VAL_IMAGE_DIR = OUT_DIR / "val" / "images"

# Output COCO annotation files.
TRAIN_JSON = OUT_DIR / "train" / "annotations.json"
VAL_JSON = OUT_DIR / "val" / "annotations.json"


# --------------------------------------------------
# Dataset settings
# --------------------------------------------------

# This must exactly match the label you typed in LabelMe.
CLASS_NAME = "tray"

# COCO category IDs usually start at 1.
CATEGORY_ID = 1

# 20% validation, 80% training.
# For 120 images:
#   train = 96
#   val   = 24
VAL_SPLIT = 0.2

# Random seed makes the train/val split repeatable.
SEED = 42

# Accepted image file extensions.
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]


def polygon_area(points):
    """
    Compute the area of a polygon using the shoelace formula.

    LabelMe stores polygon points as:

        [
            [x1, y1],
            [x2, y2],
            [x3, y3],
            ...
        ]

    COCO requires an "area" field for each annotation.
    This function computes the polygon area in pixel^2.
    """

    points = np.asarray(points, dtype=np.float32)

    x = points[:, 0]
    y = points[:, 1]

    area = 0.5 * abs(
        np.dot(x, np.roll(y, -1)) -
        np.dot(y, np.roll(x, -1))
    )

    return float(area)


def find_image_for_json(json_path):
    """
    Find the image file that belongs to a LabelMe JSON file.

    LabelMe JSON usually contains:

        "imagePath": "tray_00000.png"

    This function first tries to use that imagePath.

    If that fails, it tries matching by filename stem:

        tray_00000.json
        tray_00000.png
        tray_00000.jpg
        tray_00000.jpeg

    Returns:
        image_path : Path or None
        data       : loaded LabelMe JSON dictionary
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    # First attempt: use imagePath saved by LabelMe.
    labelme_image_path = data.get("imagePath", "")
    image_path = json_path.parent / labelme_image_path

    if image_path.exists():
        return image_path, data

    # Second attempt: search for an image with same base filename.
    stem = json_path.stem

    for ext in IMAGE_EXTENSIONS:
        candidate = json_path.parent / f"{stem}{ext}"

        if candidate.exists():
            return candidate, data

    # No matching image was found.
    return None, data


def convert_labelme_to_coco(json_files, output_image_dir, output_json_path):
    """
    Convert a list of LabelMe JSON files into one COCO annotations.json file.

    This function is called twice:

        1. once for training files
        2. once for validation files

    Each LabelMe polygon becomes one COCO instance annotation.

    COCO instance segmentation format contains:

        images:
            Information about each image.

        annotations:
            Object masks, bounding boxes, area, category ID.

        categories:
            Class names and class IDs.
    """

    # Create output folders if they do not exist.
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    # Basic COCO file structure.
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": CATEGORY_ID,
                "name": CLASS_NAME,
                "supercategory": "object",
            }
        ],
    }

    # COCO needs unique integer IDs for images and annotations.
    image_id = 1
    annotation_id = 1

    # Process every LabelMe annotation file.
    for json_path in json_files:
        image_path, data = find_image_for_json(json_path)

        # Skip annotation if its image is missing.
        if image_path is None:
            print(f"[WARNING] No matching image found for: {json_path.name}")
            continue

        # Read image to get width and height.
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[WARNING] Could not read image: {image_path}")
            continue

        height, width = image.shape[:2]

        # Copy image into train/images or val/images.
        output_image_path = output_image_dir / image_path.name
        shutil.copy2(image_path, output_image_path)

        # Add image record to COCO JSON.
        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "height": height,
                "width": width,
            }
        )

        valid_shapes = 0

        # LabelMe stores all drawn shapes in data["shapes"].
        for shape in data.get("shapes", []):
            label = shape.get("label", "")

            # Only keep the tray class.
            # This protects you from typos like "Tray", "sink", etc.
            if label != CLASS_NAME:
                print(
                    f"[WARNING] Skipping label '{label}' in {json_path.name}. "
                    f"Expected '{CLASS_NAME}'."
                )
                continue

            # Mask R-CNN needs polygon masks.
            # Rectangle/circle/line annotations are skipped.
            shape_type = shape.get("shape_type", "polygon")

            if shape_type != "polygon":
                print(
                    f"[WARNING] Skipping non-polygon shape "
                    f"'{shape_type}' in {json_path.name}"
                )
                continue

            points = shape.get("points", [])

            # A polygon needs at least 3 points.
            if len(points) < 3:
                print(f"[WARNING] Skipping invalid polygon in {json_path.name}")
                continue

            polygon = np.asarray(points, dtype=np.float32)

            # Bounding box in COCO format:
            #   [x_min, y_min, width, height]
            x_min = float(np.min(polygon[:, 0]))
            y_min = float(np.min(polygon[:, 1]))
            x_max = float(np.max(polygon[:, 0]))
            y_max = float(np.max(polygon[:, 1]))

            bbox = [
                x_min,
                y_min,
                x_max - x_min,
                y_max - y_min,
            ]

            # COCO stores polygon segmentation as a flattened list:
            #
            # LabelMe:
            #   [[x1, y1], [x2, y2], [x3, y3]]
            #
            # COCO:
            #   [x1, y1, x2, y2, x3, y3]
            segmentation = [polygon.flatten().tolist()]

            # Area of mask in pixel^2.
            area = polygon_area(points)

            # Add one object annotation.
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": CATEGORY_ID,
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                }
            )

            annotation_id += 1
            valid_shapes += 1

        # Warn if an image has no usable tray polygon.
        if valid_shapes == 0:
            print(
                f"[WARNING] Image has no valid '{CLASS_NAME}' annotation: "
                f"{json_path.name}"
            )

        image_id += 1

    # Save the COCO JSON file.
    with open(output_json_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"[DONE] Saved: {output_json_path}")
    print(f"[INFO] Images: {len(coco['images'])}")
    print(f"[INFO] Annotations: {len(coco['annotations'])}")


def main():
    """
    Main dataset preparation pipeline.

    Steps:
        1. Find all LabelMe JSON files.
        2. Shuffle them.
        3. Split into train and validation sets.
        4. Delete old COCO output folder.
        5. Convert train annotations.
        6. Convert validation annotations.
    """

    # Make sure raw folder exists.
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw dataset folder not found: {RAW_DIR}")

    # Collect all LabelMe JSON annotation files.
    json_files = sorted(RAW_DIR.glob("*.json"))

    if len(json_files) == 0:
        raise RuntimeError(f"No LabelMe JSON files found in: {RAW_DIR}")

    print(f"[INFO] Found LabelMe JSON files: {len(json_files)}")

    # Shuffle files before splitting.
    # This avoids putting only early-collected images into validation.
    random.seed(SEED)
    random.shuffle(json_files)

    # Compute number of validation examples.
    num_val = max(1, int(len(json_files) * VAL_SPLIT))

    val_files = json_files[:num_val]
    train_files = json_files[num_val:]

    print(f"[INFO] Train annotations: {len(train_files)}")
    print(f"[INFO] Val annotations:   {len(val_files)}")

    # Remove old output to prevent stale images/annotations.
    if OUT_DIR.exists():
        print(f"[INFO] Removing old output folder: {OUT_DIR}")
        shutil.rmtree(OUT_DIR)

    # Convert training split.
    convert_labelme_to_coco(
        json_files=train_files,
        output_image_dir=TRAIN_IMAGE_DIR,
        output_json_path=TRAIN_JSON,
    )

    # Convert validation split.
    convert_labelme_to_coco(
        json_files=val_files,
        output_image_dir=VAL_IMAGE_DIR,
        output_json_path=VAL_JSON,
    )

    print()
    print("[DONE] Tray dataset prepared in COCO format.")
    print(f"Train images: {TRAIN_IMAGE_DIR}")
    print(f"Train JSON:   {TRAIN_JSON}")
    print(f"Val images:   {VAL_IMAGE_DIR}")
    print(f"Val JSON:     {VAL_JSON}")


if __name__ == "__main__":
    main()