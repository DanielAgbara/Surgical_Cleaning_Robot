#!/usr/bin/env python3

import json
from pathlib import Path
import cv2

IMAGE_DIR = Path(
    "data/dataset/tray_raw/images"
)

IMAGE_EXTENSIONS = [
    "*.png",
    "*.jpg",
    "*.jpeg",
]

count = 0

for pattern in IMAGE_EXTENSIONS:

    for image_path in IMAGE_DIR.glob(pattern):

        json_path = image_path.with_suffix(".json")

        # Skip already annotated images
        if json_path.exists():
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        h, w = image.shape[:2]

        labelme_json = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [],
            "imagePath": image_path.name,
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w,
        }

        with open(json_path, "w") as f:
            json.dump(labelme_json, f, indent=2)

        count += 1

print(f"Created {count} empty annotations.")