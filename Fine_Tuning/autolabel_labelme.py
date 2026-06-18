#!/usr/bin/env python3

"""
autolabel_labelme.py

Purpose:
    Use a trained Detectron2 Mask R-CNN model to automatically create
    LabelMe JSON annotations for raw images that do not already have annotations.

Why this is useful:
    You can manually annotate a small dataset, train a first model,
    then use that model to pre-label the remaining images.

This version is adaptable:
    - Works for tray now.
    - Can be reused later for other classes.
    - Lets you choose the class name to save into LabelMe JSON.
    - Lets you choose the raw image folder.
    - Lets you choose the model output folder.
    - Can either skip already annotated images or add a new class to
      existing LabelMe JSON files.

Example for your current tray dataset:

    python autolabel_labelme.py \
        --raw-dir data/dataset/tray_raw/images \
        --model-dir output/maskrcnn_tray \
        --label tray \
        --mode missing_json

Example later for another class, such as sponge:

    python autolabel_labelme.py \
        --raw-dir data/dataset/sponge_raw/images \
        --model-dir output/maskrcnn_sponge \
        --label sponge \
        --mode missing_json

Example to add a class into existing LabelMe JSON files:

    python autolabel_labelme.py \
        --raw-dir data/dataset/tray_raw/images \
        --model-dir output/maskrcnn_multi \
        --label sponge \
        --class-index 1 \
        --mode missing_label
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


# --------------------------------------------------
# Defaults
# --------------------------------------------------

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]

DEFAULT_SCORE_THRESHOLD = 0.70
DEFAULT_MIN_CONTOUR_AREA = 1000
DEFAULT_POLYGON_APPROX_EPSILON_RATIO = 0.003


def parse_args():
    """
    Read command-line arguments.

    This makes the script reusable for different datasets and classes.

    Important arguments:
        --raw-dir:
            Folder containing raw images.

        --model-dir:
            Folder containing config.yaml and model_final.pth.

        --label:
            LabelMe class name to save, for example "tray".

        --class-index:
            Detectron2 predicted class index.
            For a one-class tray model, this is 0.

        --mode:
            missing_json:
                Only annotate images with no JSON file at all.

            missing_label:
                If a JSON exists, add the label only if that label is missing.
                Useful later for multi-class datasets.
    """

    parser = argparse.ArgumentParser(
        description="Auto-label raw images with a Detectron2 Mask R-CNN model and save LabelMe JSON files."
    )

    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/dataset/tray_raw/images",
        help="Folder containing raw images and LabelMe JSON files.",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="output/maskrcnn_tray",
        help="Folder containing Detectron2 config.yaml and model_final.pth.",
    )

    parser.add_argument(
        "--label",
        type=str,
        default="tray",
        help="LabelMe class name to write into JSON, e.g. tray, sponge, brush.",
    )

    parser.add_argument(
        "--class-index",
        type=int,
        default=0,
        help="Detectron2 predicted class index to use. For one-class models, use 0.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="missing_json",
        choices=["missing_json", "missing_label", "overwrite"],
        help=(
            "missing_json: annotate only images with no JSON. "
            "missing_label: add label if JSON exists but label is missing. "
            "overwrite: replace/create JSON with this prediction."
        ),
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help="Minimum confidence score required to save prediction.",
    )

    parser.add_argument(
        "--min-contour-area",
        type=float,
        default=DEFAULT_MIN_CONTOUR_AREA,
        help="Minimum mask contour area in pixels.",
    )

    parser.add_argument(
        "--epsilon-ratio",
        type=float,
        default=DEFAULT_POLYGON_APPROX_EPSILON_RATIO,
        help="Polygon simplification strength. Smaller = tighter polygon, more points.",
    )

    parser.add_argument(
        "--config-name",
        type=str,
        default="config.yaml",
        help="Config filename inside model-dir.",
    )

    parser.add_argument(
        "--weights-name",
        type=str,
        default="model_final.pth",
        help="Weights filename inside model-dir.",
    )

    return parser.parse_args()


def resolve_path(path_string):
    """
    Convert a string path to an absolute Path.

    This lets you run the script from Fine_Tuning with relative paths like:

        data/dataset/tray_raw/images

    or with absolute paths.
    """

    path = Path(path_string).expanduser()

    if path.is_absolute():
        return path

    return Path(__file__).resolve().parent / path


def build_predictor(model_dir, config_name, weights_name, score_threshold):
    """
    Load the trained Detectron2 Mask R-CNN model.

    Expected files:

        model_dir/
        ├── config.yaml
        └── model_final.pth

    The config contains the model architecture and dataset settings.
    The .pth file contains the learned weights.
    """

    model_dir = Path(model_dir)
    #load the files
    config_path = model_dir / config_name
    weights_path = model_dir / weights_name

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Missing model weights: {weights_path}")

    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))

    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold

    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
        print("[INFO] Using CUDA for inference.")
    else:
        cfg.MODEL.DEVICE = "cpu"
        print("[WARNING] CUDA not available. Using CPU.")

    predictor = DefaultPredictor(cfg)

    return predictor


def get_image_paths(raw_dir):
    """
    Find all images in the raw image folder.

    Supported extensions:
        .png
        .jpg
        .jpeg
    """

    image_paths = []

    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(raw_dir.glob(f"*{ext}"))

    return sorted(image_paths)


def load_labelme_json(json_path):
    """
    Load an existing LabelMe JSON file.

    Returns None if the JSON does not exist.
    """

    if not json_path.exists():
        return None

    with open(json_path, "r") as f:
        return json.load(f)


def label_exists(labelme_data, label_name):
    """
    Check whether a LabelMe JSON already contains a shape
    with the requested label.

    This is used in missing_label mode.
    """

    if labelme_data is None:
        return False

    for shape in labelme_data.get("shapes", []):
        if shape.get("label") == label_name:
            return True

    return False


def should_process_image(image_path, label_name, mode):
    """
    Decide whether the script should auto-label this image.

    Modes:

        missing_json:
            Process only if image has no JSON file.

        missing_label:
            Process if JSON is missing OR if JSON does not contain this label.

        overwrite:
            Always process image and overwrite/create JSON.
    """

    json_path = image_path.with_suffix(".json")
    labelme_data = load_labelme_json(json_path)

    if mode == "missing_json":
        return labelme_data is None

    if mode == "missing_label":
        return not label_exists(labelme_data, label_name)

    if mode == "overwrite":
        return True

    raise ValueError(f"Unknown mode: {mode}")


def mask_to_polygon(mask, min_contour_area, epsilon_ratio):
    """
    Convert a binary Mask R-CNN mask into a LabelMe polygon.

    Steps:
        1. Convert mask to uint8.
        2. Find object contours.
        3. Keep largest contour.
        4. Simplify contour into polygon points.

    Detectron2 mask:
        H x W boolean array

    LabelMe polygon:
        [[x1, y1], [x2, y2], ...]
    """

    mask_uint8 = mask.astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    if area < min_contour_area:
        return None

    perimeter = cv2.arcLength(largest_contour, True)
    epsilon = epsilon_ratio * perimeter

    approx = cv2.approxPolyDP(
        largest_contour,
        epsilon,
        True,
    )

    points = approx.reshape(-1, 2).astype(float).tolist()

    if len(points) < 3:
        return None

    return points


def make_empty_labelme_json(image_path, image_shape):
    """
    Create a blank LabelMe JSON structure.

    This is used when an image has no existing JSON file.
    """

    height, width = image_shape[:2]

    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": [],
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def make_labelme_shape(label_name, polygon_points, score):
    """
    Create one LabelMe polygon shape.

    The model confidence score is stored in the description field.
    That way, when reviewing in LabelMe, you can know it was auto-generated.
    """

    return {
        "label": label_name,
        "points": polygon_points,
        "group_id": None,
        "description": f"auto_labeled_score={score:.3f}",
        "shape_type": "polygon",
        "flags": {},
        "mask": None,
    }


def select_prediction(instances, class_index):
    """
    Select the best prediction for the requested class.

    For your current tray model:
        class_index = 0

    If later you train a multi-class model:
        class_index = 0 might be tray
        class_index = 1 might be sponge
        class_index = 2 might be brush

    This function:
        1. Filters predictions by class_index.
        2. Keeps the highest-confidence prediction.
    """

    if len(instances) == 0:
        return None, None

    pred_classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    masks = instances.pred_masks.numpy()

    matching_indices = np.where(pred_classes == class_index)[0]

    if len(matching_indices) == 0:
        return None, None

    best_idx = matching_indices[np.argmax(scores[matching_indices])]

    return masks[best_idx], float(scores[best_idx])


def save_labelme_json(json_path, labelme_data):
    """
    Save LabelMe JSON to disk.
    """

    with open(json_path, "w") as f:
        json.dump(labelme_data, f, indent=2)


def autolabel_one_image(
    predictor,
    image_path,
    label_name,
    class_index,
    mode,
    min_contour_area,
    epsilon_ratio,
):
    """
    Auto-label one image.

    Steps:
        1. Read image.
        2. Run Mask R-CNN.
        3. Select best prediction for target class.
        4. Convert mask to polygon.
        5. Create or update LabelMe JSON.
        6. Save JSON beside image.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[WARNING] Could not read image: {image_path}")
        return False

    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")

    mask, score = select_prediction(instances, class_index)

    if mask is None:
        print(f"[NO DETECTION] {image_path.name}")
        return False

    polygon_points = mask_to_polygon(
        mask=mask,
        min_contour_area=min_contour_area,
        epsilon_ratio=epsilon_ratio,
    )

    if polygon_points is None:
        print(f"[BAD MASK] {image_path.name} | score={score:.3f}")
        return False

    json_path = image_path.with_suffix(".json")

    existing_data = load_labelme_json(json_path)

    if mode == "overwrite" or existing_data is None:
        labelme_data = make_empty_labelme_json(image_path, image.shape)
    else:
        labelme_data = existing_data

    # In overwrite mode, remove old shapes with the same label before adding new one.
    # This prevents duplicate tray polygons.
    if mode == "overwrite":
        labelme_data["shapes"] = [
            shape for shape in labelme_data.get("shapes", [])
            if shape.get("label") != label_name
        ]

    new_shape = make_labelme_shape(
        label_name=label_name,
        polygon_points=polygon_points,
        score=score,
    )

    labelme_data.setdefault("shapes", []).append(new_shape)

    # Make sure image metadata is correct.
    height, width = image.shape[:2]
    labelme_data["imagePath"] = image_path.name
    labelme_data["imageHeight"] = height
    labelme_data["imageWidth"] = width
    labelme_data["imageData"] = None

    save_labelme_json(json_path, labelme_data)

    print(
        f"[AUTO-LABELED] {image_path.name} "
        f"| label={label_name} "
        f"| score={score:.3f} "
        f"| points={len(polygon_points)}"
    )

    return True


def main():
    args = parse_args()

    raw_dir = resolve_path(args.raw_dir)
    model_dir = resolve_path(args.model_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw image directory not found: {raw_dir}")

    print("[INFO] Auto-label settings")
    print(f"       Raw dir:          {raw_dir}")
    print(f"       Model dir:        {model_dir}")
    print(f"       LabelMe label:    {args.label}")
    print(f"       Class index:      {args.class_index}")
    print(f"       Mode:             {args.mode}")
    print(f"       Score threshold:  {args.score_threshold}")
    print(f"       Min contour area: {args.min_contour_area}")
    print(f"       Epsilon ratio:    {args.epsilon_ratio}")
    print()

    predictor = build_predictor(
        model_dir=model_dir,
        config_name=args.config_name,
        weights_name=args.weights_name,
        score_threshold=args.score_threshold,
    )

    image_paths = get_image_paths(raw_dir)

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in: {raw_dir}")

    images_to_process = [
        image_path for image_path in image_paths
        if should_process_image(
            image_path=image_path,
            label_name=args.label,
            mode=args.mode,
        )
    ]

    print(f"[INFO] Total images found:     {len(image_paths)}")
    print(f"[INFO] Images to auto-label:  {len(images_to_process)}")
    print()

    success_count = 0
    fail_count = 0

    for image_path in images_to_process:
        success = autolabel_one_image(
            predictor=predictor,
            image_path=image_path,
            label_name=args.label,
            class_index=args.class_index,
            mode=args.mode,
            min_contour_area=args.min_contour_area,
            epsilon_ratio=args.epsilon_ratio,
        )

        if success:
            success_count += 1
        else:
            fail_count += 1

    print()
    print("[DONE] Auto-labeling complete.")
    print(f"[INFO] Successful: {success_count}")
    print(f"[INFO] Failed:     {fail_count}")
    print()
    print("Review/correct with LabelMe:")
    print(f"  conda activate labelme_env")
    print(f"  labelme {raw_dir}")
    print()
    print("After review:")
    print("  python prepare_tray_dataset.py")
    print("  python train_maskrcnn_tray.py")


if __name__ == "__main__":
    main()