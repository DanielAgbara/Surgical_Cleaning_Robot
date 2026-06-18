#!/usr/bin/env python3

"""
train_maskrcnn_tray.py

Purpose:
    Fine-tune a COCO-pretrained Mask R-CNN model using Detectron2
    for one custom class:

        tray

Expected dataset folder:

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

Output folder:

    Fine_Tuning/
    └── output/
        └── maskrcnn_tray/
            ├── model_final.pth
            ├── config.yaml
            ├── metrics.json
            └── ...
"""

import os
from pathlib import Path

import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultPredictor


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATASET_DIR = ROOT / "data" / "dataset" / "tray_coco"

TRAIN_IMAGE_DIR = DATASET_DIR / "train" / "images"
VAL_IMAGE_DIR = DATASET_DIR / "val" / "images"

TRAIN_JSON = DATASET_DIR / "train" / "annotations.json"
VAL_JSON = DATASET_DIR / "val" / "annotations.json"

OUTPUT_DIR = ROOT / "output" / "maskrcnn_tray"


# --------------------------------------------------
# Dataset names used internally by Detectron2
# --------------------------------------------------

TRAIN_DATASET_NAME = "tray_train"
VAL_DATASET_NAME = "tray_val"

CLASS_NAMES = ["tray"]


# --------------------------------------------------
# Training settings
# --------------------------------------------------

# Pretrained Mask R-CNN model from Detectron2 model zoo.
# This starts from COCO-pretrained weights.
MODEL_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

# Number of object classes in your custom dataset.
# You only have one class: tray.
NUM_CLASSES = 1

# For your 120-image pilot dataset, this is a reasonable first run.
MAX_ITER = 4000

# Smaller learning rate is safer for fine-tuning on a small custom dataset.
BASE_LR = 0.0001

# Number of images per training batch.
# Use 2 for RTX 2080 Ti / moderate GPU memory.
IMS_PER_BATCH = 2

# Number of sampled RoIs per image used by ROI heads.
# 128 is good for small datasets.
ROI_BATCH_SIZE_PER_IMAGE = 256

# Save model checkpoints every N iterations.
CHECKPOINT_PERIOD = 500

# Evaluate on validation set every N iterations.
EVAL_PERIOD = 500

# Detection threshold used later for inference/testing.
SCORE_THRESH_TEST = 0.75


def check_dataset_paths():
    """
    Verify that the COCO dataset exists before training.

    This prevents Detectron2 from failing later with a less clear error.
    """

    required_paths = [
        TRAIN_IMAGE_DIR,
        VAL_IMAGE_DIR,
        TRAIN_JSON,
        VAL_JSON,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing required dataset path: {path}")

    print("[INFO] Dataset paths found.")
    print(f"[INFO] Train images: {TRAIN_IMAGE_DIR}")
    print(f"[INFO] Train JSON:   {TRAIN_JSON}")
    print(f"[INFO] Val images:   {VAL_IMAGE_DIR}")
    print(f"[INFO] Val JSON:     {VAL_JSON}")


def register_datasets():
    """
    Register the train and validation datasets with Detectron2.

    Detectron2 does not automatically know where your dataset is.
    You must register:
        - dataset name
        - annotation JSON
        - image folder

    After registration, you can refer to the dataset by name:
        tray_train
        tray_val
    """

    # If you rerun this script in an interactive session,
    # old registrations may still exist.
    # Remove them to avoid duplicate registration errors.
    for dataset_name in [TRAIN_DATASET_NAME, VAL_DATASET_NAME]:
        if dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_name)
            MetadataCatalog.remove(dataset_name)

    register_coco_instances(
        TRAIN_DATASET_NAME,
        {},
        str(TRAIN_JSON),
        str(TRAIN_IMAGE_DIR),
    )

    register_coco_instances(
        VAL_DATASET_NAME,
        {},
        str(VAL_JSON),
        str(VAL_IMAGE_DIR),
    )

    MetadataCatalog.get(TRAIN_DATASET_NAME).set(thing_classes=CLASS_NAMES)
    MetadataCatalog.get(VAL_DATASET_NAME).set(thing_classes=CLASS_NAMES)

    print("[INFO] Registered Detectron2 datasets:")
    print(f"       {TRAIN_DATASET_NAME}")
    print(f"       {VAL_DATASET_NAME}")


def build_config():
    """
    Build the Detectron2 training configuration.

    This tells Detectron2:
        - which pretrained model to use
        - where the dataset is
        - how many classes to predict
        - where to save output
        - learning rate, iterations, batch size, etc.
    """

    cfg = get_cfg()

    # Load base Mask R-CNN config.
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG))

    # Assign your custom dataset names.
    cfg.DATASETS.TRAIN = (TRAIN_DATASET_NAME,)
    cfg.DATASETS.TEST = (VAL_DATASET_NAME,)

    # Number of CPU workers for loading data.
    # You can increase this if your system handles it well.
    cfg.DATALOADER.NUM_WORKERS = 2

    # Start from COCO-pretrained Mask R-CNN weights.
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_CONFIG)

    # Training batch size.
    cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH

    # Fine-tuning learning rate.
    cfg.SOLVER.BASE_LR = BASE_LR

    # Maximum number of training iterations.
    cfg.SOLVER.MAX_ITER = MAX_ITER

    # No step decay for this first simple pilot run.
    cfg.SOLVER.STEPS = []

    # Checkpoint and validation evaluation periods.
    cfg.SOLVER.CHECKPOINT_PERIOD = CHECKPOINT_PERIOD
    cfg.TEST.EVAL_PERIOD = EVAL_PERIOD

    # ROI head batch size.
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = ROI_BATCH_SIZE_PER_IMAGE

    # Very important:
    # Replace COCO's 80-class prediction head with your 1-class tray head.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES

    # Score threshold used by DefaultPredictor after training.
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESH_TEST

    # Save all training outputs here.
    cfg.OUTPUT_DIR = str(OUTPUT_DIR)

    # Use GPU if available, otherwise CPU.
    # Training on CPU will be extremely slow.
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
        print("[INFO] Using CUDA GPU for training.")
    else:
        cfg.MODEL.DEVICE = "cpu"
        print("[WARNING] CUDA not available. Training on CPU will be very slow.")

    return cfg


class TrayTrainer(DefaultTrainer):
    """
    Custom trainer class.

    DefaultTrainer already handles most training logic.
    We override build_evaluator so Detectron2 knows how to evaluate
    the validation set using COCO-style metrics.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = str(Path(cfg.OUTPUT_DIR) / "eval")

        return COCOEvaluator(
            dataset_name,
            cfg,
            False,
            output_dir=output_folder,
        )


def save_config(cfg):
    """
    Save the final training configuration to config.yaml.

    This is useful because later you can see exactly what settings
    produced a given model_final.pth checkpoint.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config_path = OUTPUT_DIR / "config.yaml"

    with open(config_path, "w") as f:
        f.write(cfg.dump())

    print(f"[INFO] Saved config to: {config_path}")


def train_model(cfg):
    """
    Start training Mask R-CNN.

    resume_or_load(resume=False):
        Starts fresh from the pretrained COCO weights.

    If you want to continue from an interrupted run later,
    you can change resume=False to resume=True.
    """

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = TrayTrainer(cfg)

    trainer.resume_or_load(resume=False)

    print("[INFO] Starting training...")
    trainer.train()

    print("[DONE] Training complete.")
    print(f"[INFO] Final model saved in: {cfg.OUTPUT_DIR}")


def main():
    """
    Full training pipeline:

        1. Check that train/val COCO dataset exists.
        2. Register the dataset with Detectron2.
        3. Build Mask R-CNN training config.
        4. Save config for future inference.
        5. Train model.
    """

    check_dataset_paths()

    register_datasets()

    cfg = build_config()

    save_config(cfg)

    train_model(cfg)


if __name__ == "__main__":
    main()