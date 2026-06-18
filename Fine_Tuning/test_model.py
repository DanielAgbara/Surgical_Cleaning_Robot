#!/usr/bin/env python3

"""
test_tray_model_zed.py

Simple live test for a fine-tuned Mask R-CNN tray model.

Loads:
    output/maskrcnn_tray/config.yaml
    output/maskrcnn_tray/model_final.pth

Runs inference on live ZED images.

Controls:
    q = quit
"""

from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl
import torch

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

MODEL_DIR = ROOT / "output" / "maskrcnn_tray"

CONFIG_PATH = MODEL_DIR / "config.yaml"
MODEL_PATH = MODEL_DIR / "model_0000499.pth"


# --------------------------------------------------
# Detection settings
# --------------------------------------------------

SCORE_THRESHOLD = 0.70

CLASS_NAMES = ["tray"]

WINDOW_NAME = "Tray Detection"


# --------------------------------------------------
# Build predictor
# --------------------------------------------------

def build_predictor():

    cfg = get_cfg()

    cfg.merge_from_file(str(CONFIG_PATH))

    cfg.MODEL.WEIGHTS = str(MODEL_PATH)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = SCORE_THRESHOLD

    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
        print("[INFO] Using CUDA")
    else:
        cfg.MODEL.DEVICE = "cpu"
        print("[WARNING] Using CPU")

    return DefaultPredictor(cfg)


# --------------------------------------------------
# Open ZED
# --------------------------------------------------

def open_zed():

    zed = sl.Camera()

    init_params = sl.InitParameters()

    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30

    init_params.depth_mode = sl.DEPTH_MODE.NONE

    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open ZED: {status}")

    return zed


# --------------------------------------------------
# Draw detections
# --------------------------------------------------

def draw_predictions(frame, outputs):

    instances = outputs["instances"].to("cpu")

    if len(instances) == 0:
        return frame

    scores = instances.scores.numpy()
    masks = instances.pred_masks.numpy()

    for score, mask in zip(scores, masks):

        mask_uint8 = mask.astype(np.uint8)

        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        overlay = frame.copy()

        overlay[mask] = (0, 255, 0)

        frame = cv2.addWeighted(
            overlay,
            0.30,
            frame,
            0.70,
            0,
        )

        cv2.drawContours(
            frame,
            contours,
            -1,
            (0, 255, 0),
            2,
        )

        largest = max(contours, key=cv2.contourArea)

        M = cv2.moments(largest)

        if M["m00"] > 0:

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(
                frame,
                (cx, cy),
                6,
                (0, 0, 255),
                -1,
            )

            cv2.putText(
                frame,
                f"tray {score:.2f}",
                (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

    return frame


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    predictor = build_predictor()

    zed = open_zed()

    image = sl.Mat()

    print("[INFO] Press q to quit")

    try:

        while True:

            if zed.grab() == sl.ERROR_CODE.SUCCESS:

                zed.retrieve_image(
                    image,
                    sl.VIEW.LEFT,
                )

                frame = image.get_data()

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGRA2BGR,
                )

                outputs = predictor(frame)

                display = draw_predictions(
                    frame.copy(),
                    outputs,
                )

                cv2.imshow(
                    WINDOW_NAME,
                    display,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

    finally:

        zed.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()