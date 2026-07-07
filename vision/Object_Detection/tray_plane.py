#!/usr/bin/env python3

"""
tray_plane_zed.py

Use fine-tuned Mask R-CNN tray model with ZED camera.

Pipeline:
1. Grab ZED RGB image
2. Grab ZED XYZ point cloud
3. Detect tray using fine-tuned Mask R-CNN
4. Extract only 3D points belonging to tray mask pixels
5. Fit RANSAC plane to tray points
6. Display:
   - tray mask overlay
   - tray plane overlay
   - depth map
   - optional Open3D tray plane point cloud

Controls:
    q = quit
    o = open Open3D viewer for current tray plane
"""

from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl
import torch
import open3d as o3d

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


# Current file:
# Surgical_Cleaning_Robot/vision/tray_plane_zed.py
VISION_DIR = Path(__file__).resolve().parent.parent

# Project root:
# Surgical_Cleaning_Robot
ROOT = VISION_DIR.parent

# Fine-tuned model directory:
# Surgical_Cleaning_Robot/Fine_Tuning/output/maskrcnn_tray
MODEL_DIR = ROOT / "Fine_Tuning" / "output" / "maskrcnn_tray"

CONFIG_PATH = MODEL_DIR / "config.yaml"
MODEL_PATH = MODEL_DIR / "model_final.pth"

# --------------------------------------------------
# Settings
# --------------------------------------------------

SCORE_THRESHOLD = 0.70

CLASS_NAMES = ["tray"]

WINDOW_TRAY = "Tray Detection"
WINDOW_PLANE = "Tray Plane Detection"
WINDOW_DEPTH = "ZED Depth Map"

# ZED point cloud is in meters
MIN_DEPTH_M = 0.20
MAX_DEPTH_M = 3.00

# Plane fitting
PLANE_DISTANCE_THRESHOLD_M = 0.015
PLANE_RANSAC_POINTS = 3
PLANE_RANSAC_ITERATIONS = 1000
MAX_TRAY_POINTS = 60000

# Mask cleanup
MASK_KERNEL_SIZE = 5


# --------------------------------------------------
# Detectron2 Predictor
# --------------------------------------------------

def build_predictor():
    """
    Load the fine-tuned tray model.
    """

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

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

    predictor = DefaultPredictor(cfg)

    return predictor


# --------------------------------------------------
# ZED Setup
# --------------------------------------------------

def open_zed():
    """
    Open ZED camera with depth enabled.

    IMPORTANT:
    depth_mode cannot be NONE because we need XYZ points.
    """

    zed = sl.Camera()

    init_params = sl.InitParameters()

    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30

    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open ZED: {status}")

    runtime_params = sl.RuntimeParameters()
    runtime_params.confidence_threshold = 50
    runtime_params.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA

    print("[INFO] ZED camera opened")

    return zed, runtime_params


# --------------------------------------------------
# Tray Mask Extraction
# --------------------------------------------------

def get_best_tray_mask(outputs, frame_shape):
    """
    Get the highest-confidence tray mask.

    Since your model has only one class, every predicted instance
    is treated as a tray candidate.

    Returns:
        best_mask: bool H x W mask, or None
        best_score: float, or None
    """

    instances = outputs["instances"].to("cpu")

    if len(instances) == 0:
        return None, None

    scores = instances.scores.numpy()
    masks = instances.pred_masks.numpy()

    best_idx = int(np.argmax(scores))

    best_mask = masks[best_idx].astype(bool)
    best_score = float(scores[best_idx])

    # Clean the mask a little
    kernel = np.ones(
        (MASK_KERNEL_SIZE, MASK_KERNEL_SIZE),
        np.uint8,
    )

    mask_uint8 = best_mask.astype(np.uint8) * 255

    mask_uint8 = cv2.morphologyEx(
        mask_uint8,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask_uint8 = cv2.morphologyEx(
        mask_uint8,
        cv2.MORPH_CLOSE,
        kernel,
    )

    best_mask = mask_uint8 > 0

    return best_mask, best_score


# --------------------------------------------------
# Plane Fitting
# --------------------------------------------------

def fit_plane_to_mask_points(xyz, mask):
    """
    Fit a plane only using 3D points inside the tray mask.

    xyz:
        H x W x 3 ZED point cloud in meters

    mask:
        H x W boolean tray mask

    Returns:
        result dictionary, or None
    """

    if mask is None:
        return None

    x = xyz[:, :, 0]
    y = xyz[:, :, 1]
    z = xyz[:, :, 2]

    valid_depth = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
        & (z > MIN_DEPTH_M)
        & (z < MAX_DEPTH_M)
    )

    tray_valid_mask = mask & valid_depth

    tray_points = xyz[tray_valid_mask]

    if tray_points.shape[0] < 100:
        print("[WARNING] Not enough valid tray 3D points")
        return None

    if tray_points.shape[0] > MAX_TRAY_POINTS:
        random_indices = np.random.choice(
            tray_points.shape[0],
            MAX_TRAY_POINTS,
            replace=False,
        )

        tray_points = tray_points[random_indices]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tray_points)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=PLANE_DISTANCE_THRESHOLD_M,
        ransac_n=PLANE_RANSAC_POINTS,
        num_iterations=PLANE_RANSAC_ITERATIONS,
    )

    a, b, c, d = plane_model

    normal = np.array([a, b, c], dtype=float)

    normal_norm = np.linalg.norm(normal)

    if normal_norm == 0:
        return None

    normal = normal / normal_norm

    plane_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    plane_cloud.paint_uniform_color([1.0, 0.0, 0.0])
    outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])

    return {
        "plane_model": plane_model,
        "normal": normal,
        "inliers": inliers,
        "num_tray_points": tray_points.shape[0],
        "plane_cloud": plane_cloud,
        "outlier_cloud": outlier_cloud,
    }


# --------------------------------------------------
# Visualization
# --------------------------------------------------

def draw_tray_detection(frame, mask, score):
    """
    Draw tray mask, contour, centroid, and score.
    """

    display = frame.copy()

    if mask is None:
        cv2.putText(
            display,
            "No tray detected",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )

        return display

    overlay = display.copy()
    overlay[mask] = (0, 255, 0)

    display = cv2.addWeighted(
        overlay,
        0.30,
        display,
        0.70,
        0,
    )

    mask_uint8 = mask.astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)

        cv2.drawContours(
            display,
            [largest],
            -1,
            (0, 255, 0),
            2,
        )

        M = cv2.moments(largest)

        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(
                display,
                (cx, cy),
                6,
                (0, 0, 255),
                -1,
            )

            cv2.putText(
                display,
                f"tray {score:.2f}",
                (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

    return display


def draw_plane_overlay(frame, xyz, mask, plane_result):
    """
    Draw only the tray pixels that are close to the fitted tray plane.
    """

    display = frame.copy()

    if mask is None or plane_result is None:
        cv2.putText(
            display,
            "No tray plane",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )

        return display

    a, b, c, d = plane_result["plane_model"]

    x = xyz[:, :, 0]
    y = xyz[:, :, 1]
    z = xyz[:, :, 2]

    valid_depth = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
        & (z > MIN_DEPTH_M)
        & (z < MAX_DEPTH_M)
    )

    plane_distance = np.abs(
        a * x + b * y + c * z + d
    ) / np.sqrt(a * a + b * b + c * c)

    tray_plane_mask = (
        mask
        & valid_depth
        & (plane_distance < PLANE_DISTANCE_THRESHOLD_M)
    )

    overlay = display.copy()

    # Blue = tray plane pixels
    overlay[tray_plane_mask] = (255, 0, 0)

    display = cv2.addWeighted(
        overlay,
        0.40,
        display,
        0.60,
        0,
    )

    normal = plane_result["normal"]

    text_1 = (
        f"Plane: {a:.3f}x + {b:.3f}y + "
        f"{c:.3f}z + {d:.3f} = 0"
    )

    text_2 = (
        f"Normal: [{normal[0]:.2f}, "
        f"{normal[1]:.2f}, {normal[2]:.2f}]"
    )

    text_3 = (
        f"Tray pts: {plane_result['num_tray_points']} | "
        f"Inliers: {len(plane_result['inliers'])}"
    )

    cv2.putText(
        display,
        text_1,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        display,
        text_2,
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        display,
        text_3,
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    return display


def create_depth_colormap(xyz):
    """
    Create depth visualization from ZED XYZ point cloud.
    """

    depth = xyz[:, :, 2]

    valid_depth = (
        np.isfinite(depth)
        & (depth > MIN_DEPTH_M)
        & (depth < MAX_DEPTH_M)
    )

    depth_vis = np.zeros_like(depth, dtype=np.uint8)

    if np.any(valid_depth):
        depth_valid = depth[valid_depth]

        min_depth = np.percentile(depth_valid, 2)
        max_depth = np.percentile(depth_valid, 98)

        depth_range = max_depth - min_depth

        if depth_range > 0:
            depth_clipped = np.clip(
                depth,
                min_depth,
                max_depth,
            )

            depth_norm = (
                (depth_clipped - min_depth)
                / depth_range
                * 255
            )

            depth_norm = np.nan_to_num(depth_norm)
            depth_norm = depth_norm.astype(np.uint8)

            depth_vis[valid_depth] = depth_norm[valid_depth]

    depth_colormap = cv2.applyColorMap(
        depth_vis,
        cv2.COLORMAP_JET,
    )

    return depth_colormap


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    predictor = build_predictor()

    zed, runtime_params = open_zed()

    image_zed = sl.Mat()
    point_cloud_zed = sl.Mat()

    latest_plane_result = None

    print("[INFO] Press q to quit")
    print("[INFO] Press o to open Open3D viewer for current tray plane")

    try:
        while True:
            grab_status = zed.grab(runtime_params)

            if grab_status != sl.ERROR_CODE.SUCCESS:
                continue

            # ------------------------------------------
            # Get RGB image
            # ------------------------------------------

            zed.retrieve_image(
                image_zed,
                sl.VIEW.LEFT,
            )

            frame_rgba = image_zed.get_data()

            frame_bgr = cv2.cvtColor(
                frame_rgba,
                cv2.COLOR_BGRA2BGR,
            )

            # ------------------------------------------
            # Get XYZ point cloud
            # ------------------------------------------

            zed.retrieve_measure(
                point_cloud_zed,
                sl.MEASURE.XYZRGBA,
            )

            point_cloud_np = point_cloud_zed.get_data()

            xyz = point_cloud_np[:, :, :3]

            # ------------------------------------------
            # Detect tray
            # ------------------------------------------

            outputs = predictor(frame_bgr)

            tray_mask, tray_score = get_best_tray_mask(
                outputs,
                frame_bgr.shape,
            )

            # ------------------------------------------
            # Fit plane only to tray mask pixels
            # ------------------------------------------

            plane_result = fit_plane_to_mask_points(
                xyz,
                tray_mask,
            )

            latest_plane_result = plane_result

            if plane_result is not None:
                a, b, c, d = plane_result["plane_model"]

                print(
                    "[INFO] Tray plane: "
                    f"{a:.4f}x + {b:.4f}y + "
                    f"{c:.4f}z + {d:.4f} = 0 | "
                    f"normal={plane_result['normal']} | "
                    f"inliers={len(plane_result['inliers'])}"
                )

            # ------------------------------------------
            # Display
            # ------------------------------------------

            tray_display = draw_tray_detection(
                frame_bgr,
                tray_mask,
                tray_score if tray_score is not None else 0.0,
            )

            plane_display = draw_plane_overlay(
                frame_bgr,
                xyz,
                tray_mask,
                plane_result,
            )

            depth_display = create_depth_colormap(xyz)

            cv2.imshow(WINDOW_TRAY, tray_display)
            cv2.imshow(WINDOW_PLANE, plane_display)
            cv2.imshow(WINDOW_DEPTH, depth_display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("o"):
                if latest_plane_result is not None:
                    print("[INFO] Opening Open3D viewer")
                    print("[INFO] Red = tray plane inliers")
                    print("[INFO] Gray = tray mask outliers")

                    o3d.visualization.draw_geometries(
                        [
                            latest_plane_result["plane_cloud"],
                            latest_plane_result["outlier_cloud"],
                        ]
                    )
                else:
                    print("[WARNING] No tray plane available yet")

    finally:
        zed.close()
        cv2.destroyAllWindows()
        print("[INFO] ZED camera closed")


if __name__ == "__main__":
    main()