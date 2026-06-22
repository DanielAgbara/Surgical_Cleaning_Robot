import cv2
import numpy as np
import torch
import open3d as o3d

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from pathlib import Path



def setup_detectron2(detection_threshold=0.5):
    """
    Load fine-tuned Detectron2 Mask R-CNN model.
    """

    ROOT = Path(
        "/home/agbara-admin/Documents/Surgical_Cleaning_Robot"
    )

    MODEL_WEIGHTS = (
        ROOT /
        "Fine_Tuning" /
        "output" / 
        "maskrcnn_tray" /
        "model_final.pth"
    )

    cfg = get_cfg()

    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )

    cfg.MODEL.WEIGHTS = str(MODEL_WEIGHTS)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = detection_threshold

    cfg.MODEL.DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    predictor = DefaultPredictor(cfg)

    print(f"[INFO] Detectron2 loaded on {cfg.MODEL.DEVICE}")
    print(f"[INFO] Weights: {MODEL_WEIGHTS}")

    return predictor


def detect_object(frame_bgr, predictor, object_name="tray"):
    """
    Detect tray using fine-tuned one-class Mask R-CNN model.

    Since the fine-tuned model only has one class, we do not use
    COCO class_names anymore.
    """

    outputs = predictor(frame_bgr)
    instances = outputs["instances"].to("cpu")

    if len(instances) == 0:
        return None

    if not instances.has("pred_masks"):
        return None

    scores = instances.scores.numpy()
    masks = instances.pred_masks.numpy()

    best_idx = int(np.argmax(scores))

    return {
        "class_name": object_name,
        "score": float(scores[best_idx]),
        "mask": masks[best_idx].astype(bool),
    }

def get_valid_xyz_from_mask(xyz, mask):
    """
    Extract valid 3D points from object mask.
    """

    object_xyz = xyz[mask]

    valid = np.isfinite(object_xyz).all(axis=1)
    valid = valid & (object_xyz[:, 2] > 0)

    return object_xyz[valid]


def compute_2d_centroid_from_mask(mask):
    """
    Compute 2D image centroid from object mask.
    """

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    return np.array(
        [float(np.mean(xs)), float(np.mean(ys))],
        dtype=float
    )


def compute_3d_centroid_from_points(points_3d):
    """
    Compute 3D centroid from valid object point cloud.
    """

    if points_3d is None or len(points_3d) == 0:
        return None

    return np.mean(points_3d, axis=0)


def fit_plane_to_points(
    points_3d,
    distance_threshold_mm=15.0,
    ransac_n=3,
    num_iterations=1000,
    max_points=60000,
):
    """
    Fit a plane to object 3D points using Open3D RANSAC.
    """

    if points_3d is None or len(points_3d) < 100:
        return None

    points = points_3d.copy()

    if len(points) > max_points:
        idx = np.random.choice(
            len(points),
            max_points,
            replace=False
        )
        points = points[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold_mm,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )

    a, b, c, d = plane_model

    normal = np.array([a, b, c], dtype=float)
    normal_norm = np.linalg.norm(normal)

    if normal_norm > 0:
        normal = normal / normal_norm

    return {
        "plane_equation": {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "form": "ax + by + cz + d = 0",
        },
        "plane_normal": normal,
        "num_plane_points_used": len(points),
        "num_inliers": len(inliers),
        "distance_threshold_mm": distance_threshold_mm,
    }


def process_detected_object(
    object_data,
    xyz,
    plane_distance_threshold_mm=15.0,
    plane_ransac_points=3,
    plane_ransac_iterations=1000,
    max_object_plane_points=60000,
):
    """
    Calculate mask centroid, 3D centroid, depth, and plane for detected object.
    """

    if object_data is None:
        return None

    mask = object_data["mask"]

    centroid_2d = compute_2d_centroid_from_mask(mask)
    object_points_3d = get_valid_xyz_from_mask(xyz, mask)
    centroid_3d = compute_3d_centroid_from_points(object_points_3d)

    plane_data = fit_plane_to_points(
        object_points_3d,
        distance_threshold_mm=plane_distance_threshold_mm,
        ransac_n=plane_ransac_points,
        num_iterations=plane_ransac_iterations,
        max_points=max_object_plane_points,
    )

    depth_mm = None

    if centroid_3d is not None:
        depth_mm = float(centroid_3d[2])

    return {
        "class_name": object_data["class_name"],
        "score": object_data["score"],
        "mask": mask,
        "centroid_2d_px": centroid_2d,
        "centroid_3d_mm": centroid_3d,
        "depth_at_centroid_mm": depth_mm,
        "num_valid_3d_points": len(object_points_3d),
        "plane": plane_data,
    }


def make_object_mask_and_plane_visualization(
    frame_bgr,
    object_result,
    object_name,
    plane_distance_threshold_mm=15.0,
):
    """
    Visualize only the selected object mask and selected object plane.

    If the selected object is not detected, no masks for other objects are shown.
    """

    vis = frame_bgr.copy()

    if object_result is None:
        cv2.putText(
            vis,
            f"{object_name.upper()} not detected",
            (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        return np.ascontiguousarray(vis)

    mask = object_result["mask"]

    mask_overlay = vis.copy()
    mask_overlay[mask] = [0, 255, 255]

    vis = cv2.addWeighted(
        vis,
        0.65,
        mask_overlay,
        0.35,
        0,
    )

    centroid_2d = object_result["centroid_2d_px"]

    if centroid_2d is not None:
        cx, cy = centroid_2d.astype(int)

        cv2.circle(
            vis,
            (cx, cy),
            8,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            vis,
            f"{object_name} centroid",
            (cx + 10, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if object_result["plane"] is not None:
        cv2.putText(
            vis,
            f"{object_name} mask + plane detected",
            (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            vis,
            f"{object_name} detected | plane not found",
            (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return np.ascontiguousarray(vis)


def draw_plane_pixels_on_object(
    frame_bgr,
    object_result,
    xyz,
    plane_distance_threshold_mm=15.0,
):
    """
    Draw only plane pixels inside the detected object mask.
    """

    if object_result is None:
        return np.ascontiguousarray(frame_bgr)

    if object_result["plane"] is None:
        return np.ascontiguousarray(frame_bgr)

    mask = object_result["mask"]

    plane = object_result["plane"]
    a = plane["plane_equation"]["a"]
    b = plane["plane_equation"]["b"]
    c = plane["plane_equation"]["c"]
    d = plane["plane_equation"]["d"]

    x = xyz[:, :, 0]
    y = xyz[:, :, 1]
    z = xyz[:, :, 2]

    valid_depth = np.isfinite(z) & (z > 0)

    plane_distance = np.abs(
        a * x + b * y + c * z + d
    ) / np.sqrt(a * a + b * b + c * c)

    plane_mask = (
        mask
        & valid_depth
        & (plane_distance < plane_distance_threshold_mm)
    )

    plane_overlay = frame_bgr.copy()
    plane_overlay[plane_mask] = [0, 255, 0]

    vis = cv2.addWeighted(
        frame_bgr,
        0.7,
        plane_overlay,
        0.3,
        0,
    )

    return np.ascontiguousarray(vis)