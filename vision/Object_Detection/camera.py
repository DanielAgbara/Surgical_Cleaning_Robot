import cv2
import numpy as np
import pyzed.sl as sl
import torch
import open3d as o3d

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


# -------------------------------------------------
# Detectron2 Setup
# -------------------------------------------------

cfg = get_cfg()

cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
)

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
class_names = metadata.thing_classes

print(f"[INFO] Detectron2 loaded on {cfg.MODEL.DEVICE}")


# -------------------------------------------------
# ZED Setup
# -------------------------------------------------

zed = sl.Camera()

init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080
init_params.camera_fps = 30
init_params.depth_mode = sl.DEPTH_MODE.NEURAL
init_params.coordinate_units = sl.UNIT.METER

status = zed.open(init_params)

if status != sl.ERROR_CODE.SUCCESS:
    raise RuntimeError(f"[ERROR] Could not open ZED camera: {status}")

runtime_params = sl.RuntimeParameters()
runtime_params.confidence_threshold = 50
runtime_params.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA

image_zed = sl.Mat()
point_cloud_zed = sl.Mat()

print("[INFO] ZED camera opened")


# -------------------------------------------------
# Grab One Frame
# -------------------------------------------------

try:
    grab_status = zed.grab(runtime_params)

    if grab_status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"[ERROR] Failed to grab frame: {grab_status}")

    # -------------------------------------------------
    # Get RGB Image
    # -------------------------------------------------

    zed.retrieve_image(image_zed, sl.VIEW.LEFT)

    frame_rgba = image_zed.get_data()

    # Convert ZED image from BGRA to OpenCV BGR
    frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_BGRA2BGR)

    # -------------------------------------------------
    # Get Point Cloud
    # -------------------------------------------------

    # XYZRGBA gives one 3D point per image pixel
    # Shape is usually: height x width x 4
    # Channels are: X, Y, Z, color
    zed.retrieve_measure(point_cloud_zed, sl.MEASURE.XYZRGBA)

    point_cloud_np = point_cloud_zed.get_data()

    print("[INFO] Image shape:", frame_bgr.shape)
    print("[INFO] Point cloud shape:", point_cloud_np.shape)

    # -------------------------------------------------
    # Run Mask R-CNN Instance Segmentation
    # -------------------------------------------------

    outputs = predictor(frame_bgr)
    instances = outputs["instances"].to("cpu")

    pred_classes = instances.pred_classes
    scores = instances.scores

    print("\n[INFO] Detected objects:")

    for class_id, score in zip(pred_classes, scores):
        object_name = class_names[int(class_id)]
        confidence = float(score)
        print(f" - {object_name}: {confidence:.2f}")

    # -------------------------------------------------
    # Visualize Mask R-CNN Result
    # -------------------------------------------------

    visualizer = Visualizer(
        frame_bgr[:, :, ::-1],
        metadata=metadata,
        scale=1.0
    )

    vis_output = visualizer.draw_instance_predictions(instances)

    seg_result_bgr = vis_output.get_image()[:, :, ::-1]

    # -------------------------------------------------
    # Prepare XYZ Point Cloud Data
    # -------------------------------------------------

    xyz = point_cloud_np[:, :, :3]

    depth = xyz[:, :, 2]

    # Valid depth means:
    # - not NaN
    # - not infinity
    # - greater than zero
    valid_depth = np.isfinite(depth) & (depth > 0)

    # -------------------------------------------------
    # Create Depth Visualization
    # -------------------------------------------------

    depth_vis = np.zeros_like(depth, dtype=np.uint8)

    if np.any(valid_depth):

        depth_valid = depth[valid_depth]

        min_depth = np.percentile(depth_valid, 2)
        max_depth = np.percentile(depth_valid, 98)

        depth_range = max_depth - min_depth

        if depth_range > 0:

            depth_clipped = np.clip(depth, min_depth, max_depth)

            depth_norm = (
                (depth_clipped - min_depth)
                / depth_range
                * 255
            )

            # Replace NaN/inf values before converting to uint8
            depth_norm = np.nan_to_num(depth_norm)

            depth_norm = depth_norm.astype(np.uint8)

            depth_vis[valid_depth] = depth_norm[valid_depth]

    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    # -------------------------------------------------
    # Convert ZED Point Cloud to Open3D Point Cloud
    # -------------------------------------------------

    # Flatten H x W x 3 point cloud into N x 3
    xyz_flat = xyz.reshape(-1, 3)

    # Flatten RGB image into N x 3 colors
    rgb_flat = frame_bgr.reshape(-1, 3)[:, ::-1] / 255.0

    # Keep only valid 3D points
    valid_points = np.isfinite(xyz_flat).all(axis=1)

    # Remove very close and very far points
    valid_points = valid_points & (xyz_flat[:, 2] > 0.2) & (xyz_flat[:, 2] < 5.0)

    xyz_valid = xyz_flat[valid_points]
    rgb_valid = rgb_flat[valid_points]

    # Optional downsample so RANSAC is faster
    # For HD1080, the point cloud is large
    max_points = 150000

    if xyz_valid.shape[0] > max_points:
        random_indices = np.random.choice(
            xyz_valid.shape[0],
            max_points,
            replace=False
        )

        xyz_valid = xyz_valid[random_indices]
        rgb_valid = rgb_valid[random_indices]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_valid)
    pcd.colors = o3d.utility.Vector3dVector(rgb_valid)

    print("[INFO] Valid Open3D points:", len(pcd.points))

    # -------------------------------------------------
    # RANSAC Plane Detection
    # -------------------------------------------------

    # This finds the largest plane in the point cloud.
    # For your setup, this will often be the table, wall, or floor.
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.015,  # 1.5 cm plane tolerance
        ransac_n=3,                # 3 points define a plane
        num_iterations=1000
    )

    a, b, c, d = plane_model

    print("\n[INFO] Detected plane equation:")
    print(f"{a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    print("[INFO] Plane inlier points:", len(inliers))

    # Plane normal vector
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)

    print("[INFO] Plane normal:", normal)

    # -------------------------------------------------
    # Separate Plane and Non-Plane Points
    # -------------------------------------------------

    plane_cloud = pcd.select_by_index(inliers)
    non_plane_cloud = pcd.select_by_index(inliers, invert=True)

    # Color detected plane red
    plane_cloud.paint_uniform_color([1.0, 0.0, 0.0])

    # Color everything else gray
    non_plane_cloud.paint_uniform_color([0.6, 0.6, 0.6])

    # -------------------------------------------------
    # Create 2D Plane Mask For Image Visualization
    # -------------------------------------------------

    # The Open3D plane was detected on downsampled points,
    # so for the image mask, recompute distance of every pixel point to the plane.

    x = xyz[:, :, 0]
    y = xyz[:, :, 1]
    z = xyz[:, :, 2]

    # Distance from every point to plane:
    # distance = |ax + by + cz + d| / sqrt(a^2 + b^2 + c^2)
    plane_distance = np.abs(a * x + b * y + c * z + d) / np.sqrt(a*a + b*b + c*c)

    # Plane pixels are valid depth pixels close to the detected plane
    plane_mask = valid_depth & (plane_distance < 0.015)

    # Convert boolean mask to image
    plane_mask_uint8 = (plane_mask.astype(np.uint8)) * 255

    # Clean mask slightly
    kernel = np.ones((5, 5), np.uint8)
    plane_mask_uint8 = cv2.morphologyEx(plane_mask_uint8, cv2.MORPH_OPEN, kernel)
    plane_mask_uint8 = cv2.morphologyEx(plane_mask_uint8, cv2.MORPH_CLOSE, kernel)

    # Make overlay image
    plane_overlay = frame_bgr.copy()

    # Color plane pixels green
    plane_overlay[plane_mask_uint8 > 0] = [0, 255, 0]

    # Blend original image with green plane overlay
    plane_result_bgr = cv2.addWeighted(
        frame_bgr,
        0.6,
        plane_overlay,
        0.4,
        0
    )

    # -------------------------------------------------
    # Show Separate Visualization Windows
    # -------------------------------------------------

    cv2.imshow(
        "Mask R-CNN Instance Segmentation",
        seg_result_bgr
    )

    cv2.imshow(
        "ZED Depth Map",
        depth_colormap
    )

    cv2.imshow(
        "RANSAC Plane Detection",
        plane_result_bgr
    )

    print("\n[INFO] Press any key in an OpenCV window to continue")

    cv2.waitKey(0)

    # -------------------------------------------------
    # Open3D 3D Visualization
    # -------------------------------------------------

    print("[INFO] Opening Open3D viewer")
    print("[INFO] Red = detected plane")
    print("[INFO] Gray = non-plane points")

    o3d.visualization.draw_geometries(
        [plane_cloud, non_plane_cloud]
    )

finally:
    zed.close()
    cv2.destroyAllWindows()
    print("[INFO] ZED camera closed")