"""
Script for all the functions for the camera:
- ZED Function
- Object Detection
- Plane Detection
"""


from pathlib import Path
import json
import time
import cv2 as cv
import numpy as np
import pyzed.sl as sl
import torch
import open3d as o3d

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


# --------------------------------------------------
# ZED Camera Functions
# --------------------------------------------------

ZED_RESOLUTION = sl.RESOLUTION.HD720
ZED_FPS = 15
ZED_UNITS = sl.UNIT.METER
ZED_DEPTH = sl.DEPTH_MODE.NEURAL_PLUS


def open_zed():
    """
    Open the ZED camera with depth enabled.

    Returns
    -------
    zed : sl.Camera
        Open ZED camera object.

    runtime_params : sl.RuntimeParameters
        Runtime settings used when grabbing frames.

    image_zed : sl.Mat
        Reusable image buffer for retrieving ZED images.
    """

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = ZED_RESOLUTION
    init_params.camera_fps = ZED_FPS

    # Depth must be enabled because the integration pipeline
    # will also retrieve depth and XYZ point-cloud information.
    init_params.depth_mode = ZED_DEPTH
    init_params.coordinate_units = ZED_UNITS
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open ZED camera: {status}")

    runtime_params = sl.RuntimeParameters()
    runtime_params.confidence_threshold = 50
    runtime_params.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA

    # Reusable buffer for the left camera image.
    image_zed = sl.Mat()

    print("[INFO] ZED camera opened")

    return zed, runtime_params, image_zed

def grab_frame(
    zed: sl.Camera,
    runtime_params: sl.RuntimeParameters,
) -> bool:
    """
    Grab one synchronized ZED frame.
    """

    status = zed.grab(runtime_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(f"[WARNING] Could not grab ZED frame: {status}")
        return False

    return True

def get_image(
    zed: sl.Camera,
    runtime_params: sl.RuntimeParameters,
    image_zed: sl.Mat,
):
    """
    Grab a new ZED frame and return the rectified left image
    as an OpenCV-ready BGR NumPy array.

    Parameters
    ----------
    zed : sl.Camera
        Open ZED camera.

    runtime_params : sl.RuntimeParameters
        Runtime settings passed to zed.grab().

    image_zed : sl.Mat
        Reusable ZED image buffer created in open_zed().

    Returns
    -------
    image_bgr : numpy.ndarray | None
        Rectified left-camera image in OpenCV BGR format.

        Returns None if the camera could not grab a frame.
    """

    grab_status = zed.grab(runtime_params)

    if grab_status != sl.ERROR_CODE.SUCCESS:
        print(f"[WARNING] Could not grab ZED frame: {grab_status}")
        return None

    # Retrieve the rectified left image.
    zed.retrieve_image(
        image_zed,
        sl.VIEW.LEFT
    )

    # Convert the ZED sl.Mat into a NumPy array.
    image_bgra = image_zed.get_data()

    if image_bgra is None or image_bgra.size == 0:
        print("[WARNING] Retrieved an empty ZED image")
        return None

    # ZED images are returned in BGRA format.
    # OpenCV and Detectron2 normally expect BGR.
    image_bgr = cv.cvtColor(
        image_bgra,
        cv.COLOR_BGRA2BGR,
    )

    return image_bgr

def get_point_cloud(
    zed: sl.Camera,
    point_cloud_zed: sl.Mat,
):
    """
    Retrieve the XYZ point cloud from the most recently grabbed ZED frame.

    Parameters
    ----------
    zed : sl.Camera
        Open ZED camera.

    point_cloud_zed : sl.Mat
        Reusable ZED point-cloud buffer.

    Returns
    -------
    xyz : numpy.ndarray | None
        Organized point cloud with shape:

            (image_height, image_width, 3)

        Each pixel contains:

            xyz[v, u] = [x, y, z]

        Coordinates are in meters because the camera was opened using
        sl.UNIT.METER.

    valid_mask : numpy.ndarray | None
        Boolean array with shape:

            (image_height, image_width)

        True where all XYZ coordinates are finite.
    """

    retrieve_status = zed.retrieve_measure(
        point_cloud_zed,
        sl.MEASURE.XYZ
    )

    if retrieve_status != sl.ERROR_CODE.SUCCESS:
        print(
            "[WARNING] Could not retrieve ZED point cloud: "
            f"{retrieve_status}"
        )
        return None, None

    # Convert the ZED SDK matrix to a NumPy array.
    point_cloud_data = point_cloud_zed.get_data()

    if point_cloud_data is None or point_cloud_data.size == 0:
        print("[WARNING] Retrieved an empty point cloud")
        return None, None

    # XYZ data arranged pixel-by-pixel.
    xyz = point_cloud_data[:, :, :3].copy()

    # Invalid or rejected depth points normally contain NaN or infinity.
    valid_mask = np.isfinite(xyz).all(axis=2)

    return xyz, valid_mask


def get_zed_left_intrinsics_rectified(zed):
    """
    Return the intrinsic matrix for sl.VIEW.LEFT.

    sl.VIEW.LEFT is rectified, so zero distortion is used.
    """

    camera_information = (
        zed.get_camera_information()
    )

    left_camera = (
        camera_information
        .camera_configuration
        .calibration_parameters
        .left_cam
    )

    K = np.array(
        [
            [
                left_camera.fx,
                0.0,
                left_camera.cx,
            ],
            [
                0.0,
                left_camera.fy,
                left_camera.cy,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=np.float64,
    )

    distortion = np.zeros(
        (5, 1),
        dtype=np.float64,
    )

    return K, distortion

# --------------------------------------------------
# Object Detection Functions
# --------------------------------------------------
INTEGRATION_DIR = Path(__file__).resolve().parent
INTEGRATION_DATA_DIR = INTEGRATION_DIR / "data"
TRAY_MODEL_DIR = (
    INTEGRATION_DATA_DIR
    / "models"
    / "maskrcnn_tray"
)
TRAY_CONFIG_PATH = TRAY_MODEL_DIR / "config.yaml"
TRAY_MODEL_PATH = TRAY_MODEL_DIR / "model_final.pth"

TRAY_DATA_DIR = INTEGRATION_DATA_DIR / "tray_data"
TRAY_PLANE_FILE = TRAY_DATA_DIR / "tray_plane.json"
TRAY_CENTROID_FILE = TRAY_DATA_DIR / "tray_centroid.json"

TRAY_SCORE_THRESHOLD = 0.95
TRAY_MIN_DEPTH_M = 0.10
TRAY_MAX_DEPTH_M = 3.00
TRAY_PLANE_DISTANCE_THRESHOLD_M = 0.015
TRAY_MIN_3D_POINTS = 100
TRAY_MAX_RANSAC_POINTS = 60000


def build_tray_predictor(
    score_threshold=TRAY_SCORE_THRESHOLD,
    device=None,
):
    """Load the fine-tuned one-class Mask R-CNN tray model."""
    if not TRAY_CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Tray model configuration not found: {TRAY_CONFIG_PATH}"
        )
    if not TRAY_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Tray model weights not found: {TRAY_MODEL_PATH}"
        )
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be between 0 and 1.")

    config = get_cfg()
    config.merge_from_file(str(TRAY_CONFIG_PATH))
    config.MODEL.WEIGHTS = str(TRAY_MODEL_PATH)
    config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_threshold)
    config.MODEL.DEVICE = (
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"[INFO] Loading tray detector on {config.MODEL.DEVICE}")
    return DefaultPredictor(config)


def detect_tray(image_bgr, predictor, mask_kernel_size=5):
    """Return the highest-confidence tray mask and its score.

    Returns ``None`` when no tray is detected. The trained model contains one
    class, so every returned instance is a tray candidate.
    """
    if image_bgr is None or not isinstance(image_bgr, np.ndarray):
        raise ValueError("image_bgr must be a NumPy image.")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must have shape (height, width, 3).")
    if mask_kernel_size < 1:
        raise ValueError("mask_kernel_size must be positive.")

    outputs = predictor(image_bgr)
    if "instances" not in outputs:
        return None

    instances = outputs["instances"].to("cpu")
    if len(instances) == 0 or not instances.has("pred_masks"):
        return None

    scores = instances.scores.numpy()
    masks = instances.pred_masks.numpy()
    best_index = int(np.argmax(scores))
    mask = masks[best_index].astype(np.uint8) * 255

    # Remove isolated pixels and fill small gaps in the segmentation.
    kernel = np.ones(
        (mask_kernel_size, mask_kernel_size),
        dtype=np.uint8,
    )
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = mask > 0

    if mask.shape != image_bgr.shape[:2] or not np.any(mask):
        return None

    return {
        "mask": mask,
        "score": float(scores[best_index]),
    }


def get_tray_points(
    xyz,
    tray_mask,
    valid_mask=None,
    min_depth_m=TRAY_MIN_DEPTH_M,
    max_depth_m=TRAY_MAX_DEPTH_M,
):
    """Extract finite ZED points inside the tray mask."""
    xyz = np.asarray(xyz)
    tray_mask = np.asarray(tray_mask, dtype=bool)

    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError("xyz must have shape (height, width, 3).")
    if tray_mask.shape != xyz.shape[:2]:
        raise ValueError("tray_mask and xyz must have matching image sizes.")
    if not 0 <= min_depth_m < max_depth_m:
        raise ValueError("Depth limits are invalid.")

    usable = np.isfinite(xyz).all(axis=2)
    usable &= xyz[:, :, 2] > min_depth_m
    usable &= xyz[:, :, 2] < max_depth_m

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != tray_mask.shape:
            raise ValueError("valid_mask and tray_mask must have matching sizes.")
        usable &= valid_mask

    return np.asarray(xyz[tray_mask & usable], dtype=np.float64)


def calculate_tray_plane(
    xyz,
    tray_mask,
    valid_mask=None,
    distance_threshold_m=TRAY_PLANE_DISTANCE_THRESHOLD_M,
    ransac_iterations=1000,
):
    """Fit the largest RANSAC plane inside the detected tray mask."""
    if distance_threshold_m <= 0:
        raise ValueError("distance_threshold_m must be positive.")
    if ransac_iterations < 1:
        raise ValueError("ransac_iterations must be positive.")

    tray_points = get_tray_points(xyz, tray_mask, valid_mask)
    if len(tray_points) < TRAY_MIN_3D_POINTS:
        return None

    # Bound RANSAC cost without changing the mask used for the centroid.
    fit_points = tray_points
    if len(fit_points) > TRAY_MAX_RANSAC_POINTS:
        indices = np.linspace(
            0,
            len(fit_points) - 1,
            TRAY_MAX_RANSAC_POINTS,
            dtype=int,
        )
        fit_points = fit_points[indices]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(fit_points)
    coefficients, inliers = cloud.segment_plane(
        distance_threshold=float(distance_threshold_m),
        ransac_n=3,
        num_iterations=int(ransac_iterations),
    )

    coefficients = np.asarray(coefficients, dtype=np.float64)
    normal_length = np.linalg.norm(coefficients[:3])
    if normal_length <= np.finfo(float).eps:
        return None
    coefficients /= normal_length

    # Keep the normal direction consistent: it points from the tray to camera.
    fitted_centroid = np.mean(fit_points[np.asarray(inliers, dtype=int)], axis=0)
    if np.dot(coefficients[:3], fitted_centroid) > 0:
        coefficients *= -1.0

    return {
        "coefficients": coefficients,
        "normal": coefficients[:3].copy(),
        "number_of_mask_points": int(len(tray_points)),
        "number_of_ransac_points": int(len(fit_points)),
        "number_of_inliers": int(len(inliers)),
        "distance_threshold_m": float(distance_threshold_m),
    }


def calculate_tray_centroid(
    tray_mask,
    camera_matrix,
    plane,
):
    """Intersect the 2D mask-centroid ray with the fitted tray plane."""
    if plane is None:
        return None

    mask = np.asarray(tray_mask, dtype=np.uint8)
    moments = cv.moments(mask)
    if moments["m00"] <= 0:
        return None

    pixel = np.array(
        [
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
            1.0,
        ],
        dtype=np.float64,
    )
    camera_matrix = np.asarray(
        camera_matrix,
        dtype=np.float64,
    ).reshape(3, 3)
    ray = np.linalg.solve(camera_matrix, pixel)

    coefficients = np.asarray(
        plane["coefficients"],
        dtype=np.float64,
    ).reshape(4)
    denominator = float(coefficients[:3] @ ray)
    if abs(denominator) <= 1e-8:
        return None

    distance_along_ray = -coefficients[3] / denominator
    if not np.isfinite(distance_along_ray) or distance_along_ray <= 0:
        return None

    return distance_along_ray * ray


def process_tray(
    image_bgr,
    xyz,
    camera_matrix,
    predictor,
    valid_mask=None,
):
    """Detect a tray and calculate its plane and 3D centroid."""
    detection = detect_tray(image_bgr, predictor)
    if detection is None:
        return None

    plane = calculate_tray_plane(
        xyz,
        detection["mask"],
        valid_mask,
    )
    if plane is None:
        return {
            "detection": detection,
            "plane": None,
            "centroid": None,
        }

    centroid = calculate_tray_centroid(
        detection["mask"],
        camera_matrix,
        plane,
    )
    return {
        "detection": detection,
        "plane": plane,
        "centroid": centroid,
    }


def _write_json(path, value):
    """Write JSON atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
        file.write("\n")
    temporary.replace(path)


def save_tray_data(
    plane,
    centroid,
    plane_path=TRAY_PLANE_FILE,
    centroid_path=TRAY_CENTROID_FILE,
):
    """Save the tray plane and centroid as two camera-frame JSON files."""
    if plane is None or centroid is None:
        raise ValueError("A valid tray plane and centroid are required.")

    coefficients = np.asarray(plane["coefficients"], dtype=float).reshape(4)
    centroid = np.asarray(centroid, dtype=float).reshape(3)
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(centroid)):
        raise ValueError("Tray plane and centroid must contain finite values.")

    plane_data = {
        "coordinate_frame": "zed_left_camera",
        "units": "meters",
        "equation": "a*x + b*y + c*z + d = 0",
        "a": float(coefficients[0]),
        "b": float(coefficients[1]),
        "c": float(coefficients[2]),
        "d": float(coefficients[3]),
        "normal": coefficients[:3].tolist(),
        "number_of_inliers": int(plane["number_of_inliers"]),
        "number_of_ransac_points": int(plane["number_of_ransac_points"]),
    }
    centroid_data = {
        "coordinate_frame": "zed_left_camera",
        "units": "meters",
        "x": float(centroid[0]),
        "y": float(centroid[1]),
        "z": float(centroid[2]),
    }

    _write_json(plane_path, plane_data)
    _write_json(centroid_path, centroid_data)
    print(f"[INFO] Tray plane saved to {Path(plane_path).resolve()}")
    print(f"[INFO] Tray centroid saved to {Path(centroid_path).resolve()}")
    return Path(plane_path).resolve(), Path(centroid_path).resolve()


# --------------------------------------------------
# Arm Tracking Functions
# --------------------------------------------------
BODY_FORMAT = sl.BODY_FORMAT.BODY_18

# Right arm BODY_18 indices
RIGHT_SHOULDER = 2
RIGHT_ELBOW = 3
RIGHT_WRIST = 4

# Left arm BODY_18 indices
LEFT_SHOULDER = 5
LEFT_ELBOW = 6
LEFT_WRIST = 7

def setup_body_tracking(zed):
    """
    Enable BODY_34 body tracking on an already-opened ZED camera.

    Parameters
    ----------
    zed : sl.Camera
        Already opened ZED camera object.

    Returns
    -------
    body_runtime : sl.BodyTrackingRuntimeParameters
        Runtime parameters used by zed.retrieve_bodies(...).
    """

    # -------------------------------------------------
    # Create body tracking parameter object
    # -------------------------------------------------

    body_params = sl.BodyTrackingParameters()

    # -------------------------------------------------
    # Use the accurate human body tracking model
    # -------------------------------------------------

    body_params.detection_model = (
        sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    )

    # -------------------------------------------------
    # Enable tracking so the ZED can keep person IDs
    # consistent between frames
    # -------------------------------------------------

    body_params.enable_tracking = True

    # -------------------------------------------------
    # Enable body fitting for smoother skeleton points
    # -------------------------------------------------

    body_params.enable_body_fitting = True

    # -------------------------------------------------
    # Use the BODY_34 skeleton format
    # -------------------------------------------------

    body_params.body_format = BODY_FORMAT

    # -------------------------------------------------
    # Positional tracking is required for ZED body tracking
    # -------------------------------------------------

    positional_params = sl.PositionalTrackingParameters()

    err = zed.enable_positional_tracking(positional_params)

    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(
            f"Failed to enable positional tracking: {err}"
        )

    # -------------------------------------------------
    # Enable body tracking
    # -------------------------------------------------

    err = zed.enable_body_tracking(body_params)

    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(
            f"Failed to enable body tracking: {err}"
        )

    # -------------------------------------------------
    # Create runtime body tracking parameters
    # -------------------------------------------------

    body_runtime = sl.BodyTrackingRuntimeParameters()

    # -------------------------------------------------
    # Ignore low-confidence body detections
    # -------------------------------------------------

    body_runtime.detection_confidence_threshold = 40

    print("Body tracking enabled.")

    return body_runtime


def get_single_body(bodies, mode="closest"):
    """
    Select one person from all detected people.

    Parameters
    ----------
    bodies : sl.Bodies
        Body container returned by zed.retrieve_bodies(...).

    mode : str
        Selection method.

        "closest":
            Select the detected person closest to the camera.

        "first":
            Select the first detected person.

    Returns
    -------
    body : sl.BodyData or None
        One selected body.

        Returns None if no bodies are detected.
    """

    # -------------------------------------------------
    # If no human is detected, return None
    # -------------------------------------------------

    if len(bodies.body_list) == 0:
        print("No detected bodies!")
        return None

    # -------------------------------------------------
    # Option 1: use the first detected person
    # -------------------------------------------------

    if mode == "first":
        return bodies.body_list[0]

    # -------------------------------------------------
    # Option 2: use the closest detected person
    # -------------------------------------------------
    #
    # body.position[2] is the depth value.
    #
    # Smaller positive z value usually means the person
    # is closer to the camera.
    #
    # -------------------------------------------------

    if mode == "closest":
        return min(
            bodies.body_list,
            key=lambda b: float(b.position[2])
            if b.position[2] > 0
            else float("inf")
        )

    # -------------------------------------------------
    # Reject invalid selection modes
    # -------------------------------------------------

    raise ValueError("mode must be 'closest' or 'first'")

def get_arm_indices(arm="right"):
    """
    Return BODY_18 indices for the selected arm.

    Parameters
    ----------
    arm : str
        "right" or "left".

    Returns
    -------
    shoulder_idx, elbow_idx, wrist_idx : tuple
        BODY_18 joint indices for the selected arm.
    """

    # -------------------------------------------------
    # Normalize user input
    # -------------------------------------------------

    arm = arm.lower()

    # -------------------------------------------------
    # Return right arm indices
    # -------------------------------------------------

    if arm == "right":
        return RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST

    # -------------------------------------------------
    # Return left arm indices
    # -------------------------------------------------

    if arm == "left":
        return LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST

    # -------------------------------------------------
    # Reject invalid arm names
    # -------------------------------------------------

    raise ValueError("arm must be 'right' or 'left'")


def get_arm_points(body, arm="right"):
    """
    Extract shoulder, elbow, and wrist points for one arm.

    Parameters
    ----------
    body : sl.BodyData
        One detected person from bodies.body_list.

    arm : str
        Arm to extract.
        Options:
            "right"
            "left"

    Returns
    -------
    arm_data : dict or None
        Dictionary containing 2D and 3D points.

        Returns None if the selected arm points are invalid.
    """

    # -------------------------------------------------
    # Get correct BODY_18 indices
    # -------------------------------------------------

    shoulder_idx, elbow_idx, wrist_idx = get_arm_indices(arm)

    # -------------------------------------------------
    # Get 2D pixel keypoints
    # -------------------------------------------------
    #
    # These are used for drawing on the OpenCV image.
    #
    # -------------------------------------------------

    keypoints_2d = body.keypoint_2d

    # -------------------------------------------------
    # Get 3D keypoints
    # -------------------------------------------------
    #
    # These are used for distance calculations and
    # later robot control.
    #
    # -------------------------------------------------

    keypoints_3d = body.keypoint

    # -------------------------------------------------
    # Check that the selected indices exist
    # -------------------------------------------------

    max_idx = max(shoulder_idx, elbow_idx, wrist_idx)

    if len(keypoints_2d) <= max_idx:
        return None

    if len(keypoints_3d) <= max_idx:
        return None

    # -------------------------------------------------
    # Extract 2D points
    # -------------------------------------------------

    shoulder_2d = np.array(keypoints_2d[shoulder_idx], dtype=float)
    elbow_2d = np.array(keypoints_2d[elbow_idx], dtype=float)
    wrist_2d = np.array(keypoints_2d[wrist_idx], dtype=float)

    # -------------------------------------------------
    # Extract 3D points
    # -------------------------------------------------

    shoulder_3d = np.array(keypoints_3d[shoulder_idx], dtype=float)
    elbow_3d = np.array(keypoints_3d[elbow_idx], dtype=float)
    wrist_3d = np.array(keypoints_3d[wrist_idx], dtype=float)

    # -------------------------------------------------
    # Check 2D points
    # -------------------------------------------------
    #
    # Invalid 2D points are often [0, 0] or negative.
    #
    # -------------------------------------------------

    for point in [shoulder_2d, elbow_2d, wrist_2d]:
        if point[0] <= 0 or point[1] <= 0:
            return None

    # -------------------------------------------------
    # Check 3D points
    # -------------------------------------------------
    #
    # Invalid 3D points can contain nan or inf.
    #
    # -------------------------------------------------

    for point in [shoulder_3d, elbow_3d, wrist_3d]:
        if not np.isfinite(point).all():
            return None

    # -------------------------------------------------
    # Package arm data
    # -------------------------------------------------

    arm_data = {
        "arm": arm,

        "shoulder_2d": shoulder_2d,
        "elbow_2d": elbow_2d,
        "wrist_2d": wrist_2d,

        "shoulder_3d": shoulder_3d,
        "elbow_3d": elbow_3d,
        "wrist_3d": wrist_3d,
    }

    return arm_data


def get_arm_vectors(body, arm="right"):
    """
    Return the three 3D arm vectors for a ZED BODY_18 skeleton.

    Vector direction follows the joint order in each name:

        shoulder_to_elbow = elbow - shoulder
        elbow_to_wrist = wrist - elbow
        shoulder_to_wrist = wrist - shoulder

    Parameters
    ----------
    body : sl.BodyData
        One detected person from bodies.body_list.

    arm : str
        Arm to use: "right" or "left".

    Returns
    -------
    shoulder_to_elbow, elbow_to_wrist, shoulder_to_wrist : tuple
        Three NumPy arrays of shape (3,), in the ZED camera coordinate
        frame and the configured ZED units (meters in this module).

        Returns (None, None, None) if get_arm_points(...) cannot produce
        valid shoulder, elbow, and wrist points.
    """

    arm_data = get_arm_points(body, arm)

    if arm_data is None:
        return None, None, None

    shoulder = arm_data["shoulder_3d"]
    elbow = arm_data["elbow_3d"]
    wrist = arm_data["wrist_3d"]

    shoulder_to_elbow = elbow - shoulder
    elbow_to_wrist = wrist - elbow
    shoulder_to_wrist = wrist - shoulder

    return shoulder_to_elbow, elbow_to_wrist, shoulder_to_wrist


def draw_arm_points_and_lines(image, arm_data):
    """
    Draw shoulder, elbow, and wrist on an OpenCV image.

    Parameters
    ----------
    image : np.ndarray
        OpenCV image.

    arm_data : dict or None
        Output from get_arm_points(...).

    Returns
    -------
    image : np.ndarray
        Image with arm overlay.
    """

    # -------------------------------------------------
    # If arm data is invalid, return image unchanged
    # -------------------------------------------------

    if arm_data is None:
        return image

    # -------------------------------------------------
    # Extract 2D points
    # -------------------------------------------------

    shoulder = arm_data["shoulder_2d"]
    elbow = arm_data["elbow_2d"]
    wrist = arm_data["wrist_2d"]

    # -------------------------------------------------
    # Convert float pixel coordinates to integer pixels
    # -------------------------------------------------

    shoulder = (int(shoulder[0]), int(shoulder[1]))
    elbow = (int(elbow[0]), int(elbow[1]))
    wrist = (int(wrist[0]), int(wrist[1]))

    # -------------------------------------------------
    # Draw arm links first
    # -------------------------------------------------

    cv.line(image, shoulder, elbow, (0, 255, 255), 3)
    cv.line(image, elbow, wrist, (0, 255, 255), 3)

    # -------------------------------------------------
    # Draw arm joints
    # -------------------------------------------------

    cv.circle(image, shoulder, 7, (0, 255, 0), -1)
    cv.circle(image, elbow, 7, (0, 255, 0), -1)
    cv.circle(image, wrist, 7, (0, 255, 0), -1)

    # -------------------------------------------------
    # Label joints
    # -------------------------------------------------

    cv.putText(
        image,
        "Shoulder",
        (shoulder[0] + 10, shoulder[1] - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

    cv.putText(
        image,
        "Elbow",
        (elbow[0] + 10, elbow[1] - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

    cv.putText(
        image,
        "Wrist",
        (wrist[0] + 10, wrist[1] - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

    return image

def get_arm_length(shoulder_pos, wrist_pos):
    """
    Compute shoulder-to-hand arm length using full 3D distance.

    For BODY_18:
        shoulder -> wrist
    """

    shoulder_pos = np.asarray(shoulder_pos, dtype=float).reshape(3)
    wrist_pos = np.asarray(wrist_pos, dtype=float).reshape(3)

    return np.linalg.norm(wrist_pos - shoulder_pos)
