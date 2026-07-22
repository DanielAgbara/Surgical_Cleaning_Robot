"""
Script for all the functions for the camera:
- ZED Function
- Object Detection
- Plane Detection
"""


from pathlib import Path
import cv2
import numpy as np
import pyzed.sl as sl
import torch
import open3d as o3d

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


# --------------------------------------------------
# ZED Camera Functions
# --------------------------------------------------

ZED_RESOLUTION = sl.RESOLUTION.HD2K
ZED_FPS = 15
ZED_UNITS = sl.UNIT.METER


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
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = ZED_UNITS
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open ZED camera: {status}")

    runtime_params = sl.RuntimeParameters()
    runtime_params.confidence_threshold = 30
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
    image_bgr = cv2.cvtColor(
        image_bgra,
        cv2.COLOR_BGRA2BGR,
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