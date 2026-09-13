#!/usr/bin/env python3

"""
ROS 2 ChArUco tracker for the human cleaning device.

Subscribes
----------
/camera/left/image_raw
    sensor_msgs/msg/Image
    Encoding: bgr8

/camera/left/camera_info
    sensor_msgs/msg/CameraInfo

Publishes
---------
/human/marker_pose
    geometry_msgs/msg/PoseStamped

    Pose of the ChArUco marker in the ZED left-camera frame.

/human/cleaning_pose
    geometry_msgs/msg/PoseStamped

    Pose of the cleaning-head center in the ZED left-camera frame.

/human/charuco/reprojection_error
    std_msgs/msg/Float32

Project length convention
-------------------------
Translations published in the PoseStamped messages are in millimeters.

Rotations are represented as unit quaternions.

Transform chain
---------------
T_camera_cleaning_head =
    T_camera_marker @ T_marker_cleaning_head

No cv_bridge is used.
"""

import json
from pathlib import Path

import cv2 as cv
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo

from geometry_msgs.msg import PoseStamped

from std_msgs.msg import Float32

from ament_index_python.packages import (
    get_package_share_directory,
)


# ============================================================
# Package / ChArUco Configuration
# ============================================================

PACKAGE_NAME = "cleaning_robot_perception"

# Human cleaning-device board.
CHARUCO_SQUARES_X = 3
CHARUCO_SQUARES_Y = 3

# Use millimeters so solvePnP returns translation in millimeters.
CHARUCO_SQUARE_LENGTH_MM = 32.0
CHARUCO_MARKER_LENGTH_MM = 25.8

CHARUCO_DICTIONARY = cv.aruco.DICT_4X4_50

MIN_CORNERS_FOR_POSE = 4


# Proper 180-degree X rotation.
#
# This flips Y and Z together while preserving determinant +1.
T_MARKER_Z_FLIP = np.diag(
    [
        1.0,
        -1.0,
        -1.0,
        1.0,
    ]
)


# ============================================================
# Helper Functions
# ============================================================

def validate_transform(
    transform,
    name="transform",
):
    """
    Validate a 4x4 rigid transform.
    """

    transform = np.asarray(
        transform,
        dtype=np.float64,
    )

    if transform.shape != (4, 4):

        raise ValueError(
            f"{name} must have shape (4, 4)"
        )

    if not np.all(
        np.isfinite(transform)
    ):

        raise ValueError(
            f"{name} contains non-finite values"
        )

    if not np.allclose(
        transform[3],
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        atol=1e-9,
    ):

        raise ValueError(
            f"{name} has invalid homogeneous row"
        )

    rotation = transform[
        :3,
        :3,
    ]

    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=1e-6,
    ):

        raise ValueError(
            f"{name} rotation is not orthonormal"
        )

    if not np.isclose(
        np.linalg.det(rotation),
        1.0,
        atol=1e-6,
    ):

        raise ValueError(
            f"{name} rotation determinant is not +1"
        )

    return transform.copy()


# ============================================================
# Rotation Helpers
# ============================================================

def rotation_log(rotation):
    """
    Convert rotation matrix to axis-angle rotation vector.
    """

    rotation = np.asarray(
        rotation,
        dtype=np.float64,
    ).reshape(
        3,
        3,
    )

    cosine = np.clip(
        (
            np.trace(rotation)
            - 1.0
        )
        / 2.0,
        -1.0,
        1.0,
    )

    angle = float(
        np.arccos(cosine)
    )

    skew_vector = np.array(
        [
            rotation[2, 1]
            - rotation[1, 2],

            rotation[0, 2]
            - rotation[2, 0],

            rotation[1, 0]
            - rotation[0, 1],
        ],
        dtype=np.float64,
    )

    if angle < 1e-8:

        return (
            0.5
            * skew_vector
        )

    if (
        np.pi - angle
        < 1e-5
    ):

        eigenvalues, eigenvectors = (
            np.linalg.eig(
                rotation
            )
        )

        axis_index = int(
            np.argmin(
                np.abs(
                    eigenvalues
                    - 1.0
                )
            )
        )

        axis = np.real(
            eigenvectors[
                :,
                axis_index,
            ]
        )

        axis /= np.linalg.norm(
            axis
        )

        return (
            angle
            * axis
        )

    return (
        angle
        * skew_vector
        / (
            2.0
            * np.sin(angle)
        )
    )


def rotation_exp(
    rotation_vector,
):
    """
    Convert axis-angle rotation vector to rotation matrix.
    """

    rotation_vector = np.asarray(
        rotation_vector,
        dtype=np.float64,
    ).reshape(3)

    angle = float(
        np.linalg.norm(
            rotation_vector
        )
    )

    if angle < 1e-12:

        return np.eye(
            3,
            dtype=np.float64,
        )

    axis = (
        rotation_vector
        / angle
    )

    skew = np.array(
        [
            [
                0.0,
                -axis[2],
                axis[1],
            ],
            [
                axis[2],
                0.0,
                -axis[0],
            ],
            [
                -axis[1],
                axis[0],
                0.0,
            ],
        ],
        dtype=np.float64,
    )

    return (
        np.eye(3)
        + np.sin(angle) * skew
        + (
            1.0
            - np.cos(angle)
        )
        * (
            skew @ skew
        )
    )


def rotation_difference_deg(
    first,
    second,
):
    """
    Angular difference between two rotation matrices.
    """

    relative = (
        first.T
        @ second
    )

    cosine = np.clip(
        (
            np.trace(relative)
            - 1.0
        )
        / 2.0,
        -1.0,
        1.0,
    )

    return float(
        np.rad2deg(
            np.arccos(
                cosine
            )
        )
    )


# ============================================================
# One Euro Filter
# ============================================================

def low_pass_alpha(
    cutoff_hz,
    elapsed_s,
):
    """
    One Euro filter low-pass coefficient.
    """

    time_constant = (
        1.0
        / (
            2.0
            * np.pi
            * cutoff_hz
        )
    )

    return float(
        elapsed_s
        / (
            elapsed_s
            + time_constant
        )
    )


class PoseOneEuroFilter:
    """
    One Euro filter for SE(3).

    Translation is internally converted from millimeters to
    meters so the tuning values remain equivalent to the old
    human.py implementation.
    """

    def __init__(
        self,
        translation_min_cutoff_hz=0.5,
        translation_beta=0.05,
        rotation_min_cutoff_hz=0.5,
        rotation_beta=0.05,
        derivative_cutoff_hz=1.0,
    ):

        self.translation_min_cutoff_hz = (
            float(
                translation_min_cutoff_hz
            )
        )

        self.translation_beta = float(
            translation_beta
        )

        self.rotation_min_cutoff_hz = float(
            rotation_min_cutoff_hz
        )

        self.rotation_beta = float(
            rotation_beta
        )

        self.derivative_cutoff_hz = float(
            derivative_cutoff_hz
        )

        self.reset()

    def reset(self):

        self.previous_timestamp = None

        self.previous_raw_translation_m = None

        self.previous_filtered_translation_m = (
            None
        )

        self.previous_translation_velocity = (
            np.zeros(
                3,
                dtype=np.float64,
            )
        )

        self.previous_raw_rotation = None

        self.previous_filtered_rotation = (
            None
        )

        self.previous_angular_velocity = (
            np.zeros(
                3,
                dtype=np.float64,
            )
        )

    def update(
        self,
        transform_mm,
        timestamp_s,
    ):

        transform_mm = validate_transform(
            transform_mm,
            "filter input",
        )

        # Work in meters internally to preserve the old
        # One Euro tuning values.
        translation_m = (
            transform_mm[
                :3,
                3,
            ]
            / 1000.0
        )

        rotation = transform_mm[
            :3,
            :3,
        ]

        # ----------------------------------------------------
        # First observation
        # ----------------------------------------------------

        if (
            self.previous_timestamp
            is None
        ):

            filtered_translation_m = (
                translation_m.copy()
            )

            filtered_rotation = (
                rotation.copy()
            )

        else:

            elapsed_s = (
                timestamp_s
                - self.previous_timestamp
            )

            if elapsed_s <= 0.0:

                raise ValueError(
                    "Filter timestamps must increase"
                )

            derivative_alpha = (
                low_pass_alpha(
                    self.derivative_cutoff_hz,
                    elapsed_s,
                )
            )

            # =================================================
            # Translation
            # =================================================

            raw_velocity = (
                translation_m
                - self.previous_raw_translation_m
            ) / elapsed_s

            filtered_velocity = (
                derivative_alpha
                * raw_velocity
                + (
                    1.0
                    - derivative_alpha
                )
                * self.previous_translation_velocity
            )

            translation_cutoff = (
                self.translation_min_cutoff_hz
                + self.translation_beta
                * np.linalg.norm(
                    filtered_velocity
                )
            )

            translation_alpha = (
                low_pass_alpha(
                    translation_cutoff,
                    elapsed_s,
                )
            )

            filtered_translation_m = (
                translation_alpha
                * translation_m
                + (
                    1.0
                    - translation_alpha
                )
                * self.previous_filtered_translation_m
            )

            # =================================================
            # Rotation
            # =================================================

            raw_rotation_step = (
                rotation_log(
                    self.previous_raw_rotation.T
                    @ rotation
                )
            )

            raw_angular_velocity = (
                raw_rotation_step
                / elapsed_s
            )

            filtered_angular_velocity = (
                derivative_alpha
                * raw_angular_velocity
                + (
                    1.0
                    - derivative_alpha
                )
                * self.previous_angular_velocity
            )

            rotation_cutoff = (
                self.rotation_min_cutoff_hz
                + self.rotation_beta
                * np.linalg.norm(
                    filtered_angular_velocity
                )
            )

            rotation_alpha = (
                low_pass_alpha(
                    rotation_cutoff,
                    elapsed_s,
                )
            )

            filtered_rotation_step = (
                rotation_log(
                    self.previous_filtered_rotation.T
                    @ rotation
                )
            )

            filtered_rotation = (
                self.previous_filtered_rotation
                @ rotation_exp(
                    rotation_alpha
                    * filtered_rotation_step
                )
            )

            self.previous_translation_velocity = (
                filtered_velocity
            )

            self.previous_angular_velocity = (
                filtered_angular_velocity
            )

        # ----------------------------------------------------
        # Save state
        # ----------------------------------------------------

        self.previous_timestamp = (
            timestamp_s
        )

        self.previous_raw_translation_m = (
            translation_m.copy()
        )

        self.previous_filtered_translation_m = (
            filtered_translation_m.copy()
        )

        self.previous_raw_rotation = (
            rotation.copy()
        )

        self.previous_filtered_rotation = (
            filtered_rotation.copy()
        )

        # ----------------------------------------------------
        # Return millimeter transform
        # ----------------------------------------------------

        filtered_transform = np.eye(
            4,
            dtype=np.float64,
        )

        filtered_transform[
            :3,
            :3,
        ] = filtered_rotation

        filtered_transform[
            :3,
            3,
        ] = (
            filtered_translation_m
            * 1000.0
        )

        return validate_transform(
            filtered_transform,
            "filtered pose",
        )


# ============================================================
# Quaternion Conversion
# ============================================================

def rotation_matrix_to_quaternion(
    rotation,
):
    """
    Convert 3x3 rotation matrix to quaternion:

        [x, y, z, w]
    """

    rotation = np.asarray(
        rotation,
        dtype=np.float64,
    ).reshape(
        3,
        3,
    )

    trace = np.trace(
        rotation
    )

    if trace > 0.0:

        s = np.sqrt(
            trace + 1.0
        ) * 2.0

        qw = 0.25 * s

        qx = (
            rotation[2, 1]
            - rotation[1, 2]
        ) / s

        qy = (
            rotation[0, 2]
            - rotation[2, 0]
        ) / s

        qz = (
            rotation[1, 0]
            - rotation[0, 1]
        ) / s

    elif (
        rotation[0, 0]
        > rotation[1, 1]
        and rotation[0, 0]
        > rotation[2, 2]
    ):

        s = np.sqrt(
            1.0
            + rotation[0, 0]
            - rotation[1, 1]
            - rotation[2, 2]
        ) * 2.0

        qw = (
            rotation[2, 1]
            - rotation[1, 2]
        ) / s

        qx = 0.25 * s

        qy = (
            rotation[0, 1]
            + rotation[1, 0]
        ) / s

        qz = (
            rotation[0, 2]
            + rotation[2, 0]
        ) / s

    elif (
        rotation[1, 1]
        > rotation[2, 2]
    ):

        s = np.sqrt(
            1.0
            + rotation[1, 1]
            - rotation[0, 0]
            - rotation[2, 2]
        ) * 2.0

        qw = (
            rotation[0, 2]
            - rotation[2, 0]
        ) / s

        qx = (
            rotation[0, 1]
            + rotation[1, 0]
        ) / s

        qy = 0.25 * s

        qz = (
            rotation[1, 2]
            + rotation[2, 1]
        ) / s

    else:

        s = np.sqrt(
            1.0
            + rotation[2, 2]
            - rotation[0, 0]
            - rotation[1, 1]
        ) * 2.0

        qw = (
            rotation[1, 0]
            - rotation[0, 1]
        ) / s

        qx = (
            rotation[0, 2]
            + rotation[2, 0]
        ) / s

        qy = (
            rotation[1, 2]
            + rotation[2, 1]
        ) / s

        qz = 0.25 * s

    quaternion = np.array(
        [
            qx,
            qy,
            qz,
            qw,
        ],
        dtype=np.float64,
    )

    quaternion /= np.linalg.norm(
        quaternion
    )

    return quaternion


# ============================================================
# ChArUco Tracker Node
# ============================================================

class CharucoTrackerNode(Node):

    def __init__(self):

        super().__init__(
            "charuco_tracker"
        )

        # ====================================================
        # Parameters
        # ====================================================

        self.declare_parameter(
            "min_markers",
            4,
        )

        self.declare_parameter(
            "min_corners",
            4,
        )

        self.declare_parameter(
            "max_mean_reprojection_error_px",
            1.0,
        )

        self.declare_parameter(
            "max_corner_reprojection_error_px",
            2.0,
        )

        # Same physical limits as human.py,
        # converted from meters to millimeters.
        self.declare_parameter(
            "max_translation_jump_mm",
            33.0,
        )

        self.declare_parameter(
            "max_rotation_jump_deg",
            30.0,
        )

        self.declare_parameter(
            "max_linear_speed_mm_s",
            1000.0,
        )

        self.declare_parameter(
            "max_angular_speed_deg_s",
            90.0,
        )

        self.declare_parameter(
            "max_timestamp_gap_s",
            0.120,
        )

        self.min_markers = int(
            self.get_parameter(
                "min_markers"
            ).value
        )

        self.min_corners = int(
            self.get_parameter(
                "min_corners"
            ).value
        )

        self.max_mean_error = float(
            self.get_parameter(
                "max_mean_reprojection_error_px"
            ).value
        )

        self.max_corner_error = float(
            self.get_parameter(
                "max_corner_reprojection_error_px"
            ).value
        )

        self.max_translation_jump_mm = float(
            self.get_parameter(
                "max_translation_jump_mm"
            ).value
        )

        self.max_rotation_jump_deg = float(
            self.get_parameter(
                "max_rotation_jump_deg"
            ).value
        )

        self.max_linear_speed_mm_s = float(
            self.get_parameter(
                "max_linear_speed_mm_s"
            ).value
        )

        self.max_angular_speed_deg_s = float(
            self.get_parameter(
                "max_angular_speed_deg_s"
            ).value
        )

        self.max_timestamp_gap_s = float(
            self.get_parameter(
                "max_timestamp_gap_s"
            ).value
        )

        # ====================================================
        # Camera Calibration
        # ====================================================

        self.camera_matrix = None

        self.dist_coeffs = np.zeros(
            (
                5,
                1,
            ),
            dtype=np.float64,
        )

        # ====================================================
        # ChArUco Board
        # ====================================================

        dictionary = (
            cv.aruco.getPredefinedDictionary(
                CHARUCO_DICTIONARY
            )
        )

        self.board = (
            cv.aruco.CharucoBoard(
                (
                    CHARUCO_SQUARES_X,
                    CHARUCO_SQUARES_Y,
                ),
                CHARUCO_SQUARE_LENGTH_MM,
                CHARUCO_MARKER_LENGTH_MM,
                dictionary,
            )
        )

        self.board.setLegacyPattern(
            False
        )

        # Detector created after CameraInfo arrives.
        self.detector = None

        # ====================================================
        # Fixed Marker -> Cleaning Head Transform
        # ====================================================

        self.T_marker_cleaning_head = (
            self.load_marker_cleaning_transform()
        )

        # ====================================================
        # Filtering / Quality History
        # ====================================================

        self.pose_filter = (
            PoseOneEuroFilter()
        )

        self.previous_timestamp = None

        self.previous_cleaning_pose = None

        # ====================================================
        # Subscribers
        # ====================================================

        self.camera_info_subscription = (
            self.create_subscription(
                CameraInfo,
                "/camera/left/camera_info",
                self.camera_info_callback,
                qos_profile_sensor_data,
            )
        )

        self.image_subscription = (
            self.create_subscription(
                Image,
                "/camera/left/image_raw",
                self.image_callback,
                qos_profile_sensor_data,
            )
        )

        # ====================================================
        # Publishers
        # ====================================================

        self.marker_pose_publisher = (
            self.create_publisher(
                PoseStamped,
                "/human/marker_pose",
                10,
            )
        )

        self.cleaning_pose_publisher = (
            self.create_publisher(
                PoseStamped,
                "/human/cleaning_pose",
                10,
            )
        )

        self.error_publisher = (
            self.create_publisher(
                Float32,
                "/human/charuco/reprojection_error",
                10,
            )
        )

        self.get_logger().info(
            "ChArUco tracker started"
        )

        self.get_logger().info(
            "Human tool board: "
            "3x3, 32 mm squares, "
            "25.8 mm markers"
        )

        self.get_logger().info(
            "Output translations: millimeters"
        )

    # ========================================================
    # Load Fixed Tool Transform
    # ========================================================

    def load_marker_cleaning_transform(
        self,
    ):
        """
        Load T_marker_cleaning_head.

        The existing JSON stores translation in meters.
        Convert only its translation column to millimeters.
        """

        package_share = (
            get_package_share_directory(
                PACKAGE_NAME
            )
        )

        path = (
            Path(package_share)
            / "config"
            / "marker_to_cleaning_head.json"
        )

        if not path.exists():

            raise FileNotFoundError(
                "Marker-to-cleaning-head transform "
                f"not found: {path}"
            )

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            payload = json.load(
                file
            )

        if (
            not isinstance(
                payload,
                dict,
            )
            or "data" not in payload
        ):

            raise ValueError(
                f"Invalid transform JSON: {path}"
            )

        transform = np.asarray(
            payload["data"],
            dtype=np.float64,
        )

        transform = validate_transform(
            transform,
            "T_marker_cleaning_head",
        )

        # Stored JSON translation is meters.
        transform[
            :3,
            3,
        ] *= 1000.0

        self.get_logger().info(
            "Loaded T_marker_cleaning_head "
            f"from {path}"
        )

        return transform

    # ========================================================
    # CameraInfo
    # ========================================================

    def camera_info_callback(
        self,
        msg,
    ):

        self.camera_matrix = np.asarray(
            msg.k,
            dtype=np.float64,
        ).reshape(
            3,
            3,
        )

        self.create_detector()

    # ========================================================
    # Create ChArUco Detector
    # ========================================================

    def create_detector(
        self,
    ):

        detector_parameters = (
            cv.aruco.DetectorParameters()
        )

        detector_parameters.cornerRefinementMethod = (
            cv.aruco.CORNER_REFINE_NONE
        )

        charuco_parameters = (
            cv.aruco.CharucoParameters()
        )

        charuco_parameters.minMarkers = 2

        charuco_parameters.tryRefineMarkers = (
            True
        )

        charuco_parameters.cameraMatrix = (
            self.camera_matrix
        )

        charuco_parameters.distCoeffs = (
            self.dist_coeffs
        )

        self.detector = (
            cv.aruco.CharucoDetector(
                self.board,
                charuco_parameters,
                detector_parameters,
            )
        )

    # ========================================================
    # ROS Image -> NumPy
    # ========================================================

    def ros_image_to_bgr(
        self,
        msg,
    ):
        """
        Convert bgr8 sensor_msgs/Image directly to NumPy.
        """

        if msg.encoding != "bgr8":

            raise ValueError(
                "Expected bgr8 image, "
                f"received {msg.encoding}"
            )

        height = int(
            msg.height
        )

        width = int(
            msg.width
        )

        expected_row_bytes = (
            width * 3
        )

        if msg.step < expected_row_bytes:

            raise ValueError(
                "Image row step is too small"
            )

        raw = np.frombuffer(
            msg.data,
            dtype=np.uint8,
        )

        required = (
            height
            * msg.step
        )

        if raw.size < required:

            raise ValueError(
                "Image buffer is smaller "
                "than expected"
            )

        rows = raw[
            :required
        ].reshape(
            height,
            msg.step,
        )

        image = rows[
            :,
            :expected_row_bytes,
        ].reshape(
            height,
            width,
            3,
        )

        return np.ascontiguousarray(
            image
        ).copy()

    # ========================================================
    # Detect ChArUco
    # ========================================================

    def detect_charuco(
        self,
        image_bgr,
    ):

        gray = cv.cvtColor(
            image_bgr,
            cv.COLOR_BGR2GRAY,
        )

        (
            charuco_corners,
            charuco_ids,
            marker_corners,
            marker_ids,
        ) = self.detector.detectBoard(
            gray
        )

        num_markers = (
            0
            if marker_ids is None
            else len(marker_ids)
        )

        num_corners = (
            0
            if charuco_ids is None
            else len(charuco_ids)
        )

        if charuco_ids is None:

            collinear = True

        else:

            collinear = bool(
                self.board.checkCharucoCornersCollinear(
                    charuco_ids
                )
            )

        return {
            "charuco_corners": charuco_corners,
            "charuco_ids": charuco_ids,
            "marker_corners": marker_corners,
            "marker_ids": marker_ids,
            "num_markers": num_markers,
            "num_corners": num_corners,
            "collinear": collinear,
        }

    # ========================================================
    # Pose Estimation
    # ========================================================

    def estimate_pose(
        self,
        detection,
    ):
        """
        Estimate T_camera_marker.

        ChArUco board dimensions are in millimeters,
        therefore solvePnP translation is also millimeters.
        """

        if (
            detection["num_corners"]
            < MIN_CORNERS_FOR_POSE
        ):

            return None

        if detection["collinear"]:

            return None

        (
            object_points,
            image_points,
        ) = self.board.matchImagePoints(
            detection["charuco_corners"],
            detection["charuco_ids"],
        )

        object_points = np.asarray(
            object_points,
            dtype=np.float64,
        )

        image_points = np.asarray(
            image_points,
            dtype=np.float64,
        )

        success, rvec, tvec = (
            cv.solvePnP(
                object_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv.SOLVEPNP_IPPE,
            )
        )

        if not success:

            return None

        # Nonlinear pose refinement.
        rvec, tvec = (
            cv.solvePnPRefineLM(
                object_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                rvec,
                tvec,
            )
        )

        rotation_matrix, _ = (
            cv.Rodrigues(
                rvec
            )
        )

        transform = np.eye(
            4,
            dtype=np.float64,
        )

        transform[
            :3,
            :3,
        ] = rotation_matrix

        transform[
            :3,
            3,
        ] = tvec.reshape(3)

        # ----------------------------------------------------
        # Reprojection error
        # ----------------------------------------------------

        projected_points, _ = (
            cv.projectPoints(
                object_points,
                rvec,
                tvec,
                self.camera_matrix,
                self.dist_coeffs,
            )
        )

        measured = image_points.reshape(
            -1,
            2,
        )

        projected = projected_points.reshape(
            -1,
            2,
        )

        errors = np.linalg.norm(
            measured
            - projected,
            axis=1,
        )

        return {
            "transform": transform,
            "mean_error": float(
                np.mean(errors)
            ),
            "max_error": float(
                np.max(errors)
            ),
        }

    # ========================================================
    # Marker Z Convention
    # ========================================================

    def make_marker_z_point_into_board(
        self,
        transform,
    ):
        """
        Ensure marker +Z points into the physical board.

        With the visible printed face facing the camera, inward
        generally points away from the camera.

        Therefore +Z should have a positive dot product with the
        camera-to-marker translation.
        """

        transform = validate_transform(
            transform,
            "raw marker transform",
        )

        camera_to_marker = (
            transform[
                :3,
                3,
            ]
        )

        if (
            np.linalg.norm(
                camera_to_marker
            )
            < 1e-6
        ):

            raise ValueError(
                "Marker is too close to camera origin"
            )

        inward_score = float(
            np.dot(
                transform[
                    :3,
                    2,
                ],
                camera_to_marker,
            )
        )

        if inward_score <= 0.0:

            transform = (
                transform
                @ T_MARKER_Z_FLIP
            )

        return validate_transform(
            transform,
            "T_camera_marker",
        )

    # ========================================================
    # Quality Gate
    # ========================================================

    def passes_quality_gate(
        self,
        detection,
        pose,
        cleaning_transform,
        timestamp_s,
    ):
        """
        Reject clearly unreliable ChArUco measurements.
        """

        if (
            detection["num_markers"]
            < self.min_markers
        ):

            return False

        if (
            detection["num_corners"]
            < self.min_corners
        ):

            return False

        if detection["collinear"]:

            return False

        if (
            pose["mean_error"]
            > self.max_mean_error
        ):

            return False

        if (
            pose["max_error"]
            > self.max_corner_error
        ):

            return False

        # First valid observation.
        if (
            self.previous_timestamp
            is None
        ):

            return True

        elapsed = (
            timestamp_s
            - self.previous_timestamp
        )

        if elapsed <= 0.0:

            return False

        # Long dropout:
        #
        # allow a new segment instead of comparing speed with
        # stale history.
        if (
            elapsed
            > self.max_timestamp_gap_s
        ):

            self.previous_timestamp = None
            self.previous_cleaning_pose = None

            self.pose_filter.reset()

            return True

        previous = (
            self.previous_cleaning_pose
        )

        translation_jump_mm = float(
            np.linalg.norm(
                cleaning_transform[
                    :3,
                    3,
                ]
                - previous[
                    :3,
                    3,
                ]
            )
        )

        rotation_jump_deg = (
            rotation_difference_deg(
                previous[
                    :3,
                    :3,
                ],
                cleaning_transform[
                    :3,
                    :3,
                ],
            )
        )

        linear_speed_mm_s = (
            translation_jump_mm
            / elapsed
        )

        angular_speed_deg_s = (
            rotation_jump_deg
            / elapsed
        )

        if (
            translation_jump_mm
            > self.max_translation_jump_mm
        ):

            return False

        if (
            rotation_jump_deg
            > self.max_rotation_jump_deg
        ):

            return False

        if (
            linear_speed_mm_s
            > self.max_linear_speed_mm_s
        ):

            return False

        if (
            angular_speed_deg_s
            > self.max_angular_speed_deg_s
        ):

            return False

        return True

    # ========================================================
    # Transform -> PoseStamped
    # ========================================================

    def transform_to_pose_msg(
        self,
        transform,
        source_header,
    ):

        transform = validate_transform(
            transform
        )

        quaternion = (
            rotation_matrix_to_quaternion(
                transform[
                    :3,
                    :3,
                ]
            )
        )

        msg = PoseStamped()

        msg.header = (
            source_header
        )

        # Project convention: millimeters.
        msg.pose.position.x = float(
            transform[
                0,
                3,
            ]
        )

        msg.pose.position.y = float(
            transform[
                1,
                3,
            ]
        )

        msg.pose.position.z = float(
            transform[
                2,
                3,
            ]
        )

        msg.pose.orientation.x = float(
            quaternion[0]
        )

        msg.pose.orientation.y = float(
            quaternion[1]
        )

        msg.pose.orientation.z = float(
            quaternion[2]
        )

        msg.pose.orientation.w = float(
            quaternion[3]
        )

        return msg

    # ========================================================
    # Image Callback
    # ========================================================

    def image_callback(
        self,
        msg,
    ):

        if (
            self.camera_matrix
            is None
            or self.detector
            is None
        ):

            return

        # ----------------------------------------------------
        # ROS timestamp -> seconds
        # ----------------------------------------------------

        timestamp_s = (
            float(
                msg.header.stamp.sec
            )
            + float(
                msg.header.stamp.nanosec
            )
            * 1e-9
        )

        # ----------------------------------------------------
        # ROS Image -> BGR
        # ----------------------------------------------------

        try:

            image_bgr = (
                self.ros_image_to_bgr(
                    msg
                )
            )

        except Exception as error:

            self.get_logger().error(
                "Could not convert camera image: "
                f"{error}"
            )

            return

        # ----------------------------------------------------
        # Detect board
        # ----------------------------------------------------

        detection = (
            self.detect_charuco(
                image_bgr
            )
        )

        if (
            detection["num_markers"]
            < self.min_markers
        ):

            return

        # ----------------------------------------------------
        # Estimate marker pose
        # ----------------------------------------------------

        pose = self.estimate_pose(
            detection
        )

        if pose is None:

            return

        # ----------------------------------------------------
        # Marker frame convention
        # ----------------------------------------------------

        try:

            T_camera_marker = (
                self.make_marker_z_point_into_board(
                    pose["transform"]
                )
            )

        except Exception:

            return

        # ----------------------------------------------------
        # Marker -> Cleaning Head
        # ----------------------------------------------------

        T_camera_cleaning_head = (
            T_camera_marker
            @ self.T_marker_cleaning_head
        )

        try:

            T_camera_cleaning_head = (
                validate_transform(
                    T_camera_cleaning_head,
                    "T_camera_cleaning_head",
                )
            )

        except Exception:

            return

        # ----------------------------------------------------
        # Quality checks
        # ----------------------------------------------------

        if not self.passes_quality_gate(
            detection,
            pose,
            T_camera_cleaning_head,
            timestamp_s,
        ):

            return

        # ----------------------------------------------------
        # Filter marker pose
        # ----------------------------------------------------

        try:

            filtered_T_camera_marker = (
                self.pose_filter.update(
                    T_camera_marker,
                    timestamp_s,
                )
            )

        except Exception:

            self.pose_filter.reset()

            filtered_T_camera_marker = (
                T_camera_marker
            )

        # ----------------------------------------------------
        # Compose cleaning pose AFTER filtering marker pose
        #
        # This preserves the exact rigid marker/head relation.
        # ----------------------------------------------------

        filtered_T_camera_cleaning_head = (
            filtered_T_camera_marker
            @ self.T_marker_cleaning_head
        )

        # ----------------------------------------------------
        # Save accepted raw history for quality checks
        # ----------------------------------------------------

        self.previous_timestamp = (
            timestamp_s
        )

        self.previous_cleaning_pose = (
            T_camera_cleaning_head.copy()
        )

        # ----------------------------------------------------
        # Publish marker pose
        # ----------------------------------------------------

        marker_msg = (
            self.transform_to_pose_msg(
                filtered_T_camera_marker,
                msg.header,
            )
        )

        self.marker_pose_publisher.publish(
            marker_msg
        )

        # ----------------------------------------------------
        # Publish cleaning-head pose
        # ----------------------------------------------------

        cleaning_msg = (
            self.transform_to_pose_msg(
                filtered_T_camera_cleaning_head,
                msg.header,
            )
        )

        self.cleaning_pose_publisher.publish(
            cleaning_msg
        )

        # ----------------------------------------------------
        # Publish reprojection error
        # ----------------------------------------------------

        error_msg = Float32()

        error_msg.data = float(
            pose["mean_error"]
        )

        self.error_publisher.publish(
            error_msg
        )


# ============================================================
# Main
# ============================================================

def main(args=None):

    rclpy.init(
        args=args
    )

    node = None

    try:

        node = CharucoTrackerNode()

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    except Exception as error:

        if node is not None:

            node.get_logger().error(
                "ChArUco tracker failed: "
                f"{error}"
            )

        else:

            print(
                "[ERROR] ChArUco tracker failed: "
                f"{error}"
            )

        raise

    finally:

        if node is not None:

            node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


if __name__ == "__main__":

    main()