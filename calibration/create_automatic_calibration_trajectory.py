#!/usr/bin/env python3
"""
Create an autonomous SO-101 calibration trajectory from keyboard-selected
configurations while showing a relaxed live ZED LEFT ChArUco preview.

This file is intentionally self-contained on the camera side. It does not
import collect_eye_to_hand_data_charuco4x4.py. The camera helper section can
later be moved into a dedicated camera.py module in the calibration folder.

INTERPOLATION GROUPS
--------------------
Not every saved configuration must be connected to the next saved
configuration.

    s : save the current measured configuration in the CURRENT group
    n : start a NEW interpolation group

Interpolation occurs only between consecutive configurations inside the same
group. No interpolation is generated between the end of one group and the
start of the next group.

Example
-------
Save A, B, C in group 0:
    A -> B -> C is interpolated.

Press n, then save D and E in group 1:
    D -> E is interpolated.

There is NO generated interpolation between C and D.

Every selected configuration is preserved exactly in the final trajectory.
The remaining points needed to reach --points are distributed among the
allowed interpolation segments according to joint-space segment length.

RELAXED CAMERA PREVIEW
----------------------
The preview is advisory only:

    - Any detected ArUco marker is reported.
    - Available ChArUco corners are drawn.
    - A board pose/axis is drawn when enough corners are available.
    - Reprojection thresholds and detection streaks are not used.
    - Pressing s is never blocked by camera detection.

The operator is responsible for deciding whether the board is sufficiently
visible before saving a configuration.

KEYBOARD CONTROLS
-----------------
TAB         select the next joint
LEFT        move the selected joint negative by STEP_DEG
RIGHT       move the selected joint positive by STEP_DEG
h           move smoothly to robot.py home
r           move smoothly to robot.py rest
p           print current FK from measured motor feedback
s           save configuration in the current interpolation group
n           finish current group and start a new group
u           undo the most recently saved configuration
q           finish and generate the trajectory
x           abort without generating a trajectory

UP, DOWN, and mouse events are deliberately ignored. Some terminals translate
mouse-wheel scrolling into UP/DOWN escape sequences.

SAFETY
------
Every saved and generated configuration is validated against the joint
limits imported from robot.py. Meaningful violations are rejected. Only tiny
encoder or floating-point boundary overshoot is clipped to the exact limit.

This script still does not perform collision checking. Every allowed
interpolation segment must be physically safe. The autonomous collector may
move directly from the end of one group to the start of the next group, but it
will not collect interpolated calibration points during that transition.
"""

import argparse
import csv
import curses
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl

from lerobot.robots.so_follower import (
    SO101Follower,
    SO101FollowerConfig,
)


# ============================================================
# Project paths and robot imports
# ============================================================

ROOT = Path("/home/agbara-admin/Documents/Surgical_Cleaning_Robot")
UTIL_PATH = ROOT / "robot_control" / "Util"
sys.path.insert(0, str(UTIL_PATH))

from fk import space_product_of_exponentials
from robot import (
    M,
    S_list,
    JOINT_OFFSETS_DEG,
    theta_min_robot_deg,
    theta_max_robot_deg,
    home as robot_home,
    rest as robot_rest,
)


# ============================================================
# Robot and output configuration
# ============================================================

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "dbot"

DEFAULT_NUM_POINTS = 200

DEFAULT_OUTPUT_JSON = (
    ROOT
    / "data"
    / "eye_to_hand"
    / "automatic_calibration_trajectory.json"
)

JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

DISPLAY_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

JOINT_LIMITS = {
    name: (
        float(theta_min_robot_deg[index]),
        float(theta_max_robot_deg[index]),
    )
    for index, name in enumerate(JOINT_NAMES)
}

STEP_DEG = 1.0
COMMAND_DELAY_SECONDS = 0.03

FEEDBACK_WINDOW_SAMPLES = 10
FEEDBACK_SAMPLE_DELAY_SECONDS = 0.03

# Values farther outside a robot limit than this are rejected. Values only
# slightly outside because of encoder quantization or floating-point noise are
# clipped back to the exact limit.
JOINT_LIMIT_TOLERANCE_DEG = 0.05

home = dict(robot_home)
rest = dict(robot_rest)
current_action = dict(rest)


# ============================================================
# Self-contained ZED LEFT camera configuration
# ============================================================

ZED_RESOLUTION = sl.RESOLUTION.HD2K
ZED_FPS = 15
PREVIEW_WINDOW_NAME = "ZED LEFT - ChArUco Trajectory Preview"

# Printed 4x4 ChArUco board dimensions.
SQUARES_X = 4
SQUARES_Y = 4
SQUARE_LENGTH_M = 0.0482
MARKER_LENGTH_M = 0.0361
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50

MAX_CHARUCO_CORNERS = (
    (SQUARES_X - 1)
    * (SQUARES_Y - 1)
)

# This threshold is used only to attempt a visual pose/axis.
# It does not control whether a configuration can be saved.
MIN_CHARUCO_CORNERS_FOR_POSE = 4
AXIS_LENGTH_M = 0.05


# ============================================================
# Generic helpers
# ============================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def array_to_configuration(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(6)

    return {
        name: float(values[index])
        for index, name in enumerate(JOINT_NAMES)
    }


def configuration_to_array(configuration):
    return np.array(
        [
            float(configuration[name])
            for name in JOINT_NAMES
        ],
        dtype=np.float64,
    )



def validate_configuration_within_joint_limits(
    configuration,
    context="configuration",
    tolerance_deg=JOINT_LIMIT_TOLERANCE_DEG,
):
    """
    Validate one six-joint configuration against robot.py limits.

    Returns a new dictionary guaranteed to satisfy every configured limit.
    Missing, non-finite, or meaningfully out-of-range values raise ValueError.
    Only tiny overshoot within tolerance_deg is clipped to the exact boundary.
    """

    missing = [
        name
        for name in JOINT_NAMES
        if name not in configuration
    ]

    if missing:
        raise ValueError(
            f"{context} is missing joints: "
            + ", ".join(missing)
        )

    validated = {}

    for name in JOINT_NAMES:
        value = float(configuration[name])
        lower, upper = JOINT_LIMITS[name]

        if not np.isfinite(value):
            raise ValueError(
                f"{context}: {name} is not finite."
            )

        if (
            value < lower - tolerance_deg
            or value > upper + tolerance_deg
        ):
            raise ValueError(
                f"{context}: {name}={value:.6f} deg is outside "
                f"[{lower:.6f}, {upper:.6f}] deg."
            )

        validated[name] = float(
            np.clip(value, lower, upper)
        )

    return validated


def validate_array_within_joint_limits(
    values,
    context="joint array",
    tolerance_deg=JOINT_LIMIT_TOLERANCE_DEG,
):
    configuration = array_to_configuration(values)

    validated = validate_configuration_within_joint_limits(
        configuration,
        context=context,
        tolerance_deg=tolerance_deg,
    )

    return configuration_to_array(validated)


def validate_groups_within_joint_limits(groups):
    """Validate every manually selected configuration."""

    for group in groups:
        group_index = int(group["group_index"])

        for configuration_index, configuration in enumerate(
            group["configurations"]
        ):
            configuration["joints_deg"] = (
                validate_configuration_within_joint_limits(
                    configuration["joints_deg"],
                    context=(
                        f"group {group_index}, manual configuration "
                        f"{configuration_index}"
                    ),
                )
            )

    return groups


def validate_generated_trajectory_within_joint_limits(generated):
    """Final hard safety gate for every generated trajectory point."""

    for sample_index, sample in enumerate(generated):
        sample["joints_deg"] = (
            validate_configuration_within_joint_limits(
                sample["joints_deg"],
                context=(
                    f"generated trajectory sample {sample_index}"
                ),
            )
        )

    return generated


def joint_limit_clearance_report(generated):
    """
    Return the minimum distance from the trajectory to each lower/upper limit.
    """

    values = np.array(
        [
            configuration_to_array(
                sample["joints_deg"]
            )
            for sample in generated
        ],
        dtype=np.float64,
    )

    report = {}

    for index, name in enumerate(JOINT_NAMES):
        lower, upper = JOINT_LIMITS[name]

        lower_clearance = (
            values[:, index] - lower
        )
        upper_clearance = (
            upper - values[:, index]
        )

        report[name] = {
            "minimum_lower_limit_clearance_deg": float(
                np.min(lower_clearance)
            ),
            "minimum_upper_limit_clearance_deg": float(
                np.min(upper_clearance)
            ),
            "minimum_any_limit_clearance_deg": float(
                np.min(
                    np.minimum(
                        lower_clearance,
                        upper_clearance,
                    )
                )
            ),
        }

    return report


def save_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(temporary, "w") as file:
        json.dump(payload, file, indent=4)

    temporary.replace(path)


def finite_float_or_none(value):
    if value is None:
        return None

    value = float(value)

    if not np.isfinite(value):
        return None

    return value


def count_saved_configurations(groups):
    return sum(
        len(group["configurations"])
        for group in groups
    )


def remove_empty_groups(groups):
    cleaned = [
        group
        for group in groups
        if group["configurations"]
    ]

    for group_index, group in enumerate(cleaned):
        group["group_index"] = int(group_index)

        for configuration_index, configuration in enumerate(
            group["configurations"]
        ):
            configuration["group_index"] = int(group_index)
            configuration["configuration_index"] = int(
                configuration_index
            )

    return cleaned


# ============================================================
# Robot feedback and movement
# ============================================================

def read_feedback_once(robot):
    observation = robot.get_observation()

    missing = [
        name
        for name in JOINT_NAMES
        if name not in observation
    ]

    if missing:
        raise RuntimeError(
            "robot.get_observation() is missing: "
            + ", ".join(missing)
        )

    return np.array(
        [
            float(observation[name])
            for name in JOINT_NAMES
        ],
        dtype=np.float64,
    )


def read_feedback_median(
    robot,
    samples=FEEDBACK_WINDOW_SAMPLES,
    delay=FEEDBACK_SAMPLE_DELAY_SECONDS,
):
    measurements = []

    for sample_index in range(samples):
        measurements.append(
            read_feedback_once(robot)
        )

        if sample_index + 1 < samples:
            time.sleep(delay)

    measurements = np.asarray(
        measurements,
        dtype=np.float64,
    )

    return {
        "median_deg": np.median(
            measurements,
            axis=0,
        ),
        "mean_deg": np.mean(
            measurements,
            axis=0,
        ),
        "std_deg": np.std(
            measurements,
            axis=0,
        ),
        "minimum_deg": np.min(
            measurements,
            axis=0,
        ),
        "maximum_deg": np.max(
            measurements,
            axis=0,
        ),
    }


def send_and_save_pose(robot, action):
    final_action = {}

    for name in JOINT_NAMES:
        lower, upper = JOINT_LIMITS[name]

        final_action[name] = float(
            clamp(
                float(action[name]),
                lower,
                upper,
            )
        )

    robot.send_action(final_action)
    return final_action


def initialize_current_action_from_feedback(robot):
    global current_action

    measured = read_feedback_once(robot)

    current_action = validate_configuration_within_joint_limits(
        array_to_configuration(measured),
        context="initial measured robot feedback",
    )

    return measured


def move_smooth(robot, target_action):
    global current_action

    final_action = dict(current_action)

    for name in JOINT_NAMES:
        if name not in target_action:
            continue

        lower, upper = JOINT_LIMITS[name]

        final_action[name] = clamp(
            float(target_action[name]),
            lower,
            upper,
        )

    current = configuration_to_array(
        current_action
    )
    target = configuration_to_array(
        final_action
    )

    difference = target - current
    maximum_difference = float(
        np.max(np.abs(difference))
    )

    number_of_steps = max(
        1,
        int(
            np.ceil(
                maximum_difference
                / STEP_DEG
            )
        ),
    )

    for step_index in range(
        1,
        number_of_steps + 1,
    ):
        alpha = (
            step_index
            / number_of_steps
        )

        intermediate = (
            current
            + alpha * difference
        )

        action = array_to_configuration(
            intermediate
        )

        current_action = send_and_save_pose(
            robot,
            action,
        )

        time.sleep(
            COMMAND_DELAY_SECONDS
        )

    current_action = dict(final_action)


def robot_deg_to_model_deg(theta_robot_deg):
    theta_robot_deg = np.asarray(
        theta_robot_deg,
        dtype=np.float64,
    ).reshape(6)

    return (
        theta_robot_deg
        - np.asarray(
            JOINT_OFFSETS_DEG,
            dtype=np.float64,
        ).reshape(6)
    )


def configuration_to_T_base_ee(configuration):
    theta_robot_deg = configuration_to_array(
        configuration
    )

    theta_model_rad = np.radians(
        robot_deg_to_model_deg(
            theta_robot_deg
        )
    )

    return space_product_of_exponentials(
        M,
        S_list,
        theta_model_rad,
    )


def print_current_fk(robot):
    measured_deg = read_feedback_once(robot)

    configuration = array_to_configuration(
        measured_deg
    )

    T_base_ee = configuration_to_T_base_ee(
        configuration
    )

    print("\n" + "=" * 72)
    print("CURRENT ROBOT FK FROM MOTOR FEEDBACK")
    print("=" * 72)

    for name, value in zip(
        JOINT_NAMES,
        measured_deg,
    ):
        print(
            f"  {name:20s}: "
            f"{value: .3f} deg"
        )

    print("\nT_base_to_ee:")
    print(T_base_ee)
    print("=" * 72 + "\n")


def disable_terminal_mouse_reporting():
    try:
        sys.stdout.write(
            "\x1b[?1000l"
            "\x1b[?1002l"
            "\x1b[?1003l"
            "\x1b[?1006l"
        )
        sys.stdout.flush()
    except Exception:
        pass


# ============================================================
# ZED LEFT camera helpers
# ============================================================

def open_zed_camera():
    """
    Open the ZED for synchronized image capture without depth.
    """

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = ZED_RESOLUTION
    init_params.camera_fps = ZED_FPS
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.coordinate_system = (
        sl.COORDINATE_SYSTEM.IMAGE
    )

    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(
            f"ZED open failed: {status}"
        )

    camera_information = (
        zed.get_camera_information()
    )
    camera_configuration = (
        camera_information
        .camera_configuration
    )

    print("[INFO] ZED opened successfully.")
    print("[INFO] Actual ZED mode:")
    print(
        "  width:  "
        f"{camera_configuration.resolution.width}"
    )
    print(
        "  height: "
        f"{camera_configuration.resolution.height}"
    )
    print(
        "  fps:    "
        f"{camera_configuration.fps}"
    )
    print("[INFO] Image: rectified ZED LEFT")
    print("[INFO] Depth mode: NONE")

    return zed


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


def create_charuco_board():
    """
    Create the printed 4x4 ChArUco board and a compatible detector.
    """

    aruco = cv2.aruco

    dictionary = (
        aruco.getPredefinedDictionary(
            ARUCO_DICT_ID
        )
    )

    if hasattr(aruco, "CharucoBoard"):
        board = aruco.CharucoBoard(
            (
                SQUARES_X,
                SQUARES_Y,
            ),
            SQUARE_LENGTH_M,
            MARKER_LENGTH_M,
            dictionary,
        )
    elif hasattr(
        aruco,
        "CharucoBoard_create",
    ):
        board = aruco.CharucoBoard_create(
            SQUARES_X,
            SQUARES_Y,
            SQUARE_LENGTH_M,
            MARKER_LENGTH_M,
            dictionary,
        )
    else:
        raise RuntimeError(
            "OpenCV ChArUco support is unavailable. "
            "Install opencv-contrib-python."
        )

    charuco_detector = None

    if hasattr(
        aruco,
        "CharucoDetector",
    ):
        charuco_detector = (
            aruco.CharucoDetector(board)
        )

    return (
        board,
        dictionary,
        charuco_detector,
    )


def grab_left_bgr_and_gray(
    zed,
    runtime,
    image_zed,
):
    if (
        zed.grab(runtime)
        != sl.ERROR_CODE.SUCCESS
    ):
        return None, None

    zed.retrieve_image(
        image_zed,
        sl.VIEW.LEFT,
    )

    frame = image_zed.get_data()

    if frame is None:
        return None, None

    if (
        frame.ndim == 3
        and frame.shape[2] == 4
    ):
        frame_bgr = cv2.cvtColor(
            frame,
            cv2.COLOR_BGRA2BGR,
        )
    else:
        frame_bgr = frame.copy()

    gray = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    return frame_bgr, gray


def make_detector_parameters():
    aruco = cv2.aruco

    if hasattr(
        aruco,
        "DetectorParameters",
    ):
        return aruco.DetectorParameters()

    if hasattr(
        aruco,
        "DetectorParameters_create",
    ):
        return (
            aruco.DetectorParameters_create()
        )

    return None


def detect_charuco_relaxed(
    gray,
    board,
    dictionary,
    charuco_detector,
    K,
    distortion,
):
    """
    Detect available markers and ChArUco corners without imposing a save gate.

    The returned `board_detected` value means that at least one board marker was
    found. Pose estimation is optional and may fail without affecting saving.
    """

    aruco = cv2.aruco

    marker_corners = None
    marker_ids = None
    charuco_corners = None
    charuco_ids = None

    if charuco_detector is not None:
        (
            charuco_corners,
            charuco_ids,
            marker_corners,
            marker_ids,
        ) = charuco_detector.detectBoard(gray)

    else:
        detector_parameters = (
            make_detector_parameters()
        )

        marker_corners, marker_ids, _ = (
            aruco.detectMarkers(
                gray,
                dictionary,
                parameters=detector_parameters,
            )
        )

        if (
            marker_ids is not None
            and len(marker_ids) > 0
        ):
            (
                _,
                charuco_corners,
                charuco_ids,
            ) = aruco.interpolateCornersCharuco(
                marker_corners,
                marker_ids,
                gray,
                board,
                K,
                distortion,
            )

    marker_count = (
        0
        if marker_ids is None
        else int(len(marker_ids))
    )

    charuco_count = (
        0
        if charuco_corners is None
        else int(len(charuco_corners))
    )

    result = {
        "board_detected": bool(
            marker_count > 0
        ),
        "marker_count": marker_count,
        "charuco_count": charuco_count,
        "marker_corners": marker_corners,
        "marker_ids": marker_ids,
        "charuco_corners": (
            charuco_corners
        ),
        "charuco_ids": charuco_ids,
        "T_camera_to_board": None,
        "rvec": None,
        "tvec": None,
        "mean_reprojection_px": None,
        "max_reprojection_px": None,
        "pose_available": False,
    }

    if (
        charuco_corners is None
        or charuco_ids is None
        or charuco_count
        < MIN_CHARUCO_CORNERS_FOR_POSE
    ):
        return result

    object_points = None
    image_points = None

    if hasattr(
        board,
        "matchImagePoints",
    ):
        try:
            (
                object_points,
                image_points,
            ) = board.matchImagePoints(
                charuco_corners,
                charuco_ids,
            )
        except Exception:
            object_points = None
            image_points = None

    if (
        object_points is not None
        and image_points is not None
    ):
        object_points = np.asarray(
            object_points,
            dtype=np.float64,
        ).reshape(-1, 3)

        image_points = np.asarray(
            image_points,
            dtype=np.float64,
        ).reshape(-1, 2)

        try:
            success, rvec, tvec = (
                cv2.solvePnP(
                    object_points,
                    image_points,
                    K,
                    distortion,
                    flags=(
                        cv2.SOLVEPNP_ITERATIVE
                    ),
                )
            )
        except Exception:
            success = False
            rvec = None
            tvec = None

        if success:
            projected_points, _ = (
                cv2.projectPoints(
                    object_points,
                    rvec,
                    tvec,
                    K,
                    distortion,
                )
            )

            projected_points = (
                np.asarray(
                    projected_points,
                    dtype=np.float64,
                ).reshape(-1, 2)
            )

            errors = np.linalg.norm(
                image_points
                - projected_points,
                axis=1,
            )

            rotation, _ = cv2.Rodrigues(
                rvec
            )

            transform = np.eye(
                4,
                dtype=np.float64,
            )
            transform[:3, :3] = rotation
            transform[:3, 3] = (
                np.asarray(
                    tvec,
                    dtype=np.float64,
                ).reshape(3)
            )

            result.update(
                {
                    "T_camera_to_board": (
                        transform
                    ),
                    "rvec": rvec,
                    "tvec": tvec,
                    "mean_reprojection_px": (
                        float(np.mean(errors))
                    ),
                    "max_reprojection_px": (
                        float(np.max(errors))
                    ),
                    "pose_available": True,
                }
            )

            return result

    # Old OpenCV fallback.
    if hasattr(
        aruco,
        "estimatePoseCharucoBoard",
    ):
        try:
            (
                success,
                rvec,
                tvec,
            ) = aruco.estimatePoseCharucoBoard(
                charuco_corners,
                charuco_ids,
                board,
                K,
                distortion,
                None,
                None,
            )
        except Exception:
            success = False
            rvec = None
            tvec = None

        if success:
            rotation, _ = cv2.Rodrigues(
                rvec
            )

            transform = np.eye(
                4,
                dtype=np.float64,
            )
            transform[:3, :3] = rotation
            transform[:3, 3] = (
                np.asarray(
                    tvec,
                    dtype=np.float64,
                ).reshape(3)
            )

            result.update(
                {
                    "T_camera_to_board": (
                        transform
                    ),
                    "rvec": rvec,
                    "tvec": tvec,
                    "pose_available": True,
                }
            )

    return result


def draw_charuco_relaxed(
    frame_bgr,
    result,
    K,
    distortion,
):
    """
    Draw every available detection without applying quality restrictions.
    """

    aruco = cv2.aruco

    marker_corners = result[
        "marker_corners"
    ]
    marker_ids = result["marker_ids"]

    if (
        marker_corners is not None
        and marker_ids is not None
    ):
        try:
            aruco.drawDetectedMarkers(
                frame_bgr,
                marker_corners,
                marker_ids,
            )
        except Exception:
            pass

    charuco_corners = result[
        "charuco_corners"
    ]
    charuco_ids = result[
        "charuco_ids"
    ]

    if charuco_corners is not None:
        try:
            aruco.drawDetectedCornersCharuco(
                frame_bgr,
                charuco_corners,
                charuco_ids,
            )
        except Exception:
            points = np.asarray(
                charuco_corners
            ).reshape(-1, 2)

            for point in points:
                cv2.circle(
                    frame_bgr,
                    (
                        int(round(point[0])),
                        int(round(point[1])),
                    ),
                    5,
                    (255, 0, 255),
                    thickness=-1,
                )

    if result["pose_available"]:
        try:
            cv2.drawFrameAxes(
                frame_bgr,
                K,
                distortion,
                result["rvec"],
                result["tvec"],
                AXIS_LENGTH_M,
            )
        except Exception:
            pass


def draw_preview_overlay(
    frame_bgr,
    result,
    groups,
):
    """
    Display detection information only. No state shown here blocks saving.
    """

    frame = frame_bgr
    height, width = frame.shape[:2]

    if result["charuco_count"] > 0:
        banner = (
            "CHARUCO DETECTED - "
            "OPERATOR MAY SAVE"
        )
        banner_color = (0, 190, 0)

    elif result["marker_count"] > 0:
        banner = (
            "ARUCO MARKERS DETECTED - "
            "OPERATOR MAY SAVE"
        )
        banner_color = (0, 190, 255)

    else:
        banner = (
            "BOARD NOT DETECTED - "
            "S STILL ALLOWED"
        )
        banner_color = (0, 0, 220)

    cv2.rectangle(
        frame,
        (0, 0),
        (width, 112),
        (20, 20, 20),
        thickness=-1,
    )

    cv2.putText(
        frame,
        banner,
        (25, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        banner_color,
        2,
        cv2.LINE_AA,
    )

    reprojection = (
        result[
            "mean_reprojection_px"
        ]
    )

    reprojection_text = (
        "N/A"
        if reprojection is None
        else f"{reprojection:.3f} px"
    )

    information = (
        f"Markers: {result['marker_count']}   "
        f"ChArUco corners: "
        f"{result['charuco_count']}/"
        f"{MAX_CHARUCO_CORNERS}   "
        f"Pose reprojection: "
        f"{reprojection_text}"
    )

    cv2.putText(
        frame,
        information,
        (25, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    group_index = len(groups) - 1
    group_count = len(
        groups[-1]["configurations"]
    )
    total_count = (
        count_saved_configurations(groups)
    )

    lower_text = (
        f"Current group: {group_index}   "
        f"Configurations in group: "
        f"{group_count}   "
        f"Total saved: {total_count}   "
        f"s=save, n=new group"
    )

    cv2.putText(
        frame,
        lower_text,
        (25, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.51,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    return frame


def update_camera_preview(
    zed,
    runtime,
    image_zed,
    board,
    dictionary,
    charuco_detector,
    K,
    distortion,
    groups,
):
    frame_bgr, gray = (
        grab_left_bgr_and_gray(
            zed,
            runtime,
            image_zed,
        )
    )

    if frame_bgr is None:
        result = {
            "board_detected": False,
            "marker_count": 0,
            "charuco_count": 0,
            "marker_corners": None,
            "marker_ids": None,
            "charuco_corners": None,
            "charuco_ids": None,
            "T_camera_to_board": None,
            "rvec": None,
            "tvec": None,
            "mean_reprojection_px": None,
            "max_reprojection_px": None,
            "pose_available": False,
        }

        return result, None

    result = detect_charuco_relaxed(
        gray,
        board,
        dictionary,
        charuco_detector,
        K,
        distortion,
    )

    draw_charuco_relaxed(
        frame_bgr,
        result,
        K,
        distortion,
    )

    annotated = draw_preview_overlay(
        frame_bgr,
        result,
        groups,
    )

    cv2.imshow(
        PREVIEW_WINDOW_NAME,
        annotated,
    )
    cv2.waitKey(1)

    return result, annotated


# ============================================================
# Curses display
# ============================================================

def safe_addstr(
    window,
    row,
    column,
    text,
    attributes=0,
):
    height, width = (
        window.getmaxyx()
    )

    if (
        row < 0
        or row >= height
        or column < 0
        or column >= width
    ):
        return

    available = (
        width
        - column
        - 1
    )

    if available <= 0:
        return

    clipped = str(text)[:available]

    try:
        window.addstr(
            row,
            column,
            clipped,
            attributes,
        )
    except curses.error:
        pass


def draw_terminal(
    stdscr,
    measured_deg,
    active_joint_index,
    groups,
    camera_result,
    status_message,
):
    stdscr.erase()

    current_group_index = (
        len(groups) - 1
    )
    current_group_count = len(
        groups[-1]["configurations"]
    )
    total_count = (
        count_saved_configurations(groups)
    )

    safe_addstr(
        stdscr,
        0,
        0,
        (
            "SO-101 CALIBRATION TRAJECTORY "
            "+ RELAXED ZED LEFT PREVIEW"
        ),
        curses.A_BOLD,
    )

    safe_addstr(
        stdscr,
        2,
        0,
        "Robot controls:",
        curses.A_BOLD,
    )
    safe_addstr(
        stdscr,
        3,
        2,
        "TAB         : switch joint",
    )
    safe_addstr(
        stdscr,
        4,
        2,
        "LEFT/RIGHT  : move selected joint",
    )
    safe_addstr(
        stdscr,
        5,
        2,
        "h / r       : home / rest",
    )
    safe_addstr(
        stdscr,
        6,
        2,
        "p           : print measured FK",
    )

    safe_addstr(
        stdscr,
        8,
        0,
        "Trajectory controls:",
        curses.A_BOLD,
    )
    safe_addstr(
        stdscr,
        9,
        2,
        (
            "s : save configuration in "
            "CURRENT interpolation group"
        ),
    )
    safe_addstr(
        stdscr,
        10,
        2,
        (
            "n : start NEW group; no "
            "interpolation across boundary"
        ),
    )
    safe_addstr(
        stdscr,
        11,
        2,
        "u : undo most recent saved configuration",
    )
    safe_addstr(
        stdscr,
        12,
        2,
        "q : finish and generate trajectory",
    )
    safe_addstr(
        stdscr,
        13,
        2,
        "x : abort",
    )

    if camera_result["charuco_count"] > 0:
        detection_text = (
            "ChArUco corners detected"
        )
    elif camera_result["marker_count"] > 0:
        detection_text = (
            "ArUco markers detected"
        )
    else:
        detection_text = (
            "board not detected"
        )

    safe_addstr(
        stdscr,
        15,
        0,
        (
            f"Camera preview       : "
            f"{detection_text}"
        ),
        curses.A_BOLD,
    )

    safe_addstr(
        stdscr,
        16,
        0,
        (
            f"Markers / corners    : "
            f"{camera_result['marker_count']} / "
            f"{camera_result['charuco_count']}"
        ),
    )

    safe_addstr(
        stdscr,
        17,
        0,
        (
            "Camera save policy   : advisory only; "
            "s is never blocked"
        ),
    )

    safe_addstr(
        stdscr,
        19,
        0,
        (
            f"Current group        : "
            f"{current_group_index}"
        ),
        curses.A_BOLD,
    )
    safe_addstr(
        stdscr,
        20,
        0,
        (
            f"Configs in group     : "
            f"{current_group_count}"
        ),
    )
    safe_addstr(
        stdscr,
        21,
        0,
        (
            f"Total configs saved  : "
            f"{total_count}"
        ),
    )
    safe_addstr(
        stdscr,
        22,
        0,
        (
            "Mouse / UP / DOWN    : "
            "ignored for safety"
        ),
    )

    safe_addstr(
        stdscr,
        24,
        0,
        "Current joints:",
        curses.A_BOLD,
    )

    for index, name in enumerate(
        JOINT_NAMES
    ):
        row = 26 + index
        lower, upper = (
            JOINT_LIMITS[name]
        )

        marker = (
            "->"
            if index
            == active_joint_index
            else "  "
        )

        attributes = (
            curses.A_REVERSE
            | curses.A_BOLD
            if index
            == active_joint_index
            else 0
        )

        safe_addstr(
            stdscr,
            row,
            0,
            (
                f"{marker} "
                f"{DISPLAY_NAMES[index]:15s}: "
                f"measured={measured_deg[index]:8.2f} "
                f"command={current_action[name]:8.2f} "
                f"limits=[{lower:.1f}, {upper:.1f}]"
            ),
            attributes,
        )

    safe_addstr(
        stdscr,
        34,
        0,
        status_message,
        curses.A_BOLD,
    )

    safe_addstr(
        stdscr,
        36,
        0,
        (
            "Keep the terminal focused. "
            "The OpenCV window is visual only."
        ),
    )

    stdscr.refresh()


# ============================================================
# Keyboard configuration capture
# ============================================================

def create_empty_group(group_index):
    return {
        "group_index": int(group_index),
        "configurations": [],
    }


def undo_last_configuration(groups):
    """
    Undo the most recently saved configuration across all groups.
    """

    while (
        len(groups) > 1
        and not groups[-1]["configurations"]
    ):
        groups.pop()

    if not groups[-1]["configurations"]:
        return None

    removed = (
        groups[-1]["configurations"].pop()
    )

    image_path = (
        removed
        .get("camera_detection", {})
        .get("annotated_preview_image")
    )

    if image_path:
        image_file = Path(image_path)

        if image_file.exists():
            image_file.unlink()

    return removed


def capture_configurations_with_keyboard(
    stdscr,
    robot,
    zed,
    runtime,
    image_zed,
    board,
    dictionary,
    charuco_detector,
    K,
    distortion,
    preview_directory,
    requested_points,
):
    global current_action

    try:
        curses.curs_set(0)
    except curses.error:
        pass

    stdscr.nodelay(True)
    stdscr.keypad(True)
    stdscr.timeout(30)

    try:
        curses.mousemask(0)
    except curses.error:
        pass

    active_joint_index = 0

    groups = [
        create_empty_group(0)
    ]

    status_message = (
        "Select configurations. "
        "Press n to break interpolation."
    )

    measured_deg = (
        initialize_current_action_from_feedback(
            robot
        )
    )

    camera_result = {
        "board_detected": False,
        "marker_count": 0,
        "charuco_count": 0,
        "marker_corners": None,
        "marker_ids": None,
        "charuco_corners": None,
        "charuco_ids": None,
        "T_camera_to_board": None,
        "rvec": None,
        "tvec": None,
        "mean_reprojection_px": None,
        "max_reprojection_px": None,
        "pose_available": False,
    }

    latest_annotated_frame = None

    while True:
        try:
            measured_deg = (
                read_feedback_once(robot)
            )
        except Exception as exception:
            status_message = (
                "Robot feedback error: "
                f"{exception}"
            )

        (
            camera_result,
            latest_annotated_frame,
        ) = update_camera_preview(
            zed,
            runtime,
            image_zed,
            board,
            dictionary,
            charuco_detector,
            K,
            distortion,
            groups,
        )

        draw_terminal(
            stdscr,
            measured_deg,
            active_joint_index,
            groups,
            camera_result,
            status_message,
        )

        key = stdscr.getch()

        if (
            key == -1
            or key == curses.KEY_RESIZE
        ):
            continue

        if key == getattr(
            curses,
            "KEY_MOUSE",
            -9999,
        ):
            status_message = (
                "Mouse event ignored."
            )
            continue

        if key in (
            curses.KEY_UP,
            curses.KEY_DOWN,
        ):
            status_message = (
                "UP/DOWN ignored. "
                "Use LEFT/RIGHT."
            )
            continue

        if key == ord("q"):
            cleaned_groups = (
                remove_empty_groups(groups)
            )

            total_configurations = (
                count_saved_configurations(
                    cleaned_groups
                )
            )

            if total_configurations < 2:
                status_message = (
                    "Save at least two configurations "
                    "before finishing."
                )
                continue

            if total_configurations > requested_points:
                status_message = (
                    f"You saved {total_configurations} configurations, "
                    f"but --points={requested_points}. "
                    "Undo configurations or restart with a larger --points."
                )
                continue

            has_interpolation_edge = any(
                len(group["configurations"])
                >= 2
                for group in cleaned_groups
            )

            if (
                requested_points
                > total_configurations
                and not has_interpolation_edge
            ):
                status_message = (
                    "No interpolation edge exists. "
                    "Put at least two configurations "
                    "in one group."
                )
                continue

            return cleaned_groups

        elif key == ord("x"):
            raise KeyboardInterrupt(
                "Trajectory capture aborted "
                "by the operator."
            )

        elif key == ord("h"):
            status_message = (
                "Moving smoothly to HOME..."
            )

            draw_terminal(
                stdscr,
                measured_deg,
                active_joint_index,
                groups,
                camera_result,
                status_message,
            )

            move_smooth(robot, home)

            status_message = (
                "Reached HOME."
            )

        elif key == ord("r"):
            status_message = (
                "Moving smoothly to REST..."
            )

            draw_terminal(
                stdscr,
                measured_deg,
                active_joint_index,
                groups,
                camera_result,
                status_message,
            )

            move_smooth(robot, rest)

            status_message = (
                "Reached REST."
            )

        elif key == ord("p"):
            print_current_fk(robot)

            status_message = (
                "Printed measured FK."
            )

        elif key == ord("n"):
            if not groups[-1][
                "configurations"
            ]:
                status_message = (
                    "Current group is empty. "
                    "Save a configuration first."
                )
                continue

            new_group_index = len(groups)

            groups.append(
                create_empty_group(
                    new_group_index
                )
            )

            status_message = (
                f"Started interpolation group "
                f"{new_group_index}. "
                "No interpolation will connect "
                "the previous group to this one."
            )

        elif key == ord("s"):
            status_message = (
                "Reading stable robot feedback..."
            )

            draw_terminal(
                stdscr,
                measured_deg,
                active_joint_index,
                groups,
                camera_result,
                status_message,
            )

            feedback = read_feedback_median(
                robot
            )
            saved_deg = feedback[
                "median_deg"
            ]
            std_deg = feedback[
                "std_deg"
            ]

            # HARD SAFETY GATE:
            # The saved waypoint comes from measured motor feedback, so the
            # measured configuration itself must be within robot.py limits.
            try:
                saved_deg = validate_array_within_joint_limits(
                    saved_deg,
                    context="measured configuration selected with s",
                )
            except ValueError as exception:
                status_message = (
                    "Configuration not saved: "
                    f"{exception}"
                )
                continue

            current_group = groups[-1]
            group_index = current_group[
                "group_index"
            ]
            configuration_index = len(
                current_group[
                    "configurations"
                ]
            )

            # Avoid accidental duplicate saves inside one group.
            if current_group[
                "configurations"
            ]:
                previous = (
                    configuration_to_array(
                        current_group[
                            "configurations"
                        ][-1]["joints_deg"]
                    )
                )

                separation = float(
                    np.linalg.norm(
                        saved_deg - previous
                    )
                )

                if separation < 0.5:
                    status_message = (
                        "Configuration not saved: "
                        "less than 0.5 deg from the "
                        "previous configuration in "
                        "this group."
                    )
                    continue

            preview_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            preview_path = (
                preview_directory
                / (
                    f"group_{group_index:02d}_"
                    f"config_{configuration_index:03d}.png"
                )
            )

            if latest_annotated_frame is not None:
                cv2.imwrite(
                    str(preview_path),
                    latest_annotated_frame,
                )

            T_camera_to_board = (
                camera_result[
                    "T_camera_to_board"
                ]
            )

            camera_transform_list = (
                None
                if T_camera_to_board is None
                else np.asarray(
                    T_camera_to_board,
                    dtype=np.float64,
                ).reshape(4, 4).tolist()
            )

            saved_configuration = {
                "group_index": int(
                    group_index
                ),
                "configuration_index": int(
                    configuration_index
                ),
                "joints_deg": (
                    array_to_configuration(
                        saved_deg
                    )
                ),
                "feedback_std_deg": (
                    array_to_configuration(
                        std_deg
                    )
                ),
                "camera_detection": {
                    "advisory_only": True,
                    "board_detected": bool(
                        camera_result[
                            "board_detected"
                        ]
                    ),
                    "marker_count": int(
                        camera_result[
                            "marker_count"
                        ]
                    ),
                    "charuco_count": int(
                        camera_result[
                            "charuco_count"
                        ]
                    ),
                    "pose_available": bool(
                        camera_result[
                            "pose_available"
                        ]
                    ),
                    "mean_reprojection_px": (
                        finite_float_or_none(
                            camera_result[
                                "mean_reprojection_px"
                            ]
                        )
                    ),
                    "max_reprojection_px": (
                        finite_float_or_none(
                            camera_result[
                                "max_reprojection_px"
                            ]
                        )
                    ),
                    "T_camera_to_board_snapshot": (
                        camera_transform_list
                    ),
                    "annotated_preview_image": (
                        str(preview_path)
                    ),
                },
            }

            current_group[
                "configurations"
            ].append(
                saved_configuration
            )

            current_action = (
                array_to_configuration(
                    saved_deg
                )
            )

            detection_note = (
                "board visible"
                if camera_result[
                    "board_detected"
                ]
                else "board not detected; "
                "saved by operator judgement"
            )

            status_message = (
                f"Saved group {group_index}, "
                f"configuration "
                f"{configuration_index}; "
                f"{detection_note}. "
                f"Max feedback std: "
                f"{np.max(std_deg):.3f} deg."
            )

        elif key == ord("u"):
            removed = undo_last_configuration(
                groups
            )

            if removed is None:
                status_message = (
                    "Nothing to undo."
                )
            else:
                status_message = (
                    "Removed group "
                    f"{removed['group_index']}, "
                    "configuration "
                    f"{removed['configuration_index']}."
                )

        elif key == 9:
            active_joint_index = (
                active_joint_index + 1
            ) % len(JOINT_NAMES)

            status_message = (
                "Selected "
                f"{DISPLAY_NAMES[active_joint_index]}."
            )

        elif key in (
            curses.KEY_LEFT,
            curses.KEY_RIGHT,
        ):
            joint = JOINT_NAMES[
                active_joint_index
            ]

            direction = (
                -1.0
                if key == curses.KEY_LEFT
                else 1.0
            )

            lower, upper = (
                JOINT_LIMITS[joint]
            )

            current_action[joint] = clamp(
                current_action[joint]
                + direction * STEP_DEG,
                lower,
                upper,
            )

            current_action = (
                send_and_save_pose(
                    robot,
                    dict(current_action),
                )
            )

            time.sleep(
                COMMAND_DELAY_SECONDS
            )

            status_message = (
                f"{DISPLAY_NAMES[active_joint_index]} "
                f"command: "
                f"{current_action[joint]:.2f} deg."
            )

        else:
            status_message = (
                f"Unrecognized key ignored: "
                f"{key}"
            )


# ============================================================
# Explicit interpolation-group trajectory generation
# ============================================================

def build_allowed_edges(groups):
    edges = []

    for group in groups:
        configurations = group[
            "configurations"
        ]

        for edge_index in range(
            len(configurations) - 1
        ):
            start_configuration = (
                configurations[edge_index]
            )
            end_configuration = (
                configurations[
                    edge_index + 1
                ]
            )

            start_array = (
                configuration_to_array(
                    start_configuration[
                        "joints_deg"
                    ]
                )
            )
            end_array = (
                configuration_to_array(
                    end_configuration[
                        "joints_deg"
                    ]
                )
            )

            length = float(
                np.linalg.norm(
                    end_array
                    - start_array
                )
            )

            if length <= 1e-12:
                raise ValueError(
                    "An interpolation group contains "
                    "two identical consecutive "
                    "configurations."
                )

            edges.append(
                {
                    "global_edge_index": int(
                        len(edges)
                    ),
                    "group_index": int(
                        group["group_index"]
                    ),
                    "edge_index_in_group": int(
                        edge_index
                    ),
                    "start_configuration_index": int(
                        edge_index
                    ),
                    "end_configuration_index": int(
                        edge_index + 1
                    ),
                    "start": start_array,
                    "end": end_array,
                    "length_deg": length,
                }
            )

    return edges


def allocate_extra_intervals(
    edges,
    extra_intervals,
):
    """
    Allocate extra intervals by edge length using largest remainders.

    Every edge already receives one interval so every manually selected
    configuration appears exactly in the generated trajectory.
    """

    allocations = np.zeros(
        len(edges),
        dtype=int,
    )

    if extra_intervals <= 0:
        return allocations

    lengths = np.array(
        [
            edge["length_deg"]
            for edge in edges
        ],
        dtype=np.float64,
    )

    total_length = float(
        np.sum(lengths)
    )

    if total_length <= 1e-12:
        raise ValueError(
            "Cannot distribute interpolation "
            "points over zero-length edges."
        )

    exact = (
        extra_intervals
        * lengths
        / total_length
    )

    base = np.floor(exact).astype(int)
    allocations += base

    remaining = int(
        extra_intervals
        - int(np.sum(base))
    )

    if remaining > 0:
        remainders = exact - base
        order = np.argsort(
            -remainders,
            kind="stable",
        )

        for index in order[:remaining]:
            allocations[index] += 1

    return allocations


def generate_grouped_trajectory(
    groups,
    num_points,
):
    """
    Generate exactly num_points without interpolating across group boundaries.

    Every manually saved configuration is included exactly once.
    """

    groups = remove_empty_groups(
        groups
    )

    groups = validate_groups_within_joint_limits(
        groups
    )

    total_anchors = (
        count_saved_configurations(
            groups
        )
    )

    if total_anchors == 0:
        raise ValueError(
            "No configurations were saved."
        )

    if num_points < total_anchors:
        raise ValueError(
            f"--points={num_points} is smaller "
            f"than the {total_anchors} manually "
            "selected configurations. Increase "
            "--points or save fewer configurations."
        )

    edges = build_allowed_edges(
        groups
    )

    extra_intervals = (
        num_points
        - total_anchors
    )

    if (
        extra_intervals > 0
        and not edges
    ):
        raise ValueError(
            "Additional points were requested, but "
            "no interpolation group contains two "
            "configurations."
        )

    extra_allocations = (
        allocate_extra_intervals(
            edges,
            extra_intervals,
        )
    )

    interval_count_by_edge = {}

    for edge, extra in zip(
        edges,
        extra_allocations,
    ):
        interval_count_by_edge[
            (
                edge["group_index"],
                edge[
                    "edge_index_in_group"
                ],
            )
        ] = int(1 + extra)

    generated = []
    global_sample_index = 0

    for group in groups:
        group_index = int(
            group["group_index"]
        )
        configurations = group[
            "configurations"
        ]

        if len(configurations) == 1:
            generated.append(
                {
                    "sample_index": int(
                        global_sample_index
                    ),
                    "interpolation_group": int(
                        group_index
                    ),
                    "edge_index_in_group": None,
                    "segment_alpha": None,
                    "is_manual_configuration": True,
                    "manual_configuration_index": 0,
                    "joints_deg": dict(
                        configurations[0][
                            "joints_deg"
                        ]
                    ),
                }
            )

            global_sample_index += 1
            continue

        for edge_index in range(
            len(configurations) - 1
        ):
            start = (
                configuration_to_array(
                    configurations[
                        edge_index
                    ]["joints_deg"]
                )
            )
            end = (
                configuration_to_array(
                    configurations[
                        edge_index + 1
                    ]["joints_deg"]
                )
            )

            interval_count = (
                interval_count_by_edge[
                    (
                        group_index,
                        edge_index,
                    )
                ]
            )

            # Include alpha=0 for the first edge only.
            first_step = (
                0
                if edge_index == 0
                else 1
            )

            for step_index in range(
                first_step,
                interval_count + 1,
            ):
                alpha = (
                    step_index
                    / interval_count
                )

                point = (
                    start
                    + alpha * (end - start)
                )

                # Linear interpolation between in-limit endpoints should remain
                # in range. Keep this explicit check as a hard safety gate.
                point = validate_array_within_joint_limits(
                    point,
                    context=(
                        f"generated point in group {group_index}, "
                        f"edge {edge_index}, alpha={alpha:.9f}"
                    ),
                )

                is_start_anchor = (
                    edge_index == 0
                    and step_index == 0
                )
                is_end_anchor = (
                    step_index
                    == interval_count
                )

                manual_index = None

                if is_start_anchor:
                    manual_index = 0
                elif is_end_anchor:
                    manual_index = (
                        edge_index + 1
                    )

                generated.append(
                    {
                        "sample_index": int(
                            global_sample_index
                        ),
                        "interpolation_group": int(
                            group_index
                        ),
                        "edge_index_in_group": int(
                            edge_index
                        ),
                        "segment_alpha": float(
                            alpha
                        ),
                        "is_manual_configuration": bool(
                            manual_index
                            is not None
                        ),
                        "manual_configuration_index": (
                            manual_index
                        ),
                        "joints_deg": (
                            array_to_configuration(
                                point
                            )
                        ),
                    }
                )

                global_sample_index += 1

    if len(generated) != num_points:
        raise RuntimeError(
            "Internal trajectory allocation error: "
            f"generated {len(generated)} points, "
            f"expected {num_points}."
        )

    generated = validate_generated_trajectory_within_joint_limits(
        generated
    )

    return (
        generated,
        edges,
        interval_count_by_edge,
    )


# ============================================================
# FK diversity report
# ============================================================

def rotation_distance_deg(
    rotation_a,
    rotation_b,
):
    relative = (
        np.asarray(rotation_a).T
        @ np.asarray(rotation_b)
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
        np.degrees(
            np.arccos(cosine)
        )
    )


def calculate_fk_diversity(
    generated,
):
    transforms = [
        configuration_to_T_base_ee(
            sample["joints_deg"]
        )
        for sample in generated
    ]

    pairwise_rotations = []

    for first in range(
        len(transforms)
    ):
        for second in range(
            first + 1,
            len(transforms),
        ):
            pairwise_rotations.append(
                rotation_distance_deg(
                    transforms[first][
                        :3,
                        :3,
                    ],
                    transforms[second][
                        :3,
                        :3,
                    ],
                )
            )

    positions = np.array(
        [
            transform[:3, 3]
            for transform in transforms
        ],
        dtype=np.float64,
    )

    joint_values = np.array(
        [
            configuration_to_array(
                sample["joints_deg"]
            )
            for sample in generated
        ],
        dtype=np.float64,
    )

    return {
        "pairwise_ee_rotation_deg": {
            "minimum": float(
                np.min(
                    pairwise_rotations
                )
            ),
            "median": float(
                np.median(
                    pairwise_rotations
                )
            ),
            "maximum": float(
                np.max(
                    pairwise_rotations
                )
            ),
        },
        "ee_position_range_m": {
            axis: [
                float(
                    np.min(
                        positions[
                            :,
                            axis_index,
                        ]
                    )
                ),
                float(
                    np.max(
                        positions[
                            :,
                            axis_index,
                        ]
                    )
                ),
            ]
            for axis_index, axis in enumerate(
                ["x", "y", "z"]
            )
        },
        "joint_span_deg": {
            name: float(
                np.max(
                    joint_values[
                        :,
                        index,
                    ]
                )
                - np.min(
                    joint_values[
                        :,
                        index,
                    ]
                )
            )
            for index, name in enumerate(
                JOINT_NAMES
            )
        },
    }


# ============================================================
# Save trajectory
# ============================================================

def flatten_manual_configurations(
    groups,
):
    flattened = []

    for group in groups:
        for configuration in group[
            "configurations"
        ]:
            flattened.append(
                configuration
            )

    return flattened


def save_trajectory(
    output_json,
    groups,
    generated,
    edges,
    interval_count_by_edge,
    diversity,
):
    output_json = Path(output_json)
    output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    groups = validate_groups_within_joint_limits(
        groups
    )
    generated = validate_generated_trajectory_within_joint_limits(
        generated
    )
    limit_clearance = joint_limit_clearance_report(
        generated
    )

    edge_metadata = []

    for edge in edges:
        edge_metadata.append(
            {
                "group_index": int(
                    edge["group_index"]
                ),
                "edge_index_in_group": int(
                    edge[
                        "edge_index_in_group"
                    ]
                ),
                "start_configuration_index": int(
                    edge[
                        "start_configuration_index"
                    ]
                ),
                "end_configuration_index": int(
                    edge[
                        "end_configuration_index"
                    ]
                ),
                "joint_space_length_deg": float(
                    edge["length_deg"]
                ),
                "generated_intervals": int(
                    interval_count_by_edge[
                        (
                            edge["group_index"],
                            edge[
                                "edge_index_in_group"
                            ],
                        )
                    ]
                ),
            }
        )

    payload = {
        "format_version": 6,
        "description": (
            "SO-101 calibration trajectory with "
            "operator-defined interpolation groups "
            "and advisory ZED LEFT ChArUco preview."
        ),
        "units": (
            "robot motor-feedback degrees"
        ),
        "joint_order": JOINT_NAMES,
        "number_of_interpolation_groups": int(
            len(groups)
        ),
        "number_of_manual_configurations": int(
            count_saved_configurations(
                groups
            )
        ),
        "number_of_samples": int(
            len(generated)
        ),
        "sampling_method": (
            "all manually selected configurations "
            "preserved; extra points allocated by "
            "joint-space edge length within explicit "
            "interpolation groups only"
        ),
        "keyboard_controller": {
            "movement_keys": [
                "LEFT",
                "RIGHT",
            ],
            "home_key": "h",
            "rest_key": "r",
            "save_configuration_key": "s",
            "new_interpolation_group_key": "n",
            "undo_key": "u",
            "step_deg": float(STEP_DEG),
            "mouse_events_ignored": True,
            "up_down_keys_ignored": True,
        },
        "camera_preview": {
            "self_contained": True,
            "view": "ZED rectified LEFT",
            "save_gate_enabled": False,
            "operator_judgement": True,
            "squares_x": int(SQUARES_X),
            "squares_y": int(SQUARES_Y),
            "square_length_m": float(
                SQUARE_LENGTH_M
            ),
            "marker_length_m": float(
                MARKER_LENGTH_M
            ),
            "minimum_corners_for_optional_pose_axis": int(
                MIN_CHARUCO_CORNERS_FOR_POSE
            ),
        },
        "joint_limits_deg": {
            name: [
                float(
                    JOINT_LIMITS[name][0]
                ),
                float(
                    JOINT_LIMITS[name][1]
                ),
            ]
            for name in JOINT_NAMES
        },
        "joint_limit_validation": {
            "passed": True,
            "tolerance_deg": float(
                JOINT_LIMIT_TOLERANCE_DEG
            ),
            "out_of_limit_behavior": (
                "reject meaningful violations; clip only tiny "
                "encoder or floating-point boundary overshoot"
            ),
            "validation_stages": [
                "manual measured waypoint save",
                "every interpolated point",
                "final JSON and CSV write",
            ],
            "minimum_limit_clearance_by_joint": (
                limit_clearance
            ),
        },
        "interpolation_groups": groups,
        "waypoint_configurations": (
            flatten_manual_configurations(
                groups
            )
        ),
        "allowed_interpolation_edges": (
            edge_metadata
        ),
        "generated_configurations": (
            generated
        ),
        "predicted_fk_diversity": (
            diversity
        ),
    }

    save_json_atomic(
        output_json,
        payload,
    )

    csv_path = (
        output_json.with_suffix(".csv")
    )

    with open(
        csv_path,
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "sample_index",
                "interpolation_group",
                "edge_index_in_group",
                "segment_alpha",
                "is_manual_configuration",
                "manual_configuration_index",
                *JOINT_NAMES,
            ]
        )

        for sample in generated:
            writer.writerow(
                [
                    sample[
                        "sample_index"
                    ],
                    sample[
                        "interpolation_group"
                    ],
                    sample[
                        "edge_index_in_group"
                    ],
                    sample[
                        "segment_alpha"
                    ],
                    sample[
                        "is_manual_configuration"
                    ],
                    sample[
                        "manual_configuration_index"
                    ],
                    *[
                        sample[
                            "joints_deg"
                        ][name]
                        for name in JOINT_NAMES
                    ],
                ]
            )

    return output_json, csv_path


def print_summary(
    json_path,
    csv_path,
    preview_directory,
    groups,
    generated,
    edges,
    diversity,
):
    pairwise = diversity[
        "pairwise_ee_rotation_deg"
    ]

    print("\n" + "=" * 72)
    print("GROUPED CALIBRATION TRAJECTORY CREATED")
    print("=" * 72)

    print(
        "Interpolation groups:       "
        f"{len(groups)}"
    )
    print(
        "Manual configurations:      "
        f"{count_saved_configurations(groups)}"
    )
    print(
        "Allowed interpolation edges:"
        f" {len(edges)}"
    )
    print(
        "Generated configurations:   "
        f"{len(generated)}"
    )
    print(f"JSON:                       {json_path}")
    print(f"CSV:                        {csv_path}")
    print(
        "Annotated previews:         "
        f"{preview_directory}"
    )

    print(
        "\nPredicted end-effector "
        "rotation diversity:"
    )
    print(
        "  Pairwise minimum: "
        f"{pairwise['minimum']:.3f} deg"
    )
    print(
        "  Pairwise median:  "
        f"{pairwise['median']:.3f} deg"
    )
    print(
        "  Pairwise maximum: "
        f"{pairwise['maximum']:.3f} deg"
    )

    if pairwise["maximum"] < 20.0:
        print(
            "\n[WARNING] Maximum predicted "
            "orientation separation is under "
            "20 degrees."
        )

    print(
        "\nNo interpolation was created "
        "between different groups."
    )
    print(
        "Camera detection was advisory only; "
        "the operator was allowed to save "
        "every selected configuration."
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Capture operator-defined SO-101 "
            "interpolation groups with a relaxed "
            "self-contained ZED LEFT preview."
        )
    )

    parser.add_argument(
        "--points",
        type=int,
        default=DEFAULT_NUM_POINTS,
        help=(
            "Total generated configurations. "
            "Default: 50."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=(
            "Output trajectory JSON path."
        ),
    )

    parser.add_argument(
        "--port",
        default=ROBOT_PORT,
        help=(
            "SO-101 serial port. "
            f"Default: {ROBOT_PORT}"
        ),
    )

    parser.add_argument(
        "--robot-id",
        default=ROBOT_ID,
        help=(
            "LeRobot robot ID. "
            f"Default: {ROBOT_ID}"
        ),
    )

    args = parser.parse_args()

    validate_configuration_within_joint_limits(
        home,
        context="robot.py home configuration",
        tolerance_deg=0.0,
    )

    validate_configuration_within_joint_limits(
        rest,
        context="robot.py rest configuration",
        tolerance_deg=0.0,
    )

    if args.points < 2:
        raise ValueError(
            "--points must be at least 2."
        )

    preview_directory = (
        args.output.parent
        / (
            "automatic_calibration_"
            "configuration_previews"
        )
    )

    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.port,
            id=args.robot_id,
        )
    )

    zed = None
    robot_connected = False

    try:
        print(
            "[INFO] Connecting to SO-101..."
        )

        robot.connect(
            calibrate=False
        )
        robot_connected = True

        print(
            "[INFO] Opening ZED LEFT camera..."
        )

        zed = open_zed_camera()

        runtime = sl.RuntimeParameters()
        image_zed = sl.Mat()

        K, distortion = (
            get_zed_left_intrinsics_rectified(
                zed
            )
        )

        (
            board,
            dictionary,
            charuco_detector,
        ) = create_charuco_board()

        print(
            "[INFO] ZED LEFT intrinsic matrix:"
        )
        print(
            np.array2string(
                K,
                precision=6,
                suppress_small=True,
            )
        )

        cv2.namedWindow(
            PREVIEW_WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )

        disable_terminal_mouse_reporting()

        groups = curses.wrapper(
            capture_configurations_with_keyboard,
            robot,
            zed,
            runtime,
            image_zed,
            board,
            dictionary,
            charuco_detector,
            K,
            distortion,
            preview_directory,
            args.points,
        )

        disable_terminal_mouse_reporting()

        (
            generated,
            edges,
            interval_count_by_edge,
        ) = generate_grouped_trajectory(
            groups,
            args.points,
        )

        generated = validate_generated_trajectory_within_joint_limits(
            generated
        )

        diversity = calculate_fk_diversity(
            generated
        )

        (
            json_path,
            csv_path,
        ) = save_trajectory(
            output_json=args.output,
            groups=groups,
            generated=generated,
            edges=edges,
            interval_count_by_edge=(
                interval_count_by_edge
            ),
            diversity=diversity,
        )

        print_summary(
            json_path=json_path,
            csv_path=csv_path,
            preview_directory=(
                preview_directory
            ),
            groups=groups,
            generated=generated,
            edges=edges,
            diversity=diversity,
        )

    except KeyboardInterrupt as exception:
        print(
            f"\n[ABORTED] {exception}"
        )
        print(
            "No new trajectory was generated."
        )

    finally:
        disable_terminal_mouse_reporting()
        cv2.destroyAllWindows()

        if zed is not None:
            try:
                zed.close()
            except Exception:
                pass

        if robot_connected:
            try:
                robot.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()

