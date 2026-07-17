#!/usr/bin/env python3

"""
Eye-to-hand calibration data collection for the SO-101 + ZED setup.

This version uses the ZED as a normal monocular camera:
    - LEFT rectified image only
    - no ZED depth
    - no ZED point cloud
    - ChArUco 4x4 board pose from solvePnP

For each saved calibration sample:
    1. The robot is stationary.
    2. T_base_to_ee is computed from measured motor feedback/encoder positions
       using robot.get_observation(), not only from the commanded action.
    3. T_camera_to_board is averaged over a short camera window.
    4. The OpenCV-ready rotation and translation lists are appended to:
        - R_ee_base.json
        - t_ee_base.json
        - R_base_ee.json
        - t_base_ee.json
        - R_cam_board.json
        - t_cam_board.json
        - R_board_cam.json
        - t_board_cam.json

Each transformation file contains only one JSON list. No metadata, frame labels,
or debugging information is written into these eight files.

Important frame definitions:
    T_base_to_ee = ^B T_E:
        maps end-effector coordinates into the robot-base frame

    T_camera_to_board = ^C T_W:
        maps ChArUco-board coordinates into the ZED LEFT optical frame

    T_board_to_camera = ^W T_C:
        rigid inverse of T_camera_to_board; maps ZED LEFT camera coordinates
        into the ChArUco-board frame

Camera convention for ZED LEFT / OpenCV image coordinates:
    +X = right in image
    +Y = down in image
    +Z = forward away from camera
"""

import sys
import time
import curses
import json
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


# ============================================================
# Project paths
# ============================================================

ROOT = Path("/home/agbara-admin/Documents/Surgical_Cleaning_Robot")
UTIL_PATH = ROOT / "robot_control" / "Util"
sys.path.insert(0, str(UTIL_PATH))

from so3 import RToQuaternion
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
# Files / folders
# ============================================================

OUTPUT_DIR = ROOT / "data" / "eye_to_hand"

# Set False to create only the six calibration transformation files below.
# Set True to also create current-pose, FK, camera-pose, and metadata files.
SAVE_DEBUG_FILES = True

# Calibration files. Each file contains only a JSON list.
R_EE_BASE_FILE = OUTPUT_DIR / "R_ee_base.json"
T_EE_BASE_FILE = OUTPUT_DIR / "t_ee_base.json"
R_BASE_EE_FILE = OUTPUT_DIR / "R_base_ee.json"
T_BASE_EE_FILE = OUTPUT_DIR / "t_base_ee.json"
R_CAM_BOARD_FILE = OUTPUT_DIR / "R_cam_board.json"
T_CAM_BOARD_FILE = OUTPUT_DIR / "t_cam_board.json"
R_BOARD_CAM_FILE = OUTPUT_DIR / "R_board_cam.json"
T_BOARD_CAM_FILE = OUTPUT_DIR / "t_board_cam.json"

# Optional debugging files.
POSE_FILE = OUTPUT_DIR / "current_robot_pose.json"
FK_FILE = OUTPUT_DIR / "current_robot_fk.json"
CURRENT_CAMERA_BOARD_FILE = OUTPUT_DIR / "current_camera_board.json"
SAMPLE_METADATA_FILE = OUTPUT_DIR / "calibration_sample_metadata.json"


# ============================================================
# Robot configuration
# ============================================================

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "dbot"

STEP_DEG = 1.0
COMMAND_DELAY = 0.03


# ============================================================
# ZED configuration
# ============================================================

# Use the same resolution/FPS for calibration collection that you plan to use
# when computing camera intrinsics. The intrinsics are read after opening the
# camera, so they correspond to this exact mode.
ZED_RESOLUTION = sl.RESOLUTION.HD1080
ZED_FPS = 15


# ============================================================
# 4x4 ChArUco board settings
# ============================================================

# Your 4x4 board is intended to be 200 mm x 200 mm.
# After printing, you measured the square length as about 48.2 mm.
# If your marker measurement differs, update MARKER_LENGTH_M.
SQUARES_X = 4
SQUARES_Y = 4
SQUARE_LENGTH_M = 0.0482     # measured square length, meters
MARKER_LENGTH_M = 0.0361     # approx measured marker length, meters
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50

# 4x4 ChArUco has (4-1)*(4-1) = 9 inner ChArUco corners.
MAX_CHARUCO_CORNERS = (SQUARES_X - 1) * (SQUARES_Y - 1)

# For final calibration, require all 9 corners before accepting a sample.
MIN_CHARUCO_CORNERS_DETECT = 7
MIN_CHARUCO_CORNERS_SAVE = 9

# Quality thresholds for saving camera-window detections.
MAX_MEAN_REPROJ_ERROR_PX = 0.6
MAX_MAX_REPROJ_ERROR_PX = 1.2

# Camera averaging window used when the user presses 's'.
CAMERA_SETTLE_SECONDS = 0.50
CAMERA_AVG_SECONDS = 1.50
MIN_VALID_CAMERA_FRAMES = 10

# Visualization only.
AXIS_LENGTH_M = 0.05


# ============================================================
# Joint definitions
# ============================================================

joint_names = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

joint_labels = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

joint_limits = {
    name: (
        float(theta_min_robot_deg[i]),
        float(theta_max_robot_deg[i]),
    )
    for i, name in enumerate(joint_names)
}

home = dict(robot_home)
rest = dict(robot_rest)
current_action = dict(rest)


# ============================================================
# File cleanup
# ============================================================

def delete_old_calibration_samples():
    """
    Delete old calibration files before starting a new collection session.

    This keeps the JSON files aligned so the Nth robot pose corresponds to
    the Nth camera pose.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    patterns = [
        "R_ee_base.json",
        "t_ee_base.json",
        "R_base_ee.json",
        "t_base_ee.json",
        "R_cam_board.json",
        "t_cam_board.json",
        "R_board_cam.json",
        "t_board_cam.json",
        # Remove legacy calibration outputs from older collector versions.
        "robot_q.json",
        "robot_t.json",
        "camera_q.json",
        "camera_t.json",
        # Optional debugging outputs.
        "current_robot_pose.json",
        "current_robot_fk.json",
        "calibration_sample_metadata.json",
        "current_camera_board.json",
        "T_base_to_camera.npy",
        "T_ee_to_board.npy",
    ]

    deleted = 0

    for pattern in patterns:
        for file in OUTPUT_DIR.glob(pattern):
            file.unlink()
            deleted += 1

    print(f"[INFO] Deleted {deleted} old calibration files.")


# ============================================================
# Basic helpers
# ============================================================

def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def action_to_theta_robot_deg(action):
    """
    Convert a robot action dictionary to physical robot command angles.
    """

    return np.array(
        [float(action[name]) for name in joint_names],
        dtype=float,
    )


def robot_deg_to_model_deg(theta_robot_deg):
    """
    Convert robot command angles to FK/model angles.

    Important for joint 6:
        robot command 50 deg = FK/model 0 deg

    Therefore:
        theta_model_deg = theta_robot_deg - JOINT_OFFSETS_DEG
    """

    theta_robot_deg = np.asarray(theta_robot_deg, dtype=float).reshape(6)
    return theta_robot_deg - JOINT_OFFSETS_DEG


def action_to_T_base_to_ee(action):
    """
    Compute T_base_to_ee from the current robot command.

    No robot-side averaging is used because current_action is commanded state.
    The camera side is averaged because image detection has frame-to-frame noise.
    """

    theta_robot_deg = action_to_theta_robot_deg(action)
    theta_model_deg = robot_deg_to_model_deg(theta_robot_deg)
    theta_model_rad = np.radians(theta_model_deg)

    T_base_to_ee = space_product_of_exponentials(
        M,
        S_list,
        theta_model_rad,
    )

    return T_base_to_ee


def get_robot_feedback_angles_deg(robot, fallback_action=None):
    """
    Read measured motor positions from the SO-101 feedback/encoder observation.

    This is the preferred robot-side source for calibration because
    current_action is only the command that was sent, while get_observation()
    reports the motor positions currently measured by the robot.

    Parameters
    ----------
    robot : SO101Follower
        Connected LeRobot SO101Follower object.

    fallback_action : dict or None
        If feedback cannot be read, this action dictionary is used as a
        fallback. Passing None makes missing feedback a hard error.

    Returns
    -------
    theta_robot_deg : np.ndarray, shape (6,)
        Motor feedback angles in robot-command degrees, ordered according to
        joint_names.
    """

    try:
        obs = robot.get_observation()
    except Exception as e:
        print(f"[WARN] robot.get_observation() failed: {e}")

        if fallback_action is None:
            raise

        print("[WARN] Falling back to commanded current_action.")
        return action_to_theta_robot_deg(fallback_action)

    values = []
    missing = []

    for name in joint_names:
        if name in obs:
            values.append(float(obs[name]))
        else:
            missing.append(name)

    if missing:
        print("[WARN] Missing joint feedback keys from robot observation:")
        for name in missing:
            print(f"  {name}")

        print("[WARN] Available observation keys:")
        if hasattr(obs, "keys"):
            for key in obs.keys():
                print(f"  {key}")

        if fallback_action is None:
            raise RuntimeError("Could not read all joint feedback values.")

        print("[WARN] Falling back to commanded current_action.")
        return action_to_theta_robot_deg(fallback_action)

    return np.array(values, dtype=float)


def feedback_to_T_base_to_ee(robot, fallback_action=None):
    """
    Compute T_base_to_ee from actual measured motor feedback.

    This should be used for saving eye-to-hand calibration samples. The camera
    sees the physical board, so the robot side should use measured motor
    positions rather than only the commanded current_action.

    Returns
    -------
    T_base_to_ee : np.ndarray, shape (4, 4)
        End-effector pose in the robot base frame.

    info : dict
        JSON-serializable debug information containing command angles, feedback
        angles, model/FK angles, and feedback-command difference.
    """

    theta_feedback_deg = get_robot_feedback_angles_deg(
        robot,
        fallback_action=fallback_action,
    )

    theta_model_deg = robot_deg_to_model_deg(theta_feedback_deg)
    theta_model_rad = np.radians(theta_model_deg)

    T_base_to_ee = space_product_of_exponentials(
        M,
        S_list,
        theta_model_rad,
    )

    info = {
        "source": "robot.get_observation() motor feedback",
        "robot_feedback_degrees": {
            name: float(theta_feedback_deg[i])
            for i, name in enumerate(joint_names)
        },
        "model_fk_degrees_from_feedback": {
            name: float(theta_model_deg[i])
            for i, name in enumerate(joint_names)
        },
        "model_fk_radians_from_feedback": {
            name: float(theta_model_rad[i])
            for i, name in enumerate(joint_names)
        },
        "joint_offsets_degrees": {
            name: float(JOINT_OFFSETS_DEG[i])
            for i, name in enumerate(joint_names)
        },
    }

    if fallback_action is not None:
        theta_command_deg = action_to_theta_robot_deg(fallback_action)
        theta_command_model_deg = robot_deg_to_model_deg(theta_command_deg)
        diff_deg = theta_feedback_deg - theta_command_deg

        info["robot_command_degrees"] = {
            name: float(theta_command_deg[i])
            for i, name in enumerate(joint_names)
        }
        info["model_fk_degrees_from_command"] = {
            name: float(theta_command_model_deg[i])
            for i, name in enumerate(joint_names)
        }
        info["feedback_minus_command_degrees"] = {
            name: float(diff_deg[i])
            for i, name in enumerate(joint_names)
        }

    return T_base_to_ee, info


def save_feedback_fk_json(T_base_to_ee, feedback_info):
    """
    Save FK information computed from measured motor feedback when debugging
    output is enabled.
    """

    if not SAVE_DEBUG_FILES:
        return

    FK_FILE.parent.mkdir(parents=True, exist_ok=True)

    fk_data = dict(feedback_info)
    fk_data["T_base_to_ee"] = T_base_to_ee.tolist()
    fk_data["ee_position_m"] = {
        "x": float(T_base_to_ee[0, 3]),
        "y": float(T_base_to_ee[1, 3]),
        "z": float(T_base_to_ee[2, 3]),
    }

    with open(FK_FILE, "w") as f:
        json.dump(fk_data, f, indent=4)


def save_fk_json(action):
    """
    Save FK information to JSON when debugging output is enabled.
    """

    if not SAVE_DEBUG_FILES:
        return

    FK_FILE.parent.mkdir(parents=True, exist_ok=True)

    theta_robot_deg = action_to_theta_robot_deg(action)
    theta_model_deg = robot_deg_to_model_deg(theta_robot_deg)
    T_base_to_ee = action_to_T_base_to_ee(action)

    fk_data = {
        "robot_command_degrees": {
            name: float(theta_robot_deg[i])
            for i, name in enumerate(joint_names)
        },
        "model_fk_degrees": {
            name: float(theta_model_deg[i])
            for i, name in enumerate(joint_names)
        },
        "joint_offsets_degrees": {
            name: float(JOINT_OFFSETS_DEG[i])
            for i, name in enumerate(joint_names)
        },
        "T_base_to_ee": T_base_to_ee.tolist(),
        "ee_position_m": {
            "x": float(T_base_to_ee[0, 3]),
            "y": float(T_base_to_ee[1, 3]),
            "z": float(T_base_to_ee[2, 3]),
        },
    }

    with open(FK_FILE, "w") as f:
        json.dump(fk_data, f, indent=4)


def save_current_pose(action):
    """
    Save current robot command pose and FK when debugging output is enabled.
    """

    if not SAVE_DEBUG_FILES:
        return

    POSE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(POSE_FILE, "w") as f:
        json.dump(action, f, indent=4)

    save_fk_json(action)


def load_current_pose_if_available():
    """
    Load the previous debug pose when debugging output is enabled.
    """

    global current_action

    if not SAVE_DEBUG_FILES:
        return

    if POSE_FILE.exists():
        with open(POSE_FILE, "r") as f:
            loaded = json.load(f)

        for name in joint_names:
            if name in loaded:
                current_action[name] = float(loaded[name])

    for name in joint_names:
        min_lim, max_lim = joint_limits[name]
        current_action[name] = clamp(
            current_action[name],
            min_lim,
            max_lim,
        )


def print_robot_transform(action, robot=None):
    """
    Print FK debug info to terminal and save FK JSON.

    If a connected robot object is provided, this prints and saves FK from
    measured motor feedback. Otherwise it falls back to the commanded action.
    """

    theta_command_deg = action_to_theta_robot_deg(action)
    theta_command_model_deg = robot_deg_to_model_deg(theta_command_deg)
    T_command = action_to_T_base_to_ee(action)

    print("\n" + "=" * 60)
    print("CURRENT ROBOT FK DEBUG")
    print("=" * 60)

    print("\nCommanded robot angles [deg]:")
    for name, value in zip(joint_names, theta_command_deg):
        print(f"  {name:20s}: {value: .3f}")

    print("\nCommanded offset-corrected FK/model angles [deg]:")
    for name, value in zip(joint_names, theta_command_model_deg):
        print(f"  {name:20s}: {value: .3f}")

    print("\nT_base_to_ee from commanded current_action:")
    print(T_command)

    if robot is not None:
        try:
            T_feedback, feedback_info = feedback_to_T_base_to_ee(
                robot,
                fallback_action=action,
            )
            save_feedback_fk_json(T_feedback, feedback_info)

            feedback_deg = np.array([
                feedback_info["robot_feedback_degrees"][name]
                for name in joint_names
            ], dtype=float)

            model_feedback_deg = np.array([
                feedback_info["model_fk_degrees_from_feedback"][name]
                for name in joint_names
            ], dtype=float)

            diff_deg = feedback_deg - theta_command_deg

            print("\nMeasured motor feedback angles [deg]:")
            for name, value in zip(joint_names, feedback_deg):
                print(f"  {name:20s}: {value: .3f}")

            print("\nFeedback - command [deg]:")
            for name, value in zip(joint_names, diff_deg):
                print(f"  {name:20s}: {value: .3f}")

            print("\nFeedback offset-corrected FK/model angles [deg]:")
            for name, value in zip(joint_names, model_feedback_deg):
                print(f"  {name:20s}: {value: .3f}")

            print("\nT_base_to_ee from measured motor feedback:")
            print(T_feedback)

        except Exception as e:
            print(f"\n[WARN] Could not print feedback FK: {e}")
            save_fk_json(action)
    else:
        save_fk_json(action)

    print(f"\n[SAVED FK JSON] {FK_FILE}")
    print("=" * 60 + "\n")


# ============================================================
# Transform and rotation averaging helpers
# ============================================================

def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_T(T):
    """
    Invert a rigid homogeneous transform without using np.linalg.inv().

    If T is ^A T_B, the returned matrix is ^B T_A.
    """

    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def rvec_tvec_to_T(rvec, tvec):
    """
    Convert solvePnP output to T_camera_to_board.
    """

    R, _ = cv2.Rodrigues(rvec)
    return make_T(R, tvec)


def normalize_quaternion(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)

    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    return q / n


def quaternion_to_R(q):
    """
    Convert quaternion [w, x, y, z] to a rotation matrix.
    """

    w, x, y, z = normalize_quaternion(q)

    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def average_quaternions(quaternions):
    """
    Average quaternions using the Markley eigenvector method.

    Input quaternions use [w, x, y, z].
    """

    if len(quaternions) == 0:
        raise ValueError("No quaternions to average.")

    Q = np.array([normalize_quaternion(q) for q in quaternions], dtype=np.float64)

    # Resolve sign ambiguity so all quaternions lie in roughly the same hemisphere.
    q_ref = Q[0]
    for i in range(len(Q)):
        if np.dot(Q[i], q_ref) < 0.0:
            Q[i] = -Q[i]

    A = np.zeros((4, 4), dtype=np.float64)

    for q in Q:
        A += np.outer(q, q)

    A /= len(Q)

    eigvals, eigvecs = np.linalg.eigh(A)
    q_avg = eigvecs[:, np.argmax(eigvals)]

    if q_avg[0] < 0.0:
        q_avg = -q_avg

    return normalize_quaternion(q_avg)


def average_transforms(T_list):
    """
    Average a list of homogeneous transforms.

    Translation:
        median, for robustness against outliers

    Rotation:
        quaternion mean, then convert back to rotation matrix
    """

    if len(T_list) == 0:
        raise ValueError("Cannot average an empty transform list.")

    translations = np.array([T[:3, 3] for T in T_list], dtype=np.float64)
    t_avg = np.median(translations, axis=0)

    quaternions = [RToQuaternion(T[:3, :3]) for T in T_list]
    q_avg = average_quaternions(quaternions)
    R_avg = quaternion_to_R(q_avg)

    return make_T(R_avg, t_avg)


def translation_window_stats(T_list):
    """
    Compute translation median/std in meters and millimeters.
    """

    translations = np.array([T[:3, 3] for T in T_list], dtype=np.float64)

    return {
        "median_m": np.median(translations, axis=0),
        "std_m": np.std(translations, axis=0),
        "std_mm": 1000.0 * np.std(translations, axis=0),
    }


# ============================================================
# Robot movement helpers
# ============================================================

def send_and_save_pose(robot, action):
    """
    Send action to robot, save pose JSON, and save FK JSON.
    """

    final_action = {}

    for name in joint_names:
        min_lim, max_lim = joint_limits[name]

        final_action[name] = float(
            clamp(
                float(action[name]),
                min_lim,
                max_lim,
            )
        )

    robot.send_action(final_action)
    save_current_pose(final_action)

    return final_action


def move_smooth(robot, target_action):
    """
    Smoothly move robot to target joint pose.
    """

    global current_action

    final_action = dict(current_action)

    for name in joint_names:
        if name in target_action:
            min_lim, max_lim = joint_limits[name]
            final_action[name] = clamp(float(target_action[name]), min_lim, max_lim)

    current = np.array([current_action[name] for name in joint_names], dtype=float)
    target = np.array([final_action[name] for name in joint_names], dtype=float)

    diff = target - current
    max_diff = np.max(np.abs(diff))
    n_steps = max(1, int(np.ceil(max_diff / STEP_DEG)))

    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        intermediate = current + alpha * diff

        action = {
            name: float(intermediate[idx])
            for idx, name in enumerate(joint_names)
        }

        current_action = send_and_save_pose(robot, action)
        time.sleep(COMMAND_DELAY)

    current_action = final_action
    save_current_pose(current_action)


# ============================================================
# ZED helpers
# ============================================================

def get_zed_left_intrinsics_rectified(zed):
    """
    Get rectified ZED left-camera intrinsics.

    Because the script uses sl.VIEW.LEFT, the image is rectified by the ZED SDK.
    Therefore, OpenCV solvePnP should use zero distortion coefficients.
    """

    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters.left_cam

    K = np.array(
        [
            [calib.fx, 0.0, calib.cx],
            [0.0, calib.fy, calib.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    dist = np.zeros((5, 1), dtype=np.float64)

    return K, dist


def open_zed_camera():
    """
    Open ZED as a regular camera. No depth and no point cloud.
    """

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = ZED_RESOLUTION
    init_params.camera_fps = ZED_FPS
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {status}")

    cam_info = zed.get_camera_information()
    cam_config = cam_info.camera_configuration

    print("[INFO] ZED opened successfully.")
    print("[INFO] Actual ZED mode:")
    print(f"  width:  {cam_config.resolution.width}")
    print(f"  height: {cam_config.resolution.height}")
    print(f"  fps:    {cam_config.fps}")
    print("[INFO] Depth mode: NONE")

    return zed


# ============================================================
# ChArUco pose estimation
# ============================================================

def create_charuco_board():
    """
    Create OpenCV ChArUco board matching the printed 4x4 board.
    """

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT_ID)

    if hasattr(aruco, "CharucoBoard"):
        board = aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y),
            SQUARE_LENGTH_M,
            MARKER_LENGTH_M,
            dictionary,
        )
    elif hasattr(aruco, "CharucoBoard_create"):
        board = aruco.CharucoBoard_create(
            SQUARES_X,
            SQUARES_Y,
            SQUARE_LENGTH_M,
            MARKER_LENGTH_M,
            dictionary,
        )
    else:
        raise RuntimeError(
            "OpenCV ChArUco support not found. Install opencv-contrib-python."
        )

    return board, dictionary


def compute_reprojection_error(object_points, image_points, rvec, tvec, K, dist):
    """
    Compute mean/max reprojection error in pixels.
    """

    projected_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        K,
        dist,
    )

    image_points_2d = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    projected_points_2d = np.asarray(projected_points, dtype=np.float64).reshape(-1, 2)

    errors = np.linalg.norm(image_points_2d - projected_points_2d, axis=1)

    return float(np.mean(errors)), float(np.max(errors)), projected_points_2d


def solve_planar_pnp(object_points, image_points, K, dist):
    """
    Solve board pose using planar PnP.

    IPPE is preferred because the ChArUco board is planar. RefineLM improves
    the final pose estimate when OpenCV provides it.
    """

    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)

    pnp_method = "SOLVEPNP_IPPE + RefineLM"

    try:
        ok, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
            object_points,
            image_points,
            K,
            dist,
            flags=cv2.SOLVEPNP_IPPE,
        )

        if not ok or len(rvecs) == 0:
            return None, None, "SOLVEPNP_IPPE failed"

        best_idx = int(np.argmin(np.asarray(reproj_errs).reshape(-1)))
        rvec = rvecs[best_idx]
        tvec = tvecs[best_idx]

    except Exception:
        pnp_method = "SOLVEPNP_ITERATIVE"

        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            K,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            return None, None, "SOLVEPNP_ITERATIVE failed"

    if hasattr(cv2, "solvePnPRefineLM"):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points,
                image_points,
                K,
                dist,
                rvec,
                tvec,
            )
        except Exception:
            pass

    return rvec, tvec, pnp_method


def estimate_charuco_pose(gray, board, dictionary, K, dist):
    """
    Detect the 4x4 ChArUco board and estimate T_camera_to_board.

    Returns:
        result dict on success, None on failure.
    """

    aruco = cv2.aruco

    charuco_corners = None
    charuco_ids = None
    marker_corners = None
    marker_ids = None

    # Newer OpenCV path.
    if hasattr(aruco, "CharucoDetector") and hasattr(board, "matchImagePoints"):
        detector = aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    # Older OpenCV fallback.
    else:
        detector_params = aruco.DetectorParameters()
        marker_corners, marker_ids, rejected = aruco.detectMarkers(
            gray,
            dictionary,
            parameters=detector_params,
        )

        if marker_ids is None or len(marker_ids) == 0:
            return None

        ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board,
            K,
            dist,
        )

    if charuco_corners is None or charuco_ids is None:
        return None

    num_charuco = len(charuco_corners)

    if num_charuco < MIN_CHARUCO_CORNERS_DETECT:
        return None

    if not hasattr(board, "matchImagePoints"):
        ok, rvec, tvec = aruco.estimatePoseCharucoBoard(
            charuco_corners,
            charuco_ids,
            board,
            K,
            dist,
            None,
            None,
        )

        if not ok:
            return None

        object_points = np.empty((0, 3), dtype=np.float64)
        image_points = np.empty((0, 2), dtype=np.float64)
        mean_err_px = float("nan")
        max_err_px = float("nan")
        pnp_method = "estimatePoseCharucoBoard"

    else:
        object_points, image_points = board.matchImagePoints(
            charuco_corners,
            charuco_ids,
        )

        if object_points is None or image_points is None:
            return None

        object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
        image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)

        if len(object_points) < MIN_CHARUCO_CORNERS_DETECT:
            return None

        rvec, tvec, pnp_method = solve_planar_pnp(
            object_points,
            image_points,
            K,
            dist,
        )

        if rvec is None or tvec is None:
            return None

        mean_err_px, max_err_px, projected_points = compute_reprojection_error(
            object_points,
            image_points,
            rvec,
            tvec,
            K,
            dist,
        )

    T_camera_to_board = rvec_tvec_to_T(rvec, tvec)

    return {
        "T_camera_to_board": T_camera_to_board,
        "rvec": rvec,
        "tvec": tvec,
        "pnp_method": pnp_method,
        "num_charuco": int(num_charuco),
        "marker_corners": marker_corners,
        "marker_ids": marker_ids,
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
        "object_points": object_points,
        "image_points": image_points,
        "mean_reproj_error_px": float(mean_err_px),
        "max_reproj_error_px": float(max_err_px),
    }


def is_good_camera_result(result, require_save_quality=False):
    """
    Decide whether a ChArUco pose result is good enough.

    require_save_quality=True is used during the save-window collection.
    """

    if result is None:
        return False

    min_corners = MIN_CHARUCO_CORNERS_SAVE if require_save_quality else MIN_CHARUCO_CORNERS_DETECT

    if result["num_charuco"] < min_corners:
        return False

    mean_err = result["mean_reproj_error_px"]
    max_err = result["max_reproj_error_px"]

    if np.isfinite(mean_err) and mean_err > MAX_MEAN_REPROJ_ERROR_PX:
        return False

    if np.isfinite(max_err) and max_err > MAX_MAX_REPROJ_ERROR_PX:
        return False

    return True


def draw_charuco_debug(frame_bgr, result, K, dist):
    """
    Draw markers, ChArUco corners, board axes, and board center.
    """

    if result is None:
        return

    aruco = cv2.aruco

    marker_corners = result["marker_corners"]
    marker_ids = result["marker_ids"]
    charuco_corners = result["charuco_corners"]

    if marker_corners is not None and marker_ids is not None:
        try:
            aruco.drawDetectedMarkers(frame_bgr, marker_corners, marker_ids)
        except Exception:
            pass

    if charuco_corners is not None:
        try:
            aruco.drawDetectedCornersCharuco(frame_bgr, charuco_corners, None)
        except Exception:
            pts = np.asarray(charuco_corners).reshape(-1, 2)
            for p in pts:
                cv2.circle(frame_bgr, (int(round(p[0])), int(round(p[1]))), 5, (255, 0, 255), -1)

    rvec = result["rvec"]
    tvec = result["tvec"]

    try:
        cv2.drawFrameAxes(frame_bgr, K, dist, rvec, tvec, AXIS_LENGTH_M)
    except Exception:
        pass

    # Draw the physical ChArUco pattern center.
    center_board = np.array(
        [
            0.5 * SQUARES_X * SQUARE_LENGTH_M,
            0.5 * SQUARES_Y * SQUARE_LENGTH_M,
            0.0,
        ],
        dtype=np.float64,
    ).reshape(1, 1, 3)

    center_px, _ = cv2.projectPoints(center_board, rvec, tvec, K, dist)
    u, v = center_px.reshape(2)
    cv2.circle(frame_bgr, (int(round(u)), int(round(v))), 8, (0, 255, 255), -1)
    cv2.putText(
        frame_bgr,
        "board center",
        (int(round(u)) + 10, int(round(v)) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
    )


def grab_left_bgr_and_gray(zed, runtime, image_zed):
    """
    Grab one ZED left image and return BGR + grayscale images.
    """

    if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        return None, None

    zed.retrieve_image(image_zed, sl.VIEW.LEFT)
    frame = image_zed.get_data()

    if frame.shape[2] == 4:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        frame_bgr = frame.copy()

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    return frame_bgr, gray


def collect_average_camera_pose(
    zed,
    runtime,
    image_zed,
    board,
    dictionary,
    K,
    dist,
):
    """
    Collect a short stationary camera window and average T_camera_to_board.

    This function is called when the user presses 's'. It blocks briefly while
    collecting frames. The robot/board should not move during this function.
    """

    print("\n[INFO] Saving sample: waiting for board/robot to settle...")
    time.sleep(CAMERA_SETTLE_SECONDS)

    T_list = []
    result_list = []
    start_time = time.time()
    frames_seen = 0

    while time.time() - start_time < CAMERA_AVG_SECONDS:
        frame_bgr, gray = grab_left_bgr_and_gray(zed, runtime, image_zed)

        if frame_bgr is None:
            continue

        frames_seen += 1

        result = estimate_charuco_pose(gray, board, dictionary, K, dist)

        if is_good_camera_result(result, require_save_quality=True):
            T_list.append(result["T_camera_to_board"])
            result_list.append(result)

        draw_charuco_debug(frame_bgr, result, K, dist)

        text = f"COLLECTING WINDOW | valid={len(T_list)} | frames={frames_seen}"
        cv2.putText(
            frame_bgr,
            text,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.imshow("ZED Eye-to-Hand Collection", frame_bgr)
        cv2.waitKey(1)

    if len(T_list) < MIN_VALID_CAMERA_FRAMES:
        info = {
            "success": False,
            "reason": "not enough valid camera frames",
            "frames_seen": frames_seen,
            "valid_frames": len(T_list),
            "min_valid_frames": MIN_VALID_CAMERA_FRAMES,
        }
        return None, info

    T_avg = average_transforms(T_list)
    stats = translation_window_stats(T_list)

    mean_errors = np.array(
        [r["mean_reproj_error_px"] for r in result_list if np.isfinite(r["mean_reproj_error_px"])],
        dtype=np.float64,
    )
    max_errors = np.array(
        [r["max_reproj_error_px"] for r in result_list if np.isfinite(r["max_reproj_error_px"])],
        dtype=np.float64,
    )
    corner_counts = np.array([r["num_charuco"] for r in result_list], dtype=int)

    info = {
        "success": True,
        "frames_seen": int(frames_seen),
        "valid_frames": int(len(T_list)),
        "window_seconds": float(CAMERA_AVG_SECONDS),
        "settle_seconds": float(CAMERA_SETTLE_SECONDS),
        "translation_median_m": stats["median_m"].tolist(),
        "translation_std_m": stats["std_m"].tolist(),
        "translation_std_mm": stats["std_mm"].tolist(),
        "mean_reproj_error_px_mean": float(np.mean(mean_errors)) if len(mean_errors) else None,
        "mean_reproj_error_px_max": float(np.max(mean_errors)) if len(mean_errors) else None,
        "max_reproj_error_px_max": float(np.max(max_errors)) if len(max_errors) else None,
        "charuco_corners_min": int(np.min(corner_counts)),
        "charuco_corners_max": int(np.max(corner_counts)),
        "charuco_corners_required": int(MIN_CHARUCO_CORNERS_SAVE),
    }

    return T_avg, info


# ============================================================
# Save current camera -> board transform
# ============================================================

def save_current_camera_board(T_camera_to_board, result=None):
    """
    Save the current camera-board transform when debugging output is enabled.
    """

    if not SAVE_DEBUG_FILES:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    R = T_camera_to_board[:3, :3]
    t = T_camera_to_board[:3, 3]
    q = RToQuaternion(R)

    data = {
        "quaternion": [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
        "translation": [float(t[0]), float(t[1]), float(t[2])],
        "transform": T_camera_to_board.tolist(),
    }

    if result is not None:
        data["num_charuco"] = int(result["num_charuco"])
        data["mean_reproj_error_px"] = float(result["mean_reproj_error_px"])
        data["max_reproj_error_px"] = float(result["max_reproj_error_px"])
        data["pnp_method"] = result["pnp_method"]

    with open(CURRENT_CAMERA_BOARD_FILE, "w") as f:
        json.dump(data, f, indent=4)


# ============================================================
# Calibration sample saving
# ============================================================

def append_json_list_entry(file_path, value):
    """
    Append one matrix or vector to a plain JSON list file.

    The file contains only the list itself, with no metadata wrapper.
    """

    if file_path.exists():
        with open(file_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {file_path}")
    else:
        data = []

    data.append(np.asarray(value, dtype=np.float64).tolist())

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def append_sample_metadata(metadata):
    """
    Append sample metadata only when debugging output is enabled.
    """

    if not SAVE_DEBUG_FILES:
        return

    if SAMPLE_METADATA_FILE.exists():
        with open(SAMPLE_METADATA_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(metadata)

    with open(SAMPLE_METADATA_FILE, "w") as f:
        json.dump(data, f, indent=4)


def save_sample(T_base_to_ee, T_camera_to_board, metadata=None):
    """
    Save one synchronized eye-to-hand calibration sample.

    The eight calibration files contain only these lists:
        R_ee_base.json    : list of 3x3 rotations from ^E T_B
        t_ee_base.json    : list of 3-element translations from ^E T_B
        R_base_ee.json    : list of 3x3 rotations from ^B T_E
        t_base_ee.json    : list of 3-element translations from ^B T_E
        R_cam_board.json  : list of 3x3 rotations from ^C T_W
        t_cam_board.json  : list of 3-element translations from ^C T_W
        R_board_cam.json  : list of 3x3 rotations from ^W T_C
        t_board_cam.json  : list of 3-element translations from ^W T_C

    Here W is the ChArUco board frame and C is the ZED left camera frame.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    T_base_to_ee = np.asarray(T_base_to_ee, dtype=np.float64).reshape(4, 4)
    T_camera_to_board = np.asarray(
        T_camera_to_board,
        dtype=np.float64,
    ).reshape(4, 4)

    # Robot FK gives ^B T_E. Its rigid inverse is ^E T_B.
    T_ee_to_base = invert_T(T_base_to_ee)

    R_base_ee = T_base_to_ee[:3, :3]
    t_base_ee = T_base_to_ee[:3, 3]

    R_ee_base = T_ee_to_base[:3, :3]
    t_ee_base = T_ee_to_base[:3, 3]

    # solvePnP gives ^C T_W:
    # board coordinates transformed into camera coordinates.
    R_cam_board = T_camera_to_board[:3, :3]
    t_cam_board = T_camera_to_board[:3, 3]

    # Compute the rigid inverse ^W T_C:
    # camera coordinates transformed into board coordinates.
    #
    # For a rigid transform:
    #     R_board_cam = R_cam_board.T
    #     t_board_cam = -R_cam_board.T @ t_cam_board
    #
    # Using invert_T() keeps the direction explicit and avoids mistakes.
    T_board_to_camera = invert_T(T_camera_to_board)

    R_board_cam = T_board_to_camera[:3, :3]
    t_board_cam = T_board_to_camera[:3, 3]

    append_json_list_entry(R_EE_BASE_FILE, R_ee_base)
    append_json_list_entry(T_EE_BASE_FILE, t_ee_base)
    append_json_list_entry(R_BASE_EE_FILE, R_base_ee)
    append_json_list_entry(T_BASE_EE_FILE, t_base_ee)
    append_json_list_entry(R_CAM_BOARD_FILE, R_cam_board)
    append_json_list_entry(T_CAM_BOARD_FILE, t_cam_board)
    append_json_list_entry(R_BOARD_CAM_FILE, R_board_cam)
    append_json_list_entry(T_BOARD_CAM_FILE, t_board_cam)

    if metadata is not None:
        append_sample_metadata(metadata)


# ============================================================
# Safe curses display
# ============================================================

def safe_addstr(stdscr, y, x, text):
    h, w = stdscr.getmaxyx()

    if y >= h:
        return

    max_len = w - x - 1

    if max_len <= 0:
        return

    stdscr.addstr(y, x, str(text)[:max_len])


def draw_screen(
    stdscr,
    active_joint_idx,
    detected,
    sample_count,
    num_charuco,
    mean_reproj_error,
):
    stdscr.clear()

    safe_addstr(stdscr, 0, 0, "SO-101 Eye-to-Hand Keyboard Collection")
    safe_addstr(stdscr, 1, 0, "--------------------------------------")

    safe_addstr(stdscr, 3, 0, "Controls:")
    safe_addstr(stdscr, 4, 2, "TAB         : switch joint")
    safe_addstr(stdscr, 5, 2, "LEFT arrow  : move negative")
    safe_addstr(stdscr, 6, 2, "RIGHT arrow : move positive")
    safe_addstr(stdscr, 7, 2, "h           : move home")
    safe_addstr(stdscr, 8, 2, "r           : move rest")
    safe_addstr(stdscr, 9, 2, "s           : save averaged calibration sample")
    safe_addstr(stdscr, 10, 2, "p           : print/save current T_base_to_ee")
    safe_addstr(stdscr, 11, 2, "q           : quit")

    safe_addstr(stdscr, 13, 0, f"Board pose detected : {detected}")
    safe_addstr(stdscr, 14, 0, f"ChArUco corners     : {num_charuco}/{MAX_CHARUCO_CORNERS}")
    safe_addstr(stdscr, 15, 0, f"Mean reproj error   : {mean_reproj_error}")
    safe_addstr(stdscr, 16, 0, f"Samples saved       : {sample_count}")
    safe_addstr(stdscr, 17, 0, f"Debug files         : {SAVE_DEBUG_FILES}")
    safe_addstr(stdscr, 18, 0, f"Output directory    : {OUTPUT_DIR}")
    safe_addstr(stdscr, 19, 0, "Board               : 4x4 ChArUco, rectified ZED LEFT")
    safe_addstr(stdscr, 20, 0, "Joint 6             : free / offset-corrected in FK")

    safe_addstr(stdscr, 22, 0, "Current joints:")

    for i, name in enumerate(joint_names):
        marker = "->" if i == active_joint_idx else "  "
        label = joint_labels[i]
        value = current_action[name]
        min_lim, max_lim = joint_limits[name]

        safe_addstr(
            stdscr,
            24 + i,
            0,
            f"{marker} {label:15s}: {value:8.2f} deg   limits [{min_lim:.1f}, {max_lim:.1f}]",
        )

    stdscr.refresh()


# ============================================================
# Main keyboard + camera loop
# ============================================================

def keyboard_control(stdscr, robot):
    global current_action

    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    active_joint_idx = 0

    load_current_pose_if_available()
    save_current_pose(current_action)

    # --------------------------------------------------------
    # ZED setup
    # --------------------------------------------------------

    zed = open_zed_camera()
    runtime = sl.RuntimeParameters()
    image_zed = sl.Mat()

    K, dist = get_zed_left_intrinsics_rectified(zed)

    print("[INFO] ZED left camera matrix:")
    print(np.array2string(K, precision=4, suppress_small=True))
    print("[INFO] Distortion coefficients used:")
    print(dist.reshape(-1))

    # --------------------------------------------------------
    # ChArUco setup
    # --------------------------------------------------------

    board, dictionary = create_charuco_board()

    sample_id = 0

    detected = False
    T_camera_to_board = None
    num_charuco = 0
    mean_reproj_error_text = "N/A"

    try:
        while True:
            frame_bgr, gray = grab_left_bgr_and_gray(zed, runtime, image_zed)

            if frame_bgr is not None:
                result = estimate_charuco_pose(gray, board, dictionary, K, dist)

                detected = is_good_camera_result(result, require_save_quality=False)
                T_camera_to_board = None
                num_charuco = 0
                mean_reproj_error_text = "N/A"

                if result is not None:
                    T_camera_to_board = result["T_camera_to_board"]
                    num_charuco = result["num_charuco"]
                    mean_reproj_error_text = f"{result['mean_reproj_error_px']:.3f} px"

                    save_current_camera_board(T_camera_to_board, result=result)
                    draw_charuco_debug(frame_bgr, result, K, dist)

                status_text = (
                    f"4x4 ChArUco: {detected} | "
                    f"Corners: {num_charuco}/{MAX_CHARUCO_CORNERS} | "
                    f"Reproj: {mean_reproj_error_text} | "
                    f"Samples: {sample_id}"
                )

                cv2.putText(
                    frame_bgr,
                    status_text,
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0) if detected else (0, 0, 255),
                    2,
                )

                cv2.imshow("ZED Eye-to-Hand Collection", frame_bgr)
                cv2.waitKey(1)

            draw_screen(
                stdscr,
                active_joint_idx,
                detected,
                sample_id,
                num_charuco,
                mean_reproj_error_text,
            )

            key = stdscr.getch()

            if key == -1:
                time.sleep(0.01)
                continue

            if key == ord("q"):
                break

            elif key == ord("h"):
                move_smooth(robot, home)

            elif key == ord("r"):
                move_smooth(robot, rest)

            elif key == ord("p"):
                print_robot_transform(current_action, robot=robot)

            elif key == ord("s"):
                # Save an averaged camera pose over a stationary window.
                # Robot FK is computed AFTER the camera window from measured
                # motor feedback using robot.get_observation().
                T_camera_to_board_avg, camera_info = collect_average_camera_pose(
                    zed,
                    runtime,
                    image_zed,
                    board,
                    dictionary,
                    K,
                    dist,
                )

                if T_camera_to_board_avg is None:
                    safe_addstr(
                        stdscr,
                        32,
                        0,
                        (
                            "Cannot save: "
                            f"{camera_info['valid_frames']}/"
                            f"{camera_info['frames_seen']} valid camera frames."
                        ),
                    )
                    stdscr.refresh()
                    print("\n[WARN] Sample not saved:")
                    print(json.dumps(camera_info, indent=4))
                    time.sleep(0.75)
                    continue

                # IMPORTANT:
                # Use measured motor feedback for FK, not only current_action.
                # current_action is still saved as the commanded target for debug.
                T_base_to_ee, robot_feedback_info = feedback_to_T_base_to_ee(
                    robot,
                    fallback_action=current_action,
                )

                theta_command_deg = action_to_theta_robot_deg(current_action)
                theta_command_model_deg = robot_deg_to_model_deg(theta_command_deg)

                metadata = None

                if SAVE_DEBUG_FILES:
                    metadata = {
                        "sample_id": int(sample_id),
                        "timestamp_unix": float(time.time()),
                        "board_type": "charuco_4x4",
                        "squares_x": int(SQUARES_X),
                        "squares_y": int(SQUARES_Y),
                        "square_length_m": float(SQUARE_LENGTH_M),
                        "marker_length_m": float(MARKER_LENGTH_M),
                        "camera_resolution": str(ZED_RESOLUTION),
                        "camera_fps_requested": int(ZED_FPS),
                        "camera_info": camera_info,
                        "robot_command_degrees": {
                            name: float(theta_command_deg[i])
                            for i, name in enumerate(joint_names)
                        },
                        "robot_model_degrees_from_command": {
                            name: float(theta_command_model_deg[i])
                            for i, name in enumerate(joint_names)
                        },
                        "robot_feedback_info": robot_feedback_info,
                        "T_base_to_ee": T_base_to_ee.tolist(),
                        "T_base_to_ee_source": "motor_feedback_observation",
                        "T_camera_to_board_avg": T_camera_to_board_avg.tolist(),
                    }

                save_sample(
                    T_base_to_ee,
                    T_camera_to_board_avg,
                    metadata=metadata,
                )

                save_feedback_fk_json(T_base_to_ee, robot_feedback_info)
                save_current_camera_board(T_camera_to_board_avg)

                print("\n[SAVED] Sample", sample_id)
                print("T_base_to_ee from motor feedback:")
                print(T_base_to_ee)
                print("Robot feedback info:")
                print(json.dumps(robot_feedback_info, indent=4))
                print("T_camera_to_board_avg:")
                print(T_camera_to_board_avg)
                print("Camera window info:")
                print(json.dumps(camera_info, indent=4))
                print("Calibration lists saved to:")
                print(f"  {R_EE_BASE_FILE}")
                print(f"  {T_EE_BASE_FILE}")
                print(f"  {R_BASE_EE_FILE}")
                print(f"  {T_BASE_EE_FILE}")
                print(f"  {R_CAM_BOARD_FILE}")
                print(f"  {T_CAM_BOARD_FILE}")
                print(f"  {R_BOARD_CAM_FILE}")
                print(f"  {T_BOARD_CAM_FILE}")

                if SAVE_DEBUG_FILES:
                    print(f"FK debug JSON saved to: {FK_FILE}")
                    print(f"Metadata saved to: {SAMPLE_METADATA_FILE}")

                sample_id += 1

            elif key == 9:
                active_joint_idx = (active_joint_idx + 1) % len(joint_names)

            elif key in [curses.KEY_LEFT, curses.KEY_RIGHT]:
                joint = joint_names[active_joint_idx]
                direction = -1.0 if key == curses.KEY_LEFT else 1.0

                min_lim, max_lim = joint_limits[joint]

                current_action[joint] = clamp(
                    current_action[joint] + direction * STEP_DEG,
                    min_lim,
                    max_lim,
                )

                current_action = send_and_save_pose(robot, dict(current_action))
                time.sleep(COMMAND_DELAY)

    finally:
        zed.close()
        cv2.destroyAllWindows()


# ============================================================
# Entry point
# ============================================================

def main():
    delete_old_calibration_samples()

    robot = SO101Follower(
        SO101FollowerConfig(
            port=ROBOT_PORT,
            id=ROBOT_ID,
        )
    )

    try:
        robot.connect(calibrate=False)
        curses.wrapper(keyboard_control, robot)

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()

