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

import argparse
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
ZED_RESOLUTION = sl.RESOLUTION.HD2K
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
    t_avg = np.mean(translations, axis=0)

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
# Autonomous trajectory collection
# ============================================================

DEFAULT_TRAJECTORY_FILE = (
    ROOT
    / "data"
    / "eye_to_hand"
    / "automatic_calibration_trajectory.json"
)

DEFAULT_AUTOMATIC_OUTPUT_DIR = (
    ROOT
    / "data"
    / "eye_to_hand"
    / "automatic_calibration"
)

DEFAULT_VALID_CAMERA_FRAMES = 10
DEFAULT_CAMERA_TIMEOUT_SECONDS = 10.0

# Hold each trajectory pose for this long before beginning camera collection.
# This lets motor motion, board vibration, and camera auto-exposure settle.
DEFAULT_POINT_WAIT_SECONDS = 10.0

DEFAULT_FEEDBACK_SAMPLES = 10
DEFAULT_FEEDBACK_DELAY_SECONDS = 0.03

AUTOMATIC_SUMMARY_FILE_NAME = "automatic_collection_summary.json"
TRAJECTORY_COPY_FILE_NAME = "trajectory_used.json"
SKIPPED_POINTS_FILE_NAME = "skipped_trajectory_points.json"


def configure_output_directory(output_directory):
    """
    Point all eight calibration files and optional debug files at one directory.
    """

    global OUTPUT_DIR
    global R_EE_BASE_FILE
    global T_EE_BASE_FILE
    global R_BASE_EE_FILE
    global T_BASE_EE_FILE
    global R_CAM_BOARD_FILE
    global T_CAM_BOARD_FILE
    global R_BOARD_CAM_FILE
    global T_BOARD_CAM_FILE
    global POSE_FILE
    global FK_FILE
    global CURRENT_CAMERA_BOARD_FILE
    global SAMPLE_METADATA_FILE

    OUTPUT_DIR = Path(output_directory)

    R_EE_BASE_FILE = OUTPUT_DIR / "R_ee_base.json"
    T_EE_BASE_FILE = OUTPUT_DIR / "t_ee_base.json"
    R_BASE_EE_FILE = OUTPUT_DIR / "R_base_ee.json"
    T_BASE_EE_FILE = OUTPUT_DIR / "t_base_ee.json"
    R_CAM_BOARD_FILE = OUTPUT_DIR / "R_cam_board.json"
    T_CAM_BOARD_FILE = OUTPUT_DIR / "t_cam_board.json"
    R_BOARD_CAM_FILE = OUTPUT_DIR / "R_board_cam.json"
    T_BOARD_CAM_FILE = OUTPUT_DIR / "t_board_cam.json"

    POSE_FILE = OUTPUT_DIR / "current_robot_pose.json"
    FK_FILE = OUTPUT_DIR / "current_robot_fk.json"
    CURRENT_CAMERA_BOARD_FILE = OUTPUT_DIR / "current_camera_board.json"
    SAMPLE_METADATA_FILE = OUTPUT_DIR / "calibration_sample_metadata.json"


def atomic_write_json(file_path, value):
    """
    Write JSON through a temporary file and replace the destination.
    """

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = file_path.with_suffix(file_path.suffix + ".tmp")

    with open(temporary_path, "w") as file:
        json.dump(value, file, indent=4)

    temporary_path.replace(file_path)


def load_sample_metadata():
    """
    Load metadata for successful calibration samples.

    The transform lists contain only successful synchronized samples. Metadata
    maps each successful list entry back to its original trajectory index.
    """

    if not SAMPLE_METADATA_FILE.exists():
        return []

    with open(SAMPLE_METADATA_FILE, "r") as file:
        metadata = json.load(file)

    if not isinstance(metadata, list):
        raise ValueError(
            f"Expected a JSON list in {SAMPLE_METADATA_FILE}."
        )

    return metadata


def load_skipped_points():
    """
    Load trajectory points that were attempted and deliberately skipped.
    """

    skipped_file = OUTPUT_DIR / SKIPPED_POINTS_FILE_NAME

    if not skipped_file.exists():
        return []

    with open(skipped_file, "r") as file:
        skipped = json.load(file)

    if not isinstance(skipped, list):
        raise ValueError(
            f"Expected a JSON list in {skipped_file}."
        )

    return skipped


def successful_trajectory_indices():
    """
    Return original trajectory indexes represented by successful samples.

    Older metadata used sample_id for the trajectory index. New metadata writes
    both trajectory_index and calibration_sample_index.
    """

    indexes = set()

    for metadata_index, entry in enumerate(load_sample_metadata()):
        if not isinstance(entry, dict):
            raise ValueError(
                "Calibration metadata entry "
                f"{metadata_index} is not a dictionary."
            )

        trajectory_index = entry.get(
            "trajectory_index",
            entry.get("sample_id"),
        )

        if trajectory_index is None:
            raise ValueError(
                "Calibration metadata entry "
                f"{metadata_index} has no trajectory index."
            )

        trajectory_index = int(trajectory_index)

        if trajectory_index in indexes:
            raise RuntimeError(
                "Duplicate successful trajectory index in metadata: "
                f"{trajectory_index}"
            )

        indexes.add(trajectory_index)

    return indexes


def skipped_trajectory_indices():
    indexes = set()

    for skipped_index, entry in enumerate(load_skipped_points()):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Skipped-point entry {skipped_index} is not a dictionary."
            )

        if "trajectory_index" not in entry:
            raise ValueError(
                f"Skipped-point entry {skipped_index} has no trajectory_index."
            )

        trajectory_index = int(entry["trajectory_index"])

        if trajectory_index in indexes:
            raise RuntimeError(
                "Duplicate skipped trajectory index: "
                f"{trajectory_index}"
            )

        indexes.add(trajectory_index)

    return indexes


def validate_processed_trajectory_indices(total_targets):
    """
    Validate success/skip bookkeeping and return both processed-index sets.
    """

    successful = successful_trajectory_indices()
    skipped = skipped_trajectory_indices()

    overlap = successful & skipped

    if overlap:
        raise RuntimeError(
            "Trajectory indexes appear in both successful and skipped logs: "
            f"{sorted(overlap)}"
        )

    out_of_range = {
        index
        for index in successful | skipped
        if index < 0 or index >= total_targets
    }

    if out_of_range:
        raise RuntimeError(
            "Processed trajectory indexes are outside the current trajectory: "
            f"{sorted(out_of_range)}"
        )

    successful_count = existing_sample_count()

    if successful_count != len(successful):
        raise RuntimeError(
            "Transform-list count and successful trajectory metadata disagree: "
            f"{successful_count} transform entries versus "
            f"{len(successful)} successful trajectory indexes."
        )

    return successful, skipped


def record_skipped_trajectory_point(
    target_record,
    stage,
    reason,
    target_action,
    camera_information=None,
):
    """
    Atomically append one skipped trajectory point.

    A skipped point is never appended to any of the eight calibration transform
    files, preserving synchronization of successful robot/camera samples.
    """

    skipped = load_skipped_points()
    trajectory_index = int(target_record["sample_index"])

    existing_indexes = {
        int(entry["trajectory_index"])
        for entry in skipped
    }

    if trajectory_index in existing_indexes:
        raise RuntimeError(
            "Attempted to record an already-skipped trajectory index: "
            f"{trajectory_index}"
        )

    entry = {
        "trajectory_index": trajectory_index,
        "timestamp_unix": float(time.time()),
        "failure_stage": str(stage),
        "reason": str(reason),
        "target_command_degrees": {
            name: float(target_action[name])
            for name in joint_names
        },
        "trajectory": {
            "interpolation_group": target_record[
                "interpolation_group"
            ],
            "edge_index_in_group": target_record[
                "edge_index_in_group"
            ],
            "segment_alpha": target_record[
                "segment_alpha"
            ],
            "is_manual_configuration": target_record[
                "is_manual_configuration"
            ],
            "manual_configuration_index": target_record[
                "manual_configuration_index"
            ],
        },
    }

    if camera_information is not None:
        entry["camera_information"] = camera_information

    skipped.append(entry)

    atomic_write_json(
        OUTPUT_DIR / SKIPPED_POINTS_FILE_NAME,
        skipped,
    )

    return entry


def load_trajectory(trajectory_file):
    """
    Load generated_configurations from the grouped trajectory JSON.
    """

    trajectory_file = Path(trajectory_file)

    with open(trajectory_file, "r") as file:
        payload = json.load(file)

    generated = payload.get("generated_configurations")

    if not isinstance(generated, list) or not generated:
        raise ValueError(
            "Trajectory JSON must contain a non-empty "
            "'generated_configurations' list."
        )

    targets = []

    for expected_index, entry in enumerate(generated):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Trajectory entry {expected_index} is not a dictionary."
            )

        joint_configuration = entry.get("joints_deg", entry)

        missing = [
            name
            for name in joint_names
            if name not in joint_configuration
        ]

        if missing:
            raise ValueError(
                f"Trajectory entry {expected_index} is missing: "
                + ", ".join(missing)
            )

        validated = {}

        for name in joint_names:
            value = float(joint_configuration[name])
            minimum, maximum = joint_limits[name]

            if not np.isfinite(value):
                raise ValueError(
                    f"Trajectory entry {expected_index}, {name} is not finite."
                )

            if value < minimum or value > maximum:
                raise ValueError(
                    f"Trajectory entry {expected_index}, {name}={value:.3f} "
                    f"is outside [{minimum:.3f}, {maximum:.3f}] deg."
                )

            validated[name] = value

        sample_index = int(entry.get("sample_index", expected_index))

        if sample_index != expected_index:
            raise ValueError(
                "Trajectory sample indexes must be consecutive from zero. "
                f"Expected {expected_index}, received {sample_index}."
            )

        targets.append(
            {
                "sample_index": sample_index,
                "joints_deg": validated,
                "interpolation_group": entry.get("interpolation_group"),
                "edge_index_in_group": entry.get("edge_index_in_group"),
                "segment_alpha": entry.get("segment_alpha"),
                "is_manual_configuration": entry.get(
                    "is_manual_configuration"
                ),
                "manual_configuration_index": entry.get(
                    "manual_configuration_index"
                ),
            }
        )

    return payload, targets


def get_expected_board_marker_ids(board):
    """
    Return the exact marker IDs belonging to the ChArUco board.
    """

    marker_ids = None

    if hasattr(board, "getIds"):
        marker_ids = board.getIds()
    elif hasattr(board, "ids"):
        marker_ids = board.ids

    if marker_ids is None:
        raise RuntimeError(
            "Could not obtain marker IDs from the ChArUco board object."
        )

    return {
        int(marker_id)
        for marker_id in np.asarray(marker_ids).reshape(-1)
    }


def detect_complete_charuco_pose(
    gray,
    board,
    dictionary,
    K,
    dist,
    expected_marker_ids,
):
    """
    Detect all board markers and all ChArUco corners.

    A result is complete only when:
        1. every marker ID belonging to the board is visible,
        2. all 9 internal ChArUco corners are visible,
        3. pose estimation succeeds.
    """

    aruco = cv2.aruco

    marker_corners = None
    marker_ids = None
    charuco_corners = None
    charuco_ids = None

    if hasattr(aruco, "CharucoDetector") and hasattr(
        board,
        "matchImagePoints",
    ):
        detector = aruco.CharucoDetector(board)
        (
            charuco_corners,
            charuco_ids,
            marker_corners,
            marker_ids,
        ) = detector.detectBoard(gray)

    else:
        if hasattr(aruco, "DetectorParameters"):
            detector_parameters = aruco.DetectorParameters()
        else:
            detector_parameters = aruco.DetectorParameters_create()

        marker_corners, marker_ids, _ = aruco.detectMarkers(
            gray,
            dictionary,
            parameters=detector_parameters,
        )

        if marker_ids is not None and len(marker_ids) > 0:
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
                dist,
            )

    detected_marker_ids = set()

    if marker_ids is not None:
        detected_marker_ids = {
            int(marker_id)
            for marker_id in np.asarray(marker_ids).reshape(-1)
        }

    detected_board_marker_ids = (
        detected_marker_ids & expected_marker_ids
    )

    all_markers_detected = (
        detected_board_marker_ids == expected_marker_ids
    )

    number_of_charuco_corners = (
        0
        if charuco_corners is None
        else int(len(charuco_corners))
    )

    all_corners_detected = (
        number_of_charuco_corners == MAX_CHARUCO_CORNERS
    )

    result = {
        "complete": False,
        "all_markers_detected": bool(all_markers_detected),
        "all_corners_detected": bool(all_corners_detected),
        "marker_count": int(len(detected_board_marker_ids)),
        "expected_marker_count": int(len(expected_marker_ids)),
        "detected_marker_ids": sorted(detected_board_marker_ids),
        "expected_marker_ids": sorted(expected_marker_ids),
        "num_charuco": int(number_of_charuco_corners),
        "expected_charuco": int(MAX_CHARUCO_CORNERS),
        "marker_corners": marker_corners,
        "marker_ids": marker_ids,
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
        "T_camera_to_board": None,
        "rvec": None,
        "tvec": None,
        "pnp_method": None,
        "mean_reproj_error_px": None,
        "max_reproj_error_px": None,
    }

    if (
        not all_markers_detected
        or not all_corners_detected
        or charuco_ids is None
    ):
        return result

    if hasattr(board, "matchImagePoints"):
        object_points, image_points = board.matchImagePoints(
            charuco_corners,
            charuco_ids,
        )

        if object_points is None or image_points is None:
            return result

        object_points = np.asarray(
            object_points,
            dtype=np.float64,
        ).reshape(-1, 3)

        image_points = np.asarray(
            image_points,
            dtype=np.float64,
        ).reshape(-1, 2)

        if len(object_points) != MAX_CHARUCO_CORNERS:
            return result

        rvec, tvec, pnp_method = solve_planar_pnp(
            object_points,
            image_points,
            K,
            dist,
        )

        if rvec is None or tvec is None:
            return result

        (
            mean_error_px,
            max_error_px,
            _,
        ) = compute_reprojection_error(
            object_points,
            image_points,
            rvec,
            tvec,
            K,
            dist,
        )

    else:
        (
            success,
            rvec,
            tvec,
        ) = aruco.estimatePoseCharucoBoard(
            charuco_corners,
            charuco_ids,
            board,
            K,
            dist,
            None,
            None,
        )

        if not success:
            return result

        pnp_method = "estimatePoseCharucoBoard"
        mean_error_px = None
        max_error_px = None

    T_camera_to_board = rvec_tvec_to_T(rvec, tvec)

    result.update(
        {
            "complete": True,
            "T_camera_to_board": T_camera_to_board,
            "rvec": rvec,
            "tvec": tvec,
            "pnp_method": pnp_method,
            "mean_reproj_error_px": mean_error_px,
            "max_reproj_error_px": max_error_px,
        }
    )

    return result


def draw_complete_detection(frame_bgr, result, K, dist):
    """
    Draw every available marker/corner and draw axes for complete detections.
    """

    aruco = cv2.aruco

    if (
        result["marker_corners"] is not None
        and result["marker_ids"] is not None
    ):
        try:
            aruco.drawDetectedMarkers(
                frame_bgr,
                result["marker_corners"],
                result["marker_ids"],
            )
        except Exception:
            pass

    if result["charuco_corners"] is not None:
        try:
            aruco.drawDetectedCornersCharuco(
                frame_bgr,
                result["charuco_corners"],
                result["charuco_ids"],
            )
        except Exception:
            pass

    if result["complete"]:
        try:
            cv2.drawFrameAxes(
                frame_bgr,
                K,
                dist,
                result["rvec"],
                result["tvec"],
                AXIS_LENGTH_M,
            )
        except Exception:
            pass


def collect_complete_camera_average(
    zed,
    runtime,
    image_zed,
    board,
    dictionary,
    K,
    dist,
    expected_marker_ids,
    required_valid_frames,
    settle_seconds,
    timeout_seconds,
    trajectory_index,
    trajectory_count,
):
    """
    Wait for N complete-board frames and average their transforms.

    Invalid frames are ignored. The N valid frames do not need to be
    consecutive. timeout_seconds <= 0 waits indefinitely.
    """

    time.sleep(settle_seconds)

    transforms = []
    valid_results = []

    frames_seen = 0
    start_time = time.time()

    while len(transforms) < required_valid_frames:
        elapsed = time.time() - start_time

        if timeout_seconds > 0.0 and elapsed >= timeout_seconds:
            return None, {
                "success": False,
                "reason": "camera timeout",
                "frames_seen": int(frames_seen),
                "valid_frames": int(len(transforms)),
                "required_valid_frames": int(required_valid_frames),
                "timeout_seconds": float(timeout_seconds),
            }

        frame_bgr, gray = grab_left_bgr_and_gray(
            zed,
            runtime,
            image_zed,
        )

        if frame_bgr is None:
            continue

        frames_seen += 1

        result = detect_complete_charuco_pose(
            gray,
            board,
            dictionary,
            K,
            dist,
            expected_marker_ids,
        )

        if result["complete"]:
            transforms.append(result["T_camera_to_board"])
            valid_results.append(result)

        draw_complete_detection(
            frame_bgr,
            result,
            K,
            dist,
        )

        banner_color = (
            (0, 200, 0)
            if result["complete"]
            else (0, 0, 220)
        )

        cv2.rectangle(
            frame_bgr,
            (0, 0),
            (frame_bgr.shape[1], 95),
            (20, 20, 20),
            thickness=-1,
        )

        cv2.putText(
            frame_bgr,
            (
                f"POINT {trajectory_index + 1}/{trajectory_count} | "
                f"VALID {len(transforms)}/{required_valid_frames}"
            ),
            (25, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            banner_color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame_bgr,
            (
                f"Markers {result['marker_count']}/"
                f"{result['expected_marker_count']} | "
                f"Corners {result['num_charuco']}/"
                f"{result['expected_charuco']}"
            ),
            (25, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(
            "Automatic ZED Eye-to-Hand Collection",
            frame_bgr,
        )

        pressed_key = cv2.waitKey(1) & 0xFF

        if pressed_key == ord("q"):
            raise KeyboardInterrupt(
                "Collection stopped from the camera window."
            )

    T_average = average_transforms(transforms)

    translations = np.array(
        [
            transform[:3, 3]
            for transform in transforms
        ],
        dtype=np.float64,
    )

    rotation_deviations_deg = np.array(
        [
            rotation_distance_degrees(
                T_average[:3, :3],
                transform[:3, :3],
            )
            for transform in transforms
        ],
        dtype=np.float64,
    )

    mean_errors = np.array(
        [
            result["mean_reproj_error_px"]
            for result in valid_results
            if result["mean_reproj_error_px"] is not None
        ],
        dtype=np.float64,
    )

    max_errors = np.array(
        [
            result["max_reproj_error_px"]
            for result in valid_results
            if result["max_reproj_error_px"] is not None
        ],
        dtype=np.float64,
    )

    information = {
        "success": True,
        "frames_seen": int(frames_seen),
        "valid_frames": int(len(transforms)),
        "required_valid_frames": int(required_valid_frames),
        "all_markers_required": True,
        "all_charuco_corners_required": True,
        "expected_marker_ids": sorted(expected_marker_ids),
        "expected_marker_count": int(len(expected_marker_ids)),
        "expected_charuco_count": int(MAX_CHARUCO_CORNERS),
        "translation_average_m": np.mean(
            translations,
            axis=0,
        ).tolist(),
        "translation_std_mm": (
            1000.0 * np.std(translations, axis=0)
        ).tolist(),
        "rotation_deviation_deg_mean": float(
            np.mean(rotation_deviations_deg)
        ),
        "rotation_deviation_deg_max": float(
            np.max(rotation_deviations_deg)
        ),
        "mean_reproj_error_px_mean": (
            float(np.mean(mean_errors))
            if len(mean_errors)
            else None
        ),
        "mean_reproj_error_px_max": (
            float(np.max(mean_errors))
            if len(mean_errors)
            else None
        ),
        "max_reproj_error_px_max": (
            float(np.max(max_errors))
            if len(max_errors)
            else None
        ),
    }

    return T_average, information


def rotation_distance_degrees(R_a, R_b):
    """
    Angular difference between two rotation matrices.
    """

    R_relative = (
        np.asarray(R_a, dtype=np.float64).T
        @ np.asarray(R_b, dtype=np.float64)
    )

    cosine = np.clip(
        (np.trace(R_relative) - 1.0) / 2.0,
        -1.0,
        1.0,
    )

    return float(np.degrees(np.arccos(cosine)))


def get_robot_feedback_window(
    robot,
    number_of_samples,
    sample_delay_seconds,
):
    """
    Read a stationary window of measured motor degrees.
    """

    measurements = []

    for sample_index in range(number_of_samples):
        measurements.append(
            get_robot_feedback_angles_deg(
                robot,
                fallback_action=None,
            )
        )

        if sample_index + 1 < number_of_samples:
            time.sleep(sample_delay_seconds)

    measurements = np.asarray(
        measurements,
        dtype=np.float64,
    )

    return {
        "mean_deg": np.mean(measurements, axis=0),
        "median_deg": np.median(measurements, axis=0),
        "std_deg": np.std(measurements, axis=0),
        "minimum_deg": np.min(measurements, axis=0),
        "maximum_deg": np.max(measurements, axis=0),
    }


def feedback_window_to_T_base_to_ee(
    robot,
    number_of_samples,
    sample_delay_seconds,
):
    """
    Compute ^B T_E from the mean measured motor degrees in a feedback window.

    Commanded joint values are never used for this FK result.
    """

    feedback_window = get_robot_feedback_window(
        robot,
        number_of_samples,
        sample_delay_seconds,
    )

    theta_feedback_deg = feedback_window["mean_deg"]
    theta_model_deg = robot_deg_to_model_deg(theta_feedback_deg)
    theta_model_rad = np.radians(theta_model_deg)

    T_base_to_ee = space_product_of_exponentials(
        M,
        S_list,
        theta_model_rad,
    )

    information = {
        "source": (
            "mean measured degrees from robot.get_observation()"
        ),
        "number_of_feedback_samples": int(number_of_samples),
        "feedback_sample_delay_seconds": float(sample_delay_seconds),
        "robot_feedback_degrees_mean": {
            name: float(theta_feedback_deg[index])
            for index, name in enumerate(joint_names)
        },
        "robot_feedback_degrees_median": {
            name: float(feedback_window["median_deg"][index])
            for index, name in enumerate(joint_names)
        },
        "robot_feedback_degrees_std": {
            name: float(feedback_window["std_deg"][index])
            for index, name in enumerate(joint_names)
        },
        "model_fk_degrees_from_feedback_mean": {
            name: float(theta_model_deg[index])
            for index, name in enumerate(joint_names)
        },
    }

    return T_base_to_ee, information


def initialize_current_action_from_feedback(robot):
    """
    Initialize the motion state from actual motor feedback.
    """

    global current_action

    measured = get_robot_feedback_angles_deg(
        robot,
        fallback_action=None,
    )

    current_action = {
        name: float(measured[index])
        for index, name in enumerate(joint_names)
    }


def calibration_files():
    return [
        R_EE_BASE_FILE,
        T_EE_BASE_FILE,
        R_BASE_EE_FILE,
        T_BASE_EE_FILE,
        R_CAM_BOARD_FILE,
        T_CAM_BOARD_FILE,
        R_BOARD_CAM_FILE,
        T_BOARD_CAM_FILE,
    ]


def existing_sample_count():
    """
    Verify that all eight transform lists and metadata have the same length.

    This prevents resume from silently pairing the wrong robot and camera
    samples after an interrupted write.
    """

    lengths = {}

    for file_path in calibration_files():
        if not file_path.exists():
            lengths[str(file_path)] = 0
            continue

        with open(file_path, "r") as file:
            value = json.load(file)

        if not isinstance(value, list):
            raise ValueError(
                f"Expected a JSON list in {file_path}."
            )

        lengths[str(file_path)] = len(value)

    if SAMPLE_METADATA_FILE.exists():
        with open(SAMPLE_METADATA_FILE, "r") as file:
            metadata = json.load(file)

        if not isinstance(metadata, list):
            raise ValueError(
                f"Expected a JSON list in {SAMPLE_METADATA_FILE}."
            )

        lengths[str(SAMPLE_METADATA_FILE)] = len(metadata)

    elif any(length > 0 for length in lengths.values()):
        lengths[str(SAMPLE_METADATA_FILE)] = 0

    unique_lengths = set(lengths.values())

    if len(unique_lengths) != 1:
        raise RuntimeError(
            "Calibration files are not aligned:\n"
            + json.dumps(lengths, indent=2)
        )

    return next(iter(unique_lengths))


def prepare_automatic_output(resume, overwrite):
    """
    Prepare fresh or resumed automatic output.
    """

    if resume and overwrite:
        raise ValueError(
            "--resume and --overwrite cannot be used together."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files_exist = (
        any(
            file_path.exists()
            for file_path in calibration_files()
        )
        or SAMPLE_METADATA_FILE.exists()
        or (OUTPUT_DIR / SKIPPED_POINTS_FILE_NAME).exists()
    )

    if resume:
        if not files_exist:
            raise RuntimeError(
                "--resume was requested, but no calibration files exist."
            )

        return existing_sample_count()

    if files_exist and not overwrite:
        raise RuntimeError(
            "Calibration files already exist. Use --resume or --overwrite."
        )

    if overwrite:
        delete_old_calibration_samples()

        for file_name in [
            AUTOMATIC_SUMMARY_FILE_NAME,
            TRAJECTORY_COPY_FILE_NAME,
            SKIPPED_POINTS_FILE_NAME,
        ]:
            file_path = OUTPUT_DIR / file_name

            if file_path.exists():
                file_path.unlink()

    return 0


def save_automatic_summary(
    status,
    trajectory_file,
    total_targets,
    successful_samples,
    skipped_samples,
    error=None,
):
    processed_points = (
        int(successful_samples)
        + int(skipped_samples)
    )

    summary = {
        "status": status,
        "trajectory_file": str(trajectory_file),
        "total_targets": int(total_targets),
        "successful_samples": int(successful_samples),
        "skipped_samples": int(skipped_samples),
        "processed_points": int(processed_points),
        "remaining_points": int(
            max(0, total_targets - processed_points)
        ),
        "timestamp_unix": float(time.time()),
    }

    if error is not None:
        summary["error"] = str(error)

    atomic_write_json(
        OUTPUT_DIR / AUTOMATIC_SUMMARY_FILE_NAME,
        summary,
    )


def wait_before_point_collection(wait_seconds, point_index, total_points):
    """
    Hold the robot stationary before collecting data at a trajectory point.

    The delay is separate from camera-frame collection. It gives the arm,
    mounted board, and camera exposure time to settle after motion.
    """

    wait_seconds = float(wait_seconds)

    if wait_seconds <= 0.0:
        return

    print(
        f"[WAIT] Point {point_index + 1}/{total_points}: "
        f"holding position for {wait_seconds:.2f} seconds "
        "before camera collection..."
    )

    finish_time = time.time() + wait_seconds
    last_reported_second = None

    while True:
        remaining = finish_time - time.time()

        if remaining <= 0.0:
            break

        # Print a compact countdown only when the displayed whole second changes.
        displayed_second = int(np.ceil(remaining))

        if displayed_second != last_reported_second:
            print(f"       collecting begins in {displayed_second} s")
            last_reported_second = displayed_second

        time.sleep(min(0.10, remaining))

    print("[WAIT] Settling delay complete. Starting camera collection.")


def collect_trajectory_automatically(args):
    """
    Execute every unprocessed trajectory point.

    Point-level failures are logged and skipped. They do not terminate the
    trajectory. KeyboardInterrupt remains an immediate operator stop.

    File-write failures remain fatal because continuing after a partial write
    could misalign the eight calibration transform lists.
    """

    (
        trajectory_payload,
        targets,
    ) = load_trajectory(args.trajectory)

    prepare_automatic_output(
        resume=args.resume,
        overwrite=args.overwrite,
    )

    successful_indexes, skipped_indexes = (
        validate_processed_trajectory_indices(
            len(targets)
        )
    )

    processed_indexes = (
        successful_indexes
        | skipped_indexes
    )

    pending_indexes = [
        index
        for index in range(len(targets))
        if index not in processed_indexes
    ]

    atomic_write_json(
        OUTPUT_DIR / TRAJECTORY_COPY_FILE_NAME,
        trajectory_payload,
    )

    if not pending_indexes:
        print(
            "[INFO] Every trajectory point has already been "
            "successfully collected or skipped."
        )
        print(
            f"[INFO] Successful samples: {len(successful_indexes)}"
        )
        print(
            f"[INFO] Skipped points:     {len(skipped_indexes)}"
        )
        return

    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.port,
            id=args.robot_id,
        )
    )

    zed = None
    robot_connected = False

    print("\n" + "=" * 72)
    print("AUTONOMOUS EYE-TO-HAND TRAJECTORY COLLECTION")
    print("=" * 72)
    print(f"Trajectory:          {args.trajectory}")
    print(f"Output directory:    {OUTPUT_DIR}")
    print(f"Total points:        {len(targets)}")
    print(f"Already successful:  {len(successful_indexes)}")
    print(f"Already skipped:     {len(skipped_indexes)}")
    print(f"Pending points:      {len(pending_indexes)}")
    print(f"Valid frames:        {args.valid_frames}")
    print(f"Point wait:          {args.point_wait:.2f} seconds")
    print(
        "Camera rule:        every board marker + all 9 ChArUco corners"
    )
    print(
        "FK source:          averaged measured motor feedback"
    )
    print(
        "Failure behavior:   log failed point, skip it, continue"
    )
    print(
        "Press q in the camera window or Ctrl+C to preserve progress."
    )

    confirmation = input(
        "\nType RUN to begin autonomous motion: "
    ).strip()

    if confirmation != "RUN":
        raise RuntimeError(
            "Collection cancelled before robot motion."
        )

    try:
        robot.connect(calibrate=False)
        robot_connected = True

        initialize_current_action_from_feedback(robot)

        zed = open_zed_camera()
        runtime = sl.RuntimeParameters()
        image_zed = sl.Mat()

        K, dist = get_zed_left_intrinsics_rectified(zed)
        board, dictionary = create_charuco_board()
        expected_marker_ids = get_expected_board_marker_ids(board)

        print(
            "[INFO] Required board marker IDs: "
            f"{sorted(expected_marker_ids)}"
        )
        print("[INFO] ZED LEFT camera matrix:")
        print(
            np.array2string(
                K,
                precision=6,
                suppress_small=True,
            )
        )

        cv2.namedWindow(
            "Automatic ZED Eye-to-Hand Collection",
            cv2.WINDOW_NORMAL,
        )

        for target_index in pending_indexes:
            target_record = targets[target_index]
            target_action = target_record["joints_deg"]

            print("\n" + "-" * 72)
            print(
                f"POINT {target_index + 1}/{len(targets)}"
            )
            print("-" * 72)

            for name in joint_names:
                print(
                    f"  {name:20s}: "
                    f"{target_action[name]:9.3f} deg"
                )

            stage = "move_to_target"
            camera_information = None

            try:
                move_smooth(
                    robot,
                    target_action,
                )

                stage = "point_wait"

                wait_before_point_collection(
                    wait_seconds=args.point_wait,
                    point_index=target_index,
                    total_points=len(targets),
                )

                stage = "camera_collection"

                (
                    T_camera_to_board_average,
                    camera_information,
                ) = collect_complete_camera_average(
                    zed=zed,
                    runtime=runtime,
                    image_zed=image_zed,
                    board=board,
                    dictionary=dictionary,
                    K=K,
                    dist=dist,
                    expected_marker_ids=expected_marker_ids,
                    required_valid_frames=args.valid_frames,
                    settle_seconds=args.camera_settle,
                    timeout_seconds=args.camera_timeout,
                    trajectory_index=target_index,
                    trajectory_count=len(targets),
                )

                if T_camera_to_board_average is None:
                    raise RuntimeError(
                        "Could not collect the requested number of "
                        "complete-board camera frames: "
                        + json.dumps(
                            camera_information,
                            indent=2,
                        )
                    )

                stage = "robot_feedback_and_fk"

                (
                    T_base_to_ee,
                    robot_feedback_information,
                ) = feedback_window_to_T_base_to_ee(
                    robot=robot,
                    number_of_samples=args.feedback_samples,
                    sample_delay_seconds=args.feedback_delay,
                )

                target_array = action_to_theta_robot_deg(
                    target_action
                )

                measured_mean = np.array(
                    [
                        robot_feedback_information[
                            "robot_feedback_degrees_mean"
                        ][name]
                        for name in joint_names
                    ],
                    dtype=np.float64,
                )

                calibration_sample_index = (
                    existing_sample_count()
                )

                metadata = {
                    # Backward-compatible field: in prior files sample_id was
                    # the original trajectory index.
                    "sample_id": int(target_index),
                    "trajectory_index": int(target_index),
                    "calibration_sample_index": int(
                        calibration_sample_index
                    ),
                    "timestamp_unix": float(time.time()),
                    "trajectory": {
                        "interpolation_group": target_record[
                            "interpolation_group"
                        ],
                        "edge_index_in_group": target_record[
                            "edge_index_in_group"
                        ],
                        "segment_alpha": target_record[
                            "segment_alpha"
                        ],
                        "is_manual_configuration": target_record[
                            "is_manual_configuration"
                        ],
                        "manual_configuration_index": target_record[
                            "manual_configuration_index"
                        ],
                    },
                    "target_command_degrees": {
                        name: float(target_action[name])
                        for name in joint_names
                    },
                    "robot_feedback_information": (
                        robot_feedback_information
                    ),
                    "feedback_mean_minus_command_degrees": {
                        name: float(
                            measured_mean[index]
                            - target_array[index]
                        )
                        for index, name in enumerate(
                            joint_names
                        )
                    },
                    "T_base_to_ee": (
                        T_base_to_ee.tolist()
                    ),
                    "T_base_to_ee_source": (
                        "mean measured motor feedback"
                    ),
                    "T_camera_to_board_average": (
                        T_camera_to_board_average.tolist()
                    ),
                    "camera_information": (
                        camera_information
                    ),
                }

            except KeyboardInterrupt:
                raise

            except Exception as exception:
                skipped_entry = (
                    record_skipped_trajectory_point(
                        target_record=target_record,
                        stage=stage,
                        reason=exception,
                        target_action=target_action,
                        camera_information=(
                            camera_information
                        ),
                    )
                )

                skipped_indexes.add(
                    target_index
                )

                print(
                    f"\n[SKIPPED] Trajectory point "
                    f"{target_index}"
                )
                print(
                    f"  Failure stage: {stage}"
                )
                print(
                    f"  Reason:        {exception}"
                )
                print(
                    "  Continuing to the next trajectory point."
                )

                save_automatic_summary(
                    status="running_with_skips",
                    trajectory_file=args.trajectory,
                    total_targets=len(targets),
                    successful_samples=len(
                        successful_indexes
                    ),
                    skipped_samples=len(
                        skipped_indexes
                    ),
                )

                continue

            # Saving is deliberately outside the point-failure handler.
            # A file-write failure may partially update synchronized lists;
            # continuing would risk corrupting calibration alignment.
            stage = "save_synchronized_sample"

            save_sample(
                T_base_to_ee,
                T_camera_to_board_average,
                metadata=metadata,
            )

            save_feedback_fk_json(
                T_base_to_ee,
                robot_feedback_information,
            )

            save_current_camera_board(
                T_camera_to_board_average
            )

            successful_indexes.add(
                target_index
            )

            successful_count = (
                existing_sample_count()
            )

            save_automatic_summary(
                status=(
                    "running_with_skips"
                    if skipped_indexes
                    else "running"
                ),
                trajectory_file=args.trajectory,
                total_targets=len(targets),
                successful_samples=successful_count,
                skipped_samples=len(skipped_indexes),
            )

            maximum_tracking_error = float(
                np.max(
                    np.abs(
                        measured_mean
                        - target_array
                    )
                )
            )

            print(
                f"[SAVED] Trajectory point "
                f"{target_index} as calibration sample "
                f"{successful_count - 1}"
            )
            print(
                "  Camera valid frames: "
                f"{camera_information['valid_frames']}"
            )
            print(
                "  Camera frames seen:  "
                f"{camera_information['frames_seen']}"
            )
            print(
                "  Max tracking error:  "
                f"{maximum_tracking_error:.3f} deg"
            )
            print(
                "  Successful samples:  "
                f"{successful_count}"
            )
            print(
                "  Skipped points:       "
                f"{len(skipped_indexes)}"
            )

        print(
            "\n[INFO] Every pending trajectory point "
            "has been processed."
        )
        print(
            "[INFO] Moving robot smoothly to the REST position..."
        )

        rest_error = None

        try:
            move_smooth(
                robot,
                rest,
            )
            print(
                "[INFO] Robot reached the REST position."
            )
        except Exception as exception:
            rest_error = exception
            print(
                "[WARN] Collection processing is complete, but "
                "the robot could not return to REST: "
                f"{exception}"
            )

        successful_count = existing_sample_count()
        skipped_count = len(
            load_skipped_points()
        )

        if rest_error is not None:
            completion_status = (
                "complete_rest_failed"
            )
        elif skipped_count > 0:
            completion_status = (
                "complete_with_skips"
            )
        else:
            completion_status = "complete"

        save_automatic_summary(
            status=completion_status,
            trajectory_file=args.trajectory,
            total_targets=len(targets),
            successful_samples=successful_count,
            skipped_samples=skipped_count,
            error=rest_error,
        )

        print("\n" + "=" * 72)
        print("AUTONOMOUS COLLECTION FINISHED")
        print("=" * 72)
        print(
            f"Successful samples: {successful_count}"
        )
        print(
            f"Skipped points:     {skipped_count}"
        )
        print(
            f"Total processed:    "
            f"{successful_count + skipped_count}/"
            f"{len(targets)}"
        )
        print(
            f"Output directory:   {OUTPUT_DIR}"
        )
        print(
            "Skipped-point log:  "
            f"{OUTPUT_DIR / SKIPPED_POINTS_FILE_NAME}"
        )
        print(
            "Final robot state:  "
            + (
                "REST"
                if rest_error is None
                else (
                    "REST MOVE FAILED - inspect robot "
                    "before continuing"
                )
            )
        )

    except KeyboardInterrupt:
        successful_count = existing_sample_count()
        skipped_count = len(
            load_skipped_points()
        )

        save_automatic_summary(
            status="stopped_by_user",
            trajectory_file=args.trajectory,
            total_targets=len(targets),
            successful_samples=successful_count,
            skipped_samples=skipped_count,
            error="KeyboardInterrupt",
        )

        print(
            "\n[STOPPED] Successful samples and skipped-point "
            "records were preserved. Run again with --resume."
        )

    except Exception as exception:
        successful_count = existing_sample_count()
        skipped_count = len(
            load_skipped_points()
        )

        save_automatic_summary(
            status="failed",
            trajectory_file=args.trajectory,
            total_targets=len(targets),
            successful_samples=successful_count,
            skipped_samples=skipped_count,
            error=exception,
        )

        raise

    finally:
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


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Collect SO-101 eye-to-hand calibration data autonomously "
            "from a generated trajectory."
        )
    )

    parser.add_argument(
        "--trajectory",
        type=Path,
        default=DEFAULT_TRAJECTORY_FILE,
        help=(
            "Trajectory JSON produced by "
            "create_automatic_calibration_trajectory.py."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_AUTOMATIC_OUTPUT_DIR,
        help=(
            "Output directory for the eight plain-list transform files."
        ),
    )

    parser.add_argument(
        "--valid-frames",
        type=int,
        default=DEFAULT_VALID_CAMERA_FRAMES,
        help=(
            "Complete-board camera frames averaged per point. Default: 5."
        ),
    )

    parser.add_argument(
        "--camera-settle",
        type=float,
        default=0.0,
        help=(
            "Optional additional camera-only delay after --point-wait. "
            "Default: 0.0."
        ),
    )

    parser.add_argument(
        "--camera-timeout",
        type=float,
        default=DEFAULT_CAMERA_TIMEOUT_SECONDS,
        help=(
            "Maximum seconds to obtain all valid frames. "
            "Use 0 to wait indefinitely. Default: 30."
        ),
    )

    parser.add_argument(
        "--point-wait",
        "--robot-settle",
        dest="point_wait",
        type=float,
        default=DEFAULT_POINT_WAIT_SECONDS,
        help=(
            "Seconds to hold each trajectory point before collecting camera "
            "data. --robot-settle is retained as an alias. Default: 5.0."
        ),
    )

    parser.add_argument(
        "--feedback-samples",
        type=int,
        default=DEFAULT_FEEDBACK_SAMPLES,
        help=(
            "Measured motor-feedback readings averaged for FK. Default: 10."
        ),
    )

    parser.add_argument(
        "--feedback-delay",
        type=float,
        default=DEFAULT_FEEDBACK_DELAY_SECONDS,
        help=(
            "Seconds between measured feedback readings. Default: 0.03."
        ),
    )

    parser.add_argument(
        "--port",
        default=ROBOT_PORT,
        help=f"SO-101 port. Default: {ROBOT_PORT}",
    )

    parser.add_argument(
        "--robot-id",
        default=ROBOT_ID,
        help=f"LeRobot robot ID. Default: {ROBOT_ID}",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue from the number of aligned samples already saved."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete old automatic calibration files and start at point zero."
        ),
    )

    args = parser.parse_args()

    if args.valid_frames < 1:
        raise ValueError(
            "--valid-frames must be at least 1."
        )

    if args.feedback_samples < 1:
        raise ValueError(
            "--feedback-samples must be at least 1."
        )

    if args.camera_settle < 0.0:
        raise ValueError(
            "--camera-settle cannot be negative."
        )

    if args.point_wait < 0.0:
        raise ValueError(
            "--point-wait cannot be negative."
        )

    if args.feedback_delay < 0.0:
        raise ValueError(
            "--feedback-delay cannot be negative."
        )

    configure_output_directory(args.output_dir)
    collect_trajectory_automatically(args)


if __name__ == "__main__":
    main()

