#!/usr/bin/env python3

"""
Direct eye-to-hand consistency test for the SO-ARM101 + fixed ZED camera.

This script follows the same keyboard-control and ChArUco collection pipeline as
collect_eye_to_hand_data_charuco4x4.py, but it does not run an OpenCV hand-eye
solver. Instead, it uses the measured board mounting transform directly:

    T_base_cam = T_base_ee @ T_ee_board @ inverse(T_cam_board)

Frame convention:
    ^A T_B maps coordinates from frame B into frame A.

Transforms:
    T_base_ee  = ^B T_E
        Computed from measured SO-ARM101 motor feedback using forward kinematics.

    T_ee_board = ^E T_W
        Known rigid mounting transform from the ChArUco board frame W into the
        end-effector frame E.

    T_cam_board = ^C T_W
        Estimated by OpenCV solvePnP from the rectified ZED LEFT image.

    T_base_cam = ^B T_C
        Calculated independently for every saved robot pose. Since the camera
        and robot base are fixed, this transform should remain approximately
        constant across all samples.

Output:
    A single human-readable text file is written:
        data/eye_to_hand/base_camera_consistency_samples.txt

For every sample, the text file stores:
    - measured robot joint feedback
    - T_base_ee
    - averaged T_cam_board
    - calculated T_base_cam
    - translation and rotation difference from sample 0
    - simple ChArUco quality information

Camera convention for the rectified ZED LEFT image:
    +X = right in image
    +Y = down in image
    +Z = forward away from camera
"""

import sys
import time
import curses
from datetime import datetime
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

# This is the only file written by this script.
SAMPLE_FILE = OUTPUT_DIR / "base_camera_consistency_samples.txt"

# True:
#     erase the previous text file whenever the script starts.
#
# False:
#     append new samples to the existing text file.
RESET_SAMPLE_FILE_ON_START = True

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

# Use exactly one OpenCV window throughout the live view and save window.
# Reusing one name prevents OpenCV from creating a second camera window when
# the user presses `s`.
WINDOW_NAME = "ZED Base-Camera Consistency Test"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720


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
# Known board mounting transform: ^E T_W
# ============================================================

# Assumption supplied for this test:
#     - the end-effector frame and ChArUco board frame have the same orientation
#     - the ChArUco board-frame origin is 10.5 mm along the EE +X axis
#
# Therefore:
#     ^E R_W = I
#     ^E t_W = [0.0105, 0, 0] meters
#
# This transform must remain constant for every robot pose.
T_EE_BOARD = np.array(
    [
        [1.0, 0.0, 0.0, 0.0105],
        [0.0, 1.0, 0.0, 0.0000],
        [0.0, 0.0, 1.0, 0.0000],
        [0.0, 0.0, 0.0, 1.0000],
    ],
    dtype=np.float64,
)


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
# Output-file initialization
# ============================================================

def format_matrix(T):
    """
    Format a matrix for the human-readable text file.

    suppress_small=True prevents tiny floating-point values such as 1e-16
    from making the file difficult to read.
    """

    return np.array2string(
        np.asarray(T, dtype=np.float64),
        precision=7,
        suppress_small=True,
        floatmode="fixed",
    )


def initialize_sample_file():
    """
    Create the single text output file and write a short experiment header.

    When RESET_SAMPLE_FILE_ON_START is True, any previous consistency test is
    removed. No JSON, NPY, or additional debugging files are created.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mode = "w" if RESET_SAMPLE_FILE_ON_START else "a"

    with open(SAMPLE_FILE, mode, encoding="utf-8") as f:
        if mode == "a" and SAMPLE_FILE.stat().st_size > 0:
            f.write("\n\n")

        f.write("=" * 78 + "\n")
        f.write("SO-ARM101 FIXED-CAMERA TRANSFORM CONSISTENCY TEST\n")
        f.write("=" * 78 + "\n")
        f.write(f"Started: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write("\nFrame convention:\n")
        f.write("  ^A T_B maps coordinates from frame B into frame A.\n")
        f.write("\nEquation used for every sample:\n")
        f.write("  T_base_cam = T_base_ee @ T_ee_board @ inverse(T_cam_board)\n")
        f.write("\nKnown T_ee_board = ^E T_W:\n")
        f.write(format_matrix(T_EE_BOARD) + "\n")
        f.write("\nBoard settings:\n")
        f.write(f"  squares:       {SQUARES_X} x {SQUARES_Y}\n")
        f.write(f"  square length: {SQUARE_LENGTH_M:.7f} m\n")
        f.write(f"  marker length: {MARKER_LENGTH_M:.7f} m\n")
        f.write("\nThe camera is fixed, so T_base_cam should remain constant.\n")
        f.write("=" * 78 + "\n")

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


def print_robot_transform(action, robot=None):
    """
    Print the current FK to the terminal.

    This function does not write a separate file. The only persistent output
    produced by this script is SAMPLE_FILE.
    """

    theta_command_deg = action_to_theta_robot_deg(action)
    theta_command_model_deg = robot_deg_to_model_deg(theta_command_deg)
    T_command = action_to_T_base_to_ee(action)

    print("\n" + "=" * 70)
    print("CURRENT ROBOT FK")
    print("=" * 70)

    print("\nCommanded robot angles [deg]:")
    for name, value in zip(joint_names, theta_command_deg):
        print(f"  {name:20s}: {value: .3f}")

    print("\nCommanded offset-corrected FK/model angles [deg]:")
    for name, value in zip(joint_names, theta_command_model_deg):
        print(f"  {name:20s}: {value: .3f}")

    print("\nT_base_ee from commanded action:")
    print(format_matrix(T_command))

    if robot is not None:
        try:
            T_feedback, feedback_info = feedback_to_T_base_to_ee(
                robot,
                fallback_action=action,
            )

            feedback_deg = np.array(
                [
                    feedback_info["robot_feedback_degrees"][name]
                    for name in joint_names
                ],
                dtype=float,
            )

            model_feedback_deg = np.array(
                [
                    feedback_info["model_fk_degrees_from_feedback"][name]
                    for name in joint_names
                ],
                dtype=float,
            )

            print("\nMeasured motor feedback angles [deg]:")
            for name, value in zip(joint_names, feedback_deg):
                print(f"  {name:20s}: {value: .3f}")

            print("\nFeedback offset-corrected FK/model angles [deg]:")
            for name, value in zip(joint_names, model_feedback_deg):
                print(f"  {name:20s}: {value: .3f}")

            print("\nT_base_ee from measured motor feedback:")
            print(format_matrix(T_feedback))

        except Exception as e:
            print(f"\n[WARN] Could not read feedback FK: {e}")

    print("=" * 70 + "\n")


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

def send_robot_pose(robot, action):
    """
    Clamp and send one joint action.

    Robot motion is not written to a separate pose file. A measured FK pose is
    written only when the user explicitly saves a consistency-test sample.
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
    return final_action


def move_smooth(robot, target_action):
    """
    Smoothly move the robot from current_action to target_action.
    """

    global current_action

    final_action = dict(current_action)

    for name in joint_names:
        if name in target_action:
            min_lim, max_lim = joint_limits[name]
            final_action[name] = clamp(
                float(target_action[name]),
                min_lim,
                max_lim,
            )

    current = np.array(
        [current_action[name] for name in joint_names],
        dtype=float,
    )
    target = np.array(
        [final_action[name] for name in joint_names],
        dtype=float,
    )

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

        current_action = send_robot_pose(robot, action)
        time.sleep(COMMAND_DELAY)

    current_action = dict(final_action)

# ============================================================
# OpenCV window helpers
# ============================================================

def open_display_window():
    """
    Create the single OpenCV display window used by the entire script.

    Calling namedWindow once and always using WINDOW_NAME prevents the save
    routine from creating an additional window.
    """

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)


def close_display_window():
    """
    Close all OpenCV windows and flush the Linux HighGUI event queue.

    On some Linux desktop environments, destroyAllWindows() does not visually
    remove a window until waitKey() processes one final GUI event.
    """

    try:
        cv2.destroyWindow(WINDOW_NAME)
    except cv2.error:
        pass

    cv2.destroyAllWindows()

    # Pump the HighGUI event queue so the window disappears immediately.
    for _ in range(3):
        try:
            cv2.waitKey(1)
        except cv2.error:
            break
        time.sleep(0.01)


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

        # Reuse the same window as the normal live view.
        cv2.imshow(WINDOW_NAME, frame_bgr)

        # Allow `q` to stop the script even when the OpenCV window has focus.
        cv_key = cv2.waitKey(1) & 0xFF
        if cv_key == ord("q"):
            info = {
                "success": False,
                "quit_requested": True,
                "reason": "user pressed q in OpenCV window",
                "frames_seen": int(frames_seen),
                "valid_frames": int(len(T_list)),
            }
            return None, info

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
# Direct T_base_cam calculation and text-file sample saving
# ============================================================

def rotation_difference_deg(R_reference, R_current):
    """
    Return the relative rotation angle between two 3x3 rotation matrices.

    The result is the geodesic SO(3) angle in degrees.
    """

    R_reference = np.asarray(
        R_reference,
        dtype=np.float64,
    ).reshape(3, 3)

    R_current = np.asarray(
        R_current,
        dtype=np.float64,
    ).reshape(3, 3)

    R_delta = R_reference.T @ R_current

    cos_angle = np.clip(
        (np.trace(R_delta) - 1.0) / 2.0,
        -1.0,
        1.0,
    )

    return float(np.degrees(np.arccos(cos_angle)))


def calculate_T_base_cam(T_base_ee, T_cam_board):
    """
    Calculate the fixed-camera transform for one synchronized sample.

    Known transforms:
        T_base_ee  = ^B T_E
        T_ee_board = ^E T_W
        T_cam_board = ^C T_W

    We need:
        T_base_cam = ^B T_C

    Since:
        inverse(T_cam_board) = ^W T_C

    the valid frame chain is:
        ^B T_E @ ^E T_W @ ^W T_C = ^B T_C
    """

    T_base_ee = np.asarray(
        T_base_ee,
        dtype=np.float64,
    ).reshape(4, 4)

    T_cam_board = np.asarray(
        T_cam_board,
        dtype=np.float64,
    ).reshape(4, 4)

    return (
        T_base_ee
        @ T_EE_BOARD
        @ invert_T(T_cam_board)
    )


def append_sample_to_text_file(
    sample_id,
    T_base_ee,
    T_cam_board,
    T_base_cam,
    robot_feedback_info,
    camera_info,
    T_reference=None,
):
    """
    Append one easy-to-read sample block to SAMPLE_FILE.

    This is the only persistent write performed while collecting data.

    Parameters
    ----------
    sample_id : int
        Sequential sample number.

    T_base_ee : np.ndarray, shape (4, 4)
        Robot FK calculated from measured motor feedback.

    T_cam_board : np.ndarray, shape (4, 4)
        Averaged ChArUco pose returned by solvePnP.

    T_base_cam : np.ndarray, shape (4, 4)
        Directly calculated fixed-camera transform.

    robot_feedback_info : dict
        Joint feedback information returned by feedback_to_T_base_to_ee().

    camera_info : dict
        Camera-window quality information.

    T_reference : np.ndarray or None
        T_base_cam from sample 0. When supplied, the file includes the
        translation and rotation difference from that first sample.

    Returns
    -------
    translation_difference_mm : float
        Difference between this sample and sample 0 translations.

    rotation_difference_deg_value : float
        Relative rotation angle between this sample and sample 0.
    """

    if T_reference is None:
        translation_difference_mm = 0.0
        rotation_difference_deg_value = 0.0
    else:
        T_reference = np.asarray(
            T_reference,
            dtype=np.float64,
        ).reshape(4, 4)

        translation_difference_mm = float(
            1000.0
            * np.linalg.norm(
                T_base_cam[:3, 3]
                - T_reference[:3, 3]
            )
        )

        rotation_difference_deg_value = rotation_difference_deg(
            T_reference[:3, :3],
            T_base_cam[:3, :3],
        )

    feedback_degrees = robot_feedback_info[
        "robot_feedback_degrees"
    ]

    mean_reproj = camera_info.get(
        "mean_reproj_error_px_mean",
        None,
    )
    max_reproj = camera_info.get(
        "max_reproj_error_px_max",
        None,
    )

    with open(SAMPLE_FILE, "a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write("=" * 78 + "\n")
        f.write(f"SAMPLE {sample_id:03d}\n")
        f.write("=" * 78 + "\n")
        f.write(
            f"Timestamp: "
            f"{datetime.now().isoformat(timespec='milliseconds')}\n"
        )

        f.write("\nRobot feedback angles [deg]:\n")
        for name in joint_names:
            f.write(
                f"  {name:20s}: "
                f"{feedback_degrees[name]: .6f}\n"
            )

        f.write("\nCamera-window quality:\n")
        f.write(
            f"  valid frames: "
            f"{camera_info.get('valid_frames', 'N/A')}\n"
        )
        f.write(
            f"  frames seen:  "
            f"{camera_info.get('frames_seen', 'N/A')}\n"
        )

        if mean_reproj is not None:
            f.write(
                f"  mean reprojection error: "
                f"{mean_reproj:.6f} px\n"
            )
        else:
            f.write(
                "  mean reprojection error: N/A\n"
            )

        if max_reproj is not None:
            f.write(
                f"  maximum reprojection error: "
                f"{max_reproj:.6f} px\n"
            )
        else:
            f.write(
                "  maximum reprojection error: N/A\n"
            )

        f.write("\nT_base_ee = ^B T_E:\n")
        f.write(format_matrix(T_base_ee) + "\n")

        f.write("\nT_cam_board = ^C T_W:\n")
        f.write(format_matrix(T_cam_board) + "\n")

        f.write("\nT_base_cam = ^B T_C:\n")
        f.write(format_matrix(T_base_cam) + "\n")

        f.write("\nDifference from sample 0:\n")
        f.write(
            f"  translation: "
            f"{translation_difference_mm:.6f} mm\n"
        )
        f.write(
            f"  rotation:    "
            f"{rotation_difference_deg_value:.6f} deg\n"
        )

    return (
        translation_difference_mm,
        rotation_difference_deg_value,
    )

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
    last_translation_difference_mm,
    last_rotation_difference_deg,
):
    """
    Draw the keyboard-control state without assuming a large terminal.
    """

    stdscr.clear()

    safe_addstr(
        stdscr,
        0,
        0,
        "SO-101 Fixed-Camera Transform Consistency Test",
    )
    safe_addstr(
        stdscr,
        1,
        0,
        "------------------------------------------------",
    )

    safe_addstr(stdscr, 3, 0, "Controls:")
    safe_addstr(stdscr, 4, 2, "TAB         : switch joint")
    safe_addstr(stdscr, 5, 2, "LEFT arrow  : move negative")
    safe_addstr(stdscr, 6, 2, "RIGHT arrow : move positive")
    safe_addstr(stdscr, 7, 2, "h           : move home")
    safe_addstr(stdscr, 8, 2, "r           : move rest")
    safe_addstr(
        stdscr,
        9,
        2,
        "s           : calculate and save T_base_cam sample",
    )
    safe_addstr(
        stdscr,
        10,
        2,
        "p           : print current robot FK",
    )
    safe_addstr(stdscr, 11, 2, "q           : quit")

    safe_addstr(
        stdscr,
        13,
        0,
        f"Board pose detected : {detected}",
    )
    safe_addstr(
        stdscr,
        14,
        0,
        f"ChArUco corners     : "
        f"{num_charuco}/{MAX_CHARUCO_CORNERS}",
    )
    safe_addstr(
        stdscr,
        15,
        0,
        f"Mean reproj error   : {mean_reproj_error}",
    )
    safe_addstr(
        stdscr,
        16,
        0,
        f"Samples saved       : {sample_count}",
    )
    safe_addstr(
        stdscr,
        17,
        0,
        f"Last translation Δ  : "
        f"{last_translation_difference_mm}",
    )
    safe_addstr(
        stdscr,
        18,
        0,
        f"Last rotation Δ     : "
        f"{last_rotation_difference_deg}",
    )
    safe_addstr(
        stdscr,
        19,
        0,
        f"Output file         : {SAMPLE_FILE}",
    )
    safe_addstr(
        stdscr,
        20,
        0,
        "Known T_ee_board    : R=I, t=[0.0105, 0, 0] m",
    )

    safe_addstr(stdscr, 22, 0, "Current commanded joints:")

    for i, name in enumerate(joint_names):
        marker = "->" if i == active_joint_idx else "  "
        label = joint_labels[i]
        value = current_action[name]
        min_lim, max_lim = joint_limits[name]

        safe_addstr(
            stdscr,
            24 + i,
            0,
            (
                f"{marker} {label:15s}: {value:8.2f} deg   "
                f"limits [{min_lim:.1f}, {max_lim:.1f}]"
            ),
        )

    stdscr.refresh()


# ============================================================
# Main keyboard + camera loop
# ============================================================

def keyboard_control(stdscr, robot):
    """
    Run keyboard robot control and collect direct T_base_cam samples.
    """

    global current_action

    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    active_joint_idx = 0

    # Initialize the commanded-state dictionary from the robot's actual
    # measured joint positions. This prevents the first arrow-key command from
    # jumping from the hard-coded rest pose when the robot starts elsewhere.
    try:
        theta_feedback_deg = get_robot_feedback_angles_deg(
            robot,
            fallback_action=None,
        )

        current_action = {
            name: float(theta_feedback_deg[i])
            for i, name in enumerate(joint_names)
        }

        print(
            "[INFO] Initialized keyboard state from "
            "measured motor feedback."
        )

    except Exception as e:
        print(
            "[WARN] Could not initialize keyboard state "
            f"from feedback: {e}"
        )
        print("[WARN] Using the configured rest pose.")

        current_action = dict(rest)

    # --------------------------------------------------------
    # ZED setup
    # --------------------------------------------------------

    zed = open_zed_camera()
    runtime = sl.RuntimeParameters()
    image_zed = sl.Mat()

    # Create one display window and reuse it in both the live loop and the
    # camera averaging/save loop.
    open_display_window()

    K, dist = get_zed_left_intrinsics_rectified(zed)

    print("[INFO] ZED left camera matrix:")
    print(
        np.array2string(
            K,
            precision=4,
            suppress_small=True,
        )
    )
    print("[INFO] Distortion coefficients used:")
    print(dist.reshape(-1))

    # --------------------------------------------------------
    # ChArUco setup
    # --------------------------------------------------------

    board, dictionary = create_charuco_board()

    sample_id = 0

    # Sample 0 is used only as a comparison reference. No averaging of
    # T_base_cam is performed because we first want to see the raw consistency.
    T_reference_base_cam = None

    detected = False
    num_charuco = 0
    mean_reproj_error_text = "N/A"

    last_translation_difference_text = "N/A"
    last_rotation_difference_text = "N/A"

    try:
        while True:
            frame_bgr, gray = grab_left_bgr_and_gray(
                zed,
                runtime,
                image_zed,
            )

            if frame_bgr is not None:
                result = estimate_charuco_pose(
                    gray,
                    board,
                    dictionary,
                    K,
                    dist,
                )

                detected = is_good_camera_result(
                    result,
                    require_save_quality=False,
                )

                num_charuco = 0
                mean_reproj_error_text = "N/A"

                if result is not None:
                    num_charuco = result["num_charuco"]

                    mean_reproj_error_text = (
                        f"{result['mean_reproj_error_px']:.3f} px"
                    )

                    draw_charuco_debug(
                        frame_bgr,
                        result,
                        K,
                        dist,
                    )

                status_text = (
                    f"4x4 ChArUco: {detected} | "
                    f"Corners: {num_charuco}/"
                    f"{MAX_CHARUCO_CORNERS} | "
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

                cv2.imshow(WINDOW_NAME, frame_bgr)

                # Read keyboard input from the OpenCV window too. This means
                # `q` exits even when the camera window, rather than the
                # curses terminal, currently has focus.
                cv_key = cv2.waitKey(1) & 0xFF
                if cv_key == ord("q"):
                    break

            draw_screen(
                stdscr,
                active_joint_idx,
                detected,
                sample_id,
                num_charuco,
                mean_reproj_error_text,
                last_translation_difference_text,
                last_rotation_difference_text,
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
                print_robot_transform(
                    current_action,
                    robot=robot,
                )

            elif key == ord("s"):
                # ------------------------------------------------
                # 1. Keep the robot stationary and average several
                #    ChArUco solvePnP estimates.
                # ------------------------------------------------

                T_cam_board_avg, camera_info = (
                    collect_average_camera_pose(
                        zed,
                        runtime,
                        image_zed,
                        board,
                        dictionary,
                        K,
                        dist,
                    )
                )

                if T_cam_board_avg is None:
                    # `q` may be pressed while the averaging/save window is
                    # active. Exit the full keyboard loop instead of treating
                    # that action as a failed sample.
                    if camera_info.get("quit_requested", False):
                        break

                    safe_addstr(
                        stdscr,
                        32,
                        0,
                        (
                            "Cannot save: "
                            f"{camera_info['valid_frames']}/"
                            f"{camera_info['frames_seen']} "
                            "valid camera frames."
                        ),
                    )
                    stdscr.refresh()

                    print("\n[WARN] Sample not saved.")
                    print(
                        "Valid camera frames: "
                        f"{camera_info.get('valid_frames', 0)}/"
                        f"{camera_info.get('frames_seen', 0)}"
                    )

                    time.sleep(0.75)
                    continue

                # ------------------------------------------------
                # 2. Read actual motor feedback and compute FK.
                # ------------------------------------------------

                T_base_ee, robot_feedback_info = (
                    feedback_to_T_base_to_ee(
                        robot,
                        fallback_action=current_action,
                    )
                )

                # ------------------------------------------------
                # 3. Apply the known board mounting transform:
                #
                # T_base_cam =
                #     T_base_ee
                #     @ T_ee_board
                #     @ inverse(T_cam_board)
                # ------------------------------------------------

                T_base_cam = calculate_T_base_cam(
                    T_base_ee,
                    T_cam_board_avg,
                )

                # ------------------------------------------------
                # 4. Save sample 0 as the reference transform.
                #    Every later sample is compared against it.
                # ------------------------------------------------

                reference_for_comparison = T_reference_base_cam

                (
                    translation_difference_mm,
                    rotation_difference_value_deg,
                ) = append_sample_to_text_file(
                    sample_id=sample_id,
                    T_base_ee=T_base_ee,
                    T_cam_board=T_cam_board_avg,
                    T_base_cam=T_base_cam,
                    robot_feedback_info=robot_feedback_info,
                    camera_info=camera_info,
                    T_reference=reference_for_comparison,
                )

                if T_reference_base_cam is None:
                    T_reference_base_cam = T_base_cam.copy()

                last_translation_difference_text = (
                    f"{translation_difference_mm:.3f} mm"
                )
                last_rotation_difference_text = (
                    f"{rotation_difference_value_deg:.3f} deg"
                )

                print("\n" + "=" * 78)
                print(f"[SAVED] SAMPLE {sample_id:03d}")
                print("=" * 78)

                print("\nT_base_ee = ^B T_E:")
                print(format_matrix(T_base_ee))

                print("\nKnown T_ee_board = ^E T_W:")
                print(format_matrix(T_EE_BOARD))

                print("\nAveraged T_cam_board = ^C T_W:")
                print(format_matrix(T_cam_board_avg))

                print("\nCalculated T_base_cam = ^B T_C:")
                print(format_matrix(T_base_cam))

                print("\nDifference from sample 0:")
                print(
                    f"  translation: "
                    f"{translation_difference_mm:.6f} mm"
                )
                print(
                    f"  rotation:    "
                    f"{rotation_difference_value_deg:.6f} deg"
                )

                print(f"\nSaved to: {SAMPLE_FILE}")

                sample_id += 1

            elif key == 9:
                active_joint_idx = (
                    active_joint_idx + 1
                ) % len(joint_names)

            elif key in [
                curses.KEY_LEFT,
                curses.KEY_RIGHT,
            ]:
                joint = joint_names[active_joint_idx]

                direction = (
                    -1.0
                    if key == curses.KEY_LEFT
                    else 1.0
                )

                min_lim, max_lim = joint_limits[joint]

                current_action[joint] = clamp(
                    current_action[joint]
                    + direction * STEP_DEG,
                    min_lim,
                    max_lim,
                )

                current_action = send_robot_pose(
                    robot,
                    dict(current_action),
                )

                time.sleep(COMMAND_DELAY)

    finally:
        # Close the GUI first while the HighGUI event loop is still active,
        # then release the ZED camera.
        close_display_window()
        zed.close()


# ============================================================
# Entry point
# ============================================================

def main():
    initialize_sample_file()

    print("[INFO] Output file:")
    print(f"  {SAMPLE_FILE}")
    print("[INFO] Known T_ee_board:")
    print(format_matrix(T_EE_BOARD))

    robot = SO101Follower(
        SO101FollowerConfig(
            port=ROBOT_PORT,
            id=ROBOT_ID,
        )
    )

    try:
        robot.connect(calibrate=False)
        curses.wrapper(
            keyboard_control,
            robot,
        )

    finally:
        # Repeating GUI cleanup here is harmless and ensures windows are closed
        # even if setup or curses exits with an exception.
        close_display_window()

        try:
            robot.disconnect()
        except Exception as exc:
            print(f"[WARN] Robot disconnect failed: {exc}")

if __name__ == "__main__":
    main()