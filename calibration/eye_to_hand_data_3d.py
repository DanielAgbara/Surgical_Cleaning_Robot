#!/usr/bin/env python3
"""
Eye-to-hand calibration data collector using ZED 3D-3D board pose.

This file intentionally mirrors collect_eye_to_hand_data.py for:
    - robot connection
    - keyboard joint control
    - current_action handling
    - robot-side FK / T_base_to_ee calculation
    - JSON saving format
    - current pose / FK debug files

The only intended difference is how T_camera_to_board is computed:
    collect_eye_to_hand_data.py : ArUco corners + solvePnP
    eye_to_hand_data_3d.py     : ArUco corners + ZED point cloud + 3D-3D registration

The estimated transform has the same convention as solvePnP:

    P_camera = R_camera_board * P_board + t_camera_board

So T_camera_to_board here is the board pose expressed in the camera frame.
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

IK_PATH = ROOT / "robot_control" / "Teleoperation"
sys.path.insert(0, str(IK_PATH))

from robot import (  # noqa: E402
    M,
    S_list,
    JOINT_OFFSETS_DEG,
    theta_min_robot_deg,
    theta_max_robot_deg,
    home as robot_home,
    rest as robot_rest,
)

from fk import space_product_of_exponentials  # noqa: E402

UTIL_PATH = ROOT / "robot_control" / "Util"
sys.path.insert(0, str(UTIL_PATH))

from so3 import RToQuaternion  # noqa: E402


# ============================================================
# Files / folders
# ============================================================

OUTPUT_DIR = ROOT / "data" / "eye_to_hand"

POSE_FILE = OUTPUT_DIR / "current_robot_pose.json"
FK_FILE = OUTPUT_DIR / "current_robot_fk.json"
CURRENT_CAMERA_BOARD_FILE = OUTPUT_DIR / "current_camera_board.json"

ROBOT_Q_FILE = OUTPUT_DIR / "robot_q.json"
ROBOT_T_FILE = OUTPUT_DIR / "robot_t.json"
CAMERA_Q_FILE = OUTPUT_DIR / "camera_q.json"
CAMERA_T_FILE = OUTPUT_DIR / "camera_t.json"


# ============================================================
# Robot configuration
# ============================================================

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "dbot"

STEP_DEG = 1.0
COMMAND_DELAY = 0.03


# ============================================================
# ArUco board settings
# ============================================================

MARKERS_X = 3
MARKERS_Y = 3

MARKER_LENGTH_M = 0.0615
MARKER_SEPARATION_M = 0.00372

DICTIONARY_ID = cv2.aruco.DICT_5X5_100


# ============================================================
# ZED settings
# ============================================================

CAMERA_RESOLUTION = sl.RESOLUTION.HD720
CAMERA_FPS = 15
DEPTH_MODE = sl.DEPTH_MODE.NEURAL
COORDINATE_SYSTEM = sl.COORDINATE_SYSTEM.IMAGE
COORDINATE_UNITS = sl.UNIT.METER


# ============================================================
# 3D corner extraction / registration settings
# ============================================================

# Window around each detected ArUco image corner used for local depth averaging.
CORNER_WINDOW = 5

# ZED IMAGE coordinates: X right, Y down, Z forward.
# For this calibration, valid board depth should be positive and close enough.
MAX_DEPTH_M = 2.0

# Minimum valid ZED points inside one corner window.
MIN_VALID_POINTS_PER_CORNER = 6

# Need at least two detected markers -> 8 corners.
MIN_VALID_CORRESPONDENCES = 8

# Reject very noisy 3D-3D registrations when saving.
# 0.005 m = 5 mm.
MAX_RMS_ERROR_TO_SAVE_M = 0.005


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
    Delete old calibration sample files before starting a new session.

    This mirrors collect_eye_to_hand_data.py and keeps the current robot
    pose/FK debug files available for starting from the last saved pose.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    patterns = [
        "robot_q.json",
        "robot_t.json",
        "camera_q.json",
        "camera_t.json",
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
    Convert an action dictionary to physical robot command angles.
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

    This is copied in structure from collect_eye_to_hand_data.py so the
    robot-side transform is generated exactly the same way.
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


def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


# ============================================================
# Robot pose / FK debug saving
# ============================================================

def save_fk_json(action):
    """
    Save FK debug information to JSON.
    """

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
    Save current robot action and FK.

    Files written:
        data/eye_to_hand/current_robot_pose.json
        data/eye_to_hand/current_robot_fk.json
    """

    POSE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(POSE_FILE, "w") as f:
        json.dump(action, f, indent=4)

    save_fk_json(action)


def load_current_pose_if_available():
    """
    Load previous saved robot action if it exists.
    """

    global current_action

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


def print_robot_transform(action):
    """
    Print FK debug info to terminal and save FK JSON.
    """

    theta_robot_deg = action_to_theta_robot_deg(action)
    theta_model_deg = robot_deg_to_model_deg(theta_robot_deg)
    T_base_to_ee = action_to_T_base_to_ee(action)

    save_fk_json(action)

    print("\n" + "=" * 60)
    print("CURRENT ROBOT FK DEBUG")
    print("=" * 60)

    print("\nRobot command angles [deg]:")
    for name, value in zip(joint_names, theta_robot_deg):
        print(f"  {name:20s}: {value: .3f}")

    print("\nOffset-corrected FK/model angles [deg]:")
    for name, value in zip(joint_names, theta_model_deg):
        print(f"  {name:20s}: {value: .3f}")

    print("\nT_base_to_ee:")
    print(T_base_to_ee)

    print(f"\n[SAVED FK JSON] {FK_FILE}")
    print("=" * 60 + "\n")


# ============================================================
# Robot movement helpers
# ============================================================

def send_and_save_pose(robot, action):
    """
    Send action to robot, save pose JSON, and save FK JSON.

    Joint 6 is free and offset-corrected only for FK.
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

        current_action = send_and_save_pose(robot, action)

        time.sleep(COMMAND_DELAY)

    current_action = final_action
    save_current_pose(current_action)


# ============================================================
# ZED helpers
# ============================================================

def get_zed_intrinsics(zed):
    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters.left_cam

    K = np.array([
        [calib.fx, 0.0, calib.cx],
        [0.0, calib.fy, calib.cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    dist = np.array(
        calib.disto,
        dtype=np.float64,
    ).reshape(-1, 1)

    return K, dist


def is_valid_camera_point(p):
    """
    Check whether a ZED point-cloud value is usable.
    """

    if p is None:
        return False

    p = np.asarray(p, dtype=float).reshape(3)

    if not np.all(np.isfinite(p)):
        return False

    if p[2] <= 0.0:
        return False

    if p[2] > MAX_DEPTH_M:
        return False

    return True


def get_point_cloud_xyz(point_cloud, u, v):
    """
    Read one XYZ point from the ZED point cloud.
    """

    err, point = point_cloud.get_value(int(u), int(v))

    if err != sl.ERROR_CODE.SUCCESS:
        return None

    p = np.asarray(point[:3], dtype=float)

    if not is_valid_camera_point(p):
        return None

    return p


def get_average_camera_point(point_cloud, u, v, window=CORNER_WINDOW):
    """
    Estimate a stable 3D camera point around a detected ArUco corner.

    A single stereo depth value at an edge/corner can be noisy. This samples
    a small window, rejects invalid values, removes local outliers, then
    averages the remaining points.
    """

    u = int(round(u))
    v = int(round(v))

    half = window // 2
    points = []

    for yy in range(v - half, v + half + 1):
        for xx in range(u - half, u + half + 1):
            p = get_point_cloud_xyz(point_cloud, xx, yy)

            if p is not None:
                points.append(p)

    if len(points) < MIN_VALID_POINTS_PER_CORNER:
        return None

    points = np.asarray(points, dtype=float)

    # --------------------------------------------------------
    # Outlier rejection using median distance from local median.
    # --------------------------------------------------------

    median = np.median(points, axis=0)

    distances = np.linalg.norm(
        points - median,
        axis=1,
    )

    distance_median = np.median(distances)
    mad = np.median(np.abs(distances - distance_median))

    if mad < 1e-9:
        filtered = points
    else:
        threshold = distance_median + 3.0 * mad
        filtered = points[distances <= threshold]

    if len(filtered) < MIN_VALID_POINTS_PER_CORNER:
        return None

    return np.mean(filtered, axis=0)


# ============================================================
# ArUco board setup / correspondence building
# ============================================================

def create_aruco_grid_board():
    """
    Create OpenCV GridBoard matching the physical board.

    Physical board:
        3 x 3 markers
        marker length = 61.5 mm
        marker gap = 3.72 mm
    """

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(DICTIONARY_ID)

    board = aruco.GridBoard(
        (MARKERS_X, MARKERS_Y),
        MARKER_LENGTH_M,
        MARKER_SEPARATION_M,
        dictionary,
    )

    return board, dictionary


def build_board_correspondences(board, marker_corners, marker_ids):
    """
    Build matching board-frame 3D points and image pixels.

    Returns:
        success
        P_board      : (N, 3), board coordinates with z = 0
        image_pixels : (N, 2), detected image coordinates
    """

    if marker_ids is None or len(marker_ids) < 2:
        return False, None, None

    board_ids = board.getIds().flatten()
    board_obj_points = board.getObjPoints()

    P_board = []
    image_pixels = []

    for detected_idx, detected_id in enumerate(marker_ids.flatten()):
        detected_id = int(detected_id)

        matches = np.where(board_ids == detected_id)[0]

        if len(matches) == 0:
            continue

        board_idx = int(matches[0])

        obj_corners = np.asarray(
            board_obj_points[board_idx],
            dtype=np.float64,
        ).reshape(4, 3)

        img_corners = np.asarray(
            marker_corners[detected_idx],
            dtype=np.float64,
        ).reshape(4, 2)

        for obj_corner, img_corner in zip(obj_corners, img_corners):
            P_board.append(obj_corner)
            image_pixels.append(img_corner)

    if len(P_board) < MIN_VALID_CORRESPONDENCES:
        return False, None, None

    return (
        True,
        np.asarray(P_board, dtype=np.float64),
        np.asarray(image_pixels, dtype=np.float64),
    )


def build_camera_correspondences_from_depth(point_cloud, P_board, image_pixels):
    """
    Convert detected ArUco image corners into 3D ZED camera points.

    Output correspondences are ready for rigid registration:

        P_camera = R * P_board_valid + t
    """

    P_board_valid = []
    P_camera = []

    for p_board, pixel in zip(P_board, image_pixels):
        u, v = pixel

        p_camera = get_average_camera_point(
            point_cloud,
            u,
            v,
            window=CORNER_WINDOW,
        )

        if p_camera is None:
            continue

        P_board_valid.append(p_board)
        P_camera.append(p_camera)

    P_board_valid = np.asarray(P_board_valid, dtype=np.float64)
    P_camera = np.asarray(P_camera, dtype=np.float64)

    if len(P_camera) < MIN_VALID_CORRESPONDENCES:
        return False, None, None

    return True, P_board_valid, P_camera


# ============================================================
# 3D-3D registration for T_camera_to_board
# ============================================================

def solve_camera_to_board_transform(P_board, P_camera):
    """
    Estimate the rigid transform from board frame to camera frame.

    Convention:
        P_camera = R_camera_board * P_board + t_camera_board

    This gives the same type of board pose that solvePnP returns.
    """

    if len(P_board) != len(P_camera):
        raise ValueError("Point correspondence mismatch.")

    if len(P_board) < 3:
        return False, None, None

    centroid_board = np.mean(P_board, axis=0)
    centroid_camera = np.mean(P_camera, axis=0)

    board_centered = P_board - centroid_board
    camera_centered = P_camera - centroid_camera

    H = board_centered.T @ camera_centered

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # Reflection correction.
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_camera - R @ centroid_board

    T_camera_to_board = make_T(R, t)

    P_est = (R @ P_board.T).T + t
    errors = np.linalg.norm(P_est - P_camera, axis=1)
    rms_error = float(np.sqrt(np.mean(errors**2)))

    return True, T_camera_to_board, rms_error


def estimate_board_pose_from_aruco_depth(
    board,
    marker_corners,
    marker_ids,
    point_cloud,
):
    """
    Estimate T_camera_to_board using ArUco corner IDs and ZED 3D points.

    This replaces collect_eye_to_hand_data.py's solvePnP camera-side method.
    """

    success_board, P_board, image_pixels = build_board_correspondences(
        board,
        marker_corners,
        marker_ids,
    )

    if not success_board:
        return False, None, None, 0, False

    success_depth, P_board_valid, P_camera = build_camera_correspondences_from_depth(
        point_cloud,
        P_board,
        image_pixels,
    )

    if not success_depth:
        return False, None, None, 0, False

    success_registration, T_camera_to_board, rms_error = solve_camera_to_board_transform(
        P_board_valid,
        P_camera,
    )

    num_valid_points = len(P_camera)

    return (
        success_registration,
        T_camera_to_board,
        rms_error,
        num_valid_points,
        True,
    )


# ============================================================
# Save current camera -> board transform
# ============================================================

def save_current_camera_board(
    T_camera_to_board,
    rms_error=None,
    num_valid_points=None,
):
    """
    Save the current camera-to-board transform for live debugging.
    """

    R = T_camera_to_board[:3, :3]
    t = T_camera_to_board[:3, 3]

    q = RToQuaternion(R)

    data = {
        "quaternion": [
            float(q[0]),
            float(q[1]),
            float(q[2]),
            float(q[3]),
        ],
        "translation": [
            float(t[0]),
            float(t[1]),
            float(t[2]),
        ],
        "transform": T_camera_to_board.tolist(),
        "method": "aruco_corners_zed_point_cloud_3d_3d_registration",
    }

    if rms_error is not None:
        data["registration_rms_error_m"] = float(rms_error)
        data["registration_rms_error_mm"] = float(rms_error * 1000.0)

    if num_valid_points is not None:
        data["valid_3d_corner_points"] = int(num_valid_points)

    with open(CURRENT_CAMERA_BOARD_FILE, "w") as f:
        json.dump(data, f, indent=4)


# ============================================================
# Calibration sample saving
# ============================================================

def append_transform_to_json(T, q_file, t_file):
    """
    Append one transform to quaternion and translation JSON files.
    """

    R = T[:3, :3]
    t = T[:3, 3]

    q = RToQuaternion(R)

    if q_file.exists():
        with open(q_file, "r") as f:
            q_data = json.load(f)
    else:
        q_data = []

    if t_file.exists():
        with open(t_file, "r") as f:
            t_data = json.load(f)
    else:
        t_data = []

    q_data.append([
        float(q[0]),
        float(q[1]),
        float(q[2]),
        float(q[3]),
    ])

    t_data.append([
        float(t[0]),
        float(t[1]),
        float(t[2]),
    ])

    with open(q_file, "w") as f:
        json.dump(q_data, f, indent=4)

    with open(t_file, "w") as f:
        json.dump(t_data, f, indent=4)


def save_sample(T_base_to_ee, T_camera_to_board):
    """
    Save one calibration sample.

    Same output format as collect_eye_to_hand_data.py:
        robot_q.json
        robot_t.json
        camera_q.json
        camera_t.json
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    append_transform_to_json(
        T_base_to_ee,
        ROBOT_Q_FILE,
        ROBOT_T_FILE,
    )

    append_transform_to_json(
        T_camera_to_board,
        CAMERA_Q_FILE,
        CAMERA_T_FILE,
    )


def get_sample_count():
    if CAMERA_Q_FILE.exists():
        with open(CAMERA_Q_FILE, "r") as f:
            return len(json.load(f))

    return 0


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
    board_detected,
    depth_success,
    registration_success,
    sample_count,
    num_markers,
    num_valid_points,
    rms_error,
):
    stdscr.clear()

    safe_addstr(stdscr, 0, 0, "SO-101 Eye-to-Hand Keyboard Collection - 3D Board Pose")
    safe_addstr(stdscr, 1, 0, "------------------------------------------------------")

    safe_addstr(stdscr, 3, 0, "Controls:")
    safe_addstr(stdscr, 4, 2, "TAB         : switch joint")
    safe_addstr(stdscr, 5, 2, "LEFT arrow  : move negative")
    safe_addstr(stdscr, 6, 2, "RIGHT arrow : move positive")
    safe_addstr(stdscr, 7, 2, "h           : move home")
    safe_addstr(stdscr, 8, 2, "r           : move rest")
    safe_addstr(stdscr, 9, 2, "s           : save calibration sample")
    safe_addstr(stdscr, 10, 2, "p           : print/save current T_base_to_ee")
    safe_addstr(stdscr, 11, 2, "q           : quit")

    safe_addstr(stdscr, 13, 0, f"Markers detected       : {num_markers}")
    safe_addstr(stdscr, 14, 0, f"Board corners detected : {board_detected}")
    safe_addstr(stdscr, 15, 0, f"ZED depth success      : {depth_success}")
    safe_addstr(stdscr, 16, 0, f"3D registration success: {registration_success}")
    safe_addstr(stdscr, 17, 0, f"Valid 3D points        : {num_valid_points}")

    if rms_error is None:
        rms_text = "N/A"
    else:
        rms_text = f"{rms_error * 1000.0:.3f} mm"

    safe_addstr(stdscr, 18, 0, f"3D-3D RMS error        : {rms_text}")
    safe_addstr(stdscr, 19, 0, f"Samples saved          : {sample_count}")
    safe_addstr(stdscr, 20, 0, f"Pose JSON              : {POSE_FILE}")
    safe_addstr(stdscr, 21, 0, f"FK JSON                : {FK_FILE}")
    safe_addstr(stdscr, 22, 0, "Joint 6                : free / offset-corrected in FK")

    safe_addstr(stdscr, 24, 0, "Current joints:")

    for i, name in enumerate(joint_names):
        marker = "->" if i == active_joint_idx else "  "
        label = joint_labels[i]
        value = current_action[name]
        min_lim, max_lim = joint_limits[name]

        safe_addstr(
            stdscr,
            26 + i,
            0,
            f"{marker} {label:15s}: {value:8.2f} deg   limits [{min_lim:.1f}, {max_lim:.1f}]",
        )

    stdscr.refresh()


# ============================================================
# OpenCV display helper
# ============================================================

def draw_camera_status(
    frame_bgr,
    board_detected,
    depth_success,
    registration_success,
    num_markers,
    num_valid_points,
    rms_error,
    sample_count,
):
    status_color = (0, 255, 0) if registration_success else (0, 0, 255)

    cv2.putText(
        frame_bgr,
        (
            f"Markers: {num_markers} | "
            f"Depth: {depth_success} | "
            f"3D Reg: {registration_success} | "
            f"Samples: {sample_count}"
        ),
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2,
    )

    if rms_error is None:
        rms_text = "RMS: N/A"
    else:
        rms_text = f"RMS: {rms_error * 1000.0:.2f} mm"

    cv2.putText(
        frame_bgr,
        f"Valid 3D points: {num_valid_points} | {rms_text}",
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2,
    )

    cv2.putText(
        frame_bgr,
        f"Board corners detected: {board_detected}",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2,
    )


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

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = CAMERA_RESOLUTION
    init_params.camera_fps = CAMERA_FPS
    init_params.coordinate_units = COORDINATE_UNITS
    init_params.depth_mode = DEPTH_MODE
    init_params.coordinate_system = COORDINATE_SYSTEM

    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {status}")

    runtime = sl.RuntimeParameters()
    image_zed = sl.Mat()
    point_cloud = sl.Mat()

    K, dist = get_zed_intrinsics(zed)

    # --------------------------------------------------------
    # ArUco setup
    # --------------------------------------------------------

    aruco = cv2.aruco

    board, dictionary = create_aruco_grid_board()

    detector_params = aruco.DetectorParameters()

    detector = aruco.ArucoDetector(
        dictionary,
        detector_params,
    )

    sample_count = get_sample_count()

    board_detected = False
    depth_success = False
    registration_success = False
    T_camera_to_board = None
    latest_T_camera_to_board = None
    latest_rms_error = None
    num_markers = 0
    num_valid_points = 0
    rms_error = None

    try:
        while True:

            # ------------------------------------------------
            # Camera frame + point cloud
            # ------------------------------------------------

            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

                frame = image_zed.get_data()

                if frame.shape[2] == 4:
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

                marker_corners, marker_ids, rejected = detector.detectMarkers(
                    gray,
                )

                num_markers = 0 if marker_ids is None else len(marker_ids)
                board_detected = marker_ids is not None and len(marker_ids) >= 2
                depth_success = False
                registration_success = False
                T_camera_to_board = None
                rms_error = None
                num_valid_points = 0

                if marker_ids is not None and len(marker_ids) > 0:
                    aruco.drawDetectedMarkers(
                        frame_bgr,
                        marker_corners,
                        marker_ids,
                    )

                # ------------------------------------------------
                # This is the only major difference from
                # collect_eye_to_hand_data.py:
                #
                # Instead of solvePnP, use ZED depth at each marker
                # corner and solve a 3D-3D rigid registration.
                # ------------------------------------------------

                (
                    registration_success,
                    T_camera_to_board,
                    rms_error,
                    num_valid_points,
                    depth_success,
                ) = estimate_board_pose_from_aruco_depth(
                    board,
                    marker_corners,
                    marker_ids,
                    point_cloud,
                )

                if registration_success:
                    latest_T_camera_to_board = T_camera_to_board
                    latest_rms_error = rms_error

                    save_current_camera_board(
                        T_camera_to_board,
                        rms_error=rms_error,
                        num_valid_points=num_valid_points,
                    )

                    # Draw axes using the 3D-3D transform. This is only
                    # for visualization; the saved camera pose still comes
                    # from the point cloud registration.
                    rvec, _ = cv2.Rodrigues(T_camera_to_board[:3, :3])
                    tvec = T_camera_to_board[:3, 3].reshape(3, 1)

                    cv2.drawFrameAxes(
                        frame_bgr,
                        K,
                        dist,
                        rvec,
                        tvec,
                        0.08,
                    )

                draw_camera_status(
                    frame_bgr,
                    board_detected,
                    depth_success,
                    registration_success,
                    num_markers,
                    num_valid_points,
                    rms_error,
                    sample_count,
                )

                cv2.imshow(
                    "ZED Eye-to-Hand Collection - 3D Board Pose",
                    frame_bgr,
                )

                cv2.waitKey(1)

            draw_screen(
                stdscr,
                active_joint_idx,
                board_detected,
                depth_success,
                registration_success,
                sample_count,
                num_markers,
                num_valid_points,
                rms_error,
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
                print_robot_transform(current_action)

                if latest_T_camera_to_board is None:
                    print("[WARN] No valid 3D camera-board transform available yet.")
                else:
                    print("\n" + "=" * 60)
                    print("CURRENT CAMERA -> BOARD DEBUG, FROM 3D-3D REGISTRATION")
                    print("=" * 60)
                    print("T_camera_to_board:")
                    print(latest_T_camera_to_board)

                    if latest_rms_error is not None:
                        print(
                            f"\n3D-3D RMS error: "
                            f"{latest_rms_error * 1000.0:.3f} mm"
                        )

                    print("=" * 60 + "\n")

            elif key == ord("s"):
                if latest_T_camera_to_board is None:
                    safe_addstr(
                        stdscr,
                        34,
                        0,
                        "Cannot save: no valid 3D camera-board transform.",
                    )
                    stdscr.refresh()
                    time.sleep(0.5)
                    continue

                if latest_rms_error is not None and latest_rms_error > MAX_RMS_ERROR_TO_SAVE_M:
                    safe_addstr(
                        stdscr,
                        34,
                        0,
                        (
                            "Cannot save: 3D-3D RMS error too high "
                            f"({latest_rms_error * 1000.0:.2f} mm)."
                        ),
                    )
                    stdscr.refresh()
                    time.sleep(0.8)
                    continue

                # Robot-side data collection is exactly like collect_eye_to_hand_data.py.
                T_base_to_ee = action_to_T_base_to_ee(current_action)

                save_sample(
                    T_base_to_ee,
                    latest_T_camera_to_board,
                )

                save_fk_json(current_action)

                sample_count = get_sample_count()

                print("\n[SAVED] Sample", sample_count - 1)
                print("T_base_to_ee:")
                print(T_base_to_ee)
                print("T_camera_to_board [3D-3D]:")
                print(latest_T_camera_to_board)

                if latest_rms_error is not None:
                    print(
                        f"3D-3D RMS error: "
                        f"{latest_rms_error * 1000.0:.3f} mm"
                    )

                print(f"FK JSON saved to: {FK_FILE}")

            elif key == 9:
                active_joint_idx = (
                    active_joint_idx + 1
                ) % len(joint_names)

            elif key in [curses.KEY_LEFT, curses.KEY_RIGHT]:
                joint = joint_names[active_joint_idx]

                direction = -1.0 if key == curses.KEY_LEFT else 1.0

                min_lim, max_lim = joint_limits[joint]

                current_action[joint] = clamp(
                    current_action[joint] + direction * STEP_DEG,
                    min_lim,
                    max_lim,
                )

                current_action = send_and_save_pose(
                    robot,
                    dict(current_action),
                )

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

        curses.wrapper(
            keyboard_control,
            robot,
        )

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()