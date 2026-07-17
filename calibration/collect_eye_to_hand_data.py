#!/usr/bin/env python3

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

from robot import (
    M,
    S_list,
    JOINT_OFFSETS_DEG,
    theta_min_robot_deg,
    theta_max_robot_deg,
    home as robot_home,
    rest as robot_rest,
)



UTIL_PATH = ROOT / "robot_control" / "Util"
sys.path.insert(0, str(UTIL_PATH))
from so3 import RToQuaternion
from fk import space_product_of_exponentials


# ============================================================
# Files / folders
# ============================================================

POSE_FILE = ROOT / "data" / "eye_to_hand" / "current_robot_pose.json"
FK_FILE = ROOT / "data" / "eye_to_hand" /"current_robot_fk.json"

ROBOT_Q_FILE = ROOT / "data" / "eye_to_hand" / "robot_q.json"
ROBOT_T_FILE = ROOT / "data" / "eye_to_hand" / "robot_t.json"
CAMERA_Q_FILE = ROOT / "data" / "eye_to_hand" / "camera_q.json"
CAMERA_T_FILE = ROOT / "data" / "eye_to_hand" / "camera_t.json"

OUTPUT_DIR = ROOT / "data" / "eye_to_hand"


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
    Delete old calibration JSON files before starting a new
    eye-to-hand calibration session.
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
    Convert action dictionary to physical robot command angles.
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
    Compute forward kinematics from the current robot command.

    This uses the joint offsets from robot.py, especially joint 6.
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


def save_fk_json(action):
    """
    Save FK information to JSON.

    This creates a file similar to current_robot_pose.json, but with:
        - robot command angles
        - offset-corrected FK/model angles
        - T_base_to_ee
        - end-effector position
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
    Save current robot pose and FK.

    Files written:
        data/current_robot_pose.json
        data/current_robot_fk.json
    """

    POSE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(POSE_FILE, "w") as f:
        json.dump(action, f, indent=4)

    save_fk_json(action)


def load_current_pose_if_available():
    """
    Load previous saved pose if it exists.
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
# Transform helpers
# ============================================================

def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def rvec_tvec_to_T(rvec, tvec):
    """
    Convert solvePnP output to T_camera_to_board.
    """

    R, _ = cv2.Rodrigues(rvec)
    return make_T(R, tvec)


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


# ============================================================
# ArUco board pose estimation
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


def estimate_board_pose_from_aruco_markers(
    board,
    marker_corners,
    marker_ids,
    K,
    dist,
):
    """
    Estimate ArUco GridBoard pose using detected marker corners.
    """

    if marker_ids is None or len(marker_ids) < 2:
        return False, None, None

    board_ids = board.getIds().flatten()
    board_obj_points = board.getObjPoints()

    obj_points = []
    img_points = []

    for detected_idx, detected_id in enumerate(marker_ids.flatten()):

        detected_id = int(detected_id)

        matches = np.where(board_ids == detected_id)[0]

        if len(matches) == 0:
            continue

        board_idx = int(matches[0])

        obj_corners = np.asarray(
            board_obj_points[board_idx],
            dtype=np.float32,
        ).reshape(4, 3)

        img_corners = np.asarray(
            marker_corners[detected_idx],
            dtype=np.float32,
        ).reshape(4, 2)

        obj_points.append(obj_corners)
        img_points.append(img_corners)

    if len(obj_points) < 2:
        return False, None, None

    obj_points = np.vstack(obj_points).astype(np.float32)
    img_points = np.vstack(img_points).astype(np.float32)

    success, rvec, tvec = cv2.solvePnP(
        obj_points,
        img_points,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    return success, rvec, tvec

# ============================================================
# Save current camera -> board transform
# ============================================================

CURRENT_CAMERA_BOARD_FILE = (
    OUTPUT_DIR / "current_camera_board.json"
)


def save_current_camera_board(T_camera_to_board):
    """
    Save the current camera-to-board transform.

    The rotation is stored as a quaternion [w, x, y, z].

    The translation is stored as [x, y, z].

    This file is overwritten every frame and is intended
    only for debugging.
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
    }

    with open(CURRENT_CAMERA_BOARD_FILE, "w") as f:
        json.dump(data, f, indent=4)


# ============================================================
# Calibration sample saving
# ============================================================

def save_sample(
    T_base_to_ee,
    T_camera_to_board,
):
    """
    Save one calibration sample.

    Each sample is stored as:

        robot_q.json
        robot_t.json
        camera_q.json
        camera_t.json

    Rotations are stored as quaternions:
        [w, x, y, z]

    Translations are stored as:
        [x, y, z]
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Robot end-effector pose
    # --------------------------------------------------------

    append_transform_to_json(
        T_base_to_ee,
        ROBOT_Q_FILE,
        ROBOT_T_FILE,
    )

    # --------------------------------------------------------
    # Camera board pose
    # --------------------------------------------------------

    append_transform_to_json(
        T_camera_to_board,
        CAMERA_Q_FILE,
        CAMERA_T_FILE,
    )


# ============================================================
# Quaternion JSON export
# ============================================================

def append_transform_to_json(
    T,
    q_file,
    t_file,
):
    """
    Append one transform to the quaternion and translation files.

    Quaternion file format:
        [
            [w, x, y, z],
            [w, x, y, z],
            ...
        ]

    Translation file format:
        [
            [x, y, z],
            [x, y, z],
            ...
        ]
    """

    # --------------------------------------------------------
    # Extract rotation and translation
    # --------------------------------------------------------

    R = T[:3, :3]
    t = T[:3, 3]

    # Convert rotation matrix to quaternion using so3.py
    q = RToQuaternion(R)

    # --------------------------------------------------------
    # Load existing quaternion data
    # --------------------------------------------------------

    if q_file.exists():
        with open(q_file, "r") as f:
            q_data = json.load(f)
    else:
        q_data = []

    # --------------------------------------------------------
    # Load existing translation data
    # --------------------------------------------------------

    if t_file.exists():
        with open(t_file, "r") as f:
            t_data = json.load(f)
    else:
        t_data = []

    # --------------------------------------------------------
    # Append current sample
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Save updated JSON files
    # --------------------------------------------------------

    with open(q_file, "w") as f:
        json.dump(q_data, f, indent=4)

    with open(t_file, "w") as f:
        json.dump(t_data, f, indent=4)

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
    num_markers,
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
    safe_addstr(stdscr, 9, 2, "s           : save calibration sample")
    safe_addstr(stdscr, 10, 2, "p           : print/save current T_base_to_ee")
    safe_addstr(stdscr, 11, 2, "q           : quit")

    safe_addstr(stdscr, 13, 0, f"Board pose detected: {detected}")
    safe_addstr(stdscr, 14, 0, f"Markers detected   : {num_markers}")
    safe_addstr(stdscr, 15, 0, f"Samples saved      : {sample_count}")
    safe_addstr(stdscr, 16, 0, f"Pose JSON          : {POSE_FILE}")
    safe_addstr(stdscr, 17, 0, f"FK JSON            : {FK_FILE}")
    safe_addstr(stdscr, 18, 0, "Joint 6            : free / offset-corrected in FK")

    safe_addstr(stdscr, 20, 0, "Current joints:")

    for i, name in enumerate(joint_names):
        marker = "->" if i == active_joint_idx else "  "
        label = joint_labels[i]
        value = current_action[name]
        min_lim, max_lim = joint_limits[name]

        safe_addstr(
            stdscr,
            22 + i,
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

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {status}")

    runtime = sl.RuntimeParameters()
    image_zed = sl.Mat()

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

    T_base_to_ee_list = []
    T_camera_to_board_list = []

    sample_id = 0

    detected = False
    T_camera_to_board = None
    num_markers = 0

    try:
        while True:

            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image_zed, sl.VIEW.LEFT)
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

                detected = False
                T_camera_to_board = None
                num_markers = 0 if marker_ids is None else len(marker_ids)

                if marker_ids is not None and len(marker_ids) > 0:
                    aruco.drawDetectedMarkers(
                        frame_bgr,
                        marker_corners,
                        marker_ids,
                    )

                success, rvec, tvec = estimate_board_pose_from_aruco_markers(
                    board,
                    marker_corners,
                    marker_ids,
                    K,
                    dist,
                )

                if success:
                    detected = True
                    T_camera_to_board = rvec_tvec_to_T(rvec, tvec)
                    
                    save_current_camera_board(T_camera_to_board)

                    cv2.drawFrameAxes(
                        frame_bgr,
                        K,
                        dist,
                        rvec,
                        tvec,
                        0.08,
                    )

                cv2.putText(
                    frame_bgr,
                    f"Pose: {detected} | Markers: {num_markers} | Samples: {sample_id}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0) if detected else (0, 0, 255),
                    2,
                )

                cv2.imshow(
                    "ZED Eye-to-Hand Collection",
                    frame_bgr,
                )

                cv2.waitKey(1)

            draw_screen(
                stdscr,
                active_joint_idx,
                detected,
                sample_id,
                num_markers,
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

            elif key == ord("s"):
                if not detected or T_camera_to_board is None:
                    safe_addstr(
                        stdscr,
                        30,
                        0,
                        "Cannot save: board pose not detected.",
                    )
                    stdscr.refresh()
                    time.sleep(0.5)
                    continue

                T_base_to_ee = action_to_T_base_to_ee(current_action)
                
                save_sample(
                    T_base_to_ee,
                    T_camera_to_board,
                )

                save_fk_json(current_action)

                print("\n[SAVED] Sample", sample_id)
                print("T_base_to_ee:")
                print(T_base_to_ee)
                print("T_camera_to_board:")
                print(T_camera_to_board)
                print(f"FK JSON saved to: {FK_FILE}")

                sample_id += 1

            elif key == 9:
                active_joint_idx = (
                    active_joint_idx + 1
                ) % len(joint_names)

            elif key in [curses.KEY_LEFT, curses.KEY_RIGHT]:
                joint = joint_names[active_joint_idx]

                direction = (
                    -1.0 if key == curses.KEY_LEFT else 1.0
                )

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