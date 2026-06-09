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
IK_PATH = ROOT / "robot_control" / "IK"
sys.path.append(str(IK_PATH))

from robot import M, S_list
from fk import space_product_of_exponentials


# ============================================================
# Files / folders
# ============================================================

POSE_FILE = ROOT / "data" / "current_robot_pose.json"
OUTPUT_DIR = ROOT / "data" / "eye_to_hand"


# ============================================================
# Robot configuration
# ============================================================

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "dbot"

STEP_DEG = 2.0
COMMAND_DELAY = 0.03

# Joint 6 / gripper is locked during calibration
GRIPPER_FIXED_DEG = 90.0


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
    "shoulder_pan.pos": (-90.0, 90.0),
    "shoulder_lift.pos": (-105.0, 90.0),
    "elbow_flex.pos": (-90.0, 95.0),
    "wrist_flex.pos": (-90.0, 90.0),
    "wrist_roll.pos": (-90.0, 90.0),

    # Gripper is fixed at 90 degrees.
    "gripper.pos": (GRIPPER_FIXED_DEG, GRIPPER_FIXED_DEG),
}


home = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": 0.0,
    "elbow_flex.pos": 0.0,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": GRIPPER_FIXED_DEG,
}


rest = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": -105.0,
    "elbow_flex.pos": 95.0,
    "wrist_flex.pos": -90.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": GRIPPER_FIXED_DEG,
}


current_action = dict(rest)


# ============================================================
# File cleanup
# ============================================================

def delete_old_calibration_samples():
    """
    Delete old calibration samples at the beginning of a new run.

    This prevents accidentally mixing old calibration data with new data.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    patterns = [
        "T_base_to_ee_*.npy",
        "T_camera_to_board_*.npy",
        "eye_to_hand_samples.npz",
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
    """
    Keep a value within [min_value, max_value].
    """

    return max(min_value, min(max_value, value))


def save_current_pose(action):
    """
    Save the current commanded robot joint pose to JSON.

    The calibration sample uses this file indirectly so the robot pose
    and camera pose stay synchronized.
    """

    POSE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(POSE_FILE, "w") as f:
        json.dump(action, f, indent=4)


def load_current_pose_if_available():
    """
    Load previous robot pose if it exists.

    This allows restarting the script without losing the last known pose.
    """

    global current_action

    if POSE_FILE.exists():
        with open(POSE_FILE, "r") as f:
            current_action = json.load(f)

    current_action["gripper.pos"] = GRIPPER_FIXED_DEG


# ============================================================
# Robot FK helpers
# ============================================================

def action_to_T_base_to_ee(action):
    """
    Convert robot joint dictionary into a 4x4 FK transform.

    Input:
        action:
            dictionary with joint values in degrees

    Output:
        T_base_to_ee:
            transform from robot base frame to end-effector frame
    """

    theta_deg = np.array(
        [float(action[name]) for name in joint_names],
        dtype=float
    )

    # Force joint 6 to the fixed calibration angle.
    theta_deg[5] = GRIPPER_FIXED_DEG

    theta_rad = np.radians(theta_deg)

    T_base_to_ee = space_product_of_exponentials(
        M,
        S_list,
        theta_rad
    )

    return T_base_to_ee


# ============================================================
# Transform helpers
# ============================================================

def make_T(R, t):
    """
    Build a homogeneous transform from a rotation matrix and translation.
    """

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)

    return T


def rvec_tvec_to_T(rvec, tvec):
    """
    Convert OpenCV solvePnP output into a 4x4 transform.

    solvePnP returns:
        rvec, tvec

    Here that represents:
        T_camera_to_board

    because it maps board object points into the camera frame.
    """

    R, _ = cv2.Rodrigues(rvec)

    return make_T(R, tvec)


# ============================================================
# Robot movement helpers
# ============================================================

def send_and_save_pose(robot, action):
    """
    Send the joint command to the robot and save the same command to JSON.
    """

    action["gripper.pos"] = GRIPPER_FIXED_DEG

    robot.send_action(action)

    save_current_pose(action)


def move_smooth(robot, target_action):
    """
    Smoothly move the robot to a target joint pose.

    This is used for home/rest moves.
    """

    global current_action

    final_action = dict(current_action)

    for name in joint_names:
        if name in target_action:
            final_action[name] = float(target_action[name])

    final_action["gripper.pos"] = GRIPPER_FIXED_DEG

    current = np.array(
        [current_action[name] for name in joint_names],
        dtype=float
    )

    target = np.array(
        [final_action[name] for name in joint_names],
        dtype=float
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

        action["gripper.pos"] = GRIPPER_FIXED_DEG

        send_and_save_pose(robot, action)

        time.sleep(COMMAND_DELAY)

    current_action = final_action
    current_action["gripper.pos"] = GRIPPER_FIXED_DEG

    save_current_pose(current_action)


# ============================================================
# ZED helpers
# ============================================================

def get_zed_intrinsics(zed):
    """
    Read left-camera intrinsics from the ZED SDK.

    K:
        camera matrix used by solvePnP

    dist:
        distortion coefficients
    """

    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters.left_cam

    K = np.array([
        [calib.fx, 0.0, calib.cx],
        [0.0, calib.fy, calib.cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    dist = np.array(
        calib.disto,
        dtype=np.float64
    ).reshape(-1, 1)

    return K, dist


# ============================================================
# ArUco pose estimation helper
# ============================================================

def estimate_board_pose_from_aruco_markers(
    board,
    marker_corners,
    marker_ids,
    K,
    dist,
):
    """
    Estimate board pose using ONLY detected ArUco markers.

    This avoids CharucoDetector completely.

    Why this function exists:
        Your camera detects the ArUco markers successfully, but the
        ChArUco corner detector is not giving a valid pose. So this
        function manually matches detected marker IDs to the board's known
        3D marker corner coordinates.

    Inputs:
        board:
            OpenCV CharucoBoard object

        marker_corners:
            detected 2D image corners from ArucoDetector

        marker_ids:
            detected marker IDs from ArucoDetector

        K, dist:
            camera intrinsics and distortion

    Output:
        success:
            True if solvePnP succeeds

        rvec, tvec:
            pose of the board relative to the camera
    """

    if marker_ids is None or len(marker_ids) < 4:
        return False, None, None

    # The board knows which marker IDs exist on it.
    board_ids = board.getIds().flatten()

    # The board also knows the 3D object coordinates of every marker corner.
    # Each item is a 4x3 set of 3D points.
    board_obj_points = board.getObjPoints()

    obj_points = []
    img_points = []

    for detected_idx, detected_id in enumerate(marker_ids.flatten()):

        # Find where this detected marker ID appears in the board model.
        matches = np.where(board_ids == detected_id)[0]

        # Ignore marker IDs that are not part of this board.
        if len(matches) == 0:
            continue

        board_idx = matches[0]

        # 3D marker corners in board frame.
        obj_corners = np.asarray(
            board_obj_points[board_idx],
            dtype=np.float32
        ).reshape(4, 3)

        # 2D marker corners in image frame.
        img_corners = np.asarray(
            marker_corners[detected_idx],
            dtype=np.float32
        ).reshape(4, 2)

        obj_points.append(obj_corners)
        img_points.append(img_corners)

    # Need enough marker corners for stable PnP.
    # 4 markers = 16 points.
    if len(obj_points) < 4:
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
# Calibration sample saving
# ============================================================

def save_sample(
    sample_id,
    T_base_to_ee,
    T_camera_to_board,
    T_base_to_ee_list,
    T_camera_to_board_list,
    K,
    dist,
):
    """
    Save one hand-eye calibration sample.

    Each sample contains:

        Robot side:
            T_base_to_ee

        Camera side:
            T_camera_to_board

    Later the solver uses:
        T_base_to_ee * T_ee_to_board
        =
        T_base_to_camera * T_camera_to_board
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    T_base_to_ee_list.append(T_base_to_ee)
    T_camera_to_board_list.append(T_camera_to_board)

    np.save(
        OUTPUT_DIR / f"T_base_to_ee_{sample_id:03d}.npy",
        T_base_to_ee
    )

    np.save(
        OUTPUT_DIR / f"T_camera_to_board_{sample_id:03d}.npy",
        T_camera_to_board
    )

    np.savez(
        OUTPUT_DIR / "eye_to_hand_samples.npz",
        T_base_to_ee=np.array(T_base_to_ee_list),
        T_camera_to_board=np.array(T_camera_to_board_list),
        K=K,
        dist=dist,
    )


# ============================================================
# Safe curses display
# ============================================================

def safe_addstr(stdscr, y, x, text):
    """
    Print text in curses without crashing if terminal is too small.
    """

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
    """
    Draw terminal UI.
    """

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
    safe_addstr(stdscr, 10, 2, "q           : quit")

    safe_addstr(stdscr, 12, 0, f"Board pose detected: {detected}")
    safe_addstr(stdscr, 13, 0, f"Markers detected   : {num_markers}")
    safe_addstr(stdscr, 14, 0, f"Samples saved      : {sample_count}")
    safe_addstr(stdscr, 15, 0, f"Pose file          : {POSE_FILE}")
    safe_addstr(stdscr, 16, 0, f"Joint 6 fixed      : {GRIPPER_FIXED_DEG:.1f} deg")

    safe_addstr(stdscr, 18, 0, "Current joints:")

    for i, name in enumerate(joint_names):
        marker = "->" if i == active_joint_idx else "  "
        label = joint_labels[i]
        value = current_action[name]
        min_lim, max_lim = joint_limits[name]

        safe_addstr(
            stdscr,
            20 + i,
            0,
            f"{marker} {label:15s}: {value:8.2f} deg   limits [{min_lim:.1f}, {max_lim:.1f}]",
        )

    stdscr.refresh()


# ============================================================
# Main keyboard + camera loop
# ============================================================

def keyboard_control(stdscr, robot):
    """
    Main collection loop.

    This loop:
        1. Reads the ZED image.
        2. Detects ArUco markers.
        3. Estimates board pose from marker corners.
        4. Lets you move the robot using keyboard.
        5. Saves a calibration sample when pressing 's'.
    """

    global current_action

    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    active_joint_idx = 0

    load_current_pose_if_available()
    current_action["gripper.pos"] = GRIPPER_FIXED_DEG
    save_current_pose(current_action)

    # --------------------------------------------------------
    # ZED setup
    # --------------------------------------------------------

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL

    # OpenCV-friendly image coordinate system:
    # X right, Y down, Z forward.
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {status}")

    runtime = sl.RuntimeParameters()
    image_zed = sl.Mat()

    K, dist = get_zed_intrinsics(zed)

    # --------------------------------------------------------
    # ArUco detector setup
    # --------------------------------------------------------

    aruco = cv2.aruco

    dictionary = aruco.getPredefinedDictionary(
        aruco.DICT_5X5_100
    )

    detector_params = aruco.DetectorParameters()

    detector = aruco.ArucoDetector(
        dictionary,
        detector_params
    )

    # --------------------------------------------------------
    # Board model
    # --------------------------------------------------------
    #
    # Your printed target is visually ChArUco:
    #
    #   squares_x = 8
    #   squares_y = 8
    #   square size = 10 mm
    #   marker size = 7 mm
    #   dictionary = DICT_5X5
    #
    # We still create a CharucoBoard because it stores the correct
    # 3D coordinates of the embedded ArUco markers.
    #
    # But we do NOT use CharucoDetector here.
    # We only use ArucoDetector and solvePnP.
    # --------------------------------------------------------

    squares_x = 8
    squares_y = 8

    square_length = 0.010
    marker_length = 0.007

    board = aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length,
        marker_length,
        dictionary,
    )

    # --------------------------------------------------------
    # Calibration sample storage
    # --------------------------------------------------------

    T_base_to_ee_list = []
    T_camera_to_board_list = []

    sample_id = 0

    detected = False
    T_camera_to_board = None
    num_markers = 0

    try:
        while True:

            # ====================================================
            # Camera processing
            # ====================================================

            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                frame = image_zed.get_data()

                # ZED image can be BGRA.
                if frame.shape[2] == 4:
                    frame_bgr = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGRA2BGR
                    )
                else:
                    frame_bgr = frame.copy()

                gray = cv2.cvtColor(
                    frame_bgr,
                    cv2.COLOR_BGR2GRAY
                )

                # --------------------------------------------
                # Detect ArUco markers
                # --------------------------------------------

                marker_corners, marker_ids, rejected = detector.detectMarkers(
                    gray
                )

                detected = False
                T_camera_to_board = None

                num_markers = 0 if marker_ids is None else len(marker_ids)

                if marker_ids is not None and len(marker_ids) > 0:

                    # Draw detected marker outlines and IDs.
                    aruco.drawDetectedMarkers(
                        frame_bgr,
                        marker_corners,
                        marker_ids,
                    )

                # --------------------------------------------
                # Estimate board pose
                # --------------------------------------------

                success, rvec, tvec = estimate_board_pose_from_aruco_markers(
                    board,
                    marker_corners,
                    marker_ids,
                    K,
                    dist,
                )

                if success:
                    detected = True

                    T_camera_to_board = rvec_tvec_to_T(
                        rvec,
                        tvec
                    )

                    # Draw coordinate axes at board origin.
                    cv2.drawFrameAxes(
                        frame_bgr,
                        K,
                        dist,
                        rvec,
                        tvec,
                        0.03,
                    )

                # --------------------------------------------
                # Image overlay
                # --------------------------------------------

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
                    frame_bgr
                )

                cv2.waitKey(1)

            # ====================================================
            # Terminal UI
            # ====================================================

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

            # ----------------------------------------------------
            # Quit
            # ----------------------------------------------------

            if key == ord("q"):
                break

            # ----------------------------------------------------
            # Move to home
            # ----------------------------------------------------

            elif key == ord("h"):
                move_smooth(robot, home)

            # ----------------------------------------------------
            # Move to rest
            # ----------------------------------------------------

            elif key == ord("r"):
                move_smooth(robot, rest)

            # ----------------------------------------------------
            # Save sample
            # ----------------------------------------------------

            elif key == ord("s"):
                if not detected or T_camera_to_board is None:
                    safe_addstr(
                        stdscr,
                        28,
                        0,
                        "Cannot save: board pose not detected."
                    )
                    stdscr.refresh()
                    time.sleep(0.5)
                    continue

                # Robot-side pose from forward kinematics.
                T_base_to_ee = action_to_T_base_to_ee(
                    current_action
                )

                # Camera-side pose from marker board detection.
                save_sample(
                    sample_id,
                    T_base_to_ee,
                    T_camera_to_board,
                    T_base_to_ee_list,
                    T_camera_to_board_list,
                    K,
                    dist,
                )

                sample_id += 1

            # ----------------------------------------------------
            # Switch active joint
            # ----------------------------------------------------

            elif key == 9:
                active_joint_idx = (
                    active_joint_idx + 1
                ) % len(joint_names)

                # Skip gripper because fixed.
                if joint_names[active_joint_idx] == "gripper.pos":
                    active_joint_idx = (
                        active_joint_idx + 1
                    ) % len(joint_names)

            # ----------------------------------------------------
            # Joint movement
            # ----------------------------------------------------

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

                current_action["gripper.pos"] = GRIPPER_FIXED_DEG

                send_and_save_pose(
                    robot,
                    dict(current_action)
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
            robot
        )

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()