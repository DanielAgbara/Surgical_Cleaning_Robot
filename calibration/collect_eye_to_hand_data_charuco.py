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


ROOT = Path("/home/agbara-admin/Documents/Cleaning_Robot")
IK_PATH = ROOT / "robot_control" / "IK"
sys.path.append(str(IK_PATH))

from robot import M, S_list
from fk import space_product_of_exponentials


POSE_FILE = ROOT / "data" / "current_robot_pose.json"
OUTPUT_DIR = ROOT / "data" / "eye_to_hand"

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "dbot"

STEP_DEG = 2.0
COMMAND_DELAY = 0.03

GRIPPER_FIXED_DEG = 90.0


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


def delete_old_calibration_samples():
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


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def save_current_pose(action):
    POSE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(POSE_FILE, "w") as f:
        json.dump(action, f, indent=4)


def load_current_pose_if_available():
    global current_action

    if POSE_FILE.exists():
        with open(POSE_FILE, "r") as f:
            current_action = json.load(f)

        current_action["gripper.pos"] = GRIPPER_FIXED_DEG


def action_to_T_base_to_ee(action):
    theta_deg = np.array([float(action[name]) for name in joint_names])
    theta_deg[5] = GRIPPER_FIXED_DEG
    theta_rad = np.radians(theta_deg)

    return space_product_of_exponentials(M, S_list, theta_rad)


def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def rvec_tvec_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    return make_T(R, tvec)


def send_and_save_pose(robot, action):
    action["gripper.pos"] = GRIPPER_FIXED_DEG
    robot.send_action(action)
    save_current_pose(action)


def move_smooth(robot, target_action):
    global current_action

    final_action = dict(current_action)

    for name in joint_names:
        if name in target_action:
            final_action[name] = float(target_action[name])

    final_action["gripper.pos"] = GRIPPER_FIXED_DEG

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

        action["gripper.pos"] = GRIPPER_FIXED_DEG
        send_and_save_pose(robot, action)
        time.sleep(COMMAND_DELAY)

    current_action = final_action
    current_action["gripper.pos"] = GRIPPER_FIXED_DEG
    save_current_pose(current_action)


def get_zed_intrinsics(zed):
    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters.left_cam

    K = np.array([
        [calib.fx, 0.0, calib.cx],
        [0.0, calib.fy, calib.cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    dist = np.array(calib.disto, dtype=np.float64).reshape(-1, 1)

    return K, dist


def save_sample(
    sample_id,
    T_base_to_ee,
    T_camera_to_board,
    T_base_to_ee_list,
    T_camera_to_board_list,
    K,
    dist,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    T_base_to_ee_list.append(T_base_to_ee)
    T_camera_to_board_list.append(T_camera_to_board)

    np.save(OUTPUT_DIR / f"T_base_to_ee_{sample_id:03d}.npy", T_base_to_ee)
    np.save(OUTPUT_DIR / f"T_camera_to_board_{sample_id:03d}.npy", T_camera_to_board)

    np.savez(
        OUTPUT_DIR / "eye_to_hand_samples.npz",
        T_base_to_ee=np.array(T_base_to_ee_list),
        T_camera_to_board=np.array(T_camera_to_board_list),
        K=K,
        dist=dist,
    )


def safe_addstr(stdscr, y, x, text):
    h, w = stdscr.getmaxyx()

    if y >= h:
        return

    max_len = w - x - 1

    if max_len <= 0:
        return

    stdscr.addstr(y, x, str(text)[:max_len])


def draw_screen(stdscr, active_joint_idx, detected, sample_count, num_markers, num_charuco):
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
    safe_addstr(stdscr, 14, 0, f"ChArUco corners    : {num_charuco}")
    safe_addstr(stdscr, 15, 0, f"Samples saved      : {sample_count}")
    safe_addstr(stdscr, 16, 0, f"Pose file          : {POSE_FILE}")
    safe_addstr(stdscr, 17, 0, f"Joint 6 fixed      : {GRIPPER_FIXED_DEG:.1f} deg")

    safe_addstr(stdscr, 19, 0, "Current joints:")

    for i, name in enumerate(joint_names):
        marker = "->" if i == active_joint_idx else "  "
        label = joint_labels[i]
        value = current_action[name]
        min_lim, max_lim = joint_limits[name]

        safe_addstr(
            stdscr,
            21 + i,
            0,
            f"{marker} {label:15s}: {value:8.2f} deg   limits [{min_lim:.1f}, {max_lim:.1f}]",
        )

    stdscr.refresh()


def keyboard_control(stdscr, robot):
    global current_action

    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    active_joint_idx = 0

    load_current_pose_if_available()
    current_action["gripper.pos"] = GRIPPER_FIXED_DEG
    save_current_pose(current_action)

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL

    # Robotics-friendly ZED camera frame:
    # X = forward, Y = left, Z = up
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {status}")

    runtime = sl.RuntimeParameters()
    image_zed = sl.Mat()

    K, dist = get_zed_intrinsics(zed)

    aruco = cv2.aruco

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)

    # ChArUco board from calib.io:
    # 8x8, checker size 10 mm, marker size 7 mm, DICT_5X5
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

    detector_params = aruco.DetectorParameters()

    charuco_params = aruco.CharucoParameters()
    charuco_params.cameraMatrix = K
    charuco_params.distCoeffs = dist

    charuco_detector = aruco.CharucoDetector(
        board,
        charuco_params,
        detector_params,
    )

    T_base_to_ee_list = []
    T_camera_to_board_list = []
    sample_id = 0

    detected = False
    T_camera_to_board = None
    num_markers = 0
    num_charuco = 0

    try:
        while True:
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                frame = image_zed.get_data()

                if frame.shape[2] == 4:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                else:
                    frame_bgr = frame.copy()

                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

                charuco_corners, charuco_ids, marker_corners, marker_ids = (
                    charuco_detector.detectBoard(gray)
                )

                detected = False
                T_camera_to_board = None

                num_markers = 0 if marker_ids is None else len(marker_ids)
                num_charuco = 0 if charuco_ids is None else len(charuco_ids)

                if marker_ids is not None and len(marker_ids) > 0:
                    aruco.drawDetectedMarkers(
                        frame_bgr,
                        marker_corners,
                        marker_ids,
                    )

                if charuco_ids is not None and len(charuco_ids) > 0:
                    aruco.drawDetectedCornersCharuco(
                        frame_bgr,
                        charuco_corners,
                        charuco_ids,
                    )

                if charuco_ids is not None and len(charuco_ids) >= 4:
                    object_points, image_points = board.matchImagePoints(
                        charuco_corners,
                        charuco_ids,
                    )

                    success, rvec, tvec = cv2.solvePnP(
                        object_points,
                        image_points,
                        K,
                        dist,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )

                    if success:
                        detected = True
                        T_camera_to_board = rvec_tvec_to_T(rvec, tvec)

                        cv2.drawFrameAxes(
                            frame_bgr,
                            K,
                            dist,
                            rvec,
                            tvec,
                            0.03,
                        )

                cv2.putText(
                    frame_bgr,
                    f"Pose: {detected} | Markers: {num_markers} | ChArUco: {num_charuco} | Samples: {sample_id}",
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
                num_markers,
                num_charuco,
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

            elif key == ord("s"):
                if not detected or T_camera_to_board is None:
                    safe_addstr(stdscr, 29, 0, "Cannot save: board pose not detected.")
                    stdscr.refresh()
                    time.sleep(0.5)
                    continue

                T_base_to_ee = action_to_T_base_to_ee(current_action)

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

            elif key == 9:
                active_joint_idx = (active_joint_idx + 1) % len(joint_names)

                if joint_names[active_joint_idx] == "gripper.pos":
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

                current_action["gripper.pos"] = GRIPPER_FIXED_DEG
                send_and_save_pose(robot, dict(current_action))
                time.sleep(COMMAND_DELAY)

    finally:
        zed.close()
        cv2.destroyAllWindows()


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