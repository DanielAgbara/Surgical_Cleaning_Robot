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
    "gripper.pos": (0.0, 0.0),
}

home = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": 0.0,
    "elbow_flex.pos": 0.0,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 0.0,
}

rest = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": -105.0,
    "elbow_flex.pos": 95.0,
    "wrist_flex.pos": -90.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 0.0,
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


def action_to_T_base_to_ee(action):
    theta_deg = np.array([float(action[name]) for name in joint_names])
    theta_deg[5] = 0.0
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
    action["gripper.pos"] = 0.0
    robot.send_action(action)
    save_current_pose(action)


def move_smooth(robot, target_action):
    global current_action

    final_action = dict(current_action)

    for name in joint_names:
        if name in target_action:
            final_action[name] = float(target_action[name])

    final_action["gripper.pos"] = 0.0

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

        action["gripper.pos"] = 0.0
        send_and_save_pose(robot, action)
        time.sleep(COMMAND_DELAY)

    current_action = final_action
    current_action["gripper.pos"] = 0.0
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


def draw_screen(stdscr, active_joint_idx, detected, sample_count):
    stdscr.clear()

    stdscr.addstr(0, 0, "SO-101 Eye-to-Hand Keyboard Collection")
    stdscr.addstr(1, 0, "--------------------------------------")

    stdscr.addstr(3, 0, "Controls:")
    stdscr.addstr(4, 2, "TAB         : switch joint")
    stdscr.addstr(5, 2, "LEFT arrow  : move negative")
    stdscr.addstr(6, 2, "RIGHT arrow : move positive")
    stdscr.addstr(7, 2, "h           : move home")
    stdscr.addstr(8, 2, "r           : move rest")
    stdscr.addstr(9, 2, "s           : save calibration sample")
    stdscr.addstr(10, 2, "q           : quit")

    stdscr.addstr(12, 0, f"Board detected: {detected}")
    stdscr.addstr(13, 0, f"Samples saved : {sample_count}")
    stdscr.addstr(14, 0, f"Pose file     : {POSE_FILE}")

    stdscr.addstr(16, 0, "Current joints:")

    for i, name in enumerate(joint_names):
        marker = "->" if i == active_joint_idx else "  "
        label = joint_labels[i]
        value = current_action[name]
        min_lim, max_lim = joint_limits[name]

        stdscr.addstr(
            18 + i,
            0,
            f"{marker} {label:15s}: {value:8.2f} deg   limits [{min_lim:.1f}, {max_lim:.1f}]"
        )

    stdscr.refresh()


def keyboard_control(stdscr, robot):
    global current_action

    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    active_joint_idx = 0

    load_current_pose_if_available()
    save_current_pose(current_action)

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL

    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {status}")

    runtime = sl.RuntimeParameters()
    image_zed = sl.Mat()

    K, dist = get_zed_intrinsics(zed)

    aruco = cv2.aruco

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)

    markers_x = 5
    markers_y = 5

    marker_length = 0.010
    marker_separation = 0.002

    board = aruco.GridBoard(
        (markers_x, markers_y),
        marker_length,
        marker_separation,
        dictionary,
    )

    detector_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, detector_params)

    T_base_to_ee_list = []
    T_camera_to_board_list = []
    sample_id = 0

    detected = False
    T_camera_to_board = None

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

                corners, ids, rejected = detector.detectMarkers(gray)

                detected = False
                T_camera_to_board = None

                if ids is not None and len(ids) >= 4:
                    aruco.drawDetectedMarkers(frame_bgr, corners, ids)

                    object_points, image_points = board.matchImagePoints(corners, ids)

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
                    f"Detected: {detected} | Samples: {sample_id}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if detected else (0, 0, 255),
                    2,
                )

                cv2.imshow("ZED Eye-to-Hand Collection", frame_bgr)
                cv2.waitKey(1)

            draw_screen(stdscr, active_joint_idx, detected, sample_id)

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
                    stdscr.addstr(26, 0, "Cannot save: board not detected.")
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

                current_action["gripper.pos"] = 0.0
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