#!/usr/bin/env python3

import time
import curses
import json
import sys
from pathlib import Path

import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
UTIL_PATH = ROOT / "Util"

sys.path.insert(0, str(UTIL_PATH))

from fk import body_product_of_exponentials
from se3 import screw_axis_from_w_q, adjoint


POSE_FILE = ROOT / "data" / "current_robot_pose.json"
EE_POSE_FILE = ROOT / "data" / "end_effector_pose.txt"

# --------------------------------------------------
# Robot settings
# --------------------------------------------------

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "dbot"


# --------------------------------------------------
# Motion settings
# --------------------------------------------------

STEP_DEG = 1.0
GRIPPER_STEP = 1.0
COMMAND_DELAY = 0.03


# --------------------------------------------------
# Joint names
# --------------------------------------------------

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


# --------------------------------------------------
# Joint limits in robot command degrees
# --------------------------------------------------

joint_limits = {
    "shoulder_pan.pos": (-120.0, 120.0),
    "shoulder_lift.pos": (-95.0, 105.0),
    "elbow_flex.pos": (-90.0, 95.0),
    "wrist_flex.pos": (-90.0, 90.0),
    "wrist_roll.pos": (-180.0, 180.0),
    "gripper.pos": (0.0, 100.0),
}


# --------------------------------------------------
# Named poses in robot command degrees
# --------------------------------------------------

home = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": 0.0,
    "elbow_flex.pos": 0.0,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 50.0,
}

rest = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": -95.0,
    "elbow_flex.pos": 95.0,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 0.0,
}


# --------------------------------------------------
# Robot geometry
# Measurements are in meters
# --------------------------------------------------

w1 = np.array([0, 0, 1], dtype=float)
q1 = np.array([0.038, 0, 0.064], dtype=float)

w2 = np.array([0, 1, 0], dtype=float)
q2 = np.array([0.06874, 0, 0.117050], dtype=float)

w3 = np.array([0, 1, 0], dtype=float)
q3 = np.array([0.097, 0, 0.228], dtype=float)

w4 = np.array([0, 1, 0], dtype=float)
q4 = np.array([0.225, 0, 0.228], dtype=float)

w5 = np.array([1, 0, 0], dtype=float)
q5 = np.array([0.289, 0, 0.228], dtype=float)

w6 = np.array([0, -1, 0], dtype=float)
q6 = np.array([0.326, 0, 0.228], dtype=float)


# Home end-effector pose
M = np.array([
    [1, 0, 0, 0.430],
    [0, 1, 0, 0.000],
    [0, 0, 1, 0.228],
    [0, 0, 0, 1.000],
], dtype=float)


# Space screw axes
S_list = [
    screw_axis_from_w_q(w1, q1),
    screw_axis_from_w_q(w2, q2),
    screw_axis_from_w_q(w3, q3),
    screw_axis_from_w_q(w4, q4),
    screw_axis_from_w_q(w5, q5),
    screw_axis_from_w_q(w6, q6),
]


# Convert space screw axes to body screw axes
B_list = [
    adjoint(np.linalg.inv(M)) @ S
    for S in S_list
]


# --------------------------------------------------
# Joint offset handling
# --------------------------------------------------
# Robot command angle and IK/FK angle are not always the same.
#
# For joint 6:
#   robot command range: 0 to 100 deg
#   physical home:      50 deg
#   IK/FK zero:         50 deg robot command
#
# Therefore:
#   theta_ik_deg = theta_robot_deg - 50
#   theta_robot_deg = theta_ik_deg + 50
# --------------------------------------------------

JOINT_OFFSETS_DEG = np.array([
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    50.0,
], dtype=float)


theta_min_robot_deg = np.array([
    -120.0,
    -95.0,
    -90.0,
    -90.0,
    -180.0,
    0.0,
], dtype=float)

theta_max_robot_deg = np.array([
    120.0,
    105.0,
    95.0,
    90.0,
    180.0,
    100.0,
], dtype=float)


# IK/FK limits in radians
theta_min = np.radians(theta_min_robot_deg - JOINT_OFFSETS_DEG)
theta_max = np.radians(theta_max_robot_deg - JOINT_OFFSETS_DEG)


# Start software state at rest pose
current_action = dict(rest)


def clamp(value, min_value, max_value):
    """
    Clamp a value between minimum and maximum limits.
    """

    return max(min_value, min(max_value, value))


def save_current_pose(action):
    """
    Save the latest commanded robot pose to JSON.
    """

    POSE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(POSE_FILE, "w") as f:
        json.dump(action, f, indent=4)


def load_current_pose_if_available():
    """
    Load the previous commanded pose if the pose file exists.
    """

    global current_action

    if POSE_FILE.exists():
        with open(POSE_FILE, "r") as f:
            loaded_action = json.load(f)

        for name in joint_names:
            if name in loaded_action:
                current_action[name] = float(loaded_action[name])


def send_and_save_pose(robot, action):
    """
    Send a pose command to the robot and save it locally.
    """

    robot.send_action(action)
    save_current_pose(action)


def get_theta_rad(action):
    """
    Convert robot command angles to IK/FK joint angles in radians.

    Robot command angles are what you send to the SO-ARM.

    IK/FK angles are the mathematical joint angles used by the
    product-of-exponentials model.

    Joint 6 has an offset:
        IK/FK zero = robot command 50 deg
    """

    theta_robot_deg = np.array([
        action["shoulder_pan.pos"],
        action["shoulder_lift.pos"],
        action["elbow_flex.pos"],
        action["wrist_flex.pos"],
        action["wrist_roll.pos"],
        action["gripper.pos"],
    ], dtype=float)

    theta_ik_deg = theta_robot_deg - JOINT_OFFSETS_DEG
    theta_ik_rad = np.deg2rad(theta_ik_deg)

    return theta_ik_rad


def print_end_effector_pose(action):
    """
    Compute the current end-effector pose using FK.

    The pose is:
        • Printed to the terminal.
        • Saved to data/end_effector_pose.txt.
    """

    theta = get_theta_rad(action)

    T_ee = body_product_of_exponentials(
        M,
        B_list,
        theta
    )

    p = T_ee[:3, 3]
    R = T_ee[:3, :3]

    theta_deg = np.rad2deg(theta)

    # --------------------------------------------------
    # Print to terminal
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("CURRENT END-EFFECTOR POSE")
    print("=" * 70)

    print("\nRobot command angles [deg]")

    for name in joint_names:
        print(f"{name:20s}: {action[name]:8.3f}")

    print("\nIK/FK angles [deg]")

    for i, angle in enumerate(theta_deg):
        print(f"theta{i+1}: {angle:8.3f}")

    print("\nPosition [m]")
    print(f"x = {p[0]:.6f}")
    print(f"y = {p[1]:.6f}")
    print(f"z = {p[2]:.6f}")

    print("\nRotation Matrix")
    print(R)

    print("\nHomogeneous Transform")
    print(T_ee)

    print("=" * 70)

    # --------------------------------------------------
    # Save to text file
    # --------------------------------------------------

    EE_POSE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(EE_POSE_FILE, "w") as f:

        f.write("=" * 70 + "\n")
        f.write("CURRENT END-EFFECTOR POSE\n")
        f.write("=" * 70 + "\n\n")

        f.write("Robot command angles [deg]\n")

        for name in joint_names:
            f.write(f"{name:20s}: {action[name]:8.3f}\n")

        f.write("\nIK/FK angles [deg]\n")

        for i, angle in enumerate(theta_deg):
            f.write(f"theta{i+1}: {angle:8.3f}\n")

        f.write("\nPosition [m]\n")
        f.write(f"x = {p[0]:.6f}\n")
        f.write(f"y = {p[1]:.6f}\n")
        f.write(f"z = {p[2]:.6f}\n")

        f.write("\nRotation Matrix\n")
        f.write(np.array2string(R, precision=6))
        f.write("\n\n")

        f.write("Homogeneous Transform\n")
        f.write(np.array2string(T_ee, precision=6))
        f.write("\n")

    print(f"\nPose saved to:\n{EE_POSE_FILE}\n")


def move_smooth(robot, target_action):
    """
    Smoothly move from current_action to target_action.
    """

    global current_action

    final_action = dict(current_action)

    for name in joint_names:
        if name in target_action:
            final_action[name] = float(target_action[name])

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

    if max_diff < 1e-9:
        return

    n_steps = max(1, int(np.ceil(max_diff / STEP_DEG)))

    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        intermediate = current + alpha * diff

        action = {
            name: float(intermediate[idx])
            for idx, name in enumerate(joint_names)
        }

        send_and_save_pose(robot, action)
        time.sleep(COMMAND_DELAY)

    current_action = final_action
    save_current_pose(current_action)


def draw_screen(stdscr, active_joint_idx):
    """
    Draw the terminal control interface.
    """

    stdscr.clear()

    stdscr.addstr(0, 0, "SO-101 Keyboard Joint + Gripper Control")
    stdscr.addstr(1, 0, "---------------------------------------")

    stdscr.addstr(3, 0, "Controls:")
    stdscr.addstr(4, 2, "TAB         : switch selected joint, including gripper")
    stdscr.addstr(5, 2, "LEFT arrow  : decrease selected joint/gripper")
    stdscr.addstr(6, 2, "RIGHT arrow : increase selected joint/gripper")
    stdscr.addstr(7, 2, "h           : move home")
    stdscr.addstr(8, 2, "r           : move rest")
    stdscr.addstr(9, 2, "p           : print end-effector pose")
    stdscr.addstr(10, 2, "q           : quit")

    stdscr.addstr(12, 0, f"Pose file: {POSE_FILE}")
    stdscr.addstr(14, 0, "Current joints:")

    for i, name in enumerate(joint_names):
        marker = "->" if i == active_joint_idx else "  "
        label = joint_labels[i]
        value = current_action[name]
        min_lim, max_lim = joint_limits[name]

        unit = "%" if name == "gripper.pos" else "deg"

        stdscr.addstr(
            16 + i,
            0,
            f"{marker} {label:15s}: {value:8.2f} {unit:3s}   "
            f"limits [{min_lim:.1f}, {max_lim:.1f}]"
        )

    stdscr.refresh()


def keyboard_control(stdscr, robot):
    """
    Main keyboard control loop.
    """

    global current_action

    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    active_joint_idx = 0

    load_current_pose_if_available()
    save_current_pose(current_action)

    while True:
        draw_screen(stdscr, active_joint_idx)

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
            print_end_effector_pose(current_action)

        elif key == 9:
            active_joint_idx = (active_joint_idx + 1) % len(joint_names)

        elif key in [curses.KEY_LEFT, curses.KEY_RIGHT]:
            joint = joint_names[active_joint_idx]
            direction = -1.0 if key == curses.KEY_LEFT else 1.0

            min_lim, max_lim = joint_limits[joint]

            if joint == "gripper.pos":
                step = GRIPPER_STEP
            else:
                step = STEP_DEG

            current_action[joint] = clamp(
                current_action[joint] + direction * step,
                min_lim,
                max_lim,
            )

            send_and_save_pose(robot, dict(current_action))
            time.sleep(COMMAND_DELAY)


def main():
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