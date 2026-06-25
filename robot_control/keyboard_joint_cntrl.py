#!/usr/bin/env python3

import time
import curses
import json
from pathlib import Path

import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path("/home/agbara-admin/Documents/Cleaning_Robot")
POSE_FILE = ROOT / "data" / "current_robot_pose.json"


# --------------------------------------------------
# Robot settings
# --------------------------------------------------

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "dbot"


# --------------------------------------------------
# Motion settings
# --------------------------------------------------

# Step size for normal arm joints, in degrees
STEP_DEG = 2.0

# Separate step size for gripper.
# The gripper usually needs a larger step than revolute joints.
GRIPPER_STEP = 5.0

# Delay after each command
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
# Joint limits
# --------------------------------------------------

joint_limits = {
    "shoulder_pan.pos": (-120.0, 120.0),
    "shoulder_lift.pos": (-105.0, 90.0),
    "elbow_flex.pos": (-90.0, 95.0),
    "wrist_flex.pos": (-90.0, 90.0),
    "wrist_roll.pos": (-180.0, 180.0),

    # Gripper range.
    # If 0 to 100 does not fully open/close, adjust this range.
    "gripper.pos": (0.0, 100.0),
}


# --------------------------------------------------
# Named poses
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
    "shoulder_lift.pos": -105.0,
    "elbow_flex.pos": 95.0,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 0.0,
}


# Start software state at rest pose
current_action = dict(rest)


def clamp(value, min_value, max_value):
    """
    Clamp value between min_value and max_value.
    """

    return max(min_value, min(max_value, value))


def save_current_pose(action):
    """
    Save latest commanded pose to JSON.
    This is useful because the SO-ARM follower does not always provide
    reliable live feedback for every script.
    """

    POSE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(POSE_FILE, "w") as f:
        json.dump(action, f, indent=4)


def load_current_pose_if_available():
    """
    Load previous commanded pose if it exists.
    This helps the script continue from the last saved command.
    """

    global current_action

    if POSE_FILE.exists():
        with open(POSE_FILE, "r") as f:
            loaded_action = json.load(f)

        # Only load keys that exist in joint_names
        for name in joint_names:
            if name in loaded_action:
                current_action[name] = float(loaded_action[name])


def send_and_save_pose(robot, action):
    """
    Send command to robot and save it locally.
    """

    robot.send_action(action)
    save_current_pose(action)


def move_smooth(robot, target_action):
    """
    Smoothly move from current_action to target_action.
    This avoids sudden jumps.
    """

    global current_action

    final_action = dict(current_action)

    # Fill target values
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
    Draw terminal interface.
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
    stdscr.addstr(9, 2, "q           : quit")

    stdscr.addstr(11, 0, f"Pose file: {POSE_FILE}")

    stdscr.addstr(13, 0, "Current joints:")

    for i, name in enumerate(joint_names):
        marker = "->" if i == active_joint_idx else "  "
        label = joint_labels[i]
        value = current_action[name]
        min_lim, max_lim = joint_limits[name]

        if name == "gripper.pos":
            unit = "%"
        else:
            unit = "deg"

        stdscr.addstr(
            15 + i,
            0,
            f"{marker} {label:15s}: {value:8.2f} {unit:3s}   limits [{min_lim:.1f}, {max_lim:.1f}]"
        )

    stdscr.refresh()


def keyboard_control(stdscr, robot):
    """
    Main keyboard loop.
    """

    global current_action

    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    active_joint_idx = 0

    # Load last saved pose
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

        elif key == 9:
            # TAB switches through ALL joints, including gripper.
            # This fixes the issue where gripper was skipped.
            active_joint_idx = (active_joint_idx + 1) % len(joint_names)

        elif key in [curses.KEY_LEFT, curses.KEY_RIGHT]:
            joint = joint_names[active_joint_idx]
            direction = -1.0 if key == curses.KEY_LEFT else 1.0

            min_lim, max_lim = joint_limits[joint]

            # Use larger step for gripper
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