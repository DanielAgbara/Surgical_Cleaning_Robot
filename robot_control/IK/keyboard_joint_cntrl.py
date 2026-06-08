#!/usr/bin/env python3

import sys
import time
import curses
import json
from pathlib import Path

import numpy as np

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


ROOT = Path("/home/agbara-admin/Documents/Cleaning_Robot")
POSE_FILE = ROOT / "data" / "current_robot_pose.json"

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

        action["gripper.pos"] = 0.0
        send_and_save_pose(robot, action)
        time.sleep(COMMAND_DELAY)

    current_action = final_action
    current_action["gripper.pos"] = 0.0
    save_current_pose(current_action)


def draw_screen(stdscr, active_joint_idx):
    stdscr.clear()

    stdscr.addstr(0, 0, "SO-101 Keyboard Joint Control")
    stdscr.addstr(1, 0, "-----------------------------")

    stdscr.addstr(3, 0, "Controls:")
    stdscr.addstr(4, 2, "TAB         : switch joint")
    stdscr.addstr(5, 2, "LEFT arrow  : move selected joint negative")
    stdscr.addstr(6, 2, "RIGHT arrow : move selected joint positive")
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

        stdscr.addstr(
            15 + i,
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