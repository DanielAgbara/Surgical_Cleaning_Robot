#!/usr/bin/env python3

import json
import numpy as np
from pathlib import Path

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IK_PATH = ROOT / "IK"

sys.path.insert(0, str(IK_PATH))

from robot import SOArm101

# --------------------------------------------------
# Files
# --------------------------------------------------

TRAJECTORY_JSON = Path(
    "/home/agbara-admin/Documents/Cleaning_Robot/data/arm_motion_trajectory.json"
)

# --------------------------------------------------
# Workspace limits around home
# --------------------------------------------------

HOME = np.array([0.391, 0.0, 0.243])

MAX_X_BACK = 0.05
MAX_X_FORWARD = 0

MAX_Y_LEFT = 0.05
MAX_Y_RIGHT = 0.05

MAX_Z_UP = 0.1
MAX_Z_DOWN = 0.2


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def clamp_target(target):

    target = target.copy()

    # X
    target[0] = np.clip(
        target[0],
        HOME[0] - MAX_X_BACK,
        HOME[0] + MAX_X_FORWARD
    )

    # Y
    target[1] = np.clip(
        target[1],
        HOME[1] - MAX_Y_RIGHT,
        HOME[1] + MAX_Y_LEFT
    )

    # Z
    target[2] = np.clip(
        target[2],
        HOME[2] - MAX_Z_DOWN,
        HOME[2] + MAX_Z_UP
    )

    return target


def trajectory_point_to_target(point):

    delta = np.array(
        [
            point["x"],
            point["y"],
            point["z"]
        ],
        dtype=float
    )

    target = HOME + delta

    return clamp_target(target)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    with open(TRAJECTORY_JSON, "r") as f:
        data = json.load(f)

    trajectory = data["trajectory"]

    print("Trajectory points:", len(trajectory))

    robot = SOArm101()

    robot.connect()

    try:

        robot.move_to_home()

        theta_current_deg = np.zeros(6)

        for i, point in enumerate(trajectory):

            p_des = trajectory_point_to_target(point)

            print(
                f"[{i}] "
                f"x={p_des[0]:.3f} "
                f"y={p_des[1]:.3f} "
                f"z={p_des[2]:.3f}"
            )

            theta_target_deg = robot.inverse_kinematics(
                p_des=p_des,
                theta_init=np.radians(theta_current_deg)
            )

            target_action = {
                "shoulder_pan.pos": float(theta_target_deg[0]),
                "shoulder_lift.pos": float(theta_target_deg[1]),
                "elbow_flex.pos": float(theta_target_deg[2]),
                "wrist_flex.pos": float(theta_target_deg[3]),
                "wrist_roll.pos": float(theta_target_deg[4]),
                "gripper.pos": 0.0,
            }

            robot.moveSO101(
                target_action,
                max_step_deg=2.0,
                step_delay=0.03
            )

            theta_current_deg = theta_target_deg.copy()

    finally:

        robot.move_to_rest()
        robot.disconnect()


if __name__ == "__main__":
    main()