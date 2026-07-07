#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np


# ============================================================
# Project paths
# ============================================================

ROOT = Path("/home/agbara-admin/Documents/Surgical_Cleaning_Robot")

TELEOP_PATH = ROOT / "robot_control" / "Teleoperation"
sys.path.insert(0, str(TELEOP_PATH))

from robot import SOArm101


# ============================================================
# Calibration path
# ============================================================

T_PATH = ROOT / "data" / "eye_to_hand" / "T_base_to_camera.npy"


# ============================================================
# Helpers
# ============================================================

def transform_point(T, p):
    """
    Transform a 3D point using a 4x4 homogeneous transform.
    """

    p_h = np.array(
        [p[0], p[1], p[2], 1.0],
        dtype=float
    )

    p_out = T @ p_h

    return p_out[:3]


def is_reachable(p_base):
    """
    Rough workspace check before IK.
    """

    x, y, z = p_base

    r = np.sqrt(x**2 + y**2)

    if z < 0.02:
        return False

    if z > 0.45:
        return False

    if r < 0.05:
        return False

    if r > 0.45:
        return False

    return True


# ============================================================
# Main
# ============================================================

def main():
    if len(sys.argv) != 4:
        print("\nUsage:")
        print("  python camera_point_to_ik.py X Y Z")
        print("\nExample:")
        print("  python camera_point_to_ik.py 0.25 0.05 0.60")
        print("\nUnits are meters in the camera frame.\n")
        return

    # ------------------------------------------------------------
    # Input point in camera frame
    # ------------------------------------------------------------

    p_camera = np.array(
        [
            float(sys.argv[1]),
            float(sys.argv[2]),
            float(sys.argv[3]),
        ],
        dtype=float
    )

    # ------------------------------------------------------------
    # Load hand-eye calibration
    # ------------------------------------------------------------

    T_base_to_camera = np.load(T_PATH)

    print("\nT_base_to_camera:")
    print(T_base_to_camera)

    print("\nInput point in camera frame:")
    print(p_camera)

    # ------------------------------------------------------------
    # Transform camera point to robot base frame
    # ------------------------------------------------------------

    p_robot = transform_point(
        T_base_to_camera,
        p_camera
    )

    print("\nPoint in robot base frame:")
    print(p_robot)

    reachable = is_reachable(p_robot)

    print("\nRough reachable check:")
    print(reachable)

    if not reachable:
        print("\n[WARN] Point may be outside rough robot workspace.")
        print("Still trying IK...\n")

    # ------------------------------------------------------------
    # IK
    # ------------------------------------------------------------

    robot = SOArm101(
        port="/dev/ttyACM0",
        id="dbot"
    )

    theta_init = robot.get_joint_angles_deg()

    theta_sol_robot_deg = robot.solve_position(
        p_des=p_robot,
        theta_init=theta_init,
        max_iters=300,
        tol_converge=2e-3
    )

    print("\nIK solution [robot command degrees]:")
    print(theta_sol_robot_deg)

    action = robot.theta_to_action(
        np.radians(
            robot.robot_deg_to_ik_deg(theta_sol_robot_deg)
        )
    )

    print("\nRobot action dictionary:")
    print(action)

    print("\nThis script only computes IK. It does NOT move the robot.")


if __name__ == "__main__":
    main()