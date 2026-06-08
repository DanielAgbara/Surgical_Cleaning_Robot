#!/usr/bin/env python3

import json
import time
import numpy as np
import sys
from pathlib import Path

# --------------------------------------------------
# Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
IK_PATH = ROOT / "IK"

sys.path.insert(0, str(IK_PATH))

# --------------------------------------------------
# Robot / kinematics imports
# --------------------------------------------------

from robot import SOArm101, S_list, M
from jacobian import body_jacobian, damped_least_square_inverse, space_jacobian
from se3 import adjoint_transform_list

# --------------------------------------------------
# Files
# --------------------------------------------------

VELOCITY_JSON = Path(
    "/home/agbara-admin/Documents/Cleaning_Robot/data/arm_tracking/arm_velocity_tracking.json"
)

# --------------------------------------------------
# Joint limits
# --------------------------------------------------

JOINT_MIN_RAD = np.radians([-105, -95, -90, -90, -90, -90])
JOINT_MAX_RAD = np.radians([105, 105, 95, 90, 90, 90])

# --------------------------------------------------
# Velocity control settings
# --------------------------------------------------

MAX_JOINT_SPEED_RAD_S = np.radians(360.0)
MAX_DT = 0.05
DAMPING = 0.01

# --------------------------------------------------
# Unit conversion
# --------------------------------------------------

MM_TO_M = 0.001

# --------------------------------------------------
# Twist settings
# --------------------------------------------------

USE_ANGULAR_VELOCITY = False

THETA_HOME_RAD = np.zeros(6)


def clamp_joint_angles(theta):
    """
    Clamp joint angles to safe joint limits.
    """

    return np.clip(
        theta,
        JOINT_MIN_RAD,
        JOINT_MAX_RAD
    )


def clamp_joint_velocity(theta_dot):
    """
    Clamp joint velocities to prevent sudden robot motion.
    """

    return np.clip(
        theta_dot,
        -MAX_JOINT_SPEED_RAD_S,
        MAX_JOINT_SPEED_RAD_S
    )


def get_vector_from_record(record, key):
    """
    Extract a 3D vector from one JSON record.

    The new arm-tracking JSON stores vectors like:

        "velocity_robot_command_mm_s": [vx, vy, vz]

    or sometimes as regular JSON lists.
    """

    value = record.get(key, None)

    if value is None:
        return np.zeros(3)

    return np.asarray(value, dtype=float).reshape(3)


def velocity_sample_to_twist(record, use_angular=False):
    """
    Convert one new-format JSON record into a robot twist.

    New JSON format expected:

        {
            "timestamp": ...,
            "dt": ...,
            "velocity_robot_command_mm_s": [vx, vy, vz],
            ...
        }

    The velocity is stored in mm/s and converted to m/s here.

    Returns
    -------
    V : np.ndarray, shape (6,)
        Twist:

            [wx, wy, wz, vx, vy, vz]

        Linear velocity is in m/s.
        Angular velocity is rad/s.
    """

    # --------------------------------------------------
    # Prefer robot command velocity if available
    # --------------------------------------------------

    if "velocity_robot_command_mm_s" in record:
        linear_mm_s = get_vector_from_record(
            record,
            "velocity_robot_command_mm_s"
        )

    # --------------------------------------------------
    # Fallback to filtered arm velocity
    # --------------------------------------------------

    elif "velocity_arm_filtered_mm_s" in record:
        linear_mm_s = get_vector_from_record(
            record,
            "velocity_arm_filtered_mm_s"
        )

    # --------------------------------------------------
    # Fallback to raw arm velocity
    # --------------------------------------------------

    elif "velocity_arm_mm_s" in record:
        linear_mm_s = get_vector_from_record(
            record,
            "velocity_arm_mm_s"
        )

    else:
        linear_mm_s = np.zeros(3)

    # Convert mm/s to m/s
    linear_m_s = linear_mm_s * MM_TO_M

    vx, vy, vz = linear_m_s

    # Angular velocity is not in your current new JSON format.
    # Keep it zero unless you add angular velocity later.
    if use_angular:
        angular = get_vector_from_record(
            record,
            "angular_velocity_rad_s"
        )

        wx, wy, wz = angular

    else:
        wx = 0.0
        wy = 0.0
        wz = 0.0

    return np.array(
        [wx, wy, wz, vx, vy, vz],
        dtype=float
    )


def sample_dt(current_record, previous_record):
    """
    Compute timestep between new-format JSON records.
    """

    if "dt" in current_record:
        dt = float(current_record["dt"])
    else:
        dt = (
            float(current_record["timestamp"])
            - float(previous_record["timestamp"])
        )

    if dt <= 0:
        dt = 0.03

    return min(dt, MAX_DT)


def theta_to_action(theta_rad):
    """
    Convert joint angles in radians to SO-ARM command degrees.
    """

    theta_deg = np.degrees(theta_rad)

    return {
        "shoulder_pan.pos": float(theta_deg[0]),
        "shoulder_lift.pos": float(theta_deg[1]),
        "elbow_flex.pos": float(theta_deg[2]),
        "wrist_flex.pos": float(theta_deg[3]),
        "wrist_roll.pos": float(theta_deg[4]),
        "gripper.pos": 0.0,
    }


def main():
    """
    Replay human arm velocity data using Jacobian velocity control.

    New JSON pipeline:
        records
            ↓
        velocity_robot_command_mm_s
            ↓
        convert mm/s to m/s
            ↓
        twist V = [0, 0, 0, vx, vy, vz]
            ↓
        Jacobian inverse
            ↓
        theta_dot
            ↓
        robot command
    """

    with open(VELOCITY_JSON, "r") as f:
        data = json.load(f)

    velocity_records = data["records"]

    print(f"Velocity records: {len(velocity_records)}")
    print(f"Angular velocity enabled: {USE_ANGULAR_VELOCITY}")

    if len(velocity_records) < 2:
        print("Not enough velocity records.")
        return

    B_list = adjoint_transform_list(
        T=M,
        X_list=S_list,
        to_space=False
    )

    robot = SOArm101()
    robot.connect()

    try:
        robot.move_to_home()

        theta = THETA_HOME_RAD.copy()

        for i in range(1, len(velocity_records)):

            previous_record = velocity_records[i - 1]
            current_record = velocity_records[i]

            dt = sample_dt(
                current_record=current_record,
                previous_record=previous_record
            )

            V = velocity_sample_to_twist(
                current_record,
                use_angular=USE_ANGULAR_VELOCITY
            )

            J_b = body_jacobian(
                B_list=B_list,
                theta=theta
            )

            J_s = space_jacobian(
                S_list,
                theta
            )

            J_b_inv = damped_least_square_inverse(
                J_b,
                k=DAMPING
            )

            J_s_inv = damped_least_square_inverse(
                J_s,
                k=DAMPING
            )

            # --------------------------------------------------
            # Choose one:
            # --------------------------------------------------

            # Body-frame control
            theta_dot = J_b_inv @ V

            # Space-frame control
            # theta_dot = J_s_inv @ V

            theta_dot = clamp_joint_velocity(theta_dot)

            theta_next = theta + theta_dot * dt

            theta_next = clamp_joint_angles(theta_next)

            action = theta_to_action(theta_next)

            print(
                f"[{i}] "
                f"dt={dt:.3f} "
                f"V_m_s={np.round(V, 4)} "
                f"theta_dot_deg_s={np.round(np.degrees(theta_dot), 3)} "
                f"theta_deg={np.round(np.degrees(theta_next), 2)}"
            )

            robot.moveSO101(
                action,
                max_step_deg=1.0,
                step_delay=0.03
            )

            theta = theta_next.copy()

            time.sleep(dt)

    finally:
        robot.move_to_rest()
        robot.disconnect()


if __name__ == "__main__":
    main()