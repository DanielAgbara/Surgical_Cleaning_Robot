"""
test_pose.py

Test script for pose IK on SO-Arm101.

Goal:
    Move the end-effector to p = [0.4, 0.0, 0.1]
    while making the tool face downward.

Assumption:
    At home, the tool faces forward along its local +X axis.
    To make +X point downward in the base frame, use RotY(+90 deg).
"""

import time
import numpy as np

import robot as bot


def rot_y(theta):
    """
    Rotation matrix about the Y axis.
    """

    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


if __name__ == "__main__":

    arm = bot.SOArm101(
        port="/dev/ttyACM0",
        id="dbot"
    )

    arm.connect(calibrate=False)

    try:
        arm.move_to_home()
        time.sleep(2)

        # Desired position in meters
        p_sd = np.array([
            0.30,
            0.00,
            0.05,
        ])

        # Desired orientation:
        # tool +X axis points downward
        R_sd = rot_y(np.deg2rad(65.0))

        # Desired homogeneous transform
        T_sd = np.eye(4)
        T_sd[:3, :3] = R_sd
        T_sd[:3, 3] = p_sd

        theta_init = arm.get_joint_angles_deg()

        K = np.diag([
            0.08, 0.08, 0.08,
            0.8,  0.8,  0.8,
        ])

        (
            theta_solution,
            theta_history,
            norm_w_hist,
            norm_v_hist,
        ) = arm.solve_pose(
            T_des=T_sd,
            theta_init=theta_init,
            max_iters=500,
            tol_w=1e-3,
            tol_v=2e-3,
            K=K,
        )

        print("\nDesired pose T_sd:")
        print(T_sd)

        print("\nComputed joint angles [robot command deg]:")
        print(theta_solution)

        print("\nFinal orientation error:")
        print(norm_w_hist[-1])

        print("\nFinal position error:")
        print(norm_v_hist[-1])

        action = {
            "shoulder_pan.pos": float(theta_solution[0]),
            "shoulder_lift.pos": float(theta_solution[1]),
            "elbow_flex.pos": float(theta_solution[2]),
            "wrist_flex.pos": float(theta_solution[3]),
            "wrist_roll.pos": float(theta_solution[4]),
            "gripper.pos": float(theta_solution[5]),
        }

        print("Going to Solution now")
        arm.moveSO101(
            action,
            max_step_deg=2.0,
            step_delay=0.05,
        )

        

        time.sleep(10)
        print("done")

        arm.move_to_rest()

    finally:
        arm.disconnect()
        print("Disconnected")