import time

import numpy as np
from robot_control.utilities.fk import *
from robot_control.utilities.se3 import *
from robot_control.utilities.jacobian import *
from robot_control.utilities.so3 import *
from typing import Callable
from RR_IK import *
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

"""
This file contains the description of the robot.
The robot is a 6 DOF serial manipulator with the following joint configuration:
1. Shoulder pan
2. Shoulder lift
3. Elbow flex
4. Wrist flex
5. Wrist roll
6. Gripper
"""

# Main Robot Information
# Measurements are in meters and degrees
w1 = np.array([0, 0, 1])
q1 = np.array([0.038, 0, 0.064])

w2 = np.array([0, 1, 0])
q2 = np.array([0.06874, 0, 0.117050])

w3 = np.array([0, 1, 0])
q3 = np.array([0.097, 0, 0.228])

w4 = np.array([0, 1, 0])
q4 = np.array([0.225, 0, 0.228])

w5 = np.array([1, 0, 0])
q5 = np.array([0.289, 0, 0.228])

w6 = np.array([0, 1, 0])
q6 = np.array([0.314, 0, 0.243])

M = np.array([
    [1, 0, 0, 0.391],
    [0, 1, 0, 0.000],
    [0, 0, 1, 0.243],
    [0, 0, 0, 1.000]
])

S_list = [
    screw_axis_from_w_q(w1, q1),
    screw_axis_from_w_q(w2, q2),
    screw_axis_from_w_q(w3, q3),
    screw_axis_from_w_q(w4, q4),
    screw_axis_from_w_q(w5, q5),
    screw_axis_from_w_q(w6, q6),
]

# Convert to body frame
B_list = [adjoint(np.linalg.inv(M)) @ S for S in S_list]

theta_max = np.array([105, 105, 95, 90, 90, 90]) * np.pi / 180.0
theta_min = np.array([-105, -95, -90, -90, -90, -90]) * np.pi / 180.0

# Neutral starting pose
home = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": 0.0,
    "elbow_flex.pos": 0.0,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 0.0,
}

# Rest pose
rest = {
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": -105.0,
    "elbow_flex.pos": 95.0,
    "wrist_flex.pos": -90.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 0.0,
}


class SOArm101:
    def __init__(self, port="/dev/ttyACM0", id="dbot"):
        self.M = M
        self.S_list = S_list
        self.B_list = B_list
        self.home = home
        self.rest = rest
        self.theta_max = theta_max
        self.theta_min = theta_min
        self.robot = SO101Follower(SO101FollowerConfig(port=port, id=id))

        # Assume scripts begin with robot in rest pose
        self.current_action = dict(self.rest)
    
    def get_joint_angles_deg(self):
        joint_names = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]

        return np.array(
            [self.current_action[name] for name in joint_names],
            dtype=float
        )

    def get_T_base_to_ee(self):
        theta_deg = self.get_joint_angles_deg()

        # Keep gripper/joint 6 fixed during calibration
        theta_deg[5] = 0.0

        theta_rad = np.radians(theta_deg)

        T_base_to_ee = space_product_of_exponentials(
            self.M,
            self.S_list,
            theta_rad
        )

        return T_base_to_ee

    def moveSO101(self, target_action, max_step_deg=2.0, step_delay=0.05):
        joint_names = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]

        # Fill missing joints from current state
        final_action = dict(self.current_action)
        for name in joint_names:
            if name in target_action:
                final_action[name] = float(target_action[name])

        current = np.array([self.current_action[name] for name in joint_names], dtype=float)
        target = np.array([final_action[name] for name in joint_names], dtype=float)

        diff = target - current
        max_diff = np.max(np.abs(diff))

        if max_diff < 1e-9:
            return

        n_steps = max(1, int(np.ceil(max_diff / max_step_deg)))

        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            intermediate = current + alpha * diff

            action = {
                name: float(intermediate[idx])
                for idx, name in enumerate(joint_names)
            }
            self.robot.send_action(action)
            time.sleep(step_delay)

        # Update state after motion completes
        self.current_action = final_action

    def move_to_home(self, max_step_deg=2.0, step_delay=0.05):
        self.moveSO101(self.home, max_step_deg=max_step_deg, step_delay=step_delay)

    def move_to_rest(self, max_step_deg=2.0, step_delay=0.05):
        self.moveSO101(self.rest, max_step_deg=max_step_deg, step_delay=step_delay)

    def connect(self, calibrate=False):
        self.robot.connect(calibrate=calibrate)

    def disconnect(self):
        self.robot.disconnect()

    def inverse_kinematics(self, p_des, theta_init):
        theta_sol, _ = numerical_inverse_kinematics_position(
            M_ee=self.M,
            B_list=self.B_list,
            theta_init=theta_init,
            p_des=p_des,
            max_iters=100,
            tol_converge=1e-6,
            tol_manipulability=1e-3,
            q_min=self.theta_min,
            q_max=self.theta_max,
            k_null=0.1,
            k_damping=0.01,
            print_iterations=False,
        )

        theta_sol = np.asarray(theta_sol, dtype=float)
        theta_deg = np.degrees(theta_sol)
        return theta_deg