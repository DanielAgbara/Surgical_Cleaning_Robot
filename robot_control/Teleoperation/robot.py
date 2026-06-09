import time

import numpy as np
from fk import *
from se3 import *
from jacobian import *
from so3 import *
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