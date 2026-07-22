import time
import numpy as np
from typing import Callable
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

import sys
from pathlib import Path

ROOT = Path("/home/agbara-admin/Documents/Surgical_Cleaning_Robot")

UTIL_PATH = ROOT / "robot_control"/ "Util"
IK_PATH = ROOT / "robot_control"/ "IK"

# ---------------------------------------
# Util imports
# ---------------------------------------
sys.path.insert(0, str(UTIL_PATH))

from fk import *
from se3 import *
from jacobian import *
from so3 import *

# ---------------------------------------
# IK imports
# ---------------------------------------
sys.path.insert(0, str(IK_PATH))

from jacobian_transpose import *


"""
This file contains the description of the robot.
The robot is a 6 DOF serial manipulator with the following joint configuration:
1. Shoulder pan
2. Shoulder lift
3. Elbow flex
4. Wrist flex
5. Wrist roll
6. End-effector joint
"""


# --------------------------------------------------
# Robot geometry
# Measurements are in meters and degrees
# --------------------------------------------------

w1 = np.array([0, 0, -1])
q1 = np.array([0.038, 0, 0.065])

w2 = np.array([0, 1, 0])
q2 = np.array([0.06874, 0, 0.105])

w3 = np.array([0, 1, 0])
q3 = np.array([0.097, 0, 0.228])

w4 = np.array([0, 1, 0])
q4 = np.array([0.225, 0, 0.228])

w5 = np.array([1, 0, 0])
q5 = np.array([0.289, 0, 0.228])

w6 = np.array([0, 1, 0])
q6 = np.array([0.326, 0, 0.228])


M = np.array([
    [1, 0, 0, 0.430],
    [0, 1, 0, 0.000],
    [0, 0, 1, 0.228],
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


# Convert screw axes to body frame
B_list = [adjoint(np.linalg.inv(M)) @ S for S in S_list]


# --------------------------------------------------
# Joint offset handling
# --------------------------------------------------
# Robot command angle and IK angle are not always the same.
#
# For joint 6:
#   robot command range: 0 to 100 deg
#   physical home:      50 deg
#   IK zero:            50 deg robot command
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
    50.0
])


# Robot command limits in degrees
theta_min_robot_deg = np.array([
    -120.0,
    -106.0,
    -97.0,
    -95.0,
    -180.0,
    0.0
])

theta_max_robot_deg = np.array([
    120.0,
    106.0,
    97.0,
    95.0,
    180.0,
    100.0
])


# IK limits in radians
# These are offset-corrected limits used by FK/Jacobian/IK.
theta_min = np.radians(theta_min_robot_deg - JOINT_OFFSETS_DEG)
theta_max = np.radians(theta_max_robot_deg - JOINT_OFFSETS_DEG)


# --------------------------------------------------
# Robot poses
# These are physical robot command angles in degrees.
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
    "wrist_flex.pos": -90.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 50.0,
}


class SOArm101:
    def __init__(self, port="/dev/ttyACM0", id="dbot"):
        self.M = M
        self.S_list = S_list
        self.B_list = B_list

        self.home = home
        self.rest = rest

        self.joint_offsets_deg = JOINT_OFFSETS_DEG

        # IK limits, radians
        self.theta_max = theta_max
        self.theta_min = theta_min

        # Physical robot command limits, degrees
        self.theta_max_robot_deg = theta_max_robot_deg
        self.theta_min_robot_deg = theta_min_robot_deg

        self.robot = SO101Follower(SO101FollowerConfig(port=port, id=id))

        # Assume scripts begin with robot in rest pose
        self.current_action = dict(self.rest)

    # --------------------------------------------------
    # Joint conversion helpers
    # --------------------------------------------------

    def robot_deg_to_ik_deg(self, theta_robot_deg):
        """
        Convert physical robot command angles to IK/model angles.

        Robot joint 6:
            0 to 100 deg command

        IK joint 6:
            -50 to +50 deg

        Parameters
        ----------
        theta_robot_deg : array-like, shape (6,)
            Robot command angles in degrees.

        Returns
        -------
        theta_ik_deg : np.ndarray, shape (6,)
            Offset-corrected IK angles in degrees.
        """

        theta_robot_deg = np.asarray(theta_robot_deg, dtype=float).flatten()

        if len(theta_robot_deg) != 6:
            raise ValueError("theta_robot_deg must contain 6 joint values")

        return theta_robot_deg - self.joint_offsets_deg


    def ik_deg_to_robot_deg(self, theta_ik_deg):
        """
        Convert IK/model angles back to physical robot command angles.

        Parameters
        ----------
        theta_ik_deg : array-like, shape (6,)
            Offset-corrected IK angles in degrees.

        Returns
        -------
        theta_robot_deg : np.ndarray, shape (6,)
            Robot command angles in degrees.
        """

        theta_ik_deg = np.asarray(theta_ik_deg, dtype=float).flatten()

        if len(theta_ik_deg) != 6:
            raise ValueError("theta_ik_deg must contain 6 joint values")

        theta_robot_deg = theta_ik_deg + self.joint_offsets_deg

        theta_robot_deg = np.clip(
            theta_robot_deg,
            self.theta_min_robot_deg,
            self.theta_max_robot_deg
        )

        return theta_robot_deg


    def get_joint_angles_deg(self):
        """
        Return current physical robot command angles in degrees.
        """

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


    def get_theta_rad(self):
        """
        Get current commanded joint angles as IK/model angles in radians.
        """

        theta_robot_deg = self.get_joint_angles_deg()
        theta_ik_deg = self.robot_deg_to_ik_deg(theta_robot_deg)

        return np.radians(theta_ik_deg)


    def get_T_base_to_ee(self):
        """
        Compute FK using offset-corrected IK/model angles.
        """

        theta_robot_deg = self.get_joint_angles_deg()
        theta_ik_deg = self.robot_deg_to_ik_deg(theta_robot_deg)
        theta_ik_rad = np.radians(theta_ik_deg)

        T_base_to_ee = space_product_of_exponentials(
            self.M,
            self.S_list,
            theta_ik_rad
        )

        return T_base_to_ee


    def moveSO101(self, target_action, max_step_deg=2.0, step_delay=0.05):
        """
        Move the physical robot using robot command angles in degrees.
        """

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

        # Clip to physical robot command limits
        target_deg = np.array(
            [final_action[name] for name in joint_names],
            dtype=float
        )

        target_deg = np.clip(
            target_deg,
            self.theta_min_robot_deg,
            self.theta_max_robot_deg
        )

        final_action = {
            name: float(target_deg[i])
            for i, name in enumerate(joint_names)
        }

        current = np.array(
            [self.current_action[name] for name in joint_names],
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
        self.moveSO101(
            self.home,
            max_step_deg=max_step_deg,
            step_delay=step_delay
        )


    def move_to_rest(self, max_step_deg=2.0, step_delay=0.05):
        self.moveSO101(
            self.rest,
            max_step_deg=max_step_deg,
            step_delay=step_delay
        )


    def V_to_pos(self, previous_pos, velocity, duration, timestep):
        """
        Convert velocity command into final position using Euler integration.

        Parameters
        ----------
        previous_pos : array-like, shape (3,)
            Starting position [x, y, z]

        velocity : array-like, shape (3,)
            Linear velocity [vx, vy, vz]

        duration : float
            How long to apply the velocity [s]

        timestep : float
            Integration timestep [s]

        Returns
        -------
        final_pos : np.ndarray, shape (3,)
            Final position after applying velocity

        pos_history : np.ndarray, shape (N, 3)
            Position history during integration
        """

        previous_pos = np.asarray(previous_pos, dtype=float).reshape(3)
        velocity = np.asarray(velocity, dtype=float).reshape(3)

        if duration <= 0:
            return previous_pos.copy(), np.array([previous_pos.copy()])

        if timestep <= 0:
            raise ValueError("timestep must be greater than zero")

        pos = previous_pos.copy()
        pos_history = [pos.copy()]

        elapsed = 0.0

        while elapsed < duration:
            dt = min(timestep, duration - elapsed)

            # Euler integration
            pos = pos + velocity * dt

            pos_history.append(pos.copy())

            elapsed += dt

        return pos, np.asarray(pos_history)


    def solve_position(
        self,
        p_des,
        theta_init=None,
        max_iters=100,
        tol_converge=2e-3
    ):
        """
        Solve IK for desired EE position.

        Input theta_init is expected in physical robot command degrees.

        Returns
        -------
        theta_sol_robot_deg : np.ndarray
            Joint angles in physical robot command degrees.
        """

        p_des = np.asarray(p_des, dtype=float).reshape(3)

        if theta_init is None:
            theta_robot_deg = self.get_joint_angles_deg()
        else:
            theta_robot_deg = np.asarray(theta_init, dtype=float).flatten()

        if len(theta_robot_deg) != 6:
            raise ValueError("theta_init must contain 6 joint values")

        # Convert physical robot command angles to IK/model angles
        theta_ik_deg = self.robot_deg_to_ik_deg(theta_robot_deg)

        # Convert IK/model angles to radians for the solver
        theta_ik_rad = np.radians(theta_ik_deg)

        theta_sol_ik_rad, _ = jacobian_transpose_position(
            M_ee=self.M,
            B_list=self.B_list,
            theta_init=theta_ik_rad,
            p_des=p_des,
            max_iters=max_iters,
            tol_converge=tol_converge,
            q_min=self.theta_min,
            q_max=self.theta_max
        )

        # Convert IK result to degrees
        theta_sol_ik_deg = np.degrees(theta_sol_ik_rad)

        # Convert IK/model angles back to physical robot command angles
        theta_sol_robot_deg = self.ik_deg_to_robot_deg(theta_sol_ik_deg)

        return theta_sol_robot_deg
    
    def solve_pose(
        self,
        T_des,
        theta_init=None,
        max_iters=100,
        tol_w=1e-6,
        tol_v=1e-6,
        K=None,
    ):
        """
        Solve IK for a desired end-effector pose.

        Parameters
        ----------
        T_des : np.ndarray, shape (4,4)
            Desired end-effector pose in the base frame.

        theta_init : array-like, optional
            Initial guess in physical robot command degrees.
            If None, the current robot configuration is used.

        max_iters : int
            Maximum IK iterations.

        tol_w : float
            Orientation convergence tolerance.

        tol_v : float
            Position convergence tolerance.

        K : np.ndarray, optional
            6x6 gain matrix.

        Returns
        -------
        theta_sol_robot_deg : np.ndarray
            Solution in robot command degrees.

        theta_history_robot_deg : np.ndarray
            Robot command angle history.

        norm_w_hist : np.ndarray
            Orientation error history.

        norm_v_hist : np.ndarray
            Position error history.
        """

        T_des = np.asarray(T_des, dtype=float).reshape(4, 4)

        # Initial guess
        if theta_init is None:
            theta_robot_deg = self.get_joint_angles_deg()
        else:
            theta_robot_deg = np.asarray(theta_init, dtype=float).flatten()

        if len(theta_robot_deg) != 6:
            raise ValueError("theta_init must contain 6 joint values")

        # Robot command -> IK model
        theta_ik_deg = self.robot_deg_to_ik_deg(theta_robot_deg)
        theta_ik_rad = np.radians(theta_ik_deg)

        # Solve pose IK
        (
            theta_sol_ik_rad,
            theta_history_ik_rad,
            norm_w_hist,
            norm_v_hist,
        ) = jacobian_transpose_pose(
            M_ee=self.M,
            B_list=self.B_list,
            theta_init=theta_ik_rad,
            T_sd=T_des,
            max_iters=max_iters,
            tol_w=tol_w,
            tol_v=tol_v,
            q_min=self.theta_min,
            q_max=self.theta_max,
            K=K,
        )

        # Final solution back to robot command angles
        theta_sol_ik_deg = np.degrees(theta_sol_ik_rad)
        theta_sol_robot_deg = self.ik_deg_to_robot_deg(theta_sol_ik_deg)

        # Convert history back to robot command angles
        theta_history_robot_deg = []

        for theta in theta_history_ik_rad:
            theta_deg = np.degrees(theta)
            theta_robot = self.ik_deg_to_robot_deg(theta_deg)
            theta_history_robot_deg.append(theta_robot)

        theta_history_robot_deg = np.asarray(theta_history_robot_deg)

        return (
            theta_sol_robot_deg,
            theta_history_robot_deg,
            norm_w_hist,
            norm_v_hist,
        )

    def connect(self, calibrate=False):
        self.robot.connect(calibrate=calibrate)


    def disconnect(self):
        self.robot.disconnect()