import time

import numpy as np
from typing import Callable
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

UTIL_PATH = ROOT / "Util"
IK_PATH = ROOT / "IK"

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

    def Twist_to_dtheta(self, Twist, type="Linear"):
        """
        Convert a desired twist/velocity command into joint angle increments.

        Parameters
        ----------
        Twist : array-like
            Desired velocity command.

            If type == "Linear":
                Twist should be [vx, vy, vz]

            If type == "Angular":
                Twist should be [wx, wy, wz]

            If type == "All":
                Twist should be [wx, wy, wz, vx, vy, vz]

        type : str
            "Linear", "Angular", or "All"

        Returns
        -------
        dtheta : np.ndarray, shape (6,)
            Joint velocity / joint increment direction in radians.
        """

        Twist = np.asarray(Twist, dtype=float).flatten()

        # Get current joint angles in degrees
        theta_deg = self.get_joint_angles_deg()

        # Keep gripper fixed
        theta_deg[5] = 0.0

        # Convert to radians for Jacobian calculation
        theta_rad = np.radians(theta_deg)

        # Use Jacobian because your velocity command is in robot/base frame
        J = body_jacobian(
            self.B_list,
            theta_rad
        )

        type = type.lower()

        if type == "linear":
            if Twist.shape[0] != 3:
                raise ValueError(
                    "For type='Linear', Twist must be [vx, vy, vz]"
                )

            # Linear velocity part of Jacobian
            J_use = J[3:6, :]

            # Desired linear velocity
            V_use = Twist

        elif type == "angular":
            if Twist.shape[0] != 3:
                raise ValueError(
                    "For type='Angular', Twist must be [wx, wy, wz]"
                )

            # Angular velocity part of Jacobian
            J_use = J[0:3, :]

            # Desired angular velocity
            V_use = Twist

        elif type == "all":
            if Twist.shape[0] != 6:
                raise ValueError(
                    "For type='All', Twist must be [wx, wy, wz, vx, vy, vz]"
                )

            # Full spatial Jacobian
            J_use = J

            # Full twist
            V_use = Twist

        else:
            raise ValueError(
                "type must be 'Linear', 'Angular', or 'All'"
            )

        # Use damped least squares for stability near singularities
        J_inv = damped_least_square_inverse(
            J_use,
            k=0.01
        )

        # Convert task-space velocity to joint-space velocity
        dtheta = J_inv @ V_use

        return dtheta
    

    def move_V(self, dtheta, duration, timestep=0.05):
        """
        Move robot using joint velocities.

        Parameters
        ----------
        dtheta : array-like (6,)
            Joint velocity vector [rad/s]

        duration : float
            Total motion duration [s]

        timestep : float
            Integration timestep [s]
        """

        dtheta = np.asarray(dtheta, dtype=float).flatten()

        if len(dtheta) != 6:
            raise ValueError(
                "dtheta must contain 6 joint velocities"
            )

        joint_names = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]

        # Number of integration steps
        n_steps = int(np.ceil(duration / timestep))

        for _ in range(n_steps):

            # Current robot state [deg]
            theta_deg = self.get_joint_angles_deg()

            # Convert to radians
            theta_rad = np.radians(theta_deg)

            # Euler integration
            theta_next_rad = (
                theta_rad
                + dtheta * timestep
            )

            # Joint limits
            theta_next_rad = np.clip(
                theta_next_rad,
                self.theta_min,
                self.theta_max
            )

            # Convert back to degrees
            theta_next_deg = np.degrees(theta_next_rad)

            action = {
                name: float(theta_next_deg[i])
                for i, name in enumerate(joint_names)
            }

            # Send directly to robot
            self.robot.send_action(action)

            # Update internal state
            self.current_action = action

            time.sleep(timestep)

    def V_to_pos(previous_pos, velocity, duration, timestep):
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
    tol_converge=1e-4
    ):
        """
        Solve IK for desired EE position.
        """

        if theta_init is None:

            theta_init = np.radians(
                self.get_joint_angles_deg()
            )

        theta_sol = jacobian_transpose_position(
            M=self.M,
            B_list=self.B_list,
            theta_init=theta_init,
            p_des=p_des,
            max_iters=max_iters,
            tol_converge=tol_converge,
            q_min=self.theta_min,
            q_max=self.theta_max
        )

        return theta_sol
    
    def theta_to_action(self, theta_rad):
        """
        Convert joint angles in radians to SO-101 action dictionary.

        Parameters
        ----------
        theta_rad : array-like, shape (6,)
            Joint angles in radians.

        Returns
        -------
        action : dict
            Robot action dictionary in degrees.
        """

        theta_rad = np.asarray(theta_rad, dtype=float).flatten()

        if len(theta_rad) != 6:
            raise ValueError("theta_rad must contain 6 joint values")

        # Apply joint limits for safety
        theta_rad = np.clip(
            theta_rad,
            self.theta_min,
            self.theta_max
        )

        theta_deg = np.degrees(theta_rad)

        return {
            "shoulder_pan.pos": float(theta_deg[0]),
            "shoulder_lift.pos": float(theta_deg[1]),
            "elbow_flex.pos": float(theta_deg[2]),
            "wrist_flex.pos": float(theta_deg[3]),
            "wrist_roll.pos": float(theta_deg[4]),

            # Keep gripper fixed
            "gripper.pos": float(self.current_action["gripper.pos"]),
        }


    def get_theta_rad(self):
        """
        Get current commanded joint angles in radians.
        """

        theta_deg = self.get_joint_angles_deg()

        return np.radians(theta_deg)


    def solve_position_jacobian_transpose(
        self,
        p_des,
        theta_init=None,
        max_iters=100,
        tol_converge=1e-4,
        K=None,
        print_iterations=False
    ):
        """
        Solve position-only IK using Jacobian transpose.

        Parameters
        ----------
        p_des : array-like, shape (3,)
            Desired end-effector position in meters.

        theta_init : array-like or None
            Initial joint guess in radians.
            If None, use current robot joint angles.

        Returns
        -------
        theta_sol : np.ndarray, shape (6,)
            Solved joint angles in radians.

        theta_history : np.ndarray
            IK joint history.
        """

        p_des = np.asarray(p_des, dtype=float).reshape(3)

        if theta_init is None:
            theta_init = self.get_theta_rad()

        if K is None:
            K = 0.1 * np.eye(3)

        theta_sol, theta_history = jacobian_transpose_position(
            M_ee=self.M,
            B_list=self.B_list,
            theta_init=theta_init,
            p_des=p_des,
            max_iters=max_iters,
            tol_converge=tol_converge,
            q_min=self.theta_min,
            q_max=self.theta_max,
            K=K,
            print_iterations=print_iterations
        )

        return theta_sol, theta_history


    def move_to_position_jacobian_transpose(
        self,
        p_des,
        max_iters=100,
        tol_converge=1e-4,
        K=None,
        print_iterations=False,
        max_step_deg=2.0,
        step_delay=0.05
    ):
        """
        Solve IK for desired Cartesian position and move robot.

        Parameters
        ----------
        p_des : array-like, shape (3,)
            Desired EE position in meters.
        """

        theta_sol, theta_history = self.solve_position_jacobian_transpose(
            p_des=p_des,
            max_iters=max_iters,
            tol_converge=tol_converge,
            K=K,
            print_iterations=print_iterations
        )

        action = self.theta_to_action(theta_sol)

        self.moveSO101(
            action,
            max_step_deg=max_step_deg,
            step_delay=step_delay
        )

        return theta_sol, theta_history

    

    def connect(self, calibrate=False):
        self.robot.connect(calibrate=calibrate)

    def disconnect(self):
        self.robot.disconnect()
    
    

        

