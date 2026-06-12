#!/usr/bin/env python3

import json
import time
import numpy as np
from pathlib import Path

from robot import SOArm101

import sys

ROOT = Path("/home/agbara-admin/Documents/Surgical_Cleaning_Robot")
ROBOT_IK_PATH = ROOT / "robot_control" / "IK"



# =====================================================
# Settings
# =====================================================

INPUT_JSON = ROOT / "data" / "arm_tracking" / "processed_velocity_output.json"

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "dbot"

CONNECT_CALIBRATE = False

MOVE_TO_HOME_FIRST = True

# Velocity playback scale.
# If processed velocity is already m/s, start small.
PLAYBACK_SCALE = 0.2

# Ignore tiny velocity blocks
VELOCITY_NORM_THRESHOLD = 1e-5

# Pause between blocks
BLOCK_PAUSE = 0.05

# IK settings
IK_MAX_ITERS = 150
IK_TOL = 1e-4
IK_GAIN = 0.05 * np.eye(3)

# Robot movement settings
MAX_STEP_DEG = 2.0
STEP_DELAY = 0.03

# Safety workspace clamp in meters
X_LIMITS = (0.15, 0.45)
Y_LIMITS = (-0.25, 0.25)
Z_LIMITS = (0.03, 0.40)


# =====================================================
# Helper Functions
# =====================================================

def load_velocity_blocks(json_path):
    """
    Load block velocity data from processed_velocity_output.json.
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    if "velocity_blocks" not in data:
        raise ValueError("JSON file does not contain 'velocity_blocks'")

    return data["velocity_blocks"], data.get("settings", {})


def get_velocity_from_block(block):
    """
    Extract [vx, vy, vz] velocity from block.
    """

    v = block["velocity"]

    return np.array(
        [v["vx"], v["vy"], v["vz"]],
        dtype=float
    )


def get_duration_from_block(block):
    """
    Extract duration from block.
    """

    interval = block.get("time_interval", {})

    return float(interval.get("duration_s", 0.0))


def get_timestep_from_block(block):
    """
    Extract timestep from block.
    """

    interval = block.get("time_interval", {})

    timestep = float(interval.get("timestep_s", 0.05))

    if timestep <= 0.0:
        timestep = 0.05

    return timestep


def clamp_position(p):
    """
    Clamp desired EE position to safe workspace.
    """

    p = np.asarray(p, dtype=float).reshape(3)

    p[0] = np.clip(p[0], X_LIMITS[0], X_LIMITS[1])
    p[1] = np.clip(p[1], Y_LIMITS[0], Y_LIMITS[1])
    p[2] = np.clip(p[2], Z_LIMITS[0], Z_LIMITS[1])

    return p


def V_to_pos(previous_pos, velocity, duration, timestep):
    """
    Integrate velocity into position using Euler integration.

    Parameters
    ----------
    previous_pos : np.ndarray, shape (3,)
        Starting EE position [m].

    velocity : np.ndarray, shape (3,)
        Linear velocity [m/s].

    duration : float
        Duration to apply velocity [s].

    timestep : float
        Integration timestep [s].

    Returns
    -------
    final_pos : np.ndarray, shape (3,)
        Final desired position after integration.

    pos_history : np.ndarray
        Position history during the block.
    """

    previous_pos = np.asarray(previous_pos, dtype=float).reshape(3)
    velocity = np.asarray(velocity, dtype=float).reshape(3)

    if timestep <= 0.0:
        raise ValueError("timestep must be greater than zero")

    pos = previous_pos.copy()
    pos_history = [pos.copy()]

    elapsed = 0.0

    while elapsed < duration:
        dt = min(timestep, duration - elapsed)

        pos = pos + velocity * dt

        pos = clamp_position(pos)

        pos_history.append(pos.copy())

        elapsed += dt

    return pos, np.asarray(pos_history)


# =====================================================
# Main
# =====================================================

def main():
    """
    Replay velocity blocks using position IK.

    New pipeline:

        velocity block
        ↓
        V_to_pos()
        ↓
        desired EE position
        ↓
        Jacobian transpose IK
        ↓
        moveSO101()
    """

    velocity_blocks, settings = load_velocity_blocks(INPUT_JSON)

    print(f"Loaded velocity file: {INPUT_JSON}")
    print(f"Number of velocity blocks: {len(velocity_blocks)}")
    print(f"Velocity units: {settings.get('scaled_velocity_units', 'unknown')}")

    robot = SOArm101(
        port=ROBOT_PORT,
        id=ROBOT_ID
    )

    try:
        print("Connecting robot...")
        robot.connect(calibrate=CONNECT_CALIBRATE)

        if MOVE_TO_HOME_FIRST:
            print("Moving robot to home...")
            robot.move_to_home(
                max_step_deg=MAX_STEP_DEG,
                step_delay=STEP_DELAY
            )

        # Current EE position becomes the starting point
        T_current = robot.get_T_base_to_ee()
        previous_pos = T_current[:3, 3].copy()

        print(f"Starting EE position: {previous_pos.tolist()}")

        for block in velocity_blocks:

            block_id = block["block_id"]
            intent = block.get("intent", "unknown")

            velocity = get_velocity_from_block(block)
            velocity = velocity * PLAYBACK_SCALE

            duration = get_duration_from_block(block)
            timestep = get_timestep_from_block(block)

            print(
                f"\nBlock {block_id}"
                f"\nIntent: {intent}"
                f"\nVelocity m/s: {velocity.tolist()}"
                f"\nDuration: {duration:.3f} s"
                f"\nTimestep: {timestep:.3f} s"
            )

            if duration <= 0.0:
                print("Skipping block: invalid duration")
                continue

            if np.linalg.norm(velocity) < VELOCITY_NORM_THRESHOLD:
                print("Skipping block: velocity too small")
                time.sleep(duration)
                continue

            # Integrate velocity to get next desired EE position
            desired_pos, pos_history = V_to_pos(
                previous_pos=previous_pos,
                velocity=velocity,
                duration=duration,
                timestep=timestep
            )

            print(f"Desired EE position: {desired_pos.tolist()}")

            # Solve IK and move robot
            theta_sol, theta_history = robot.move_to_position_jacobian_transpose(
                p_des=desired_pos,
                max_iters=IK_MAX_ITERS,
                tol_converge=IK_TOL,
                K=IK_GAIN,
                print_iterations=False,
                max_step_deg=MAX_STEP_DEG,
                step_delay=STEP_DELAY
            )

            print(f"IK solution deg: {np.degrees(theta_sol).tolist()}")

            # Use desired position as next reference point
            previous_pos = desired_pos.copy()

            time.sleep(BLOCK_PAUSE)

        print("\nPlayback complete.")

    finally:
        print("Moving robot to rest...")
        robot.move_to_rest(
            max_step_deg=MAX_STEP_DEG,
            step_delay=STEP_DELAY
        )

        print("Disconnecting robot...")
        robot.disconnect()

        print("Done.")


if __name__ == "__main__":
    main()