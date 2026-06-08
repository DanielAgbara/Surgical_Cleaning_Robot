"""
Serial Version of the Trajectory Generation Code for the SOArm101 Robot

This script generates a straight-line trajectory in 3D and moves
the SOArm101 robot through the trajectory point-by-point using inverse kinematics.

IK is computed once, then the joint trajectory is replayed multiple times.
"""

import numpy as np
import robot as bot


def generate_line_trajectory(start, end, num_points, back_and_forth=False):
    """
    Generates a straight-line trajectory in 3D.

    Inputs:
        start: (x, y, z) starting point
        end: (x, y, z) ending point
        num_points: number of points along the line
        back_and_forth: if True, trajectory goes start->end->start

    Returns:
        traj: list of 3D numpy arrays
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    if start.shape != (3,) or end.shape != (3,):
        raise ValueError("start and end must be 3D vectors")

    if num_points < 2:
        raise ValueError("num_points must be at least 2")

    traj = []

    # Forward path: start -> end
    for i in range(num_points):
        alpha = i / (num_points - 1)
        point = (1.0 - alpha) * start + alpha * end
        traj.append(point)

    # Optional backward path: end -> start
    if back_and_forth:
        for i in range(num_points - 2, -1, -1):
            traj.append(traj[i].copy())

    return traj


def precompute_joint_path(arm, traj, gripper_pos=0.0):
    """
    Solve IK once for the entire Cartesian trajectory and store the resulting actions.
    """
    joint_path = []

    # Initial guess for IK in radians
    theta_init = np.zeros(6, dtype=float)

    for i, p_des in enumerate(traj):
        print(f"Precomputing IK point {i+1}/{len(traj)}: {p_des}")

        theta_solution_deg = arm.inverse_kinematics(p_des, theta_init)
        print("IK solution (deg):", theta_solution_deg)

        action = {
            "shoulder_pan.pos": float(theta_solution_deg[0]),
            "shoulder_lift.pos": float(theta_solution_deg[1]),
            "elbow_flex.pos": float(theta_solution_deg[2]),
            "wrist_flex.pos": float(theta_solution_deg[3]),
            "wrist_roll.pos": float(theta_solution_deg[4]),
            "gripper.pos": float(gripper_pos),
        }

        joint_path.append(action)

        # Use the current solution as the next IK seed, in radians
        theta_init = np.radians(theta_solution_deg)

    return joint_path


def replay_joint_path(arm, joint_path, loops=1, max_step_deg=1.5, step_delay=0.04):
    """
    Replay a precomputed joint path multiple times without rerunning IK.
    """
    for loop_idx in range(loops):
        print(f"\nStarting loop {loop_idx + 1}/{loops}")

        for point_idx, action in enumerate(joint_path):
            print(f"Replaying point {point_idx + 1}/{len(joint_path)}")

            arm.moveSO101(
                action,
                max_step_deg=max_step_deg,
                step_delay=step_delay,
            )


def main():
    arm = bot.SOArm101(port="/dev/ttyACM0", id="dbot")

    # ---------------------------
    # Line trajectory parameters
    # ---------------------------
    start = np.array([0.4, 0.0, 0.05])   # meters
    end = np.array([0.27, 0.0, 0.05])      # meters
    num_points = 30
    back_and_forth = True

    # ---------------------------
    # Motion tuning
    # ---------------------------
    max_step_deg = 1.5
    step_delay = 0.04
    gripper_pos = 0.0
    num_loops = 6

    # Generate Cartesian line trajectory
    traj = generate_line_trajectory(
        start=start,
        end=end,
        num_points=num_points,
        back_and_forth=back_and_forth,
    )

    try:
        arm.connect(calibrate=False)

        # Move to a known start pose
        arm.move_to_home(max_step_deg=2.0, step_delay=0.05)

        # Solve IK once for the whole line
        joint_path = precompute_joint_path(
            arm=arm,
            traj=traj,
            gripper_pos=gripper_pos,
        )

        # Replay the joint path multiple times
        replay_joint_path(
            arm=arm,
            joint_path=joint_path,
            loops=num_loops,
            max_step_deg=max_step_deg,
            step_delay=step_delay,
        )

        # Return robot to rest pose
        arm.move_to_rest(max_step_deg=2.0, step_delay=0.05)

    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()