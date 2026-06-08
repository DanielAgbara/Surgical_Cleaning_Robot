"""
Serial Version of the Trajectory Generation Code for the SOArm101 Robot

This script generates a square trajectory in an adjustable plane and moves
the SOArm101 robot through the trajectory point-by-point using inverse kinematics.

IK is computed once, then the joint trajectory is replayed multiple times.
"""

import numpy as np
import robot as bot


def generate_square_trajectory(center, side_length, num_points_per_edge, normal, close_loop=True):
    """
    Generates a square trajectory in an arbitrary plane.

    Inputs:
        center: (x, y, z) center of the square
        side_length: length of each side of the square
        num_points_per_edge: resolution of each edge
        normal: normal vector defining the plane
        close_loop: whether to return to the starting point

    Returns:
        traj: list of 3D points forming the square path
    """

    center = np.asarray(center, dtype=float)
    normal = np.asarray(normal, dtype=float)

    # Normalize the normal vector (defines plane orientation)
    normal = normal / np.linalg.norm(normal)

    # Choose a vector not parallel to normal (for basis construction)
    if abs(np.dot(normal, np.array([0.0, 0.0, 1.0]))) < 0.99:
        arbitrary = np.array([0.0, 0.0, 1.0])
    else:
        arbitrary = np.array([1.0, 0.0, 0.0])

    # Build orthonormal basis (u, v) for the plane
    u = np.cross(normal, arbitrary)
    u = u / np.linalg.norm(u)

    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    # Half side length (for convenience)
    h = side_length / 2.0

    # Define square corners in plane coordinates
    corners = [
        center + (-h * u - h * v),
        center + ( h * u - h * v),
        center + ( h * u + h * v),
        center + (-h * u + h * v),
    ]

    traj = []

    # Interpolate along each edge
    for i in range(4):
        p_start = corners[i]
        p_end = corners[(i + 1) % 4]

        for t in range(num_points_per_edge):
            alpha = t / num_points_per_edge
            point = (1 - alpha) * p_start + alpha * p_end
            traj.append(point)

    # Optionally close the loop
    if close_loop:
        traj.append(traj[0].copy())

    return traj


def precompute_joint_path(arm, traj, gripper_pos=0.0):
    """
    Solve IK once for all Cartesian points and store joint commands.
    """
    joint_path = []
    theta_init = np.zeros(6)

    for i, p_des in enumerate(traj):
        print(f"Precomputing IK point {i+1}/{len(traj)}: {p_des}")

        theta_solution_deg = arm.inverse_kinematics(p_des, theta_init)

        action = {
            "shoulder_pan.pos": float(theta_solution_deg[0]),
            "shoulder_lift.pos": float(theta_solution_deg[1]),
            "elbow_flex.pos": float(theta_solution_deg[2]),
            "wrist_flex.pos": float(theta_solution_deg[3]),
            "wrist_roll.pos": float(theta_solution_deg[4]),
            "gripper.pos": float(gripper_pos),
        }

        joint_path.append(action)

        # Use solution as next IK seed (in radians)
        theta_init = np.radians(theta_solution_deg)

    return joint_path


def replay_joint_path(arm, joint_path, loops=1, max_step_deg=1.5, step_delay=0.04):
    """
    Replay a precomputed joint trajectory multiple times.
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
    # Trajectory parameters
    # ---------------------------
    center = np.array([0.3, 0.00, 0.2])   # square center
    side_length = 0.1                      # square size
    num_points_per_edge = 30                # resolution per side

    # Plane selection
    # XY plane -> [0, 0, 1]
    # YZ plane -> [1, 0, 0]
    # XZ plane -> [0, 1, 0]
    normal = np.array([1.0, 0.0, 0.0])      # XY plane

    # Motion tuning
    max_step_deg = 1.5
    step_delay = 0.04
    gripper_pos = 0.0
    num_loops = 4

    # ---------------------------
    # Generate square trajectory
    # ---------------------------
    traj = generate_square_trajectory(
        center=center,
        side_length=side_length,
        num_points_per_edge=num_points_per_edge,
        normal=normal,
        close_loop=True,
    )

    try:
        arm.connect(calibrate=False)

        # Move to known start pose
        arm.move_to_home(max_step_deg=2.0, step_delay=0.05)

        # Compute IK once
        joint_path = precompute_joint_path(
            arm=arm,
            traj=traj,
            gripper_pos=gripper_pos,
        )

        # Replay trajectory multiple times
        replay_joint_path(
            arm=arm,
            joint_path=joint_path,
            loops=num_loops,
            max_step_deg=max_step_deg,
            step_delay=step_delay,
        )

        # Return to rest pose
        arm.move_to_rest(max_step_deg=2.0, step_delay=0.05)

    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()