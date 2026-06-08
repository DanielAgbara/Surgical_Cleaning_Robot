'''
Script for testing the inverse kinematics solver on SO-Arm 101. 
This script will compute the joint angles required to reach a desired end-effector position and print the results.

'''

import robot as bot
import numpy as np

import time

if __name__ == "__main__":
    # Create an instance of the SOArm101 class
    arm = bot.SOArm101(port="/dev/ttyACM0", id="dbot")

    # Connect to the robot
    arm.connect(calibrate=False)
    arm.move_to_home()

    # Desired end-effector position (x, y, z) in meters
    p_des = np.array([0.038, 0.391, 0.243])  # Example desired position

    # Initial guess for joint angles (in degrees)
    theta_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    time.sleep(2)
    # Compute inverse kinematics to find joint angles that achieve the desired end-effector position
    theta_solution = arm.inverse_kinematics(p_des, theta_init)

    print("Desired end-effector position:", p_des)
    print("Computed joint angles (degrees):", theta_solution)

    arm.moveSO101({
        "shoulder_pan.pos": theta_solution[0],
        "shoulder_lift.pos": theta_solution[1],
        "elbow_flex.pos": theta_solution[2],
        "wrist_flex.pos": theta_solution[3],
        "wrist_roll.pos": theta_solution[4],
        "gripper.pos": theta_solution[5],
    }, max_step_deg=2.0, step_delay=0.05)

    time.sleep(2)


    arm.move_to_rest()

    # Disconnect from the robot
    arm.disconnect()

    print("Disconnected")