import time
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

config = SO101FollowerConfig(port="/dev/ttyACM0", id="dbot")
robot = SO101Follower(config)

def move_and_wait(action, delay=1.5):
    print("Sending:", action)
    robot.send_action(action)
    time.sleep(delay)

robot.connect(calibrate=False)

try:
    # Neutral starting pose
    home = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 0.0,
        "elbow_flex.pos": 0.0,
        "wrist_flex.pos": 0.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 0.0,
    }

    move_and_wait(home, 2.0)

    # Raise arm a bit
    move_and_wait({
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 20.0,
        "elbow_flex.pos": -20.0,
        "wrist_flex.pos": 10.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 0.0,
    }, 2.0)

    # Wave left and right
    wave_poses = [
        {
            "shoulder_pan.pos": -20.0,
            "shoulder_lift.pos": 20.0,
            "elbow_flex.pos": -20.0,
            "wrist_flex.pos": 10.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 0.0,
        },
        {
            "shoulder_pan.pos": 20.0,
            "shoulder_lift.pos": 20.0,
            "elbow_flex.pos": -20.0,
            "wrist_flex.pos": 10.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 0.0,
        },
    ]

    for _ in range(2):
        for pose in wave_poses:
            move_and_wait(pose, 1.2)

    # Wrist twist sequence
    wrist_twist = [
        {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 20.0,
            "elbow_flex.pos": -20.0,
            "wrist_flex.pos": 10.0,
            "wrist_roll.pos": -30.0,
            "gripper.pos": 0.0,
        },
        {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 20.0,
            "elbow_flex.pos": -20.0,
            "wrist_flex.pos": 10.0,
            "wrist_roll.pos": 30.0,
            "gripper.pos": 0.0,
        },
    ]

    for _ in range(2):
        for pose in wrist_twist:
            move_and_wait(pose, 1.0)

    # Reach forward a bit
    move_and_wait({
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 30.0,
        "elbow_flex.pos": -35.0,
        "wrist_flex.pos": 15.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 0.0,
    }, 2.0)

    # Open / close gripper
    for grip in [100.0, 0.0, 100.0, 0.0]:
        move_and_wait({
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 30.0,
            "elbow_flex.pos": -35.0,
            "wrist_flex.pos": 15.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": grip,
        }, 0.8)

    # Small bow motion
    bow_poses = [
        {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 15.0,
            "elbow_flex.pos": -15.0,
            "wrist_flex.pos": 20.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 0.0,
        },
        {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 25.0,
            "elbow_flex.pos": -30.0,
            "wrist_flex.pos": -10.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 0.0,
        },
    ]

    for _ in range(2):
        for pose in bow_poses:
            move_and_wait(pose, 1.0)

    # Return home
    move_and_wait(home, 2.0)

    # Final dramatic pose: joint2 = -90, joint3 = +90
    final_pose = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": -105.0,   # joint 2
        "elbow_flex.pos": 95.0,       # joint 3
        "wrist_flex.pos": -90.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 0.0,
    }

    move_and_wait(final_pose, 3.0)

finally:
    robot.disconnect()