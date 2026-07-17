#!/usr/bin/env python3

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "dbot"

robot = SO101Follower(
    SO101FollowerConfig(
        port=ROBOT_PORT,
        id=ROBOT_ID,
    )
)

try:
    robot.connect(calibrate=False)

    obs = robot.get_observation()

    print("\nObservation type:")
    print(type(obs))

    print("\nObservation:")
    print(obs)

    print("\nObservation keys:")
    if hasattr(obs, "keys"):
        for key in obs.keys():
            print(" ", key)

finally:
    robot.disconnect()