#!/usr/bin/env python3

import time
import curses
import numpy as np

import robot as bot
from jacobian import body_jacobian, space_jacobian, damped_least_square_inverse
from se3 import adjoint_transform_list


# -----------------------------
# Robot settings
# -----------------------------
ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "dbot"

# -----------------------------
# Velocity settings
# -----------------------------
LINEAR_SPEED_M_S = 0.05          # constant EE speed, 1 cm/s
WRIST_SPEED_DEG_S = 30.0          # constant wrist roll speed
CONTROL_DT = 0.03                 # control loop timestep

MAX_JOINT_SPEED_RAD_S = np.radians(180.0)
DAMPING = 0.02

USE_BODY_FRAME = False            # False = space/base frame control

# -----------------------------
# Motion smoothing
# -----------------------------
MAX_STEP_DEG = 1.0
STEP_DELAY = 0.01

# -----------------------------
# Workspace clamp
# -----------------------------
X_LIMITS = (0.18, 0.40)
Y_LIMITS = (-0.25, 0.25)
Z_LIMITS = (0.03, 0.35)

JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def clamp(value, low, high):
    return max(low, min(high, value))


def clamp_joint_angles(theta, arm):
    return np.clip(theta, arm.theta_min, arm.theta_max)


def clamp_joint_velocity(theta_dot):
    return np.clip(
        theta_dot,
        -MAX_JOINT_SPEED_RAD_S,
        MAX_JOINT_SPEED_RAD_S
    )


def action_to_theta_rad(action):
    theta_deg = np.array(
        [float(action[name]) for name in JOINT_NAMES],
        dtype=float
    )
    return np.radians(theta_deg)


def theta_to_action(theta_rad):
    theta_deg = np.degrees(theta_rad)

    return {
        "shoulder_pan.pos": float(theta_deg[0]),
        "shoulder_lift.pos": float(theta_deg[1]),
        "elbow_flex.pos": float(theta_deg[2]),
        "wrist_flex.pos": float(theta_deg[3]),
        "wrist_roll.pos": float(theta_deg[4]),
        "gripper.pos": 0.0,
    }


def fk_position(arm, theta_rad):
    from fk import space_product_of_exponentials

    T = space_product_of_exponentials(
        arm.M,
        arm.S_list,
        theta_rad
    )
    return T[:3, 3]


def make_keyboard_twist(key):
    """
    Returns Cartesian twist:
        [wx, wy, wz, vx, vy, vz]

    Linear velocity is constant while key is pressed.
    """

    V = np.zeros(6, dtype=float)

    # W/S: x-axis
    if key in [ord("w"), ord("W")]:
        V[3] = LINEAR_SPEED_M_S       # +X forward

    elif key in [ord("s"), ord("S")]:
        V[3] = -LINEAR_SPEED_M_S      # -X backward

    # A/D switched:
    # A = left = -Y
    # D = right = +Y
    elif key in [ord("a"), ord("A")]:
        V[4] = -LINEAR_SPEED_M_S      # left

    elif key in [ord("d"), ord("D")]:
        V[4] = LINEAR_SPEED_M_S       # right

    # Up/down arrows: z-axis
    elif key == curses.KEY_UP:
        V[5] = LINEAR_SPEED_M_S       # +Z up

    elif key == curses.KEY_DOWN:
        V[5] = -LINEAR_SPEED_M_S      # -Z down

    return V


def apply_velocity_control(arm, theta, V, B_list):
    if USE_BODY_FRAME:
        J = body_jacobian(
            B_list=B_list,
            theta=theta
        )
    else:
        J = space_jacobian(
            arm.S_list,
            theta
        )

    J_inv = damped_least_square_inverse(
        J,
        k=DAMPING
    )

    theta_dot = J_inv @ V
    theta_dot = clamp_joint_velocity(theta_dot)

    theta_next = theta + theta_dot * CONTROL_DT
    theta_next = clamp_joint_angles(theta_next, arm)

    p_next = fk_position(arm, theta_next)

    # Workspace safety clamp
    if not (
        X_LIMITS[0] <= p_next[0] <= X_LIMITS[1]
        and Y_LIMITS[0] <= p_next[1] <= Y_LIMITS[1]
        and Z_LIMITS[0] <= p_next[2] <= Z_LIMITS[1]
    ):
        return theta, theta_dot, p_next, False

    action = theta_to_action(theta_next)

    arm.moveSO101(
        action,
        max_step_deg=MAX_STEP_DEG,
        step_delay=STEP_DELAY
    )

    return theta_next, theta_dot, p_next, True


def apply_wrist_velocity(arm, theta, direction):
    theta_next = theta.copy()

    wrist_speed_rad_s = np.radians(WRIST_SPEED_DEG_S)
    theta_next[4] += direction * wrist_speed_rad_s * CONTROL_DT

    theta_next = clamp_joint_angles(theta_next, arm)

    action = theta_to_action(theta_next)

    arm.moveSO101(
        action,
        max_step_deg=MAX_STEP_DEG,
        step_delay=STEP_DELAY
    )

    return theta_next


def draw_screen(stdscr, arm, theta, last_msg):
    stdscr.clear()

    p = fk_position(arm, theta)
    theta_deg = np.degrees(theta)

    stdscr.addstr(0, 0, "SO-Arm101 End-Effector Velocity Keyboard Control")
    stdscr.addstr(1, 0, "------------------------------------------------")

    stdscr.addstr(3, 0, "Controls:")
    stdscr.addstr(4, 2, "W : forward   +X")
    stdscr.addstr(5, 2, "S : backward  -X")
    stdscr.addstr(6, 2, "A : left      -Y")
    stdscr.addstr(7, 2, "D : right     +Y")
    stdscr.addstr(8, 2, "UP arrow      : up      +Z")
    stdscr.addstr(9, 2, "DOWN arrow    : down    -Z")
    stdscr.addstr(10, 2, "LEFT arrow    : wrist roll left")
    stdscr.addstr(11, 2, "RIGHT arrow   : wrist roll right")
    stdscr.addstr(12, 2, "h             : home")
    stdscr.addstr(13, 2, "r             : rest")
    stdscr.addstr(14, 2, "q             : quit")

    stdscr.addstr(
        16,
        0,
        f"EE position: x={p[0]:.4f}, y={p[1]:.4f}, z={p[2]:.4f} m"
    )

    stdscr.addstr(18, 0, "Joint angles:")
    for i, name in enumerate(JOINT_NAMES):
        stdscr.addstr(
            19 + i,
            2,
            f"{name:18s}: {theta_deg[i]:8.2f} deg"
        )

    frame = "BODY" if USE_BODY_FRAME else "SPACE / BASE"
    stdscr.addstr(27, 0, f"Control frame: {frame}")
    stdscr.addstr(28, 0, f"Linear speed : {LINEAR_SPEED_M_S:.3f} m/s")
    stdscr.addstr(29, 0, f"Status       : {last_msg}")

    stdscr.refresh()


def keyboard_control(stdscr, arm):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    B_list = adjoint_transform_list(
        T=arm.M,
        X_list=arm.S_list,
        to_space=False
    )

    theta = action_to_theta_rad(arm.current_action)
    last_msg = "Ready."

    while True:
        draw_screen(stdscr, arm, theta, last_msg)

        key = stdscr.getch()

        if key == -1:
            time.sleep(CONTROL_DT)
            continue

        if key == ord("q"):
            break

        elif key == ord("h"):
            arm.move_to_home(max_step_deg=2.0, step_delay=0.05)
            theta = action_to_theta_rad(arm.current_action)
            last_msg = "Moved home."
            continue

        elif key == ord("r"):
            arm.move_to_rest(max_step_deg=2.0, step_delay=0.05)
            theta = action_to_theta_rad(arm.current_action)
            last_msg = "Moved rest."
            continue

        elif key == curses.KEY_LEFT:
            theta = apply_wrist_velocity(
                arm=arm,
                theta=theta,
                direction=-1.0
            )
            last_msg = "Wrist rolling left."
            time.sleep(CONTROL_DT)
            continue

        elif key == curses.KEY_RIGHT:
            theta = apply_wrist_velocity(
                arm=arm,
                theta=theta,
                direction=1.0
            )
            last_msg = "Wrist rolling right."
            time.sleep(CONTROL_DT)
            continue

        V = make_keyboard_twist(key)

        if np.linalg.norm(V) < 1e-9:
            continue

        theta, theta_dot, p_next, ok = apply_velocity_control(
            arm=arm,
            theta=theta,
            V=V,
            B_list=B_list
        )

        if ok:
            last_msg = (
                f"Moving with V={np.round(V, 3)} m/s, "
                f"theta_dot={np.round(np.degrees(theta_dot), 2)} deg/s"
            )
        else:
            last_msg = (
                f"Motion blocked by workspace clamp. "
                f"p_next={np.round(p_next, 3)}"
            )

        time.sleep(CONTROL_DT)


def main():
    arm = bot.SOArm101(
        port=ROBOT_PORT,
        id=ROBOT_ID
    )

    try:
        arm.connect(calibrate=False)

        arm.move_to_home(
            max_step_deg=2.0,
            step_delay=0.05
        )

        curses.wrapper(keyboard_control, arm)

        arm.move_to_rest(
            max_step_deg=2.0,
            step_delay=0.05
        )

    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()