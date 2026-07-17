#!/usr/bin/env python3

import sys
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


# ============================================================
# Paths
# ============================================================

ROOT = Path("/home/agbara-admin/Documents/Surgical_Cleaning_Robot")
DATA_DIR = ROOT / "data" / "eye_to_hand"

UTIL_PATH = ROOT / "robot_control" / "Util"
sys.path.insert(0, str(UTIL_PATH))

from so3 import QuaternionToR, RToQuaternion


ROBOT_Q_FILE = DATA_DIR / "robot_q.json"
ROBOT_T_FILE = DATA_DIR / "robot_t.json"

CAMERA_Q_FILE = DATA_DIR / "camera_q.json"
CAMERA_T_FILE = DATA_DIR / "camera_t.json"

OUT_T_BASE_TO_CAMERA = DATA_DIR / "T_base_to_camera.npy"
OUT_T_EE_TO_BOARD = DATA_DIR / "T_ee_to_board.npy"


# ============================================================
# Quaternion helpers
# Quaternion convention: [w, x, y, z]
# ============================================================

def normalize_quaternion(q):
    q = np.asarray(q, dtype=float).reshape(4)
    norm = np.linalg.norm(q)

    if norm < 1e-12:
        raise ValueError("Quaternion norm is too small.")

    q = q / norm

    # Keep sign consistent.
    if q[0] < 0:
        q = -q

    return q


def quat_conjugate(q):
    q = normalize_quaternion(q)

    return np.array([
        q[0],
        -q[1],
        -q[2],
        -q[3],
    ], dtype=float)


def quat_multiply(q1, q2):
    """
    Hamilton product.

    q = q1 * q2
    """

    w1, x1, y1, z1 = normalize_quaternion(q1)
    w2, x2, y2, z2 = normalize_quaternion(q2)

    return normalize_quaternion(np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=float))


def quat_inverse(q):
    return quat_conjugate(q)


def rotation_error_vector(q_left, q_right):
    """
    Rotation residual between two orientations.

    q_err = inverse(q_left) * q_right

    For small rotations:
        vector part of q_err ≈ 0.5 * rotation_vector

    So we return 2 * vector part.
    """

    q_err = quat_multiply(
        quat_inverse(q_left),
        q_right,
    )

    if q_err[0] < 0:
        q_err = -q_err

    return 2.0 * q_err[1:4]


# ============================================================
# Transform helpers
# ============================================================

def make_T(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)

    return T


def transform_from_q_t(q, t):
    q = normalize_quaternion(q)
    R = QuaternionToR(q)

    return make_T(R, t)


def q_t_from_T(T):
    R = T[:3, :3]
    t = T[:3, 3]

    q = RToQuaternion(R)
    q = normalize_quaternion(q)

    return q, t


# ============================================================
# Loading data
# ============================================================

def load_json_array(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    return np.asarray(data, dtype=float)


def load_calibration_data():
    robot_q = load_json_array(ROBOT_Q_FILE)
    robot_t = load_json_array(ROBOT_T_FILE)

    camera_q = load_json_array(CAMERA_Q_FILE)
    camera_t = load_json_array(CAMERA_T_FILE)

    n = len(robot_q)

    if not (
        len(robot_t) == n
        and len(camera_q) == n
        and len(camera_t) == n
    ):
        raise RuntimeError(
            "JSON files have different sample counts."
        )

    if n < 8:
        raise RuntimeError(
            f"Only {n} samples found. Need at least 8. Prefer 25-40."
        )

    robot_q = np.array([
        normalize_quaternion(q)
        for q in robot_q
    ])

    camera_q = np.array([
        normalize_quaternion(q)
        for q in camera_q
    ])

    return robot_q, robot_t, camera_q, camera_t


# ============================================================
# Initial guess
# ============================================================

def make_initial_guess():
    """
    Unknowns:

        q_base_camera = x[0:4]
        t_base_camera = x[4:7]

        q_ee_board    = x[7:11]
        t_ee_board    = x[11:14]

    Initial values are rough guesses.
    """

    q_base_camera = np.array([1.0, 0.0, 0.0, 0.0])
    t_base_camera = np.array([-0.10, 0.60, 0.60])

    q_ee_board = np.array([1.0, 0.0, 0.0, 0.0])
    t_ee_board = np.array([0.01, 0.00, 0.0])

    return np.hstack([
        q_base_camera,
        t_base_camera,
        q_ee_board,
        t_ee_board,
    ])


def unpack_x(x):
    q_base_camera = normalize_quaternion(x[0:4])
    t_base_camera = np.asarray(x[4:7], dtype=float)

    q_ee_board = normalize_quaternion(x[7:11])
    t_ee_board = np.asarray(x[11:14], dtype=float)

    return (
        q_base_camera,
        t_base_camera,
        q_ee_board,
        t_ee_board,
    )


# ============================================================
# Residual
# ============================================================

def residual(
    x,
    robot_q,
    robot_t,
    camera_q,
    camera_t,
    rotation_weight=0.03,
):
    """
    Solve the absolute pose equation:

        T_base_to_ee(i) * T_ee_to_board
        =
        T_base_to_camera * T_camera_to_board(i)

    Rotation part:

        q_base_ee(i) * q_ee_board
        =
        q_base_camera * q_camera_board(i)

    Translation part:

        t_base_ee(i) + R_base_ee(i) * t_ee_board
        =
        t_base_camera + R_base_camera * t_camera_board(i)
    """

    (
        q_base_camera,
        t_base_camera,
        q_ee_board,
        t_ee_board,
    ) = unpack_x(x)

    R_base_camera = QuaternionToR(q_base_camera)

    errors = []

    for i in range(len(robot_q)):

        q_base_ee = normalize_quaternion(robot_q[i])
        t_base_ee = robot_t[i]

        q_camera_board = normalize_quaternion(camera_q[i])
        t_camera_board = camera_t[i]

        R_base_ee = QuaternionToR(q_base_ee)

        # ----------------------------------------------------
        # Left chain:
        # base -> ee -> board
        # ----------------------------------------------------

        q_left = quat_multiply(
            q_base_ee,
            q_ee_board,
        )

        t_left = (
            t_base_ee
            + R_base_ee @ t_ee_board
        )

        # ----------------------------------------------------
        # Right chain:
        # base -> camera -> board
        # ----------------------------------------------------

        q_right = quat_multiply(
            q_base_camera,
            q_camera_board,
        )

        t_right = (
            t_base_camera
            + R_base_camera @ t_camera_board
        )

        # ----------------------------------------------------
        # Residuals
        # ----------------------------------------------------

        trans_err = t_left - t_right
        rot_err = rotation_error_vector(
            q_left,
            q_right,
        )

        errors.extend(trans_err)
        errors.extend(rotation_weight * rot_err)

    return np.asarray(errors, dtype=float)


# ============================================================
# Solve
# ============================================================

def solve_eye_to_hand(
    robot_q,
    robot_t,
    camera_q,
    camera_t,
):
    x0 = make_initial_guess()

    result = least_squares(
        residual,
        x0,
        args=(
            robot_q,
            robot_t,
            camera_q,
            camera_t,
        ),
        loss="huber",
        f_scale=0.01,
        max_nfev=5000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        verbose=1,
    )

    (
        q_base_camera,
        t_base_camera,
        q_ee_board,
        t_ee_board,
    ) = unpack_x(result.x)

    T_base_to_camera = transform_from_q_t(
        q_base_camera,
        t_base_camera,
    )

    T_ee_to_board = transform_from_q_t(
        q_ee_board,
        t_ee_board,
    )

    return (
        T_base_to_camera,
        T_ee_to_board,
        q_base_camera,
        t_base_camera,
        q_ee_board,
        t_ee_board,
        result,
    )


# ============================================================
# Validation
# ============================================================

def rotation_angle_error_deg(q_left, q_right):
    q_err = quat_multiply(
        quat_inverse(q_left),
        q_right,
    )

    if q_err[0] < 0:
        q_err = -q_err

    q_err = normalize_quaternion(q_err)

    angle = 2.0 * np.arccos(
        np.clip(q_err[0], -1.0, 1.0)
    )

    return np.degrees(angle)


def validate_solution(
    q_base_camera,
    t_base_camera,
    q_ee_board,
    t_ee_board,
    robot_q,
    robot_t,
    camera_q,
    camera_t,
):
    R_base_camera = QuaternionToR(q_base_camera)

    pos_errors = []
    rot_errors = []

    for i in range(len(robot_q)):

        q_base_ee = normalize_quaternion(robot_q[i])
        t_base_ee = robot_t[i]

        q_camera_board = normalize_quaternion(camera_q[i])
        t_camera_board = camera_t[i]

        R_base_ee = QuaternionToR(q_base_ee)

        q_left = quat_multiply(
            q_base_ee,
            q_ee_board,
        )

        t_left = (
            t_base_ee
            + R_base_ee @ t_ee_board
        )

        q_right = quat_multiply(
            q_base_camera,
            q_camera_board,
        )

        t_right = (
            t_base_camera
            + R_base_camera @ t_camera_board
        )

        pos_errors.append(
            np.linalg.norm(t_left - t_right)
        )

        rot_errors.append(
            rotation_angle_error_deg(q_left, q_right)
        )

    pos_errors = np.asarray(pos_errors)
    rot_errors = np.asarray(rot_errors)

    print("\nValidation error:")
    print(f"Position mean   : {np.mean(pos_errors) * 1000:.2f} mm")
    print(f"Position median : {np.median(pos_errors) * 1000:.2f} mm")
    print(f"Position max    : {np.max(pos_errors) * 1000:.2f} mm")

    print(f"Rotation mean   : {np.mean(rot_errors):.2f} deg")
    print(f"Rotation median : {np.median(rot_errors):.2f} deg")
    print(f"Rotation max    : {np.max(rot_errors):.2f} deg")

    return pos_errors, rot_errors


# ============================================================
# Save debug output
# ============================================================

def save_solution_debug(
    q_base_camera,
    t_base_camera,
    q_ee_board,
    t_ee_board,
    pos_errors,
    rot_errors,
):
    debug = {
        "q_base_to_camera": [
            float(v)
            for v in q_base_camera
        ],
        "t_base_to_camera": [
            float(v)
            for v in t_base_camera
        ],
        "q_ee_to_board": [
            float(v)
            for v in q_ee_board
        ],
        "t_ee_to_board": [
            float(v)
            for v in t_ee_board
        ],
        "position_errors_m": [
            float(v)
            for v in pos_errors
        ],
        "rotation_errors_deg": [
            float(v)
            for v in rot_errors
        ],
    }

    out_path = DATA_DIR / "eye_to_hand_quaternion_solution_debug.json"

    with open(out_path, "w") as f:
        json.dump(debug, f, indent=4)

    print(f"\n[SAVED DEBUG JSON]")
    print(out_path)


# ============================================================
# Main
# ============================================================

def main():
    (
        robot_q,
        robot_t,
        camera_q,
        camera_t,
    ) = load_calibration_data()

    print(f"[INFO] Loaded {len(robot_q)} samples")

    (
        T_base_to_camera,
        T_ee_to_board,
        q_base_camera,
        t_base_camera,
        q_ee_board,
        t_ee_board,
        result,
    ) = solve_eye_to_hand(
        robot_q,
        robot_t,
        camera_q,
        camera_t,
    )

    print("\nT_base_to_camera:")
    print(T_base_to_camera)

    print("\nq_base_to_camera [w, x, y, z]:")
    print(q_base_camera)

    print("\nt_base_to_camera [m]:")
    print(t_base_camera)

    print("\nT_ee_to_board:")
    print(T_ee_to_board)

    print("\nq_ee_to_board [w, x, y, z]:")
    print(q_ee_board)

    print("\nt_ee_to_board [m]:")
    print(t_ee_board)

    (
        pos_errors,
        rot_errors,
    ) = validate_solution(
        q_base_camera,
        t_base_camera,
        q_ee_board,
        t_ee_board,
        robot_q,
        robot_t,
        camera_q,
        camera_t,
    )

    np.save(
        OUT_T_BASE_TO_CAMERA,
        T_base_to_camera,
    )

    np.save(
        OUT_T_EE_TO_BOARD,
        T_ee_to_board,
    )

    save_solution_debug(
        q_base_camera,
        t_base_camera,
        q_ee_board,
        t_ee_board,
        pos_errors,
        rot_errors,
    )

    print("\n[SAVED]")
    print(OUT_T_BASE_TO_CAMERA)
    print(OUT_T_EE_TO_BOARD)


if __name__ == "__main__":
    main()