from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "eye_to_hand" / "eye_to_hand_samples.npz"
OUT_DIR = ROOT / "data" / "eye_to_hand"


# Rotation residual weight.
# 0.03 means 1 rad rotation error is treated like about 30 mm translation error.
ROT_WEIGHT = 0.03

# Robust loss settings.
# Helps prevent a few bad samples from dominating the solution.
ROBUST_LOSS = "huber"
ROBUST_F_SCALE = 0.01


def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T_inv


def params_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return make_T(R, tvec)


def T_to_params(T):
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    tvec = T[:3, 3]
    return rvec.reshape(3), tvec.reshape(3)


def rotation_error_deg(R):
    value = (np.trace(R) - 1.0) / 2.0
    value = np.clip(value, -1.0, 1.0)
    return np.degrees(np.arccos(value))


def se3_error(T_err):
    """
    Convert transform error into optimizer residual.

    Translation is in meters.
    Rotation is in radians, scaled by ROT_WEIGHT.
    """

    rvec, _ = cv2.Rodrigues(T_err[:3, :3])
    tvec = T_err[:3, 3]

    return np.hstack([
        tvec,
        ROT_WEIGHT * rvec.reshape(3),
    ])


def residual(x, T_base_to_ee_list, T_camera_to_board_list):
    r_base_cam = x[0:3]
    t_base_cam = x[3:6]

    r_ee_board = x[6:9]
    t_ee_board = x[9:12]

    T_base_to_camera = params_to_T(r_base_cam, t_base_cam)
    T_ee_to_board = params_to_T(r_ee_board, t_ee_board)

    errors = []

    for T_base_to_ee, T_camera_to_board in zip(
        T_base_to_ee_list,
        T_camera_to_board_list,
    ):
        # Robot path:
        # base -> ee -> board
        lhs = T_base_to_ee @ T_ee_to_board

        # Camera path:
        # base -> camera -> board
        rhs = T_base_to_camera @ T_camera_to_board

        # Difference between both paths.
        T_err = invert_T(lhs) @ rhs

        errors.extend(se3_error(T_err))

    return np.array(errors, dtype=np.float64)


def validate(
    T_base_to_camera,
    T_ee_to_board,
    T_base_to_ee_list,
    T_camera_to_board_list,
    label="Validation error",
):
    pos_errors = []
    rot_errors = []

    for T_base_to_ee, T_camera_to_board in zip(
        T_base_to_ee_list,
        T_camera_to_board_list,
    ):
        lhs = T_base_to_ee @ T_ee_to_board
        rhs = T_base_to_camera @ T_camera_to_board

        pos_error = np.linalg.norm(lhs[:3, 3] - rhs[:3, 3])
        pos_errors.append(pos_error)

        dR = lhs[:3, :3].T @ rhs[:3, :3]
        rot_error = rotation_error_deg(dR)
        rot_errors.append(rot_error)

    pos_errors = np.asarray(pos_errors)
    rot_errors = np.asarray(rot_errors)

    print(f"\n{label}:")
    print(f"Position mean: {np.mean(pos_errors) * 1000:.2f} mm")
    print(f"Position median: {np.median(pos_errors) * 1000:.2f} mm")
    print(f"Position max : {np.max(pos_errors) * 1000:.2f} mm")
    print(f"Rotation mean: {np.mean(rot_errors):.2f} deg")
    print(f"Rotation median: {np.median(rot_errors):.2f} deg")
    print(f"Rotation max : {np.max(rot_errors):.2f} deg")

    return pos_errors, rot_errors


def solve_once(T_base_to_ee_list, T_camera_to_board_list):
    """
    Solve:
        T_base_to_ee * T_ee_to_board
        =
        T_base_to_camera * T_camera_to_board
    """

    # Initial guess for base -> camera.
    # Change this rough translation if your camera is obviously elsewhere.
    T_base_to_camera_init = np.eye(4, dtype=np.float64)
    T_base_to_camera_init[:3, 3] = np.array([0.1, 0.0, 0.3])

    # Since the board is mounted directly on the end effector,
    # this should be a small transform.
    T_ee_to_board_init = np.eye(4, dtype=np.float64)
    T_ee_to_board_init[:3, 3] = np.array([0.003, 0.0, 0.0])

    r_bc, t_bc = T_to_params(T_base_to_camera_init)
    r_eb, t_eb = T_to_params(T_ee_to_board_init)

    x0 = np.hstack([r_bc, t_bc, r_eb, t_eb])

    result = least_squares(
        residual,
        x0,
        args=(T_base_to_ee_list, T_camera_to_board_list),
        loss=ROBUST_LOSS,
        f_scale=ROBUST_F_SCALE,
        verbose=1,
        max_nfev=5000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )

    x = result.x

    T_base_to_camera = params_to_T(x[0:3], x[3:6])
    T_ee_to_board = params_to_T(x[6:9], x[9:12])

    return T_base_to_camera, T_ee_to_board, result


def main():
    data = np.load(DATA_PATH)

    T_base_to_ee_list = data["T_base_to_ee"]
    T_camera_to_board_list = data["T_camera_to_board"]

    n = len(T_base_to_ee_list)

    print(f"[INFO] Loaded {n} samples")

    if n < 8:
        raise RuntimeError("Need at least 8 samples. Prefer 25–40 good samples.")

    print(f"[INFO] ROT_WEIGHT = {ROT_WEIGHT}")
    print(f"[INFO] Robust loss = {ROBUST_LOSS}, f_scale = {ROBUST_F_SCALE}")

    T_base_to_camera, T_ee_to_board, result = solve_once(
        T_base_to_ee_list,
        T_camera_to_board_list,
    )

    print("\nT_base_to_camera:")
    print(T_base_to_camera)

    print("\nT_ee_to_board:")
    print(T_ee_to_board)

    pos_errors, rot_errors = validate(
        T_base_to_camera,
        T_ee_to_board,
        T_base_to_ee_list,
        T_camera_to_board_list,
    )

    t_ee_board = T_ee_to_board[:3, 3]
    print("\nT_ee_to_board translation:")
    print(f"x = {t_ee_board[0] * 1000:.2f} mm")
    print(f"y = {t_ee_board[1] * 1000:.2f} mm")
    print(f"z = {t_ee_board[2] * 1000:.2f} mm")
    print(f"norm = {np.linalg.norm(t_ee_board) * 1000:.2f} mm")

    np.save(OUT_DIR / "T_base_to_camera.npy", T_base_to_camera)
    np.save(OUT_DIR / "T_ee_to_board.npy", T_ee_to_board)

    np.savez(
        OUT_DIR / "eye_to_hand_solution_debug.npz",
        T_base_to_camera=T_base_to_camera,
        T_ee_to_board=T_ee_to_board,
        pos_errors=pos_errors,
        rot_errors=rot_errors,
        optimizer_x=result.x,
        optimizer_cost=result.cost,
    )

    print("\n[SAVED]")
    print(OUT_DIR / "T_base_to_camera.npy")
    print(OUT_DIR / "T_ee_to_board.npy")
    print(OUT_DIR / "eye_to_hand_solution_debug.npz")


if __name__ == "__main__":
    main()