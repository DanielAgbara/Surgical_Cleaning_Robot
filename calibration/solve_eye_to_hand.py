from pathlib import Path
import numpy as np
import cv2
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "eye_to_hand" / "eye_to_hand_samples.npz"
OUT_DIR = ROOT / "data" / "eye_to_hand"


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def params_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec).reshape(3, 1))
    return make_T(R, tvec)


def T_to_params(T):
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    tvec = T[:3, 3]
    return rvec.reshape(3), tvec.reshape(3)


def se3_error(T_err):
    rvec, _ = cv2.Rodrigues(T_err[:3, :3])
    tvec = T_err[:3, 3]

    # Rotation in radians, translation in meters
    return np.hstack([
        tvec,
        0.05 * rvec.reshape(3)
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
        T_camera_to_board_list
    ):
        lhs = T_base_to_ee @ T_ee_to_board
        rhs = T_base_to_camera @ T_camera_to_board

        T_err = invert_T(lhs) @ rhs
        errors.extend(se3_error(T_err))

    return np.array(errors)


def rotation_error_deg(R):
    value = (np.trace(R) - 1.0) / 2.0
    value = np.clip(value, -1.0, 1.0)
    return np.degrees(np.arccos(value))


def validate(T_base_to_camera, T_ee_to_board, T_base_to_ee_list, T_camera_to_board_list):
    pos_errors = []
    rot_errors = []

    for T_base_to_ee, T_camera_to_board in zip(
        T_base_to_ee_list,
        T_camera_to_board_list
    ):
        lhs = T_base_to_ee @ T_ee_to_board
        rhs = T_base_to_camera @ T_camera_to_board

        dp = lhs[:3, 3] - rhs[:3, 3]
        pos_errors.append(np.linalg.norm(dp))

        dR = lhs[:3, :3].T @ rhs[:3, :3]
        rot_errors.append(rotation_error_deg(dR))

    pos_errors = np.array(pos_errors)
    rot_errors = np.array(rot_errors)

    print("\nValidation error:")
    print(f"Position mean: {np.mean(pos_errors) * 1000:.2f} mm")
    print(f"Position max : {np.max(pos_errors) * 1000:.2f} mm")
    print(f"Rotation mean: {np.mean(rot_errors):.2f} deg")
    print(f"Rotation max : {np.max(rot_errors):.2f} deg")


def main():
    data = np.load(DATA_PATH)

    T_base_to_ee_list = data["T_base_to_ee"]
    T_camera_to_board_list = data["T_camera_to_board"]

    n = len(T_base_to_ee_list)

    print(f"[INFO] Loaded {n} samples")

    if n < 8:
        raise RuntimeError("Need at least 8 samples. Prefer 15–25.")

    # Initial guess
    T_base_to_camera_init = np.eye(4)
    T_base_to_camera_init[:3, 3] = np.array([0.1, 0.0, 0.3])

    T_ee_to_board_init = np.eye(4)
    T_ee_to_board_init[:3, 3] = np.array([0.05, 0.0, 0.0])

    r_bc, t_bc = T_to_params(T_base_to_camera_init)
    r_eb, t_eb = T_to_params(T_ee_to_board_init)

    x0 = np.hstack([r_bc, t_bc, r_eb, t_eb])

    result = least_squares(
        residual,
        x0,
        args=(T_base_to_ee_list, T_camera_to_board_list),
        verbose=1,
        max_nfev=5000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )

    x = result.x

    T_base_to_camera = params_to_T(x[0:3], x[3:6])
    T_ee_to_board = params_to_T(x[6:9], x[9:12])

    print("\nT_base_to_camera:")
    print(T_base_to_camera)

    print("\nT_ee_to_board:")
    print(T_ee_to_board)

    validate(
        T_base_to_camera,
        T_ee_to_board,
        T_base_to_ee_list,
        T_camera_to_board_list,
    )

    np.save(OUT_DIR / "T_base_to_camera.npy", T_base_to_camera)
    np.save(OUT_DIR / "T_ee_to_board.npy", T_ee_to_board)

    print("\n[SAVED]")
    print(OUT_DIR / "T_base_to_camera.npy")
    print(OUT_DIR / "T_ee_to_board.npy")


if __name__ == "__main__":
    main()