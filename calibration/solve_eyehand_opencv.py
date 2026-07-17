#!/usr/bin/env python3

"""
Solve the SO-ARM101 eye-to-hand calibration using OpenCV.

Two OpenCV calibration functions are evaluated:

    1. cv2.calibrateHandEye(...)
    2. cv2.calibrateRobotWorldHandEye(...)

Physical setup
--------------
The ZED camera is fixed relative to the robot base, while the 4x4 ChArUco
board is rigidly attached to the robot end effector.

Frames
------
    B : robot base
    E : robot end effector
    C : ZED LEFT camera
    W : ChArUco board

Transform convention
--------------------
This script uses the convention:

    T_A_B = ^A T_B

which maps a point expressed in frame B into frame A:

    p_A = T_A_B @ p_B

The measured transforms for sample i are:

    T_base_ee[i]  = ^B T_E[i]
        Obtained from robot forward kinematics.

    T_ee_base[i]  = ^E T_B[i]
        Rigid inverse of T_base_ee[i].

    T_cam_board[i] = ^C T_W[i]
        Obtained directly from solvePnP for the ChArUco board.

The two constant transforms we want are:

    T_base_cam = ^B T_C
        Fixed camera pose in the robot-base frame.

    T_ee_board = ^E T_W
        Fixed ChArUco-board pose in the end-effector frame.

For every synchronized sample, the physical closed-loop equation is:

    ^B T_E[i] @ ^E T_W = ^B T_C @ ^C T_W[i]

or, using the variable names in this script:

    T_base_ee[i] @ T_ee_board
        =
    T_base_cam @ T_cam_board[i]

Input files
-----------
The data collector creates six plain JSON-list files in:

    Surgical_Cleaning_Robot/data/eye_to_hand/

Files:
    R_ee_base.json
    t_ee_base.json
    R_base_ee.json
    t_base_ee.json
    R_cam_board.json
    t_cam_board.json

Each rotation file must have shape [N, 3, 3].
Each translation file may have shape [N, 3] or [N, 3, 1].
All translations must use the same unit. This project uses meters.

Outputs
-------
Results are written into:

    data/eye_to_hand/opencv_results/

The script saves:
    - one JSON summary containing all successful methods and residuals
    - one T_base_camera_<method>.npy file per successful method
    - one T_ee_board_<method>.npy file per successful method
    - T_base_camera_best.npy
    - T_ee_board_best.npy

The "best" result is selected using a transparent geometric score that
combines translation residual with rotation-induced displacement at a
user-configurable reference radius.
"""

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ============================================================
# Project paths
# ============================================================

ROOT = Path("/home/agbara-admin/Documents/Surgical_Cleaning_Robot")
DATA_DIR = ROOT / "data" / "eye_to_hand"
RESULTS_DIR = DATA_DIR / "opencv_results"

R_EE_BASE_FILE = DATA_DIR / "R_ee_base.json"
T_EE_BASE_FILE = DATA_DIR / "t_ee_base.json"
R_BASE_EE_FILE = DATA_DIR / "R_base_ee.json"
T_BASE_EE_FILE = DATA_DIR / "t_base_ee.json"
R_CAM_BOARD_FILE = DATA_DIR / "R_cam_board.json"
T_CAM_BOARD_FILE = DATA_DIR / "t_cam_board.json"

SUMMARY_FILE = RESULTS_DIR / "opencv_eye_to_hand_results.json"
BEST_T_BASE_CAM_FILE = RESULTS_DIR / "T_base_camera_best.npy"
BEST_T_EE_BOARD_FILE = RESULTS_DIR / "T_ee_board_best.npy"


# ============================================================
# Solver configuration
# ============================================================

# OpenCV mathematically requires at least three distinct poses for hand-eye
# calibration. In practice, many more diverse poses should be used.
MINIMUM_NUMBER_OF_POSES = 3
RECOMMENDED_NUMBER_OF_POSES = 15

# When ranking methods, a rotation error is converted into an approximate
# displacement error using:
#
#     displacement ~= radius * angle_radians
#
# A 0.10 m radius is reasonable for comparing errors around a 200 mm board.
# This does not alter any calibration result; it only ranks the methods.
ROTATION_ERROR_EQUIVALENT_RADIUS_M = 0.10

# The collector writes both T_base_ee and its inverse T_ee_base. These
# tolerances verify that the six files are synchronized and internally valid.
INVERSE_CHECK_TRANSLATION_TOL_M = 1e-8
INVERSE_CHECK_ROTATION_TOL_DEG = 1e-5

# Small numerical drift can make a rotation matrix slightly non-orthogonal.
# Rotations are projected to the nearest valid SO(3) matrix by SVD. A large
# correction indicates corrupt or incorrectly shaped input data.
MAX_ROTATION_PROJECTION_CHANGE = 1e-2

# Set False to run only one method from each OpenCV function.
RUN_ALL_HAND_EYE_METHODS = True
RUN_ALL_ROBOT_WORLD_METHODS = True


# ============================================================
# OpenCV methods
# ============================================================

HAND_EYE_METHODS = {
    "handeye_tsai": cv2.CALIB_HAND_EYE_TSAI,
    "handeye_park": cv2.CALIB_HAND_EYE_PARK,
    "handeye_horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "handeye_andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "handeye_daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}

ROBOT_WORLD_METHODS = {
    "robotworld_shah": cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
    "robotworld_li": cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI,
}

if not RUN_ALL_HAND_EYE_METHODS:
    HAND_EYE_METHODS = {
        "handeye_tsai": cv2.CALIB_HAND_EYE_TSAI,
    }

if not RUN_ALL_ROBOT_WORLD_METHODS:
    ROBOT_WORLD_METHODS = {
        "robotworld_shah": cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
    }


# ============================================================
# Basic rigid-transform helpers
# ============================================================

def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Construct a 4x4 homogeneous rigid transform.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix.

    t : array-like, shape (3,), (3, 1), or (1, 3)
        Translation vector.

    Returns
    -------
    T : np.ndarray, shape (4, 4)
        Homogeneous transform containing R and t.
    """

    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    """
    Invert a rigid homogeneous transform without np.linalg.inv().

    If:
        T = ^A T_B = [R, t]

    then:
        inverse(T) = ^B T_A = [R.T, -R.T @ t]
    """

    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def project_rotation_to_so3(R: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Project a nearly valid rotation matrix to the nearest SO(3) matrix.

    The projection is performed with singular-value decomposition:

        R ~= U @ Vt

    A determinant correction prevents a reflection matrix.

    Returns
    -------
    R_valid : np.ndarray, shape (3, 3)
        Nearest proper rotation matrix.

    change : float
        Frobenius norm of R_valid - R. A large value means the input was not
        close to a valid rotation matrix.
    """

    R = np.asarray(R, dtype=np.float64).reshape(3, 3)

    if not np.all(np.isfinite(R)):
        raise ValueError("Rotation matrix contains NaN or infinity.")

    U, _, Vt = np.linalg.svd(R)
    R_valid = U @ Vt

    if np.linalg.det(R_valid) < 0.0:
        U[:, -1] *= -1.0
        R_valid = U @ Vt

    change = float(np.linalg.norm(R_valid - R, ord="fro"))

    if change > MAX_ROTATION_PROJECTION_CHANGE:
        raise ValueError(
            "Input is too far from a valid rotation matrix. "
            f"Projection change = {change:.6e}"
        )

    return R_valid, change


def rotation_angle_deg(R: np.ndarray) -> float:
    """
    Return the magnitude of a rotation matrix in degrees.

    The trace formula is used with clipping for numerical safety:

        theta = acos((trace(R) - 1) / 2)
    """

    R, _ = project_rotation_to_so3(R)
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def average_rotation_matrices(R_list: list[np.ndarray]) -> np.ndarray:
    """
    Compute a chordal/SVD mean of several rotation matrices.

    This is used only to recover T_ee_board after calibrateHandEye(), because
    calibrateHandEye() returns T_base_cam but does not return the board mount.
    Each sample gives an independent estimate of T_ee_board; these rotations
    are averaged and projected back to SO(3).
    """

    if not R_list:
        raise ValueError("Cannot average an empty rotation list.")

    R_sum = np.zeros((3, 3), dtype=np.float64)

    for R in R_list:
        R_valid, _ = project_rotation_to_so3(R)
        R_sum += R_valid

    # R_sum is not itself expected to be a rotation matrix, so do not apply
    # the input-validity threshold used for measured rotations. Project the
    # accumulated matrix directly to SO(3).
    U, _, Vt = np.linalg.svd(R_sum)
    R_average = U @ Vt

    if np.linalg.det(R_average) < 0.0:
        U[:, -1] *= -1.0
        R_average = U @ Vt

    return R_average


def average_transforms(T_list: list[np.ndarray]) -> np.ndarray:
    """
    Average a list of rigid transforms.

    Rotation:
        SVD/chordal rotation mean.

    Translation:
        Coordinate-wise median, which is less sensitive to an occasional
        bad pose than the arithmetic mean.
    """

    if not T_list:
        raise ValueError("Cannot average an empty transform list.")

    R_average = average_rotation_matrices([T[:3, :3] for T in T_list])
    translations = np.array([T[:3, 3] for T in T_list], dtype=np.float64)
    t_average = np.median(translations, axis=0)

    return make_T(R_average, t_average)


# ============================================================
# JSON input helpers
# ============================================================

def load_json_array(path: Path) -> np.ndarray:
    """Load a JSON file and convert its top-level list to float64."""

    if not path.exists():
        raise FileNotFoundError(f"Required calibration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a top-level JSON list in: {path}")

    array = np.asarray(data, dtype=np.float64)

    if not np.all(np.isfinite(array)):
        raise ValueError(f"NaN or infinity found in: {path}")

    return array


def load_rotation_list(path: Path) -> tuple[list[np.ndarray], float]:
    """
    Load a JSON rotation list with expected shape [N, 3, 3].

    Every rotation is projected to the nearest valid SO(3) matrix. The largest
    numerical correction is returned for reporting.
    """

    array = load_json_array(path)

    if array.ndim != 3 or array.shape[1:] != (3, 3):
        raise ValueError(
            f"Expected shape [N, 3, 3] in {path}, received {array.shape}."
        )

    rotations = []
    max_change = 0.0

    for index, R in enumerate(array):
        try:
            R_valid, change = project_rotation_to_so3(R)
        except ValueError as e:
            raise ValueError(f"Invalid rotation at sample {index} in {path}: {e}") from e

        rotations.append(R_valid)
        max_change = max(max_change, change)

    return rotations, max_change


def load_translation_list(path: Path) -> list[np.ndarray]:
    """
    Load translations and return a list of OpenCV-style 3x1 vectors.

    Accepted JSON shapes:
        [N, 3]
        [N, 3, 1]
        [N, 1, 3]
    """

    array = load_json_array(path)

    if array.ndim == 2 and array.shape[1] == 3:
        normalized = array.reshape(-1, 3, 1)

    elif array.ndim == 3 and array.shape[1:] in ((3, 1), (1, 3)):
        normalized = array.reshape(-1, 3, 1)

    else:
        raise ValueError(
            f"Expected translation shape [N,3], [N,3,1], or [N,1,3] "
            f"in {path}; received {array.shape}."
        )

    return [t.astype(np.float64, copy=True) for t in normalized]


def build_transform_list(
    rotations: list[np.ndarray],
    translations: list[np.ndarray],
    label: str,
) -> list[np.ndarray]:
    """Combine corresponding rotation and translation entries into transforms."""

    if len(rotations) != len(translations):
        raise ValueError(
            f"Length mismatch for {label}: "
            f"{len(rotations)} rotations and {len(translations)} translations."
        )

    return [make_T(R, t) for R, t in zip(rotations, translations)]


def load_calibration_data() -> dict[str, Any]:
    """
    Load all six synchronized transformation-list files.

    Returns a dictionary containing both the OpenCV-ready R/t lists and the
    corresponding 4x4 transform lists used for validation.
    """

    R_ee_base, change_ee_base = load_rotation_list(R_EE_BASE_FILE)
    t_ee_base = load_translation_list(T_EE_BASE_FILE)

    R_base_ee, change_base_ee = load_rotation_list(R_BASE_EE_FILE)
    t_base_ee = load_translation_list(T_BASE_EE_FILE)

    R_cam_board, change_cam_board = load_rotation_list(R_CAM_BOARD_FILE)
    t_cam_board = load_translation_list(T_CAM_BOARD_FILE)

    lengths = {
        "R_ee_base": len(R_ee_base),
        "t_ee_base": len(t_ee_base),
        "R_base_ee": len(R_base_ee),
        "t_base_ee": len(t_base_ee),
        "R_cam_board": len(R_cam_board),
        "t_cam_board": len(t_cam_board),
    }

    if len(set(lengths.values())) != 1:
        formatted = ", ".join(f"{name}={count}" for name, count in lengths.items())
        raise ValueError(f"Calibration file lengths are not synchronized: {formatted}")

    number_of_samples = next(iter(lengths.values()))

    if number_of_samples < MINIMUM_NUMBER_OF_POSES:
        raise ValueError(
            f"Only {number_of_samples} poses were loaded. OpenCV requires at "
            f"least {MINIMUM_NUMBER_OF_POSES} distinct poses."
        )

    T_ee_base = build_transform_list(R_ee_base, t_ee_base, "T_ee_base")
    T_base_ee = build_transform_list(R_base_ee, t_base_ee, "T_base_ee")
    T_cam_board = build_transform_list(R_cam_board, t_cam_board, "T_cam_board")

    return {
        "number_of_samples": number_of_samples,
        "R_ee_base": R_ee_base,
        "t_ee_base": t_ee_base,
        "R_base_ee": R_base_ee,
        "t_base_ee": t_base_ee,
        "R_cam_board": R_cam_board,
        "t_cam_board": t_cam_board,
        "T_ee_base": T_ee_base,
        "T_base_ee": T_base_ee,
        "T_cam_board": T_cam_board,
        "maximum_rotation_projection_change": max(
            change_ee_base,
            change_base_ee,
            change_cam_board,
        ),
    }


# ============================================================
# Input consistency checks
# ============================================================

def check_stored_robot_inverses(
    T_base_ee_list: list[np.ndarray],
    T_ee_base_list: list[np.ndarray],
) -> dict[str, float]:
    """
    Verify that each stored T_ee_base equals inverse(T_base_ee).

    This catches:
        - sample-order mismatches
        - stale files from another collection run
        - incorrect inverse translations
        - accidental use of R.T with an uninverted translation
    """

    translation_errors_m = []
    rotation_errors_deg = []

    for T_base_ee, T_ee_base_stored in zip(T_base_ee_list, T_ee_base_list):
        T_ee_base_expected = invert_T(T_base_ee)
        residual = invert_T(T_ee_base_expected) @ T_ee_base_stored

        translation_errors_m.append(float(np.linalg.norm(residual[:3, 3])))
        rotation_errors_deg.append(rotation_angle_deg(residual[:3, :3]))

    max_translation_error_m = float(np.max(translation_errors_m))
    max_rotation_error_deg = float(np.max(rotation_errors_deg))

    if (
        max_translation_error_m > INVERSE_CHECK_TRANSLATION_TOL_M
        or max_rotation_error_deg > INVERSE_CHECK_ROTATION_TOL_DEG
    ):
        raise ValueError(
            "Stored T_ee_base data does not match inverse(T_base_ee).\n"
            f"Maximum translation disagreement: {max_translation_error_m:.6e} m\n"
            f"Maximum rotation disagreement:    {max_rotation_error_deg:.6e} deg\n"
            "Delete the old calibration JSON files and recollect synchronized data."
        )

    return {
        "max_translation_error_m": max_translation_error_m,
        "max_rotation_error_deg": max_rotation_error_deg,
    }


def compute_motion_diversity(T_base_ee_list: list[np.ndarray]) -> dict[str, float]:
    """
    Report pairwise robot rotation diversity.

    Hand-eye calibration needs rotations about multiple, non-parallel axes.
    This metric does not prove the axes are ideal, but it provides a useful
    warning when almost every robot orientation is nearly identical.
    """

    pair_angles_deg = []

    for i in range(len(T_base_ee_list)):
        for j in range(i + 1, len(T_base_ee_list)):
            relative = invert_T(T_base_ee_list[j]) @ T_base_ee_list[i]
            pair_angles_deg.append(rotation_angle_deg(relative[:3, :3]))

    pair_angles = np.asarray(pair_angles_deg, dtype=np.float64)

    return {
        "pair_count": int(len(pair_angles)),
        "minimum_pair_rotation_deg": float(np.min(pair_angles)),
        "median_pair_rotation_deg": float(np.median(pair_angles)),
        "maximum_pair_rotation_deg": float(np.max(pair_angles)),
        "pairs_above_10_deg": int(np.sum(pair_angles > 10.0)),
        "pairs_above_20_deg": int(np.sum(pair_angles > 20.0)),
    }


def check_translation_scale(
    T_base_ee_list: list[np.ndarray],
    T_cam_board_list: list[np.ndarray],
) -> dict[str, float | bool]:
    """
    Perform a simple meter-versus-millimeter sanity check.

    This is only a warning heuristic. Robot and camera translations do not need
    to have equal magnitudes, but a ratio near 1000 often indicates that one
    source uses millimeters and the other uses meters.
    """

    robot_norms = np.array(
        [np.linalg.norm(T[:3, 3]) for T in T_base_ee_list],
        dtype=np.float64,
    )
    camera_norms = np.array(
        [np.linalg.norm(T[:3, 3]) for T in T_cam_board_list],
        dtype=np.float64,
    )

    robot_median = float(np.median(robot_norms))
    camera_median = float(np.median(camera_norms))

    denominator = max(min(robot_median, camera_median), 1e-12)
    magnitude_ratio = max(robot_median, camera_median) / denominator

    return {
        "median_robot_translation_norm": robot_median,
        "median_camera_translation_norm": camera_median,
        "larger_to_smaller_ratio": float(magnitude_ratio),
        "possible_unit_mismatch": bool(magnitude_ratio > 100.0),
    }


# ============================================================
# Calibration validation
# ============================================================

def validate_closed_loop(
    T_base_cam: np.ndarray,
    T_ee_board: np.ndarray,
    T_base_ee_list: list[np.ndarray],
    T_cam_board_list: list[np.ndarray],
) -> dict[str, Any]:
    """
    Validate one calibration solution on every captured sample.

    The correct physical equation is:

        T_base_ee[i] @ T_ee_board
            ~=
        T_base_cam @ T_cam_board[i]

    Both sides are ^B T_W: the board pose expressed in robot-base coordinates.

    Residual definition:

        residual_i = inverse(left_i) @ right_i

    A perfect calibration gives the 4x4 identity matrix for every residual.
    """

    translation_errors_m = []
    rotation_errors_deg = []
    per_sample = []

    for sample_id, (T_base_ee, T_cam_board) in enumerate(
        zip(T_base_ee_list, T_cam_board_list)
    ):
        left = T_base_ee @ T_ee_board
        right = T_base_cam @ T_cam_board
        residual = invert_T(left) @ right

        translation_error_m = float(np.linalg.norm(residual[:3, 3]))
        rotation_error_deg = rotation_angle_deg(residual[:3, :3])

        translation_errors_m.append(translation_error_m)
        rotation_errors_deg.append(rotation_error_deg)

        per_sample.append(
            {
                "sample_id": int(sample_id),
                "translation_error_m": translation_error_m,
                "translation_error_mm": 1000.0 * translation_error_m,
                "rotation_error_deg": rotation_error_deg,
            }
        )

    translation = np.asarray(translation_errors_m, dtype=np.float64)
    rotation = np.asarray(rotation_errors_deg, dtype=np.float64)

    translation_rms_m = float(np.sqrt(np.mean(translation**2)))
    rotation_rms_deg = float(np.sqrt(np.mean(rotation**2)))
    rotation_rms_rad = float(np.radians(rotation_rms_deg))

    # Convert rotation residual into approximate displacement at a known radius.
    # This gives a single score in meters only for ranking the tested methods.
    ranking_score_m = (
        translation_rms_m
        + ROTATION_ERROR_EQUIVALENT_RADIUS_M * rotation_rms_rad
    )

    return {
        "translation": {
            "mean_m": float(np.mean(translation)),
            "median_m": float(np.median(translation)),
            "rms_m": translation_rms_m,
            "max_m": float(np.max(translation)),
            "mean_mm": 1000.0 * float(np.mean(translation)),
            "median_mm": 1000.0 * float(np.median(translation)),
            "rms_mm": 1000.0 * translation_rms_m,
            "max_mm": 1000.0 * float(np.max(translation)),
        },
        "rotation": {
            "mean_deg": float(np.mean(rotation)),
            "median_deg": float(np.median(rotation)),
            "rms_deg": rotation_rms_deg,
            "max_deg": float(np.max(rotation)),
        },
        "ranking_score_m": float(ranking_score_m),
        "ranking_score_mm": 1000.0 * float(ranking_score_m),
        "rotation_equivalent_radius_m": ROTATION_ERROR_EQUIVALENT_RADIUS_M,
        "per_sample": per_sample,
    }


# ============================================================
# cv2.calibrateHandEye solver
# ============================================================

def solve_with_calibrate_hand_eye(
    method_name: str,
    method_value: int,
    data: dict[str, Any],
) -> dict[str, Any]:
    """
    Solve fixed-camera eye-to-hand calibration with calibrateHandEye().

    Important OpenCV frame mapping
    ------------------------------
    The documented argument names describe the common eye-in-hand case, but
    OpenCV also documents the eye-to-hand form of AX = XB.

    For this fixed-camera setup, pass:

        R_gripper2base <- R_ee_base = ^E R_B
        t_gripper2base <- t_ee_base = ^E t_B

        R_target2cam   <- R_cam_board = ^C R_W
        t_target2cam   <- t_cam_board = ^C t_W

    The returned nominal "cam2gripper" transform is physically:

        T_base_cam = ^B T_C

    Why?
    ----
    Define:

        G_i = ^E T_B[i]
        X   = ^B T_C
        C_i = ^C T_W[i]
        Y   = ^E T_W

    The rigid board attachment gives:

        G_i @ X @ C_i = Y

    Comparing two samples creates:

        inverse(G_j) @ G_i @ X
            =
        X @ C_j @ inverse(C_i)

    which is AX = XB. This is exactly the relative-motion construction used by
    calibrateHandEye().

    calibrateHandEye() returns only T_base_cam. Once it is known, each sample
    independently provides:

        T_ee_board_i
            =
        T_ee_base_i @ T_base_cam @ T_cam_board_i

    Those per-sample board transforms are averaged to obtain T_ee_board.
    """

    R_base_cam, t_base_cam = cv2.calibrateHandEye(
        R_gripper2base=data["R_ee_base"],
        t_gripper2base=data["t_ee_base"],
        R_target2cam=data["R_cam_board"],
        t_target2cam=data["t_cam_board"],
        method=method_value,
    )

    T_base_cam = make_T(R_base_cam, t_base_cam)

    if not np.all(np.isfinite(T_base_cam)):
        raise RuntimeError("calibrateHandEye returned NaN or infinity.")

    # Recover the board-to-end-effector mounting transform from every sample.
    T_ee_board_samples = []

    for T_ee_base, T_cam_board in zip(
        data["T_ee_base"],
        data["T_cam_board"],
    ):
        T_ee_board_i = T_ee_base @ T_base_cam @ T_cam_board
        T_ee_board_samples.append(T_ee_board_i)

    T_ee_board = average_transforms(T_ee_board_samples)

    metrics = validate_closed_loop(
        T_base_cam=T_base_cam,
        T_ee_board=T_ee_board,
        T_base_ee_list=data["T_base_ee"],
        T_cam_board_list=data["T_cam_board"],
    )

    return {
        "name": method_name,
        "solver_family": "cv2.calibrateHandEye",
        "opencv_method_value": int(method_value),
        "T_base_camera": T_base_cam,
        "T_ee_board": T_ee_board,
        "metrics": metrics,
    }


# ============================================================
# cv2.calibrateRobotWorldHandEye solver
# ============================================================

def solve_with_calibrate_robot_world_hand_eye(
    method_name: str,
    method_value: int,
    data: dict[str, Any],
) -> dict[str, Any]:
    """
    Solve both constant transforms with calibrateRobotWorldHandEye().

    OpenCV solves the robot-world/hand-eye equation using abstract frames:

        ^C T_W[i] @ ^W T_B
            =
        ^C T_G @ ^G T_B[i]

    and returns:

        ^W T_B
        ^C T_G

    To make this equation represent the physical eye-to-hand setup, relabel
    OpenCV's abstract frames as follows:

        OpenCV "world"   W  -> physical ChArUco board W
        OpenCV "camera"  C  -> physical fixed camera C
        OpenCV "base"    B  -> physical end effector E
        OpenCV "gripper" G  -> physical robot base B

    Under this relabeling, the required OpenCV inputs become:

        ^C T_W[i] -> physical T_cam_board[i]
        ^G T_B[i] -> physical ^B T_E[i] = T_base_ee[i]

    Therefore this function receives DIRECT FK, not inverse FK:

        R_world2cam    <- R_cam_board
        t_world2cam    <- t_cam_board
        R_base2gripper <- R_base_ee
        t_base2gripper <- t_base_ee

    OpenCV returns:

        nominal ^W T_B -> physical ^W T_E = T_board_ee
        nominal ^C T_G -> physical ^C T_B = T_cam_base

    The transforms wanted by this project are their rigid inverses:

        T_ee_board = inverse(T_board_ee)
        T_base_cam = inverse(T_cam_base)
    """

    (
        R_board_ee,
        t_board_ee,
        R_cam_base,
        t_cam_base,
    ) = cv2.calibrateRobotWorldHandEye(
        R_world2cam=data["R_cam_board"],
        t_world2cam=data["t_cam_board"],
        R_base2gripper=data["R_base_ee"],
        t_base2gripper=data["t_base_ee"],
        method=method_value,
    )

    T_board_ee = make_T(R_board_ee, t_board_ee)
    T_cam_base = make_T(R_cam_base, t_cam_base)

    if not np.all(np.isfinite(T_board_ee)):
        raise RuntimeError("calibrateRobotWorldHandEye returned invalid T_board_ee.")

    if not np.all(np.isfinite(T_cam_base)):
        raise RuntimeError("calibrateRobotWorldHandEye returned invalid T_cam_base.")

    T_ee_board = invert_T(T_board_ee)
    T_base_cam = invert_T(T_cam_base)

    metrics = validate_closed_loop(
        T_base_cam=T_base_cam,
        T_ee_board=T_ee_board,
        T_base_ee_list=data["T_base_ee"],
        T_cam_board_list=data["T_cam_board"],
    )

    return {
        "name": method_name,
        "solver_family": "cv2.calibrateRobotWorldHandEye",
        "opencv_method_value": int(method_value),
        "T_base_camera": T_base_cam,
        "T_ee_board": T_ee_board,
        "opencv_raw_outputs": {
            "T_board_ee": T_board_ee,
            "T_camera_base": T_cam_base,
        },
        "metrics": metrics,
    }


# ============================================================
# Result formatting and saving
# ============================================================

def numpy_to_json_compatible(value: Any) -> Any:
    """Recursively convert NumPy values into ordinary JSON-compatible values."""

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(key): numpy_to_json_compatible(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [numpy_to_json_compatible(item) for item in value]

    return value


def save_results(
    results: list[dict[str, Any]],
    best_result: dict[str, Any],
    dataset_info: dict[str, Any],
) -> None:
    """Save all method-specific transforms and one complete JSON summary."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for result in results:
        method_name = result["name"]

        np.save(
            RESULTS_DIR / f"T_base_camera_{method_name}.npy",
            result["T_base_camera"],
        )
        np.save(
            RESULTS_DIR / f"T_ee_board_{method_name}.npy",
            result["T_ee_board"],
        )

    np.save(BEST_T_BASE_CAM_FILE, best_result["T_base_camera"])
    np.save(BEST_T_EE_BOARD_FILE, best_result["T_ee_board"])

    summary = {
        "opencv_version": cv2.__version__,
        "frame_convention": "T_A_B maps coordinates from frame B into frame A",
        "physical_equation": (
            "T_base_ee[i] @ T_ee_board = "
            "T_base_camera @ T_camera_board[i]"
        ),
        "dataset": dataset_info,
        "selection": {
            "best_method": best_result["name"],
            "best_solver_family": best_result["solver_family"],
            "ranking_metric": (
                "translation_rms_m + rotation_equivalent_radius_m "
                "* rotation_rms_rad"
            ),
            "rotation_equivalent_radius_m": (
                ROTATION_ERROR_EQUIVALENT_RADIUS_M
            ),
        },
        "results": results,
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(
            numpy_to_json_compatible(summary),
            f,
            indent=4,
        )


def print_transform(label: str, T: np.ndarray) -> None:
    """Print a homogeneous transform with readable fixed precision."""

    print(f"\n{label}:")
    print(np.array2string(T, precision=7, suppress_small=True))


def print_method_result(result: dict[str, Any]) -> None:
    """Print one method's transforms and closed-loop residual summary."""

    metrics = result["metrics"]
    translation = metrics["translation"]
    rotation = metrics["rotation"]

    print("\n" + "=" * 78)
    print(f"METHOD: {result['name']}")
    print(f"FAMILY: {result['solver_family']}")
    print("=" * 78)

    print_transform("T_base_camera = ^B T_C", result["T_base_camera"])
    print_transform("T_ee_board = ^E T_W", result["T_ee_board"])

    print("\nClosed-loop residuals:")
    print(
        "  Translation: "
        f"mean={translation['mean_mm']:.3f} mm, "
        f"median={translation['median_mm']:.3f} mm, "
        f"RMS={translation['rms_mm']:.3f} mm, "
        f"max={translation['max_mm']:.3f} mm"
    )
    print(
        "  Rotation:    "
        f"mean={rotation['mean_deg']:.3f} deg, "
        f"median={rotation['median_deg']:.3f} deg, "
        f"RMS={rotation['rms_deg']:.3f} deg, "
        f"max={rotation['max_deg']:.3f} deg"
    )
    print(f"  Ranking score: {metrics['ranking_score_mm']:.3f} mm-equivalent")


def print_ranking(results: list[dict[str, Any]]) -> None:
    """Print all successful methods ordered by the geometric ranking score."""

    print("\n" + "=" * 78)
    print("METHOD RANKING")
    print("=" * 78)
    print(
        f"{'Rank':<6} {'Method':<26} "
        f"{'RMS trans [mm]':>16} {'RMS rot [deg]':>16} {'Score [mm]':>13}"
    )
    print("-" * 78)

    for rank, result in enumerate(results, start=1):
        metrics = result["metrics"]
        print(
            f"{rank:<6} "
            f"{result['name']:<26} "
            f"{metrics['translation']['rms_mm']:>16.3f} "
            f"{metrics['rotation']['rms_deg']:>16.3f} "
            f"{metrics['ranking_score_mm']:>13.3f}"
        )


# ============================================================
# Main program
# ============================================================

def main() -> int:
    """Load data, run all requested OpenCV solvers, validate, rank, and save."""

    print("=" * 78)
    print("SO-ARM101 EYE-TO-HAND CALIBRATION WITH OPENCV")
    print("=" * 78)
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Input directory: {DATA_DIR}")

    try:
        data = load_calibration_data()

        inverse_check = check_stored_robot_inverses(
            data["T_base_ee"],
            data["T_ee_base"],
        )

        motion_diversity = compute_motion_diversity(data["T_base_ee"])
        translation_scale = check_translation_scale(
            data["T_base_ee"],
            data["T_cam_board"],
        )

    except Exception as e:
        print(f"\n[ERROR] Could not load or validate calibration data:\n{e}")
        return 1

    number_of_samples = data["number_of_samples"]

    print(f"\nLoaded synchronized poses: {number_of_samples}")
    print(
        "Maximum rotation SO(3) projection correction: "
        f"{data['maximum_rotation_projection_change']:.6e}"
    )
    print(
        "Stored inverse check: "
        f"{inverse_check['max_translation_error_m']:.3e} m, "
        f"{inverse_check['max_rotation_error_deg']:.3e} deg"
    )

    print("\nRobot motion diversity:")
    print(
        f"  Pairwise rotation range: "
        f"{motion_diversity['minimum_pair_rotation_deg']:.2f} to "
        f"{motion_diversity['maximum_pair_rotation_deg']:.2f} deg"
    )
    print(
        f"  Median pairwise rotation: "
        f"{motion_diversity['median_pair_rotation_deg']:.2f} deg"
    )
    print(
        f"  Pairs above 10 deg: {motion_diversity['pairs_above_10_deg']} | "
        f"above 20 deg: {motion_diversity['pairs_above_20_deg']}"
    )

    print("\nTranslation-scale check:")
    print(
        "  Median |t_base_ee|:  "
        f"{translation_scale['median_robot_translation_norm']:.6f}"
    )
    print(
        "  Median |t_cam_board|: "
        f"{translation_scale['median_camera_translation_norm']:.6f}"
    )
    print(
        "  Larger/smaller ratio: "
        f"{translation_scale['larger_to_smaller_ratio']:.3f}"
    )

    if number_of_samples < RECOMMENDED_NUMBER_OF_POSES:
        print(
            f"\n[WARNING] Only {number_of_samples} poses are available. "
            f"At least {RECOMMENDED_NUMBER_OF_POSES} diverse poses are "
            "recommended for a practical calibration."
        )

    if motion_diversity["maximum_pair_rotation_deg"] < 15.0:
        print(
            "\n[WARNING] Robot orientation diversity is low. Include larger "
            "rotations about several non-parallel axes."
        )

    if translation_scale["possible_unit_mismatch"]:
        print(
            "\n[WARNING] The robot and camera translation magnitudes differ "
            "by more than 100x. Check for a meter/millimeter mismatch."
        )

    successful_results = []
    failed_methods = {}

    print("\nRunning cv2.calibrateHandEye methods...")

    for method_name, method_value in HAND_EYE_METHODS.items():
        try:
            result = solve_with_calibrate_hand_eye(
                method_name=method_name,
                method_value=method_value,
                data=data,
            )
            successful_results.append(result)
            print(f"  [OK] {method_name}")
        except Exception as e:
            failed_methods[method_name] = str(e)
            print(f"  [FAILED] {method_name}: {e}")

    print("\nRunning cv2.calibrateRobotWorldHandEye methods...")

    for method_name, method_value in ROBOT_WORLD_METHODS.items():
        try:
            result = solve_with_calibrate_robot_world_hand_eye(
                method_name=method_name,
                method_value=method_value,
                data=data,
            )
            successful_results.append(result)
            print(f"  [OK] {method_name}")
        except Exception as e:
            failed_methods[method_name] = str(e)
            print(f"  [FAILED] {method_name}: {e}")

    if not successful_results:
        print("\n[ERROR] Every OpenCV calibration method failed.")
        return 1

    successful_results.sort(
        key=lambda result: result["metrics"]["ranking_score_m"]
    )

    for result in successful_results:
        print_method_result(result)

    print_ranking(successful_results)

    best_result = successful_results[0]

    dataset_info = {
        "number_of_samples": number_of_samples,
        "input_directory": str(DATA_DIR),
        "input_files": {
            "R_ee_base": str(R_EE_BASE_FILE),
            "t_ee_base": str(T_EE_BASE_FILE),
            "R_base_ee": str(R_BASE_EE_FILE),
            "t_base_ee": str(T_BASE_EE_FILE),
            "R_cam_board": str(R_CAM_BOARD_FILE),
            "t_cam_board": str(T_CAM_BOARD_FILE),
        },
        "inverse_check": inverse_check,
        "motion_diversity": motion_diversity,
        "translation_scale_check": translation_scale,
        "failed_methods": failed_methods,
    }

    try:
        save_results(
            results=successful_results,
            best_result=best_result,
            dataset_info=dataset_info,
        )
    except Exception as e:
        print(f"\n[ERROR] Calibration ran, but result saving failed:\n{e}")
        return 1

    print("\n" + "=" * 78)
    print("SELECTED RESULT")
    print("=" * 78)
    print(f"Best method: {best_result['name']}")
    print_transform("Best T_base_camera = ^B T_C", best_result["T_base_camera"])
    print_transform("Best T_ee_board = ^E T_W", best_result["T_ee_board"])

    print("\nSaved outputs:")
    print(f"  Summary:          {SUMMARY_FILE}")
    print(f"  Best base-camera: {BEST_T_BASE_CAM_FILE}")
    print(f"  Best EE-board:    {BEST_T_EE_BOARD_FILE}")
    print(f"  Method files:     {RESULTS_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())