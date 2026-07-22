#!/usr/bin/env python3

"""
Test how cv2.calibrateHandEye() output should be interpreted for this
SO-ARM101 fixed-camera eye-to-hand setup.

This script does NOT recollect data. It loads the synchronized JSON files:

    R_ee_base.json
    t_ee_base.json
    R_base_ee.json
    t_base_ee.json
    R_cam_board.json
    t_cam_board.json

For every OpenCV hand-eye method, the script:

    1. Calls cv2.calibrateHandEye() using:
           R_gripper2base = R_base_ee = ^B R_E
           t_gripper2base = t_base_ee = ^B t_E
           R_target2cam   = R_cam_board = ^C R_W
           t_target2cam   = t_cam_board = ^C t_W

    2. Treats the returned transform in two ways:
           candidate A: raw OpenCV output
           candidate B: inverse(raw OpenCV output)

    3. For each candidate T_base_cam, recovers one T_ee_board per sample:

           T_ee_board_i
               =
           inverse(T_base_ee_i)
           @ T_base_cam_candidate
           @ T_cam_board_i

    4. Measures how constant the recovered T_ee_board_i transforms are.

The correct interpretation should produce:
    - a physically plausible T_base_cam translation
    - small translation spread among T_ee_board_i
    - small rotation spread among T_ee_board_i
    - small closed-loop residuals

A readable report is written to:

    data/eye_to_hand/opencv_direction_test.txt
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Paths
# ============================================================

ROOT = Path("/home/agbara-admin/Documents/Surgical_Cleaning_Robot")
DATA_DIR = ROOT / "data" / "eye_to_hand"

R_EE_BASE_FILE = DATA_DIR / "automatic_calibration" / "R_ee_base.json"
T_EE_BASE_FILE = DATA_DIR / "automatic_calibration" /  "t_ee_base.json"
R_BASE_EE_FILE = DATA_DIR / "automatic_calibration" / "R_base_ee.json"
T_BASE_EE_FILE = DATA_DIR / "automatic_calibration" / "t_base_ee.json"
R_CAM_BOARD_FILE = DATA_DIR / "automatic_calibration" / "R_cam_board.json"
T_CAM_BOARD_FILE = DATA_DIR / "automatic_calibration" / "t_cam_board.json"

REPORT_FILE = DATA_DIR / "opencv_direction_test.txt"


# ============================================================
# Physical sanity ranges for camera translation
# ============================================================

# These ranges are only used for a human-readable sanity check.
# They do NOT change the calibration or ranking.
EXPECTED_X_MIN_M = -0.25
EXPECTED_X_MAX_M = -0.05

EXPECTED_Y_MIN_M = 0.10
EXPECTED_Z_MIN_M = 0.10


# ============================================================
# OpenCV methods
# ============================================================

HAND_EYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


# ============================================================
# Transform helpers
# ============================================================

def make_T(R, t):
    """
    Build a 4x4 rigid transform.

    T_A_B maps coordinates from frame B into frame A.
    """

    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def invert_T(T):
    """
    Rigid inverse:

        [R, t]^-1 = [R.T, -R.T @ t]
    """

    T = np.asarray(T, dtype=np.float64).reshape(4, 4)

    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T_inv


def project_to_so3(R):
    """
    Project a nearly valid rotation matrix to the nearest proper rotation.
    """

    R = np.asarray(R, dtype=np.float64).reshape(3, 3)

    U, _, Vt = np.linalg.svd(R)
    R_valid = U @ Vt

    if np.linalg.det(R_valid) < 0.0:
        U[:, -1] *= -1.0
        R_valid = U @ Vt

    return R_valid


def rotation_angle_deg(R):
    """
    Return the magnitude of a relative rotation in degrees.
    """

    R = project_to_so3(R)

    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return float(np.degrees(np.arccos(cos_theta)))


def average_rotation_matrices(R_list):
    """
    Chordal/SVD rotation mean.
    """

    R_sum = np.zeros((3, 3), dtype=np.float64)

    for R in R_list:
        R_sum += project_to_so3(R)

    return project_to_so3(R_sum)


def average_transforms(T_list):
    """
    Average transforms using:
        - SVD rotation mean
        - coordinate-wise median translation
    """

    R_mean = average_rotation_matrices(
        [T[:3, :3] for T in T_list]
    )

    translations = np.asarray(
        [T[:3, 3] for T in T_list],
        dtype=np.float64,
    )

    t_median = np.median(translations, axis=0)

    return make_T(R_mean, t_median)


def format_matrix(T, precision=7):
    """
    Format a matrix for the text report.
    """

    return np.array2string(
        np.asarray(T),
        precision=precision,
        suppress_small=True,
    )


# ============================================================
# Input loading
# ============================================================

def load_json_array(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    array = np.asarray(data, dtype=np.float64)

    if not np.all(np.isfinite(array)):
        raise ValueError(f"NaN or infinity found in {path}")

    return array


def load_rotation_list(path):
    array = load_json_array(path)

    if array.ndim != 3 or array.shape[1:] != (3, 3):
        raise ValueError(
            f"{path.name} must have shape [N,3,3], got {array.shape}"
        )

    return [
        project_to_so3(R)
        for R in array
    ]


def load_translation_list(path):
    array = load_json_array(path)

    if array.ndim == 2 and array.shape[1] == 3:
        array = array.reshape(-1, 3, 1)

    elif array.ndim == 3 and array.shape[1:] in ((3, 1), (1, 3)):
        array = array.reshape(-1, 3, 1)

    else:
        raise ValueError(
            f"{path.name} must have shape [N,3], [N,3,1], or [N,1,3], "
            f"got {array.shape}"
        )

    return [
        t.astype(np.float64, copy=True)
        for t in array
    ]


def build_transform_list(R_list, t_list, label):
    if len(R_list) != len(t_list):
        raise ValueError(
            f"Length mismatch for {label}: "
            f"{len(R_list)} rotations vs {len(t_list)} translations"
        )

    return [
        make_T(R, t)
        for R, t in zip(R_list, t_list)
    ]


def load_data():
    R_ee_base = load_rotation_list(R_EE_BASE_FILE)
    t_ee_base = load_translation_list(T_EE_BASE_FILE)

    R_base_ee = load_rotation_list(R_BASE_EE_FILE)
    t_base_ee = load_translation_list(T_BASE_EE_FILE)

    R_cam_board = load_rotation_list(R_CAM_BOARD_FILE)
    t_cam_board = load_translation_list(T_CAM_BOARD_FILE)

    lengths = {
        len(R_ee_base),
        len(t_ee_base),
        len(R_base_ee),
        len(t_base_ee),
        len(R_cam_board),
        len(t_cam_board),
    }

    if len(lengths) != 1:
        raise ValueError("Input JSON files do not contain the same number of samples.")

    number_of_samples = len(R_base_ee)

    if number_of_samples < 3:
        raise ValueError("At least 3 synchronized poses are required.")

    T_ee_base = build_transform_list(
        R_ee_base,
        t_ee_base,
        "T_ee_base",
    )

    T_base_ee = build_transform_list(
        R_base_ee,
        t_base_ee,
        "T_base_ee",
    )

    T_cam_board = build_transform_list(
        R_cam_board,
        t_cam_board,
        "T_cam_board",
    )

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
    }


# ============================================================
# Candidate evaluation
# ============================================================

def physical_translation_check(T_base_cam):
    """
    Apply the expected sign/range sanity check.

    Expected:
        x = small negative, approximately -0.10 to -0.20 m
        y = positive
        z = positive
    """

    x, y, z = T_base_cam[:3, 3]

    x_ok = EXPECTED_X_MIN_M <= x <= EXPECTED_X_MAX_M
    y_ok = y >= EXPECTED_Y_MIN_M
    z_ok = z >= EXPECTED_Z_MIN_M

    return {
        "x_ok": bool(x_ok),
        "y_ok": bool(y_ok),
        "z_ok": bool(z_ok),
        "overall": bool(x_ok and y_ok and z_ok),
    }


def recover_board_mounts(T_base_cam, data):
    """
    Recover one board mounting transform per sample:

        T_ee_board_i
            =
        inverse(T_base_ee_i)
        @ T_base_cam
        @ T_cam_board_i
    """

    mounts = []

    for T_base_ee, T_cam_board in zip(
        data["T_base_ee"],
        data["T_cam_board"],
    ):
        T_ee_board_i = (
            invert_T(T_base_ee)
            @ T_base_cam
            @ T_cam_board
        )

        mounts.append(T_ee_board_i)

    return mounts


def compute_mount_consistency(T_ee_board_samples):
    """
    Measure how tightly all recovered T_ee_board_i transforms cluster.
    """

    T_mean = average_transforms(T_ee_board_samples)

    translation_errors_m = []
    rotation_errors_deg = []

    for T_i in T_ee_board_samples:
        residual = invert_T(T_mean) @ T_i

        translation_errors_m.append(
            float(np.linalg.norm(residual[:3, 3]))
        )

        rotation_errors_deg.append(
            rotation_angle_deg(residual[:3, :3])
        )

    translation_errors_m = np.asarray(
        translation_errors_m,
        dtype=np.float64,
    )

    rotation_errors_deg = np.asarray(
        rotation_errors_deg,
        dtype=np.float64,
    )

    translations = np.asarray(
        [T[:3, 3] for T in T_ee_board_samples],
        dtype=np.float64,
    )

    return {
        "T_ee_board_mean": T_mean,
        "translation_median_m": np.median(translations, axis=0),
        "translation_std_mm_xyz": np.std(translations, axis=0) * 1000.0,
        "translation_rms_mm": (
            1000.0
            * float(np.sqrt(np.mean(translation_errors_m ** 2)))
        ),
        "translation_max_mm": (
            1000.0
            * float(np.max(translation_errors_m))
        ),
        "rotation_rms_deg": float(
            np.sqrt(np.mean(rotation_errors_deg ** 2))
        ),
        "rotation_max_deg": float(
            np.max(rotation_errors_deg)
        ),
        "per_sample_translation_mm": (
            translation_errors_m * 1000.0
        ),
        "per_sample_rotation_deg": rotation_errors_deg,
    }


def compute_closed_loop_metrics(T_base_cam, T_ee_board, data):
    """
    Validate:

        T_base_ee_i @ T_ee_board
            =
        T_base_cam @ T_cam_board_i
    """

    translation_errors_m = []
    rotation_errors_deg = []

    for T_base_ee, T_cam_board in zip(
        data["T_base_ee"],
        data["T_cam_board"],
    ):
        left = T_base_ee @ T_ee_board
        right = T_base_cam @ T_cam_board

        residual = invert_T(left) @ right

        translation_errors_m.append(
            float(np.linalg.norm(residual[:3, 3]))
        )

        rotation_errors_deg.append(
            rotation_angle_deg(residual[:3, :3])
        )

    translation_errors_m = np.asarray(
        translation_errors_m,
        dtype=np.float64,
    )

    rotation_errors_deg = np.asarray(
        rotation_errors_deg,
        dtype=np.float64,
    )

    return {
        "translation_rms_mm": (
            1000.0
            * float(np.sqrt(np.mean(translation_errors_m ** 2)))
        ),
        "translation_max_mm": (
            1000.0
            * float(np.max(translation_errors_m))
        ),
        "rotation_rms_deg": float(
            np.sqrt(np.mean(rotation_errors_deg ** 2))
        ),
        "rotation_max_deg": float(
            np.max(rotation_errors_deg)
        ),
    }


def evaluate_candidate(candidate_name, T_base_cam, data):
    """
    Evaluate one interpretation of the OpenCV output.
    """

    T_ee_board_samples = recover_board_mounts(
        T_base_cam,
        data,
    )

    consistency = compute_mount_consistency(
        T_ee_board_samples
    )

    closed_loop = compute_closed_loop_metrics(
        T_base_cam,
        consistency["T_ee_board_mean"],
        data,
    )

    physical_check = physical_translation_check(
        T_base_cam
    )

    return {
        "candidate_name": candidate_name,
        "T_base_cam": T_base_cam,
        "T_ee_board_samples": T_ee_board_samples,
        "consistency": consistency,
        "closed_loop": closed_loop,
        "physical_check": physical_check,
    }


# ============================================================
# Report writing
# ============================================================

def write_candidate_section(f, candidate):
    T_base_cam = candidate["T_base_cam"]
    consistency = candidate["consistency"]
    closed_loop = candidate["closed_loop"]
    physical = candidate["physical_check"]

    x, y, z = T_base_cam[:3, 3]

    f.write("\n")
    f.write("-" * 78 + "\n")
    f.write(f"CANDIDATE: {candidate['candidate_name']}\n")
    f.write("-" * 78 + "\n")

    f.write("\nT_base_cam candidate = ^B T_C\n")
    f.write(format_matrix(T_base_cam) + "\n")

    f.write("\nCamera translation in robot base frame [m]\n")
    f.write(f"  x = {x:+.6f}\n")
    f.write(f"  y = {y:+.6f}\n")
    f.write(f"  z = {z:+.6f}\n")

    f.write("\nPhysical translation sanity check\n")
    f.write(
        f"  expected x in [{EXPECTED_X_MIN_M:+.3f}, "
        f"{EXPECTED_X_MAX_M:+.3f}] m : "
        f"{'PASS' if physical['x_ok'] else 'FAIL'}\n"
    )
    f.write(
        f"  expected y >= {EXPECTED_Y_MIN_M:+.3f} m : "
        f"{'PASS' if physical['y_ok'] else 'FAIL'}\n"
    )
    f.write(
        f"  expected z >= {EXPECTED_Z_MIN_M:+.3f} m : "
        f"{'PASS' if physical['z_ok'] else 'FAIL'}\n"
    )
    f.write(
        f"  overall physical sanity           : "
        f"{'PASS' if physical['overall'] else 'FAIL'}\n"
    )

    f.write("\nRecovered mean T_ee_board = ^E T_W\n")
    f.write(format_matrix(consistency["T_ee_board_mean"]) + "\n")

    med = consistency["translation_median_m"]
    std = consistency["translation_std_mm_xyz"]

    f.write("\nRecovered T_ee_board translation statistics\n")
    f.write(
        "  median [m] = "
        f"[{med[0]:+.6f}, {med[1]:+.6f}, {med[2]:+.6f}]\n"
    )
    f.write(
        "  std [mm]   = "
        f"[{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]\n"
    )

    f.write("\nRecovered T_ee_board consistency\n")
    f.write(
        f"  translation RMS = "
        f"{consistency['translation_rms_mm']:.3f} mm\n"
    )
    f.write(
        f"  translation max = "
        f"{consistency['translation_max_mm']:.3f} mm\n"
    )
    f.write(
        f"  rotation RMS    = "
        f"{consistency['rotation_rms_deg']:.3f} deg\n"
    )
    f.write(
        f"  rotation max    = "
        f"{consistency['rotation_max_deg']:.3f} deg\n"
    )

    f.write("\nClosed-loop validation using the mean recovered T_ee_board\n")
    f.write(
        f"  translation RMS = "
        f"{closed_loop['translation_rms_mm']:.3f} mm\n"
    )
    f.write(
        f"  translation max = "
        f"{closed_loop['translation_max_mm']:.3f} mm\n"
    )
    f.write(
        f"  rotation RMS    = "
        f"{closed_loop['rotation_rms_deg']:.3f} deg\n"
    )
    f.write(
        f"  rotation max    = "
        f"{closed_loop['rotation_max_deg']:.3f} deg\n"
    )

    f.write("\nPer-sample recovered T_ee_board deviation from mean\n")
    f.write(
        f"{'Sample':>8} "
        f"{'Translation [mm]':>20} "
        f"{'Rotation [deg]':>18}\n"
    )
    f.write("-" * 50 + "\n")

    for i, (trans_mm, rot_deg) in enumerate(
        zip(
            consistency["per_sample_translation_mm"],
            consistency["per_sample_rotation_deg"],
        )
    ):
        f.write(
            f"{i:>8d} "
            f"{trans_mm:>20.3f} "
            f"{rot_deg:>18.3f}\n"
        )


def candidate_score(candidate):
    """
    Score candidates only by recovered rigid-mount consistency.

    Lower is better.
    """

    c = candidate["consistency"]

    # Convert rotation into an approximate displacement at 0.1 m.
    rotation_equivalent_mm = (
        100.0
        * np.radians(c["rotation_rms_deg"])
    )

    return (
        c["translation_rms_mm"]
        + rotation_equivalent_mm
    )


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 78)
    print("OPENCV HAND-EYE OUTPUT DIRECTION TEST")
    print("=" * 78)
    print(f"OpenCV version : {cv2.__version__}")
    print(f"Input folder   : {DATA_DIR}")
    print(f"Report file    : {REPORT_FILE}")

    try:
        data = load_data()
    except Exception as e:
        print(f"\n[ERROR] Could not load calibration data:\n{e}")
        return 1

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 78 + "\n")
        f.write("OPENCV HAND-EYE OUTPUT DIRECTION TEST\n")
        f.write("=" * 78 + "\n")

        f.write(f"OpenCV version       : {cv2.__version__}\n")
        f.write(f"Number of samples    : {data['number_of_samples']}\n")
        f.write(f"Input directory      : {DATA_DIR}\n")

        f.write("\nTransform convention\n")
        f.write("  T_A_B maps coordinates from frame B into frame A.\n")

        f.write("\nOpenCV input mapping used\n")
        f.write("  R_gripper2base = R_base_ee   = ^B R_E\n")
        f.write("  t_gripper2base = t_base_ee   = ^B t_E\n")
        f.write("  R_target2cam   = R_cam_board = ^C R_W\n")
        f.write("  t_target2cam   = t_cam_board = ^C t_W\n")

        f.write("\nFor each raw OpenCV output, two candidates are tested:\n")
        f.write("  1. RAW output interpreted as T_base_cam\n")
        f.write("  2. INVERSE of raw output interpreted as T_base_cam\n")

        for method_name, method_value in HAND_EYE_METHODS.items():
            f.write("\n\n")
            f.write("=" * 78 + "\n")
            f.write(f"METHOD: {method_name}\n")
            f.write("=" * 78 + "\n")

            try:
                R_raw, t_raw = cv2.calibrateHandEye(
                    R_gripper2base=data["R_base_ee"],
                    t_gripper2base=data["t_base_ee"],
                    R_target2cam=data["R_cam_board"],
                    t_target2cam=data["t_cam_board"],
                    method=method_value,
                )

                T_raw = make_T(R_raw, t_raw)
                T_inverse = invert_T(T_raw)

                f.write("\nRaw matrix returned by cv2.calibrateHandEye\n")
                f.write(format_matrix(T_raw) + "\n")

                f.write("\nRigid inverse of raw matrix\n")
                f.write(format_matrix(T_inverse) + "\n")

                raw_result = evaluate_candidate(
                    "RAW OUTPUT AS T_base_cam",
                    T_raw,
                    data,
                )

                inverse_result = evaluate_candidate(
                    "INVERSE OUTPUT AS T_base_cam",
                    T_inverse,
                    data,
                )

                write_candidate_section(
                    f,
                    raw_result,
                )

                write_candidate_section(
                    f,
                    inverse_result,
                )

                raw_score = candidate_score(raw_result)
                inverse_score = candidate_score(inverse_result)

                preferred = (
                    raw_result
                    if raw_score <= inverse_score
                    else inverse_result
                )

                f.write("\n")
                f.write("METHOD-LEVEL DECISION\n")
                f.write("---------------------\n")
                f.write(
                    f"  RAW consistency score     = "
                    f"{raw_score:.3f} mm-equivalent\n"
                )
                f.write(
                    f"  INVERSE consistency score = "
                    f"{inverse_score:.3f} mm-equivalent\n"
                )
                f.write(
                    f"  Better interpretation     = "
                    f"{preferred['candidate_name']}\n"
                )

                all_results.append(
                    {
                        "method_name": method_name,
                        "candidate": preferred,
                        "score": min(raw_score, inverse_score),
                    }
                )

                print(f"[OK] {method_name}")

            except Exception as e:
                f.write(f"\n[FAILED] {method_name}: {e}\n")
                print(f"[FAILED] {method_name}: {e}")

        f.write("\n\n")
        f.write("=" * 78 + "\n")
        f.write("FINAL RANKING ACROSS ALL METHODS\n")
        f.write("=" * 78 + "\n")

        if all_results:
            all_results.sort(
                key=lambda item: item["score"]
            )

            f.write(
                f"{'Rank':>6} "
                f"{'Method':<15} "
                f"{'Interpretation':<32} "
                f"{'Score [mm-eq]':>15} "
                f"{'Physical':>10}\n"
            )
            f.write("-" * 90 + "\n")

            for rank, item in enumerate(all_results, start=1):
                candidate = item["candidate"]

                f.write(
                    f"{rank:>6d} "
                    f"{item['method_name']:<15} "
                    f"{candidate['candidate_name']:<32} "
                    f"{item['score']:>15.3f} "
                    f"{'PASS' if candidate['physical_check']['overall'] else 'FAIL':>10}\n"
                )

            best = all_results[0]
            best_candidate = best["candidate"]

            f.write("\nBEST CONSISTENCY RESULT\n")
            f.write("-----------------------\n")
            f.write(f"Method         : {best['method_name']}\n")
            f.write(
                f"Interpretation : "
                f"{best_candidate['candidate_name']}\n"
            )
            f.write(
                f"Score          : "
                f"{best['score']:.3f} mm-equivalent\n"
            )
            f.write(
                f"Physical sanity: "
                f"{'PASS' if best_candidate['physical_check']['overall'] else 'FAIL'}\n"
            )

            f.write("\nBest T_base_cam candidate\n")
            f.write(
                format_matrix(
                    best_candidate["T_base_cam"]
                )
                + "\n"
            )

            f.write("\nBest recovered mean T_ee_board\n")
            f.write(
                format_matrix(
                    best_candidate["consistency"]["T_ee_board_mean"]
                )
                + "\n"
            )

        else:
            f.write("No OpenCV method completed successfully.\n")

    print("\nDone.")
    print(f"Readable report saved to:\n{REPORT_FILE}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

