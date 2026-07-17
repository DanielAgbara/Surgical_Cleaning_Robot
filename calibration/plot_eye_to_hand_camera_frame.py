import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path("/home/agbara-admin/Documents/Surgical_Cleaning_Robot")
DATA_DIR = ROOT / "data" / "eye_to_hand"

R_EE_BASE_FILE = DATA_DIR / "R_ee_base.json"
T_EE_BASE_FILE = DATA_DIR / "t_ee_base.json"

R_BASE_EE_FILE = DATA_DIR / "R_base_ee.json"
T_BASE_EE_FILE = DATA_DIR / "t_base_ee.json"

R_CAM_BOARD_FILE = DATA_DIR / "R_cam_board.json"
T_CAM_BOARD_FILE = DATA_DIR / "t_cam_board.json"

RESULTS_DIR = DATA_DIR / "opencv_results"

AVAILABLE_RESULTS = {
    "1": {
        "name": "handeye_tsai",
        "function": "cv2.calibrateHandEye",
    },
    "2": {
        "name": "handeye_park",
        "function": "cv2.calibrateHandEye",
    },
    "3": {
        "name": "handeye_horaud",
        "function": "cv2.calibrateHandEye",
    },
    "4": {
        "name": "handeye_andreff",
        "function": "cv2.calibrateHandEye",
    },
    "5": {
        "name": "handeye_daniilidis",
        "function": "cv2.calibrateHandEye",
    },
    "6": {
        "name": "robotworld_shah",
        "function": "cv2.calibrateRobotWorldHandEye",
    },
    "7": {
        "name": "robotworld_li",
        "function": "cv2.calibrateRobotWorldHandEye",
    },
    "8": {
        "name": "best",
        "function": "Solver-selected best result",
    },
}


def make_T(R, t):
    """
    Construct a homogeneous transform T_A_B.

    T_A_B maps coordinates from frame B into frame A.
    """

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)

    return T


def invert_T(T):
    """
    Rigid inverse of a homogeneous transform.
    """

    T = np.asarray(T, dtype=np.float64).reshape(4, 4)

    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T_inv


def rotation_angle_deg(R):
    """
    Return the magnitude of a relative rotation matrix in degrees.
    """

    R = np.asarray(R, dtype=np.float64).reshape(3, 3)

    cosine = (np.trace(R) - 1.0) / 2.0
    cosine = float(np.clip(cosine, -1.0, 1.0))

    return float(np.degrees(np.arccos(cosine)))


def validate_rotation_matrix(R, label, tolerance=1e-6):
    """
    Confirm that R is a proper 3D rotation matrix.

    A valid rotation must satisfy:

        R.T @ R = I
        det(R)   = +1
    """

    R = np.asarray(R, dtype=np.float64).reshape(3, 3)

    if not np.all(np.isfinite(R)):
        raise ValueError(f"{label} contains NaN or infinity.")

    orthogonality_error = np.linalg.norm(
        R.T @ R - np.eye(3),
        ord="fro",
    )

    determinant = float(np.linalg.det(R))

    if orthogonality_error > tolerance:
        raise ValueError(
            f"{label} is not orthogonal. "
            f"||R.T @ R - I|| = {orthogonality_error:.6e}"
        )

    if abs(determinant - 1.0) > tolerance:
        raise ValueError(
            f"{label} is not a proper rotation. "
            f"det(R) = {determinant:.9f}"
        )


def validate_transform(T, label):
    """
    Validate a 4x4 rigid homogeneous transform.
    """

    T = np.asarray(T, dtype=np.float64)

    if T.shape != (4, 4):
        raise ValueError(
            f"{label} must have shape (4, 4), received {T.shape}."
        )

    if not np.all(np.isfinite(T)):
        raise ValueError(f"{label} contains NaN or infinity.")

    expected_last_row = np.array(
        [0.0, 0.0, 0.0, 1.0],
        dtype=np.float64,
    )

    if not np.allclose(
        T[3],
        expected_last_row,
        atol=1e-8,
    ):
        raise ValueError(
            f"{label} has an invalid homogeneous last row: {T[3]}"
        )

    validate_rotation_matrix(
        T[:3, :3],
        f"{label} rotation",
    )

    return T


def load_json_array(path):
    """
    Load one JSON array and perform basic validation.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Required input file does not exist:\n{path}"
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {path}:\n{error}"
        ) from error

    array = np.asarray(data, dtype=np.float64)

    if array.size == 0:
        raise ValueError(f"{path.name} is empty.")

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{path.name} contains NaN or infinity."
        )

    return array


def load_rotation_list(path):
    """
    Load a list of 3x3 rotation matrices.
    """

    array = load_json_array(path)

    if array.ndim != 3 or array.shape[1:] != (3, 3):
        raise ValueError(
            f"{path.name} should have shape [N,3,3], "
            f"received {array.shape}."
        )

    rotations = []

    for sample_id, R in enumerate(array):
        validate_rotation_matrix(
            R,
            f"{path.name}, sample {sample_id}",
        )
        rotations.append(R.copy())

    return rotations


def load_translation_list(path):
    """
    Load translations and normalize every entry to shape (3,).
    """

    array = load_json_array(path)

    if array.ndim == 2 and array.shape[1] == 3:
        normalized = array.reshape(-1, 3)

    elif (
        array.ndim == 3
        and array.shape[1:] in ((3, 1), (1, 3))
    ):
        normalized = array.reshape(-1, 3)

    else:
        raise ValueError(
            f"{path.name} must have shape [N,3], "
            f"[N,3,1], or [N,1,3]. "
            f"Received {array.shape}."
        )

    return [
        translation.copy()
        for translation in normalized
    ]


def load_calibration_data():
    """
    Load the synchronized measured poses needed for plotting.

    Required measurements:

        T_base_ee[i]   = ^B T_E[i]
        T_cam_board[i] = ^C T_W[i]
    """

    R_base_ee = load_rotation_list(
        R_BASE_EE_FILE
    )
    t_base_ee = load_translation_list(
        T_BASE_EE_FILE
    )

    R_cam_board = load_rotation_list(
        R_CAM_BOARD_FILE
    )
    t_cam_board = load_translation_list(
        T_CAM_BOARD_FILE
    )

    lengths = {
        "R_base_ee": len(R_base_ee),
        "t_base_ee": len(t_base_ee),
        "R_cam_board": len(R_cam_board),
        "t_cam_board": len(t_cam_board),
    }

    if len(set(lengths.values())) != 1:
        details = ", ".join(
            f"{name}={count}"
            for name, count in lengths.items()
        )

        raise ValueError(
            "Collected sample files are not synchronized:\n"
            + details
        )

    number_of_samples = len(R_base_ee)

    if number_of_samples == 0:
        raise ValueError("No calibration samples were found.")

    T_base_ee = build_transform_list(
        R_base_ee,
        t_base_ee,
        label="T_base_ee",
    )

    T_cam_board = build_transform_list(
        R_cam_board,
        t_cam_board,
        label="T_cam_board",
    )

    return {
        "number_of_samples": number_of_samples,
        "T_base_ee": T_base_ee,
        "T_cam_board": T_cam_board,
    }


def load_solved_transforms(result_name):
    """
    Load T_base_camera and T_ee_board previously saved by
    solve_eyehand_opencv.py.
    """

    if result_name == "best":
        base_cam_file = (
            RESULTS_DIR
            / "T_base_camera_best.npy"
        )
        ee_board_file = (
            RESULTS_DIR
            / "T_ee_board_best.npy"
        )
    else:
        base_cam_file = (
            RESULTS_DIR
            / f"T_base_camera_{result_name}.npy"
        )
        ee_board_file = (
            RESULTS_DIR
            / f"T_ee_board_{result_name}.npy"
        )

    missing = [
        path
        for path in (base_cam_file, ee_board_file)
        if not path.exists()
    ]

    if missing:
        formatted = "\n".join(
            f"  {path}"
            for path in missing
        )

        raise FileNotFoundError(
            "The selected calibration result is incomplete.\n"
            "Run solve_eyehand_opencv.py first.\n"
            f"Missing files:\n{formatted}"
        )

    T_base_cam = np.asarray(
        np.load(base_cam_file),
        dtype=np.float64,
    )

    T_ee_board = np.asarray(
        np.load(ee_board_file),
        dtype=np.float64,
    )

    validate_transform(
        T_base_cam,
        "Loaded T_base_cam",
    )

    validate_transform(
        T_ee_board,
        "Loaded T_ee_board",
    )

    return {
        "T_base_cam": T_base_cam,
        "T_ee_board": T_ee_board,
        "base_cam_file": base_cam_file,
        "ee_board_file": ee_board_file,
    }




def build_transform_list(R_list, t_list, label):
    """
    Combine synchronized rotation and translation entries.
    """

    if len(R_list) != len(t_list):
        raise ValueError(
            f"{label} has {len(R_list)} rotations but "
            f"{len(t_list)} translations."
        )

    transforms = []

    for sample_id, (R, t) in enumerate(
        zip(R_list, t_list)
    ):
        T = make_T(R, t)

        validate_transform(
            T,
            f"{label}, sample {sample_id}",
        )

        transforms.append(T)

    return transforms

def ask_result_selection():
    """
    Ask which saved calibration result should be visualized.
    """

    print("\n" + "=" * 70)
    print("AVAILABLE SAVED CALIBRATION RESULTS")
    print("=" * 70)

    for key, entry in AVAILABLE_RESULTS.items():
        print(
            f"  {key}. "
            f"{entry['name']:<24s} "
            f"{entry['function']}"
        )

    while True:
        choice = input(
            "\nSelect result number, or q to quit: "
        ).strip().lower()

        if choice == "q":
            return None

        if choice in AVAILABLE_RESULTS:
            return AVAILABLE_RESULTS[choice]

        print(
            f"Invalid choice. Enter one of: "
            f"{', '.join(AVAILABLE_RESULTS.keys())}"
        )

def print_residual_summary(samples):
    """
    Print numerical differences between the two board reconstructions.
    """

    translation_mm = np.array(
        [
            sample["translation_error_m"] * 1000.0
            for sample in samples
        ],
        dtype=np.float64,
    )

    rotation_deg = np.array(
        [
            sample["rotation_error_deg"]
            for sample in samples
        ],
        dtype=np.float64,
    )

    print("\n" + "=" * 70)
    print("BOARD-FRAME RECONSTRUCTION RESIDUALS")
    print("=" * 70)

    print(
        "Translation [mm]: "
        f"mean={np.mean(translation_mm):.3f}, "
        f"median={np.median(translation_mm):.3f}, "
        f"RMS={np.sqrt(np.mean(translation_mm**2)):.3f}, "
        f"max={np.max(translation_mm):.3f}"
    )

    print(
        "Rotation [deg]:   "
        f"mean={np.mean(rotation_deg):.3f}, "
        f"median={np.median(rotation_deg):.3f}, "
        f"RMS={np.sqrt(np.mean(rotation_deg**2)):.3f}, "
        f"max={np.max(rotation_deg):.3f}"
    )

    print("\nPer-sample errors:")

    for sample in samples:
        print(
            f"  Sample {sample['sample_id']:02d}: "
            f"{sample['translation_error_m'] * 1000.0:8.3f} mm, "
            f"{sample['rotation_error_deg']:8.3f} deg"
        )


def reconstruct_frames_in_camera(
    data,
    T_base_cam,
    T_ee_board,
):
    """
    Express all frames in the fixed camera coordinate frame.

    Camera is the origin, so T_cam_cam = I.
    """

    T_cam_base = invert_T(T_base_cam)
    samples = []

    for sample_id, (T_base_ee, T_cam_board) in enumerate(
        zip(data["T_base_ee"], data["T_cam_board"])
    ):
        T_cam_ee = T_cam_base @ T_base_ee
        T_cam_board_robot = T_cam_ee @ T_ee_board
        T_cam_board_camera = T_cam_board.copy()

        board_residual = invert_T(T_cam_board_robot) @ T_cam_board_camera

        translation_error_m = float(
            np.linalg.norm(
                T_cam_board_robot[:3, 3] - T_cam_board_camera[:3, 3]
            )
        )
        rotation_error_deg = rotation_angle_deg(board_residual[:3, :3])

        samples.append({
            "sample_id": sample_id,
            "T_cam_cam": np.eye(4, dtype=np.float64),
            "T_cam_base": T_cam_base,
            "T_cam_ee": T_cam_ee,
            "T_cam_board_robot": T_cam_board_robot,
            "T_cam_board_camera": T_cam_board_camera,
            "T_base_ee": T_base_ee,
            "T_cam_board_measured": T_cam_board,
            "translation_error_m": translation_error_m,
            "rotation_error_deg": rotation_error_deg,
        })

    return samples

def draw_frame(
    ax,
    T,
    axis_length,
    label=None,
    alpha=1.0,
    linewidth=1.5,
    linestyle="solid",
    draw_axis_labels=False,
):
    """
    Draw one 3D coordinate frame.

    Standard axis colors:
        +X = red
        +Y = green
        +Z = blue
    """

    T = validate_transform(
        np.asarray(T, dtype=np.float64),
        label or "Plot transform",
    )

    origin = T[:3, 3]
    R = T[:3, :3]

    axes = (
        ("x", R[:, 0], "red"),
        ("y", R[:, 1], "green"),
        ("z", R[:, 2], "blue"),
    )

    for axis_name, direction, color in axes:
        endpoint = (
            origin
            + axis_length * direction
        )

        ax.plot(
            [origin[0], endpoint[0]],
            [origin[1], endpoint[1]],
            [origin[2], endpoint[2]],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            linestyle=linestyle,
        )

        if draw_axis_labels:
            ax.text(
                endpoint[0],
                endpoint[1],
                endpoint[2],
                f"+{axis_name.upper()}",
                color=color,
                fontsize=7,
            )

    if label:
        ax.text(
            origin[0],
            origin[1],
            origin[2],
            label,
            fontsize=8,
            alpha=alpha,
        )

    return origin


def set_axes_equal(ax, points, padding_fraction=0.10):
    """
    Give X, Y, and Z the same physical scale.
    """

    points = np.asarray(
        points,
        dtype=np.float64,
    ).reshape(-1, 3)

    if len(points) == 0:
        return

    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)

    center = 0.5 * (minimum + maximum)
    full_range = maximum - minimum

    radius = 0.5 * float(
        np.max(full_range)
    )

    if radius < 1e-6:
        radius = 0.1

    radius *= 1.0 + padding_fraction

    ax.set_xlim(
        center[0] - radius,
        center[0] + radius,
    )
    ax.set_ylim(
        center[1] - radius,
        center[1] + radius,
    )
    ax.set_zlim(
        center[2] - radius,
        center[2] + radius,
    )

    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def get_key_sample_indices(number_of_samples):
    """Return the first, middle, and last sample indices."""
    if number_of_samples <= 0:
        raise ValueError("At least one sample is required for plotting.")
    indices = [0, number_of_samples // 2, number_of_samples - 1]
    return list(dict.fromkeys(indices))


def plot_all_frames(samples, function_name, method_name):
    """Plot all trajectories in the fixed camera coordinate frame."""

    if not samples:
        raise ValueError("No reconstructed samples to plot.")

    fig = plt.figure(figsize=(15, 11))
    ax = fig.add_subplot(111, projection="3d")

    sample_axis_length = 0.055
    fixed_axis_length = 0.10

    ee_path_color = "tab:purple"
    board_robot_path_color = "tab:orange"
    board_camera_path_color = "tab:cyan"
    disagreement_color = "gray"

    all_points = []

    camera_origin = draw_frame(
        ax, np.eye(4), axis_length=fixed_axis_length,
        label="CAMERA", alpha=1.0, linewidth=3.0,
        draw_axis_labels=True,
    )
    all_points.append(camera_origin)

    T_cam_base = samples[0]["T_cam_base"]
    base_origin = draw_frame(
        ax, T_cam_base, axis_length=fixed_axis_length,
        label="BASE", alpha=1.0, linewidth=3.0,
        draw_axis_labels=True,
    )
    all_points.append(base_origin)

    ee_origins = np.asarray([s["T_cam_ee"][:3, 3] for s in samples])
    board_robot_origins = np.asarray([s["T_cam_board_robot"][:3, 3] for s in samples])
    board_camera_origins = np.asarray([s["T_cam_board_camera"][:3, 3] for s in samples])

    all_points.extend(ee_origins)
    all_points.extend(board_robot_origins)
    all_points.extend(board_camera_origins)

    ax.plot(
        ee_origins[:, 0], ee_origins[:, 1], ee_origins[:, 2],
        color=ee_path_color, marker="o", markersize=5,
        linewidth=1.8, alpha=0.85, label="EE path in camera frame",
    )
    ax.plot(
        board_robot_origins[:, 0], board_robot_origins[:, 1], board_robot_origins[:, 2],
        color=board_robot_path_color, marker="o", markersize=5,
        linewidth=2.0, alpha=0.90, label="Board path: robot chain",
    )
    ax.plot(
        board_camera_origins[:, 0], board_camera_origins[:, 1], board_camera_origins[:, 2],
        color=board_camera_path_color, marker="x", markersize=6,
        linewidth=2.0, linestyle="dashed", alpha=0.90,
        label="Board path: solvePnP",
    )

    key_indices = get_key_sample_indices(len(samples))

    for index in key_indices:
        sample = samples[index]
        sample_id = sample["sample_id"]

        draw_frame(
            ax, sample["T_cam_ee"], axis_length=sample_axis_length,
            label=f"EE{sample_id}", alpha=1.0, linewidth=2.2,
        )
        draw_frame(
            ax, sample["T_cam_board_robot"], axis_length=sample_axis_length,
            label=f"WR{sample_id}", alpha=1.0, linewidth=2.2,
            linestyle="solid",
        )
        draw_frame(
            ax, sample["T_cam_board_camera"], axis_length=sample_axis_length,
            label=f"WC{sample_id}", alpha=1.0, linewidth=2.2,
            linestyle="dashed",
        )

        r = sample["T_cam_board_robot"][:3, 3]
        c = sample["T_cam_board_camera"][:3, 3]
        ax.plot(
            [r[0], c[0]], [r[1], c[1]], [r[2], c[2]],
            color=disagreement_color, linewidth=1.4,
            linestyle="dotted", alpha=0.85,
        )

    ax.set_xlabel("Camera X [m]")
    ax.set_ylabel("Camera Y [m]")
    ax.set_zlabel("Camera Z [m]")

    displayed_ids = ", ".join(str(samples[i]["sample_id"]) for i in key_indices)
    ax.set_title(
        "Eye-to-Hand Frames in Camera Coordinates\n"
        f"{function_name} — {method_name}\n"
        f"Camera is the origin | Full frames: {displayed_ids}"
    )

    set_axes_equal(ax, all_points, padding_fraction=0.12)
    ax.legend(loc="upper left")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    """
    Load measured poses and solved transforms, reconstruct all
    frames in robot-base coordinates, and display the plot.
    """

    print("=" * 70)
    print("EYE-TO-HAND CAMERA-FRAME PLOTTER")
    print("=" * 70)
    print(f"Data directory:    {DATA_DIR}")
    print(f"Results directory: {RESULTS_DIR}")

    try:
        selection = ask_result_selection()

        if selection is None:
            print("Plotting cancelled.")
            return 0

        result_name = selection["name"]
        function_name = selection["function"]

        data = load_calibration_data()

        solved = load_solved_transforms(
            result_name
        )

        T_base_cam = solved["T_base_cam"]
        T_ee_board = solved["T_ee_board"]

        samples = reconstruct_frames_in_camera(
            data=data,
            T_base_cam=T_base_cam,
            T_ee_board=T_ee_board,
        )

    except Exception as error:
        print(f"\n[ERROR]\n{error}")
        return 1

    print("\nLoaded result:")
    print(f"  Function: {function_name}")
    print(f"  Method:   {result_name}")
    print(
        f"  Samples:  "
        f"{data['number_of_samples']}"
    )

    print("\nLoaded files:")
    print(
        f"  T_base_cam: {solved['base_cam_file']}"
    )
    print(
        f"  T_ee_board: {solved['ee_board_file']}"
    )

    print("\nT_base_cam = ^B T_C:")
    print(
        np.array2string(
            T_base_cam,
            precision=7,
            suppress_small=True,
        )
    )


    T_cam_base = invert_T(T_base_cam)

    print("\nT_cam_base = ^C T_B used for the camera-frame plot:")
    print(
        np.array2string(
            T_cam_base,
            precision=7,
            suppress_small=True,
        )
    )

    print("\nT_ee_board = ^E T_W:")
    print(
        np.array2string(
            T_ee_board,
            precision=7,
            suppress_small=True,
        )
    )

    print_residual_summary(samples)

    plot_all_frames(
        samples=samples,
        function_name=function_name,
        method_name=result_name,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

