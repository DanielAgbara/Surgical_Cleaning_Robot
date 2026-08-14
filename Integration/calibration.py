"""
Main calibration program for the surgical cleaning robot.
This script contains the complete calibration workflow:
- Creating calibration trajectories
- Collecting calibration data
- Solving the calibration
- Validating the calibration using object detection
- Allowing the user to select which calibration operation to run
"""

# ============================================================
# Imports
# ============================================================

import argparse
import curses
import json
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2, robot
import numpy as np

from camera import (
    open_zed,
    get_image,
    get_zed_left_intrinsics_rectified,
)

# ============================================================
# ChArUco configuration
# ============================================================
@dataclass(frozen=True)
class CharucoBoardConfig:
    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float

    dictionary_id: int = cv2.aruco.DICT_4X4_50
    marker_ids: Optional[tuple[int, ...]] = None
    legacy_pattern: bool = False

    min_markers_per_corner: int = 2
    try_refine_markers: bool = True
    marker_corner_refinement: int = (
        cv2.aruco.CORNER_REFINE_NONE
    )

    @property
    def max_charuco_corners(self) -> int:
        return (
            (self.squares_x - 1)
            * (self.squares_y - 1)
        )

    @property
    def axis_length_m(self) -> float:
        board_short_side = min(
            self.squares_x,
            self.squares_y,
        ) * self.square_length_m

        return 0.25 * board_short_side

    def validate(self) -> None:
        if self.squares_x < 2 or self.squares_y < 2:
            raise ValueError(
                "ChArUco board dimensions must be at least 2x2."
            )

        if self.square_length_m <= 0:
            raise ValueError(
                "square_length_m must be positive."
            )

        if self.marker_length_m <= 0:
            raise ValueError(
                "marker_length_m must be positive."
            )

        if self.marker_length_m >= self.square_length_m:
            raise ValueError(
                "marker_length_m must be smaller than "
                "square_length_m."
            )
        
DEFAULT_CHARUCO_CONFIG = CharucoBoardConfig(
    squares_x=4,
    squares_y=4,
    square_length_m=0.0482,
    marker_length_m=0.0361,
    dictionary_id=cv2.aruco.DICT_4X4_50,
)

HUMAN_TOOL_CHARUCO_CONFIG = CharucoBoardConfig(
    squares_x=3,
    squares_y=3,
    square_length_m=0.032,
    marker_length_m=0.0258,
    dictionary_id=cv2.aruco.DICT_4X4_50,
)
        

def create_charuco_detector(
    config: CharucoBoardConfig,
    camera_matrix: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
):
    """
    Create an OpenCV ChArUco board and detector.

    Parameters
    ----------
    config
        Description of the physical printed board.

    camera_matrix
        Optional 3x3 camera intrinsic matrix.

    dist_coeffs
        Optional distortion coefficients. For a rectified ZED image,
        these will normally be zeros.

    Returns
    -------
    board
        cv2.aruco.CharucoBoard object.

    detector
        cv2.aruco.CharucoDetector object.
    """

    config.validate()

    dictionary = cv2.aruco.getPredefinedDictionary(
        config.dictionary_id
    )

    board_size = (
        config.squares_x,
        config.squares_y,
    )

    # First create a default board so that we can determine
    # how many ArUco markers this board geometry requires.
    default_board = cv2.aruco.CharucoBoard(
        board_size,
        config.square_length_m,
        config.marker_length_m,
        dictionary,
    )

    if config.marker_ids is None:
        board = default_board

    else:
        marker_ids = np.asarray(
            config.marker_ids,
            dtype=np.int32,
        ).reshape(-1, 1)

        expected_marker_count = len(
            default_board.getIds()
        )

        if len(marker_ids) != expected_marker_count:
            raise ValueError(
                f"This board requires {expected_marker_count} "
                f"marker IDs, but {len(marker_ids)} were provided."
            )

        if len(np.unique(marker_ids)) != len(marker_ids):
            raise ValueError(
                "All custom marker IDs must be unique."
            )

        board = cv2.aruco.CharucoBoard(
            board_size,
            config.square_length_m,
            config.marker_length_m,
            dictionary,
            marker_ids,
        )

    # Compatibility for boards generated before OpenCV 4.6.
    board.setLegacyPattern(config.legacy_pattern)

    detector_parameters = cv2.aruco.DetectorParameters()

    detector_parameters.cornerRefinementMethod = (
        config.marker_corner_refinement
    )

    charuco_parameters = cv2.aruco.CharucoParameters()

    charuco_parameters.minMarkers = (
        config.min_markers_per_corner
    )

    charuco_parameters.tryRefineMarkers = (
        config.try_refine_markers
    )

    # Providing camera intrinsics improves interpolation,
    # especially near the edges of the image.
    if camera_matrix is not None:
        camera_matrix = np.asarray(
            camera_matrix,
            dtype=np.float64,
        ).reshape(3, 3)

        if dist_coeffs is None:
            dist_coeffs = np.zeros(
                (5, 1),
                dtype=np.float64,
            )
        else:
            dist_coeffs = np.asarray(
                dist_coeffs,
                dtype=np.float64,
            ).reshape(-1, 1)

        charuco_parameters.cameraMatrix = camera_matrix
        charuco_parameters.distCoeffs = dist_coeffs

    detector = cv2.aruco.CharucoDetector(
        board,
        charuco_parameters,
        detector_parameters,
    )

    print(
        "[INFO] ChArUco detector created: "
        f"{config.squares_x}x{config.squares_y}, "
        f"{config.max_charuco_corners} corners"
    )

    return board, detector

#Detection
@dataclass
class CharucoDetection:
    charuco_corners: np.ndarray | None
    charuco_ids: np.ndarray | None

    marker_corners: tuple
    marker_ids: np.ndarray | None

    num_charuco_corners: int
    num_markers: int

    all_corners_detected: bool
    corners_are_collinear: bool
    can_estimate_pose: bool

def detect_charuco_board(
    image_bgr: np.ndarray,
    board,
    detector,
    min_corners_for_pose: int = 4,
) -> CharucoDetection:
    """
    Detect ArUco markers and ChArUco corners in an OpenCV image.

    Partial board detection is allowed. The entire board does not
    have to be visible.

    Parameters
    ----------
    image_bgr
        OpenCV BGR image.

    board
        cv2.aruco.CharucoBoard returned by
        create_charuco_detector().

    detector
        cv2.aruco.CharucoDetector returned by
        create_charuco_detector().

    min_corners_for_pose
        Minimum number of non-collinear ChArUco corners required
        before pose estimation is attempted.

    Returns
    -------
    CharucoDetection
        Detection information.
    """

    if image_bgr is None:
        raise ValueError("image_bgr cannot be None.")

    if not isinstance(image_bgr, np.ndarray):
        raise TypeError(
            "image_bgr must be a NumPy array."
        )

    if image_bgr.size == 0:
        raise ValueError("image_bgr is empty.")

    if image_bgr.ndim == 3:
        gray = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2GRAY,
        )
    elif image_bgr.ndim == 2:
        gray = image_bgr
    else:
        raise ValueError(
            "Image must be either grayscale or BGR."
        )

    (
        charuco_corners,
        charuco_ids,
        marker_corners,
        marker_ids,
    ) = detector.detectBoard(gray)

    if marker_corners is None:
        marker_corners = tuple()
    else:
        marker_corners = tuple(marker_corners)

    num_markers = (
        0
        if marker_ids is None
        else len(marker_ids)
    )

    num_charuco_corners = (
        0
        if charuco_ids is None
        else len(charuco_ids)
    )

    total_board_corners = len(
        board.getChessboardCorners()
    )

    all_corners_detected = (
        num_charuco_corners == total_board_corners
    )

    # solvePnP cannot determine a reliable pose when all
    # detected points lie on one straight line.
    if charuco_ids is None:
        corners_are_collinear = True
    else:
        corners_are_collinear = bool(
            board.checkCharucoCornersCollinear(
                charuco_ids
            )
        )

    can_estimate_pose = (
        num_charuco_corners >= min_corners_for_pose
        and not corners_are_collinear
    )

    return CharucoDetection(
        charuco_corners=charuco_corners,
        charuco_ids=charuco_ids,
        marker_corners=marker_corners,
        marker_ids=marker_ids,
        num_charuco_corners=num_charuco_corners,
        num_markers=num_markers,
        all_corners_detected=all_corners_detected,
        corners_are_collinear=corners_are_collinear,
        can_estimate_pose=can_estimate_pose,
    )

#Estimation
@dataclass
class CharucoPose:
    rotation_matrix: np.ndarray
    rotation_vector: np.ndarray
    translation_vector: np.ndarray

    T_camera_board: np.ndarray

    mean_reprojection_error_px: float
    max_reprojection_error_px: float


def estimate_charuco_pose(
    detection: CharucoDetection,
    board,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
) -> CharucoPose | None:
    """
    Estimate the board pose relative to the camera.

    Returns
    -------
    CharucoPose | None
        T_camera_board maps board-frame points into the
        camera coordinate frame:

            p_camera = T_camera_board @ p_board
    """

    if not detection.can_estimate_pose:
        return None

    camera_matrix = np.asarray(
        camera_matrix,
        dtype=np.float64,
    ).reshape(3, 3)

    if dist_coeffs is None:
        dist_coeffs = np.zeros(
            (5, 1),
            dtype=np.float64,
        )
    else:
        dist_coeffs = np.asarray(
            dist_coeffs,
            dtype=np.float64,
        ).reshape(-1, 1)

    object_points, image_points = board.matchImagePoints(
        detection.charuco_corners,
        detection.charuco_ids,
    )

    object_points = np.asarray(
        object_points,
        dtype=np.float64,
    )

    image_points = np.asarray(
        image_points,
        dtype=np.float64,
    )

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE,
    )

    if not success:
        return None

    # Nonlinear refinement after the planar IPPE estimate.
    rvec, tvec = cv2.solvePnPRefineLM(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec,
    )

    rotation_matrix, _ = cv2.Rodrigues(rvec)

    T_camera_board = np.eye(
        4,
        dtype=np.float64,
    )

    T_camera_board[:3, :3] = rotation_matrix
    T_camera_board[:3, 3] = tvec.reshape(3)

    projected_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    measured_points = image_points.reshape(-1, 2)
    projected_points = projected_points.reshape(-1, 2)

    reprojection_errors = np.linalg.norm(
        measured_points - projected_points,
        axis=1,
    )

    return CharucoPose(
        rotation_matrix=rotation_matrix,
        rotation_vector=rvec.reshape(3),
        translation_vector=tvec.reshape(3),
        T_camera_board=T_camera_board,
        mean_reprojection_error_px=float(
            np.mean(reprojection_errors)
        ),
        max_reprojection_error_px=float(
            np.max(reprojection_errors)
        ),
    )

def generateCharucoPDF(
    config=DEFAULT_CHARUCO_CONFIG,
    name=None,
):
    """Generate a printable PDF from a ChArUco board configuration.

    Returns
    -------
    pathlib.Path
        Absolute path to the generated PDF. Print it at 100% (actual size).

    Notes
    -----
    All board geometry, marker sizing, dictionary selection, and legacy
    pattern behavior are taken from ``config``.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to create the PDF: pip install Pillow"
        ) from exc

    config.validate()

    square_length_mm = config.square_length_m * 1000.0
    marker_length_mm = config.marker_length_m * 1000.0
    width_mm = config.squares_x * square_length_mm
    height_mm = config.squares_y * square_length_mm

    dictionary = cv2.aruco.getPredefinedDictionary(
        config.dictionary_id
    )
    board = cv2.aruco.CharucoBoard(
        (config.squares_x, config.squares_y),
        square_length_mm,
        marker_length_mm,
        dictionary,
    )
    board.setLegacyPattern(config.legacy_pattern)

    dpi = 300
    image_size = (
        int(round(width_mm / 25.4 * dpi)),
        int(round(height_mm / 25.4 * dpi)),
    )
    if min(image_size) < 1:
        raise ValueError("boardSize is too small to render at 300 DPI.")

    if hasattr(board, "generateImage"):
        board_image = board.generateImage(
            image_size, marginSize=0, borderBits=1
        )
    else:  # OpenCV < 4.7 compatibility
        board_image = board.draw(image_size, marginSize=0, borderBits=1)

    size_label = f"{width_mm:g}x{height_mm:g}mm"
    board_directory = Path(__file__).resolve().parent / "data" / "charuco_boards"
    board_directory.mkdir(parents=True, exist_ok=True)
    name_prefix = "" if name is None else f"{name}_"
    pdf_path = (
        board_directory
        / f"charuco_{name_prefix}{config.squares_x}x{config.squares_y}_{size_label}.pdf"
    )
    Image.fromarray(board_image).convert("L").save(
        pdf_path, "PDF", resolution=dpi
    )

    print(f"Saved printable ChArUco board: {pdf_path}")
    print("Print at 100% / actual size; disable 'fit to page'.")
    print(f"Verify that each printed square is {square_length_mm:.3f} mm.")
    return pdf_path

# ======================================================
# Trajectory files and shared calibration paths
# ======================================================

INTEGRATION_DIR = Path(__file__).resolve().parent
INTEGRATION_DATA_DIR = INTEGRATION_DIR / "data"
EYE_HAND_DIR = INTEGRATION_DATA_DIR / "eyehand"
TRAJECTORY_FILE = EYE_HAND_DIR / "calibration_trajectory.json"
LITE6_TRAJECTORY_FILE = EYE_HAND_DIR / "lite6_calibration_trajectory.json"
R_BASE_EE_FILE = EYE_HAND_DIR / "R_base_ee.json"
T_BASE_EE_FILE = EYE_HAND_DIR / "t_base_ee.json"
R_CAM_BOARD_FILE = EYE_HAND_DIR / "R_cam_board.json"
T_CAM_BOARD_FILE = EYE_HAND_DIR / "t_cam_board.json"
SOLUTION_FILE = EYE_HAND_DIR / "eye_hand_calibration.json"
TSAI_SOLUTION_FILE = EYE_HAND_DIR / "eye_hand_calibration_tsai.json"
COLLECTION_WINDOW_NAME = "Collect eye-hand calibration"

JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
LITE6_POSE_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")
LITE6_JOINT_NAMES = (
    "joint_1_deg",
    "joint_2_deg",
    "joint_3_deg",
    "joint_4_deg",
    "joint_5_deg",
    "joint_6_deg",
)

# OpenCV arrow-key codes vary with the active GUI backend.
LEFT_ARROW_KEYS = {81, 2424832, 65361}
RIGHT_ARROW_KEYS = {83, 2555904, 65363}


def write_json(path, value):
    """Write JSON atomically so an interrupted write does not corrupt a file."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
        file.write("\n")
    temporary.replace(path)
    return path


def numeric_json(description, data):
    """Build a JSON payload whose data section contains numbers only."""

    if not isinstance(description, dict):
        raise TypeError("description must be a dictionary.")
    array = np.asarray(data, dtype=float)
    if array.ndim == 0:
        raise ValueError("JSON data must be an array, not a scalar.")
    if not np.all(np.isfinite(array)):
        raise ValueError("JSON data must contain finite numbers.")
    return {
        "description": description,
        "data": array.tolist(),
    }


def read_json_data(path):
    """Read numeric data from the new format or a legacy bare JSON array."""

    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, dict) and "data" in payload:
        description = payload.get("description", {})
        if not isinstance(description, dict):
            raise ValueError(f"{path}: description must be an object.")
        try:
            numeric_data = np.asarray(payload["data"], dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}: data must contain numbers only.") from error
        if not np.all(np.isfinite(numeric_data)):
            raise ValueError(f"{path}: data contains non-finite numbers.")
        return payload["data"], description
    return payload, {}


def numeric_rows_to_points(data, names):
    """Convert an N-column numeric matrix to internal named point records."""

    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data

    rows = np.asarray(data, dtype=float)
    expected_columns = len(names)
    if rows.ndim != 2 or rows.shape[1] != expected_columns:
        raise ValueError(
            f"Trajectory data must have shape (N, {expected_columns})."
        )
    if not np.all(np.isfinite(rows)):
        raise ValueError("Trajectory data contains non-finite numbers.")
    return [
        {
            name: float(row[index])
            for index, name in enumerate(names)
        }
        for row in rows
    ]


def validate_trajectory(points):
    """Validate and normalize a list of six-joint robot poses."""
    if not isinstance(points, list) or not points:
        raise ValueError("Trajectory must be a non-empty JSON list.")

    clean_points = []
    lower = np.asarray(robot.theta_min_robot_deg, dtype=float)
    upper = np.asarray(robot.theta_max_robot_deg, dtype=float)

    for point_index, point in enumerate(points):
        if not isinstance(point, dict):
            raise TypeError(f"Point {point_index} must be a dictionary.")

        if set(point) != set(JOINT_NAMES):
            raise ValueError(
                f"Point {point_index} must contain exactly: {JOINT_NAMES}"
            )

        values = np.array([point[name] for name in JOINT_NAMES], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Point {point_index} contains a non-finite value.")
        if np.any(values < lower) or np.any(values > upper):
            raise ValueError(f"Point {point_index} exceeds the robot limits.")

        clean_points.append(
            {name: float(values[index]) for index, name in enumerate(JOINT_NAMES)}
        )

    return clean_points


def save_trajectory_file(points, path=TRAJECTORY_FILE):
    """Save ordered joint poses as a numeric matrix."""
    path = Path(path).resolve()
    if path.suffix.lower() != ".json":
        raise ValueError("Trajectory path must end in .json.")
    points = validate_trajectory(points)
    data = [
        [point[name] for name in JOINT_NAMES]
        for point in points
    ]
    payload = numeric_json(
        {
            "summary": "SO-101 calibration joint trajectory.",
            "columns": list(JOINT_NAMES),
            "units": "degrees",
        },
        data,
    )
    write_json(path, payload)
    print(f"[INFO] Saved {len(points)} points to {path}")
    return path


def load_trajectory_file(path=TRAJECTORY_FILE):
    """Load a numeric or legacy SO-101 trajectory."""
    path = Path(path).resolve()
    data, _ = read_json_data(path)
    return validate_trajectory(numeric_rows_to_points(data, JOINT_NAMES))


def validate_lite6_trajectory(points):
    """Validate absolute Lite 6 TCP poses in millimetres and degrees."""
    if not isinstance(points, list) or not points:
        raise ValueError("Trajectory must be a non-empty JSON list.")

    clean_points = []
    for point_index, point in enumerate(points):
        if not isinstance(point, dict) or set(point) != set(LITE6_POSE_NAMES):
            raise ValueError(
                f"Point {point_index} must contain exactly: {LITE6_POSE_NAMES}"
            )
        values = np.asarray(
            [point[name] for name in LITE6_POSE_NAMES], dtype=float
        )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Point {point_index} contains a non-finite value.")
        clean_points.append(
            {
                name: float(values[index])
                for index, name in enumerate(LITE6_POSE_NAMES)
            }
        )
    return clean_points


def save_lite6_trajectory_file(
    points,
    path=LITE6_TRAJECTORY_FILE,
    group_ranges=None,
    joint_limits_checked=False,
):
    """Save absolute Lite 6 TCP poses as a numeric matrix."""
    path = Path(path).resolve()
    if path.suffix.lower() != ".json":
        raise ValueError("Trajectory path must end in .json.")
    points = validate_lite6_trajectory(points)
    data = [
        [point[name] for name in LITE6_POSE_NAMES]
        for point in points
    ]
    description = {
        "summary": "Lite 6 absolute TCP calibration trajectory.",
        "columns": list(LITE6_POSE_NAMES),
        "units": [
            "millimeters",
            "millimeters",
            "millimeters",
            "degrees",
            "degrees",
            "degrees",
        ],
        "lite6_joint_min_deg": list(robot.Lite6.JOINT_MIN_DEG),
        "lite6_joint_max_deg": list(robot.Lite6.JOINT_MAX_DEG),
        "all_poses_ik_joint_limit_checked": bool(joint_limits_checked),
    }
    if group_ranges is not None:
        description["interpolation_groups"] = group_ranges
        description["interpolation_crosses_groups"] = False
    payload = numeric_json(description, data)
    write_json(path, payload)
    print(f"[INFO] Saved {len(points)} Lite 6 poses to {path}")
    return path


def load_lite6_trajectory_file(path=LITE6_TRAJECTORY_FILE):
    """Load and validate an absolute Lite 6 TCP trajectory."""
    path = Path(path).resolve()
    data, _ = read_json_data(path)
    return validate_lite6_trajectory(
        numeric_rows_to_points(data, LITE6_POSE_NAMES)
    )


def interpolate_lite6_trajectory(points, points_between=0):
    """Insert evenly spaced TCP poses between each captured Lite 6 pose.

    XYZ is interpolated linearly in millimetres. Euler angles use the shortest
    signed angular difference in degrees, avoiding unnecessary 360-degree
    rotations across the wrap boundary.
    """
    points = validate_lite6_trajectory(points)
    if (
        isinstance(points_between, bool)
        or not isinstance(points_between, (int, np.integer))
        or points_between < 0
    ):
        raise ValueError("points_between must be a non-negative integer.")
    if points_between == 0 or len(points) == 1:
        return points

    values = np.asarray(
        [[point[name] for name in LITE6_POSE_NAMES] for point in points],
        dtype=float,
    )
    interpolated = []
    for start, end in zip(values[:-1], values[1:]):
        interpolated.append(start)
        delta = end - start
        delta[3:] = (delta[3:] + 180.0) % 360.0 - 180.0
        for step in range(1, points_between + 1):
            fraction = step / (points_between + 1)
            interpolated.append(start + fraction * delta)
    interpolated.append(values[-1])

    return [
        {
            name: float(pose[index])
            for index, name in enumerate(LITE6_POSE_NAMES)
        }
        for pose in interpolated
    ]


def interpolate_lite6_groups(
    groups,
    points_between=0,
    requested_points=None,
):
    """Interpolate only inside explicit visibility-safe pose groups."""
    clean_groups = [
        validate_lite6_trajectory(group)
        for group in groups
        if group
    ]
    if not clean_groups:
        raise ValueError("At least one non-empty group is required.")
    if points_between < 0:
        raise ValueError("points_between cannot be negative.")

    captured_count = sum(len(group) for group in clean_groups)
    edges = [
        (group_index, edge_index)
        for group_index, group in enumerate(clean_groups)
        for edge_index in range(len(group) - 1)
    ]

    if requested_points is not None:
        if requested_points < captured_count:
            raise ValueError(
                f"--points must be at least the {captured_count} captured poses."
            )
        extra_points = requested_points - captured_count
        if extra_points and not edges:
            raise ValueError(
                "Interpolation requires at least two poses in one group."
            )
        base, remainder = divmod(extra_points, len(edges)) if edges else (0, 0)
        edge_counts = {
            edge: base + (index < remainder)
            for index, edge in enumerate(edges)
        }
    else:
        edge_counts = {edge: int(points_between) for edge in edges}

    generated = []
    group_ranges = []
    for group_index, group in enumerate(clean_groups):
        start_index = len(generated)
        if len(group) == 1:
            generated.extend(group)
        else:
            for edge_index, (start, end) in enumerate(zip(group[:-1], group[1:])):
                segment = interpolate_lite6_trajectory(
                    [start, end],
                    edge_counts[(group_index, edge_index)],
                )
                generated.extend(segment[:-1])
            generated.append(group[-1])
        group_ranges.append(
            {
                "group_index": group_index,
                "captured_pose_count": len(group),
                "trajectory_start_index": start_index,
                "trajectory_end_index": len(generated) - 1,
            }
        )

    return generated, group_ranges


def validate_lite6_joint_trajectory(points, clip=False):
    """Validate Lite 6 joint records, optionally clipping every joint."""
    if not isinstance(points, list) or not points:
        raise ValueError("Lite 6 joint trajectory must be a non-empty list.")
    clean = []
    for index, point in enumerate(points):
        if not isinstance(point, dict) or set(point) != set(LITE6_JOINT_NAMES):
            raise ValueError(
                f"Joint point {index} must contain exactly: {LITE6_JOINT_NAMES}"
            )
        angles = np.asarray(
            [point[name] for name in LITE6_JOINT_NAMES], dtype=float
        )
        angles = (
            robot.Lite6.clip_joint_angles_deg(angles)
            if clip
            else robot.Lite6.validate_joint_angles_deg(
                angles, context=f"Lite 6 trajectory point {index}"
            )
        )
        clean.append(
            {
                name: float(angles[joint_index])
                for joint_index, name in enumerate(LITE6_JOINT_NAMES)
            }
        )
    return clean


def interpolate_lite6_joint_groups(
    groups,
    points_between=0,
    requested_points=None,
):
    """Linearly interpolate clipped joint angles only within each group."""
    clean_groups = [
        validate_lite6_joint_trajectory(group, clip=True)
        for group in groups
        if group
    ]
    if not clean_groups:
        raise ValueError("At least one non-empty joint group is required.")

    captured_count = sum(len(group) for group in clean_groups)
    edges = [
        (group_index, edge_index)
        for group_index, group in enumerate(clean_groups)
        for edge_index in range(len(group) - 1)
    ]
    if requested_points is not None:
        if requested_points < captured_count:
            raise ValueError(
                f"--points must be at least the {captured_count} captured poses."
            )
        extra = requested_points - captured_count
        if extra and not edges:
            raise ValueError(
                "Interpolation requires at least two poses in one group."
            )
        base, remainder = divmod(extra, len(edges)) if edges else (0, 0)
        edge_counts = {
            edge: base + (edge_number < remainder)
            for edge_number, edge in enumerate(edges)
        }
    else:
        if points_between < 0:
            raise ValueError("points_between cannot be negative.")
        edge_counts = {edge: int(points_between) for edge in edges}

    generated = []
    group_ranges = []
    for group_index, group in enumerate(clean_groups):
        start_index = len(generated)
        for edge_index, (start, end) in enumerate(zip(group[:-1], group[1:])):
            start_angles = np.asarray(
                [start[name] for name in LITE6_JOINT_NAMES], dtype=float
            )
            end_angles = np.asarray(
                [end[name] for name in LITE6_JOINT_NAMES], dtype=float
            )
            generated.append(start)
            number_between = edge_counts[(group_index, edge_index)]
            for step in range(1, number_between + 1):
                fraction = step / (number_between + 1)
                angles = robot.Lite6.clip_joint_angles_deg(
                    start_angles + fraction * (end_angles - start_angles)
                )
                generated.append(
                    {
                        name: float(angles[index])
                        for index, name in enumerate(LITE6_JOINT_NAMES)
                    }
                )
        generated.append(group[-1])
        group_ranges.append(
            {
                "group_index": group_index,
                "captured_pose_count": len(group),
                "trajectory_start_index": start_index,
                "trajectory_end_index": len(generated) - 1,
            }
        )
    return validate_lite6_joint_trajectory(generated), group_ranges


def save_lite6_joint_trajectory_file(
    points,
    path=LITE6_TRAJECTORY_FILE,
    group_ranges=None,
):
    """Save a clipped Lite 6 joint-space calibration trajectory."""
    path = Path(path).resolve()
    if path.suffix.lower() != ".json":
        raise ValueError("Trajectory path must end in .json.")
    points = validate_lite6_joint_trajectory(points, clip=True)
    data = [[point[name] for name in LITE6_JOINT_NAMES] for point in points]
    description = {
        "summary": "Lite 6 joint-space calibration trajectory.",
        "columns": list(LITE6_JOINT_NAMES),
        "units": "degrees",
        "lite6_joint_min_deg": list(robot.Lite6.JOINT_MIN_DEG),
        "lite6_joint_max_deg": list(robot.Lite6.JOINT_MAX_DEG),
        "interpolation_space": "joint_angles",
        "interpolation_crosses_groups": False,
    }
    if group_ranges is not None:
        description["interpolation_groups"] = group_ranges
    path = write_json(path, numeric_json(description, data))
    print(f"[INFO] Saved {len(points)} Lite 6 joint poses to {path}")
    return path


def load_lite6_joint_trajectory_file(path=LITE6_TRAJECTORY_FILE):
    """Load a Lite 6 joint-space trajectory and enforce its limits."""
    data, description = read_json_data(path)
    if description.get("columns") != list(LITE6_JOINT_NAMES):
        raise ValueError(
            "Lite 6 collection requires a joint-space trajectory. "
            "Recreate any legacy or XYZ/RPY trajectory with the current "
            "create command."
        )
    points = numeric_rows_to_points(data, LITE6_JOINT_NAMES)
    return validate_lite6_joint_trajectory(points)


def delete_trajectory_file(path=TRAJECTORY_FILE):
    """Delete the selected trajectory file. Return False when it is absent."""
    path = Path(path).resolve()
    if path.suffix.lower() != ".json":
        raise ValueError("Trajectory path must end in .json.")
    if not path.exists():
        return False
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    path.unlink()
    print(f"[INFO] Deleted {path}")
    return True


# ======================================================
# Reusable preview and transform helpers
# ======================================================

def show_image(window_name, image, lines=(), detection=None, pose=None,
               camera_matrix=None, dist_coeffs=None, wait_ms=1):
    """Show an image with optional ChArUco results and status text."""
    if image is None or image.size == 0:
        return -1

    view = image.copy()
    if detection is not None:
        if detection.marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(
                view, list(detection.marker_corners), detection.marker_ids
            )
        if detection.charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(
                view, detection.charuco_corners, detection.charuco_ids
            )

    if pose is not None and camera_matrix is not None:
        cv2.drawFrameAxes(
            view,
            camera_matrix,
            dist_coeffs,
            pose.rotation_vector,
            pose.translation_vector,
            DEFAULT_CHARUCO_CONFIG.axis_length_m,
        )

    for index, line in enumerate(lines):
        cv2.putText(
            view,
            str(line),
            (15, 30 + 28 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow(window_name, view)
    return cv2.waitKeyEx(wait_ms)


def close_image_windows():
    """Close OpenCV windows without masking an earlier hardware error."""
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def return_robot_to_rest(robot_arm):
    """Attempt a rest move during cleanup without hiding prior errors."""
    try:
        print("[INFO] Returning robot to the rest position...")
        robot_arm.move_to_rest(max_step_deg=2.0, step_delay=0.05)
        print("[INFO] Robot reached the rest position.")
    except Exception as error:
        print(f"[WARNING] Robot could not return to rest: {error}")


def invert_transform(T):
    """Return the rigid inverse of a 4x4 transform."""
    T = np.asarray(T, dtype=float).reshape(4, 4)
    inverse = np.eye(4)
    inverse[:3, :3] = T[:3, :3].T
    inverse[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return inverse


def make_transform(rotation, translation):
    """Build a 4x4 transform from a rotation and translation."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return T


# ======================================================
# Interactive trajectory creation
# ======================================================

def control_lite6_with_keyboard(
    ip_address,
    translation_step_mm=5.0,
    rotation_step_deg=2.0,
    speed_mm_s=30.0,
    trajectory_path=None,
    control_mode="keyboard",
    points_between=0,
):
    """Jog a UFACTORY Lite 6 from the terminal, one step per keypress.

    Translation keys are W/S (X), A/D (Y), and R/F (Z). Rotation keys are
    I/K (roll), J/L (pitch), and U/O (yaw). When ``trajectory_path`` is
    supplied, C captures the current pose and Backspace removes the last one.
    """
    if translation_step_mm <= 0:
        raise ValueError("translation_step_mm must be positive.")
    if rotation_step_deg <= 0:
        raise ValueError("rotation_step_deg must be positive.")
    if speed_mm_s <= 0:
        raise ValueError("speed_mm_s must be positive.")
    control_mode = str(control_mode).strip().lower()
    if control_mode not in ("keyboard", "manual"):
        raise ValueError("control_mode must be 'keyboard' or 'manual'.")
    if points_between < 0:
        raise ValueError("points_between cannot be negative.")

    lite6 = robot.Lite6(ip_address)
    points = []
    zed = None
    runtime = None
    image_zed = None
    camera_matrix = None
    dist_coeffs = None
    board = None
    detector = None
    preview_window = "Lite 6 trajectory board preview"

    jogs = {
        ord("w"): ("x", translation_step_mm),
        ord("s"): ("x", -translation_step_mm),
        ord("a"): ("y", translation_step_mm),
        ord("d"): ("y", -translation_step_mm),
        ord("r"): ("z", translation_step_mm),
        ord("f"): ("z", -translation_step_mm),
        ord("i"): ("roll", rotation_step_deg),
        ord("k"): ("roll", -rotation_step_deg),
        ord("j"): ("pitch", rotation_step_deg),
        ord("l"): ("pitch", -rotation_step_deg),
        ord("u"): ("yaw", rotation_step_deg),
        ord("o"): ("yaw", -rotation_step_deg),
    }

    def keyboard_loop(screen):
        curses.curs_set(0)
        screen.keypad(True)
        screen.timeout(20)
        status = "Ready"
        board_status = "Camera preview disabled"

        while True:
            preview_key = -1
            if zed is not None:
                image = get_image(zed, runtime, image_zed)
                if image is not None:
                    detection = detect_charuco_board(image, board, detector)
                    pose = estimate_charuco_pose(
                        detection,
                        board,
                        camera_matrix,
                        dist_coeffs,
                    )
                    board_status = (
                        f"markers={detection.num_markers}/{len(board.getIds())}, "
                        f"corners={detection.num_charuco_corners}/"
                        f"{DEFAULT_CHARUCO_CONFIG.max_charuco_corners}, "
                        f"pose={'yes' if pose is not None else 'no'}"
                    )
                    preview_key = show_image(
                        preview_window,
                        image,
                        (
                            "Lite 6 trajectory creation",
                            board_status,
                            "C capture | Backspace undo | Q save | ESC cancel",
                        ),
                        detection,
                        pose,
                        camera_matrix,
                        dist_coeffs,
                        wait_ms=1,
                    )

            screen.erase()
            title = (
                "Lite 6 free-drive trajectory capture"
                if control_mode == "manual"
                else "Lite 6 keyboard control"
            )
            screen.addstr(0, 0, title, curses.A_BOLD)
            if control_mode == "manual":
                screen.addstr(2, 0, "Move the robot by hand (Mode 2 free-drive)")
                screen.addstr(3, 0, "Confirm mounting and payload are configured")
            else:
                screen.addstr(2, 0, "W/S: +X/-X     A/D: +Y/-Y")
                screen.addstr(3, 0, "R/F: +Z/-Z")
                screen.addstr(4, 0, "I/K: +roll/-roll")
                screen.addstr(5, 0, "J/L: +pitch/-pitch")
                screen.addstr(6, 0, "U/O: +yaw/-yaw")
            if trajectory_path is None:
                screen.addstr(8, 0, "Q: disconnect and quit")
            else:
                screen.addstr(
                    8,
                    0,
                    "C: capture  Backspace: undo  Q: save  ESC: cancel",
                )
            if control_mode == "manual":
                screen.addstr(
                    10,
                    0,
                    f"Interpolation: {points_between} pose(s) per segment",
                )
            else:
                screen.addstr(
                    10,
                    0,
                    f"Steps: {translation_step_mm:g} mm, "
                    f"{rotation_step_deg:g} deg; speed {speed_mm_s:g} mm/s",
                )

            code, pose = lite6.arm.get_position(is_radian=False)
            if code == 0:
                screen.addstr(
                    12,
                    0,
                    "Pose [x y z r p y]: "
                    + " ".join(f"{value:8.2f}" for value in pose[:6]),
                )
            else:
                status = f"Pose read failed with Lite 6 code {code}"

            screen.addstr(14, 0, f"Status: {status}")
            if trajectory_path is not None:
                screen.addstr(15, 0, f"Captured poses: {len(points)}")
                screen.addstr(16, 0, f"Board: {board_status}")
            screen.refresh()

            key = preview_key if preview_key != -1 else screen.getch()
            if key == -1:
                continue
            if key == 27:
                return None
            key = ord(chr(key).lower()) if 0 <= key <= 255 else key
            if key == ord("q"):
                if trajectory_path is not None:
                    if not points:
                        status = "Capture at least one pose before saving"
                        continue
                    saved_points = interpolate_lite6_trajectory(
                        points, points_between
                    )
                    return save_lite6_trajectory_file(
                        saved_points, trajectory_path
                    )
                return None
            if trajectory_path is not None and key == ord("c"):
                code, pose = lite6.arm.get_position(is_radian=False)
                if code != 0:
                    raise RuntimeError(
                        f"Could not capture Lite 6 pose; error code {code}"
                    )
                points.append(
                    {
                        name: float(pose[index])
                        for index, name in enumerate(LITE6_POSE_NAMES)
                    }
                )
                status = f"Captured pose {len(points)}"
                continue
            if trajectory_path is not None and key in (
                curses.KEY_BACKSPACE,
                8,
                127,
            ):
                if points:
                    points.pop()
                    status = "Removed the last pose"
                else:
                    status = "No captured poses to remove"
                continue
            if key not in jogs:
                status = "Unknown key"
                continue
            if control_mode == "manual":
                status = "Manual mode: drag the robot by hand; C captures"
                continue

            axis, amount = jogs[key]
            command = {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            }
            command[axis] = amount
            code = lite6.arm.set_position(
                **command,
                speed=speed_mm_s,
                wait=True,
                relative=True,
                is_radian=False,
            )
            if code != 0:
                raise RuntimeError(
                    f"Lite 6 {axis} jog failed with error code {code}"
                )
            status = f"Moved {axis} by {amount:+g}"

    initial_position_reached = False
    try:
        lite6.connect()
        print("[INFO] Moving Lite 6 to its initial position...")
        lite6.move_to_initial()
        initial_position_reached = True
        if trajectory_path is not None:
            zed, runtime, image_zed = open_zed()
            camera_matrix, dist_coeffs = get_zed_left_intrinsics_rectified(zed)
            board, detector = create_charuco_detector(
                DEFAULT_CHARUCO_CONFIG,
                camera_matrix,
                dist_coeffs,
            )
            cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
        if control_mode == "manual":
            print(
                "[WARNING] Entering Mode 2 free-drive. The configured mounting "
                "direction and payload must match the physical robot."
            )
            lite6.enter_manual_mode()
        return curses.wrapper(keyboard_loop)
    finally:
        close_image_windows()
        if zed is not None:
            zed.close()
        if lite6.arm is not None:
            try:
                if initial_position_reached:
                    print("[INFO] Returning Lite 6 to its initial position...")
                    lite6.move_to_initial()
                    print("[INFO] Lite 6 reached its initial position.")
                else:
                    lite6.reset_state()
            except Exception as error:
                print(
                    "[WARNING] Could not return the Lite 6 to its initial "
                    "position "
                    f"during cleanup: {error}"
                )
        lite6.disconnect()


def create_lite6_trajectory_with_keyboard(
    ip_address,
    path=LITE6_TRAJECTORY_FILE,
    translation_step_mm=5.0,
    rotation_step_deg=2.0,
    speed_mm_s=30.0,
    control_mode="keyboard",
    points_between=0,
):
    """Capture Lite 6 TCP poses using keyboard jogging or manual free-drive."""
    return control_lite6_with_keyboard(
        ip_address,
        translation_step_mm,
        rotation_step_deg,
        speed_mm_s,
        trajectory_path=path,
        control_mode=control_mode,
        points_between=points_between,
    )


def create_lite6_trajectory_with_preview(
    ip_address,
    path=LITE6_TRAJECTORY_FILE,
    points_between=0,
    requested_points=None,
):
    """Capture UFactory-Studio-controlled poses from the camera preview.

    This function never commands robot motion or changes the controller mode.
    Move the Lite 6 with UFactory Studio, then use the OpenCV preview keys to
    record clipped six-joint configurations. Interpolation is performed in
    joint space, restricted to the current group, and never crosses an ``N``
    group boundary. TCP XYZ/RPY is displayed for operator reference only.
    """
    if points_between < 0:
        raise ValueError("points_between cannot be negative.")
    if requested_points is not None and requested_points < 2:
        raise ValueError("requested_points must be at least 2.")

    lite6 = robot.Lite6(ip_address)
    groups = [[]]
    zed = None
    preview_window = "Lite 6 UFactory Studio trajectory capture"
    status = "Move the robot in UFactory Studio, then press S to save."

    try:
        # Passive connection: do not clear faults, change modes, or command
        # motion while UFactory Studio owns robot control.
        lite6.connect(prepare=False)
        zed, runtime, image_zed = open_zed()
        camera_matrix, dist_coeffs = get_zed_left_intrinsics_rectified(zed)
        board, detector = create_charuco_detector(
            DEFAULT_CHARUCO_CONFIG,
            camera_matrix,
            dist_coeffs,
        )
        cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)

        while True:
            image = get_image(zed, runtime, image_zed)
            if image is None:
                continue

            detection = detect_charuco_board(image, board, detector)
            board_pose = estimate_charuco_pose(
                detection,
                board,
                camera_matrix,
                dist_coeffs,
            )
            code, tcp_pose = lite6.arm.get_position(is_radian=False)
            if code != 0:
                raise RuntimeError(
                    f"Could not read Lite 6 TCP pose; error code: {code}"
                )
            tcp_pose = np.asarray(tcp_pose[:6], dtype=float)
            joint_code, joint_angles = lite6.arm.get_servo_angle(
                is_radian=False
            )
            if joint_code != 0:
                raise RuntimeError(
                    "Could not read Lite 6 joint angles; "
                    f"error code: {joint_code}"
                )
            joint_angles = np.asarray(joint_angles[:6], dtype=float)
            clipped_joint_angles = robot.Lite6.clip_joint_angles_deg(
                joint_angles
            )
            saved_count = sum(len(group) for group in groups)
            board_status = (
                f"markers={detection.num_markers}/{len(board.getIds())}, "
                f"corners={detection.num_charuco_corners}/"
                f"{DEFAULT_CHARUCO_CONFIG.max_charuco_corners}, "
                f"pose={'yes' if board_pose is not None else 'no'}"
            )
            lines = (
                "Move in UFactory Studio; this preview saves JOINT ANGLES only",
                "S/C save | N new group | U/Backspace undo | Q finish | ESC cancel",
                f"Group {len(groups) - 1}: {len(groups[-1])} poses | total: {saved_count}",
                board_status,
                "TCP [mm, deg]: " + " ".join(f"{value:.2f}" for value in tcp_pose),
                "Joints [deg]: "
                + " ".join(f"{value:.2f}" for value in clipped_joint_angles),
                status,
            )
            key = show_image(
                preview_window,
                image,
                lines,
                detection,
                board_pose,
                camera_matrix,
                dist_coeffs,
                wait_ms=20,
            )
            if key == -1:
                continue
            if key == 27:
                print("[INFO] Lite 6 trajectory creation cancelled.")
                return None
            key = ord(chr(key).lower()) if 0 <= key <= 255 else key

            if key in (ord("s"), ord("c")):
                point = {
                    name: float(clipped_joint_angles[index])
                    for index, name in enumerate(LITE6_JOINT_NAMES)
                }
                if groups[-1]:
                    previous = np.asarray(
                        [groups[-1][-1][name] for name in LITE6_JOINT_NAMES]
                    )
                    if np.allclose(previous, clipped_joint_angles, atol=1e-4):
                        status = "Pose not saved: it duplicates the previous pose."
                        continue
                groups[-1].append(point)
                visibility = (
                    "full board visible"
                    if detection.all_corners_detected
                    else "board is not fully visible"
                )
                status = (
                    f"Saved group {len(groups) - 1}, pose "
                    f"{len(groups[-1]) - 1}; {visibility}."
                )
                if not np.allclose(joint_angles, clipped_joint_angles):
                    status += " One or more joint angles were clipped."
            elif key == ord("n"):
                if not groups[-1]:
                    status = "Current group is empty; save a pose first."
                    continue
                groups.append([])
                status = (
                    f"Started group {len(groups) - 1}; interpolation will not "
                    "cross this boundary."
                )
            elif key in (ord("u"), 8, 127):
                if not groups[-1] and len(groups) > 1:
                    groups.pop()
                if groups[-1]:
                    groups[-1].pop()
                    status = "Removed the most recently saved pose."
                else:
                    status = "There is no saved pose to remove."
            elif key == ord("q"):
                clean_groups = [group for group in groups if group]
                captured_count = sum(len(group) for group in clean_groups)
                if captured_count < 2:
                    status = "Save at least two poses before finishing."
                    continue
                try:
                    generated, group_ranges = interpolate_lite6_joint_groups(
                        clean_groups,
                        points_between=points_between,
                        requested_points=requested_points,
                    )
                except ValueError as error:
                    status = str(error)
                    continue
                return save_lite6_joint_trajectory_file(
                    generated,
                    path,
                    group_ranges=group_ranges,
                )
    finally:
        close_image_windows()
        if zed is not None:
            zed.close()
        lite6.disconnect()


def create_trajectory_with_keyboard(
    path=TRAJECTORY_FILE,
    port="/dev/ttyACM0",
    robot_id="dbot",
    step_deg=2.0,
):
    """Manually move the robot while previewing the board and save poses.

    Keys: 1-6 select joint, LEFT/RIGHT move, H home, R rest, S save,
    U undo, Q finish and save, ESC cancel.
    """
    if step_deg <= 0:
        raise ValueError("step_deg must be positive.")

    arm = robot.SOArm101(port=port, id=robot_id)
    zed = None
    connected = False
    motion_ready = False
    points = []
    selected_joint = 0
    window = "Create calibration trajectory"

    try:
        arm.connect(calibrate=False)
        connected = True
        current = arm.get_joint_angles_deg()
        arm.current_action = {
            name: float(current[index])
            for index, name in enumerate(JOINT_NAMES)
        }
        motion_ready = True

        zed, runtime, image_zed = open_zed()
        camera_matrix, dist_coeffs = get_zed_left_intrinsics_rectified(zed)
        board, detector = create_charuco_detector(
            DEFAULT_CHARUCO_CONFIG, camera_matrix, dist_coeffs
        )
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        while True:
            image = get_image(zed, runtime, image_zed)
            if image is None:
                continue

            detection = detect_charuco_board(image, board, detector)
            pose = estimate_charuco_pose(
                detection, board, camera_matrix, dist_coeffs
            )
            measured = arm.get_joint_angles_deg()
            lines = [
                "1-6 joint | LEFT/RIGHT move | S save | U undo | Q finish",
                f"Selected: {JOINT_NAMES[selected_joint]}  step={step_deg:g} deg",
                f"Saved points: {len(points)}  corners: {detection.num_charuco_corners}",
                "Joints: " + " ".join(f"{value:.1f}" for value in measured),
            ]
            key = show_image(
                window, image, lines, detection, pose,
                camera_matrix, dist_coeffs, wait_ms=20
            )

            if ord("1") <= key <= ord("6"):
                selected_joint = key - ord("1")
            elif key in LEFT_ARROW_KEYS | RIGHT_ARROW_KEYS:
                direction = -1.0 if key in LEFT_ARROW_KEYS else 1.0
                target = dict(arm.current_action)
                name = JOINT_NAMES[selected_joint]
                target[name] += direction * step_deg
                arm.moveSO101(target, max_step_deg=step_deg, step_delay=0.04)
            elif key in (ord("h"), ord("H")):
                arm.move_to_home()
            elif key in (ord("r"), ord("R")):
                arm.move_to_rest()
            elif key in (ord("s"), ord("S")):
                measured = arm.get_joint_angles_deg()
                points.append(
                    {
                        name: float(measured[index])
                        for index, name in enumerate(JOINT_NAMES)
                    }
                )
                print(f"[INFO] Saved trajectory point {len(points) - 1}")
            elif key in (ord("u"), ord("U")) and points:
                points.pop()
                print("[INFO] Removed the last trajectory point")
            elif key in (ord("q"), ord("Q")):
                if not points:
                    print(
                        "[INFO] No trajectory points were saved; "
                        "no trajectory file was created."
                    )
                    return None
                return save_trajectory_file(points, path)
            elif key == 27:
                print("[INFO] Trajectory creation cancelled")
                return None
    finally:
        close_image_windows()
        if zed is not None:
            zed.close()
        if connected and motion_ready:
            return_robot_to_rest(arm)
        if connected:
            arm.disconnect()


# ======================================================
# Automatic calibration-data collection
# ======================================================

def collect_camera_transform(
    zed,
    runtime,
    image_zed,
    board,
    detector,
    camera_matrix,
    dist_coeffs,
    timeout_seconds=12.0,
    window_name=COLLECTION_WINDOW_NAME,
):
    """Return the first pose in which every board marker is detected."""
    required_markers = len(board.getIds())
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        image = get_image(zed, runtime, image_zed)
        if image is None:
            continue

        detection = detect_charuco_board(
            image, board, detector, min_corners_for_pose=4
        )
        pose = estimate_charuco_pose(
            detection, board, camera_matrix, dist_coeffs
        )
        all_markers_detected = detection.num_markers == required_markers

        key = show_image(
            window_name,
            image,
            (
                f"Markers: {detection.num_markers}/{required_markers}",
                f"Corners: {detection.num_charuco_corners}",
                "Q aborts collection",
            ),
            detection,
            pose,
            camera_matrix,
            dist_coeffs,
            wait_ms=1,
        )
        if key in (ord("q"), ord("Q"), 27):
            raise KeyboardInterrupt("Collection stopped by operator.")
        if all_markers_detected and pose is not None:
            return pose.T_camera_board

    return None


def save_collected_rt(T_base_ee_list, T_cam_board_list):
    """Save only the four synchronized rotation and translation lists."""
    if len(T_base_ee_list) != len(T_cam_board_list):
        raise ValueError("Robot and camera sample counts do not match.")

    values = {
        R_BASE_EE_FILE: (
            "R_base_ee rotation matrices; data shape is [sample, row, column].",
            [T[:3, :3].tolist() for T in T_base_ee_list],
        ),
        T_BASE_EE_FILE: (
            "t_base_ee vectors; columns are [x, y, z] in meters.",
            [T[:3, 3].tolist() for T in T_base_ee_list],
        ),
        R_CAM_BOARD_FILE: (
            "R_camera_board rotation matrices; data shape is "
            "[sample, row, column].",
            [T[:3, :3].tolist() for T in T_cam_board_list],
        ),
        T_CAM_BOARD_FILE: (
            "t_camera_board vectors; columns are [x, y, z] in meters.",
            [T[:3, 3].tolist() for T in T_cam_board_list],
        ),
    }
    for path, (summary, data) in values.items():
        write_json(
            path,
            numeric_json({"summary": summary}, data),
        )


def collect_calibration_data(
    trajectory_path=TRAJECTORY_FILE,
    port="/dev/ttyACM0",
    robot_id="dbot",
    settle_seconds=10.0,
):
    """Execute the saved trajectory and collect synchronized R/t lists."""
    if settle_seconds < 0:
        raise ValueError("settle_seconds cannot be negative.")
    trajectory = load_trajectory_file(trajectory_path)
    arm = robot.SOArm101(port=port, id=robot_id)
    zed = None
    connected = False
    motion_ready = False
    T_base_ee_list = []
    T_cam_board_list = []

    confirmation = input(
        f"This will move through {len(trajectory)} poses and overwrite old "
        "R/t data. Type RUN to continue: "
    ).strip()
    if confirmation != "RUN":
        print("[INFO] Collection cancelled")
        return None

    # Start a new synchronized dataset; no summary or debug files are created.
    save_collected_rt([], [])

    try:
        arm.connect(calibrate=False)
        connected = True
        current = arm.get_joint_angles_deg()
        arm.current_action = {
            name: float(current[index])
            for index, name in enumerate(JOINT_NAMES)
        }
        motion_ready = True

        zed, runtime, image_zed = open_zed()
        camera_matrix, dist_coeffs = get_zed_left_intrinsics_rectified(zed)
        board, detector = create_charuco_detector(
            DEFAULT_CHARUCO_CONFIG, camera_matrix, dist_coeffs
        )

        # Create one persistent preview and reuse it for the entire trajectory.
        # Its last frame remains visible while the robot moves and settles.
        cv2.namedWindow(COLLECTION_WINDOW_NAME, cv2.WINDOW_NORMAL)

        for index, target in enumerate(trajectory):
            print(f"[INFO] Pose {index + 1}/{len(trajectory)}")
            arm.moveSO101(target, max_step_deg=2.0, step_delay=0.05)
            time.sleep(settle_seconds)

            T_cam_board = collect_camera_transform(
                zed,
                runtime,
                image_zed,
                board,
                detector,
                camera_matrix,
                dist_coeffs,
                window_name=COLLECTION_WINDOW_NAME,
            )
            if T_cam_board is None:
                print(f"[WARNING] Skipped pose {index}: board was not detected")
                continue

            # Read FK once so it is paired with the accepted camera transform.
            T_base_ee_list.append(arm.get_T_base_to_ee())
            T_cam_board_list.append(T_cam_board)
            save_collected_rt(T_base_ee_list, T_cam_board_list)
            print(f"[INFO] Saved sample {len(T_base_ee_list) - 1}")

        print(f"[INFO] Collected {len(T_base_ee_list)} synchronized samples")
        return len(T_base_ee_list)
    finally:
        # Close the persistent preview only after collection ends or is stopped.
        close_image_windows()
        if zed is not None:
            zed.close()
        if connected and motion_ready:
            return_robot_to_rest(arm)
        if connected:
            arm.disconnect()


def collect_lite6_calibration_data(
    ip_address,
    trajectory_path=LITE6_TRAJECTORY_FILE,
    settle_seconds=3.0,
    speed_deg_s=30.0,
):
    """Replay Lite 6 joint angles and collect synchronized transforms."""
    if settle_seconds < 0:
        raise ValueError("settle_seconds cannot be negative.")
    if speed_deg_s <= 0:
        raise ValueError("speed_deg_s must be positive.")

    trajectory = load_lite6_joint_trajectory_file(trajectory_path)
    lite6 = robot.Lite6(ip_address)
    zed = None
    T_base_ee_list = []
    T_cam_board_list = []

    confirmation = input(
        f"This will move the Lite 6 through {len(trajectory)} joint poses "
        "poses and overwrite old R/t data. Type RUN to continue: "
    ).strip()
    if confirmation != "RUN":
        print("[INFO] Collection cancelled")
        return None

    try:
        lite6.connect()
        print(
            f"[INFO] Loaded {len(trajectory)} clipped, joint-limit-checked poses."
        )
        save_collected_rt([], [])
        zed, runtime, image_zed = open_zed()
        camera_matrix, dist_coeffs = get_zed_left_intrinsics_rectified(zed)
        board, detector = create_charuco_detector(
            DEFAULT_CHARUCO_CONFIG, camera_matrix, dist_coeffs
        )
        cv2.namedWindow(COLLECTION_WINDOW_NAME, cv2.WINDOW_NORMAL)

        for index, target in enumerate(trajectory):
            print(f"[INFO] Lite 6 pose {index + 1}/{len(trajectory)}")
            target_angles = [target[name] for name in LITE6_JOINT_NAMES]
            code = lite6.arm.set_servo_angle(
                angle=target_angles,
                speed=speed_deg_s,
                wait=True,
                is_radian=False,
            )
            if code != 0:
                raise RuntimeError(
                    f"Lite 6 move to pose {index} failed with error code {code}"
                )
            time.sleep(settle_seconds)

            T_cam_board = collect_camera_transform(
                zed,
                runtime,
                image_zed,
                board,
                detector,
                camera_matrix,
                dist_coeffs,
                window_name=COLLECTION_WINDOW_NAME,
            )
            if T_cam_board is None:
                print(f"[WARNING] Skipped pose {index}: board was not detected")
                continue

            T_base_ee_list.append(lite6.get_T_base_to_ee())
            T_cam_board_list.append(T_cam_board)
            save_collected_rt(T_base_ee_list, T_cam_board_list)
            print(f"[INFO] Saved sample {len(T_base_ee_list) - 1}")

        print(f"[INFO] Collected {len(T_base_ee_list)} synchronized samples")
        return len(T_base_ee_list)
    finally:
        close_image_windows()
        if zed is not None:
            zed.close()
        lite6.disconnect()


# ======================================================
# Eye-hand calibration solvers
# ======================================================

def load_collected_transforms():
    """Load the four synchronized R/t files as two transform lists."""
    paths = (
        R_BASE_EE_FILE,
        T_BASE_EE_FILE,
        R_CAM_BOARD_FILE,
        T_CAM_BOARD_FILE,
    )
    arrays = []
    for path in paths:
        data, _ = read_json_data(path)
        arrays.append(data)

    arrays = [
        np.asarray(values, dtype=float)
        for values in arrays
    ]
    R_base_ee, t_base_ee, R_cam_board, t_cam_board = arrays
    counts = {len(values) for values in arrays}
    if len(counts) != 1 or not counts or next(iter(counts)) < 3:
        raise ValueError("The four R/t files must contain at least 3 samples.")
    sample_count = next(iter(counts))
    expected_shapes = (
        (sample_count, 3, 3),
        (sample_count, 3),
        (sample_count, 3, 3),
        (sample_count, 3),
    )
    for path, values, expected_shape in zip(paths, arrays, expected_shapes):
        if values.shape != expected_shape or not np.all(np.isfinite(values)):
            raise ValueError(
                f"{path} must contain finite data shaped {expected_shape}."
            )

    T_base_ee = [
        make_transform(R, t) for R, t in zip(R_base_ee, t_base_ee)
    ]
    T_cam_board = [
        make_transform(R, t) for R, t in zip(R_cam_board, t_cam_board)
    ]
    return T_base_ee, T_cam_board


def solve_eye_hand_li(output_path=SOLUTION_FILE):
    """Solve eye-to-hand calibration using only OpenCV's Li method."""
    T_base_ee, T_cam_board = load_collected_transforms()

    R_board_ee, t_board_ee, R_cam_base, t_cam_base = (
        cv2.calibrateRobotWorldHandEye(
            R_world2cam=[T[:3, :3] for T in T_cam_board],
            t_world2cam=[T[:3, 3] for T in T_cam_board],
            R_base2gripper=[T[:3, :3] for T in T_base_ee],
            t_base2gripper=[T[:3, 3] for T in T_base_ee],
            method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI,
        )
    )

    T_base_camera = invert_transform(
        make_transform(R_cam_base, t_cam_base)
    )
    T_ee_board = invert_transform(
        make_transform(R_board_ee, t_board_ee)
    )
    if not (
        np.all(np.isfinite(T_base_camera))
        and np.all(np.isfinite(T_ee_board))
    ):
        raise RuntimeError("The Li solver returned a non-finite transform.")
    result = numeric_json(
        {
            "summary": "Li eye-to-hand calibration transforms.",
            "axis_0_order": ["T_base_camera", "T_ee_board"],
            "matrix_layout": "4x4 homogeneous transforms",
            "translation_units": "meters",
        },
        np.stack((T_base_camera, T_ee_board)),
    )
    output_path = write_json(output_path, result)
    print(f"[INFO] Li calibration saved to {output_path}")
    return T_base_camera, T_ee_board


def solve_eye_hand_tsai(output_path=TSAI_SOLUTION_FILE):
    """Solve fixed-camera eye-to-hand calibration with the Tsai method.

    For this eye-to-hand arrangement, ``calibrateHandEye`` receives
    ``T_ee_base`` (the inverse of each measured ``T_base_ee``) as its
    gripper-to-base input. Its returned camera-to-gripper transform is
    therefore physically ``T_base_camera``.

    Only ``T_base_camera`` is saved and returned; the board mounting
    transform is intentionally not recovered.
    """
    T_base_ee, T_cam_board = load_collected_transforms()
    T_ee_base = [invert_transform(T) for T in T_base_ee]

    R_base_camera, t_base_camera = cv2.calibrateHandEye(
        R_gripper2base=[T[:3, :3] for T in T_ee_base],
        t_gripper2base=[T[:3, 3] for T in T_ee_base],
        R_target2cam=[T[:3, :3] for T in T_cam_board],
        t_target2cam=[T[:3, 3] for T in T_cam_board],
        method=cv2.CALIB_HAND_EYE_TSAI,
    )

    T_base_camera = make_transform(R_base_camera, t_base_camera)
    if not np.all(np.isfinite(T_base_camera)):
        raise RuntimeError("The Tsai solver returned a non-finite transform.")

    output_path = write_json(
        output_path,
        numeric_json(
            {
                "summary": "Tsai eye-to-hand calibration transform.",
                "axis_0_order": ["T_base_camera"],
                "matrix_layout": "4x4 homogeneous transform",
                "translation_units": "meters",
            },
            T_base_camera[np.newaxis, :, :],
        ),
    )
    print(f"[INFO] Tsai calibration saved to {output_path}")
    return T_base_camera


def solve_eye_hand(method="li", output_path=None):
    """Run the selected calibration solver and save its JSON result."""
    method = str(method).strip().lower()
    if method == "li":
        return solve_eye_hand_li(output_path or SOLUTION_FILE)
    if method == "tsai":
        return solve_eye_hand_tsai(output_path or TSAI_SOLUTION_FILE)
    raise ValueError("method must be either 'li' or 'tsai'.")


# ======================================================
# Plotting
# ======================================================

def set_3d_axes_equal(ax, points, padding=0.15):
    """Apply equal metric scaling around a collection of 3D points."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float(np.max(maximum - minimum)) / 2.0, 0.05)
    radius *= 1.0 + padding
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def draw_coordinate_frame(ax, transform, label, axis_length=0.08):
    """Draw a labelled RGB coordinate frame on a 3D axis."""
    transform = np.asarray(transform, dtype=float).reshape(4, 4)
    origin = transform[:3, 3]
    for axis_index, (color, axis_name) in enumerate(
        (("red", "x"), ("green", "y"), ("blue", "z"))
    ):
        direction = transform[:3, axis_index] * axis_length
        ax.quiver(
            *origin,
            *direction,
            color=color,
            linewidth=2.0,
            arrow_length_ratio=0.18,
        )
        endpoint = origin + direction
        ax.text(*endpoint, f"{label}.{axis_name}", color=color, fontsize=8)

    ax.scatter(*origin, color="black", s=35, depthshade=False)
    ax.text(*origin, f"  {label}", color="black", weight="bold")
    return origin


def plot_transform_set(ax, transforms, label, color, connect=False):
    """Plot transform origins and short coordinate axes on a 3D axis."""
    transforms = [np.asarray(T, dtype=float).reshape(4, 4) for T in transforms]
    positions = np.array([T[:3, 3] for T in transforms])
    marker = "o" if connect else "x"
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        marker=marker,
        linestyle="-" if connect else "None",
        color=color,
        label=label,
    )

    axis_length = 0.02
    axis_colors = ("r", "g", "b")
    for T in transforms:
        origin = T[:3, 3]
        for axis_index, axis_color in enumerate(axis_colors):
            direction = T[:3, axis_index] * axis_length
            ax.quiver(*origin, *direction, color=axis_color, linewidth=0.7)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()
    set_3d_axes_equal(ax, np.vstack((positions, np.zeros((1, 3)))))
    ax.grid(True, alpha=0.35)
    ax.view_init(elev=25, azim=-55)


def plot_trajectory(path=TRAJECTORY_FILE):
    """Plot an SO-101 joint trajectory or a Lite 6 TCP trajectory."""
    import matplotlib.pyplot as plt

    path = Path(path).resolve()
    raw_points, description = read_json_data(path)

    is_lite6_joint = (
        description.get("columns") == list(LITE6_JOINT_NAMES)
        or (
            raw_points
            and isinstance(raw_points[0], dict)
            and set(raw_points[0]) == set(LITE6_JOINT_NAMES)
        )
    )
    is_lite6_tcp = (
        description.get("columns") == list(LITE6_POSE_NAMES)
        or (
            raw_points
            and isinstance(raw_points[0], dict)
            and set(raw_points[0]) == set(LITE6_POSE_NAMES)
        )
    )

    if is_lite6_joint:
        points = validate_lite6_joint_trajectory(
            numeric_rows_to_points(raw_points, LITE6_JOINT_NAMES)
        )
        values = np.asarray(
            [[point[name] for name in LITE6_JOINT_NAMES] for point in points],
            dtype=float,
        )
        figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
        sample_indices = np.arange(len(values))
        lower = robot.Lite6.JOINT_MIN_DEG
        upper = robot.Lite6.JOINT_MAX_DEG
        for index, axis in enumerate(axes.flat):
            axis.plot(sample_indices, values[:, index], color="tab:blue")
            axis.axhline(lower[index], color="red", linestyle="--")
            axis.axhline(upper[index], color="red", linestyle="--")
            axis.set_title(f"Lite 6 Joint {index + 1}")
            axis.set_ylabel("Angle [deg]")
            axis.grid(True, alpha=0.35)
        for axis in axes[-1]:
            axis.set_xlabel("Trajectory point")
        figure.suptitle(f"{path.name} — joint-space trajectory")
    elif is_lite6_tcp:
        points = validate_lite6_trajectory(
            numeric_rows_to_points(raw_points, LITE6_POSE_NAMES)
        )
        values = np.asarray(
            [[point[name] for name in LITE6_POSE_NAMES] for point in points],
            dtype=float,
        )
        figure = plt.figure(figsize=(14, 5))
        path_axis = figure.add_subplot(131, projection="3d")
        xyz_axis = figure.add_subplot(132)
        rpy_axis = figure.add_subplot(133)
        sample_indices = np.arange(len(values))

        group_ranges = description.get("interpolation_groups", [])
        if group_ranges:
            colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(group_ranges)))
            previous_end = None
            transition_labeled = False
            for group, color in zip(group_ranges, colors):
                start = int(group["trajectory_start_index"])
                end = int(group["trajectory_end_index"]) + 1
                group_values = values[start:end, :3] / 1000.0
                path_axis.plot(
                    group_values[:, 0],
                    group_values[:, 1],
                    group_values[:, 2],
                    marker="o",
                    markersize=3,
                    color=color,
                    label=f"group {group['group_index']}",
                )
                if previous_end is not None:
                    path_axis.plot(
                        [previous_end[0], group_values[0, 0]],
                        [previous_end[1], group_values[0, 1]],
                        [previous_end[2], group_values[0, 2]],
                        linestyle=":",
                        color="0.6",
                        label=(
                            "non-interpolated group transition"
                            if not transition_labeled
                            else None
                        ),
                    )
                    transition_labeled = True
                previous_end = group_values[-1]
        else:
            path_axis.plot(
                values[:, 0] / 1000.0,
                values[:, 1] / 1000.0,
                values[:, 2] / 1000.0,
                marker="o",
                markersize=3,
                color="tab:blue",
                label="trajectory",
            )
        path_axis.scatter(
            values[0, 0] / 1000.0,
            values[0, 1] / 1000.0,
            values[0, 2] / 1000.0,
            color="green",
            s=70,
            label="start",
        )
        path_axis.scatter(
            values[-1, 0] / 1000.0,
            values[-1, 1] / 1000.0,
            values[-1, 2] / 1000.0,
            color="red",
            s=70,
            label="end",
        )
        path_axis.set_title("Lite 6 Cartesian path")
        path_axis.set_xlabel("X [m]")
        path_axis.set_ylabel("Y [m]")
        path_axis.set_zlabel("Z [m]")
        path_axis.legend()
        path_axis.grid(True, alpha=0.35)
        path_axis.view_init(elev=25, azim=-55)
        set_3d_axes_equal(path_axis, values[:, :3] / 1000.0)

        for index, name in enumerate(LITE6_POSE_NAMES[:3]):
            xyz_axis.plot(
                sample_indices, values[:, index], marker=".", label=name
            )
        xyz_axis.set_title("TCP translation")
        xyz_axis.set_xlabel("Trajectory point")
        xyz_axis.set_ylabel("Position [mm]")
        xyz_axis.grid(True, alpha=0.35)
        xyz_axis.legend()

        for index, name in enumerate(LITE6_POSE_NAMES[3:], start=3):
            rpy_axis.plot(
                sample_indices, values[:, index], marker=".", label=name
            )
        rpy_axis.set_title("TCP orientation")
        rpy_axis.set_xlabel("Trajectory point")
        rpy_axis.set_ylabel("Angle [deg]")
        rpy_axis.grid(True, alpha=0.35)
        rpy_axis.legend()
    else:
        points = validate_trajectory(
            numeric_rows_to_points(raw_points, JOINT_NAMES)
        )
        values = np.array(
            [[point[name] for name in JOINT_NAMES] for point in points],
            dtype=float,
        )
        figure, axis = plt.subplots(figsize=(10, 6))
        for joint_index, name in enumerate(JOINT_NAMES):
            axis.plot(values[:, joint_index], marker=".", label=name)
        axis.set_title("SO-101 joint trajectory")
        axis.set_xlabel("Trajectory point")
        axis.set_ylabel("Robot command [deg]")
        axis.grid(True, alpha=0.35)
        axis.legend(ncol=2)

    figure.suptitle(path.name)
    figure.tight_layout()
    plt.show()


def plot_collected_data():
    """Plot unsolved robot and camera measurements in separate 3D panels."""
    import matplotlib.pyplot as plt

    T_base_ee, T_cam_board = load_collected_transforms()
    figure = plt.figure(figsize=(12, 5))
    robot_axis = figure.add_subplot(121, projection="3d")
    camera_axis = figure.add_subplot(122, projection="3d")

    # Robot samples form a connected commanded path.
    plot_transform_set(
        robot_axis, T_base_ee, "T_base_ee", "tab:blue", connect=True
    )
    robot_axis.set_title("Measured robot poses")
    set_3d_axes_equal(
        robot_axis,
        np.array([T[:3, 3] for T in T_base_ee] + [np.zeros(3)]),
    )

    # Board observations are independent camera measurements, not a path.
    plot_transform_set(
        camera_axis, T_cam_board, "T_camera_board", "tab:orange", connect=False
    )
    camera_axis.set_title("Unsolved camera observations")
    set_3d_axes_equal(
        camera_axis,
        np.array([T[:3, 3] for T in T_cam_board] + [np.zeros(3)]),
    )

    figure.tight_layout()
    plt.show()


def plot_solved_calibration(solution_path=SOLUTION_FILE):
    """Plot solved camera and optional board poses in their parent frames."""
    import matplotlib.pyplot as plt

    solution_path = Path(solution_path).resolve()
    with open(solution_path, "r", encoding="utf-8") as file:
        result = json.load(file)

    if "data" in result:
        transform_data = result["data"]
        transform_names = result.get("description", {}).get(
            "axis_0_order",
            [],
        )
        result = {
            name: transform_data[index]
            for index, name in enumerate(transform_names)
        }

    if "T_base_camera" not in result:
        raise ValueError(f"{solution_path} does not contain T_base_camera.")

    named_transforms = [
        (
            "Base",
            "Camera",
            "Camera pose in robot base frame",
            np.asarray(result["T_base_camera"], dtype=float),
        )
    ]
    if "T_ee_board" in result:
        named_transforms.append(
            (
                "End effector",
                "Board",
                "Board pose in end-effector frame",
                np.asarray(result["T_ee_board"], dtype=float),
            )
        )

    for parent, child, _, transform in named_transforms:
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError(f"T_{parent}_{child} must be a finite 4x4 matrix.")

    figure = plt.figure(figsize=(7 * len(named_transforms), 6.5))
    identity = np.eye(4)
    for index, (parent, child, title, transform) in enumerate(
        named_transforms, start=1
    ):
        axis = figure.add_subplot(
            1, len(named_transforms), index, projection="3d"
        )
        parent_origin = draw_coordinate_frame(
            axis, identity, parent, axis_length=0.08
        )
        child_origin = draw_coordinate_frame(
            axis, transform, child, axis_length=0.08
        )
        axis.plot(
            [parent_origin[0], child_origin[0]],
            [parent_origin[1], child_origin[1]],
            [parent_origin[2], child_origin[2]],
            linestyle="--",
            color="0.45",
            linewidth=1.5,
        )
        translation = transform[:3, 3]
        axis.text2D(
            0.02,
            0.96,
            "translation [m]\n"
            f"x={translation[0]:+.4f}\n"
            f"y={translation[1]:+.4f}\n"
            f"z={translation[2]:+.4f}",
            transform=axis.transAxes,
            va="top",
            family="monospace",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
        )
        set_3d_axes_equal(
            axis,
            np.vstack((parent_origin, child_origin)),
            padding=0.35,
        )
        axis.set_xlabel("X [m]")
        axis.set_ylabel("Y [m]")
        axis.set_zlabel("Z [m]")
        axis.grid(True, alpha=0.35)
        axis.view_init(elev=25, azim=-55)
        axis.set_title(title)

    figure.suptitle(f"Solved calibration: {solution_path.name}")
    figure.tight_layout()
    plt.show()


# ======================================================
# Command line
# ======================================================

def main():
    parser = argparse.ArgumentParser(description="Robot eye-hand calibration")
    commands = parser.add_subparsers(dest="command", required=True)

    pdf = commands.add_parser("pdf", help="Generate a printable ChArUco board")
    pdf.add_argument(
        "--board",
        choices=("calibration", "human-tool"),
        default="calibration",
        help="Board design to generate, default: calibration",
    )

    create = commands.add_parser(
        "create",
        help="Create a trajectory (Lite 6 uses UFactory Studio + preview)",
    )
    create.add_argument("--robot", choices=("so101", "lite6"), default="so101")
    create.add_argument("--output", type=Path)
    create.add_argument("--port", default="/dev/ttyACM0")
    create.add_argument("--robot-id", default="dbot")
    create.add_argument("--step", type=float, default=2.0)
    create.add_argument("--ip")
    interpolation = create.add_mutually_exclusive_group()
    interpolation.add_argument(
        "--interpolate",
        type=int,
        default=0,
        metavar="N",
        help="insert N joint poses per segment, only within each Lite 6 group",
    )
    interpolation.add_argument(
        "--points",
        type=int,
        help="generate exactly this many Lite 6 joint poses across all groups",
    )

    lite6_control = commands.add_parser(
        "lite6-control", help="Jog a Lite 6 with the keyboard"
    )
    lite6_control.add_argument("--ip", required=True)
    lite6_control.add_argument("--translation-step", type=float, default=5.0)
    lite6_control.add_argument("--rotation-step", type=float, default=2.0)
    lite6_control.add_argument("--speed", type=float, default=30.0)

    delete = commands.add_parser("delete", help="Delete the trajectory")
    delete.add_argument("--trajectory", type=Path, default=TRAJECTORY_FILE)

    collect = commands.add_parser("collect", help="Collect calibration R/t data")
    collect.add_argument("--robot", choices=("so101", "lite6"), default="so101")
    collect.add_argument("--trajectory", type=Path)
    collect.add_argument("--port", default="/dev/ttyACM0")
    collect.add_argument("--robot-id", default="dbot")
    collect.add_argument("--settle", type=float, default=10.0)
    collect.add_argument("--ip")
    collect.add_argument(
        "--speed",
        type=float,
        default=30.0,
        help="Lite 6 joint speed in degrees/second",
    )

    solve = commands.add_parser(
        "solve", help="Solve with Li or Tsai"
    )
    solve.add_argument(
        "--method",
        choices=("li", "tsai"),
        default="li",
        help="Li returns base-camera and EE-board; Tsai returns base-camera only",
    )
    solve.add_argument(
        "--output",
        type=Path,
        help="Optional result JSON path",
    )

    plot_trajectory_command = commands.add_parser(
        "plot-trajectory", help="Plot the trajectory joints"
    )
    plot_trajectory_command.add_argument(
        "--trajectory", type=Path, default=TRAJECTORY_FILE
    )
    commands.add_parser("plot-collected", help="Plot unsolved collected poses")
    plot_solved = commands.add_parser(
        "plot-solved", help="Plot solved calibration transforms"
    )
    plot_solved.add_argument(
        "--solution", type=Path, default=SOLUTION_FILE
    )

    args = parser.parse_args()

    if args.command == "pdf":
        if args.board == "human-tool":
            generateCharucoPDF(HUMAN_TOOL_CHARUCO_CONFIG, "human_tool")
        else:
            generateCharucoPDF(DEFAULT_CHARUCO_CONFIG)
    elif args.command == "create":
        if args.robot == "lite6":
            if not args.ip:
                parser.error("create --robot lite6 requires --ip")
            create_lite6_trajectory_with_preview(
                args.ip,
                args.output or LITE6_TRAJECTORY_FILE,
                args.interpolate,
                args.points,
            )
        else:
            create_trajectory_with_keyboard(
                args.output or TRAJECTORY_FILE,
                args.port,
                args.robot_id,
                args.step,
            )
    elif args.command == "lite6-control":
        control_lite6_with_keyboard(
            args.ip,
            args.translation_step,
            args.rotation_step,
            args.speed,
        )
    elif args.command == "delete":
        delete_trajectory_file(args.trajectory)
    elif args.command == "collect":
        if args.robot == "lite6":
            if not args.ip:
                parser.error("collect --robot lite6 requires --ip")
            collect_lite6_calibration_data(
                args.ip,
                args.trajectory or LITE6_TRAJECTORY_FILE,
                args.settle,
                args.speed,
            )
        else:
            collect_calibration_data(
                args.trajectory or TRAJECTORY_FILE,
                args.port,
                args.robot_id,
                args.settle,
            )
    elif args.command == "solve":
        solve_eye_hand(args.method, args.output)
    elif args.command == "plot-trajectory":
        plot_trajectory(args.trajectory)
    elif args.command == "plot-collected":
        plot_collected_data()
    elif args.command == "plot-solved":
        plot_solved_calibration(args.solution)


if __name__ == "__main__":
    main()
