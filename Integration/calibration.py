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

def generateCharucoPDF():
    """Generate a printable PDF from ``DEFAULT_CHARUCO_CONFIG``.

    Returns
    -------
    pathlib.Path
        Absolute path to the generated PDF. Print it at 100% (actual size).

    Notes
    -----
    All board geometry, marker sizing, dictionary selection, and legacy
    pattern behavior are taken from ``DEFAULT_CHARUCO_CONFIG``.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to create the PDF: pip install Pillow"
        ) from exc

    config = DEFAULT_CHARUCO_CONFIG
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
    pdf_path = Path(
        f"charuco_{config.squares_x}x{config.squares_y}_{size_label}.pdf"
    ).resolve()
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

ROOT = Path(__file__).resolve().parents[1]
EYE_HAND_DIR = ROOT / "data" / "eye_to_hand"
TRAJECTORY_FILE = EYE_HAND_DIR / "calibration_trajectory.json"
R_BASE_EE_FILE = EYE_HAND_DIR / "R_base_ee.json"
T_BASE_EE_FILE = EYE_HAND_DIR / "t_base_ee.json"
R_CAM_BOARD_FILE = EYE_HAND_DIR / "R_cam_board.json"
T_CAM_BOARD_FILE = EYE_HAND_DIR / "t_cam_board.json"
SOLUTION_FILE = EYE_HAND_DIR / "eye_hand_calibration.json"
COLLECTION_WINDOW_NAME = "Collect eye-hand calibration"

JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
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
    """Save only the ordered trajectory points."""
    path = Path(path).resolve()
    if path.suffix.lower() != ".json":
        raise ValueError("Trajectory path must end in .json.")
    points = validate_trajectory(points)
    write_json(path, points)
    print(f"[INFO] Saved {len(points)} points to {path}")
    return path


def load_trajectory_file(path=TRAJECTORY_FILE):
    """Load the bare trajectory point list."""
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as file:
        return validate_trajectory(json.load(file))


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
                return save_trajectory_file(points, path)
            elif key == 27:
                print("[INFO] Trajectory creation cancelled")
                return None
    finally:
        close_image_windows()
        if zed is not None:
            zed.close()
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
        R_BASE_EE_FILE: [T[:3, :3].tolist() for T in T_base_ee_list],
        T_BASE_EE_FILE: [T[:3, 3].tolist() for T in T_base_ee_list],
        R_CAM_BOARD_FILE: [T[:3, :3].tolist() for T in T_cam_board_list],
        T_CAM_BOARD_FILE: [T[:3, 3].tolist() for T in T_cam_board_list],
    }
    for path, value in values.items():
        write_json(path, value)


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
        if connected:
            arm.disconnect()


# ======================================================
# Li robot-world/hand-eye solve
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
        with open(path, "r", encoding="utf-8") as file:
            arrays.append(json.load(file))

    R_base_ee, t_base_ee, R_cam_board, t_cam_board = arrays
    counts = {len(values) for values in arrays}
    if len(counts) != 1 or not counts or next(iter(counts)) < 3:
        raise ValueError("The four R/t files must contain at least 3 samples.")

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
    result = {
        "T_base_camera": T_base_camera.tolist(),
        "T_ee_board": T_ee_board.tolist(),
    }
    output_path = write_json(output_path, result)
    print(f"[INFO] Li calibration saved to {output_path}")
    return T_base_camera, T_ee_board


# ======================================================
# Plotting
# ======================================================

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
    ax.set_box_aspect((1, 1, 1))


def plot_trajectory(path=TRAJECTORY_FILE):
    """Plot every commanded joint against trajectory point index."""
    import matplotlib.pyplot as plt

    points = load_trajectory_file(path)
    values = np.array(
        [[point[name] for name in JOINT_NAMES] for point in points], dtype=float
    )
    figure, axis = plt.subplots()
    for joint_index, name in enumerate(JOINT_NAMES):
        axis.plot(values[:, joint_index], marker=".", label=name)
    axis.set_xlabel("Trajectory point")
    axis.set_ylabel("Robot command [deg]")
    axis.grid(True)
    axis.legend()
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

    # Board observations are independent camera measurements, not a path.
    plot_transform_set(
        camera_axis, T_cam_board, "T_camera_board", "tab:orange", connect=False
    )
    camera_axis.set_title("Unsolved camera observations")

    figure.tight_layout()
    plt.show()


# ======================================================
# Command line
# ======================================================

def main():
    parser = argparse.ArgumentParser(description="SO-101 eye-hand calibration")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("pdf", help="Generate the printable ChArUco board")

    create = commands.add_parser("create", help="Create a trajectory manually")
    create.add_argument("--output", type=Path, default=TRAJECTORY_FILE)
    create.add_argument("--port", default="/dev/ttyACM0")
    create.add_argument("--robot-id", default="dbot")
    create.add_argument("--step", type=float, default=2.0)

    delete = commands.add_parser("delete", help="Delete the trajectory")
    delete.add_argument("--trajectory", type=Path, default=TRAJECTORY_FILE)

    collect = commands.add_parser("collect", help="Collect calibration R/t data")
    collect.add_argument("--trajectory", type=Path, default=TRAJECTORY_FILE)
    collect.add_argument("--port", default="/dev/ttyACM0")
    collect.add_argument("--robot-id", default="dbot")
    collect.add_argument("--settle", type=float, default=10.0)

    solve = commands.add_parser("solve", help="Solve using the Li method")
    solve.add_argument("--output", type=Path, default=SOLUTION_FILE)

    plot_trajectory_command = commands.add_parser(
        "plot-trajectory", help="Plot the trajectory joints"
    )
    plot_trajectory_command.add_argument(
        "--trajectory", type=Path, default=TRAJECTORY_FILE
    )
    commands.add_parser("plot-collected", help="Plot unsolved collected poses")

    args = parser.parse_args()

    if args.command == "pdf":
        generateCharucoPDF()
    elif args.command == "create":
        create_trajectory_with_keyboard(
            args.output, args.port, args.robot_id, args.step
        )
    elif args.command == "delete":
        delete_trajectory_file(args.trajectory)
    elif args.command == "collect":
        collect_calibration_data(
            args.trajectory,
            args.port,
            args.robot_id,
            args.settle,
        )
    elif args.command == "solve":
        solve_eye_hand_li(args.output)
    elif args.command == "plot-trajectory":
        plot_trajectory(args.trajectory)
    elif args.command == "plot-collected":
        plot_collected_data()


if __name__ == "__main__":
    main()
