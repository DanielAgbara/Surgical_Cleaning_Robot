#!/usr/bin/env python3

"""
Test a 4x4 ChArUco board pose using the ZED as a normal monocular camera.

This script:
    - Opens the ZED camera
    - Uses only the LEFT image
    - Does NOT use depth
    - Does NOT use the ZED point cloud
    - Detects a 4x4 ChArUco board
    - Estimates T_camera_board using solvePnP
    - Prints detailed debugging metrics

Coordinate convention for ZED LEFT camera:
    +X = right in image
    +Y = down in image
    +Z = forward away from camera

Coordinate convention for ChArUco board:
    The board frame is defined by OpenCV.
    The translation vector tvec gives the board origin position in the camera frame.

Important:
    The board origin is NOT the board center.

    For intuitive debugging, this script also computes the physical pattern center:
        center_board = [
            0.5 * SQUARES_X * SQUARE_LENGTH_M,
            0.5 * SQUARES_Y * SQUARE_LENGTH_M,
            0.0
        ]

    Then:
        center_camera = R_camera_board @ center_board + tvec

Use this script to compare the 4x4 board against your 5x5 board.

Good result:
    - corners detected: 9 / 9
    - mean reprojection error: preferably < 0.5 px
    - max reprojection error: preferably < 1.0 px
    - low jitter while board is stationary
"""

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl


# --------------------------------------------------
# ChArUco board settings
# --------------------------------------------------

SQUARES_X = 4
SQUARES_Y = 4

# --------------------------------------------------
# IMPORTANT:
# These are intended values for the generated board.
#
# After printing, measure the actual square and marker size.
# Then update these two values.
#
# Example:
#   If 50 mm prints as 48.2 mm:
#       SQUARE_LENGTH_M = 0.0482
#
#   If 37.5 mm prints as 36.1 mm:
#       MARKER_LENGTH_M = 0.0361
# --------------------------------------------------

SQUARE_LENGTH_M = 0.0482     # 50.0 mm intended
MARKER_LENGTH_M = 0.0361     # 37.5 mm intended

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50

# 4x4 ChArUco has:
#   (4 - 1) * (4 - 1) = 9 ChArUco corners.
#
# For final calibration samples, prefer 9/9 corners.
# For testing, you can allow 7 or 8.
MIN_CHARUCO_CORNERS = 7

# Print debug information every N seconds.
PRINT_INTERVAL_SEC = 5.0

# Axis length drawn on the image, in meters.
# This does not affect pose estimation.
AXIS_LENGTH_M = 0.05

# Number of recent good frames used to compute jitter.
TVEC_HISTORY_SIZE = 20

# Human-readable report written every PRINT_INTERVAL_SEC seconds.
ROOT = Path("/home/agbara-admin/Documents/Surgical_Cleaning_Robot")
OUTPUT_DIR = ROOT / "data" / "eye_to_hand"
REPORT_FILE = OUTPUT_DIR / "solvepnp_axis_test.txt"

# Numerical tolerances used only for reporting whether the rotation matrix
# represents an orthonormal, right-handed coordinate frame.
AXIS_DOT_TOL = 1e-6
AXIS_NORM_TOL = 1e-6
ORTHOGONALITY_MATRIX_TOL = 1e-6
DETERMINANT_TOL = 1e-6


# --------------------------------------------------
# ZED settings
# --------------------------------------------------

# Use the same resolution that you plan to use for calibration collection.
# HD720 is faster. HD1080 usually gives better pixel precision.
ZED_RESOLUTION = sl.RESOLUTION.HD1080
ZED_FPS = 15


# --------------------------------------------------
# Helper functions: board and camera setup
# --------------------------------------------------

def make_charuco_board():
    """
    Create the OpenCV ChArUco board object.

    This board object is used for:
        1. detecting ChArUco corners
        2. matching detected 2D corners to 3D board points
        3. solving for T_camera_board
    """

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)

    if hasattr(cv2.aruco, "CharucoBoard"):
        board = cv2.aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y),
            SQUARE_LENGTH_M,
            MARKER_LENGTH_M,
            dictionary,
        )
    elif hasattr(cv2.aruco, "CharucoBoard_create"):
        board = cv2.aruco.CharucoBoard_create(
            SQUARES_X,
            SQUARES_Y,
            SQUARE_LENGTH_M,
            MARKER_LENGTH_M,
            dictionary,
        )
    else:
        raise RuntimeError(
            "Your OpenCV install does not support ChArUco boards. "
            "Install opencv-contrib-python."
        )

    return board, dictionary


def open_zed_camera():
    """
    Open the ZED camera.

    We explicitly set the resolution and FPS so the camera does not use a default.
    The intrinsics we later read from the ZED correspond to this opened mode.
    """

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = ZED_RESOLUTION
    init_params.camera_fps = ZED_FPS

    # We are using the ZED as a normal camera.
    # No depth, no point cloud.
    init_params.depth_mode = sl.DEPTH_MODE.NONE

    # ZED/OpenCV image-style coordinate convention:
    #   +X right
    #   +Y down
    #   +Z forward
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

    err = zed.open(init_params)

    if err != sl.ERROR_CODE.SUCCESS:
        print(f"[ERROR] Could not open ZED camera: {err}")
        sys.exit(1)

    print("[INFO] ZED opened successfully.")

    cam_info = zed.get_camera_information()
    cam_config = cam_info.camera_configuration

    print("[INFO] Actual ZED resolution:")
    print(f"  width:  {cam_config.resolution.width}")
    print(f"  height: {cam_config.resolution.height}")
    print(f"  fps:    {cam_config.fps}")

    return zed


def get_zed_left_camera_matrix(zed):
    """
    Get the ZED left camera intrinsics.

    We retrieve:
        sl.VIEW.LEFT

    sl.VIEW.LEFT is the rectified left image from the ZED SDK.
    That means the SDK has already removed lens distortion.

    So for solvePnP, we use:
        rectified left intrinsics
        zero distortion coefficients

    Do NOT mix:
        rectified image + raw distortion
    or:
        raw image + zero distortion
    """

    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters
    left_cam = calib.left_cam

    fx = left_cam.fx
    fy = left_cam.fy
    cx = left_cam.cx
    cy = left_cam.cy

    camera_matrix = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # Rectified image means use zero distortion in OpenCV.
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    return camera_matrix, dist_coeffs


# --------------------------------------------------
# Helper functions: geometry and projection
# --------------------------------------------------

def make_transform_from_rvec_tvec(rvec, tvec):
    """
    Convert OpenCV rvec/tvec into a 4x4 homogeneous transform.

    T_camera_board maps a point from board frame to camera frame:

        p_camera = T_camera_board @ p_board_homogeneous
    """

    R, _ = cv2.Rodrigues(rvec)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)

    return T


def transform_point(T, p_board):
    """
    Transform one 3D point from board frame to camera frame.

    Input:
        T:
            4x4 transform from board to camera

        p_board:
            3D point in board coordinates

    Output:
        3D point in camera coordinates
    """

    p_h = np.array(
        [p_board[0], p_board[1], p_board[2], 1.0],
        dtype=np.float64,
    )

    p_camera_h = T @ p_h

    return p_camera_h[:3]


def get_board_center_point():
    """
    Get the physical center of the ChArUco pattern in board coordinates.

    For a 4x4 board:
        width  = 4 * square_length
        height = 4 * square_length

    The center is:
        x = width / 2
        y = height / 2
        z = 0
    """

    return np.array(
        [
            0.5 * SQUARES_X * SQUARE_LENGTH_M,
            0.5 * SQUARES_Y * SQUARE_LENGTH_M,
            0.0,
        ],
        dtype=np.float64,
    )


def project_board_point(p_board, rvec, tvec, camera_matrix, dist_coeffs):
    """
    Project one 3D point from board coordinates into image pixel coordinates.

    This is useful for checking:
        - where the estimated board origin lands in the image
        - where the estimated board center lands in the image
    """

    p = np.array([p_board], dtype=np.float64).reshape(1, 1, 3)

    img_pt, _ = cv2.projectPoints(
        p,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    return img_pt.reshape(2)


def compute_reprojection_error(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs):
    """
    Compute mean and max reprojection error.

    Reprojection error checks how well the estimated pose explains the detected points.

    Steps:
        1. Take the known 3D board points.
        2. Project them into the image using the estimated pose.
        3. Compare projected pixels against detected pixels.

    Low reprojection error means the pose is fitting the detected corners well.
    """

    projected_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    image_points_2d = np.asarray(image_points).reshape(-1, 2)
    projected_points_2d = np.asarray(projected_points).reshape(-1, 2)

    errors = np.linalg.norm(
        image_points_2d - projected_points_2d,
        axis=1,
    )

    mean_err = float(np.mean(errors))
    max_err = float(np.max(errors))

    return mean_err, max_err, projected_points_2d



def compute_axis_diagnostics(T_camera_board):
    """
    Numerically verify the three board axes contained in R_camera_board.

    For T_camera_board = ^C T_W, the columns of R_camera_board are:
        column 0: board +X axis expressed in the camera frame
        column 1: board +Y axis expressed in the camera frame
        column 2: board +Z axis expressed in the camera frame

    A valid rotation matrix must have:
        unit-length columns
        pairwise dot products equal to zero
        x cross y equal to z
        R.T @ R equal to identity
        determinant equal to +1
    """

    R = np.asarray(T_camera_board[:3, :3], dtype=np.float64)

    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]

    dot_xy = float(np.dot(x_axis, y_axis))
    dot_xz = float(np.dot(x_axis, z_axis))
    dot_yz = float(np.dot(y_axis, z_axis))

    norm_x = float(np.linalg.norm(x_axis))
    norm_y = float(np.linalg.norm(y_axis))
    norm_z = float(np.linalg.norm(z_axis))

    cross_xy = np.cross(x_axis, y_axis)
    cross_error = float(np.linalg.norm(cross_xy - z_axis))

    rt_r = R.T @ R
    orthogonality_error = float(np.linalg.norm(rt_r - np.eye(3), ord="fro"))
    determinant = float(np.linalg.det(R))

    axes_perpendicular = (
        abs(dot_xy) <= AXIS_DOT_TOL
        and abs(dot_xz) <= AXIS_DOT_TOL
        and abs(dot_yz) <= AXIS_DOT_TOL
    )

    axes_unit_length = (
        abs(norm_x - 1.0) <= AXIS_NORM_TOL
        and abs(norm_y - 1.0) <= AXIS_NORM_TOL
        and abs(norm_z - 1.0) <= AXIS_NORM_TOL
    )

    right_handed = (
        cross_error <= ORTHOGONALITY_MATRIX_TOL
        and abs(determinant - 1.0) <= DETERMINANT_TOL
    )

    valid_rotation = (
        axes_perpendicular
        and axes_unit_length
        and right_handed
        and orthogonality_error <= ORTHOGONALITY_MATRIX_TOL
    )

    return {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "dot_xy": dot_xy,
        "dot_xz": dot_xz,
        "dot_yz": dot_yz,
        "norm_x": norm_x,
        "norm_y": norm_y,
        "norm_z": norm_z,
        "cross_xy": cross_xy,
        "cross_error": cross_error,
        "rt_r": rt_r,
        "orthogonality_error": orthogonality_error,
        "determinant": determinant,
        "axes_perpendicular": axes_perpendicular,
        "axes_unit_length": axes_unit_length,
        "right_handed": right_handed,
        "valid_rotation": valid_rotation,
    }


def angle_between_vectors_deg(a, b):
    """Return the angle between two nonzero vectors in degrees."""

    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator < 1e-12:
        return float("nan")

    cosine = float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def compute_projected_axis_diagnostics(rvec, tvec, camera_matrix, dist_coeffs):
    """
    Project the origin and three axis endpoints into the image.

    These 2D image angles are reported only to explain the visualization.
    Perspective projection does not preserve 3D right angles, so the projected
    image angles are not expected to equal 90 degrees.
    """

    axis_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [AXIS_LENGTH_M, 0.0, 0.0],
            [0.0, AXIS_LENGTH_M, 0.0],
            [0.0, 0.0, AXIS_LENGTH_M],
        ],
        dtype=np.float64,
    )

    pixels, _ = cv2.projectPoints(
        axis_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)

    origin_px = pixels[0]
    x_vector_px = pixels[1] - origin_px
    y_vector_px = pixels[2] - origin_px
    z_vector_px = pixels[3] - origin_px

    return {
        "origin_px": origin_px,
        "x_endpoint_px": pixels[1],
        "y_endpoint_px": pixels[2],
        "z_endpoint_px": pixels[3],
        "angle_xy_deg": angle_between_vectors_deg(x_vector_px, y_vector_px),
        "angle_xz_deg": angle_between_vectors_deg(x_vector_px, z_vector_px),
        "angle_yz_deg": angle_between_vectors_deg(y_vector_px, z_vector_px),
    }


def append_pose_report(result, camera_matrix, frame_w, frame_h, tvec_history):
    """Append one easy-to-read solvePnP and axis-validation report block."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    T_camera_board = result["T_camera_board"]
    axis = compute_axis_diagnostics(T_camera_board)
    projected = compute_projected_axis_diagnostics(
        result["rvec"],
        result["tvec"],
        camera_matrix,
        np.zeros((5, 1), dtype=np.float64),
    )

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("=" * 78)
    lines.append(f"SOLVEPNP AXIS TEST   {timestamp}")
    lines.append("=" * 78)
    lines.append(f"Detected ChArUco corners : {result['num_charuco']}")
    lines.append(f"PnP method               : {result['pnp_method']}")
    lines.append(f"Image size               : {frame_w} x {frame_h}")
    lines.append(
        f"Reprojection error       : mean={result['mean_reproj_error_px']:.6f} px, "
        f"max={result['max_reproj_error_px']:.6f} px"
    )

    lines.append("")
    lines.append("T_camera_board = ^C T_W")
    lines.append("Maps board-frame coordinates into the ZED LEFT camera frame.")
    lines.append(np.array2string(T_camera_board, precision=9, suppress_small=True))

    lines.append("")
    lines.append("BOARD AXES EXPRESSED IN CAMERA FRAME")
    lines.append(f"x_axis = {np.array2string(axis['x_axis'], precision=9, suppress_small=True)}")
    lines.append(f"y_axis = {np.array2string(axis['y_axis'], precision=9, suppress_small=True)}")
    lines.append(f"z_axis = {np.array2string(axis['z_axis'], precision=9, suppress_small=True)}")

    lines.append("")
    lines.append("3D PERPENDICULARITY CHECK")
    lines.append(f"x dot y = {axis['dot_xy']:+.12e}")
    lines.append(f"x dot z = {axis['dot_xz']:+.12e}")
    lines.append(f"y dot z = {axis['dot_yz']:+.12e}")
    lines.append(f"Axes perpendicular: {'PASS' if axis['axes_perpendicular'] else 'FAIL'}")

    lines.append("")
    lines.append("UNIT-LENGTH CHECK")
    lines.append(f"|x| = {axis['norm_x']:.12f}")
    lines.append(f"|y| = {axis['norm_y']:.12f}")
    lines.append(f"|z| = {axis['norm_z']:.12f}")
    lines.append(f"Axes unit length: {'PASS' if axis['axes_unit_length'] else 'FAIL'}")

    lines.append("")
    lines.append("RIGHT-HANDED ROTATION CHECK")
    lines.append(f"x cross y = {np.array2string(axis['cross_xy'], precision=9, suppress_small=True)}")
    lines.append(f"z axis    = {np.array2string(axis['z_axis'], precision=9, suppress_small=True)}")
    lines.append(f"|x cross y - z| = {axis['cross_error']:.12e}")
    lines.append(f"det(R) = {axis['determinant']:.12f}")
    lines.append(f"Right handed: {'PASS' if axis['right_handed'] else 'FAIL'}")

    lines.append("")
    lines.append("ROTATION-MATRIX CHECK")
    lines.append("R.T @ R =")
    lines.append(np.array2string(axis["rt_r"], precision=12, suppress_small=True))
    lines.append(f"||R.T @ R - I||_F = {axis['orthogonality_error']:.12e}")
    lines.append(f"Valid SO(3) rotation: {'PASS' if axis['valid_rotation'] else 'FAIL'}")

    lines.append("")
    lines.append("2D PROJECTED AXIS ANGLES IN THE IMAGE")
    lines.append("These are NOT expected to be 90 degrees because perspective projection")
    lines.append("does not preserve 3D angles.")
    lines.append(f"projected x-y angle = {projected['angle_xy_deg']:.6f} deg")
    lines.append(f"projected x-z angle = {projected['angle_xz_deg']:.6f} deg")
    lines.append(f"projected y-z angle = {projected['angle_yz_deg']:.6f} deg")

    lines.append("")
    lines.append("POSE LOCATION")
    origin = result["origin_camera"]
    center = result["center_camera"]
    lines.append(
        f"Board origin in camera [m] : x={origin[0]:+.6f}, "
        f"y={origin[1]:+.6f}, z={origin[2]:+.6f}"
    )
    lines.append(
        f"Board center in camera [m] : x={center[0]:+.6f}, "
        f"y={center[1]:+.6f}, z={center[2]:+.6f}"
    )

    if len(tvec_history) >= 5:
        hist = np.asarray(tvec_history, dtype=np.float64)
        med = np.median(hist, axis=0)
        std = np.std(hist, axis=0)
        lines.append("")
        lines.append(f"RECENT TVEC JITTER ({len(tvec_history)} good frames)")
        lines.append(
            f"Median [mm] : x={med[0]*1000:+.3f}, "
            f"y={med[1]*1000:+.3f}, z={med[2]*1000:+.3f}"
        )
        lines.append(
            f"Std [mm]    : x={std[0]*1000:.3f}, "
            f"y={std[1]*1000:.3f}, z={std[2]*1000:.3f}"
        )

    lines.append("")

    report = "\n".join(lines)

    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"[SAVED] Axis report appended to: {REPORT_FILE}")


# --------------------------------------------------
# Helper functions: ChArUco detection
# --------------------------------------------------

def solve_planar_pnp(object_points, image_points, camera_matrix, dist_coeffs):
    """
    Solve pose for a planar board.

    Since ChArUco is a flat target, we prefer SOLVEPNP_IPPE.
    IPPE is designed for planar pose estimation.

    Then we refine the pose using solvePnPRefineLM when available.
    """

    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)

    pnp_method = "SOLVEPNP_IPPE + RefineLM"

    try:
        ok, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )

        if not ok or len(rvecs) == 0:
            return None, None, "SOLVEPNP_IPPE failed"

        # solvePnPGeneric can return multiple planar solutions.
        # Choose the one with the lowest reprojection error.
        best_idx = int(np.argmin(np.asarray(reproj_errs).reshape(-1)))

        rvec = rvecs[best_idx]
        tvec = tvecs[best_idx]

    except Exception:
        # Fallback if IPPE is unavailable or fails.
        pnp_method = "SOLVEPNP_ITERATIVE"

        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            return None, None, "SOLVEPNP_ITERATIVE failed"

    # Refine the pose using Levenberg-Marquardt if OpenCV has it.
    if hasattr(cv2, "solvePnPRefineLM"):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                rvec,
                tvec,
            )
        except Exception:
            # Refinement is optional. If it fails, keep the original pose.
            pass

    return rvec, tvec, pnp_method


def detect_charuco_pose_new_api(gray, board, camera_matrix, dist_coeffs):
    """
    Detect ChArUco board using the newer OpenCV API.

    Newer workflow:
        1. cv2.aruco.CharucoDetector(board)
        2. detector.detectBoard(gray)
        3. board.matchImagePoints(charuco_corners, charuco_ids)
        4. solvePnP with the matched object/image points
    """

    detector = cv2.aruco.CharucoDetector(board)

    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    if charuco_corners is None or charuco_ids is None:
        return None

    if len(charuco_corners) < MIN_CHARUCO_CORNERS:
        return None

    object_points, image_points = board.matchImagePoints(
        charuco_corners,
        charuco_ids,
    )

    if object_points is None or image_points is None:
        return None

    if len(object_points) < MIN_CHARUCO_CORNERS:
        return None

    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)

    rvec, tvec, pnp_method = solve_planar_pnp(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
    )

    if rvec is None or tvec is None:
        return None

    mean_err_px, max_err_px, projected_points = compute_reprojection_error(
        object_points,
        image_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    T_camera_board = make_transform_from_rvec_tvec(rvec, tvec)

    origin_board = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    center_board = get_board_center_point()

    origin_camera = transform_point(T_camera_board, origin_board)
    center_camera = transform_point(T_camera_board, center_board)

    projected_origin_px = project_board_point(
        origin_board,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    projected_center_px = project_board_point(
        center_board,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    return {
        "rvec": rvec,
        "tvec": tvec,
        "T_camera_board": T_camera_board,
        "pnp_method": pnp_method,
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
        "marker_corners": marker_corners,
        "marker_ids": marker_ids,
        "object_points": object_points,
        "image_points": image_points,
        "projected_points": projected_points,
        "num_charuco": len(charuco_corners),
        "mean_reproj_error_px": mean_err_px,
        "max_reproj_error_px": max_err_px,
        "origin_camera": origin_camera,
        "center_camera": center_camera,
        "projected_origin_px": projected_origin_px,
        "projected_center_px": projected_center_px,
    }


def detect_charuco_pose_old_api(gray, board, dictionary, camera_matrix, dist_coeffs):
    """
    Detect ChArUco board using the older OpenCV API.

    Older workflow:
        1. detectMarkers
        2. interpolateCornersCharuco
        3. matchImagePoints or estimatePoseCharucoBoard
    """

    aruco_params = cv2.aruco.DetectorParameters()

    marker_corners, marker_ids, rejected = cv2.aruco.detectMarkers(
        gray,
        dictionary,
        parameters=aruco_params,
    )

    if marker_ids is None or len(marker_ids) == 0:
        return None

    ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        gray,
        board,
        camera_matrix,
        dist_coeffs,
    )

    if charuco_corners is None or charuco_ids is None:
        return None

    if len(charuco_corners) < MIN_CHARUCO_CORNERS:
        return None

    # Prefer board.matchImagePoints if available.
    if hasattr(board, "matchImagePoints"):
        object_points, image_points = board.matchImagePoints(
            charuco_corners,
            charuco_ids,
        )

        object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
        image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)

        rvec, tvec, pnp_method = solve_planar_pnp(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
        )

        if rvec is None or tvec is None:
            return None

    else:
        # Full fallback for old OpenCV.
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners,
            charuco_ids,
            board,
            camera_matrix,
            dist_coeffs,
            None,
            None,
        )

        if not ok:
            return None

        pnp_method = "estimatePoseCharucoBoard"

        # In this old fallback, object_points/image_points may not be accessible.
        object_points = np.empty((0, 3), dtype=np.float64)
        image_points = np.empty((0, 2), dtype=np.float64)

    if len(object_points) > 0:
        mean_err_px, max_err_px, projected_points = compute_reprojection_error(
            object_points,
            image_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
    else:
        mean_err_px = float("nan")
        max_err_px = float("nan")
        projected_points = np.empty((0, 2), dtype=np.float64)

    T_camera_board = make_transform_from_rvec_tvec(rvec, tvec)

    origin_board = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    center_board = get_board_center_point()

    origin_camera = transform_point(T_camera_board, origin_board)
    center_camera = transform_point(T_camera_board, center_board)

    projected_origin_px = project_board_point(
        origin_board,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    projected_center_px = project_board_point(
        center_board,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    return {
        "rvec": rvec,
        "tvec": tvec,
        "T_camera_board": T_camera_board,
        "pnp_method": pnp_method,
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
        "marker_corners": marker_corners,
        "marker_ids": marker_ids,
        "object_points": object_points,
        "image_points": image_points,
        "projected_points": projected_points,
        "num_charuco": len(charuco_corners),
        "mean_reproj_error_px": mean_err_px,
        "max_reproj_error_px": max_err_px,
        "origin_camera": origin_camera,
        "center_camera": center_camera,
        "projected_origin_px": projected_origin_px,
        "projected_center_px": projected_center_px,
    }


def detect_charuco_pose(gray, board, dictionary, camera_matrix, dist_coeffs):
    """
    Try new OpenCV ChArUco API first.
    Fall back to old API if needed.
    """

    if hasattr(cv2.aruco, "CharucoDetector") and hasattr(board, "matchImagePoints"):
        try:
            result = detect_charuco_pose_new_api(
                gray,
                board,
                camera_matrix,
                dist_coeffs,
            )

            if result is not None:
                return result

        except Exception as e:
            print(f"[WARN] New ChArUco API failed once: {e}")

    if hasattr(cv2.aruco, "estimatePoseCharucoBoard"):
        try:
            result = detect_charuco_pose_old_api(
                gray,
                board,
                dictionary,
                camera_matrix,
                dist_coeffs,
            )

            return result

        except Exception as e:
            print(f"[WARN] Old ChArUco API failed once: {e}")

    return None


# --------------------------------------------------
# Helper functions: drawing
# --------------------------------------------------

def draw_detection(frame_bgr, result, camera_matrix, dist_coeffs):
    """
    Draw detected markers, ChArUco corners, pose axes, and board center.

    The colored axes are drawn at the OpenCV board origin.
    The yellow dot is the physical board-pattern center.
    """

    marker_corners = result["marker_corners"]
    marker_ids = result["marker_ids"]
    charuco_corners = result["charuco_corners"]
    charuco_ids = result["charuco_ids"]
    rvec = result["rvec"]
    tvec = result["tvec"]

    # Draw ArUco marker boundaries.
    if marker_corners is not None and marker_ids is not None:
        try:
            cv2.aruco.drawDetectedMarkers(
                frame_bgr,
                marker_corners,
                marker_ids,
            )
        except Exception as e:
            print(f"[WARN] Could not draw detected markers: {e}")

    # Draw ChArUco corners safely.
    # Some OpenCV versions complain if corner/id shapes do not match perfectly.
    if charuco_corners is not None:
        try:
            cv2.aruco.drawDetectedCornersCharuco(
                frame_bgr,
                charuco_corners,
                None,
            )
        except Exception:
            pts = np.asarray(charuco_corners).reshape(-1, 2)

            ids = None
            if charuco_ids is not None:
                ids = np.asarray(charuco_ids).reshape(-1)

            for i, p in enumerate(pts):
                x, y = int(round(p[0])), int(round(p[1]))

                cv2.circle(frame_bgr, (x, y), 5, (255, 0, 255), -1)

                if ids is not None and i < len(ids):
                    cv2.putText(
                        frame_bgr,
                        str(int(ids[i])),
                        (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 0, 255),
                        1,
                    )

    # Draw the OpenCV board coordinate frame.
    try:
        cv2.drawFrameAxes(
            frame_bgr,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
            AXIS_LENGTH_M,
        )
    except Exception as e:
        print(f"[WARN] Could not draw frame axes: {e}")

    # Numerically test the actual 3D rotation matrix and display the result.
    axis_check = compute_axis_diagnostics(result["T_camera_board"])
    axis_text = (
        "3D axes orthogonal: PASS"
        if axis_check["valid_rotation"]
        else "3D axes orthogonal: FAIL"
    )
    axis_color = (0, 255, 0) if axis_check["valid_rotation"] else (0, 0, 255)
    cv2.putText(
        frame_bgr,
        axis_text,
        (25, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        axis_color,
        2,
    )

    # Draw physical board center.
    center_px = result["projected_center_px"]
    cx, cy = int(round(center_px[0])), int(round(center_px[1]))

    cv2.circle(frame_bgr, (cx, cy), 8, (0, 255, 255), -1)
    cv2.putText(
        frame_bgr,
        "physical board center",
        (cx + 10, cy - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
    )


def draw_status_text(frame_bgr, result):
    """
    Draw a compact status line in the upper-left corner of the image.
    """

    if result is None:
        text = "No valid 4x4 ChArUco pose"
        color = (0, 0, 255)
    else:
        text = (
            f"4x4 ChArUco pose | "
            f"corners={result['num_charuco']} | "
            f"reproj={result['mean_reproj_error_px']:.2f}px"
        )
        color = (0, 255, 0)

    cv2.putText(
        frame_bgr,
        text,
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("[INFO] OpenCV version:", cv2.__version__)
    print("[INFO] OpenCV path:", cv2.__file__)

    board, dictionary = make_charuco_board()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("SOLVEPNP / CHARUCO AXIS VALIDATION REPORT\n")
        f.write("A new block is appended every 5 seconds while a valid pose is visible.\n\n")

    zed = open_zed_camera()

    camera_matrix, dist_coeffs = get_zed_left_camera_matrix(zed)

    print("")
    print("[INFO] ZED left camera matrix:")
    print(np.array2string(camera_matrix, precision=4, suppress_small=True))

    print("")
    print("[INFO] Distortion coefficients used:")
    print(dist_coeffs.reshape(-1))

    print("")
    print("[INFO] 4x4 ChArUco board parameters:")
    print(f"  squares_x:       {SQUARES_X}")
    print(f"  squares_y:       {SQUARES_Y}")
    print(f"  square_length_m: {SQUARE_LENGTH_M}")
    print(f"  marker_length_m: {MARKER_LENGTH_M}")
    print(f"  max corners:     {(SQUARES_X - 1) * (SQUARES_Y - 1)}")
    print(f"  min corners:     {MIN_CHARUCO_CORNERS}")

    print("")
    print("[INFO] Press 'q' or ESC to quit.")
    print("")

    runtime_params = sl.RuntimeParameters()
    left_image = sl.Mat()

    last_print_time = 0.0
    tvec_history = deque(maxlen=TVEC_HISTORY_SIZE)

    try:
        while True:
            err = zed.grab(runtime_params)

            if err != sl.ERROR_CODE.SUCCESS:
                continue

            # Retrieve rectified left image only.
            # No depth. No point cloud.
            zed.retrieve_image(left_image, sl.VIEW.LEFT)

            frame_bgra = left_image.get_data()

            # ZED image usually comes as BGRA.
            frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            frame_h, frame_w = frame_bgr.shape[:2]

            result = detect_charuco_pose(
                gray,
                board,
                dictionary,
                camera_matrix,
                dist_coeffs,
            )

            if result is not None:
                tvec_history.append(result["tvec"].reshape(3))

                draw_detection(
                    frame_bgr,
                    result,
                    camera_matrix,
                    dist_coeffs,
                )

            draw_status_text(frame_bgr, result)

            now = time.time()

            if result is not None and now - last_print_time >= PRINT_INTERVAL_SEC:
                last_print_time = now

                append_pose_report(
                    result=result,
                    camera_matrix=camera_matrix,
                    frame_w=frame_w,
                    frame_h=frame_h,
                    tvec_history=tvec_history,
                )

            cv2.imshow("ZED Left - 4x4 ChArUco Board Test", frame_bgr)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    finally:
        cv2.destroyAllWindows()
        zed.close()
        print("[INFO] ZED closed.")


if __name__ == "__main__":
    main()

