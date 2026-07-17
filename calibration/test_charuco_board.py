#!/usr/bin/env python3

"""
Debug ChArUco board pose using the ZED as a normal monocular camera.

This script:
    - Opens the ZED camera
    - Uses only the LEFT rectified image
    - Does NOT use ZED depth or point cloud
    - Detects a 5x5 ChArUco board
    - Estimates T_camera_board using solvePnP
    - Prints detailed debugging information for position errors:
        * image shape
        * camera matrix
        * object point min/max
        * image point min/max
        * reprojection error
        * board origin in camera frame
        * board center in camera frame
        * projected origin/center pixels
        * T_camera_board

Your measured printed board:
    - 5 x 5 squares
    - square length ≈ 38.5 mm
    - marker length ≈ 26.9 mm

Coordinate meaning:
    tvec = [x, y, z] is the location of the ChArUco board origin
           expressed in the ZED left camera frame.

For ZED IMAGE/OpenCV camera convention:
    +X = right in image
    +Y = down in image
    +Z = forward away from camera
"""

import sys
import time
from collections import deque

import cv2
import numpy as np
import pyzed.sl as sl


# --------------------------------------------------
# ChArUco board settings
# --------------------------------------------------

SQUARES_X = 5
SQUARES_Y = 5

# Use your measured printed dimensions.
# If you later measure horizontal and vertical square sizes separately,
# use the average here first. OpenCV CharucoBoard assumes true squares.
SQUARE_LENGTH_M = 0.0385   # 38.5 mm
MARKER_LENGTH_M = 0.0269   # 26.9 mm

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50

# For 5x5 ChArUco, max internal ChArUco corners = 4x4 = 16.
# For debugging/calibration, only trust strong detections.
MIN_CHARUCO_CORNERS = 12

# Print debug info every N seconds instead of spamming every frame.
PRINT_INTERVAL_SEC = 5.0

# Axis length drawn on the image, in meters. This is visualization only.
AXIS_LENGTH_M = 0.05

# Keep a short history to show whether position is jittering.
TVEC_HISTORY_LEN = 20
GOOD_REPROJ_MEAN_THRESHOLD_PX = 1.0


# --------------------------------------------------
# ZED settings
# --------------------------------------------------

# Important: use the same resolution you will use during final calibration.
ZED_RESOLUTION = sl.RESOLUTION.HD720
ZED_FPS = 15


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def make_charuco_board():
    """
    Create OpenCV ChArUco board.
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


def print_board_model_debug(board):
    """
    Print what OpenCV thinks the board geometry is.

    This is useful because if object points are in the wrong units/range,
    tvec will be wrong even if rotation looks correct.
    """

    print("")
    print("[DEBUG] OpenCV board model:")

    if hasattr(board, "getChessboardSize"):
        print(f"  chessboard size: {board.getChessboardSize()}")

    if hasattr(board, "getSquareLength"):
        print(f"  square length from board: {board.getSquareLength()} m")

    if hasattr(board, "getMarkerLength"):
        print(f"  marker length from board: {board.getMarkerLength()} m")

    if hasattr(board, "getRightBottomCorner"):
        rb = np.asarray(board.getRightBottomCorner(), dtype=np.float64).reshape(3)
        print("  right-bottom/full-board corner from board:")
        print(f"    x={rb[0]:+.6f} m, y={rb[1]:+.6f} m, z={rb[2]:+.6f} m")

    if hasattr(board, "getChessboardCorners"):
        corners = np.asarray(board.getChessboardCorners(), dtype=np.float64).reshape(-1, 3)
        print("  internal ChArUco corner object-point range:")
        print(f"    min: {corners.min(axis=0)}")
        print(f"    max: {corners.max(axis=0)}")
        print(f"    count: {len(corners)}")

    expected_width = SQUARES_X * SQUARE_LENGTH_M
    expected_height = SQUARES_Y * SQUARE_LENGTH_M
    print("  expected full pattern size from constants:")
    print(f"    width:  {expected_width:.6f} m ({expected_width * 1000:.1f} mm)")
    print(f"    height: {expected_height:.6f} m ({expected_height * 1000:.1f} mm)")


def get_zed_left_camera_matrix(zed):
    """
    Get ZED left camera intrinsics.

    We use sl.VIEW.LEFT, which is the ZED rectified left image.
    For a rectified image, use calibration_parameters and zero distortion.

    If you switch to sl.VIEW.LEFT_UNRECTIFIED, then you should use
    calibration_parameters_raw and the raw distortion coefficients instead.
    """

    cam_info = zed.get_camera_information()

    # Rectified calibration, matching sl.VIEW.LEFT.
    rect_calib = cam_info.camera_configuration.calibration_parameters
    rect_left = rect_calib.left_cam

    fx = rect_left.fx
    fy = rect_left.fy
    cx = rect_left.cx
    cy = rect_left.cy

    camera_matrix = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # sl.VIEW.LEFT is already rectified by the ZED SDK, so OpenCV solvePnP
    # should not apply lens distortion again.
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    # Optional raw calibration printout for sanity checking.
    print("")
    print("[DEBUG] ZED calibration sanity check:")
    print("  Using sl.VIEW.LEFT = rectified image")
    print("  Using camera_configuration.calibration_parameters = rectified intrinsics")
    print("  Using zero distortion coefficients")

    if hasattr(cam_info.camera_configuration, "calibration_parameters_raw"):
        raw_calib = cam_info.camera_configuration.calibration_parameters_raw
        raw_left = raw_calib.left_cam
        print("  Raw left distortion exists but is NOT used with sl.VIEW.LEFT:")
        print(f"    raw fx={raw_left.fx:.6f}, fy={raw_left.fy:.6f}")
        print(f"    raw cx={raw_left.cx:.6f}, cy={raw_left.cy:.6f}")
        print(f"    raw disto={raw_left.disto}")

    return camera_matrix, dist_coeffs


def open_zed_camera():
    """
    Open ZED camera.
    """

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = ZED_RESOLUTION
    init_params.camera_fps = ZED_FPS
    init_params.depth_mode = sl.DEPTH_MODE.NONE
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


def get_charuco_object_image_points(board, charuco_corners, charuco_ids):
    """
    Convert detected ChArUco corners/IDs into 3D object points and 2D image points.

    Uses board.matchImagePoints when available. Falls back to indexing the
    board's ChArUco corner table when needed.
    """

    if charuco_corners is None or charuco_ids is None:
        return None, None

    if hasattr(board, "matchImagePoints"):
        object_points, image_points = board.matchImagePoints(
            charuco_corners,
            charuco_ids,
        )

        if object_points is not None and image_points is not None:
            object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
            image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
            return object_points, image_points

    # Fallback path for older OpenCV versions.
    ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    image_points = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)

    if hasattr(board, "getChessboardCorners"):
        all_corners = np.asarray(board.getChessboardCorners(), dtype=np.float64).reshape(-1, 3)
    elif hasattr(board, "chessboardCorners"):
        all_corners = np.asarray(board.chessboardCorners, dtype=np.float64).reshape(-1, 3)
    else:
        return None, None

    if np.any(ids < 0) or np.any(ids >= len(all_corners)):
        return None, None

    object_points = all_corners[ids]

    return object_points, image_points


def compute_reprojection_error(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs):
    """
    Compute mean and max reprojection error in pixels.

    This is the most important number for deciding whether the pose actually
    matches the detected corners.
    """

    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)

    projected_points, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    projected_points = np.asarray(projected_points, dtype=np.float64).reshape(-1, 2)

    errors = np.linalg.norm(image_points - projected_points, axis=1)

    return float(np.mean(errors)), float(np.max(errors)), errors


def estimate_pose_from_points(object_points, image_points, camera_matrix, dist_coeffs):
    """
    Estimate pose from 3D board points and 2D image points.

    For a flat ChArUco board, IPPE can be more appropriate than the default
    iterative method. We try IPPE first, then refine with solvePnPRefineLM.
    If IPPE fails, we fall back to SOLVEPNP_ITERATIVE.
    """

    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)

    if len(object_points) < MIN_CHARUCO_CORNERS:
        return None

    method = "UNKNOWN"
    rvec = None
    tvec = None

    # Try IPPE for planar target.
    if hasattr(cv2, "solvePnPGeneric") and hasattr(cv2, "SOLVEPNP_IPPE"):
        try:
            ok, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE,
            )

            if ok and len(rvecs) > 0:
                if reproj_errs is not None and len(reproj_errs) > 0:
                    best_idx = int(np.argmin(np.asarray(reproj_errs).reshape(-1)))
                else:
                    best_idx = 0

                rvec = np.asarray(rvecs[best_idx], dtype=np.float64).reshape(3, 1)
                tvec = np.asarray(tvecs[best_idx], dtype=np.float64).reshape(3, 1)
                method = "SOLVEPNP_IPPE"
        except Exception as e:
            print(f"[WARN] SOLVEPNP_IPPE failed, falling back to ITERATIVE: {e}")

    # Fallback to standard iterative solvePnP.
    if rvec is None or tvec is None:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            return None

        method = "SOLVEPNP_ITERATIVE"

    # Refine pose if available.
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
            method += " + RefineLM"
        except Exception as e:
            print(f"[WARN] solvePnPRefineLM failed: {e}")

    mean_err, max_err, per_point_err = compute_reprojection_error(
        object_points,
        image_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    return {
        "rvec": rvec,
        "tvec": tvec,
        "pnp_method": method,
        "mean_reproj_error_px": mean_err,
        "max_reproj_error_px": max_err,
        "per_point_reproj_error_px": per_point_err,
    }


def detect_charuco_pose_new_api(gray, board, camera_matrix, dist_coeffs):
    """
    Detect ChArUco pose using the newer OpenCV API:
        cv2.aruco.CharucoDetector
        board.matchImagePoints
        solvePnP / solvePnPGeneric
    """

    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    if charuco_corners is None or charuco_ids is None:
        return None

    if len(charuco_corners) < MIN_CHARUCO_CORNERS:
        return None

    object_points, image_points = get_charuco_object_image_points(
        board,
        charuco_corners,
        charuco_ids,
    )

    if object_points is None or image_points is None:
        return None

    if len(object_points) < MIN_CHARUCO_CORNERS:
        return None

    pose = estimate_pose_from_points(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
    )

    if pose is None:
        return None

    result = {
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
        "marker_corners": marker_corners,
        "marker_ids": marker_ids,
        "num_charuco": len(charuco_corners),
        "object_points": object_points,
        "image_points": image_points,
    }
    result.update(pose)

    return result


def detect_charuco_pose_old_api(gray, board, dictionary, camera_matrix, dist_coeffs):
    """
    Detect ChArUco pose using older OpenCV API.

    This still tries to use our own solvePnP path so the same debug information
    is available.
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

    object_points, image_points = get_charuco_object_image_points(
        board,
        charuco_corners,
        charuco_ids,
    )

    if object_points is None or image_points is None:
        return None

    pose = estimate_pose_from_points(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
    )

    if pose is None:
        return None

    result = {
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
        "marker_corners": marker_corners,
        "marker_ids": marker_ids,
        "num_charuco": len(charuco_corners),
        "object_points": object_points,
        "image_points": image_points,
    }
    result.update(pose)

    return result


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


def make_transform_from_rvec_tvec(rvec, tvec):
    """
    Convert rvec/tvec to a 4x4 homogeneous transform T_camera_board.
    """

    R, _ = cv2.Rodrigues(rvec)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)

    return T


def transform_point(T, p_board):
    """
    Transform a 3D point from board frame into camera frame.
    """

    p_h = np.array(
        [p_board[0], p_board[1], p_board[2], 1.0],
        dtype=np.float64,
    )

    p_cam_h = T @ p_h

    return p_cam_h[:3]


def project_board_point(p_board, rvec, tvec, camera_matrix, dist_coeffs):
    """
    Project one board-frame point into image pixels.
    """

    p = np.asarray(p_board, dtype=np.float64).reshape(1, 1, 3)

    img_pt, _ = cv2.projectPoints(
        p,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    return img_pt.reshape(2)


def get_board_center_point(board):
    """
    Return full-board center in board coordinates.

    Prefer OpenCV's own right-bottom corner if available, because that tells us
    what OpenCV thinks the full board size is.
    """

    if hasattr(board, "getRightBottomCorner"):
        rb = np.asarray(board.getRightBottomCorner(), dtype=np.float64).reshape(3)
        return np.array([0.5 * rb[0], 0.5 * rb[1], 0.0], dtype=np.float64)

    return np.array(
        [
            0.5 * SQUARES_X * SQUARE_LENGTH_M,
            0.5 * SQUARES_Y * SQUARE_LENGTH_M,
            0.0,
        ],
        dtype=np.float64,
    )


def draw_detection(frame_bgr, result, camera_matrix, dist_coeffs, board_center_px=None):
    """
    Draw detected markers, ChArUco corners, coordinate axes, and projected center.
    """

    marker_corners = result["marker_corners"]
    marker_ids = result["marker_ids"]
    charuco_corners = result["charuco_corners"]
    charuco_ids = result["charuco_ids"]
    rvec = result["rvec"]
    tvec = result["tvec"]

    # Draw ArUco marker boxes.
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
                        0.4,
                        (255, 0, 255),
                        1,
                    )

    # Draw board coordinate frame.
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

    # Draw projected full-board center.
    if board_center_px is not None:
        u, v = board_center_px
        u_i, v_i = int(round(u)), int(round(v))
        cv2.circle(frame_bgr, (u_i, v_i), 8, (0, 255, 255), -1)
        cv2.putText(
            frame_bgr,
            "projected board center",
            (u_i + 10, v_i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )


def print_pose_debug(result, T_camera_board, board_center_board, camera_matrix, dist_coeffs, frame_bgr):
    """
    Print all useful numbers for debugging wrong position/tvec.
    """

    rvec = result["rvec"]
    tvec = result["tvec"]
    object_points = np.asarray(result["object_points"], dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(result["image_points"], dtype=np.float64).reshape(-1, 2)

    origin_camera = tvec.reshape(3)
    center_camera = transform_point(T_camera_board, board_center_board)

    origin_px = project_board_point(
        [0.0, 0.0, 0.0],
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    center_px = project_board_point(
        board_center_board,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )

    h, w = frame_bgr.shape[:2]

    print("--------------------------------------------------")
    print(f"Detected ChArUco corners: {result['num_charuco']}")
    print(f"PnP method: {result['pnp_method']}")
    print(f"Image shape used for detection: width={w}, height={h}")
    print("")
    print("Camera matrix used:")
    print(np.array2string(camera_matrix, precision=4, suppress_small=True))
    print("")
    print("Object points used by solvePnP, board frame, meters:")
    print(f"  min xyz: {object_points.min(axis=0)}")
    print(f"  max xyz: {object_points.max(axis=0)}")
    print(f"  count:   {len(object_points)}")
    print("")
    print("Image points used by solvePnP, pixels:")
    print(f"  min uv: {image_points.min(axis=0)}")
    print(f"  max uv: {image_points.max(axis=0)}")
    print(f"  count:  {len(image_points)}")
    print("")
    print("Reprojection error:")
    print(f"  mean: {result['mean_reproj_error_px']:.4f} px")
    print(f"  max:  {result['max_reproj_error_px']:.4f} px")
    print("")
    print("Board ORIGIN in ZED LEFT camera frame:")
    print(f"  meters:      x={origin_camera[0]:+.4f}, y={origin_camera[1]:+.4f}, z={origin_camera[2]:+.4f}")
    print(f"  millimeters: x={origin_camera[0]*1000:+.1f}, y={origin_camera[1]*1000:+.1f}, z={origin_camera[2]*1000:+.1f}")
    print("")
    print("Board CENTER in ZED LEFT camera frame:")
    print(f"  board center used: x={board_center_board[0]:+.4f}, y={board_center_board[1]:+.4f}, z={board_center_board[2]:+.4f} m")
    print(f"  meters:      x={center_camera[0]:+.4f}, y={center_camera[1]:+.4f}, z={center_camera[2]:+.4f}")
    print(f"  millimeters: x={center_camera[0]*1000:+.1f}, y={center_camera[1]*1000:+.1f}, z={center_camera[2]*1000:+.1f}")
    print("")
    print("Projected pixels from the estimated pose:")
    print(f"  projected origin pixel: u={origin_px[0]:.1f}, v={origin_px[1]:.1f}")
    print(f"  projected center pixel: u={center_px[0]:.1f}, v={center_px[1]:.1f}")
    print(f"  image center pixel:     u={w/2:.1f}, v={h/2:.1f}")
    print("")
    print("T_camera_board:")
    print(np.array2string(T_camera_board, precision=4, suppress_small=True))


def print_median_tvec_debug(tvec_history):
    """
    Print median/std of recent good tvecs so you can see jitter.
    """

    if len(tvec_history) < 5:
        return

    arr = np.asarray(tvec_history, dtype=np.float64).reshape(-1, 3)
    med = np.median(arr, axis=0)
    std = np.std(arr, axis=0)

    print("")
    print(f"Recent good tvec history: {len(arr)} frames")
    print("Median origin position:")
    print(f"  x={med[0]*1000:+.1f} mm, y={med[1]*1000:+.1f} mm, z={med[2]*1000:+.1f} mm")
    print("Std / jitter:")
    print(f"  x={std[0]*1000:.2f} mm, y={std[1]*1000:.2f} mm, z={std[2]*1000:.2f} mm")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("[INFO] OpenCV version:", cv2.__version__)
    print("[INFO] OpenCV path:", cv2.__file__)

    board, dictionary = make_charuco_board()
    print_board_model_debug(board)

    board_center_board = get_board_center_point(board)

    print("")
    print("[DEBUG] Board center point used for debugging:")
    print(f"  x={board_center_board[0]:+.6f} m")
    print(f"  y={board_center_board[1]:+.6f} m")
    print(f"  z={board_center_board[2]:+.6f} m")

    zed = open_zed_camera()

    camera_matrix, dist_coeffs = get_zed_left_camera_matrix(zed)

    print("")
    print("[INFO] ZED left camera matrix:")
    print(camera_matrix)
    print("")
    print("[INFO] Distortion coefficients used:")
    print(dist_coeffs.reshape(-1))
    print("")
    print("[INFO] ChArUco board parameters:")
    print(f"  squares_x:       {SQUARES_X}")
    print(f"  squares_y:       {SQUARES_Y}")
    print(f"  square_length_m: {SQUARE_LENGTH_M}")
    print(f"  marker_length_m: {MARKER_LENGTH_M}")
    print("")
    print("[INFO] Press 'q' or ESC to quit.")
    print("")

    runtime_params = sl.RuntimeParameters()
    left_image = sl.Mat()

    last_print_time = 0.0
    printed_frame_shape_once = False
    tvec_history = deque(maxlen=TVEC_HISTORY_LEN)

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

            if not printed_frame_shape_once:
                printed_frame_shape_once = True
                print("[DEBUG] First retrieved frame shape:")
                print(f"  frame_bgra.shape = {frame_bgra.shape}")
                print(f"  frame_bgr.shape  = {frame_bgr.shape}")
                print(f"  gray.shape       = {gray.shape}")
                print("  If you resize this frame before detection, you must also scale camera_matrix.")
                print("")

            result = detect_charuco_pose(
                gray,
                board,
                dictionary,
                camera_matrix,
                dist_coeffs,
            )

            if result is not None:
                rvec = result["rvec"]
                tvec = result["tvec"]

                T_camera_board = make_transform_from_rvec_tvec(rvec, tvec)

                board_center_px = project_board_point(
                    board_center_board,
                    rvec,
                    tvec,
                    camera_matrix,
                    dist_coeffs,
                )

                draw_detection(
                    frame_bgr,
                    result,
                    camera_matrix,
                    dist_coeffs,
                    board_center_px=board_center_px,
                )

                # Store only strong poses in the jitter history.
                if (
                    result["num_charuco"] >= MIN_CHARUCO_CORNERS
                    and result["mean_reproj_error_px"] <= GOOD_REPROJ_MEAN_THRESHOLD_PX
                ):
                    tvec_history.append(tvec.reshape(3))

                now = time.time()

                if now - last_print_time >= PRINT_INTERVAL_SEC:
                    last_print_time = now

                    print_pose_debug(
                        result,
                        T_camera_board,
                        board_center_board,
                        camera_matrix,
                        dist_coeffs,
                        frame_bgr,
                    )
                    print_median_tvec_debug(tvec_history)

                cv2.putText(
                    frame_bgr,
                    f"ChArUco pose | corners={result['num_charuco']} | reproj={result['mean_reproj_error_px']:.2f}px",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame_bgr,
                    "No valid ChArUco pose",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("ZED Left - ChArUco Debug", frame_bgr)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    finally:
        cv2.destroyAllWindows()
        zed.close()
        print("[INFO] ZED closed.")


if __name__ == "__main__":
    main()