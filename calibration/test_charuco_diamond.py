#!/usr/bin/env python3

"""
Test ChArUco diamond-style board pose using the ZED as a normal monocular camera.

This script:
    - Opens the ZED camera
    - Uses only the LEFT image
    - Does NOT use ZED depth or point cloud
    - Detects the 4 ArUco markers on the diamond board
    - Uses the known 3D marker-corner layout
    - Estimates T_camera_board using solvePnP
    - Prints board origin and board center in the ZED left camera frame
    - Prints reprojection error and position jitter

Important:
    This treats the diamond as a custom 4-marker planar ArUco board.
    That is more robust than relying on cv2.aruco.detectCharucoDiamond,
    especially because your OpenCV 5 build had missing/weird ChArUco APIs.

Coordinate convention for the board:
    Board origin = top-left corner of the full printed board/page
    Board +X     = right across the printed board
    Board +Y     = down across the printed board
    Board +Z     = out of the board plane

Coordinate convention for ZED LEFT camera:
    Camera +X = right in image
    Camera +Y = down in image
    Camera +Z = forward away from camera
"""

import sys
import time
from collections import deque

import cv2
import numpy as np
import pyzed.sl as sl


# --------------------------------------------------
# Diamond board settings
# --------------------------------------------------

# Your generated diamond board was:
#   full board/page: 200 mm x 200 mm
#   pattern:         180 mm x 180 mm
#   margin:          10 mm each side
#   square length:   60 mm
#   marker length:   42 mm
#
# IMPORTANT:
# If your print was scaled, measure these values physically and update them.
#
# If the same printer scaling happened as your 5x5 board:
#   40 mm printed as 38.5 mm
#   scale = 38.5 / 40 = 0.9625
#
# Then approximate diamond values would be:
#   BOARD_SIZE_M      = 0.1925
#   MARGIN_M          = 0.009625
#   SQUARE_LENGTH_M   = 0.05775
#   MARKER_LENGTH_M   = 0.040425
#
# For now, this uses the intended/original generated values.

BOARD_SIZE_M = 0.1737
MARGIN_M = 0.0

SQUARE_LENGTH_M = 0.0579
MARKER_LENGTH_M = 0.0404
# Marker IDs from the generator:
#   ID 0 = top marker
#   ID 1 = left marker
#   ID 2 = right marker
#   ID 3 = bottom marker
MARKER_IDS = {
    0: "top",
    1: "left",
    2: "right",
    3: "bottom",
}

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50

# Require all 4 markers for the cleanest test.
MIN_MARKERS_REQUIRED = 4

PRINT_INTERVAL_SEC = 5.0

AXIS_LENGTH_M = 0.05

# Store recent good translations to measure jitter
TVEC_HISTORY_SIZE = 20


# --------------------------------------------------
# ZED settings
# --------------------------------------------------

ZED_RESOLUTION = sl.RESOLUTION.HD720
ZED_FPS = 15


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def get_dictionary():
    return cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)


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


def get_zed_left_camera_matrix(zed):
    """
    Get ZED left camera intrinsics.

    We are using sl.VIEW.LEFT, which is the rectified left image.
    Therefore, use rectified intrinsics and zero distortion.
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

    # sl.VIEW.LEFT is rectified, so use zero distortion.
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    return camera_matrix, dist_coeffs


def make_marker_layout_object_points():
    """
    Build the 3D object points for each marker corner.

    Object point frame:
        origin = top-left corner of the full printed board/page
        +X     = right on the board
        +Y     = down on the board
        +Z     = out of board plane

    OpenCV marker corner order:
        top-left, top-right, bottom-right, bottom-left
    """

    s = SQUARE_LENGTH_M
    m = MARKER_LENGTH_M
    margin = MARGIN_M

    # Marker is centered inside its square.
    inset = 0.5 * (s - m)

    # Square locations in the 3x3 diamond pattern:
    #
    #   row 0, col 1 -> top marker, ID 0
    #   row 1, col 0 -> left marker, ID 1
    #   row 1, col 2 -> right marker, ID 2
    #   row 2, col 1 -> bottom marker, ID 3
    marker_square_locations = {
        0: (0, 1),  # top
        1: (1, 0),  # left
        2: (1, 2),  # right
        3: (2, 1),  # bottom
    }

    layout = {}

    for marker_id, (row, col) in marker_square_locations.items():
        square_x0 = margin + col * s
        square_y0 = margin + row * s

        marker_x0 = square_x0 + inset
        marker_y0 = square_y0 + inset

        marker_x1 = marker_x0 + m
        marker_y1 = marker_y0 + m

        corners_3d = np.array(
            [
                [marker_x0, marker_y0, 0.0],  # top-left
                [marker_x1, marker_y0, 0.0],  # top-right
                [marker_x1, marker_y1, 0.0],  # bottom-right
                [marker_x0, marker_y1, 0.0],  # bottom-left
            ],
            dtype=np.float64,
        )

        layout[marker_id] = corners_3d

    return layout


def detect_aruco_markers(gray, dictionary):
    """
    Detect ArUco markers with OpenCV new API if available,
    otherwise fall back to old API.
    """

    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, rejected = detector.detectMarkers(gray)
        return corners, ids, rejected

    params = cv2.aruco.DetectorParameters()
    corners, ids, rejected = cv2.aruco.detectMarkers(
        gray,
        dictionary,
        parameters=params,
    )
    return corners, ids, rejected


def build_pnp_points(marker_corners, marker_ids, marker_layout):
    """
    Match detected 2D marker corners to known 3D marker corners.
    """

    if marker_ids is None or len(marker_ids) == 0:
        return None, None, []

    object_points_list = []
    image_points_list = []
    used_ids = []

    ids_flat = np.asarray(marker_ids).reshape(-1)

    for i, marker_id_raw in enumerate(ids_flat):
        marker_id = int(marker_id_raw)

        if marker_id not in marker_layout:
            continue

        # 3D object corners for this marker
        obj_corners = marker_layout[marker_id]  # shape: 4x3

        # 2D detected image corners for this marker
        img_corners = np.asarray(marker_corners[i], dtype=np.float64).reshape(4, 2)

        object_points_list.append(obj_corners)
        image_points_list.append(img_corners)
        used_ids.append(marker_id)

    if len(used_ids) == 0:
        return None, None, []

    object_points = np.vstack(object_points_list).astype(np.float64)
    image_points = np.vstack(image_points_list).astype(np.float64)

    return object_points, image_points, used_ids


def solve_planar_pnp(object_points, image_points, camera_matrix, dist_coeffs):
    """
    Solve pose for planar marker layout.

    Uses SOLVEPNP_IPPE when available, then refines with solvePnPRefineLM.
    """

    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)

    # Prefer IPPE for planar targets
    try:
        ok, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )

        if ok and len(rvecs) > 0:
            best_idx = int(np.argmin(np.asarray(reproj_errs).reshape(-1)))
            rvec = rvecs[best_idx]
            tvec = tvecs[best_idx]
        else:
            return None, None

    except Exception:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not ok:
            return None, None

    # Refine pose
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
            pass

    return rvec, tvec


def compute_reprojection_error(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs):
    """
    Compute mean and max reprojection error in pixels.
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

    return float(np.mean(errors)), float(np.max(errors)), projected_points_2d


def make_transform_from_rvec_tvec(rvec, tvec):
    """
    Convert rvec/tvec to 4x4 T_camera_board.
    """

    R, _ = cv2.Rodrigues(rvec)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)

    return T


def transform_point(T, p_board):
    """
    Transform a 3D point from board frame to camera frame.
    """

    p_h = np.array(
        [p_board[0], p_board[1], p_board[2], 1.0],
        dtype=np.float64,
    )

    p_camera = T @ p_h

    return p_camera[:3]


def project_board_point(p_board, rvec, tvec, camera_matrix, dist_coeffs):
    """
    Project one board-frame point into image pixels.
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


def get_board_center_point():
    """
    Center of the full printed board/page in board coordinates.
    """

    return np.array(
        [
            0.5 * BOARD_SIZE_M,
            0.5 * BOARD_SIZE_M,
            0.0,
        ],
        dtype=np.float64,
    )


def draw_detection(frame_bgr, marker_corners, marker_ids, used_ids, result, camera_matrix, dist_coeffs):
    """
    Draw detected markers and pose axes.
    """

    if marker_corners is not None and marker_ids is not None:
        try:
            cv2.aruco.drawDetectedMarkers(
                frame_bgr,
                marker_corners,
                marker_ids,
            )
        except Exception:
            pass

    if result is None:
        return

    rvec = result["rvec"]
    tvec = result["tvec"]

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

    # Draw projected board center
    center_px = result["projected_center_px"]
    cx, cy = int(round(center_px[0])), int(round(center_px[1]))

    cv2.circle(frame_bgr, (cx, cy), 8, (0, 255, 255), -1)
    cv2.putText(
        frame_bgr,
        "board center",
        (cx + 10, cy - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )


def estimate_diamond_pose(gray, dictionary, marker_layout, camera_matrix, dist_coeffs):
    """
    Detect diamond markers and estimate board pose.
    """

    marker_corners, marker_ids, rejected = detect_aruco_markers(gray, dictionary)

    object_points, image_points, used_ids = build_pnp_points(
        marker_corners,
        marker_ids,
        marker_layout,
    )

    if object_points is None or image_points is None:
        return None, marker_corners, marker_ids, []

    if len(used_ids) < MIN_MARKERS_REQUIRED:
        return None, marker_corners, marker_ids, used_ids

    rvec, tvec = solve_planar_pnp(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
    )

    if rvec is None or tvec is None:
        return None, marker_corners, marker_ids, used_ids

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

    result = {
        "rvec": rvec,
        "tvec": tvec,
        "T_camera_board": T_camera_board,
        "object_points": object_points,
        "image_points": image_points,
        "used_ids": used_ids,
        "mean_reproj_error_px": mean_err_px,
        "max_reproj_error_px": max_err_px,
        "projected_points": projected_points,
        "origin_camera": origin_camera,
        "center_camera": center_camera,
        "projected_origin_px": projected_origin_px,
        "projected_center_px": projected_center_px,
    }

    return result, marker_corners, marker_ids, used_ids


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("[INFO] OpenCV version:", cv2.__version__)
    print("[INFO] OpenCV path:", cv2.__file__)

    dictionary = get_dictionary()
    marker_layout = make_marker_layout_object_points()

    print("")
    print("[INFO] Diamond marker layout, board frame, meters:")
    for marker_id in sorted(marker_layout.keys()):
        print(f"  ID {marker_id} ({MARKER_IDS[marker_id]}):")
        print(marker_layout[marker_id])

    zed = open_zed_camera()

    camera_matrix, dist_coeffs = get_zed_left_camera_matrix(zed)

    print("")
    print("[INFO] ZED left camera matrix:")
    print(camera_matrix)
    print("")
    print("[INFO] Distortion coefficients used:")
    print(dist_coeffs.reshape(-1))
    print("")
    print("[INFO] Diamond board parameters:")
    print(f"  board_size_m:      {BOARD_SIZE_M}")
    print(f"  margin_m:          {MARGIN_M}")
    print(f"  square_length_m:   {SQUARE_LENGTH_M}")
    print(f"  marker_length_m:   {MARKER_LENGTH_M}")
    print(f"  marker_ids:        {sorted(MARKER_IDS.keys())}")
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

            # Retrieve left image only.
            # No depth. No point cloud.
            zed.retrieve_image(left_image, sl.VIEW.LEFT)

            frame_bgra = left_image.get_data()
            frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            frame_h, frame_w = frame_bgr.shape[:2]

            result, marker_corners, marker_ids, used_ids = estimate_diamond_pose(
                gray,
                dictionary,
                marker_layout,
                camera_matrix,
                dist_coeffs,
            )

            draw_detection(
                frame_bgr,
                marker_corners,
                marker_ids,
                used_ids,
                result,
                camera_matrix,
                dist_coeffs,
            )

            now = time.time()

            if result is not None:
                tvec_history.append(result["tvec"].reshape(3))

                cv2.putText(
                    frame_bgr,
                    f"Diamond pose detected | markers: {used_ids}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

                if now - last_print_time >= PRINT_INTERVAL_SEC:
                    last_print_time = now

                    object_points = result["object_points"]
                    image_points = result["image_points"]

                    origin_camera = result["origin_camera"]
                    center_camera = result["center_camera"]

                    print("--------------------------------------------------")
                    print(f"Detected diamond marker IDs used: {sorted(result['used_ids'])}")
                    print(f"PnP method: SOLVEPNP_IPPE + RefineLM")
                    print(f"Image shape used for detection: width={frame_w}, height={frame_h}")

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
                    center_board = get_board_center_point()
                    print(f"  board center used: x={center_board[0]:+.4f}, y={center_board[1]:+.4f}, z={center_board[2]:+.4f} m")
                    print(f"  meters:      x={center_camera[0]:+.4f}, y={center_camera[1]:+.4f}, z={center_camera[2]:+.4f}")
                    print(f"  millimeters: x={center_camera[0]*1000:+.1f}, y={center_camera[1]*1000:+.1f}, z={center_camera[2]*1000:+.1f}")

                    print("")
                    print("Projected pixels from the estimated pose:")
                    print(f"  projected origin pixel: u={result['projected_origin_px'][0]:.1f}, v={result['projected_origin_px'][1]:.1f}")
                    print(f"  projected center pixel: u={result['projected_center_px'][0]:.1f}, v={result['projected_center_px'][1]:.1f}")
                    print(f"  image center pixel:     u={frame_w / 2.0:.1f}, v={frame_h / 2.0:.1f}")

                    print("")
                    print("T_camera_board:")
                    print(np.array2string(result["T_camera_board"], precision=4, suppress_small=True))

                    if len(tvec_history) >= 5:
                        hist = np.asarray(tvec_history)
                        med = np.median(hist, axis=0)
                        std = np.std(hist, axis=0)

                        print("")
                        print(f"Recent good tvec history: {len(tvec_history)} frames")
                        print("Median origin position:")
                        print(f"  x={med[0]*1000:+.1f} mm, y={med[1]*1000:+.1f} mm, z={med[2]*1000:+.1f} mm")
                        print("Std / jitter:")
                        print(f"  x={std[0]*1000:.2f} mm, y={std[1]*1000:.2f} mm, z={std[2]*1000:.2f} mm")

            else:
                cv2.putText(
                    frame_bgr,
                    f"No valid diamond pose | markers seen: {used_ids}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("ZED Left - Diamond ChArUco Board Test", frame_bgr)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    finally:
        cv2.destroyAllWindows()
        zed.close()
        print("[INFO] ZED closed.")


if __name__ == "__main__":
    main()