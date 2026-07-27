"""Detect a tray centroid and optionally move the Lite 6 TCP to it."""

import argparse
import json

from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl
import robot

from camera import (
    build_tray_predictor,
    get_image,
    get_point_cloud,
    get_zed_left_intrinsics_rectified,
    open_zed,
    process_tray,
    save_tray_data,
)


WINDOW_NAME = "Tray camera test"
DEFAULT_CALIBRATION_FILE = (
    Path(__file__).resolve().parent
    / "data"
    / "eyehand"
    / "eye_hand_calibration.json"
)
TSAI_CALIBRATION_FILE = (
    Path(__file__).resolve().parent
    / "data"
    / "eyehand"
    / "eye_hand_calibration_tsai.json"
)


def select_calibration_file(method, explicit_path=None):
    """Select the Li or Tsai result unless an explicit path was supplied."""
    if explicit_path is not None:
        return Path(explicit_path).resolve()

    method = str(method).strip().lower()
    if method == "li":
        return DEFAULT_CALIBRATION_FILE
    if method == "tsai":
        return TSAI_CALIBRATION_FILE
    raise ValueError("method must be either 'li' or 'tsai'.")


def load_T_base_camera(path):
    """Load and validate T_base_camera from a calibration JSON file."""
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    transform_value = data.get("T_base_camera", data.get("T_base_cam"))
    if transform_value is None:
        raise ValueError(
            f"{path} must contain 'T_base_camera' or 'T_base_cam'."
        )

    transform = np.asarray(transform_value, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("T_base_camera must be a finite 4x4 matrix.")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("T_base_camera has an invalid homogeneous last row.")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        raise ValueError("T_base_camera rotation is not orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
        raise ValueError("T_base_camera rotation determinant must be +1.")
    return transform


def camera_centroid_to_base(centroid_camera_m, T_base_camera):
    """Transform a camera-frame centroid in metres into the robot base frame."""
    centroid = np.asarray(centroid_camera_m, dtype=float).reshape(3)
    if not np.all(np.isfinite(centroid)) or centroid[2] <= 0:
        raise ValueError("Centroid must be finite and in front of the camera.")

    centroid_h = np.append(centroid, 1.0)
    centroid_base_m = (T_base_camera @ centroid_h)[:3]
    if not np.all(np.isfinite(centroid_base_m)):
        raise RuntimeError("The transformed centroid is not finite.")
    return centroid_base_m


def move_lite6_tcp_to_centroid(
    lite6,
    centroid_camera_m,
    T_base_camera,
    speed_mm_s,
):
    """Move the Lite 6 TCP position without commanding its orientation."""
    if speed_mm_s <= 0:
        raise ValueError("speed_mm_s must be positive.")

    target_base_mm = (
        camera_centroid_to_base(centroid_camera_m, T_base_camera) * 1000.0
    )
    if np.linalg.norm(target_base_mm) > 1000.0:
        raise ValueError(
            "Transformed target is over 1 m from the robot base; "
            "check the calibration and units."
        )

    if lite6.arm is None:
        lite6.connect()
    else:
        lite6.reset_state()

    print(
        "[WARNING] This command moves the TCP directly to the tray centroid."
    )
    print(
        "[INFO] Target position [x, y, z] mm = "
        + ", ".join(f"{value:.2f}" for value in target_base_mm)
    )
    print("[INFO] End-effector orientation is not commanded.")
    confirmation = input("Type MOVE to execute this motion: ").strip()
    if confirmation != "MOVE":
        print("[INFO] Robot motion cancelled.")
        return None

    code = lite6.arm.set_position(
        x=float(target_base_mm[0]),
        y=float(target_base_mm[1]),
        z=float(target_base_mm[2]),
        speed=float(speed_mm_s),
        wait=True,
        is_radian=False,
    )
    if code != 0:
        raise RuntimeError(
            f"Lite 6 centroid move failed with error code: {code}"
        )

    print("[INFO] Lite 6 TCP reached the commanded centroid.")
    return target_base_mm


def draw_result(image, result, camera_matrix):
    """Draw the detected mask, observed plane support, and 3D centroid."""
    display = image.copy()
    lines = ["S save | M move TCP to centroid | Q or ESC quit"]

    if result is None:
        lines.append("Tray not detected")
    else:
        detection = result["detection"]
        mask = detection["mask"]
        lines.append(f"Tray score: {detection['score']:.3f}")

        plane = result["plane"]
        if plane is None:
            # Green detection mask is shown only when plane fitting failed.
            overlay = display.copy()
            overlay[mask] = (0, 180, 0)
            display = cv2.addWeighted(display, 0.70, overlay, 0.30, 0)
            lines.append("Tray detected, but plane could not be calculated")
        else:
            # A valid RANSAC plane generalizes to the full detected tray mask.
            plane_overlay = display.copy()
            plane_overlay[mask] = (255, 0, 0)
            display = cv2.addWeighted(
                display,
                0.65,
                plane_overlay,
                0.35,
                0,
            )
            lines.append(
                "Plane inliers: "
                f"{plane['number_of_inliers']}/"
                f"{plane['number_of_ransac_points']}"
            )

        centroid = result["centroid"]
        if centroid is not None and centroid[2] > 0:
            pixel = camera_matrix @ centroid
            pixel = pixel[:2] / pixel[2]
            center = tuple(np.rint(pixel).astype(int))
            cv2.circle(display, center, 8, (0, 0, 255), -1)
            lines.append(
                "Centroid [m]: "
                + ", ".join(f"{value:.3f}" for value in centroid)
            )

    for index, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (15, 30 + 28 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return display


def main():
    parser = argparse.ArgumentParser(
        description="Detect a tray centroid and move the Lite 6 TCP to it"
    )
    parser.add_argument("--ip", required=True, help="Lite 6 controller IP")
    parser.add_argument(
        "--method",
        choices=("li", "tsai"),
        default="li",
        help="Select the saved eye-to-hand calibration result",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Override --method with a specific calibration JSON",
    )
    parser.add_argument("--speed", type=float, default=20.0)
    args = parser.parse_args()

    calibration_file = select_calibration_file(
        args.method,
        args.calibration,
    )
    T_base_camera = load_T_base_camera(calibration_file)
    calibration_label = (
        "custom" if args.calibration is not None else args.method.upper()
    )
    print(
        f"[INFO] Using {calibration_label} calibration: "
        f"{calibration_file}"
    )
    lite6 = robot.Lite6(args.ip)
    predictor = build_tray_predictor()
    zed = None
    latest_result = None

    try:
        zed, runtime, image_zed = open_zed()
        point_cloud_zed = sl.Mat()
        camera_matrix, _ = get_zed_left_intrinsics_rectified(zed)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        while True:
            # get_image grabs once; get_point_cloud retrieves that same frame.
            image = get_image(zed, runtime, image_zed)
            if image is None:
                continue

            xyz, valid_mask = get_point_cloud(zed, point_cloud_zed)
            if xyz is None:
                continue

            latest_result = process_tray(
                image,
                xyz,
                camera_matrix,
                predictor,
                valid_mask,
            )
            display = draw_result(image, latest_result, camera_matrix)
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKeyEx(1)

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                if (
                    latest_result is None
                    or latest_result["plane"] is None
                    or latest_result["centroid"] is None
                ):
                    print("[WARNING] No valid tray plane and centroid to save.")
                else:
                    save_tray_data(
                        latest_result["plane"],
                        latest_result["centroid"],
                    )
            if key in (ord("m"), ord("M")):
                if (
                    latest_result is None
                    or latest_result["plane"] is None
                    or latest_result["centroid"] is None
                ):
                    print("[WARNING] No valid tray centroid for robot motion.")
                else:
                    move_lite6_tcp_to_centroid(
                        lite6,
                        latest_result["centroid"],
                        T_base_camera,
                        args.speed,
                    )
    finally:
        cv2.destroyAllWindows()
        if zed is not None:
            zed.close()
        lite6.disconnect()


if __name__ == "__main__":
    main()
