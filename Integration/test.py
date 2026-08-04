"""Test tray localization or apply a force to a detected tray with a Lite 6."""

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
DEFAULT_ORIENTATION_X_MM = 200.0
DEFAULT_ORIENTATION_Y_MM = 0.9
DEFAULT_ORIENTATION_CLEARANCE_MM = 100.0
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
DEFAULT_FORCE_CALIBRATION_FILE = (
    Path(__file__).resolve().parent
    / "data"
    / "force_data"
    / "force_sensor_calibration.json"
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

    if "data" in data:
        names = data.get("description", {}).get("axis_0_order", [])
        try:
            transform_index = names.index("T_base_camera")
        except ValueError:
            transform_value = None
        else:
            transform_value = data["data"][transform_index]
    else:
        transform_value = data.get("T_base_camera", data.get("T_base_cam"))

    if transform_value is None:
        raise ValueError(
            f"{path} must describe a T_base_camera transform."
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


def tray_tool_rotation(normal_camera, T_base_camera, current_R_base_tool,
                       tool_z_sign=1.0):
    """Build a tool rotation with tool Z aligned to the tray push direction.

    ``camera.calculate_tray_plane`` makes the normal point away from the tray,
    toward the camera.  The push direction is consequently its negative.
    ``tool_z_sign=1`` aligns tool +Z with the push direction; use -1 when the
    contact face is on the tool's -Z side.
    """
    normal_camera = np.asarray(normal_camera, dtype=float).reshape(3)
    length = np.linalg.norm(normal_camera)
    if not np.isfinite(length) or length < 1e-9:
        raise ValueError("Tray normal must be a finite nonzero vector.")
    normal_camera /= length

    normal_base = T_base_camera[:3, :3] @ normal_camera
    normal_base /= np.linalg.norm(normal_base)
    push_direction_base = -normal_base
    z_tool = float(tool_z_sign) * push_direction_base

    # Retain the current tool twist by projecting its X axis onto the plane
    # perpendicular to the new tool Z axis.
    x_reference = np.asarray(current_R_base_tool, dtype=float)[:3, 0]
    x_tool = x_reference - np.dot(x_reference, z_tool) * z_tool
    if np.linalg.norm(x_tool) < 1e-6:
        candidates = np.eye(3)
        x_reference = min(
            candidates,
            key=lambda axis: abs(np.dot(axis, z_tool)),
        )
        x_tool = x_reference - np.dot(x_reference, z_tool) * z_tool
    x_tool /= np.linalg.norm(x_tool)
    y_tool = np.cross(z_tool, x_tool)
    y_tool /= np.linalg.norm(y_tool)
    x_tool = np.cross(y_tool, z_tool)
    x_tool /= np.linalg.norm(x_tool)
    return np.column_stack((x_tool, y_tool, z_tool)), push_direction_base


def rotation_to_lite6_rpy_deg(rotation):
    """Convert a rotation matrix to Lite 6 Rz(yaw)Ry(pitch)Rx(roll) RPY."""
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt

    pitch = np.arctan2(
        -rotation[2, 0],
        np.hypot(rotation[0, 0], rotation[1, 0]),
    )
    if abs(np.cos(pitch)) > 1e-7:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-rotation[0, 1], rotation[1, 1])
    return np.degrees([roll, pitch, yaw])


def command_cartesian_pose(lite6, position_base_mm, rpy_deg, speed_mm_s):
    """Send one blocking Cartesian pose command and validate its result."""
    position_base_mm = np.asarray(position_base_mm, dtype=float).reshape(3)
    rpy_deg = np.asarray(rpy_deg, dtype=float).reshape(3)
    code = lite6.arm.set_position(
        x=float(position_base_mm[0]),
        y=float(position_base_mm[1]),
        z=float(position_base_mm[2]),
        roll=float(rpy_deg[0]),
        pitch=float(rpy_deg[1]),
        yaw=float(rpy_deg[2]),
        speed=float(speed_mm_s),
        wait=True,
        is_radian=False,
    )
    if code != 0:
        raise RuntimeError(f"Lite 6 Cartesian move failed with code: {code}")


def calculate_tray_target(lite6, frozen_result, T_base_camera, tool_z_sign):
    """Calculate the frozen tray centroid, push direction, and tool RPY."""
    centroid_base_mm = camera_centroid_to_base(
        frozen_result["centroid"],
        T_base_camera,
    ) * 1000.0
    if np.linalg.norm(centroid_base_mm) > 1000.0:
        raise ValueError("Tray target is over 1 m from the robot base.")

    current_rotation = lite6.get_T_base_to_ee()[:3, :3]
    target_rotation, push_direction = tray_tool_rotation(
        frozen_result["plane"]["normal"],
        T_base_camera,
        current_rotation,
        tool_z_sign,
    )
    target_rpy_deg = rotation_to_lite6_rpy_deg(target_rotation)
    return centroid_base_mm, push_direction, target_rpy_deg


def print_frozen_target(centroid_base_mm, push_direction, target_rpy_deg):
    """Print the frozen vision result used by the upcoming robot command."""
    print("[INFO] Camera detection is now frozen for the entire motion.")
    print(
        "[INFO] Frozen centroid [mm]: "
        + ", ".join(f"{value:.2f}" for value in centroid_base_mm)
    )
    print(
        "[INFO] Push direction in base: "
        + ", ".join(f"{value:.5f}" for value in push_direction)
    )
    print(
        "[INFO] Target RPY [deg]: "
        + ", ".join(f"{value:.2f}" for value in target_rpy_deg)
    )


def orientation_staging_position(centroid_base_mm, x_mm, y_mm,
                                 clearance_mm):
    """Return the known XY staging point positioned above the tray in base Z."""
    centroid_base_mm = np.asarray(centroid_base_mm, dtype=float).reshape(3)
    position = np.array(
        [x_mm, y_mm, centroid_base_mm[2] + clearance_mm],
        dtype=float,
    )
    if not np.all(np.isfinite(position)):
        raise ValueError("Orientation staging position must be finite.")
    return position


def command_tray_orientation(lite6, staging_position_mm, target_rpy_deg,
                             speed_mm_s):
    """Command the calculated orientation at the safe staging position."""
    command_cartesian_pose(
        lite6,
        staging_position_mm,
        target_rpy_deg,
        speed_mm_s,
    )
    print(
        "[INFO] Orientation staging pose reached [mm]: "
        + ", ".join(f"{value:.2f}" for value in staging_position_mm)
    )


def run_orientation_test(lite6, frozen_result, T_base_camera, args,
                         move_to_centroid=False):
    """Test normal-to-orientation, optionally followed by centroid motion."""
    if lite6.arm is None:
        lite6.connect()
    else:
        lite6.reset_state()
    centroid_mm, push_direction, target_rpy = calculate_tray_target(
        lite6,
        frozen_result,
        T_base_camera,
        args.tool_z_sign,
    )
    print_frozen_target(centroid_mm, push_direction, target_rpy)
    staging_mm = orientation_staging_position(
        centroid_mm,
        args.orientation_x,
        args.orientation_y,
        args.orientation_clearance,
    )
    print(
        "[INFO] Orientation staging XYZ [mm]: "
        + ", ".join(f"{value:.2f}" for value in staging_mm)
    )
    action = "ORIENT_AND_MOVE" if move_to_centroid else "ORIENT"
    if input(f"Type {action} to execute: ").strip() != action:
        print("[INFO] Robot motion cancelled.")
        return None

    command_tray_orientation(lite6, staging_mm, target_rpy, args.speed)
    if move_to_centroid:
        command_cartesian_pose(lite6, centroid_mm, target_rpy, args.speed)
        print(
            "[INFO] Centroid reached while maintaining tool orientation; "
            "no force-control motion was commanded."
        )
    return target_rpy


def gravity_compensated_force(sensor, lite6, force_sign):
    """Read one fresh sample and remove modeled tool-weight loading."""
    sample = sensor.read(timeout_seconds=2.0)
    transform = lite6.get_T_base_to_ee()
    force = sensor.calibration.contact_force_from_raw(
        sample.raw_adc,
        transform,
    )
    return float(force_sign) * force


def apply_force_to_tray(lite6, frozen_result, T_base_camera, args):
    """Approach a frozen tray pose and increment along its normal to contact."""
    # Keep force-sensor/Haplink dependencies optional in transformation mode.
    from force_sensor import ForceSensor

    if lite6.arm is None:
        lite6.connect()
    else:
        lite6.reset_state()
    centroid_base_mm, push_direction, target_rpy_deg = calculate_tray_target(
        lite6,
        frozen_result,
        T_base_camera,
        args.tool_z_sign,
    )
    approach_mm = centroid_base_mm - args.approach_distance * push_direction
    staging_mm = orientation_staging_position(
        centroid_base_mm,
        args.orientation_x,
        args.orientation_y,
        args.orientation_clearance,
    )

    print_frozen_target(centroid_base_mm, push_direction, target_rpy_deg)
    print(
        "[INFO] Orientation staging XYZ [mm]: "
        + ", ".join(f"{value:.2f}" for value in staging_mm)
    )
    print(
        f"[WARNING] Motion may advance {args.max_contact_travel:.1f} mm "
        f"toward the tray. Target={args.target_force:.1f} N; "
        f"hard limit={args.max_force:.1f} N."
    )
    if input("Type APPLY to execute: ").strip() != "APPLY":
        print("[INFO] Force application cancelled.")
        return None

    sensor = ForceSensor(port=args.force_port, baudrate=args.force_baud)
    reached_target = False
    must_retract = False
    try:
        sensor.connect()
        sensor.load_calibration(args.force_calibration)
        if not sensor.calibration.has_gravity_model:
            raise RuntimeError(
                "Force calibration has no gravity model; calibrate it first."
            )

        # Orientation is deliberately a separate, observable pipeline step.
        command_tray_orientation(
            lite6,
            staging_mm,
            target_rpy_deg,
            args.speed,
        )
        command_cartesian_pose(
            lite6,
            approach_mm,
            target_rpy_deg,
            args.speed,
        )
        print("[INFO] Approach pose reached; beginning contact search.")

        travel_mm = 0.0
        consecutive_target_samples = 0
        while travel_mm <= args.max_contact_travel + 1e-9:
            force_n = gravity_compensated_force(
                sensor,
                lite6,
                args.force_sign,
            )
            print(
                f"[FORCE] {force_n:7.3f} N | "
                f"travel {travel_mm:6.2f} mm"
            )
            if abs(force_n) >= args.max_force:
                must_retract = True
                raise RuntimeError(
                    f"Hard force limit reached: {force_n:.3f} N"
                )
            if force_n >= args.target_force:
                consecutive_target_samples += 1
                if consecutive_target_samples >= args.target_samples:
                    reached_target = True
                    print(
                        f"[INFO] Target force reached: {force_n:.3f} N. "
                        "Holding the final commanded pose."
                    )
                    return force_n
                # Confirm the threshold at the same pose; do not push farther
                # merely to collect the remaining confirmation samples.
                continue
            else:
                consecutive_target_samples = 0

            next_travel_mm = travel_mm + args.contact_step
            if next_travel_mm > args.max_contact_travel + 1e-9:
                break
            target_mm = approach_mm + next_travel_mm * push_direction
            command_cartesian_pose(
                lite6,
                target_mm,
                target_rpy_deg,
                args.contact_speed,
            )
            travel_mm = next_travel_mm

        must_retract = True
        raise RuntimeError(
            "Maximum contact-search travel reached before target force."
        )
    except Exception:
        must_retract = True
        raise
    finally:
        if must_retract and not reached_target and lite6.arm is not None:
            print("[WARNING] Retracting to the approach pose.")
            try:
                command_cartesian_pose(
                    lite6,
                    approach_mm,
                    target_rpy_deg,
                    args.speed,
                )
            except Exception as retract_error:
                print(f"[ERROR] Automatic retract failed: {retract_error}")
                lite6.arm.set_state(4)
        sensor.disconnect()


def draw_result(image, result, camera_matrix, mode):
    """Draw the detected mask, observed plane support, and 3D centroid."""
    display = image.copy()
    if mode == "transformation":
        lines = ["S save | M test transformation | Q or ESC quit"]
    elif mode == "orientation":
        lines = ["S save | O test tool orientation | Q or ESC quit"]
    elif mode == "orientation_movement":
        lines = ["S save | M orient and move to centroid | Q or ESC quit"]
    else:
        lines = ["S save | F freeze detection and apply force | Q or ESC quit"]

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
        description="Test tray localization or apply force with the Lite 6"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test-transformation", action="store_true")
    mode.add_argument("--test-orientation", action="store_true")
    mode.add_argument("--test-orientation-movement", action="store_true")
    mode.add_argument("--apply-force", action="store_true")
    parser.add_argument("--ip", required=True, help="Lite 6 controller IP")
    parser.add_argument(
        "--method",
        choices=("li", "tsai"),
        default="tsai",
        help="Select the saved eye-to-hand calibration result",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Override --method with a specific calibration JSON",
    )
    parser.add_argument("--speed", type=float, default=20.0)
    parser.add_argument("--orientation-x", type=float,
                        default=DEFAULT_ORIENTATION_X_MM,
                        help="Base-frame X of orientation staging pose in mm")
    parser.add_argument("--orientation-y", type=float,
                        default=DEFAULT_ORIENTATION_Y_MM,
                        help="Base-frame Y of orientation staging pose in mm")
    parser.add_argument("--orientation-clearance", type=float,
                        default=DEFAULT_ORIENTATION_CLEARANCE_MM,
                        help="Base-Z distance above transformed tray Z in mm")
    parser.add_argument("--contact-speed", type=float, default=2.0)
    parser.add_argument("--approach-distance", type=float, default=50.0,
                        help="Approach clearance in mm")
    parser.add_argument("--contact-step", type=float, default=0.25,
                        help="Incremental contact-search step in mm")
    parser.add_argument("--max-contact-travel", type=float, default=80.0,
                        help="Maximum travel from the approach pose in mm")
    parser.add_argument("--target-force", type=float, default=10.0)
    parser.add_argument("--max-force", type=float, default=45.0,
                        help="Software abort threshold in N (must be <= 50)")
    parser.add_argument("--target-samples", type=int, default=3,
                        help="Consecutive samples required above target")
    parser.add_argument("--tool-z-sign", type=float, choices=(-1.0, 1.0),
                        default=1.0,
                        help="Tool Z sign that points into the tray")
    parser.add_argument("--force-sign", type=float, choices=(-1.0, 1.0),
                        default=1.0,
                        help="Sign making compression positive")
    parser.add_argument("--force-port", default="/dev/ttyUSB0")
    parser.add_argument("--force-baud", type=int, default=115200)
    parser.add_argument("--force-calibration", type=Path,
                        default=DEFAULT_FORCE_CALIBRATION_FILE)
    args = parser.parse_args()

    positive_values = {
        "speed": args.speed,
        "contact_speed": args.contact_speed,
        "approach_distance": args.approach_distance,
        "orientation_clearance": args.orientation_clearance,
        "contact_step": args.contact_step,
        "max_contact_travel": args.max_contact_travel,
        "target_force": args.target_force,
        "max_force": args.max_force,
        "target_samples": args.target_samples,
    }
    for name, value in positive_values.items():
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not np.isfinite(args.orientation_x) or not np.isfinite(
        args.orientation_y
    ):
        parser.error("--orientation-x and --orientation-y must be finite")
    if args.target_force >= args.max_force:
        parser.error("--target-force must be below --max-force")
    if args.max_force > 50.0:
        parser.error("--max-force cannot exceed the 50 N robot safety limit")
    args.force_calibration = args.force_calibration.resolve()
    if args.test_transformation:
        run_mode = "transformation"
    elif args.test_orientation:
        run_mode = "orientation"
    elif args.test_orientation_movement:
        run_mode = "orientation_movement"
    else:
        run_mode = "force"

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
            display = draw_result(image, latest_result, camera_matrix, run_mode)
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
            if run_mode == "transformation" and key in (ord("m"), ord("M")):
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
            if run_mode == "orientation" and key in (ord("o"), ord("O")):
                if (
                    latest_result is None
                    or latest_result["plane"] is None
                    or latest_result["centroid"] is None
                ):
                    print("[WARNING] No valid tray plane and centroid.")
                else:
                    run_orientation_test(
                        lite6,
                        latest_result,
                        T_base_camera,
                        args,
                        move_to_centroid=False,
                    )
            if (
                run_mode == "orientation_movement"
                and key in (ord("m"), ord("M"))
            ):
                if (
                    latest_result is None
                    or latest_result["plane"] is None
                    or latest_result["centroid"] is None
                ):
                    print("[WARNING] No valid tray plane and centroid.")
                else:
                    run_orientation_test(
                        lite6,
                        latest_result,
                        T_base_camera,
                        args,
                        move_to_centroid=True,
                    )
            if run_mode == "force" and key in (ord("f"), ord("F")):
                if (
                    latest_result is None
                    or latest_result["plane"] is None
                    or latest_result["centroid"] is None
                ):
                    print("[WARNING] No valid tray plane and centroid.")
                else:
                    # This blocking routine deliberately prevents additional
                    # camera grabs/detections until all motion has stopped.
                    apply_force_to_tray(
                        lite6,
                        latest_result,
                        T_base_camera,
                        args,
                    )
    finally:
        cv2.destroyAllWindows()
        if zed is not None:
            zed.close()
        lite6.disconnect()


if __name__ == "__main__":
    main()
