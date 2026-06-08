import pyzed.sl as sl
import cv2 as cv
import numpy as np
import sys
import json
import time
from pathlib import Path

from ZED_bodytracking import (
    setup_body_tracking,
    get_single_body,
    get_arm_points,
    draw_arm_points_and_lines,
)

ROOT = Path("/home/agbara-admin/Documents/Cleaning_Robot")
IK_PATH = ROOT / "robot_control" / "IK"
DATA_PATH = ROOT / "data"

sys.path.append(str(IK_PATH))

from robot import M


ARM_TO_TRACK = "right"
NUM_REFERENCE_FRAMES = 30

ALPHA = 0.15

DEADBAND_X_MM = 50.0
DEADBAND_Y_MM = 50.0
DEADBAND_Z_MM = 50.0

TRAJECTORY_THRESHOLD_M = 0.002

TRAJECTORY_JSON = DATA_PATH / "arm_motion_trajectory.json"
DEBUG_JSON = DATA_PATH / "arm_motion_debug.json"


def get_robot_arm_length(M):
    """
    Calculate robot reference arm length.

    M is the home end-effector pose in meters.
    Output is also in meters.
    """

    p_home = M[:3, 3]

    robot_arm_length_m = np.linalg.norm(p_home)

    return float(robot_arm_length_m)


def calculate_motion_scale(human_arm_length_m, robot_arm_length_m):
    """
    Calculate human-to-robot motion scale.
    """

    if human_arm_length_m <= 0:
        raise ValueError("human_arm_length_m must be positive.")

    return float(robot_arm_length_m / human_arm_length_m)


def apply_exponential_smoothing(new_value, previous_value, alpha):
    """
    Smooth noisy displacement data.
    """

    if previous_value is None:
        return new_value

    return alpha * new_value + (1.0 - alpha) * previous_value


def apply_coordinate_deadband(
    vector,
    deadband_x,
    deadband_y,
    deadband_z
):
    """
    Apply deadband independently to each coordinate.

    If only one coordinate has real motion, small noise
    in the other coordinates is removed.
    """

    filtered_vector = vector.copy()

    if abs(filtered_vector[0]) < deadband_x:
        filtered_vector[0] = 0.0

    if abs(filtered_vector[1]) < deadband_y:
        filtered_vector[1] = 0.0

    if abs(filtered_vector[2]) < deadband_z:
        filtered_vector[2] = 0.0

    return filtered_vector


def camera_delta_to_robot_delta(camera_delta):
    """
    Convert relative camera-frame motion to robot-frame motion.

    Corrected mapping:
        camera x -> robot x
        camera y -> -robot z
        camera z -> robot y

    Meaning:
        robot x = camera x
        robot y = camera z
        robot z = -camera y

    Notes:
        camera y is positive downward, so moving the hand up
        gives negative camera y. The robot z-axis is positive upward,
        so we use -camera_y.
    """

    camera_x = camera_delta[0]
    camera_y = camera_delta[1]
    camera_z = camera_delta[2]

    robot_delta = np.array(
        [
            camera_x,
            camera_z,
            -camera_y,
        ],
        dtype=float
    )

    return robot_delta


def estimate_arm_length_from_data(arm_data):
    """
    Estimate human arm length from shoulder, elbow, and wrist.

    Input ZED data is in millimeters.
    Output is in meters.
    """

    shoulder = arm_data["shoulder_3d"]
    elbow = arm_data["elbow_3d"]
    wrist = arm_data["wrist_3d"]

    shoulder_to_elbow_mm = np.linalg.norm(elbow - shoulder)
    elbow_to_wrist_mm = np.linalg.norm(wrist - elbow)
    shoulder_to_wrist_mm = np.linalg.norm(wrist - shoulder)

    return {
        "shoulder_to_elbow_m": float(shoulder_to_elbow_mm / 1000.0),
        "elbow_to_wrist_m": float(elbow_to_wrist_mm / 1000.0),
        "shoulder_to_wrist_m": float(shoulder_to_wrist_mm / 1000.0),
    }


def compute_average_arm_data(samples, arm):
    """
    Average collected reference samples.
    """

    return {
        "arm": arm,

        "shoulder_2d": np.mean(
            [sample["shoulder_2d"] for sample in samples],
            axis=0
        ),

        "elbow_2d": np.mean(
            [sample["elbow_2d"] for sample in samples],
            axis=0
        ),

        "wrist_2d": np.mean(
            [sample["wrist_2d"] for sample in samples],
            axis=0
        ),

        "shoulder_3d": np.mean(
            [sample["shoulder_3d"] for sample in samples],
            axis=0
        ),

        "elbow_3d": np.mean(
            [sample["elbow_3d"] for sample in samples],
            axis=0
        ),

        "wrist_3d": np.mean(
            [sample["wrist_3d"] for sample in samples],
            axis=0
        ),
    }


def initialize_json_files(
    trajectory_path,
    debug_path,
    reference_data,
    motion_scale,
    robot_arm_length_m
):
    """
    Create fresh trajectory and debug JSON files.
    """

    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    trajectory_data = {
        "reference": {
            "arm": reference_data["arm"],

            "shoulder_reference_m": {
                "x": float(reference_data["shoulder_reference_3d"][0] / 1000.0),
                "y": float(reference_data["shoulder_reference_3d"][1] / 1000.0),
                "z": float(reference_data["shoulder_reference_3d"][2] / 1000.0),
            },

            "wrist_reference_m": {
                "x": float(reference_data["wrist_reference_3d"][0] / 1000.0),
                "y": float(reference_data["wrist_reference_3d"][1] / 1000.0),
                "z": float(reference_data["wrist_reference_3d"][2] / 1000.0),
            },

            "arm_length_m": reference_data["arm_length_data"],
        },

        "robot": {
            "robot_arm_length_m": float(robot_arm_length_m),

            "coordinate_mapping": {
                "robot_x": "camera_z",
                "robot_y": "camera_x",
                "robot_z": "-camera_y",
            },
        },

        "motion_scale": float(motion_scale),

        "trajectory": []
    }

    debug_data = {
        "settings": {
            "arm_to_track": ARM_TO_TRACK,
            "alpha": float(ALPHA),

            "deadband_m": {
                "x": float(DEADBAND_X_MM / 1000.0),
                "y": float(DEADBAND_Y_MM / 1000.0),
                "z": float(DEADBAND_Z_MM / 1000.0),
            },

            "trajectory_threshold_m": float(TRAJECTORY_THRESHOLD_M),
            "num_reference_frames": int(NUM_REFERENCE_FRAMES),

            "coordinate_mapping": {
                "robot_x": "camera_z",
                "robot_y": "camera_x",
                "robot_z": "-camera_y",
            },
        },

        "debug_samples": []
    }

    with open(trajectory_path, "w") as f:
        json.dump(trajectory_data, f, indent=4)

    with open(debug_path, "w") as f:
        json.dump(debug_data, f, indent=4)


def append_trajectory_point(
    trajectory_path,
    start_time,
    robot_motion_m
):
    """
    Append one robot-frame motion point in meters.
    """

    with open(trajectory_path, "r") as f:
        data = json.load(f)

    t = time.time() - start_time

    point = {
        "t": round(float(t), 3),
        "x": round(float(robot_motion_m[0]), 6),
        "y": round(float(robot_motion_m[1]), 6),
        "z": round(float(robot_motion_m[2]), 6),
    }

    data["trajectory"].append(point)

    with open(trajectory_path, "w") as f:
        json.dump(data, f, indent=4)


def append_debug_sample(
    debug_path,
    start_time,
    raw_delta_camera_mm,
    smoothed_delta_camera_mm,
    deadband_delta_camera_mm,
    robot_delta_unscaled_mm,
    robot_motion_m,
    motion_scale
):
    """
    Append one debug sample.

    Camera values are stored in meters.
    Final robot output is stored in meters.
    """

    with open(debug_path, "r") as f:
        data = json.load(f)

    t = time.time() - start_time

    sample = {
        "t": round(float(t), 3),

        "camera_frame": {
            "raw_delta_m": {
                "x": round(float(raw_delta_camera_mm[0] / 1000.0), 6),
                "y": round(float(raw_delta_camera_mm[1] / 1000.0), 6),
                "z": round(float(raw_delta_camera_mm[2] / 1000.0), 6),
            },

            "smoothed_delta_m": {
                "x": round(float(smoothed_delta_camera_mm[0] / 1000.0), 6),
                "y": round(float(smoothed_delta_camera_mm[1] / 1000.0), 6),
                "z": round(float(smoothed_delta_camera_mm[2] / 1000.0), 6),
            },

            "deadband_delta_m": {
                "x": round(float(deadband_delta_camera_mm[0] / 1000.0), 6),
                "y": round(float(deadband_delta_camera_mm[1] / 1000.0), 6),
                "z": round(float(deadband_delta_camera_mm[2] / 1000.0), 6),
            },
        },

        "robot_frame": {
            "unscaled_delta_m": {
                "x": round(float(robot_delta_unscaled_mm[0] / 1000.0), 6),
                "y": round(float(robot_delta_unscaled_mm[1] / 1000.0), 6),
                "z": round(float(robot_delta_unscaled_mm[2] / 1000.0), 6),
            },

            "scaled_motion_m": {
                "x": round(float(robot_motion_m[0]), 6),
                "y": round(float(robot_motion_m[1]), 6),
                "z": round(float(robot_motion_m[2]), 6),
            },
        },

        "motion_scale": round(float(motion_scale), 6),
    }

    data["debug_samples"].append(sample)

    with open(debug_path, "w") as f:
        json.dump(data, f, indent=4)


def should_save_trajectory_point(
    robot_motion_m,
    previous_saved_motion_m,
    threshold_m
):
    """
    Save only meaningful robot motion changes.
    """

    if previous_saved_motion_m is None:
        return True

    movement = np.linalg.norm(
        robot_motion_m - previous_saved_motion_m
    )

    return movement >= threshold_m


def main():
    zed = sl.Camera()

    init_params = sl.InitParameters()

    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30

    # ZED outputs body keypoints in millimeters.
    init_params.coordinate_units = sl.UNIT.MILLIMETER

    # IMAGE coordinate system:
    # camera x = image right
    # camera y = image down
    # camera z = forward depth
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

    err = zed.open(init_params)

    if err != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera:", err)
        exit(1)

    print("Camera opened!")

    try:
        body_runtime = setup_body_tracking(zed)

        image = sl.Mat()
        bodies = sl.Bodies()
        runtime = sl.RuntimeParameters()

        window_name = "Arm Motion Tracking"

        reference_data = None
        reference_samples = []
        collecting_reference = False

        smoothed_delta_camera_mm = None
        motion_scale = None
        tracking_start_time = None
        previous_saved_motion_m = None

        robot_arm_length_m = get_robot_arm_length(M)

        print(f"Robot arm length: {robot_arm_length_m:.4f} m")
        print("Press ENTER to collect reference position.")
        print("Press q to quit.")

        while True:

            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()

            if frame.shape[2] == 4:
                frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

            zed.retrieve_bodies(bodies, body_runtime)

            body = get_single_body(bodies, mode="closest")

            arm_data = None

            if body is not None:
                arm_data = get_arm_points(
                    body,
                    arm=ARM_TO_TRACK
                )

            if arm_data is not None:
                frame = draw_arm_points_and_lines(
                    frame,
                    arm_data
                )

            if collecting_reference and arm_data is not None:

                reference_samples.append(arm_data)

                cv.putText(
                    frame,
                    f"Collecting reference: {len(reference_samples)}/{NUM_REFERENCE_FRAMES}",
                    (20, 40),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                cv.putText(
                    frame,
                    "Keep arm still and straight",
                    (20, 75),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                if len(reference_samples) >= NUM_REFERENCE_FRAMES:

                    avg_arm_data = compute_average_arm_data(
                        samples=reference_samples,
                        arm=ARM_TO_TRACK
                    )

                    arm_length_data = estimate_arm_length_from_data(
                        avg_arm_data
                    )

                    reference_data = {
                        "arm": ARM_TO_TRACK,

                        "shoulder_reference_3d": avg_arm_data["shoulder_3d"],
                        "elbow_reference_3d": avg_arm_data["elbow_3d"],
                        "wrist_reference_3d": avg_arm_data["wrist_3d"],

                        "shoulder_reference_2d": avg_arm_data["shoulder_2d"],
                        "elbow_reference_2d": avg_arm_data["elbow_2d"],
                        "wrist_reference_2d": avg_arm_data["wrist_2d"],

                        "arm_length_data": arm_length_data,

                        "num_frames": NUM_REFERENCE_FRAMES,
                    }

                    human_arm_length_m = arm_length_data[
                        "shoulder_to_wrist_m"
                    ]

                    motion_scale = calculate_motion_scale(
                        human_arm_length_m=human_arm_length_m,
                        robot_arm_length_m=robot_arm_length_m
                    )

                    smoothed_delta_camera_mm = None
                    previous_saved_motion_m = None
                    tracking_start_time = time.time()

                    initialize_json_files(
                        trajectory_path=TRAJECTORY_JSON,
                        debug_path=DEBUG_JSON,
                        reference_data=reference_data,
                        motion_scale=motion_scale,
                        robot_arm_length_m=robot_arm_length_m
                    )

                    append_trajectory_point(
                        trajectory_path=TRAJECTORY_JSON,
                        start_time=tracking_start_time,
                        robot_motion_m=np.zeros(3)
                    )

                    previous_saved_motion_m = np.zeros(3)

                    collecting_reference = False

                    print("Reference collected.")
                    print(f"Human arm length: {human_arm_length_m:.4f} m")
                    print(f"Robot arm length: {robot_arm_length_m:.4f} m")
                    print(f"Motion scale: {motion_scale:.4f}")
                    print(f"Trajectory JSON: {TRAJECTORY_JSON}")
                    print(f"Debug JSON: {DEBUG_JSON}")

            elif reference_data is not None and arm_data is not None:

                wrist_current = arm_data["wrist_3d"]
                wrist_reference = reference_data["wrist_reference_3d"]

                # Camera-frame displacement in millimeters.
                raw_delta_camera_mm = wrist_current - wrist_reference

                # Smooth camera-frame displacement in millimeters.
                smoothed_delta_camera_mm = apply_exponential_smoothing(
                    new_value=raw_delta_camera_mm,
                    previous_value=smoothed_delta_camera_mm,
                    alpha=ALPHA
                )

                # Remove small noise independently per coordinate.
                deadband_delta_camera_mm = apply_coordinate_deadband(
                    vector=smoothed_delta_camera_mm,
                    deadband_x=DEADBAND_X_MM,
                    deadband_y=DEADBAND_Y_MM,
                    deadband_z=DEADBAND_Z_MM
                )

                # Convert camera-frame relative motion to robot-frame motion.
                robot_delta_unscaled_mm = camera_delta_to_robot_delta(
                    deadband_delta_camera_mm
                )

                # Scale robot-frame motion, then convert final output to meters.
                robot_motion_mm = motion_scale * robot_delta_unscaled_mm
                robot_motion_m = robot_motion_mm / 1000.0

                append_debug_sample(
                    debug_path=DEBUG_JSON,
                    start_time=tracking_start_time,
                    raw_delta_camera_mm=raw_delta_camera_mm,
                    smoothed_delta_camera_mm=smoothed_delta_camera_mm,
                    deadband_delta_camera_mm=deadband_delta_camera_mm,
                    robot_delta_unscaled_mm=robot_delta_unscaled_mm,
                    robot_motion_m=robot_motion_m,
                    motion_scale=motion_scale
                )

                if should_save_trajectory_point(
                    robot_motion_m=robot_motion_m,
                    previous_saved_motion_m=previous_saved_motion_m,
                    threshold_m=TRAJECTORY_THRESHOLD_M
                ):
                    append_trajectory_point(
                        trajectory_path=TRAJECTORY_JSON,
                        start_time=tracking_start_time,
                        robot_motion_m=robot_motion_m
                    )

                    previous_saved_motion_m = robot_motion_m.copy()

                cv.putText(
                    frame,
                    "Tracking active",
                    (20, 40),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv.putText(
                    frame,
                    f"Robot m: x={robot_motion_m[0]:.4f}, y={robot_motion_m[1]:.4f}, z={robot_motion_m[2]:.4f}",
                    (20, 75),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv.putText(
                    frame,
                    "Mapping: robot x=cam z, robot y=cam x, robot z=-cam y",
                    (20, 110),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

            elif reference_data is None and not collecting_reference:

                cv.putText(
                    frame,
                    "Hold arm straight",
                    (20, 40),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv.putText(
                    frame,
                    "Press ENTER to collect reference",
                    (20, 75),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            cv.putText(
                frame,
                "Press q to quit",
                (20, frame.shape[0] - 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv.imshow(window_name, frame)

            key = cv.waitKey(1) & 0xFF

            if key in [10, 13] and not collecting_reference:

                reference_samples.clear()
                reference_data = None
                motion_scale = None
                smoothed_delta_camera_mm = None
                previous_saved_motion_m = None
                tracking_start_time = None

                collecting_reference = True

                print("Collecting reference...")

            if key == ord("q"):
                break

            if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
                break

    finally:
        cv.destroyAllWindows()
        zed.close()
        print("Program closed.")


if __name__ == "__main__":
    main()