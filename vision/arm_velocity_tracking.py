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
DATA_PATH = ROOT / "data" / "arm_tracking"

sys.path.append(str(IK_PATH))

# ============================================================
# Settings
# ============================================================

ARM_TO_TRACK = "right"
NUM_REFERENCE_FRAMES = 30

ALPHA_LINEAR = 0.1
ALPHA_ANGULAR = 0.1

MOTION_SCALE = 0.5

# Since ZED is set to sl.UNIT.MILLIMETER, this is 50 mm
POSITION_DEADBAND = 50

SAVE_THRESHOLD_LINEAR = 0.05
SAVE_THRESHOLD_ANGULAR = 0.05

DISPLAY_SCALE = 0.75

VELOCITY_JSON = DATA_PATH / "arm_velocity_tracking.json"

ENABLE_VIDEO_RECORDING = True
VIDEO_OUTPUT = DATA_PATH / "arm_velocity_showcase.mp4"
VIDEO_FPS = 30


def applyPositionDB(position):
    """
    Apply deadband to position vector.
    """

    position2 = np.asarray(position, dtype=float).copy()
    position2[np.abs(position2) <= POSITION_DEADBAND] = 0.0

    return position2

def Cal_velocity(p2, p1, dt):
    """
    Calculate linear velocity from two positions.
    """

    pos2 = np.asarray(p2, dtype=float)
    pos1 = np.asarray(p1, dtype=float)

    if dt <= 0:
        return np.zeros_like(pos2)

    return (pos2 - pos1) / dt


def exponential_filter(current, previous, alpha):
    """
    Apply exponential smoothing.
    """

    if previous is None:
        return current

    return alpha * current + (1.0 - alpha) * previous


def get_p_arm(p_camera, T):
    """
    Transform a 3D point from the camera frame to the arm frame.
    """

    p_camera_h = np.append(
        np.asarray(p_camera, dtype=float),
        1.0
    )

    p_arm_h = T @ p_camera_h

    return p_arm_h[:3]


def get_T_wrist_camera(reference_data, origin_point="shoulder"):
    """
    Build transform from camera frame to wrist frame.

    Camera frame:
        X = Right
        Y = Down
        Z = Forward

    Wrist frame:
        X = Forward
        Y = Right
        Z = Up

    The wrist frame orientation is fixed.

    The transform may be constructed using:

        wrist
        elbow
        shoulder

    as an intermediate origin.

    Parameters
    ----------
    reference_data : dict

    origin_point : str
        "wrist"
        "elbow"
        "shoulder"

    Returns
    -------
    T_wrist_camera : np.ndarray (4,4)

        p_wrist = T_wrist_camera @ p_camera_h
    """

    # -------------------------------------------------
    # Reference points in camera frame
    # -------------------------------------------------

    shoulder = np.asarray(
        reference_data["shoulder_reference_3d"],
        dtype=float
    )

    elbow = np.asarray(
        reference_data["elbow_reference_3d"],
        dtype=float
    )

    wrist = np.asarray(
        reference_data["wrist_reference_3d"],
        dtype=float
    )

    # -------------------------------------------------
    # Wrist frame orientation
    # -------------------------------------------------

    R_wrist_camera = np.array([
        [0.0,  0.0,  1.0],
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0]
    ])

    origin_point = origin_point.lower()

    # -------------------------------------------------
    # Direct camera -> wrist transform
    # -------------------------------------------------

    if origin_point == "wrist":

        T_wrist_camera = np.eye(4)

        T_wrist_camera[:3, :3] = R_wrist_camera

        T_wrist_camera[:3, 3] = (
            -R_wrist_camera @ wrist
        )

        return T_wrist_camera

    # -------------------------------------------------
    # Choose intermediate origin
    # -------------------------------------------------

    if origin_point == "shoulder":
        origin = shoulder

    elif origin_point == "elbow":
        origin = elbow

    else:
        raise ValueError(
            "origin_point must be "
            "'wrist', 'elbow', or 'shoulder'"
        )

    # -------------------------------------------------
    # T_origin_camera
    #
    # Camera -> origin frame
    #
    # Origin frame has SAME orientation
    # as wrist frame.
    # -------------------------------------------------

    T_origin_camera = np.eye(4)

    T_origin_camera[:3, :3] = R_wrist_camera

    T_origin_camera[:3, 3] = (
        -R_wrist_camera @ origin
    )

    # -------------------------------------------------
    # T_wrist_origin
    #
    # Origin -> wrist
    #
    # Same orientation.
    # Translation only.
    # -------------------------------------------------

    wrist_in_origin = (
        R_wrist_camera @ (wrist - origin)
    )

    T_wrist_origin = np.eye(4)

    T_wrist_origin[:3, 3] = (
        -wrist_in_origin
    )

    # -------------------------------------------------
    # Compose
    #
    # T_wrist_camera =
    # T_wrist_origin @ T_origin_camera
    # -------------------------------------------------

    T_wrist_camera = (
        T_wrist_origin @ T_origin_camera
    )

    return T_wrist_camera


def make_json_safe(data):
    """
    Convert numpy arrays/types into JSON-safe Python types.
    """

    if isinstance(data, np.ndarray):
        return data.tolist()

    if isinstance(data, np.integer):
        return int(data)

    if isinstance(data, np.floating):
        return float(data)

    if isinstance(data, dict):
        return {
            key: make_json_safe(value)
            for key, value in data.items()
        }

    if isinstance(data, list):
        return [
            make_json_safe(value)
            for value in data
        ]

    return data


def estimate_arm_length_from_points(shoulder, elbow, wrist):
    """
    Estimate arm segment lengths.
    """

    return {
        "shoulder_to_elbow": float(np.linalg.norm(elbow - shoulder)),
        "elbow_to_wrist": float(np.linalg.norm(wrist - elbow)),
        "shoulder_to_wrist": float(np.linalg.norm(wrist - shoulder)),
    }


def save_velocity_json(reference_data, records):
    """
    Save velocity tracking data to JSON.
    """

    output = {
        "reference_data": reference_data,
        "records": records
    }

    with open(VELOCITY_JSON, "w") as f:
        json.dump(
            make_json_safe(output),
            f,
            indent=4
        )


# ===========================================================
# Main Function
# ===========================================================

def main():
    DATA_PATH.mkdir(parents=True, exist_ok=True)

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

    video_writer = None

    try:
        body_runtime = setup_body_tracking(zed)

        image = sl.Mat()
        bodies = sl.Bodies()
        runtime = sl.RuntimeParameters()

        window_name = "Arm Motion Tracking"

        reference_data = None
        T_arm_camera = None

        collecting_reference = False

        shoulder_3d_samples = []
        elbow_3d_samples = []
        wrist_3d_samples = []

        shoulder_2d_samples = []
        elbow_2d_samples = []
        wrist_2d_samples = []

        prev_p_arm_db = None
        prev_time = None

        filtered_velocity = None

        records = []

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

            body = get_single_body(
                bodies,
                mode="closest"
            )

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

            # --------------------------------------------------
            # Collect reference position
            # --------------------------------------------------

            if collecting_reference and arm_data is not None:

                shoulder_3d_samples.append(arm_data["shoulder_3d"])
                elbow_3d_samples.append(arm_data["elbow_3d"])
                wrist_3d_samples.append(arm_data["wrist_3d"])

                shoulder_2d_samples.append(arm_data["shoulder_2d"])
                elbow_2d_samples.append(arm_data["elbow_2d"])
                wrist_2d_samples.append(arm_data["wrist_2d"])

                count = len(wrist_3d_samples)

                cv.putText(
                    frame,
                    f"Collecting reference: {count}/{NUM_REFERENCE_FRAMES}",
                    (20, 40),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                if count >= NUM_REFERENCE_FRAMES:

                    shoulder_ref = np.mean(shoulder_3d_samples, axis=0)
                    elbow_ref = np.mean(elbow_3d_samples, axis=0)
                    wrist_ref = np.mean(wrist_3d_samples, axis=0)

                    shoulder_ref_2d = np.mean(shoulder_2d_samples, axis=0)
                    elbow_ref_2d = np.mean(elbow_2d_samples, axis=0)
                    wrist_ref_2d = np.mean(wrist_2d_samples, axis=0)

                    arm_length_data = estimate_arm_length_from_points(
                        shoulder_ref,
                        elbow_ref,
                        wrist_ref
                    )

                    reference_data = {
                        "arm": ARM_TO_TRACK,

                        "shoulder_reference_3d": shoulder_ref,
                        "elbow_reference_3d": elbow_ref,
                        "wrist_reference_3d": wrist_ref,

                        "shoulder_reference_2d": shoulder_ref_2d,
                        "elbow_reference_2d": elbow_ref_2d,
                        "wrist_reference_2d": wrist_ref_2d,

                        "arm_length_data": arm_length_data,

                        "num_frames": NUM_REFERENCE_FRAMES,
                        "units": "millimeters"
                    }

                    T_arm_camera = get_T_wrist_camera(reference_data)

                    collecting_reference = False

                    prev_p_arm_db = None
                    prev_time = None
                    filtered_velocity = None
                    records.clear()

                    save_velocity_json(reference_data, records)

                    print("Reference collection complete.")
                    print("Wrist reference:", wrist_ref)
                    print("T_arm_camera:")
                    print(T_arm_camera)

            # --------------------------------------------------
            # Track arm motion after reference is collected
            # --------------------------------------------------

            elif reference_data is not None and T_arm_camera is not None and arm_data is not None:

                current_time = time.time()

                wrist_camera = arm_data["wrist_3d"]

                # Camera frame -> arm frame
                p_arm = get_p_arm(
                    wrist_camera,
                    T_arm_camera
                )

                # Position deadband in arm frame
                p_arm_db = applyPositionDB(p_arm)

                velocity_arm = np.zeros(3)
                velocity_arm_filtered = np.zeros(3)
                velocity_robot_command = np.zeros(3)

                if prev_p_arm_db is not None and prev_time is not None:

                    dt = current_time - prev_time

                    velocity_arm = Cal_velocity(
                        p_arm_db,
                        prev_p_arm_db,
                        dt
                    )

                    filtered_velocity = exponential_filter(
                        velocity_arm,
                        filtered_velocity,
                        ALPHA_LINEAR
                    )

                    velocity_arm_filtered = filtered_velocity

                    # Scale human velocity to robot command velocity
                    velocity_robot_command = MOTION_SCALE * velocity_arm_filtered

                    record = {
                        "timestamp": current_time,
                        "dt": dt,

                        "wrist_camera_mm": wrist_camera,

                        "position_arm_mm": p_arm,
                        "position_arm_deadband_mm": p_arm_db,

                        "velocity_arm_mm_s": velocity_arm,
                        "velocity_arm_filtered_mm_s": velocity_arm_filtered,

                        "velocity_robot_command_mm_s": velocity_robot_command,
                    }

                    records.append(record)

                    # JSON is always saved for robot control
                    save_velocity_json(reference_data, records)

                prev_p_arm_db = p_arm_db
                prev_time = current_time

                cv.putText(
                    frame,
                    f"p_arm mm: x={p_arm_db[0]:.1f}, y={p_arm_db[1]:.1f}, z={p_arm_db[2]:.1f}",
                    (20, 40),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                cv.putText(
                    frame,
                    f"v_arm mm/s: x={velocity_arm_filtered[0]:.1f}, y={velocity_arm_filtered[1]:.1f}, z={velocity_arm_filtered[2]:.1f}",
                    (20, 75),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )

                cv.putText(
                    frame,
                    f"robot cmd mm/s: x={velocity_robot_command[0]:.1f}, y={velocity_robot_command[1]:.1f}, z={velocity_robot_command[2]:.1f}",
                    (20, 110),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

            # --------------------------------------------------
            # Idle text before reference collection
            # --------------------------------------------------

            else:
                cv.putText(
                    frame,
                    "Press ENTER to collect wrist reference",
                    (20, 40),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2
                )

                cv.putText(
                    frame,
                    "Press q to quit",
                    (20, 75),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2
                )

            # --------------------------------------------------
            # Resize display
            # --------------------------------------------------

            if DISPLAY_SCALE != 1.0:
                display_frame = cv.resize(
                    frame,
                    None,
                    fx=DISPLAY_SCALE,
                    fy=DISPLAY_SCALE
                )
            else:
                display_frame = frame

            # --------------------------------------------------
            # Video recording
            # --------------------------------------------------

            if ENABLE_VIDEO_RECORDING:

                if video_writer is None:
                    h, w = display_frame.shape[:2]

                    fourcc = cv.VideoWriter_fourcc(*"mp4v")

                    video_writer = cv.VideoWriter(
                        str(VIDEO_OUTPUT),
                        fourcc,
                        VIDEO_FPS,
                        (w, h)
                    )

                video_writer.write(display_frame)

            # --------------------------------------------------
            # Show window
            # --------------------------------------------------

            cv.imshow(
                window_name,
                display_frame
            )

            key = cv.waitKey(1) & 0xFF

            # ENTER starts reference collection
            if key in [10, 13] and not collecting_reference:

                collecting_reference = True

                shoulder_3d_samples.clear()
                elbow_3d_samples.clear()
                wrist_3d_samples.clear()

                shoulder_2d_samples.clear()
                elbow_2d_samples.clear()
                wrist_2d_samples.clear()

                reference_data = None
                T_arm_camera = None

                prev_p_arm_db = None
                prev_time = None
                filtered_velocity = None
                records.clear()

                print(f"Collecting {NUM_REFERENCE_FRAMES} reference frames...")

            # q quits
            if key == ord("q"):
                break

            if cv.getWindowProperty(
                window_name,
                cv.WND_PROP_VISIBLE
            ) < 1:
                break

    finally:
        if video_writer is not None:
            video_writer.release()

        cv.destroyAllWindows()
        zed.close()

        print("Program closed.")


if __name__ == "__main__":
    main()