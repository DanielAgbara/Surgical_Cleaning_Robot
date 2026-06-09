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

# ============================================================
# File paths
# ============================================================

ROOT = Path("/home/agbara-admin/Documents/Surgical_Cleaning_Robot")

DATA_PATH = ROOT / "data" / "arm_tracking"
DATA_PATH.mkdir(parents=True, exist_ok=True)

OUTPUT_TXT = DATA_PATH / "wrist_delta_output.txt"


# ============================================================
# Settings
# ============================================================

DEADBAND_MM = 200
ARM_LENGTH_DB_MM = 100

# How often to output delta data after reference is collected
OUTPUT_INTERVAL = 1.0  # seconds


def apply_body_tracking_DB(keypoint_pos, previous_keypoint_pos):
    """
    Apply deadband to a body-tracking keypoint.

    If the current keypoint movement is smaller than DEADBAND_MM
    along an axis, that axis keeps its previous value.

    This helps reduce small ZED body-tracking jitter.

    Parameters
    ----------
    keypoint_pos : np.ndarray, shape (3,)
        Current keypoint position [x, y, z] in camera frame.

    previous_keypoint_pos : np.ndarray, shape (3,)
        Previous keypoint position [x, y, z] in camera frame.

    Returns
    -------
    keypoint_pos_DB : np.ndarray, shape (3,)
        Deadband-filtered keypoint position.
    """

    keypoint_pos = np.asarray(keypoint_pos, dtype=float).reshape(3)
    previous_keypoint_pos = np.asarray(previous_keypoint_pos, dtype=float).reshape(3)

    keypoint_pos_DB = keypoint_pos.copy()

    for i in range(len(keypoint_pos)):
        if np.abs(keypoint_pos[i] - previous_keypoint_pos[i]) < DEADBAND_MM:
            keypoint_pos_DB[i] = previous_keypoint_pos[i]

    return keypoint_pos_DB

def get_arm_length(shoulder_ref, wrist_ref):
    """
    Compute reference arm length using full 3D distance.

    This is better than using only x because the arm can move
    left/right/up/down while keeping roughly the same true length.
    """

    shoulder_ref = np.asarray(shoulder_ref, dtype=float).reshape(3)
    wrist_ref = np.asarray(wrist_ref, dtype=float).reshape(3)

    arm_vector = wrist_ref - shoulder_ref
    arm_length = np.linalg.norm(arm_vector)

    return arm_length


def apply_DB_armlength(robot_delta,
                       current_wrist_pos,
                       current_shoulder_pos,
                       ref_arm_length):
    """
    Remove false forward/backward robot motion when current arm length
    is close to the reference arm length.

    Assumption:
        robot_delta[0] = forward/backward motion

    If the measured arm length has not changed enough, then the
    forward/backward delta is treated as body-tracking noise.
    """

    robot_delta = np.asarray(robot_delta, dtype=float).reshape(3)
    current_wrist_pos = np.asarray(current_wrist_pos, dtype=float).reshape(3)
    current_shoulder_pos = np.asarray(current_shoulder_pos, dtype=float).reshape(3)

    robot_delta_DB = robot_delta.copy()

    current_arm_length = get_arm_length(
        current_shoulder_pos,
        current_wrist_pos
    )

    arm_length_error = current_arm_length - ref_arm_length

    if abs(arm_length_error) < ARM_LENGTH_DB_MM:
        robot_delta_DB[0] = 0.0

    return robot_delta_DB


def get_T_robot_camera():
    """
    Build the homogeneous transform from camera frame to robot frame.

    Camera frame:
        X = Right
        Y = Down
        Z = Forward

    Robot frame:
        X_robot =  X_camera
        Y_robot = -Z_camera
        Z_robot = -Y_camera

    This gives:

        x_robot =  x_camera
        y_robot = -z_camera
        z_robot = -y_camera

    Returns
    -------
    T_robot_camera : np.ndarray, shape (4, 4)
        Homogeneous transform from camera frame to robot frame.
    """

    R_robot_camera = np.array([
        [1.0,  0.0,  0.0],
        [0.0,  0.0, -1.0],
        [0.0, -1.0,  0.0]
    ])

    T_robot_camera = np.eye(4)
    T_robot_camera[:3, :3] = R_robot_camera

    return T_robot_camera


def apply_T(pos, T):
    """
    Transform a 3D point from camera frame to robot frame.

    This is for actual positions/points, not pure displacement vectors.

    Parameters
    ----------
    pos : array-like, shape (3,)
        Position in camera frame [x, y, z].

    T : np.ndarray, shape (4, 4)
        Homogeneous transform.

    Returns
    -------
    pos_robot : np.ndarray, shape (3,)
        Position in robot frame [x, y, z].
    """

    pos = np.asarray(pos, dtype=float).reshape(3)

    pos_h = np.append(pos, 1.0)

    pos_robot_h = T @ pos_h

    return pos_robot_h[:3]


def cal_wrist_delta(current_shoulder_pos, current_wrist_pos,
                    prev_shoulder_pos, prev_wrist_pos, T):
    """
    Calculate wrist displacement relative to the shoulder.

    Steps:
        1. Apply deadband to current shoulder and wrist positions.
        2. Compute current wrist vector relative to shoulder.
        3. Compute previous wrist vector relative to shoulder.
        4. Subtract previous relative vector from current relative vector.
        5. Rotate the delta from camera frame to robot frame.

    Using wrist relative to shoulder helps reduce errors caused by
    whole-body drift or small camera/body-tracking shifts.

    Inputs are assumed to be in millimeters.
    """

    current_shoulder_pos = np.asarray(current_shoulder_pos, dtype=float).reshape(3)
    current_wrist_pos = np.asarray(current_wrist_pos, dtype=float).reshape(3)

    prev_shoulder_pos = np.asarray(prev_shoulder_pos, dtype=float).reshape(3)
    prev_wrist_pos = np.asarray(prev_wrist_pos, dtype=float).reshape(3)

    # Apply deadband to current measurements.
    current_shoulder_pos_DB = apply_body_tracking_DB(
        current_shoulder_pos,
        prev_shoulder_pos
    )

    current_wrist_pos_DB = apply_body_tracking_DB(
        current_wrist_pos,
        prev_wrist_pos
    )

    # Current wrist position relative to current shoulder.
    current_wrist_rel = current_wrist_pos_DB - current_shoulder_pos_DB

    # Previous wrist position relative to previous shoulder.
    prev_wrist_rel = prev_wrist_pos - prev_shoulder_pos

    # Delta in camera frame.
    delta_camera = current_wrist_rel - prev_wrist_rel

    # Since this is a displacement/vector, use rotation only.
    R_robot_camera = T[:3, :3]

    delta_robot = R_robot_camera @ delta_camera

    return delta_robot


def describe_delta(delta_robot):
    """
    Convert numeric robot-frame delta into readable motion meaning.

    Robot frame used here:
        +X = forward
        -X = backward

        +Y = left
        -Y = right

        +Z = up
        -Z = down
    """

    dx, dy, dz = delta_robot

    if dx >= 0:
        x_dir = "forward"
    else:
        x_dir = "backward"

    if dy >= 0:
        y_dir = "left"
    else:
        y_dir = "right"

    if dz >= 0:
        z_dir = "up"
    else:
        z_dir = "down"

    return (
        f"Went {x_dir} by {abs(dx):.2f} mm, "
        f"went {y_dir} by {abs(dy):.2f} mm, "
        f"went {z_dir} by {abs(dz):.2f} mm"
    )


def write_output_to_txt(
    sample_number,
    delta_robot_DB,
    shoulder_pos,
    wrist_pos,
    current_arm_length
):
    """
    Write tracking data to txt file.
    """

    meaning = describe_delta(delta_robot_DB)

    output_text = (
        f"\nTime: {sample_number}\n"
        f"Current Shoulder Position: {shoulder_pos.tolist()}\n"
        f"Current Wrist Position: {wrist_pos.tolist()}\n"
        f"Current Arm Length: {current_arm_length:.2f} mm\n"
        f"Delta Robot DB: {delta_robot_DB.tolist()}\n"
        f"Meaning: {meaning}\n"
    )

    with open(OUTPUT_TXT, "a") as f:
        f.write(output_text)

    return output_text


def main():
    """
    Main ZED body-tracking loop for wrist delta tracking.

    ENTER:
        Collect reference position.

    q:
        Quit.

    After reference:
        Outputs wrist delta every OUTPUT_INTERVAL seconds.
    """

    with open(OUTPUT_TXT, "w") as f:
        f.write("Wrist Delta Output\n")
        f.write("==================\n")
        f.write(f"Deadband: {DEADBAND_MM} mm\n")
        f.write(f"Output interval: {OUTPUT_INTERVAL} seconds\n\n")

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30

    # Use millimeters since DEADBAND_MM is in mm.
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

    err = zed.open(init_params)

    if err != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera:", err)
        return

    print("Camera opened!")
    print(f"Writing wrist delta output to: {OUTPUT_TXT}")
    

    try:
        body_runtime = setup_body_tracking(zed)

        image = sl.Mat()
        bodies = sl.Bodies()
        runtime = sl.RuntimeParameters()

        window_name = "Arm Motion Tracking"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)

        T_robot_camera = get_T_robot_camera()

        reference_collected = False
        reference_arm_length = None

        prev_shoulder_pos = None
        prev_wrist_pos = None

        latest_arm_data = None
        last_output_time = time.time()
        sample_number = 0

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

            latest_arm_data = None

            if body is not None:
                latest_arm_data = get_arm_points(body, arm="right")

            if latest_arm_data is not None:
                frame = draw_arm_points_and_lines(frame, latest_arm_data)

                shoulder_pos = np.asarray(
                    latest_arm_data["shoulder_3d"],
                    dtype=float
                ).reshape(3)

                wrist_pos = np.asarray(
                    latest_arm_data["wrist_3d"],
                    dtype=float
                ).reshape(3)

                if reference_collected:
                    current_time = time.time()

                    if current_time - last_output_time >= OUTPUT_INTERVAL:
                        # ------------------------------------------------
                        # Calculate raw wrist delta
                        # ------------------------------------------------
                        delta_robot = cal_wrist_delta(
                            shoulder_pos,
                            wrist_pos,
                            prev_shoulder_pos,
                            prev_wrist_pos,
                            T_robot_camera
                        )

                        # ------------------------------------------------
                        # Apply arm-length deadband
                        #
                        # If arm length hasn't changed enough from the
                        # reference arm length, suppress forward/backward
                        # motion caused by body-tracking noise.
                        # ------------------------------------------------
                        delta_robot_DB = apply_DB_armlength(
                            delta_robot,
                            wrist_pos,
                            shoulder_pos,
                            reference_arm_length
                        )
                        current_arm_length = get_arm_length(
                            shoulder_pos,
                            wrist_pos
                        )
                        sample_number +=1

                        output_text = write_output_to_txt(
                            sample_number,
                            delta_robot_DB,
                            shoulder_pos,
                            wrist_pos,
                            current_arm_length
                        )

                        print(output_text)

                        prev_shoulder_pos = shoulder_pos.copy()
                        prev_wrist_pos = wrist_pos.copy()

                        last_output_time = current_time

            if reference_collected:
                status_text = "Reference collected | Tracking wrist delta"
                status_color = (0, 255, 0)
            else:
                status_text = "Press ENTER to collect reference"
                status_color = (0, 255, 255)

            cv.putText(
                frame,
                status_text,
                (30, 40),
                cv.FONT_HERSHEY_SIMPLEX,
                0.9,
                status_color,
                2,
                cv.LINE_AA
            )

            cv.putText(
                frame,
                "Press q to quit",
                (30, 75),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv.LINE_AA
            )

            cv.imshow(window_name, frame)

            key = cv.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
                break

            if key in [10, 13]:
                if latest_arm_data is None:
                    print("No valid right arm detected. Cannot collect reference.")
                    continue

                shoulder_pos = np.asarray(
                    latest_arm_data["shoulder_3d"],
                    dtype=float
                ).reshape(3)

                wrist_pos = np.asarray(
                    latest_arm_data["wrist_3d"],
                    dtype=float
                ).reshape(3)

                reference_collected = True

                prev_shoulder_pos = shoulder_pos.copy()
                prev_wrist_pos = wrist_pos.copy()

                reference_arm_length = get_arm_length(
                    prev_shoulder_pos,
                    prev_wrist_pos
                )

                last_output_time = time.time()

                reference_text = (
                    "\nReference collected.\n"
                    f"Reference shoulder: {prev_shoulder_pos.tolist()}\n"
                    f"Reference wrist: {prev_wrist_pos.tolist()}\n"
                )

                print(reference_text)

                with open(OUTPUT_TXT, "a") as f:
                    f.write(reference_text)

    finally:
        cv.destroyAllWindows()
        zed.close()
        print("Program closed.")


if __name__ == "__main__":
    main()