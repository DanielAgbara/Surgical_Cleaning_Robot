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

VIDEO_PATH = ROOT / "data" / "Video"
VIDEO_PATH.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = DATA_PATH / "raw_arm_tracking.json"

VIDEO_OUTPUT = VIDEO_PATH / "motion_capture_recording.mp4"


# ============================================================
# Settings
# ============================================================

OUTPUT_INTERVAL = 0.1
NUM_REFERENCE_FRAMES = 30
ARM_TO_TRACK = "left"

ENABLE_VIDEO_RECORDING = True
VIDEO_FPS = 30

# ============================================================
# Output Helper Functions
# ============================================================

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

# ============================================================
# Helper functions
# ============================================================

def make_json_safe(data):
    """
    Convert numpy arrays and numpy numbers to JSON-safe Python types.
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


def get_arm_length(shoulder_pos, wrist_pos):
    """
    Compute shoulder-to-wrist arm length using full 3D distance.
    Units are the same as the ZED coordinate units.
    """

    shoulder_pos = np.asarray(shoulder_pos, dtype=float).reshape(3)
    wrist_pos = np.asarray(wrist_pos, dtype=float).reshape(3)

    return np.linalg.norm(wrist_pos - shoulder_pos)


def save_raw_json(reference_data, records):
    """
    Save reference data and raw tracking records to JSON.
    """

    output = {
        "description": "Raw unfiltered arm tracking data",
        "units": "millimeters",
        "coordinate_system": "ZED IMAGE frame",
        "arm_tracked": ARM_TO_TRACK,
        "output_interval_s": OUTPUT_INTERVAL,
        "num_reference_frames": NUM_REFERENCE_FRAMES,
        "reference_data": reference_data,
        "records": records,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(
            make_json_safe(output),
            f,
            indent=4
        )


# ============================================================
# Main function
# ============================================================

def main():
    """
    Raw arm tracking data collection.

    ENTER:
        Collect reference position over NUM_REFERENCE_FRAMES frames.
        The averaged reference becomes the first previous position.

    q:
        Quit.

    After reference:
        Saves raw unfiltered data blocks:

            time block
            shoulder position
            wrist position
            change in wrist position from previous
            change in shoulder position from previous
            arm length
    """

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30

    # Raw body keypoints will be in millimeters.
    init_params.coordinate_units = sl.UNIT.MILLIMETER

    # ZED IMAGE frame:
    #   X = image right
    #   Y = image down
    #   Z = forward/depth
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

    err = zed.open(init_params)

    if err != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera:", err)
        return

    print("Camera opened!")
    print(f"Saving raw tracking JSON to: {OUTPUT_JSON}")

    try:
        body_runtime = setup_body_tracking(zed)

        image = sl.Mat()
        bodies = sl.Bodies()
        runtime = sl.RuntimeParameters()

        window_name = "Raw Arm Tracking"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)

        # --------------------------------------------------------
        # Reference collection state
        # --------------------------------------------------------

        collecting_reference = False
        reference_collected = False

        shoulder_ref_samples = []
        wrist_ref_samples = []

        reference_data = None

        # --------------------------------------------------------
        # Previous position state
        # --------------------------------------------------------

        prev_shoulder_pos = None
        prev_wrist_pos = None

        # --------------------------------------------------------
        # Output state
        # --------------------------------------------------------

        records = []
        sample_number = 0
        start_time = None
        last_output_time = time.time()
                
        # --------------------------------------------------------
        # Video recording state
        # --------------------------------------------------------

        video_writer = None
        video_recording_started = False

        # Create empty JSON at startup.
        save_raw_json(reference_data, records)

        print("Press ENTER to collect reference position.")
        print("Press q to quit.")

        while True:
            # ----------------------------------------------------
            # Grab frame
            # ----------------------------------------------------

            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()

            if frame.shape[2] == 4:
                frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

            # ----------------------------------------------------
            # Retrieve body tracking
            # ----------------------------------------------------

            zed.retrieve_bodies(bodies, body_runtime)

            body = get_single_body(
                bodies,
                mode="closest"
            )

            arm_data = None
            shoulder_pos = None
            wrist_pos = None

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

                shoulder_pos = np.asarray(
                    arm_data["shoulder_3d"],
                    dtype=float
                ).reshape(3)

                wrist_pos = np.asarray(
                    arm_data["wrist_3d"],
                    dtype=float
                ).reshape(3)

                # ------------------------------------------------
                # Collect averaged reference
                # ------------------------------------------------

                if collecting_reference:
                    shoulder_ref_samples.append(
                        shoulder_pos.copy()
                    )

                    wrist_ref_samples.append(
                        wrist_pos.copy()
                    )

                    count = len(wrist_ref_samples)

                    cv.putText(
                        frame,
                        f"Collecting reference: {count}/{NUM_REFERENCE_FRAMES}",
                        (30, 115),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 255),
                        2,
                        cv.LINE_AA
                    )

                    if count >= NUM_REFERENCE_FRAMES:
                        shoulder_ref = np.mean(
                            shoulder_ref_samples,
                            axis=0
                        )

                        wrist_ref = np.mean(
                            wrist_ref_samples,
                            axis=0
                        )

                        reference_arm_length = get_arm_length(
                            shoulder_ref,
                            wrist_ref
                        )

                        reference_data = {
                            "reference_method": "average_of_multiple_frames",
                            "num_reference_frames": NUM_REFERENCE_FRAMES,
                            "shoulder_position_mm": shoulder_ref,
                            "wrist_position_mm": wrist_ref,
                            "arm_length_mm": reference_arm_length,
                        }

                        # Averaged reference becomes first previous position.
                        prev_shoulder_pos = shoulder_ref.copy()
                        prev_wrist_pos = wrist_ref.copy()

                        collecting_reference = False
                        reference_collected = True

                        records.clear()
                        sample_number = 0
                        start_time = time.time()
                        last_output_time = start_time

                        save_raw_json(
                            reference_data,
                            records
                        )

                        print("\nReference collected.")
                        print("Reference shoulder:", shoulder_ref)
                        print("Reference wrist:", wrist_ref)
                        print(f"Reference arm length: {reference_arm_length:.2f} mm")

                # ------------------------------------------------
                # Save raw measurement blocks after reference
                # ------------------------------------------------

                elif reference_collected:
                    current_time = time.time()

                    if current_time - last_output_time >= OUTPUT_INTERVAL:
                        sample_number += 1

                        elapsed_time = current_time - start_time

                        current_arm_length = get_arm_length(
                            shoulder_pos,
                            wrist_pos
                        )

                        wrist_delta_from_previous = (
                            wrist_pos - prev_wrist_pos
                        )

                        shoulder_delta_from_previous = (
                            shoulder_pos - prev_shoulder_pos
                        )

                        record = {
                            "sample": sample_number,

                            "time": {
                                "elapsed_s": elapsed_time,
                                "timestamp_s": current_time,
                            },

                            "shoulder_position_mm": shoulder_pos,
                            "wrist_position_mm": wrist_pos,

                            "change_from_previous": {
                                "wrist_delta_mm": wrist_delta_from_previous,
                                "shoulder_delta_mm": shoulder_delta_from_previous,
                            },

                            "arm_length_mm": current_arm_length,
                        }

                        records.append(record)

                        save_raw_json(
                            reference_data,
                            records
                        )

                        print(
                            f"\nTime block: {sample_number}\n"
                            f"Shoulder: {shoulder_pos.tolist()}\n"
                            f"Wrist: {wrist_pos.tolist()}\n"
                            f"Wrist delta: {wrist_delta_from_previous.tolist()}\n"
                            f"Shoulder delta: {shoulder_delta_from_previous.tolist()}\n"
                            f"Arm length: {current_arm_length:.2f} mm"
                        )

                        # Current raw measurements become previous raw measurements.
                        prev_shoulder_pos = shoulder_pos.copy()
                        prev_wrist_pos = wrist_pos.copy()

                        last_output_time = current_time

            # ----------------------------------------------------
            # Display status
            # ----------------------------------------------------

            if collecting_reference:
                status_text = "Collecting reference | Keep arm still"
                status_color = (0, 255, 255)

            elif reference_collected:
                status_text = "Reference collected | Recording raw data"
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

            # ----------------------------------------------------
            # Video recording
            # ----------------------------------------------------
            # Records the displayed frame with arm tracking visuals.
            # The video starts only after reference is collected,
            # so the recording corresponds to actual motion capture.

            if ENABLE_VIDEO_RECORDING and reference_collected:

                if video_writer is None:
                    frame_height, frame_width = frame.shape[:2]

                    fourcc = cv.VideoWriter_fourcc(*"mp4v")

                    video_writer = cv.VideoWriter(
                        str(VIDEO_OUTPUT),
                        fourcc,
                        VIDEO_FPS,
                        (frame_width, frame_height)
                    )

                    video_recording_started = True

                    print(f"Video recording started: {VIDEO_OUTPUT}")

                video_writer.write(frame)
            
            cv.imshow(
                window_name,
                frame
            )

            key = cv.waitKey(1) & 0xFF

            # ----------------------------------------------------
            # Quit
            # ----------------------------------------------------

            if key == ord("q"):
                break

            if cv.getWindowProperty(
                window_name,
                cv.WND_PROP_VISIBLE
            ) < 1:
                break

            # ----------------------------------------------------
            # ENTER starts reference collection
            # ----------------------------------------------------

            if key in [10, 13]:
                if arm_data is None:
                    print("No valid right arm detected. Cannot collect reference.")
                    continue

                collecting_reference = True
                reference_collected = False

                shoulder_ref_samples.clear()
                wrist_ref_samples.clear()

                reference_data = None
                records.clear()

                prev_shoulder_pos = None
                prev_wrist_pos = None

                save_raw_json(
                    reference_data,
                    records
                )

                print(f"Collecting {NUM_REFERENCE_FRAMES} reference frames...")

    finally:
        cv.destroyAllWindows()
        zed.close()
        print("Program closed.")


if __name__ == "__main__":
    main()