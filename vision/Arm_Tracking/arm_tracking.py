import pyzed.sl as sl
import cv2 as cv
import numpy as np
import json
import time
from pathlib import Path


# ============================================================
# Choose body tracking model
# ============================================================

BODY_MODEL = 34   # Use 18 or 34

if BODY_MODEL == 34:
    from body34 import (
        setup_body_tracking,
        get_single_body,
        get_arm_points,
        draw_arm_points_and_lines,
    )

elif BODY_MODEL == 18:
    from body18 import (
        setup_body_tracking,
        get_single_body,
        get_arm_points,
        draw_arm_points_and_lines,
    )

else:
    raise ValueError("BODY_MODEL must be 18 or 34")


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
ARM_TO_TRACK = "right"

ENABLE_VIDEO_RECORDING = True
VIDEO_FPS = 30


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


def get_arm_length(shoulder_pos, hand_pos):
    """
    Compute shoulder-to-hand arm length using full 3D distance.

    For BODY_34:
        shoulder -> hand

    For BODY_18:
        hand_pos falls back to wrist_pos because BODY_18
        does not provide a separate hand keypoint.
    """

    shoulder_pos = np.asarray(shoulder_pos, dtype=float).reshape(3)
    hand_pos = np.asarray(hand_pos, dtype=float).reshape(3)

    return np.linalg.norm(hand_pos - shoulder_pos)


def save_raw_json(reference_data, records):
    """
    Save reference data and raw tracking records to JSON.
    """

    output = {
        "description": "Raw unfiltered arm tracking data",
        "units": "millimeters",
        "coordinate_system": "ZED IMAGE frame",
        "body_model": BODY_MODEL,
        "arm_tracked": ARM_TO_TRACK,
        "output_interval_s": OUTPUT_INTERVAL,
        "num_reference_frames": NUM_REFERENCE_FRAMES,
        "arm_length_definition": "shoulder_to_hand",
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

    q:
        Quit.

    After reference:
        Saves raw unfiltered data blocks:

            time
            shoulder position
            wrist position
            hand position
            change in wrist position from previous
            change in hand position from previous
            change in shoulder position from previous
            shoulder-to-hand arm length
    """

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

    err = zed.open(init_params)

    if err != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera:", err)
        return

    print("Camera opened!")
    print(f"Using BODY_{BODY_MODEL}")
    print(f"Tracking {ARM_TO_TRACK} arm")
    print(f"Saving raw tracking JSON to: {OUTPUT_JSON}")

    try:
        body_runtime = setup_body_tracking(zed)

        image = sl.Mat()
        bodies = sl.Bodies()
        runtime = sl.RuntimeParameters()

        window_name = f"Raw Arm Tracking BODY_{BODY_MODEL}"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)

        collecting_reference = False
        reference_collected = False

        shoulder_ref_samples = []
        wrist_ref_samples = []
        hand_ref_samples = []

        reference_data = None

        prev_shoulder_pos = None
        prev_wrist_pos = None
        prev_hand_pos = None

        records = []
        sample_number = 0
        start_time = None
        last_output_time = time.time()

        video_writer = None

        save_raw_json(reference_data, records)

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
            shoulder_pos = None
            wrist_pos = None
            hand_pos = None

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

                if BODY_MODEL == 34:
                    hand_pos = np.asarray(
                        arm_data["hand_3d"],
                        dtype=float
                    ).reshape(3)
                else:
                    hand_pos = wrist_pos.copy()

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

                    hand_ref_samples.append(
                        hand_pos.copy()
                    )

                    count = len(hand_ref_samples)

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

                        hand_ref = np.mean(
                            hand_ref_samples,
                            axis=0
                        )

                        reference_arm_length = get_arm_length(
                            shoulder_ref,
                            hand_ref
                        )

                        reference_data = {
                            "reference_method": "average_of_multiple_frames",
                            "num_reference_frames": NUM_REFERENCE_FRAMES,
                            "body_model": BODY_MODEL,
                            "arm": ARM_TO_TRACK,

                            "shoulder_position_mm": shoulder_ref,
                            "wrist_position_mm": wrist_ref,
                            "hand_position_mm": hand_ref,

                            "arm_length_mm": reference_arm_length,
                            "arm_length_definition": "shoulder_to_hand",
                        }

                        prev_shoulder_pos = shoulder_ref.copy()
                        prev_wrist_pos = wrist_ref.copy()
                        prev_hand_pos = hand_ref.copy()

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
                        print("Reference hand:", hand_ref)
                        print(
                            f"Reference shoulder-to-hand length: "
                            f"{reference_arm_length:.2f} mm"
                        )

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
                            hand_pos
                        )

                        wrist_delta_from_previous = (
                            wrist_pos - prev_wrist_pos
                        )

                        hand_delta_from_previous = (
                            hand_pos - prev_hand_pos
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
                            "hand_position_mm": hand_pos,

                            "change_from_previous": {
                                "wrist_delta_mm": wrist_delta_from_previous,
                                "hand_delta_mm": hand_delta_from_previous,
                                "shoulder_delta_mm": shoulder_delta_from_previous,
                            },

                            "arm_length_mm": current_arm_length,
                            "arm_length_definition": "shoulder_to_hand",
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
                            f"Hand: {hand_pos.tolist()}\n"
                            f"Wrist delta: {wrist_delta_from_previous.tolist()}\n"
                            f"Hand delta: {hand_delta_from_previous.tolist()}\n"
                            f"Shoulder delta: {shoulder_delta_from_previous.tolist()}\n"
                            f"Arm length shoulder-to-hand: {current_arm_length:.2f} mm"
                        )

                        prev_shoulder_pos = shoulder_pos.copy()
                        prev_wrist_pos = wrist_pos.copy()
                        prev_hand_pos = hand_pos.copy()

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
                f"BODY_{BODY_MODEL} | Tracking {ARM_TO_TRACK} arm",
                (30, 75),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv.LINE_AA
            )

            cv.putText(
                frame,
                "Press q to quit",
                (30, 105),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv.LINE_AA
            )

            # ----------------------------------------------------
            # Video recording
            # ----------------------------------------------------

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

                    print(f"Video recording started: {VIDEO_OUTPUT}")

                video_writer.write(frame)

            cv.imshow(
                window_name,
                frame
            )

            key = cv.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if cv.getWindowProperty(
                window_name,
                cv.WND_PROP_VISIBLE
            ) < 1:
                break

            if key in [10, 13]:
                if arm_data is None:
                    print(
                        f"No valid {ARM_TO_TRACK} arm detected. "
                        "Cannot collect reference."
                    )
                    continue

                collecting_reference = True
                reference_collected = False

                shoulder_ref_samples.clear()
                wrist_ref_samples.clear()
                hand_ref_samples.clear()

                reference_data = None
                records.clear()

                prev_shoulder_pos = None
                prev_wrist_pos = None
                prev_hand_pos = None

                save_raw_json(
                    reference_data,
                    records
                )

                print(
                    f"Collecting {NUM_REFERENCE_FRAMES} "
                    "reference frames..."
                )

    finally:
        if video_writer is not None:
            video_writer.release()

        cv.destroyAllWindows()
        zed.close()
        print("Program closed.")


if __name__ == "__main__":
    main()