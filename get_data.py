#!/usr/bin/env python3
"""
data script combining:

1. Object detection
   - Detect only the selected object, currently "sink"
   - Do not show masks for other objects
   - Calculate sink 2D centroid
   - Calculate sink 3D centroid
   - Fit plane only on detected sink points

2. Arm tracking
   - Start from a reference position
   - Record all data after reference
   - Add flag showing whether hand is close to sink

3. Output
   - JSON dataset
   - Optional video recording
"""

import cv2
import json
import time
from pathlib import Path

import numpy as np
import pyzed.sl as sl


# ============================================================
# Project Root
# ============================================================
ROOT = Path("/home/agbara-admin/Documents/Surgical_Cleaning_Robot")
ROOT = Path(__file__).resolve().parent
# ============================================================
# Import project helpers
# ============================================================

from vision.Object_Detection.helper import (
    setup_detectron2,
    detect_object,
    process_detected_object,
    make_object_mask_and_plane_visualization,
    draw_plane_pixels_on_object,
)

from vision.Arm_Tracking.helper import (
    euclidean_distance,
    extract_arm_positions,
    build_reference_data,
    build_arm_tracking_record,
    draw_distance_info,
    draw_main_status,
)


# ============================================================
# Choose body tracking model
# ============================================================

BODY_MODEL = 34   # Use 18 or 34

if BODY_MODEL == 34:
    from vision.Arm_Tracking.body34 import (
        setup_body_tracking,
        get_single_body,
        get_arm_points,
        draw_arm_points_and_lines,
    )

elif BODY_MODEL == 18:
    from vision.Arm_Tracking.body18 import (
        setup_body_tracking,
        get_single_body,
        get_arm_points,
        draw_arm_points_and_lines,
    )

else:
    raise ValueError("BODY_MODEL must be 18 or 34")


# ============================================================
# Paths
# ============================================================

DATA_PATH = ROOT / "data" / "arm_tracking"
DATA_PATH.mkdir(parents=True, exist_ok=True)

VIDEO_PATH = ROOT / "data" / "Video"
VIDEO_PATH.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = DATA_PATH / "tray_arm_tracking_raw.json"
VIDEO_OUTPUT = VIDEO_PATH / "tray_arm_tracking_recording.mp4"


# ============================================================
# Settings
# ============================================================

ARM_TO_TRACK = "right"
OBJECT_NAME_TO_TRACK = "tray"

# ------------------------------------------------------------
# Main timing control
# ------------------------------------------------------------
# Change only this value.
# Everything else below updates automatically.
#
# For offline human demonstration learning:
#   5 FPS is okay because the camera is stationary.
#
# For future real-time teleoperation:
#   use 15 or 30 FPS.
# ------------------------------------------------------------

CAMERA_FPS = 5

# Record one sample per camera frame.
OUTPUT_INTERVAL = 1.0 / CAMERA_FPS

# Collect one second of reference data.
NUM_REFERENCE_FRAMES = CAMERA_FPS

ENABLE_VIDEO_RECORDING = False

# Match saved video FPS to camera FPS.
VIDEO_FPS = CAMERA_FPS

# ZED units are millimeters in this script.
HAND_TO_OBJECT_CLOSE_DISTANCE_MM = 500.0

# Plane fitting settings.
PLANE_DISTANCE_THRESHOLD_MM = 15.0
PLANE_RANSAC_POINTS = 3
PLANE_RANSAC_ITERATIONS = 1000
MAX_OBJECT_PLANE_POINTS = 60000

# Detectron2 threshold.
DETECTION_THRESHOLD = 0.95


# ============================================================
# JSON helpers
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

    if isinstance(data, np.bool_):
        return bool(data)

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


def save_json(reference_data, records):
    """
    Save reference data and all recorded samples.
    """

    output = {
        "description": "Combined sink detection and arm tracking raw data",
        "units": "millimeters",
        "coordinate_system": "ZED IMAGE frame",
        "body_model": BODY_MODEL,
        "arm_tracked": ARM_TO_TRACK,
        "object_tracked": OBJECT_NAME_TO_TRACK,
        "output_interval_s": OUTPUT_INTERVAL,
        "num_reference_frames": NUM_REFERENCE_FRAMES,
        "hand_to_object_close_distance_mm": HAND_TO_OBJECT_CLOSE_DISTANCE_MM,
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
# ZED setup
# ============================================================

def setup_zed():
    """
    Open ZED camera.

    Important:
        coordinate_units = MILLIMETER
        coordinate_system = IMAGE

    This matches your arm tracking data convention.
    """

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = CAMERA_FPS
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

    err = zed.open(init_params)

    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED camera: {err}")

    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = 50
    runtime.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA

    print("[INFO] ZED camera opened")

    return zed, runtime


# ============================================================
# Main
# ============================================================

def main():
    """
    Controls:

    ENTER:
        Collect reference arm position.

    q:
        Quit.

    Data behavior:
        After reference is collected, every valid arm sample is recorded.
        The JSON tells you whether the hand is close to the sink.
    """

    predictor = setup_detectron2(
        detection_threshold=DETECTION_THRESHOLD
    )

    zed, runtime = setup_zed()
    body_runtime = setup_body_tracking(zed)

    image_zed = sl.Mat()
    point_cloud_zed = sl.Mat()
    bodies = sl.Bodies()

    window_name = "Tray Mask + Tray Plane + Arm Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

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

    save_json(reference_data, records)

    print("[INFO] Press ENTER to collect reference.")
    print("[INFO] Press q to quit.")
    print(f"[INFO] Saving JSON to: {OUTPUT_JSON}")

    try:
        while True:
            grab_status = zed.grab(runtime)

            if grab_status != sl.ERROR_CODE.SUCCESS:
                continue

            # ----------------------------------------------------
            # Retrieve image
            # ----------------------------------------------------

            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            frame = image_zed.get_data()

            if frame.shape[2] == 4:
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGRA2BGR
                )

            frame = np.ascontiguousarray(frame)

            # ----------------------------------------------------
            # Retrieve point cloud
            # ----------------------------------------------------

            zed.retrieve_measure(
                point_cloud_zed,
                sl.MEASURE.XYZRGBA
            )

            point_cloud_np = point_cloud_zed.get_data()
            xyz = point_cloud_np[:, :, :3]

            # ----------------------------------------------------
            # Detect only selected object
            # ----------------------------------------------------

            object_data = detect_object(
                frame,
                predictor
            )

            object_result = process_detected_object(
                object_data,
                xyz,
                plane_distance_threshold_mm=PLANE_DISTANCE_THRESHOLD_MM,
                plane_ransac_points=PLANE_RANSAC_POINTS,
                plane_ransac_iterations=PLANE_RANSAC_ITERATIONS,
                max_object_plane_points=MAX_OBJECT_PLANE_POINTS,
            )

            # ----------------------------------------------------
            # Visualize only sink mask and sink plane
            # ----------------------------------------------------

            frame = make_object_mask_and_plane_visualization(
                frame,
                object_result,
                OBJECT_NAME_TO_TRACK,
                plane_distance_threshold_mm=PLANE_DISTANCE_THRESHOLD_MM,
            )

            frame = draw_plane_pixels_on_object(
                frame,
                object_result,
                xyz,
                plane_distance_threshold_mm=PLANE_DISTANCE_THRESHOLD_MM,
            )

            # ----------------------------------------------------
            # Arm tracking
            # ----------------------------------------------------

            zed.retrieve_bodies(
                bodies,
                body_runtime
            )

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

                shoulder_pos, wrist_pos, hand_pos = extract_arm_positions(
                    arm_data,
                    BODY_MODEL
                )

            # ----------------------------------------------------
            # Reference collection
            # ----------------------------------------------------

            if collecting_reference and arm_data is not None:
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

                cv2.putText(
                    frame,
                    f"Collecting reference: {count}/{NUM_REFERENCE_FRAMES}",
                    (30, 280),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if count >= NUM_REFERENCE_FRAMES:
                    (
                        reference_data,
                        shoulder_ref,
                        wrist_ref,
                        hand_ref,
                    ) = build_reference_data(
                        shoulder_ref_samples,
                        wrist_ref_samples,
                        hand_ref_samples,
                        NUM_REFERENCE_FRAMES,
                        BODY_MODEL,
                        ARM_TO_TRACK,
                    )

                    prev_shoulder_pos = shoulder_ref.copy()
                    prev_wrist_pos = wrist_ref.copy()
                    prev_hand_pos = hand_ref.copy()

                    collecting_reference = False
                    reference_collected = True

                    records.clear()
                    sample_number = 0

                    start_time = time.time()
                    last_output_time = start_time

                    save_json(
                        reference_data,
                        records
                    )

                    print("\n[INFO] Reference collected.")
                    print("[INFO] Shoulder ref:", shoulder_ref)
                    print("[INFO] Wrist ref:", wrist_ref)
                    print("[INFO] Hand ref:", hand_ref)
                    print(
                        "[INFO] Arm length:",
                        reference_data["arm_length_mm"]
                    )

            # ----------------------------------------------------
            # Distance from hand to sink centroid
            # ----------------------------------------------------

            hand_to_object_distance_mm = None
            hand_close_to_object = False

            if (
                reference_collected
                and hand_pos is not None
                and object_result is not None
                and object_result["centroid_3d_mm"] is not None
            ):
                hand_to_object_distance_mm = euclidean_distance(
                    hand_pos,
                    object_result["centroid_3d_mm"]
                )

                hand_close_to_object = (
                    hand_to_object_distance_mm
                    <= HAND_TO_OBJECT_CLOSE_DISTANCE_MM
                )

            frame = draw_distance_info(
                frame,
                hand_to_object_distance_mm,
                hand_close_to_object,
                OBJECT_NAME_TO_TRACK,
            )

            # ----------------------------------------------------
            # Record all data after reference
            # ----------------------------------------------------

            if reference_collected and arm_data is not None:
                current_time = time.time()

                if current_time - last_output_time >= OUTPUT_INTERVAL:
                    last_output_time = current_time

                    sample_number += 1
                    elapsed_time = current_time - start_time

                    arm_tracking_record = build_arm_tracking_record(
                        shoulder_pos,
                        wrist_pos,
                        hand_pos,
                        prev_shoulder_pos,
                        prev_wrist_pos,
                        prev_hand_pos,
                    )

                    record = {
                        "sample": sample_number,

                        "time": {
                            "elapsed_s": elapsed_time,
                            "timestamp_s": current_time,
                        },

                        "arm_tracking": arm_tracking_record,

                        "object_tracking": object_result,

                        "hand_to_object": {
                            "distance_mm": hand_to_object_distance_mm,
                            "close_distance_threshold_mm": HAND_TO_OBJECT_CLOSE_DISTANCE_MM,
                            "hand_close_to_object": hand_close_to_object,
                            "hand_close_to_sink": hand_close_to_object,
                        },

                        "force_data": {
                            "available": False,
                            "force_N": None,
                            "contact": None,
                            "note": "Placeholder until force sensor is integrated",
                        },
                    }

                    records.append(record)

                    save_json(
                        reference_data,
                        records
                    )

                    if hand_close_to_object:
                        close_text = "CLOSE"
                    else:
                        close_text = "NOT CLOSE"

                    print(
                        f"\n[RECORDED] Sample {sample_number} | {close_text}\n"
                        f"Hand: {hand_pos.tolist()}\n"
                        f"Distance to {OBJECT_NAME_TO_TRACK}: {hand_to_object_distance_mm}"
                    )

                    # Always update previous position after each sample.
                    # This keeps deltas continuous even when hand is far.
                    prev_shoulder_pos = shoulder_pos.copy()
                    prev_wrist_pos = wrist_pos.copy()
                    prev_hand_pos = hand_pos.copy()

            # ----------------------------------------------------
            # Main status text
            # ----------------------------------------------------

            frame = draw_main_status(
                frame,
                BODY_MODEL,
                ARM_TO_TRACK,
                collecting_reference,
                reference_collected,
            )

            # ----------------------------------------------------
            # Optional video recording
            # ----------------------------------------------------

            if ENABLE_VIDEO_RECORDING and reference_collected:
                if video_writer is None:
                    frame_height, frame_width = frame.shape[:2]

                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                    video_writer = cv2.VideoWriter(
                        str(VIDEO_OUTPUT),
                        fourcc,
                        VIDEO_FPS,
                        (frame_width, frame_height),
                    )

                    print(f"[INFO] Video recording started: {VIDEO_OUTPUT}")

                video_writer.write(frame)

            # ----------------------------------------------------
            # Show frame
            # ----------------------------------------------------

            cv2.imshow(
                window_name,
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if cv2.getWindowProperty(
                window_name,
                cv2.WND_PROP_VISIBLE
            ) < 1:
                break

            if key in [10, 13]:
                if arm_data is None:
                    print(
                        f"[WARNING] No valid {ARM_TO_TRACK} arm detected. "
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

                save_json(
                    reference_data,
                    records
                )

                print(
                    f"[INFO] Collecting {NUM_REFERENCE_FRAMES} "
                    "reference frames..."
                )

    finally:
        if video_writer is not None:
            video_writer.release()

        zed.close()
        cv2.destroyAllWindows()

        print("[INFO] Program closed.")


if __name__ == "__main__":
    main()