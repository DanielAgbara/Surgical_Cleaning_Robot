#!/usr/bin/env python3

import os
import time
import json
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -------------------------------------------------
# Output Configuration
# -------------------------------------------------

OUTPUT_DIR = "/home/agbara-admin/Documents/Cleaning_Robot/data/arm_tracking"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "right_arm_positions_mediapipe_tasks.txt"
)

MODEL_PATH = "models/pose_landmarker.task"


# -------------------------------------------------
# MediaPipe Pose Landmark Indices
# -------------------------------------------------
# MediaPipe Pose has 33 landmarks.
# Right arm indices:
# 12 = right shoulder
# 14 = right elbow
# 16 = right wrist
# 18 = right pinky
# 20 = right index
# 22 = right thumb

RIGHT_ARM_KEYPOINTS = {
    12: "RIGHT_SHOULDER",
    14: "RIGHT_ELBOW",
    16: "RIGHT_WRIST",
    18: "RIGHT_PINKY",
    20: "RIGHT_INDEX",
    22: "RIGHT_THUMB",
}

RIGHT_ARM_BONES = [
    (12, 14),  # shoulder -> elbow
    (14, 16),  # elbow -> wrist
    (16, 18),  # wrist -> pinky
    (16, 20),  # wrist -> index
    (16, 22),  # wrist -> thumb
]


# -------------------------------------------------
# Visualization
# -------------------------------------------------

def draw_right_arm(img, landmarks, confidence_threshold=0.5):
    """
    Draw right arm landmarks and bones.

    landmarks are MediaPipe normalized image landmarks.
    x and y are normalized from 0 to 1.
    """

    h, w, _ = img.shape

    # Draw bones first
    for idx1, idx2 in RIGHT_ARM_BONES:
        lm1 = landmarks[idx1]
        lm2 = landmarks[idx2]

        if lm1.visibility < confidence_threshold:
            continue

        if lm2.visibility < confidence_threshold:
            continue

        x1 = int(lm1.x * w)
        y1 = int(lm1.y * h)

        x2 = int(lm2.x * w)
        y2 = int(lm2.y * h)

        cv2.line(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            3
        )

    # Draw joints
    for idx, name in RIGHT_ARM_KEYPOINTS.items():
        lm = landmarks[idx]

        if lm.visibility < confidence_threshold:
            continue

        x = int(lm.x * w)
        y = int(lm.y * h)

        cv2.circle(img, (x, y), 7, (0, 255, 0), -1)
        cv2.circle(img, (x, y), 10, (0, 0, 0), 2)

        cv2.putText(
            img,
            name.replace("RIGHT_", ""),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2
        )


# -------------------------------------------------
# MediaPipe Pose Landmarker
# -------------------------------------------------

def create_pose_landmarker(model_path):
    """
    Creates MediaPipe PoseLandmarker using the new Tasks API.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Download it with:\n\n"
            "mkdir -p models\n"
            "wget -O models/pose_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/"
            "pose_landmarker/pose_landmarker_full/float16/latest/"
            "pose_landmarker_full.task\n"
        )

    base_options = python.BaseOptions(
        model_asset_path=model_path
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    return vision.PoseLandmarker.create_from_options(options)


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open camera")
        return

    # Optional camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    try:
        landmarker = create_pose_landmarker(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Could not create PoseLandmarker: {e}")
        cap.release()
        return

    print("[INFO] Tracking right arm using MediaPipe PoseLandmarker Tasks API")
    print(f"[INFO] Writing data to: {OUTPUT_FILE}")
    print("[INFO] Press q to stop")

    with open(OUTPUT_FILE, "w") as f:
        f.write("# Right arm MediaPipe PoseLandmarker tracking output\n")
        f.write("# Compatible with mediapipe 0.10.35 Tasks API\n")
        f.write("# image_landmarks: normalized image coordinates\n")
        f.write("# world_landmarks: approximate 3D body/world landmarks from MediaPipe\n")
        f.write("# One JSON object per timestep\n")

        frame_id = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()

            if not ret:
                print("[WARNING] Could not read camera frame")
                break

            frame_id += 1

            current_time = time.time()
            timestamp = current_time
            timestamp_ms = int((current_time - start_time) * 1000)

            # OpenCV gives BGR.
            # MediaPipe expects RGB.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = rgb_frame.copy()

            mp_image = mp.Image(
                mp.ImageFormat.SRGB,
                rgb_frame
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            frame_data = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "timestamp_ms": timestamp_ms,
                "pose_detected": False,
                "num_poses": 0,
                "right_arm_image": {},
                "right_arm_world": {}
            }

            if result.pose_landmarks:
                frame_data["pose_detected"] = True
                frame_data["num_poses"] = len(result.pose_landmarks)

                # Since num_poses=1, use first detected body
                image_landmarks = result.pose_landmarks[0]

                for idx, name in RIGHT_ARM_KEYPOINTS.items():
                    lm = image_landmarks[idx]

                    frame_data["right_arm_image"][name] = {
                        "index": idx,
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                        "visibility": float(lm.visibility),
                        "presence": float(lm.presence),
                    }

                # Optional approximate 3D landmarks
                if result.pose_world_landmarks:
                    world_landmarks = result.pose_world_landmarks[0]

                    for idx, name in RIGHT_ARM_KEYPOINTS.items():
                        lm = world_landmarks[idx]

                        frame_data["right_arm_world"][name] = {
                            "index": idx,
                            "x": float(lm.x),
                            "y": float(lm.y),
                            "z": float(lm.z),
                            "visibility": float(lm.visibility),
                            "presence": float(lm.presence),
                        }

                draw_right_arm(
                    frame,
                    image_landmarks,
                    confidence_threshold=0.5
                )

                cv2.putText(
                    frame,
                    "MediaPipe right arm tracking",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            else:
                cv2.putText(
                    frame,
                    "No pose detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            f.write(json.dumps(frame_data) + "\n")
            f.flush()

            cv2.imshow("MediaPipe Tasks Right Arm Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()

    print("[INFO] Tracking stopped")
    print(f"[INFO] Data saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()