#!/usr/bin/env python3

import os
import time
import json
import cv2
import pyzed.sl as sl


OUTPUT_DIR = "/home/agbara-admin/Documents/Cleaning_Robot/data/arm_tracking"
os.makedirs(OUTPUT_DIR, exist_ok=True)

timestamp_name = time.strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    f"right_arm_positions_body34.txt"
)


RIGHT_ARM_KEYPOINTS = {
    11: "RIGHT_CLAVICLE",
    12: "RIGHT_SHOULDER",
    13: "RIGHT_ELBOW",
    14: "RIGHT_WRIST",
    15: "RIGHT_HAND",
    16: "RIGHT_HANDTIP",
    17: "RIGHT_THUMB",
}

RIGHT_ARM_BONES = [
    (11, 12),  # clavicle -> shoulder
    (12, 13),  # shoulder -> elbow
    (13, 14),  # elbow -> wrist
    (14, 15),  # wrist -> hand
    (15, 16),  # hand -> handtip
    (14, 17),  # wrist -> thumb
]


def make_body_tracking_params():
    params = sl.BodyTrackingParameters()
    params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    params.body_format = sl.BODY_FORMAT.BODY_34
    params.enable_tracking = True
    params.enable_body_fitting = True
    return params


def draw_right_arm(img, body, confidence_threshold=40):
    """
    Draws BODY_34 right arm keypoints and bones on the image.
    Uses 2D keypoints for visualization.
    """

    keypoints_2d = body.keypoint_2d
    confidences = body.keypoint_confidence

    # Draw bones first
    for idx1, idx2 in RIGHT_ARM_BONES:
        if confidences[idx1] < confidence_threshold:
            continue
        if confidences[idx2] < confidence_threshold:
            continue

        p1 = keypoints_2d[idx1]
        p2 = keypoints_2d[idx2]

        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 3)

    # Draw joints
    for idx, name in RIGHT_ARM_KEYPOINTS.items():
        if confidences[idx] < confidence_threshold:
            continue

        p = keypoints_2d[idx]
        x, y = int(p[0]), int(p[1])

        cv2.circle(img, (x, y), 7, (0, 255, 0), -1)
        cv2.circle(img, (x, y), 10, (0, 0, 0), 2)

        cv2.putText(
            img,
            name.replace("RIGHT_", ""),
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
        )


def main():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"[ERROR] Could not open ZED camera: {status}")
        return

    positional_tracking_params = sl.PositionalTrackingParameters()
    status = zed.enable_positional_tracking(positional_tracking_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(f"[ERROR] Could not enable positional tracking: {status}")
        zed.close()
        return

    body_tracking_params = make_body_tracking_params()
    status = zed.enable_body_tracking(body_tracking_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(f"[ERROR] Could not enable body tracking: {status}")
        zed.disable_positional_tracking()
        zed.close()
        return

    runtime_params = sl.RuntimeParameters()

    body_runtime_params = sl.BodyTrackingRuntimeParameters()
    body_runtime_params.detection_confidence_threshold = 40

    bodies = sl.Bodies()
    image = sl.Mat()

    print("[INFO] Tracking right arm using BODY_34")
    print(f"[INFO] Writing data to: {OUTPUT_FILE}")
    print("[INFO] Press q to stop")

    with open(OUTPUT_FILE, "w") as f:
        f.write("# Right arm BODY_34 tracking output\n")
        f.write("# Units: meters\n")
        f.write("# One JSON object per timestep\n")

        frame_id = 0

        while True:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                frame_id += 1

                zed.retrieve_image(image, sl.VIEW.LEFT)
                zed.retrieve_bodies(bodies, body_runtime_params)

                timestamp = zed.get_timestamp(
                    sl.TIME_REFERENCE.IMAGE
                ).get_seconds()

                frame_data = {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "num_bodies": len(bodies.body_list),
                    "right_arm": {}
                }

                img_np = image.get_data().copy()

                if len(bodies.body_list) > 0:
                    body = bodies.body_list[0]

                    keypoints_3d = body.keypoint
                    confidences = body.keypoint_confidence

                    for idx, name in RIGHT_ARM_KEYPOINTS.items():
                        p = keypoints_3d[idx]
                        conf = confidences[idx]

                        frame_data["right_arm"][name] = {
                            "index": idx,
                            "x": float(p[0]),
                            "y": float(p[1]),
                            "z": float(p[2]),
                            "confidence": float(conf)
                        }

                    draw_right_arm(
                        img_np,
                        body,
                        confidence_threshold=40
                    )

                    cv2.putText(
                        img_np,
                        f"Tracking body ID: {body.id}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                else:
                    cv2.putText(
                        img_np,
                        "No body detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )

                f.write(json.dumps(frame_data) + "\n")
                f.flush()

                cv2.imshow("ZED BODY_34 Right Arm Tracking", img_np)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cv2.destroyAllWindows()
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()

    print("[INFO] Tracking stopped")
    print(f"[INFO] Data saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()