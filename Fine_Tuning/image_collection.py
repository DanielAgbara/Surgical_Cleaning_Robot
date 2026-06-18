#!/usr/bin/env python3

import cv2
import time
import json
from pathlib import Path
import pyzed.sl as sl


# --------------------------------------------------
# Dataset settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "data" / "dataset" / "tray_raw"
IMAGE_DIR = DATASET_DIR / "images"
METADATA_FILE = DATASET_DIR / "metadata.json"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAME = "tray"

CAMERA_RESOLUTION = sl.RESOLUTION.HD720
CAMERA_FPS = 30

WINDOW_NAME = "Tray Dataset Collector"


def ask_num_images():
    """
    Ask user how many NEW images to collect.

    Example:

        Existing images: 400
        User enters: 200

    Final dataset size becomes:

        400 + 200 = 600 images
    """

    while True:
        try:
            num_images = int(
                input("How many NEW images do you want to collect? ")
            )

            if num_images > 0:
                return num_images

            print("[ERROR] Please enter a positive number.")

        except ValueError:
            print("[ERROR] Please enter a valid integer.")

def open_zed():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = CAMERA_RESOLUTION
    init_params.camera_fps = CAMERA_FPS
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.coordinate_units = sl.UNIT.METER

    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Could not open ZED camera: {status}")

    return zed


def save_metadata(new_images_requested, current_count):
    """
    Update metadata without deleting previous information.
    """

    metadata = {}

    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    metadata["class_name"] = CLASS_NAME
    metadata["camera"] = "ZED"
    metadata["resolution"] = str(CAMERA_RESOLUTION)
    metadata["fps"] = CAMERA_FPS

    metadata["current_image_count"] = current_count

    metadata["last_collection_request"] = new_images_requested

    metadata["notes"] = (
        "Raw images for instance segmentation. "
        "Dataset grows incrementally."
    )

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[INFO] Metadata updated: {METADATA_FILE}")

def main():
    saved_count = len(list(IMAGE_DIR.glob("*.png")))

    new_images_to_collect = ask_num_images()

    target_num_images = saved_count + new_images_to_collect

    save_metadata(
        new_images_requested=new_images_to_collect,
        current_count=saved_count,
    )

    zed = open_zed()
    image = sl.Mat()

    saved_count = len(list(IMAGE_DIR.glob("*.png")))

    print("[INFO] ZED camera opened.")
    print("[INFO] Controls:")
    print("  s = save current image")
    print("  q = quit")
    print(f"[INFO] Starting from image count: {saved_count}")
    print(f"[INFO] Target number of images: {target_num_images}")

    try:
        while saved_count < target_num_images:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)

                frame = image.get_data()

                # ZED image is BGRA. Convert to BGR for OpenCV.
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                display = frame_bgr.copy()

                cv2.putText(
                    display,
                    f"Class: {CLASS_NAME}",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    display,
                    f"Saved: {saved_count}/{target_num_images}",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    display,
                    "Press s = save | q = quit",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow(WINDOW_NAME, display)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("s"):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = IMAGE_DIR / f"tray_{saved_count:05d}_{timestamp}.png"

                    cv2.imwrite(str(filename), frame_bgr)
                    saved_count += 1

                    print(f"[SAVED] {filename}")

                elif key == ord("q"):
                    print("[INFO] Quit requested.")
                    break

        if saved_count >= target_num_images:
            print("[DONE] Target image count reached.")

    finally:
        zed.close()
        cv2.destroyAllWindows()
        print("[INFO] Camera closed.")
        print(f"[INFO] Total saved images: {saved_count}")
        print(f"[INFO] Images stored in: {IMAGE_DIR}")


if __name__ == "__main__":
    main()