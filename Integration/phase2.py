#!/usr/bin/env python3
"""Record and replay human cleaning-head motion relative to a fixed tray.

The tray detector and plane fitter are used only during the initial tray
calibration.  Once a valid plane and centroid have been frozen, recording uses
only the human-tool ChArUco marker detector.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "data" / "phase2"
WINDOW_NAME = "Phase 2 human cleaning recording"


def cleaning_head_measurement(
    T_camera_cleaning_head: np.ndarray,
    tray_centroid_camera_m: np.ndarray,
    tray_plane_coefficients: np.ndarray,
) -> dict:
    """Convert one accepted cleaning-head pose into tray coordinates.

    Marker detection, marker-to-tool conversion, quality rejection, and pose
    filtering all belong to ``human.py``. This function deliberately starts
    with the resulting cleaning-head pose and only performs the Phase 2 task:
    expressing its position relative to the frozen tray.
    """
    T_camera_cleaning_head = np.asarray(
        T_camera_cleaning_head, dtype=float
    ).reshape(4, 4)
    position = T_camera_cleaning_head[:3, 3]
    centroid = np.asarray(tray_centroid_camera_m, dtype=float).reshape(3)
    relative = position - centroid

    plane = _normalized_plane(tray_plane_coefficients)
    T_tray_camera = tray_frame_from_plane(centroid, plane)
    T_tray_cleaning_head = T_tray_camera @ T_camera_cleaning_head
    position_tray = T_tray_cleaning_head[:3, 3]

    return {
        "cleaning_head_position_camera_m": position.tolist(),
        "cleaning_head_position_tray_m": position_tray.tolist(),
        "distance_to_centroid_m": float(np.linalg.norm(relative)),
    }


def track_cleaning_device(image, timestamp_s, board, detector, camera_matrix,
                          dist_coeffs, quality_gate, pose_filter):
    """Return one reliable, filtered cleaning-head pose from ``human.py``.

    The human-device code owns the complete marker measurement policy:

    1. Detect the ChArUco marker and estimate its raw pose.
    2. Reject observations with missing corners, excessive reprojection error,
       physically implausible jumps, or unstable reacquisition after a gap.
    3. Convert the accepted marker pose to the cleaning-head pose using the
       calibrated marker-to-cleaning-head transform.
    4. Smooth translation and rotation with the SE(3)-safe One Euro filter.

    Returns ``(None, reason)`` for a rejected frame. Rejected poses are never
    displayed as measurements and never written to the recording.
    """
    from human import _estimate_camera_marker

    detection, raw_pose = _estimate_camera_marker(
        image, board, detector, camera_matrix, dist_coeffs
    )
    quality = quality_gate.evaluate(detection, raw_pose, timestamp_s)
    if not quality.accepted:
        return None, quality.reason

    # A confirmed reacquisition starts a separate continuous segment. Reset
    # the filter so it does not blend the new segment with a stale old pose.
    if quality.starts_new_segment:
        pose_filter.reset()
    filtered_pose = pose_filter.update(
        quality.T_camera_cleaning_head, timestamp_s
    )
    return filtered_pose, quality.reason


def _normalized_plane(coefficients):
    plane = np.asarray(coefficients, dtype=float).reshape(4)
    normal_length = np.linalg.norm(plane[:3])
    if normal_length <= np.finfo(float).eps:
        raise ValueError("Tray plane normal cannot be zero")
    return plane / normal_length


def tray_frame_from_plane(centroid_camera_m, plane_coefficients):
    """Return T_tray_camera with +Z directed from the tray to the camera.

    Tray X is the camera +X direction projected onto the tray. Tray Y completes
    a right-handed frame, and the tray centroid is the origin.
    """
    centroid = np.asarray(centroid_camera_m, dtype=float).reshape(3)
    normal = _normalized_plane(plane_coefficients)[:3]

    # camera.calculate_tray_plane() normally already points the normal toward
    # the camera. Enforce that convention here as well for imported data.
    if float(normal @ centroid) > 0.0:
        normal = -normal
    camera_x = np.array([1.0, 0.0, 0.0])
    tray_x = camera_x - float(camera_x @ normal) * normal
    if np.linalg.norm(tray_x) < 1e-8:
        camera_x = np.array([0.0, 1.0, 0.0])
        tray_x = camera_x - float(camera_x @ normal) * normal
    tray_x /= np.linalg.norm(tray_x)
    tray_y = np.cross(normal, tray_x)
    tray_y /= np.linalg.norm(tray_y)

    rotation_tray_camera = np.vstack([tray_x, tray_y, normal])
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation_tray_camera
    transform[:3, 3] = -rotation_tray_camera @ centroid
    return transform


def _project(camera_matrix, point_camera_m):
    point = np.asarray(point_camera_m, dtype=float).reshape(3)
    if point[2] <= 0.0 or not np.all(np.isfinite(point)):
        return None
    pixel = np.asarray(camera_matrix, dtype=float).reshape(3, 3) @ point
    return tuple(np.rint(pixel[:2] / pixel[2]).astype(int))


def _plane_basis(normal):
    """Return two orthonormal vectors lying in a plane."""
    normal = np.asarray(normal, dtype=float).reshape(3)
    normal /= np.linalg.norm(normal)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(normal @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    first = np.cross(normal, reference)
    first /= np.linalg.norm(first)
    return first, np.cross(normal, first)


def _draw_frozen_plane(display, camera_matrix, centroid, plane, tray_mask):
    """Overlay the frozen tray area, plane grid, centroid, and normal."""
    overlay = np.zeros_like(display)
    overlay[np.asarray(tray_mask, dtype=bool)] = (90, 60, 0)
    display = cv2.addWeighted(display, 1.0, overlay, 0.25, 0.0)

    normal = np.asarray(plane, dtype=float).reshape(4)[:3]
    normal /= np.linalg.norm(normal)
    first, second = _plane_basis(normal)
    extent_m = 0.20
    values = np.linspace(-extent_m, extent_m, 5)
    for value in values:
        for direction, offset_direction in ((first, second), (second, first)):
            endpoints = [
                centroid + value * offset_direction - extent_m * direction,
                centroid + value * offset_direction + extent_m * direction,
            ]
            pixels = [_project(camera_matrix, point) for point in endpoints]
            if all(pixel is not None for pixel in pixels):
                cv2.line(display, pixels[0], pixels[1], (255, 180, 0), 1,
                         cv2.LINE_AA)
    center_pixel = _project(camera_matrix, centroid)
    normal_pixel = _project(camera_matrix, centroid + 0.10 * normal)
    if center_pixel is not None and normal_pixel is not None:
        cv2.arrowedLine(display, center_pixel, normal_pixel, (255, 0, 255),
                        3, cv2.LINE_AA, tipLength=0.2)
    return display


def _draw_tracking(image, camera_matrix, centroid, plane, tray_mask, samples,
                   measurement, tracking_status):
    """Draw the frozen tray and the latest accepted cleaning-head position."""
    display = image.copy()
    display = _draw_frozen_plane(
        display, camera_matrix, centroid, plane, tray_mask
    )
    centroid_pixel = _project(camera_matrix, centroid)
    if centroid_pixel is not None:
        cv2.drawMarker(
            display, centroid_pixel, (0, 255, 255), cv2.MARKER_CROSS, 24, 3
        )

    trail = []
    for sample in samples[-300:]:
        pixel = _project(
            camera_matrix, sample["cleaning_head_position_camera_m"]
        )
        if pixel is not None:
            trail.append(pixel)
    if len(trail) > 1:
        cv2.polylines(
            display,
            [np.asarray(trail, dtype=np.int32).reshape(-1, 1, 2)],
            False,
            (255, 100, 0),
            2,
            cv2.LINE_AA,
        )

    if measurement is None:
        text = f"Pose rejected: {tracking_status}"
        color = (0, 0, 255)
    else:
        head_pixel = _project(
            camera_matrix, measurement["cleaning_head_position_camera_m"]
        )
        if head_pixel is not None:
            cv2.circle(display, head_pixel, 8, (0, 255, 0), -1, cv2.LINE_AA)
            if centroid_pixel is not None:
                cv2.line(display, centroid_pixel, head_pixel, (255, 255, 0), 2)
        tray_position_mm = 1000.0 * np.asarray(
            measurement["cleaning_head_position_tray_m"]
        )
        text = (
            f"tray XYZ: {tray_position_mm[0]:+.1f}, "
            f"{tray_position_mm[1]:+.1f}, {tray_position_mm[2]:+.1f} mm | norm: "
            f"{1000.0 * measurement['distance_to_centroid_m']:.1f} mm"
        )
        color = (0, 255, 0)
    cv2.putText(display, text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, color, 2, cv2.LINE_AA)
    cv2.putText(display, "Q/Esc: save and stop", (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                cv2.LINE_AA)
    return display


def _freeze_tray(zed, runtime_params, image_zed, point_cloud_zed,
                 camera_matrix, predictor):
    """Run tray detection until the operator accepts one valid result."""
    import pyzed.sl as sl
    from camera import get_point_cloud, grab_frame, process_tray

    print("[INFO] Detecting tray. Press SPACE to freeze a valid result; Q to stop.")
    while True:
        if not grab_frame(zed, runtime_params):
            continue
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        bgra = image_zed.get_data()
        if bgra is None or bgra.size == 0:
            continue
        image = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        xyz, valid_mask = get_point_cloud(zed, point_cloud_zed)
        result = None if xyz is None else process_tray(
            image, xyz, camera_matrix, predictor, valid_mask
        )

        display = image.copy()
        valid = result is not None and result["plane"] is not None \
            and result["centroid"] is not None
        if result is not None:
            mask = result["detection"]["mask"]
            overlay = np.zeros_like(display)
            overlay[mask] = (0, 120, 255)
            display = cv2.addWeighted(display, 1.0, overlay, 0.35, 0.0)
        message = (
            "CONFIRM 1/2 - SPACE: freeze tray"
            if valid else "Searching for tray plane..."
        )
        cv2.putText(display, message, (20, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0) if valid else (0, 0, 255), 2,
                    cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKeyEx(1) & 0xFF
        if key in (ord("q"), 27):
            raise KeyboardInterrupt
        if key == ord(" ") and valid:
            return result


def _confirm_collection_start(zed, runtime_params, image_zed, camera_matrix,
                              centroid, plane, tray_mask, get_image):
    """Show the frozen tray until the operator explicitly starts recording."""
    print("[CONFIRM 1/2] Tray frozen.")
    print("[CONFIRM 2/2] Press SPACE to start data collection; Q to cancel.")
    while True:
        image = get_image(zed, runtime_params, image_zed)
        if image is None:
            continue
        display = _draw_frozen_plane(
            image.copy(), camera_matrix, centroid, plane, tray_mask
        )
        cv2.putText(display, "TRAY FROZEN", (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(display, "SPACE: start data collection | Q: cancel",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKeyEx(1) & 0xFF
        if key in (ord("q"), 27):
            raise KeyboardInterrupt
        if key == ord(" "):
            print("[INFO] Data collection started.")
            return


def _default_output():
    """Return the next simple sequential recording filename."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in DATA_DIR.glob("recording_*.json"):
        match = re.fullmatch(r"recording_(\d+)\.json", path.name)
        if match:
            numbers.append(int(match.group(1)))
    return DATA_DIR / f"recording_{max(numbers, default=0) + 1:03d}.json"


def _latest_recording():
    recordings = list(DATA_DIR.glob("recording_*.json"))
    if not recordings:
        raise FileNotFoundError(
            f"No recordings found in {DATA_DIR}. Run 'phase2.py record' first."
        )
    return max(recordings, key=lambda path: path.stat().st_mtime)


def record(args):
    """Calibrate the fixed tray once, then record human-tool motion."""
    import pyzed.sl as sl
    from calibration import HUMAN_TOOL_CHARUCO_CONFIG, create_charuco_detector
    from camera import (build_tray_predictor, get_image,
                        get_zed_left_intrinsics_rectified, open_zed,
                        save_tray_data)
    from human import (PoseOneEuroFilter, PoseQualityGate,
                       get_T_marker_cleaning_head)

    output = (args.output or _default_output()).expanduser().resolve()
    zed = None
    samples = []
    try:
        predictor = build_tray_predictor(args.tray_score, args.device)
        zed, runtime_params, image_zed = open_zed()
        point_cloud_zed = sl.Mat()
        camera_matrix, dist_coeffs = get_zed_left_intrinsics_rectified(zed)
        board, detector = create_charuco_detector(
            HUMAN_TOOL_CHARUCO_CONFIG, camera_matrix, dist_coeffs
        )
        # human.py owns marker validation and filtering. The quality gate also
        # performs the calibrated marker-to-cleaning-head conversion.
        T_marker_head = get_T_marker_cleaning_head()
        quality_gate = PoseQualityGate(T_marker_head)
        pose_filter = PoseOneEuroFilter()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        tray = _freeze_tray(
            zed, runtime_params, image_zed, point_cloud_zed,
            camera_matrix, predictor
        )
        plane = np.asarray(tray["plane"]["coefficients"], dtype=float)
        centroid = np.asarray(tray["centroid"], dtype=float)
        frozen_tray_mask = np.asarray(
            tray["detection"]["mask"], dtype=bool
        ).copy()
        save_tray_data(tray["plane"], centroid)

        # The heavy detector and point-cloud objects are deliberately dropped
        # immediately after the tray is frozen.
        del predictor, point_cloud_zed, tray
        print("[INFO] Tray frozen; object and plane detection are now OFF.")
        _confirm_collection_start(
            zed, runtime_params, image_zed, camera_matrix, centroid, plane,
            frozen_tray_mask, get_image
        )

        print("[INFO] Recording marker motion. Press Q or Escape to save.")
        start = time.monotonic()
        while True:
            image = get_image(zed, runtime_params, image_zed)
            if image is None:
                continue
            elapsed = time.monotonic() - start
            T_camera_cleaning_head, tracking_status = track_cleaning_device(
                image, elapsed, board, detector, camera_matrix, dist_coeffs,
                quality_gate, pose_filter
            )
            measurement = None
            if T_camera_cleaning_head is not None:
                measurement = cleaning_head_measurement(
                    T_camera_cleaning_head, centroid, plane
                )
                measurement["time_s"] = float(elapsed)
                samples.append(measurement)
            display = _draw_tracking(
                image, camera_matrix, centroid, plane, frozen_tray_mask,
                samples, measurement, tracking_status
            )
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKeyEx(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        if zed is not None:
            zed.close()
        cv2.destroyAllWindows()

    # Plot/replay only need time and tray-frame XYZ. Keep the fixed plane and
    # centroid because they define how the tray coordinate frame was created.
    payload = {
        "tray_plane_camera": plane.tolist(),
        "tray_centroid_camera_m": centroid.tolist(),
        "samples": [
            {
                "time_s": sample["time_s"],
                "tray_xyz_m": sample["cleaning_head_position_tray_m"],
            }
            for sample in samples
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"[INFO] Saved {len(samples)} tracked samples to {output}")
    return output


def _load_recording(path):
    if path is None:
        path = _latest_recording()
    elif str(path).isdigit():
        recording_number = int(str(path))
        if recording_number < 1:
            raise ValueError("Recording number must be 1 or greater")
        path = DATA_DIR / f"recording_{recording_number:03d}.json"
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Recording not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("samples"):
        raise ValueError(f"Recording contains no tracked samples: {path}")
    return path, payload


def plot_recording(path, show=True):
    """Plot tray-frame XYZ, centroid distance, and the 3D path."""
    import matplotlib.pyplot as plt

    path, data = _load_recording(path)
    samples = data["samples"]
    times = np.asarray([sample["time_s"] for sample in samples])
    relative = 1000.0 * _tray_positions(data)
    distance = np.linalg.norm(relative, axis=1)
    plane_normal = np.array([0.0, 0.0, 1.0])

    figure = plt.figure(figsize=(12, 7))
    trajectory_axis = figure.add_subplot(121, projection="3d")
    time_axis = figure.add_subplot(222)
    distance_axis = figure.add_subplot(224)
    trajectory_axis.plot(relative[:, 0], relative[:, 1], relative[:, 2])
    trajectory_axis.scatter(0, 0, 0, c="gold", marker="x", s=90,
                            label="tray centroid")
    _plot_plane(trajectory_axis, plane_normal, relative)
    trajectory_axis.set(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)",
                        title="Cleaning head in tray coordinates")
    trajectory_axis.legend()
    for index, label in enumerate(("X", "Y", "Z")):
        time_axis.plot(times, relative[:, index], label=label)
    time_axis.set(xlabel="Time (s)", ylabel="Relative position (mm)")
    time_axis.grid(True, alpha=0.3)
    time_axis.legend()
    distance_axis.plot(times, distance, color="black")
    distance_axis.set(xlabel="Time (s)", ylabel="Centroid distance (mm)")
    distance_axis.grid(True, alpha=0.3)
    figure.suptitle(path.name)
    figure.tight_layout()
    if show:
        plt.show()
    return figure


def _tray_positions(data):
    """Return cleaning-head positions from the current recording format."""
    return np.asarray([
        sample["tray_xyz_m"] for sample in data["samples"]
    ], dtype=float)


def _plot_plane(axis, normal, points):
    """Draw a translucent tray plane through the relative-frame origin."""
    first, second = _plane_basis(normal)
    span = max(100.0, float(np.max(np.linalg.norm(points, axis=1))))
    coordinates = np.linspace(-span, span, 9)
    uu, vv = np.meshgrid(coordinates, coordinates)
    surface = uu[..., None] * first + vv[..., None] * second
    axis.plot_surface(surface[:, :, 0], surface[:, :, 1], surface[:, :, 2],
                      color="orange", alpha=0.25, linewidth=0)
    axis.quiver(0, 0, 0, *(normal * span * 0.5), color="magenta",
                label="tray normal")


def replay(path, speed=1.0, save=False):
    """Animate a recording, optionally saving it as an MP4 video."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    path, data = _load_recording(path)
    samples = data["samples"]
    times = np.asarray([sample["time_s"] for sample in samples])
    points = 1000.0 * _tray_positions(data)
    plane_normal = np.array([0.0, 0.0, 1.0])
    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(111, projection="3d")
    bounds_points = np.vstack([points, np.zeros(3)])
    padding = np.maximum(np.ptp(bounds_points, axis=0) * 0.08, 5.0)
    axis.set_xlim(bounds_points[:, 0].min() - padding[0],
                  bounds_points[:, 0].max() + padding[0])
    axis.set_ylim(bounds_points[:, 1].min() - padding[1],
                  bounds_points[:, 1].max() + padding[1])
    axis.set_zlim(bounds_points[:, 2].min() - padding[2],
                  bounds_points[:, 2].max() + padding[2])
    axis.set(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
    axis.scatter(0, 0, 0, c="gold", marker="x", s=100)
    _plot_plane(axis, plane_normal, points)
    line, = axis.plot([], [], [], color="tab:blue")
    head, = axis.plot([], [], [], "o", color="tab:red")

    def update(index):
        line.set_data(points[:index + 1, 0], points[:index + 1, 1])
        line.set_3d_properties(points[:index + 1, 2])
        head.set_data([points[index, 0]], [points[index, 1]])
        head.set_3d_properties([points[index, 2]])
        axis.set_title(f"{path.name} | t={times[index]:.2f} s")
        return line, head

    intervals = np.diff(times, append=times[-1] + 1 / 30.0)
    interval_ms = max(1.0, 1000.0 * float(np.median(intervals)) / speed)
    animation = FuncAnimation(figure, update, frames=len(points),
                              interval=interval_ms, repeat=False, blit=False)
    # Keep a reference alive until the blocking window closes.
    figure._phase2_animation = animation
    if save:
        match = re.fullmatch(r"recording_(\d+)", path.stem)
        video_name = (
            f"replay_{int(match.group(1)):03d}.mp4"
            if match else f"{path.stem}_replay.mp4"
        )
        video_path = DATA_DIR / video_name
        frames_per_second = 1000.0 / interval_ms
        print(f"[INFO] Saving replay video to {video_path} ...")
        try:
            animation.save(
                video_path,
                writer="ffmpeg",
                fps=frames_per_second,
                dpi=150,
            )
        except (FileNotFoundError, RuntimeError) as error:
            raise RuntimeError(
                "Could not save MP4. Install FFmpeg and verify that "
                "'ffmpeg -version' works."
            ) from error
        finally:
            plt.close(figure)
        print(f"[INFO] Replay video saved: {video_path.resolve()}")
        return video_path.resolve()
    plt.show()
    return None


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record_parser = commands.add_parser("record", help="calibrate tray and record")
    record_parser.add_argument("--output", type=Path)
    record_parser.add_argument("--tray-score", type=float, default=0.95)
    record_parser.add_argument("--device", choices=("cpu", "cuda"))
    plot_parser = commands.add_parser("plot", help="plot a saved recording")
    plot_parser.add_argument(
        "recording", nargs="?",
        help="recording number or JSON path (default: latest recording)"
    )
    replay_parser = commands.add_parser("replay", help="animate a saved recording")
    replay_parser.add_argument(
        "recording", nargs="?",
        help="recording number or JSON path (default: latest recording)"
    )
    replay_parser.add_argument("--speed", type=float, default=1.0)
    replay_parser.add_argument(
        "--save", action="store_true",
        help="save the replay as data/phase2/replay_###.mp4"
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "record":
        record(args)
    elif args.command == "plot":
        plot_recording(args.recording)
    elif args.command == "replay":
        if not np.isfinite(args.speed) or args.speed <= 0.0:
            raise SystemExit("--speed must be positive")
        replay(args.recording, args.speed, args.save)


if __name__ == "__main__":
    main()
