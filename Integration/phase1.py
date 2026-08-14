"""Test tray localization or apply a force to a detected tray with a Lite 6."""

import argparse
import json
import time

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl
import robot

from camera import (
    build_tray_predictor,
    get_image,
    get_point_cloud,
    get_zed_left_intrinsics_rectified,
    open_zed,
    process_tray,
    save_tray_data,
)


WINDOW_NAME = "Tray camera test"
DEFAULT_ORIENTATION_X_MM = 200.0
DEFAULT_ORIENTATION_Y_MM = 0.9
DEFAULT_ORIENTATION_CLEARANCE_MM = 100.0
DEFAULT_CALIBRATION_FILE = (
    Path(__file__).resolve().parent
    / "data"
    / "eyehand"
    / "eye_hand_calibration.json"
)
TSAI_CALIBRATION_FILE = (
    Path(__file__).resolve().parent
    / "data"
    / "eyehand"
    / "eye_hand_calibration_tsai.json"
)
DEFAULT_FORCE_CALIBRATION_FILE = (
    Path(__file__).resolve().parent
    / "data"
    / "force_data"
    / "force_sensor_calibration.json"
)
DEFAULT_TOOL_RADIUS_MM = 50.0
MASK_MARGIN_REDUCTION_MM = 50.0

# Set this to True to record force videos by default. The CLI can override it
# for one run with --video or --no-video without editing this file.
ENABLE_FORCE_VIDEO = True
DEFAULT_VIDEO_FPS = 15.0
FORCE_VIDEO_DIR = Path(__file__).resolve().parent / "data" / "force_data"


class ForceTrajectoryVideoRecorder:
    """Write clean camera frames plus trajectory points and force history."""

    CHART_WIDTH = 600

    def __init__(self, path, pixels, frame_getter, target_force, cutoff, fps):
        self.path = Path(path).resolve()
        self.pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
        self.frame_getter = frame_getter
        self.target_force = float(target_force)
        self.cutoff = float(cutoff)
        self.fps = float(fps)
        if len(self.pixels) == 0:
            raise ValueError("Video requires at least one planned point")
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("Video FPS must be positive and finite")
        self.writer = None
        self.start_time = None
        self.times = []
        self.forces = []
        self.frames_written = 0
        self.last_frame = None

    def _camera_panel(self, image, active_index):
        """Draw only the planned path on an otherwise clean camera frame."""
        panel = np.asarray(image).copy()
        points = np.rint(self.pixels).astype(np.int32)
        if len(points) > 1:
            cv2.polylines(
                panel,
                [points.reshape(-1, 1, 2)],
                False,
                (255, 180, 0),
                2,
                cv2.LINE_AA,
            )
        for index, point in enumerate(points, start=1):
            center = (int(point[0]), int(point[1]))
            color = (0, 0, 255) if index == active_index else (0, 255, 0)
            cv2.circle(panel, center, 7, color, -1, cv2.LINE_AA)
            cv2.putText(
                panel,
                str(index),
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return panel

    def _force_panel(self, height, active_index, status):
        """Render the accumulated calibrated force-versus-time plot."""
        width = self.CHART_WIDTH
        panel = np.full((height, width, 3), 245, dtype=np.uint8)
        left, right, top, bottom = 75, width - 25, 70, height - 70
        cv2.rectangle(panel, (left, top), (right, bottom), (30, 30, 30), 2)
        elapsed = np.asarray(self.times, dtype=float)
        forces = np.asarray(self.forces, dtype=float)
        time_max = max(1.0, float(elapsed[-1]) if elapsed.size else 1.0)
        force_min = min(-2.0, float(forces.min()) if forces.size else 0.0)
        force_max = max(
            self.cutoff * 1.1,
            self.target_force + 2.0,
            float(forces.max()) * 1.1 if forces.size else 1.0,
        )

        def x_coordinate(value):
            return int(left + (right - left) * float(value) / time_max)

        def y_coordinate(value):
            fraction = (float(value) - force_min) / (force_max - force_min)
            return int(bottom - (bottom - top) * fraction)

        for value, color, label in (
            (self.target_force, (0, 160, 0), "target"),
            (self.cutoff, (0, 140, 255), "cutoff"),
        ):
            y_value = y_coordinate(value)
            cv2.line(panel, (left, y_value), (right, y_value), color, 2)
            cv2.putText(panel, f"{label} {value:.1f} N",
                        (left + 8, y_value - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        cv2.LINE_AA)
        if len(elapsed) > 1:
            curve = np.column_stack(
                [[x_coordinate(t) for t in elapsed],
                 [y_coordinate(f) for f in forces]]
            ).astype(np.int32)
            cv2.polylines(panel, [curve.reshape(-1, 1, 2)], False,
                          (200, 40, 40), 2, cv2.LINE_AA)
        elif len(elapsed) == 1:
            cv2.circle(panel, (x_coordinate(elapsed[0]),
                               y_coordinate(forces[0])), 4,
                       (200, 40, 40), -1)
        latest = float(forces[-1]) if forces.size else 0.0
        cv2.putText(panel, "Force vs time", (left, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2,
                    cv2.LINE_AA)
        cv2.putText(
            panel,
            f"Point {active_index}/{len(self.pixels)} | "
            f"{latest:.2f} N | {status}",
            (left, 57),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(panel, "Time (s)", (right - 75, bottom + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
        cv2.putText(panel, f"{time_max:.1f}", (right - 20, bottom + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)
        cv2.putText(panel, f"{force_max:.1f} N", (5, top + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)
        cv2.putText(panel, f"{force_min:.1f} N", (5, bottom),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1)
        return panel

    def record_force(self, force_newtons, active_index, status="contact"):
        """Grab one clean ZED frame and append one force/time measurement."""
        image = self.frame_getter()
        if image is None:
            print("[WARNING] Force video skipped a failed camera grab.")
            return
        now = time.monotonic()
        if self.start_time is None:
            self.start_time = now
        elapsed = now - self.start_time
        self.times.append(elapsed)
        self.forces.append(float(force_newtons))
        camera_panel = self._camera_panel(image, active_index)
        chart_panel = self._force_panel(camera_panel.shape[0], active_index, status)
        composite = np.hstack([camera_panel, chart_panel])
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = cv2.VideoWriter(
                str(self.path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (composite.shape[1], composite.shape[0]),
            )
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = None
                raise RuntimeError(f"Could not open force video: {self.path}")
        target_prior_frames = int(np.floor(elapsed * self.fps))
        while self.last_frame is not None and self.frames_written < target_prior_frames:
            self.writer.write(self.last_frame)
            self.frames_written += 1
        self.writer.write(composite)
        self.frames_written += 1
        self.last_frame = composite

    def close(self):
        """Flush and close the MP4 file."""
        if self.writer is None:
            return
        for _ in range(max(1, int(round(self.fps * 0.5)))):
            self.writer.write(self.last_frame)
        self.writer.release()
        self.writer = None
        print(f"[INFO] Force video saved to: {self.path}")


def shrink_tray_mask_for_tool(
    tray_mask,
    camera_matrix,
    tray_plane,
    tool_radius_mm=DEFAULT_TOOL_RADIUS_MM,
    safety_margin_mm=0.0,
):
    """Shrink a tray mask by a physical tool radius and safety margin.

    The segmentation mask is measured in pixels, while the tool radius is a
    physical distance.  This function uses the fitted tray plane and camera
    intrinsics to estimate the closest tray depth, then converts the requested
    physical clearance to a conservative pixel clearance.  Using the closest
    depth is conservative because the same physical radius covers more pixels
    when it is nearer to the camera.

    The returned mask describes valid *tool-center* pixels: placing the tool
    center anywhere in this mask keeps its circular footprint inside the
    original detected tray mask, subject to the pinhole/pixel-scale
    approximation.

    Returns
    -------
    safe_mask : ndarray of bool
        Mask in which trajectory centers may be placed.
    clearance_pixels : int
        Effective physical clearance (radius + margin - 5 mm) in pixels.
    closest_tray_depth_m : float
        Closest valid plane-intersection depth used for the conversion.
    """
    mask = np.asarray(tray_mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError("tray_mask must be a nonempty 2D mask")

    camera_matrix = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(camera_matrix)):
        raise ValueError("camera_matrix must contain finite values")
    focal_x = float(camera_matrix[0, 0])
    focal_y = float(camera_matrix[1, 1])
    if focal_x <= 0.0 or focal_y <= 0.0:
        raise ValueError("camera focal lengths must be positive")

    tool_radius_mm = float(tool_radius_mm)
    safety_margin_mm = float(safety_margin_mm)
    if not np.isfinite(tool_radius_mm) or tool_radius_mm <= 0.0:
        raise ValueError("tool_radius_mm must be positive and finite")
    if not np.isfinite(safety_margin_mm) or safety_margin_mm < 0.0:
        raise ValueError("safety_margin_mm must be finite and nonnegative")

    if isinstance(tray_plane, dict):
        coefficients = tray_plane.get("coefficients")
    else:
        coefficients = tray_plane
    coefficients = np.asarray(coefficients, dtype=float).reshape(4)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("tray plane coefficients must be finite")
    normal_length = np.linalg.norm(coefficients[:3])
    if normal_length <= 1e-12:
        raise ValueError("tray plane normal cannot be zero")
    coefficients = coefficients / normal_length

    # Intersect every masked pixel ray with the plane. Only depth is needed
    # here; the full 3D points can be calculated later for the final path.
    rows, columns = np.nonzero(mask)
    homogeneous_pixels = np.vstack(
        [columns, rows, np.ones(columns.size, dtype=float)]
    )
    rays = np.linalg.solve(camera_matrix, homogeneous_pixels)
    denominators = coefficients[:3] @ rays
    valid = np.abs(denominators) > 1e-10
    distances = np.full(columns.size, np.nan, dtype=float)
    distances[valid] = -coefficients[3] / denominators[valid]
    depths = distances * rays[2]
    valid &= np.isfinite(depths) & (depths > 0.0)
    if not np.any(valid):
        raise ValueError("tray mask rays do not intersect the plane in front")
    closest_depth_m = float(np.min(depths[valid]))

    # Deliberately recover 5 mm from the nominal tool-edge clearance. For a
    # 50 mm tool radius and zero extra margin, the mask is therefore eroded by
    # 45 mm. Clamp at zero so small tools never create a negative erosion.
    effective_clearance_mm = max(
        0.0,
        tool_radius_mm + safety_margin_mm - MASK_MARGIN_REDUCTION_MM,
    )
    clearance_m = effective_clearance_mm / 1000.0
    clearance_pixels = int(
        np.ceil(clearance_m * max(focal_x, focal_y) / closest_depth_m)
    )

    # Padding makes the image boundary count as background. The distance
    # transform provides an efficient circular erosion even for a large tool,
    # which could otherwise require a very large morphology kernel.
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    distance_to_background = cv2.distanceTransform(
        padded,
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )[1:-1, 1:-1]
    safe_mask = distance_to_background > float(clearance_pixels)
    if not np.any(safe_mask):
        raise ValueError(
            "The tray has no safe center region after shrinking for a "
            f"{tool_radius_mm:.1f} mm tool radius and "
            f"{safety_margin_mm:.1f} mm margin minus the configured "
            f"{MASK_MARGIN_REDUCTION_MM:.1f} mm reduction "
            f"({clearance_pixels} pixels)."
        )
    return safe_mask, clearance_pixels, closest_depth_m


def generate_tray_trajectory_pixels(
    safe_mask,
    shape="square",
    discrete_points=True,
    number_of_points=12,
    continuous_step_pixels=2.0,
    path_scale=0.9,
):
    """Generate a square or circular image trajectory inside a safe mask.

    Parameters
    ----------
    safe_mask : ndarray of bool
        Output from :func:`shrink_tray_mask_for_tool`.
    shape : {"square", "circle"}
        Requested closed trajectory shape in image coordinates.
    discrete_points : bool
        True returns ``number_of_points`` independent contact locations for
        the sequence move -> apply force -> retract -> next point. False
        returns a dense, closed polyline intended for later continuous-force
        control. Robot execution is deliberately not performed here.
    number_of_points : int
        Number of locations returned in discrete mode.
    continuous_step_pixels : float
        Approximate pixel spacing along the dense continuous polyline.
    path_scale : float
        Fraction of the largest centered shape supported by the safe mask.

    Returns
    -------
    ndarray, shape (N, 2)
        Floating-point image coordinates ordered as ``[x, y]``. In continuous
        mode the first point is repeated at the end to close the path; it is
        not repeated in discrete mode.

    Notes
    -----
    Both shapes are centered at the centroid of ``safe_mask``. If that
    centroid lies outside an irregular or non-convex mask, the nearest valid
    safe-mask pixel is used. The circle radius is the available clearance at
    that center. The square uses the centered horizontal and vertical limits
    of the shrunken mask directly; it is not inscribed in the circle. Every
    sampled boundary point is still checked against the actual safe mask.
    """
    mask = np.asarray(safe_mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError("safe_mask must be a nonempty 2D mask")
    shape = str(shape).strip().lower()
    if shape not in {"square", "circle"}:
        raise ValueError("shape must be 'square' or 'circle'")
    if not isinstance(discrete_points, (bool, np.bool_)):
        raise TypeError("discrete_points must be True or False")
    if int(number_of_points) != number_of_points or number_of_points < 4:
        raise ValueError("number_of_points must be an integer of at least 4")
    continuous_step_pixels = float(continuous_step_pixels)
    if not np.isfinite(continuous_step_pixels) or continuous_step_pixels <= 0:
        raise ValueError("continuous_step_pixels must be positive and finite")
    path_scale = float(path_scale)
    if not np.isfinite(path_scale) or not 0.0 < path_scale <= 1.0:
        raise ValueError("path_scale must be in (0, 1]")

    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    distances = cv2.distanceTransform(
        padded,
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )[1:-1, 1:-1]
    # Use the center of the complete shrunken mask rather than the center of
    # its largest inscribed circle. This keeps square and circle trajectories
    # centered on the same detected safe tray region.
    safe_rows, safe_columns = np.nonzero(mask)
    center_x = float(np.mean(safe_columns))
    center_y = float(np.mean(safe_rows))
    center_column = int(np.rint(center_x))
    center_row = int(np.rint(center_y))

    # A non-convex mask can have its mathematical centroid outside the mask.
    # In that case, select the valid mask pixel nearest to the centroid.
    if (
        center_column < 0
        or center_column >= mask.shape[1]
        or center_row < 0
        or center_row >= mask.shape[0]
        or not mask[center_row, center_column]
    ):
        squared_distances = (
            np.square(safe_columns - center_x)
            + np.square(safe_rows - center_y)
        )
        nearest_index = int(np.argmin(squared_distances))
        center_x = float(safe_columns[nearest_index])
        center_y = float(safe_rows[nearest_index])
        center_column = int(safe_columns[nearest_index])
        center_row = int(safe_rows[nearest_index])

    if shape == "circle":
        available_size = float(distances[center_row, center_column])
    else:
        # Use the shrunken mask bounds directly. This makes a centered square
        # up to sqrt(2) larger than the former circle-inscribed square while
        # the validation below still protects irregular mask boundaries.
        available_size = float(
            min(
                center_x - np.min(safe_columns),
                np.max(safe_columns) - center_x,
                center_y - np.min(safe_rows),
                np.max(safe_rows) - center_y,
            )
        )
    path_size = available_size * path_scale
    if path_size < 1.0:
        raise ValueError("safe_mask is too small to generate a trajectory")

    if shape == "circle":
        perimeter_pixels = 2.0 * np.pi * path_size
    else:
        perimeter_pixels = 8.0 * path_size

    if discrete_points:
        sample_count = int(number_of_points)
    else:
        sample_count = max(
            12,
            int(np.ceil(perimeter_pixels / continuous_step_pixels)),
        )

    def make_path(current_size):
        parameters = np.arange(sample_count, dtype=float) / sample_count
        if shape == "circle":
            angles = 2.0 * np.pi * parameters
            x_values = center_x + current_size * np.cos(angles)
            y_values = center_y + current_size * np.sin(angles)
        else:
            current_half_side = current_size
            side_parameter = 4.0 * parameters
            side = np.floor(side_parameter).astype(int)
            offset = side_parameter - side
            x_values = np.empty(sample_count, dtype=float)
            y_values = np.empty(sample_count, dtype=float)
            selections = side == 0
            x_values[selections] = center_x - current_half_side + 2 * current_half_side * offset[selections]
            y_values[selections] = center_y - current_half_side
            selections = side == 1
            x_values[selections] = center_x + current_half_side
            y_values[selections] = center_y - current_half_side + 2 * current_half_side * offset[selections]
            selections = side == 2
            x_values[selections] = center_x + current_half_side - 2 * current_half_side * offset[selections]
            y_values[selections] = center_y + current_half_side
            selections = side == 3
            x_values[selections] = center_x - current_half_side
            y_values[selections] = center_y + current_half_side - 2 * current_half_side * offset[selections]
        return np.column_stack([x_values, y_values])

    # Pixel rounding can place a mathematical boundary point one pixel outside
    # an irregular mask. Reduce the path slightly until every sampled point is
    # valid, while retaining the requested shape and sample ordering.
    trajectory = None
    for _ in range(20):
        candidate = make_path(path_size)
        rounded = np.rint(candidate).astype(int)
        inside_image = (
            (rounded[:, 0] >= 0)
            & (rounded[:, 0] < mask.shape[1])
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < mask.shape[0])
        )
        if np.all(inside_image) and np.all(mask[rounded[:, 1], rounded[:, 0]]):
            trajectory = candidate
            break
        path_size *= 0.95
    if trajectory is None:
        raise RuntimeError("Could not fit the requested trajectory in safe_mask")

    if not discrete_points:
        trajectory = np.vstack([trajectory, trajectory[0]])
    return trajectory


def select_calibration_file(method, explicit_path=None):
    """Select the Li or Tsai result unless an explicit path was supplied."""
    if explicit_path is not None:
        return Path(explicit_path).resolve()

    method = str(method).strip().lower()
    if method == "li":
        return DEFAULT_CALIBRATION_FILE
    if method == "tsai":
        return TSAI_CALIBRATION_FILE
    raise ValueError("method must be either 'li' or 'tsai'.")


def load_T_base_camera(path):
    """Load and validate T_base_camera from a calibration JSON file."""
    path = Path(path).resolve()
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if "data" in data:
        names = data.get("description", {}).get("axis_0_order", [])
        try:
            transform_index = names.index("T_base_camera")
        except ValueError:
            transform_value = None
        else:
            transform_value = data["data"][transform_index]
    else:
        transform_value = data.get("T_base_camera", data.get("T_base_cam"))

    if transform_value is None:
        raise ValueError(
            f"{path} must describe a T_base_camera transform."
        )

    transform = np.asarray(transform_value, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("T_base_camera must be a finite 4x4 matrix.")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("T_base_camera has an invalid homogeneous last row.")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        raise ValueError("T_base_camera rotation is not orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
        raise ValueError("T_base_camera rotation determinant must be +1.")
    return transform


def camera_centroid_to_base(centroid_camera_m, T_base_camera):
    """Transform a camera-frame centroid in metres into the robot base frame."""
    centroid = np.asarray(centroid_camera_m, dtype=float).reshape(3)
    if not np.all(np.isfinite(centroid)) or centroid[2] <= 0:
        raise ValueError("Centroid must be finite and in front of the camera.")

    centroid_h = np.append(centroid, 1.0)
    centroid_base_m = (T_base_camera @ centroid_h)[:3]
    if not np.all(np.isfinite(centroid_base_m)):
        raise RuntimeError("The transformed centroid is not finite.")
    return centroid_base_m


def tray_pixel_to_camera_point(pixel_xy, camera_matrix, tray_plane):
    """Intersect one image pixel ray with the frozen tray plane."""
    pixel_xy = np.asarray(pixel_xy, dtype=float).reshape(2)
    camera_matrix = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(pixel_xy)):
        raise ValueError("Trajectory pixel must be finite")

    coefficients = np.asarray(
        tray_plane["coefficients"],
        dtype=float,
    ).reshape(4)
    ray = np.linalg.solve(
        camera_matrix,
        np.array([pixel_xy[0], pixel_xy[1], 1.0]),
    )
    denominator = float(coefficients[:3] @ ray)
    if abs(denominator) <= 1e-10:
        raise ValueError("Trajectory pixel ray is parallel to the tray plane")
    distance = -float(coefficients[3]) / denominator
    point_camera_m = distance * ray
    if (
        not np.all(np.isfinite(point_camera_m))
        or point_camera_m[2] <= 0.0
    ):
        raise ValueError("Trajectory pixel does not intersect the visible tray")
    return point_camera_m


def build_force_surface_targets(
    frozen_result,
    camera_matrix,
    T_base_camera,
    args,
):
    """Build centroid-only or discrete trajectory targets in base millimetres.

    With no ``--trajectory`` this returns only the existing detected centroid.
    In trajectory mode it shrinks the frozen segmentation mask for the tool,
    generates ordered image points, intersects every point with the frozen
    plane, and transforms the resulting camera points into the robot base.
    """
    if args.trajectory is None:
        centroid_base_mm = camera_centroid_to_base(
            frozen_result["centroid"],
            T_base_camera,
        ) * 1000.0
        return [centroid_base_mm], None

    safe_mask, clearance_pixels, closest_depth_m = (
        shrink_tray_mask_for_tool(
            frozen_result["detection"]["mask"],
            camera_matrix,
            frozen_result["plane"],
            tool_radius_mm=args.tool_radius_mm,
            safety_margin_mm=args.tool_safety_margin_mm,
        )
    )
    trajectory_pixels = generate_tray_trajectory_pixels(
        safe_mask,
        shape=args.trajectory,
        discrete_points=True,
        number_of_points=args.trajectory_points,
        path_scale=args.trajectory_scale,
    )

    targets_base_mm = []
    for pixel in trajectory_pixels:
        point_camera_m = tray_pixel_to_camera_point(
            pixel,
            camera_matrix,
            frozen_result["plane"],
        )
        point_base_mm = (
            camera_centroid_to_base(point_camera_m, T_base_camera) * 1000.0
        )
        if np.linalg.norm(point_base_mm) > 1000.0:
            raise ValueError(
                "A trajectory target is over 1 m from the robot base; "
                "check calibration and units"
            )
        targets_base_mm.append(point_base_mm)

    print(
        f"[INFO] Planned {len(targets_base_mm)} {args.trajectory} points; "
        f"tool clearance={clearance_pixels} px, "
        f"closest tray depth={closest_depth_m:.3f} m."
    )
    return targets_base_mm, trajectory_pixels


def move_lite6_tcp_to_centroid(
    lite6,
    centroid_camera_m,
    T_base_camera,
    speed_mm_s,
):
    """Move the Lite 6 TCP position without commanding its orientation."""
    if speed_mm_s <= 0:
        raise ValueError("speed_mm_s must be positive.")

    target_base_mm = (
        camera_centroid_to_base(centroid_camera_m, T_base_camera) * 1000.0
    )
    if np.linalg.norm(target_base_mm) > 1000.0:
        raise ValueError(
            "Transformed target is over 1 m from the robot base; "
            "check the calibration and units."
        )

    if lite6.arm is None:
        lite6.connect()
    else:
        lite6.reset_state()

    print(
        "[WARNING] This command moves the TCP directly to the tray centroid."
    )
    print(
        "[INFO] Target position [x, y, z] mm = "
        + ", ".join(f"{value:.2f}" for value in target_base_mm)
    )
    print("[INFO] End-effector orientation is not commanded.")
    confirmation = input("Type MOVE to execute this motion: ").strip()
    if confirmation != "MOVE":
        print("[INFO] Robot motion cancelled.")
        return None

    code = lite6.arm.set_position(
        x=float(target_base_mm[0]),
        y=float(target_base_mm[1]),
        z=float(target_base_mm[2]),
        speed=float(speed_mm_s),
        wait=True,
        is_radian=False,
    )
    if code != 0:
        raise RuntimeError(
            f"Lite 6 centroid move failed with error code: {code}"
        )

    print("[INFO] Lite 6 TCP reached the commanded centroid.")
    return target_base_mm


def tray_tool_rotation(normal_camera, T_base_camera, current_R_base_tool,
                       tool_z_sign=1.0):
    """Build a tool rotation with tool Z aligned to the tray push direction.

    ``camera.calculate_tray_plane`` makes the normal point away from the tray,
    toward the camera.  The push direction is consequently its negative.
    ``tool_z_sign=1`` aligns tool +Z with the push direction; use -1 when the
    contact face is on the tool's -Z side.
    """
    normal_camera = np.asarray(normal_camera, dtype=float).reshape(3)
    length = np.linalg.norm(normal_camera)
    if not np.isfinite(length) or length < 1e-9:
        raise ValueError("Tray normal must be a finite nonzero vector.")
    normal_camera /= length

    normal_base = T_base_camera[:3, :3] @ normal_camera
    normal_base /= np.linalg.norm(normal_base)
    push_direction_base = -normal_base
    z_tool = float(tool_z_sign) * push_direction_base

    # Retain the current tool twist by projecting its X axis onto the plane
    # perpendicular to the new tool Z axis.
    x_reference = np.asarray(current_R_base_tool, dtype=float)[:3, 0]
    x_tool = x_reference - np.dot(x_reference, z_tool) * z_tool
    if np.linalg.norm(x_tool) < 1e-6:
        candidates = np.eye(3)
        x_reference = min(
            candidates,
            key=lambda axis: abs(np.dot(axis, z_tool)),
        )
        x_tool = x_reference - np.dot(x_reference, z_tool) * z_tool
    x_tool /= np.linalg.norm(x_tool)
    y_tool = np.cross(z_tool, x_tool)
    y_tool /= np.linalg.norm(y_tool)
    x_tool = np.cross(y_tool, z_tool)
    x_tool /= np.linalg.norm(x_tool)
    return np.column_stack((x_tool, y_tool, z_tool)), push_direction_base


def rotation_to_lite6_rpy_deg(rotation):
    """Convert a rotation matrix to Lite 6 Rz(yaw)Ry(pitch)Rx(roll) RPY."""
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt

    pitch = np.arctan2(
        -rotation[2, 0],
        np.hypot(rotation[0, 0], rotation[1, 0]),
    )
    if abs(np.cos(pitch)) > 1e-7:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-rotation[0, 1], rotation[1, 1])
    return np.degrees([roll, pitch, yaw])


def command_cartesian_pose(lite6, position_base_mm, rpy_deg, speed_mm_s):
    """Send one blocking Cartesian pose command and validate its result."""
    position_base_mm = np.asarray(position_base_mm, dtype=float).reshape(3)
    rpy_deg = np.asarray(rpy_deg, dtype=float).reshape(3)
    code = lite6.arm.set_position(
        x=float(position_base_mm[0]),
        y=float(position_base_mm[1]),
        z=float(position_base_mm[2]),
        roll=float(rpy_deg[0]),
        pitch=float(rpy_deg[1]),
        yaw=float(rpy_deg[2]),
        speed=float(speed_mm_s),
        wait=True,
        is_radian=False,
    )
    if code != 0:
        raise RuntimeError(f"Lite 6 Cartesian move failed with code: {code}")


def calculate_tray_target(lite6, frozen_result, T_base_camera, tool_z_sign):
    """Calculate the frozen tray centroid, push direction, and tool RPY."""
    centroid_base_mm = camera_centroid_to_base(
        frozen_result["centroid"],
        T_base_camera,
    ) * 1000.0
    if np.linalg.norm(centroid_base_mm) > 1000.0:
        raise ValueError("Tray target is over 1 m from the robot base.")

    current_rotation = lite6.get_T_base_to_ee()[:3, :3]
    target_rotation, push_direction = tray_tool_rotation(
        frozen_result["plane"]["normal"],
        T_base_camera,
        current_rotation,
        tool_z_sign,
    )
    target_rpy_deg = rotation_to_lite6_rpy_deg(target_rotation)
    return centroid_base_mm, push_direction, target_rpy_deg


def print_frozen_target(centroid_base_mm, push_direction, target_rpy_deg):
    """Print the frozen vision result used by the upcoming robot command."""
    print("[INFO] Camera detection is now frozen for the entire motion.")
    print(
        "[INFO] Frozen centroid [mm]: "
        + ", ".join(f"{value:.2f}" for value in centroid_base_mm)
    )
    print(
        "[INFO] Push direction in base: "
        + ", ".join(f"{value:.5f}" for value in push_direction)
    )
    print(
        "[INFO] Target RPY [deg]: "
        + ", ".join(f"{value:.2f}" for value in target_rpy_deg)
    )


def orientation_staging_position(centroid_base_mm, x_mm, y_mm,
                                 clearance_mm):
    """Return the known XY staging point positioned above the tray in base Z."""
    centroid_base_mm = np.asarray(centroid_base_mm, dtype=float).reshape(3)
    position = np.array(
        [x_mm, y_mm, centroid_base_mm[2] + clearance_mm],
        dtype=float,
    )
    if not np.all(np.isfinite(position)):
        raise ValueError("Orientation staging position must be finite.")
    return position


def command_tray_orientation(lite6, staging_position_mm, target_rpy_deg,
                             speed_mm_s):
    """Command the calculated orientation at the safe staging position."""
    command_cartesian_pose(
        lite6,
        staging_position_mm,
        target_rpy_deg,
        speed_mm_s,
    )
    print(
        "[INFO] Orientation staging pose reached [mm]: "
        + ", ".join(f"{value:.2f}" for value in staging_position_mm)
    )


def run_orientation_test(lite6, frozen_result, T_base_camera, args,
                         move_to_centroid=False):
    """Test normal-to-orientation, optionally followed by centroid motion."""
    if lite6.arm is None:
        lite6.connect()
    else:
        lite6.reset_state()
    centroid_mm, push_direction, target_rpy = calculate_tray_target(
        lite6,
        frozen_result,
        T_base_camera,
        args.tool_z_sign,
    )
    print_frozen_target(centroid_mm, push_direction, target_rpy)
    staging_mm = orientation_staging_position(
        centroid_mm,
        args.orientation_x,
        args.orientation_y,
        args.orientation_clearance,
    )
    print(
        "[INFO] Orientation staging XYZ [mm]: "
        + ", ".join(f"{value:.2f}" for value in staging_mm)
    )
    action = "ORIENT_AND_MOVE" if move_to_centroid else "ORIENT"
    if input(f"Type {action} to execute: ").strip() != action:
        print("[INFO] Robot motion cancelled.")
        return None

    command_tray_orientation(lite6, staging_mm, target_rpy, args.speed)
    if move_to_centroid:
        command_cartesian_pose(lite6, centroid_mm, target_rpy, args.speed)
        print(
            "[INFO] Centroid reached while maintaining tool orientation; "
            "no force-control motion was commanded."
        )
    return target_rpy


def gravity_compensated_force(sensor, lite6, force_sign):
    """Read one fresh sample and remove modeled tool-weight loading."""
    sample = sensor.read(timeout_seconds=2.0)
    transform = lite6.get_T_base_to_ee()
    force = sensor.calibration.contact_force_from_raw(
        sample.raw_adc,
        transform,
    )
    return float(force_sign) * force


def search_for_target_force(
    lite6,
    sensor,
    approach_mm,
    push_direction,
    target_rpy_deg,
    args,
    point_index,
    point_count,
    video_recorder=None,
):
    """Advance from one approach pose until the configured force is reached."""
    command_cartesian_pose(
        lite6,
        approach_mm,
        target_rpy_deg,
        args.speed,
    )
    print(
        f"[INFO] Point {point_index}/{point_count}: approach reached; "
        "beginning contact search."
    )

    travel_mm = 0.0
    consecutive_target_samples = 0
    while travel_mm <= args.max_contact_travel + 1e-9:
        force_n = gravity_compensated_force(
            sensor,
            lite6,
            args.force_sign,
        )
        print(
            f"[FORCE] point {point_index}/{point_count} | "
            f"{force_n:7.3f} N | travel {travel_mm:6.2f} mm"
        )
        if video_recorder is not None:
            video_recorder.record_force(
                force_n,
                point_index,
                status="contact search",
            )
        if abs(force_n) >= args.max_force:
            print(
                f"[WARNING] Point {point_index}/{point_count}: force cutoff "
                f"reached at {force_n:.3f} N; stopping contact search."
            )
            return force_n, travel_mm, "force_limit"
        if force_n >= args.target_force:
            consecutive_target_samples += 1
            if consecutive_target_samples >= args.target_samples:
                print(
                    f"[INFO] Point {point_index}/{point_count}: target force "
                    f"reached at {force_n:.3f} N."
                )
                return force_n, travel_mm, "target_reached"
            # Stay at the current pose while confirming the force. Advancing
            # during confirmation would unnecessarily increase contact force.
            continue

        consecutive_target_samples = 0
        next_travel_mm = travel_mm + args.contact_step
        if next_travel_mm > args.max_contact_travel + 1e-9:
            break
        target_mm = approach_mm + next_travel_mm * push_direction
        command_cartesian_pose(
            lite6,
            target_mm,
            target_rpy_deg,
            args.contact_speed,
        )
        travel_mm = next_travel_mm

    raise RuntimeError(
        f"Point {point_index} reached maximum contact-search travel before "
        "target force"
    )


def apply_force_to_tray(
    lite6,
    frozen_result,
    camera_matrix,
    T_base_camera,
    args,
    video_frame_getter=None,
):
    """Apply force at the centroid or at every point of a frozen tray path.

    Centroid mode preserves the original behavior: reach the target force and
    hold the final pose. Trajectory mode executes a discrete sequence for each
    planned point: move to approach, search until target force, retract to that
    approach pose, then move to the next point. Any failure aborts the complete
    trajectory and attempts to retract from the active point.
    """
    # Keep force-sensor/Haplink dependencies optional in transformation mode.
    from force_sensor import ForceSensor

    if lite6.arm is None:
        lite6.connect()
    else:
        lite6.reset_state()
    centroid_base_mm, push_direction, target_rpy_deg = calculate_tray_target(
        lite6,
        frozen_result,
        T_base_camera,
        args.tool_z_sign,
    )
    surface_targets_mm, trajectory_pixels = build_force_surface_targets(
        frozen_result,
        camera_matrix,
        T_base_camera,
        args,
    )
    trajectory_mode = args.trajectory is not None
    staging_mm = orientation_staging_position(
        centroid_base_mm,
        args.orientation_x,
        args.orientation_y,
        args.orientation_clearance,
    )

    print_frozen_target(centroid_base_mm, push_direction, target_rpy_deg)
    print(
        "[INFO] Orientation staging XYZ [mm]: "
        + ", ".join(f"{value:.2f}" for value in staging_mm)
    )
    print(
        f"[WARNING] Motion may advance {args.max_contact_travel:.1f} mm "
        f"toward the tray. Target={args.target_force:.1f} N; "
        f"per-point force cutoff={args.max_force:.1f} N."
    )
    if trajectory_mode:
        print(
            f"[WARNING] The robot will execute {len(surface_targets_mm)} "
            "discrete contacts and retract after every point."
        )
        print("[INFO] Frozen trajectory targets:")
        for index, (pixel, target_mm) in enumerate(
            zip(trajectory_pixels, surface_targets_mm),
            start=1,
        ):
            print(
                f"  {index:02d}: pixel=({pixel[0]:.1f}, {pixel[1]:.1f}) | "
                "base mm=("
                + ", ".join(f"{value:.2f}" for value in target_mm)
                + ")"
            )
    if input("Type APPLY to execute: ").strip() != "APPLY":
        print("[INFO] Force application cancelled.")
        return None

    sensor = ForceSensor(port=args.force_port, baudrate=args.force_baud)
    video_recorder = None
    if args.video:
        if video_frame_getter is None:
            raise RuntimeError("Video enabled without an available camera")
        video_pixels = trajectory_pixels
        if video_pixels is None:
            centroid_pixel = camera_matrix @ frozen_result["centroid"]
            centroid_pixel = centroid_pixel[:2] / centroid_pixel[2]
            video_pixels = np.asarray([centroid_pixel])
        video_recorder = ForceTrajectoryVideoRecorder(
            args.video_output,
            video_pixels,
            video_frame_getter,
            args.target_force,
            args.max_force,
            args.video_fps,
        )
    active_approach_mm = None
    preserve_centroid_contact = False
    results = []
    try:
        sensor.connect()
        sensor.load_calibration(args.force_calibration)
        if not sensor.calibration.has_gravity_model:
            raise RuntimeError(
                "Force calibration has no gravity model; calibrate it first."
            )

        # Orientation is deliberately a separate, observable pipeline step.
        command_tray_orientation(
            lite6,
            staging_mm,
            target_rpy_deg,
            args.speed,
        )

        point_count = len(surface_targets_mm)
        for index, surface_mm in enumerate(surface_targets_mm, start=1):
            active_approach_mm = (
                surface_mm - args.approach_distance * push_direction
            )
            print(
                f"[INFO] Surface target {index}/{point_count} [mm]: "
                + ", ".join(f"{value:.2f}" for value in surface_mm)
            )
            force_n, travel_mm, outcome = search_for_target_force(
                lite6,
                sensor,
                active_approach_mm,
                push_direction,
                target_rpy_deg,
                args,
                index,
                point_count,
                video_recorder=video_recorder,
            )
            result = {
                "point_index": index,
                "surface_base_mm": np.asarray(surface_mm).tolist(),
                "force_newtons": float(force_n),
                "contact_travel_mm": float(travel_mm),
                "outcome": outcome,
            }
            if trajectory_pixels is not None:
                result["image_pixel_xy"] = trajectory_pixels[index - 1].tolist()
            results.append(result)

            if not trajectory_mode and outcome == "target_reached":
                preserve_centroid_contact = True
                print("[INFO] Holding the final commanded centroid pose.")
                return force_n

            print(
                f"[INFO] Retracting from point {index}/{point_count} "
                f"({outcome}) to its approach pose."
            )
            command_cartesian_pose(
                lite6,
                active_approach_mm,
                target_rpy_deg,
                args.speed,
            )
            active_approach_mm = None

            if not trajectory_mode:
                print(
                    "[INFO] Centroid force cutoff handled; no additional "
                    "points remain."
                )
                return force_n

        print(
            f"[INFO] Completed all {point_count} trajectory contact points."
        )
        return results
    finally:
        if (
            active_approach_mm is not None
            and not preserve_centroid_contact
            and lite6.arm is not None
        ):
            print("[WARNING] Retracting from the active point.")
            try:
                command_cartesian_pose(
                    lite6,
                    active_approach_mm,
                    target_rpy_deg,
                    args.speed,
                )
            except Exception as retract_error:
                print(f"[ERROR] Automatic retract failed: {retract_error}")
                lite6.arm.set_state(4)
        try:
            if video_recorder is not None:
                video_recorder.close()
        finally:
            sensor.disconnect()


def draw_result(image, result, camera_matrix, mode):
    """Draw the detected mask, observed plane support, and 3D centroid."""
    display = image.copy()
    if mode == "transformation":
        lines = ["S save | M test transformation | Q or ESC quit"]
    elif mode == "orientation":
        lines = ["S save | O test tool orientation | Q or ESC quit"]
    elif mode == "orientation_movement":
        lines = ["S save | M orient and move to centroid | Q or ESC quit"]
    else:
        lines = ["S save | F freeze detection and apply force | Q or ESC quit"]

    if result is None:
        lines.append("Tray not detected")
    else:
        detection = result["detection"]
        mask = detection["mask"]
        lines.append(f"Tray score: {detection['score']:.3f}")

        plane = result["plane"]
        if plane is None:
            # Green detection mask is shown only when plane fitting failed.
            overlay = display.copy()
            overlay[mask] = (0, 180, 0)
            display = cv2.addWeighted(display, 0.70, overlay, 0.30, 0)
            lines.append("Tray detected, but plane could not be calculated")
        else:
            # A valid RANSAC plane generalizes to the full detected tray mask.
            plane_overlay = display.copy()
            plane_overlay[mask] = (255, 0, 0)
            display = cv2.addWeighted(
                display,
                0.65,
                plane_overlay,
                0.35,
                0,
            )
            lines.append(
                "Plane inliers: "
                f"{plane['number_of_inliers']}/"
                f"{plane['number_of_ransac_points']}"
            )

        centroid = result["centroid"]
        if centroid is not None and centroid[2] > 0:
            pixel = camera_matrix @ centroid
            pixel = pixel[:2] / pixel[2]
            center = tuple(np.rint(pixel).astype(int))
            cv2.circle(display, center, 8, (0, 0, 255), -1)
            lines.append(
                "Centroid [m]: "
                + ", ".join(f"{value:.3f}" for value in centroid)
            )

    for index, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (15, 30 + 28 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return display


def main():
    parser = argparse.ArgumentParser(
        description="Test tray localization or apply force with the Lite 6"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test-transformation", action="store_true")
    mode.add_argument("--test-orientation", action="store_true")
    mode.add_argument("--test-orientation-movement", action="store_true")
    mode.add_argument("--apply-force", action="store_true")
    parser.add_argument(
        "--trajectory",
        choices=("square", "circle"),
        default=None,
        help=(
            "Optional discrete force trajectory. If omitted, apply force "
            "once at the detected tray centroid."
        ),
    )
    parser.add_argument(
        "--points",
        "--number-of-points",
        "--trajectory-points",
        dest="trajectory_points",
        type=int,
        default=12,
        help="Number of discrete trajectory contact points (default: 12)",
    )
    parser.add_argument(
        "--tool-radius-mm",
        type=float,
        default=DEFAULT_TOOL_RADIUS_MM,
        help=(
            "Physical tool radius used to shrink the tray mask "
            f"(default: {DEFAULT_TOOL_RADIUS_MM:g})"
        ),
    )
    parser.add_argument(
        "--tool-safety-margin-mm",
        type=float,
        default=0.0,
        help="Additional physical clearance inside the tray boundary",
    )
    parser.add_argument(
        "--trajectory-scale",
        type=float,
        default=0.9,
        help="Fraction of the safe inscribed shape to use, in (0, 1]",
    )
    parser.add_argument(
        "--video",
        action=argparse.BooleanOptionalAction,
        default=ENABLE_FORCE_VIDEO,
        help=(
            "Record a clean camera view with planned points and a live "
            "force-time plot; use --no-video to override the top-level default"
        ),
    )
    parser.add_argument(
        "--video-output",
        type=Path,
        default=None,
        help="Output MP4 path (default: Integration/data/force_data)",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=DEFAULT_VIDEO_FPS,
        help="Force-video playback frame rate (default: 10)",
    )
    parser.add_argument("--ip", required=True, help="Lite 6 controller IP")
    parser.add_argument(
        "--method",
        choices=("li", "tsai"),
        default="tsai",
        help="Select the saved eye-to-hand calibration result",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Override --method with a specific calibration JSON",
    )
    parser.add_argument("--speed", type=float, default=30.0)
    parser.add_argument("--orientation-x", type=float,
                        default=DEFAULT_ORIENTATION_X_MM,
                        help="Base-frame X of orientation staging pose in mm")
    parser.add_argument("--orientation-y", type=float,
                        default=DEFAULT_ORIENTATION_Y_MM,
                        help="Base-frame Y of orientation staging pose in mm")
    parser.add_argument("--orientation-clearance", type=float,
                        default=DEFAULT_ORIENTATION_CLEARANCE_MM,
                        help="Base-Z distance above transformed tray Z in mm")
    parser.add_argument("--contact-speed", type=float, default=5.0)
    parser.add_argument("--approach-distance", type=float, default=50.0,
                        help="Approach clearance in mm")
    parser.add_argument("--contact-step", type=float, default=5.0,
                        help="Incremental contact-search step in mm")
    parser.add_argument("--max-contact-travel", type=float, default=80.0,
                        help="Maximum travel from the approach pose in mm")
    parser.add_argument("--target-force", type=float, default=10.0)
    parser.add_argument("--max-force", type=float, default=45.0,
                        help=("Per-point retract-and-continue threshold in N "
                              "(must be <= 50)"))
    parser.add_argument("--target-samples", type=int, default=3,
                        help="Consecutive samples required above target")
    parser.add_argument("--tool-z-sign", type=float, choices=(-1.0, 1.0),
                        default=1.0,
                        help="Tool Z sign that points into the tray")
    parser.add_argument("--force-sign", type=float, choices=(-1.0, 1.0),
                        default=1.0,
                        help="Sign making compression positive")
    parser.add_argument("--force-port", default="/dev/ttyUSB0")
    parser.add_argument("--force-baud", type=int, default=115200)
    parser.add_argument("--force-calibration", type=Path,
                        default=DEFAULT_FORCE_CALIBRATION_FILE)
    args = parser.parse_args()

    positive_values = {
        "speed": args.speed,
        "contact_speed": args.contact_speed,
        "approach_distance": args.approach_distance,
        "orientation_clearance": args.orientation_clearance,
        "contact_step": args.contact_step,
        "max_contact_travel": args.max_contact_travel,
        "target_force": args.target_force,
        "max_force": args.max_force,
        "target_samples": args.target_samples,
        "trajectory_points": args.trajectory_points,
        "tool_radius_mm": args.tool_radius_mm,
        "trajectory_scale": args.trajectory_scale,
        "video_fps": args.video_fps,
    }
    for name, value in positive_values.items():
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not np.isfinite(args.orientation_x) or not np.isfinite(
        args.orientation_y
    ):
        parser.error("--orientation-x and --orientation-y must be finite")
    if (
        not np.isfinite(args.tool_safety_margin_mm)
        or args.tool_safety_margin_mm < 0.0
    ):
        parser.error("--tool-safety-margin-mm must be finite and nonnegative")
    if args.trajectory_points < 4:
        parser.error("--points must be at least 4")
    if args.trajectory_scale > 1.0:
        parser.error("--trajectory-scale cannot exceed 1")
    if args.trajectory is not None and not args.apply_force:
        parser.error("--trajectory can only be used with --apply-force")
    if args.video and not args.apply_force:
        parser.error("--video can only be used with --apply-force")
    if args.video_output is None:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        args.video_output = FORCE_VIDEO_DIR / f"force_trajectory_{timestamp}.mp4"
    else:
        args.video_output = args.video_output.resolve()
    if args.target_force >= args.max_force:
        parser.error("--target-force must be below --max-force")
    if args.max_force > 50.0:
        parser.error("--max-force cannot exceed the 50 N robot safety limit")
    args.force_calibration = args.force_calibration.resolve()
    if args.test_transformation:
        run_mode = "transformation"
    elif args.test_orientation:
        run_mode = "orientation"
    elif args.test_orientation_movement:
        run_mode = "orientation_movement"
    else:
        run_mode = "force"

    calibration_file = select_calibration_file(
        args.method,
        args.calibration,
    )
    T_base_camera = load_T_base_camera(calibration_file)
    calibration_label = (
        "custom" if args.calibration is not None else args.method.upper()
    )
    print(
        f"[INFO] Using {calibration_label} calibration: "
        f"{calibration_file}"
    )
    lite6 = robot.Lite6(args.ip)
    predictor = build_tray_predictor()
    zed = None
    latest_result = None

    try:
        zed, runtime, image_zed = open_zed()
        point_cloud_zed = sl.Mat()
        camera_matrix, _ = get_zed_left_intrinsics_rectified(zed)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        while True:
            # get_image grabs once; get_point_cloud retrieves that same frame.
            image = get_image(zed, runtime, image_zed)
            if image is None:
                continue

            xyz, valid_mask = get_point_cloud(zed, point_cloud_zed)
            if xyz is None:
                continue

            latest_result = process_tray(
                image,
                xyz,
                camera_matrix,
                predictor,
                valid_mask,
            )
            display = draw_result(image, latest_result, camera_matrix, run_mode)
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKeyEx(1)

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                if (
                    latest_result is None
                    or latest_result["plane"] is None
                    or latest_result["centroid"] is None
                ):
                    print("[WARNING] No valid tray plane and centroid to save.")
                else:
                    save_tray_data(
                        latest_result["plane"],
                        latest_result["centroid"],
                    )
            if run_mode == "transformation" and key in (ord("m"), ord("M")):
                if (
                    latest_result is None
                    or latest_result["plane"] is None
                    or latest_result["centroid"] is None
                ):
                    print("[WARNING] No valid tray centroid for robot motion.")
                else:
                    move_lite6_tcp_to_centroid(
                        lite6,
                        latest_result["centroid"],
                        T_base_camera,
                        args.speed,
                    )
            if run_mode == "orientation" and key in (ord("o"), ord("O")):
                if (
                    latest_result is None
                    or latest_result["plane"] is None
                    or latest_result["centroid"] is None
                ):
                    print("[WARNING] No valid tray plane and centroid.")
                else:
                    run_orientation_test(
                        lite6,
                        latest_result,
                        T_base_camera,
                        args,
                        move_to_centroid=False,
                    )
            if (
                run_mode == "orientation_movement"
                and key in (ord("m"), ord("M"))
            ):
                if (
                    latest_result is None
                    or latest_result["plane"] is None
                    or latest_result["centroid"] is None
                ):
                    print("[WARNING] No valid tray plane and centroid.")
                else:
                    run_orientation_test(
                        lite6,
                        latest_result,
                        T_base_camera,
                        args,
                        move_to_centroid=True,
                    )
            if run_mode == "force" and key in (ord("f"), ord("F")):
                if (
                    latest_result is None
                    or latest_result["plane"] is None
                    or latest_result["centroid"] is None
                ):
                    print("[WARNING] No valid tray plane and centroid.")
                else:
                    # This blocking routine deliberately prevents additional
                    # camera grabs/detections until all motion has stopped.
                    apply_force_to_tray(
                        lite6,
                        latest_result,
                        camera_matrix,
                        T_base_camera,
                        args,
                        video_frame_getter=(
                            lambda: get_image(zed, runtime, image_zed)
                        ),
                    )
    finally:
        cv2.destroyAllWindows()
        if zed is not None:
            zed.close()
        lite6.disconnect()


if __name__ == "__main__":
    main()
