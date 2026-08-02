import argparse
import pyzed.sl as sl
import cv2 as cv
import numpy as np
from collections import deque
from pathlib import Path
import json
import time
import robot

from camera import (
    ZED_FPS,
    draw_arm_points_and_lines,
    get_arm_points,
    get_arm_vectors,
    get_image,
    get_single_body,
    open_zed,
    setup_body_tracking,
)


"""
Using the ZED SDK to Track the human arm motion
"""


# --------------------------------------------------
# Data Processing
# --------------------------------------------------

ONE_EURO_MIN_CUTOFF = 1.0
ONE_EURO_BETA = 0.05
ONE_EURO_DERIVATIVE_CUTOFF = 1.0

CAMERA_FPS = ZED_FPS
REFERENCE_WINDOW_FRAMES = CAMERA_FPS
REGRESSION_WINDOW_FRAMES = 5
ARM_SIDE = "right"
REFERENCE_DELAY_SECONDS = 10.0
CAMERA_WINDOW_NAME = "ZED Arm Tracking"
CAMERA_WINDOW_SIZE = (1280, 720)
TELEOP_SCALE = 0.5
PREDICTION_SECONDS = 1.0 / CAMERA_FPS
LITE6_COMMAND_SPEED_MM_S = 30.0
LITE6_MAX_FRAME_STEP_MM = 10.0

TELEOPERATION_DIR = Path(__file__).resolve().parent
ARM_TRACKING_DATA_DIR = TELEOPERATION_DIR / "data" / "arm_tracking"
PROCESSED_DATA_FILE = (
    ARM_TRACKING_DATA_DIR / "processed_arm_tracking.json"
)

VECTOR_NAMES = (
    "shoulder_to_elbow",
    "elbow_to_wrist",
    "shoulder_to_wrist",
)
VECTOR_AXES = ("x", "y", "z")
PROCESSED_COLUMNS = (
    ("start_time_s", "end_time_s")
    + tuple(
        f"{value}_{vector}_{axis}"
        for value in ("first", "last", "velocity")
        for vector in VECTOR_NAMES
        for axis in VECTOR_AXES
    )
)
TIME_COLUMNS = slice(0, 2)
END_TIME_COLUMN = 1
FIRST_VECTOR_COLUMNS = slice(2, 11)
LAST_VECTOR_COLUMNS = slice(11, 20)
VELOCITY_COLUMNS = slice(20, 29)
PROCESSED_COLUMN_COUNT = len(PROCESSED_COLUMNS)

# ZED camera: +X right, +Y down, +Z forward.
# Lite 6 base: camera right -> +X, camera forward -> +Y, camera up -> +Z.
R_ROBOT_CAMERA = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)

LITE6_START_POSE_MM_DEG = np.array(
    [500.0, 0.9, 320.0, 180.0, -90.0, 0.0],
    dtype=float,
)
LITE6_MAX_DISPLACEMENT_MM = 200.0


class OneEuroFilter:
    """Adaptive low-pass filter for timestamped vectors or vector arrays."""

    def __init__(
        self,
        min_cutoff=ONE_EURO_MIN_CUTOFF,
        beta=ONE_EURO_BETA,
        derivative_cutoff=ONE_EURO_DERIVATIVE_CUTOFF,
    ):
        if min_cutoff <= 0.0 or derivative_cutoff <= 0.0:
            raise ValueError("cutoff frequencies must be positive")
        if beta < 0.0:
            raise ValueError("beta cannot be negative")

        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.derivative_cutoff = float(derivative_cutoff)
        self.reset()

    @staticmethod
    def _alpha(cutoff, timestep):
        time_constant = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + time_constant / timestep)

    def reset(self, value=None, timestamp=None):
        """Clear history, optionally seeding the filter with one sample."""

        self.previous_raw = None
        self.previous_filtered = None
        self.previous_derivative = None
        self.previous_time = None
        if value is not None:
            value = np.asarray(value, dtype=float)
            if not np.all(np.isfinite(value)):
                raise ValueError("filter value must contain finite numbers")
            self.previous_raw = value.copy()
            self.previous_filtered = value.copy()
            self.previous_derivative = np.zeros_like(value)
            self.previous_time = (
                time.monotonic() if timestamp is None else float(timestamp)
            )
        return self.previous_filtered

    def update(self, value, timestamp=None):
        """Filter one sample using its actual monotonic timestamp."""

        value = np.asarray(value, dtype=float)
        if value.ndim < 1 or value.shape[-1] != 3:
            raise ValueError("value must have shape (3,) or (..., 3)")
        if not np.all(np.isfinite(value)):
            raise ValueError("value must contain finite numbers")

        timestamp = time.monotonic() if timestamp is None else float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self.previous_raw is None:
            return self.reset(value, timestamp).copy()
        if value.shape != self.previous_raw.shape:
            raise ValueError("all filter samples must have the same shape")

        timestep = timestamp - self.previous_time
        if timestep <= 0.0:
            raise ValueError("timestamps must be strictly increasing")

        raw_derivative = (value - self.previous_raw) / timestep
        derivative_alpha = self._alpha(
            self.derivative_cutoff,
            timestep,
        )
        filtered_derivative = (
            derivative_alpha * raw_derivative
            + (1.0 - derivative_alpha) * self.previous_derivative
        )
        cutoff = self.min_cutoff + self.beta * np.abs(filtered_derivative)
        signal_alpha = self._alpha(cutoff, timestep)
        filtered = (
            signal_alpha * value
            + (1.0 - signal_alpha) * self.previous_filtered
        )

        self.previous_raw = value.copy()
        self.previous_filtered = filtered.copy()
        self.previous_derivative = filtered_derivative.copy()
        self.previous_time = timestamp
        return filtered


def map_displacement_to_robot(
    displacement_camera,
    scale=0.5,
    max_displacement_m=None,
):
    """
    Map a relative XYZ displacement from the ZED frame to the Lite 6 frame.

    Parameters
    ----------
    displacement_camera : array-like, shape (3,)
        Human-arm displacement [x, y, z] in the ZED camera frame, in meters.

    scale : float
        Robot-motion scale. For example, 0.25 maps 10 cm of human motion to
        2.5 cm of robot motion.

    max_displacement_m : float, optional
        Maximum mapped displacement magnitude. Larger commands are scaled
        down while preserving their direction.

    Returns
    -------
    np.ndarray, shape (3,)
        Lite 6 base-frame displacement [x, y, z] in meters.
    """

    displacement_camera = np.asarray(
        displacement_camera,
        dtype=float,
    ).reshape(3)
    if not np.all(np.isfinite(displacement_camera)):
        raise ValueError("displacement_camera must contain finite numbers")
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("scale must be finite and non-negative")
    if max_displacement_m is not None and (
        not np.isfinite(max_displacement_m)
        or max_displacement_m <= 0.0
    ):
        raise ValueError("max_displacement_m must be finite and positive")

    displacement_robot = (
        float(scale) * R_ROBOT_CAMERA @ displacement_camera
    )

    if max_displacement_m is not None:
        magnitude = np.linalg.norm(displacement_robot)
        if magnitude > max_displacement_m:
            displacement_robot *= max_displacement_m / magnitude

    return displacement_robot


def meters_to_millimeters(value):
    """Convert finite position or displacement data from meters to millimeters."""

    value = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(value)):
        raise ValueError("value must contain finite numbers")
    return value * 1000.0


def make_lite6_position_command(
    displacement_robot_m,
    start_pose=LITE6_START_POSE_MM_DEG,
    max_displacement_mm=LITE6_MAX_DISPLACEMENT_MM,
):
    """
    Convert a robot-frame displacement into a safe absolute Lite 6 pose.

    XYZ displacement is limited to 150 mm per axis by default. Positive X is
    additionally capped at its starting value, so the arm cannot move farther
    outward than the initial X position. Roll, pitch, and yaw remain fixed at
    the starting orientation.
    """

    displacement_robot_m = np.asarray(
        displacement_robot_m,
        dtype=float,
    ).reshape(3)
    start_pose = np.asarray(start_pose, dtype=float).reshape(6)

    if not np.all(np.isfinite(displacement_robot_m)):
        raise ValueError("displacement_robot_m must contain finite numbers")
    if not np.all(np.isfinite(start_pose)):
        raise ValueError("start_pose must contain finite numbers")
    if (
        not np.isfinite(max_displacement_mm)
        or max_displacement_mm <= 0.0
    ):
        raise ValueError("max_displacement_mm must be finite and positive")

    displacement_mm = meters_to_millimeters(displacement_robot_m)
    minimum_xyz = start_pose[:3] - max_displacement_mm
    maximum_xyz = start_pose[:3] + max_displacement_mm

    # X may move inward by up to the safety limit, but never beyond its
    # starting position.
    maximum_xyz[0] = start_pose[0]

    target_xyz = np.clip(
        start_pose[:3] + displacement_mm,
        minimum_xyz,
        maximum_xyz,
    )

    return {
        "x": float(target_xyz[0]),
        "y": float(target_xyz[1]),
        "z": float(target_xyz[2]),
        "roll": float(start_pose[3]),
        "pitch": float(start_pose[4]),
        "yaw": float(start_pose[5]),
    }


def predict_wrist_displacement(
    regression_block,
    prediction_seconds=PREDICTION_SECONDS,
):
    """Predict shoulder-to-wrist displacement over the next control frame."""

    block = np.asarray(
        regression_block,
        dtype=float,
    ).reshape(PROCESSED_COLUMN_COUNT)
    if not np.all(np.isfinite(block)):
        raise ValueError("regression_block must contain finite numbers")
    if not np.isfinite(prediction_seconds) or prediction_seconds <= 0.0:
        raise ValueError("prediction_seconds must be finite and positive")

    velocities = block[VELOCITY_COLUMNS].reshape(3, 3)
    shoulder_to_wrist_velocity = velocities[2]
    return shoulder_to_wrist_velocity * prediction_seconds


class Lite6TeleopController:
    """Send bounded absolute position commands from predicted human motion."""

    def __init__(
        self,
        ip_address,
        scale=TELEOP_SCALE,
        speed_mm_s=LITE6_COMMAND_SPEED_MM_S,
    ):
        if not ip_address:
            raise ValueError("A Lite 6 IP address is required")
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("scale must be finite and non-negative")
        if not np.isfinite(speed_mm_s) or speed_mm_s <= 0.0:
            raise ValueError("speed_mm_s must be finite and positive")

        self.lite6 = robot.Lite6(ip_address)
        self.scale = float(scale)
        self.speed_mm_s = float(speed_mm_s)
        self.accumulated_displacement_m = np.zeros(3, dtype=float)
        self.last_command = make_lite6_position_command(
            self.accumulated_displacement_m
        )

    def connect_and_move_to_start(self):
        """Connect and move to the fixed teleoperation starting pose."""

        self.lite6.connect()
        self.lite6.reset_state()
        command = make_lite6_position_command(np.zeros(3, dtype=float))
        code = self.lite6.arm.set_position(
            **command,
            speed=self.speed_mm_s,
            wait=True,
            is_radian=False,
        )
        if code != 0:
            raise RuntimeError(
                f"Lite 6 start-position command failed with code {code}"
            )
        self.last_command = command

    def reset_to_start(self):
        """Stop queued motion and return to the fixed starting pose."""

        self.lite6.reset_state()
        self.accumulated_displacement_m.fill(0.0)
        command = make_lite6_position_command(
            self.accumulated_displacement_m
        )
        code = self.lite6.arm.set_position(
            **command,
            speed=self.speed_mm_s,
            wait=True,
            is_radian=False,
        )
        if code != 0:
            raise RuntimeError(f"Lite 6 reset failed with code {code}")
        self.last_command = command

    def send_predicted_displacement(self, displacement_camera_m):
        """Map one predicted camera-frame displacement and command the robot."""

        displacement_robot_m = map_displacement_to_robot(
            displacement_camera_m,
            scale=self.scale,
        )
        maximum_step_m = LITE6_MAX_FRAME_STEP_MM / 1000.0
        step_magnitude = np.linalg.norm(displacement_robot_m)
        if step_magnitude > maximum_step_m:
            displacement_robot_m *= maximum_step_m / step_magnitude

        self.accumulated_displacement_m += displacement_robot_m
        command = make_lite6_position_command(
            self.accumulated_displacement_m
        )
        commanded_xyz_mm = np.array(
            [command["x"], command["y"], command["z"]],
            dtype=float,
        )
        self.accumulated_displacement_m = (
            commanded_xyz_mm - LITE6_START_POSE_MM_DEG[:3]
        ) / 1000.0
        code = self.lite6.arm.set_position(
            **command,
            speed=self.speed_mm_s,
            wait=False,
            is_radian=False,
        )
        if code != 0:
            raise RuntimeError(
                f"Lite 6 position command failed with code {code}"
            )
        self.last_command = command
        return command

    def stop(self):
        """Stop robot motion without issuing another trajectory."""

        if self.lite6.arm is not None:
            self.lite6.arm.set_state(4)

    def disconnect(self):
        """Disconnect from the Lite 6 controller."""

        self.lite6.disconnect()


def linear_regression_1d(times, coordinates):
    """
    Fit coordinate = velocity * time + intercept.

    Returns
    -------
    velocity : float
        Fitted slope in coordinate units per second.

    intercept : float
        Fitted coordinate at time zero.

    r_squared : float
        Goodness-of-fit value. Constant data returns 0.0.
    """

    times = np.asarray(times, dtype=float).reshape(-1)
    coordinates = np.asarray(coordinates, dtype=float).reshape(-1)

    if times.size != coordinates.size:
        raise ValueError("times and coordinates must have the same length")
    if times.size == 0:
        raise ValueError("at least one sample is required")
    if not np.all(np.isfinite(times)):
        raise ValueError("times must contain finite values")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates must contain finite values")

    if times.size < 2:
        return 0.0, float(coordinates[0]), 0.0

    centered_times = times - np.mean(times)
    centered_coordinates = coordinates - np.mean(coordinates)
    denominator = np.dot(centered_times, centered_times)

    if denominator <= np.finfo(float).eps:
        return 0.0, float(coordinates[-1]), 0.0

    velocity = np.dot(centered_times, centered_coordinates) / denominator
    intercept = np.mean(coordinates) - velocity * np.mean(times)
    predictions = velocity * times + intercept

    residual_sum = np.sum((coordinates - predictions) ** 2)
    total_sum = np.sum(centered_coordinates ** 2)
    r_squared = (
        0.0
        if total_sum <= np.finfo(float).eps
        else 1.0 - residual_sum / total_sum
    )

    return float(velocity), float(intercept), float(r_squared)


def linear_regression_3d(times, vectors):
    """
    Estimate a 3D vector's rate of change over a rolling frame window.

    Parameters
    ----------
    times : array-like, shape (N,)
        Sample timestamps in seconds.

    vectors : array-like, shape (N, 3)
        One-Euro-filtered arm vectors for the same timestamps.

    Returns
    -------
    velocity : np.ndarray, shape (3,)
        Per-axis regression slopes.

    r_squared : np.ndarray, shape (3,)
        Per-axis goodness-of-fit values.
    """

    times = np.asarray(times, dtype=float).reshape(-1)
    vectors = np.asarray(vectors, dtype=float)

    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("vectors must have shape (N, 3)")
    if vectors.shape[0] != times.size:
        raise ValueError("times and vectors must contain the same samples")

    velocity = np.empty(3, dtype=float)
    r_squared = np.empty(3, dtype=float)

    for axis in range(3):
        velocity[axis], _, r_squared[axis] = linear_regression_1d(
            times,
            vectors[:, axis],
        )

    return velocity, r_squared


def regress_arm_window(times, vector_frames):
    """
    Fit all three arm vectors across a processing window of any length.

    Parameters
    ----------
    times : array-like, shape (N,)
        Monotonic timestamps in seconds.

    vector_frames : array-like, shape (N, 3, 3)
        One-Euro-filtered vectors. Axis 1 follows shoulder-to-elbow,
        elbow-to-wrist, and shoulder-to-wrist. At least two frames are
        required.

    Returns
    -------
    np.ndarray, shape (29,)
        Numeric row ordered according to ``PROCESSED_COLUMNS``:
        two timestamps, nine first-vector values, nine last-vector values,
        and nine velocities.
    """

    times = np.asarray(times, dtype=float).reshape(-1)
    vector_frames = np.asarray(vector_frames, dtype=float)
    number_of_samples = times.size

    if number_of_samples < 2:
        raise ValueError("at least two samples are required for regression")
    if vector_frames.shape != (number_of_samples, 3, 3):
        raise ValueError(
            f"vector_frames must have shape ({number_of_samples}, 3, 3)"
        )
    if not np.all(np.isfinite(times)):
        raise ValueError("times must contain finite values")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("times must be strictly increasing")
    if not np.all(np.isfinite(vector_frames)):
        raise ValueError("vector_frames must contain finite values")

    relative_times = times - times[0]
    duration = float(relative_times[-1])

    first_vectors = np.empty((3, 3), dtype=float)
    last_vectors = np.empty((3, 3), dtype=float)
    velocities = np.empty((3, 3), dtype=float)

    for vector_index in range(3):
        for axis in range(3):
            velocity, intercept, _ = linear_regression_1d(
                relative_times,
                vector_frames[:, vector_index, axis],
            )
            first_vectors[vector_index, axis] = intercept
            last_vectors[vector_index, axis] = (
                velocity * duration + intercept
            )
            velocities[vector_index, axis] = velocity

    return np.concatenate(
        (
            times[[0, -1]],
            first_vectors.reshape(9),
            last_vectors.reshape(9),
            velocities.reshape(9),
        )
    )


def make_reference_block(vector_frames):
    """Return a zero-time processed row containing the estimated arm reference."""

    vector_frames = np.asarray(vector_frames, dtype=float)
    if vector_frames.ndim != 3 or vector_frames.shape[1:] != (3, 3):
        raise ValueError("vector_frames must have shape (N, 3, 3)")
    if len(vector_frames) < 1 or not np.all(np.isfinite(vector_frames)):
        raise ValueError("at least one finite reference frame is required")

    reference_vectors = np.mean(vector_frames, axis=0).reshape(9)
    return np.concatenate(
        (
            np.zeros(2, dtype=float),
            reference_vectors,
            reference_vectors,
            np.zeros(9, dtype=float),
        )
    )


def synchronize_regression_block(block, previous_block):
    """Connect a regression row to the preceding row without a pose jump."""

    block = np.asarray(block, dtype=float).reshape(PROCESSED_COLUMN_COUNT).copy()
    previous_block = np.asarray(
        previous_block,
        dtype=float,
    ).reshape(PROCESSED_COLUMN_COUNT)
    if not np.all(np.isfinite(block)) or not np.all(np.isfinite(previous_block)):
        raise ValueError("regression blocks must contain finite numbers")

    start_time_s, end_time_s = block[TIME_COLUMNS]
    duration_s = end_time_s - start_time_s
    if duration_s <= 0.0:
        raise ValueError("regression block duration must be positive")

    first_vectors = previous_block[LAST_VECTOR_COLUMNS]
    velocity = block[VELOCITY_COLUMNS]
    block[FIRST_VECTOR_COLUMNS] = first_vectors
    block[LAST_VECTOR_COLUMNS] = first_vectors + velocity * duration_s
    return block


def delete_processed_data(output_path=PROCESSED_DATA_FILE):
    """Delete saved and temporary tracking output when restarting."""

    output_path = Path(output_path)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    for path in (output_path, temporary_path):
        if path.is_file():
            path.unlink()


def save_processed_blocks(
    blocks,
    output_path=PROCESSED_DATA_FILE,
    arm=ARM_SIDE,
):
    """Save regression blocks as numeric rows with columns in description."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray(blocks, dtype=float)
    if data.size == 0:
        data = np.empty((0, PROCESSED_COLUMN_COUNT), dtype=float)
    elif data.ndim != 2 or data.shape[1] != PROCESSED_COLUMN_COUNT:
        raise ValueError(
            f"blocks must have shape (N, {PROCESSED_COLUMN_COUNT})"
        )
    if not np.all(np.isfinite(data)):
        raise ValueError("processed blocks must contain finite numbers")

    payload = {
        "description": {
            "summary": (
                "One-Euro-filtered BODY_18 arm vectors processed with "
                "linear regression. Data row 0 is the estimated reference "
                "pose; its time and velocity values are zero."
            ),
            "body_format": "BODY_18",
            "arm": arm,
            "columns": list(PROCESSED_COLUMNS),
            "camera_fps": CAMERA_FPS,
            "reference_window_frames": REFERENCE_WINDOW_FRAMES,
            "regression_window_frames": REGRESSION_WINDOW_FRAMES,
            "coordinate_frame": "zed_camera_shoulder_origin",
            "time_units": "seconds",
            "vector_units": "meters",
            "velocity_units": "meters_per_second",
            "filter": {
                "name": "One Euro",
                "min_cutoff": ONE_EURO_MIN_CUTOFF,
                "beta": ONE_EURO_BETA,
                "derivative_cutoff": ONE_EURO_DERIVATIVE_CUTOFF,
            },
        },
        "data": data.tolist(),
    }

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
    temporary_path.replace(output_path)

    return output_path.resolve()


# --------------------------------------------------
# Live 3D Arm Plotting
# --------------------------------------------------

class ArmPlot3D:
    """
    Interactive 3D stick-arm visualization for live ZED tracking.

    Call update(...) once per camera frame with the three vectors returned by
    camera.get_arm_vectors(...). One Euro filtering is performed before
    the artists are updated.
    """

    def __init__(
        self,
        axis_limit=1.0,
        trail_length=45,
        plot_every_n_frames=CAMERA_FPS,
        title="Live ZED BODY_18 Arm Tracking",
    ):
        """
        Create the interactive plot.

        Parameters
        ----------
        axis_limit : float
            Symmetric plot limit in meters around the shoulder.

        trail_length : int
            Number of filtered wrist positions retained in the motion trail.

        plot_every_n_frames : int
            Number of camera frames between plot updates. The default is
            CAMERA_FPS, so a 15 FPS stream redraws once per second.
        """

        if axis_limit <= 0.0:
            raise ValueError("axis_limit must be greater than zero")
        if trail_length < 1:
            raise ValueError("trail_length must be at least one")
        if plot_every_n_frames < 1:
            raise ValueError("plot_every_n_frames must be at least one")

        # Import lazily so data processing can run without a plotting backend.
        import matplotlib.pyplot as plt

        self._plt = plt
        self.filter = OneEuroFilter()
        self.wrist_trail = deque(maxlen=trail_length)
        self.plot_every_n_frames = int(plot_every_n_frames)
        self.frame_count = 0

        self.figure = plt.figure(figsize=(9, 7))
        self.axis = self.figure.add_subplot(111, projection="3d")
        self.axis.set_title(title)
        self.axis.set_xlabel("Camera X: right (m)")
        self.axis.set_ylabel("Camera Y: down (m)")
        self.axis.set_zlabel("Camera Z: forward (m)")
        self.axis.set_xlim(-axis_limit, axis_limit)
        self.axis.set_ylim(-axis_limit, axis_limit)
        self.axis.set_zlim(-axis_limit, axis_limit)
        self.axis.set_box_aspect((1, 1, 1))
        self.axis.grid(True)

        self.arm_line, = self.axis.plot(
            [],
            [],
            [],
            "-o",
            color="tab:blue",
            linewidth=4,
            markersize=8,
            label="Shoulder–elbow–wrist",
        )
        self.shoulder_to_wrist_line, = self.axis.plot(
            [],
            [],
            [],
            "--",
            color="tab:orange",
            linewidth=2,
            label="Shoulder→wrist vector",
        )
        self.trail_line, = self.axis.plot(
            [],
            [],
            [],
            color="tab:green",
            linewidth=1.5,
            alpha=0.7,
            label="Filtered wrist trail",
        )
        self.axis.legend(loc="upper left")

        plt.ion()
        plt.show(block=False)

    @staticmethod
    def _set_line_3d(line, points):
        """Update a Matplotlib 3D line from an array shaped (N, 3)."""

        points = np.asarray(points, dtype=float)
        line.set_data(points[:, 0], points[:, 1])
        line.set_3d_properties(points[:, 2])

    def update(
        self,
        shoulder_to_elbow,
        elbow_to_wrist,
        shoulder_to_wrist,
        pause_s=0.001,
    ):
        """
        One-Euro-filter one frame and periodically update the 3D stick arm.

        Every valid camera frame is filtered. The plot is redrawn only after
        ``plot_every_n_frames`` samples. The shoulder is fixed at [0, 0, 0],
        so all displayed joints are relative to the shoulder.

        Returns
        -------
        np.ndarray, shape (3, 3)
            The three filtered vectors for this camera frame, whether or not
            this frame triggers a plot redraw.
        """

        if any(
            vector is None
            for vector in (
                shoulder_to_elbow,
                elbow_to_wrist,
                shoulder_to_wrist,
            )
        ):
            return None

        current_vectors = np.asarray(
            [
                shoulder_to_elbow,
                elbow_to_wrist,
                shoulder_to_wrist,
            ],
            dtype=float,
        )
        filtered_vectors = self.filter.update(current_vectors)
        self.frame_count += 1

        if self.frame_count % self.plot_every_n_frames != 0:
            return filtered_vectors

        filtered_shoulder_to_elbow = filtered_vectors[0]
        filtered_elbow_to_wrist = filtered_vectors[1]
        filtered_shoulder_to_wrist = filtered_vectors[2]

        shoulder = np.zeros(3, dtype=float)
        elbow = filtered_shoulder_to_elbow
        wrist = elbow + filtered_elbow_to_wrist
        direct_wrist = filtered_shoulder_to_wrist

        self._set_line_3d(
            self.arm_line,
            np.vstack((shoulder, elbow, wrist)),
        )
        self._set_line_3d(
            self.shoulder_to_wrist_line,
            np.vstack((shoulder, direct_wrist)),
        )

        self.wrist_trail.append(wrist.copy())
        self._set_line_3d(
            self.trail_line,
            np.asarray(self.wrist_trail),
        )

        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

        if pause_s > 0.0:
            self._plt.pause(pause_s)

        return filtered_vectors

    def is_open(self):
        """Return True while the interactive plot window exists."""

        return self._plt.fignum_exists(self.figure.number)

    def close(self):
        """Close the plot window."""

        self._plt.close(self.figure)


class ProcessedArmTimeSlider:
    """Time-controlled 3D view of the continuous regression arm trajectory."""

    def __init__(self, processed_blocks, axis_limit=None, axis_padding=0.12):
        if len(processed_blocks) == 0:
            raise ValueError("processed_blocks cannot be empty")
        if axis_limit is not None and axis_limit <= 0.0:
            raise ValueError("axis_limit must be greater than zero")
        if axis_padding < 0.0:
            raise ValueError("axis_padding cannot be negative")

        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        self._plt = plt
        self.processed_blocks = np.asarray(processed_blocks, dtype=float)
        if self.processed_blocks.ndim != 2 or (
            self.processed_blocks.shape[1] != PROCESSED_COLUMN_COUNT
        ):
            raise ValueError(
                f"processed_blocks must have shape "
                f"(N, {PROCESSED_COLUMN_COUNT})"
            )
        if not np.all(np.isfinite(self.processed_blocks)):
            raise ValueError("processed_blocks must contain finite numbers")

        self.motion_blocks = self.processed_blocks[1:]
        self.reference_vectors = self.processed_blocks[
            0,
            LAST_VECTOR_COLUMNS,
        ].reshape(3, 3)
        self.end_time_s = (
            float(self.motion_blocks[-1, END_TIME_COLUMN])
            if len(self.motion_blocks)
            else 0.0
        )

        self.figure = plt.figure(figsize=(10, 8), constrained_layout=False)
        self.axis = self.figure.add_subplot(111, projection="3d")
        self.figure.subplots_adjust(
            left=0.04,
            right=0.96,
            top=0.94,
            bottom=0.16,
        )

        self.axis.set_xlabel("Camera X: right (m)")
        self.axis.set_ylabel("Camera Y: down (m)")
        self.axis.set_zlabel("Camera Z: forward (m)")
        if axis_limit is None:
            self._fit_axis_to_data(axis_padding)
        else:
            self.axis.set_xlim(-axis_limit, axis_limit)
            self.axis.set_ylim(-axis_limit, axis_limit)
            self.axis.set_zlim(-axis_limit, axis_limit)
            self.axis.set_box_aspect((1, 1, 1))
        self.axis.grid(True)

        self.arm_line, = self.axis.plot(
            [],
            [],
            [],
            "-o",
            color="tab:blue",
            linewidth=4,
            markersize=8,
            label="Regression pose",
        )
        self.direct_line, = self.axis.plot(
            [],
            [],
            [],
            "--",
            color="tab:orange",
            linewidth=2,
            label="Shoulder→wrist",
        )
        self.axis.legend(loc="upper left")

        slider_axis = self.figure.add_axes([0.18, 0.055, 0.64, 0.035])
        slider_max = max(self.end_time_s, 1.0 / CAMERA_FPS)
        self.slider = Slider(
            ax=slider_axis,
            label="Time (s)",
            valmin=0.0,
            valmax=slider_max,
            valinit=0.0,
            valstep=1.0 / CAMERA_FPS,
            valfmt="%1.2f s",
        )
        self.slider.on_changed(self.show_time)
        self.show_time(0.0)

    def _fit_axis_to_data(self, padding):
        """Fit equal 3D limits around every processed start/end arm pose."""

        poses = [self._pose_from_vectors(self.reference_vectors)]
        for block in self.motion_blocks:
            poses.append(
                self._pose_from_vectors(
                    block[FIRST_VECTOR_COLUMNS].reshape(3, 3)
                )
            )
            poses.append(
                self._pose_from_vectors(
                    block[LAST_VECTOR_COLUMNS].reshape(3, 3)
                )
            )

        points = np.vstack(poses)
        minimum = np.min(points, axis=0)
        maximum = np.max(points, axis=0)
        center = (minimum + maximum) / 2.0
        span = max(float(np.max(maximum - minimum)), 0.20)
        radius = 0.5 * span * (1.0 + 2.0 * padding)

        self.axis.set_xlim(center[0] - radius, center[0] + radius)
        self.axis.set_ylim(center[1] - radius, center[1] + radius)
        self.axis.set_zlim(center[2] - radius, center[2] + radius)
        self.axis.set_box_aspect((1, 1, 1))

    @staticmethod
    def _pose_from_vectors(vectors):
        """Build shoulder, elbow, wrist positions with shoulder at origin."""

        vectors = np.asarray(vectors, dtype=float).reshape(3, 3)
        shoulder = np.zeros(3, dtype=float)
        elbow = vectors[0]
        wrist = elbow + vectors[1]
        return np.vstack((shoulder, elbow, wrist))

    @staticmethod
    def _set_line_3d(line, points):
        points = np.asarray(points, dtype=float)
        line.set_data(points[:, 0], points[:, 1])
        line.set_3d_properties(points[:, 2])

    def vectors_at_time(self, time_s):
        """Evaluate the connected piecewise-linear regression at a time."""

        time_s = float(np.clip(time_s, 0.0, self.end_time_s))
        if len(self.motion_blocks) == 0:
            return self.reference_vectors.copy()

        end_times = self.motion_blocks[:, END_TIME_COLUMN]
        block_index = min(
            int(np.searchsorted(end_times, time_s, side="left")),
            len(self.motion_blocks) - 1,
        )
        block = self.motion_blocks[block_index]
        start_time_s, end_time_s = block[TIME_COLUMNS]

        if time_s < start_time_s:
            if block_index == 0:
                return self.reference_vectors.copy()
            return self.motion_blocks[
                block_index - 1,
                LAST_VECTOR_COLUMNS,
            ].reshape(3, 3)

        local_time_s = np.clip(
            time_s - start_time_s,
            0.0,
            end_time_s - start_time_s,
        )
        first_vectors = block[FIRST_VECTOR_COLUMNS].reshape(3, 3)
        velocities = block[VELOCITY_COLUMNS].reshape(3, 3)
        return first_vectors + velocities * local_time_s

    def show_time(self, time_s):
        """Display one continuous regression pose at the selected time."""

        time_s = min(float(time_s), self.end_time_s)
        vectors = self.vectors_at_time(time_s)
        pose = self._pose_from_vectors(vectors)
        shoulder = np.zeros(3, dtype=float)

        self._set_line_3d(self.arm_line, pose)
        self._set_line_3d(
            self.direct_line,
            np.vstack((shoulder, vectors[2])),
        )
        self.axis.set_title(f"Processed arm movement | t = {time_s:.2f} s")
        self.figure.canvas.draw_idle()

    def show(self):
        """Open the slider plot and block until the user closes it."""

        self._plt.show()


def run_arm_tracking(
    arm=ARM_SIDE,
    reference_delay_seconds=REFERENCE_DELAY_SECONDS,
    robot_controller=None,
):
    """
    Establish an arm reference, then regress each one-second sample window.

    The operator gets a positioning countdown, followed by a reference window
    of ``REFERENCE_WINDOW_FRAMES`` valid frames. Motion regression uses
    ``REGRESSION_WINDOW_FRAMES`` samples. Press ``r`` at any time to delete
    all collected data and restart reference acquisition. Press ``q`` to end.
    """

    if arm not in ("left", "right"):
        raise ValueError("arm must be 'left' or 'right'")
    if reference_delay_seconds < 0.0:
        raise ValueError("reference_delay_seconds cannot be negative")
    if REFERENCE_WINDOW_FRAMES < 1:
        raise ValueError("REFERENCE_WINDOW_FRAMES must be at least one")
    if REGRESSION_WINDOW_FRAMES < 2:
        raise ValueError("REGRESSION_WINDOW_FRAMES must be at least two")

    zed = None
    body_tracking_enabled = False
    processed_blocks = []
    window_times = []
    window_vectors = []
    arm_filter = OneEuroFilter()
    state = "countdown"
    state_start = time.monotonic()
    recording_start = None

    delete_processed_data()
    print(
        "[REFERENCE] Face the camera and hold your arm straight forward, "
        "with your wrist closer to the camera."
    )

    try:
        zed, runtime_params, image_zed = open_zed()
        body_runtime = setup_body_tracking(zed)
        body_tracking_enabled = True
        bodies = sl.Bodies()
        cv.namedWindow(
            CAMERA_WINDOW_NAME,
            cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO,
        )
        cv.resizeWindow(
            CAMERA_WINDOW_NAME,
            *CAMERA_WINDOW_SIZE,
        )

        print(
            f"[INFO] Tracking closest person's {arm} arm. "
            "Press r to restart or q to stop."
        )

        while True:
            image = get_image(zed, runtime_params, image_zed)
            if image is None:
                continue

            retrieve_status = zed.retrieve_bodies(
                bodies,
                body_runtime,
            )

            if (
                retrieve_status == sl.ERROR_CODE.SUCCESS
                and len(bodies.body_list) > 0
            ):
                body = get_single_body(bodies, mode="closest")
            else:
                body = None

            if body is not None:
                arm_data = get_arm_points(body, arm)
                vectors = get_arm_vectors(body, arm)

                if arm_data is not None:
                    draw_arm_points_and_lines(image, arm_data)

                if all(vector is not None for vector in vectors):
                    current_vectors = np.asarray(vectors, dtype=float)
                    now = time.monotonic()

                    if (
                        state == "countdown"
                        and now - state_start >= reference_delay_seconds
                    ):
                        state = "reference"
                        arm_filter.reset()
                        window_times.clear()
                        window_vectors.clear()
                        print(
                            f"[REFERENCE] Hold still for "
                            f"{REFERENCE_WINDOW_FRAMES} "
                            "valid frames."
                        )

                    if state in ("reference", "recording"):
                        filtered_vectors = arm_filter.update(
                            current_vectors,
                            now,
                        )
                        window_vectors.append(filtered_vectors)

                    if state == "reference":
                        if len(window_vectors) == REFERENCE_WINDOW_FRAMES:
                            reference_block = make_reference_block(
                                window_vectors
                            )
                            processed_blocks[:] = [reference_block]
                            reference_vectors = reference_block[
                                FIRST_VECTOR_COLUMNS
                            ].reshape(3, 3)
                            arm_filter.reset(
                                reference_vectors,
                                time.monotonic(),
                            )
                            window_vectors.clear()
                            window_times.clear()
                            recording_start = time.monotonic()
                            state = "recording"
                            save_processed_blocks(
                                processed_blocks,
                                arm=arm,
                            )
                            print(
                                "[REFERENCE] Reference captured. "
                                "Start moving your arm."
                            )

                    elif state == "recording":
                        elapsed_s = now - recording_start
                        window_times.append(elapsed_s)

                        if len(window_vectors) == REGRESSION_WINDOW_FRAMES:
                            block = synchronize_regression_block(
                                regress_arm_window(
                                    window_times,
                                    window_vectors,
                                ),
                                processed_blocks[-1],
                            )
                            processed_blocks.append(block)
                            if robot_controller is not None:
                                predicted_displacement = (
                                    predict_wrist_displacement(block)
                                )
                                robot_controller.send_predicted_displacement(
                                    predicted_displacement
                                )
                            motion_row_count = len(processed_blocks) - 1
                            if motion_row_count % CAMERA_FPS == 0:
                                output_path = save_processed_blocks(
                                    processed_blocks,
                                    arm=arm,
                                )
                                print(
                                    "[INFO] Processed arm motion through "
                                    f"{block[END_TIME_COLUMN]:.2f} s "
                                    f"saved to {output_path}"
                                )
                            # Retain the newest sample so the next camera
                            # frame forms another two-frame regression window.
                            window_times[:] = window_times[-1:]
                            window_vectors[:] = window_vectors[-1:]

            if state == "countdown":
                remaining = max(
                    0.0,
                    reference_delay_seconds - (time.monotonic() - state_start),
                )
                status_text = (
                    "Arm straight; wrist toward camera. "
                    f"Reference starts in {remaining:.1f}s"
                )
            elif state == "reference":
                status_text = (
                    "Hold still: reference samples "
                    f"{len(window_vectors)}/{REFERENCE_WINDOW_FRAMES}"
                )
            else:
                elapsed_s = time.monotonic() - recording_start
                status_text = f"Move your arm | Time: {elapsed_s:.1f} s"

            cv.putText(
                image,
                status_text,
                (20, 35),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            cv.putText(
                image,
                "R restart and delete data | Q stop",
                (20, 65),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            cv.imshow(CAMERA_WINDOW_NAME, image)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                if robot_controller is not None:
                    robot_controller.reset_to_start()
                delete_processed_data()
                processed_blocks.clear()
                window_times.clear()
                window_vectors.clear()
                arm_filter.reset()
                recording_start = None
                state = "countdown"
                state_start = time.monotonic()
                print(
                    "[RESET] Deleted collected arm data. Face the camera and "
                    "hold your arm straight forward."
                )

    except KeyboardInterrupt:
        print("\n[INFO] Arm tracking stopped.")
    finally:
        cv.destroyAllWindows()

        if zed is not None:
            if body_tracking_enabled:
                zed.disable_body_tracking()
                zed.disable_positional_tracking()
            zed.close()
        if robot_controller is not None:
            robot_controller.stop()

    if processed_blocks:
        output_path = save_processed_blocks(
            processed_blocks,
            arm=arm,
        )
        print(
            f"[INFO] Saved one reference and "
            f"{len(processed_blocks) - 1} processed blocks to {output_path}"
        )

    return processed_blocks


def parse_arguments():
    """Parse optional Lite 6 real-time control settings."""

    parser = argparse.ArgumentParser(
        description="Track a human arm and optionally control a Lite 6."
    )
    parser.add_argument(
        "--lite6-ip",
        help="Enable robot position control using this Lite 6 IP address.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=TELEOP_SCALE,
        help=f"Human-to-robot displacement scale (default: {TELEOP_SCALE}).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=LITE6_COMMAND_SPEED_MM_S,
        help=(
            "Lite 6 Cartesian command speed in mm/s "
            f"(default: {LITE6_COMMAND_SPEED_MM_S})."
        ),
    )
    return parser.parse_args()


def main():
    """Run live tracking/control, then show processed motion by time."""

    args = parse_arguments()
    robot_controller = None

    if args.lite6_ip:
        print(
            "[WARNING] Real-time Lite 6 position control is enabled. "
            f"The robot will first move to {LITE6_START_POSE_MM_DEG.tolist()}."
        )
        if input("Type MOVE to continue: ").strip() != "MOVE":
            print("[INFO] Robot control cancelled.")
            return
        robot_controller = Lite6TeleopController(
            args.lite6_ip,
            scale=args.scale,
            speed_mm_s=args.speed,
        )
        robot_controller.connect_and_move_to_start()

    try:
        processed_blocks = run_arm_tracking(
            arm=ARM_SIDE,
            robot_controller=robot_controller,
        )
    finally:
        if robot_controller is not None:
            robot_controller.disconnect()

    if not processed_blocks:
        print(
            "[WARNING] No complete regression blocks were captured; "
            "there is nothing to plot."
        )
        return

    plot = ProcessedArmTimeSlider(processed_blocks)
    plot.show()


if __name__ == "__main__":
    main()
