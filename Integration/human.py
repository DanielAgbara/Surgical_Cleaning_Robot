#!/usr/bin/env python3
"""Human cleaning-device transforms, marker tracking, and CLI utilities.

Public transform API
--------------------
``get_T_camera_marker`` estimates the cleaning-device marker pose from one
camera image. ``get_T_marker_cleaning_head`` loads the fixed transform from the
marker to the cleaning-head center.

Force calibration and reading live in force_sensor.py so robot and human
sensors share one calibration schema and serial interface. This launcher
inserts the human target into its canonical command line:

    human.py calibrate  ->  force_sensor.py calibrate human
    human.py read       ->  force_sensor.py read human
"""

import sys
import time
import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HUMAN_DATA_DIR = Path(__file__).resolve().parent / "data" / "human"
MARKER_TO_CLEANING_HEAD_FILE = (
    HUMAN_DATA_DIR / "marker_to_cleaning_head.json"
)

# Alternative representation of the same marker origin with Y and Z reversed.
# This is a proper 180-degree X rotation (determinant +1), not a reflection.
T_MARKER_Z_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass(frozen=True)
class PoseQualityConfig:
    """Thresholds for accepting a measured cleaning-device pose."""

    min_markers: int = 4
    min_charuco_corners: int = 4
    max_mean_reprojection_error_px: float = 1.0
    max_corner_reprojection_error_px: float = 2.0
    max_translation_jump_m: float = 0.033
    max_rotation_jump_deg: float = 30.0
    max_linear_speed_m_s: float = 1.0
    max_angular_speed_deg_s: float = 90.0
    max_timestamp_gap_s: float = 0.120
    reacquisition_samples: int = 2
    max_reacquisition_step_m: float = 0.015
    max_reacquisition_step_deg: float = 8.0
    max_reacquisition_translation_m: float = 0.100
    max_reacquisition_rotation_deg: float = 30.0

    def validate(self) -> None:
        if self.min_markers < 1 or self.min_charuco_corners < 4:
            raise ValueError("Marker and ChArUco corner minimums are invalid")
        if self.reacquisition_samples < 2:
            raise ValueError("Reacquisition requires at least two samples")
        numeric_limits = (
            self.max_mean_reprojection_error_px,
            self.max_corner_reprojection_error_px,
            self.max_translation_jump_m,
            self.max_rotation_jump_deg,
            self.max_linear_speed_m_s,
            self.max_angular_speed_deg_s,
            self.max_timestamp_gap_s,
            self.max_reacquisition_step_m,
            self.max_reacquisition_step_deg,
            self.max_reacquisition_translation_m,
            self.max_reacquisition_rotation_deg,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in numeric_limits):
            raise ValueError("All pose-quality limits must be finite and positive")


@dataclass(frozen=True)
class PoseQualityResult:
    """Decision and diagnostic values for one camera pose observation."""

    accepted: bool
    reason: str
    starts_new_segment: bool = False
    T_camera_marker: np.ndarray | None = None
    T_camera_cleaning_head: np.ndarray | None = None
    timestamp_gap_s: float | None = None
    translation_jump_m: float | None = None
    rotation_jump_deg: float | None = None
    linear_speed_m_s: float | None = None
    angular_speed_deg_s: float | None = None


@dataclass(frozen=True)
class PoseFilterConfig:
    """One Euro filter parameters for translation and SO(3) rotation."""

    translation_min_cutoff_hz: float = 0.5
    translation_beta: float = 0.05
    rotation_min_cutoff_hz: float = 0.5
    rotation_beta: float = 0.05
    derivative_cutoff_hz: float = 1.0

    def validate(self) -> None:
        cutoffs = (
            self.translation_min_cutoff_hz,
            self.rotation_min_cutoff_hz,
            self.derivative_cutoff_hz,
        )
        betas = (self.translation_beta, self.rotation_beta)
        if not all(np.isfinite(value) and value > 0.0 for value in cutoffs):
            raise ValueError("Pose-filter cutoffs must be finite and positive")
        if not all(np.isfinite(value) and value >= 0.0 for value in betas):
            raise ValueError("Pose-filter beta values must be finite and nonnegative")


@dataclass(frozen=True)
class PoseTrajectorySample:
    """One timestamped marker pose suitable for reuse by Phase 2.

    ``status`` is ``"measured"`` for a camera measurement or
    ``"interpolated"`` for a pose inserted across a short detection gap.
    ``starts_new_segment`` prevents consumers from connecting long gaps.
    """

    timestamp_s: float
    T_camera_marker: np.ndarray
    status: str = "measured"
    starts_new_segment: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestamp_s):
            raise ValueError("Trajectory timestamp must be finite")
        if self.status not in {"measured", "interpolated"}:
            raise ValueError("Pose status must be 'measured' or 'interpolated'")
        transform = _validate_rigid_transform(
            self.T_camera_marker, "trajectory T_camera_marker"
        )
        object.__setattr__(self, "T_camera_marker", transform)


def _low_pass_alpha(cutoff_hz: float, elapsed_s: float) -> float:
    """Return exponential smoothing alpha for a cutoff and timestep."""
    time_constant = 1.0 / (2.0 * np.pi * cutoff_hz)
    return float(elapsed_s / (elapsed_s + time_constant))


def _rotation_log(rotation: np.ndarray) -> np.ndarray:
    """Map an SO(3) rotation matrix to its axis-angle vector."""
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    skew_vector = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=float,
    )
    if angle < 1e-8:
        return 0.5 * skew_vector
    if np.pi - angle < 1e-5:
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        axis_index = int(np.argmin(np.abs(eigenvalues - 1.0)))
        axis = np.real(eigenvectors[:, axis_index])
        axis /= np.linalg.norm(axis)
        return angle * axis
    return angle * skew_vector / (2.0 * np.sin(angle))


def _rotation_exp(rotation_vector: np.ndarray) -> np.ndarray:
    """Map an axis-angle vector to an SO(3) rotation matrix."""
    rotation_vector = np.asarray(rotation_vector, dtype=float)
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1e-12:
        return np.eye(3, dtype=float)
    axis = rotation_vector / angle
    axis_skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3)
        + np.sin(angle) * axis_skew
        + (1.0 - np.cos(angle)) * (axis_skew @ axis_skew)
    )


def interpolate_pose(first_transform, second_transform, fraction: float) -> np.ndarray:
    """Interpolate one rigid pose without interpolating matrix entries.

    Translation is linear. Rotation follows the shortest geodesic on SO(3),
    which is equivalent to quaternion SLERP for the selected rotation branch.
    """
    first = _validate_rigid_transform(first_transform, "first_transform")
    second = _validate_rigid_transform(second_transform, "second_transform")
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("Interpolation fraction must be between 0 and 1")

    interpolated = np.eye(4, dtype=float)
    interpolated[:3, 3] = (
        (1.0 - fraction) * first[:3, 3]
        + fraction * second[:3, 3]
    )
    rotation_step = _rotation_log(first[:3, :3].T @ second[:3, :3])
    interpolated[:3, :3] = (
        first[:3, :3] @ _rotation_exp(fraction * rotation_step)
    )
    return _validate_rigid_transform(interpolated, "interpolated_pose")


def fill_short_pose_gaps(
    measured_samples,
    target_frequency_hz: float = 30.0,
    max_interpolation_gap_s: float = 0.100,
) -> list[PoseTrajectorySample]:
    """Fill short timestamp gaps and preserve long gaps as new segments.

    This function has no camera or plotting dependencies and is intended for
    direct use by ``phase2.py``. Input poses must already have passed quality
    rejection and filtering. No extrapolation is performed at either end.
    """
    if not np.isfinite(target_frequency_hz) or target_frequency_hz <= 0.0:
        raise ValueError("Target frequency must be finite and positive")
    if (
        not np.isfinite(max_interpolation_gap_s)
        or max_interpolation_gap_s <= 0.0
    ):
        raise ValueError("Maximum interpolation gap must be finite and positive")

    samples = list(measured_samples)
    if not samples:
        return []
    if not all(isinstance(sample, PoseTrajectorySample) for sample in samples):
        raise TypeError("All trajectory samples must be PoseTrajectorySample")

    nominal_period_s = 1.0 / target_frequency_hz
    output = [
        PoseTrajectorySample(
            samples[0].timestamp_s,
            samples[0].T_camera_marker,
            samples[0].status,
            True,
        )
    ]

    for previous, current in zip(samples, samples[1:]):
        gap_s = current.timestamp_s - previous.timestamp_s
        if gap_s <= 0.0:
            raise ValueError("Trajectory timestamps must be strictly increasing")

        # A segment explicitly created by reacquisition is never interpolated.
        long_gap = (
            current.starts_new_segment
            or gap_s > max_interpolation_gap_s
        )
        if long_gap:
            output.append(
                PoseTrajectorySample(
                    current.timestamp_s,
                    current.T_camera_marker,
                    current.status,
                    True,
                )
            )
            continue

        # Round to the most likely number of camera periods. This avoids
        # inventing a point for ordinary frame-timing jitter.
        missing_count = max(0, int(round(gap_s / nominal_period_s)) - 1)
        for missing_index in range(1, missing_count + 1):
            fraction = missing_index / (missing_count + 1)
            output.append(
                PoseTrajectorySample(
                    previous.timestamp_s + fraction * gap_s,
                    interpolate_pose(
                        previous.T_camera_marker,
                        current.T_camera_marker,
                        fraction,
                    ),
                    "interpolated",
                    False,
                )
            )
        output.append(
            PoseTrajectorySample(
                current.timestamp_s,
                current.T_camera_marker,
                current.status,
                False,
            )
        )
    return output


class PoseOneEuroFilter:
    """Filter pose translation and rotation without leaving SE(3)."""

    def __init__(self, config: PoseFilterConfig | None = None) -> None:
        self.config = config or PoseFilterConfig()
        self.config.validate()
        self.reset()

    def reset(self) -> None:
        self.previous_timestamp = None
        self.previous_raw_translation = None
        self.previous_filtered_translation = None
        self.previous_translation_velocity = np.zeros(3, dtype=float)
        self.previous_raw_rotation = None
        self.previous_filtered_rotation = None
        self.previous_angular_velocity = np.zeros(3, dtype=float)

    def update(self, transform, timestamp: float) -> np.ndarray:
        """Return a filtered pose; the first pose after reset passes through."""
        transform = _validate_rigid_transform(transform, "pose_filter_input")
        if not np.isfinite(timestamp):
            raise ValueError("Pose-filter timestamp must be finite")
        translation = transform[:3, 3]
        rotation = transform[:3, :3]

        if self.previous_timestamp is None:
            filtered_translation = translation.copy()
            filtered_rotation = rotation.copy()
        else:
            elapsed_s = timestamp - self.previous_timestamp
            if elapsed_s <= 0.0:
                raise ValueError("Pose-filter timestamps must increase")
            derivative_alpha = _low_pass_alpha(
                self.config.derivative_cutoff_hz, elapsed_s
            )

            raw_velocity = (
                translation - self.previous_raw_translation
            ) / elapsed_s
            filtered_velocity = (
                derivative_alpha * raw_velocity
                + (1.0 - derivative_alpha) * self.previous_translation_velocity
            )
            translation_cutoff = (
                self.config.translation_min_cutoff_hz
                + self.config.translation_beta * np.linalg.norm(filtered_velocity)
            )
            translation_alpha = _low_pass_alpha(translation_cutoff, elapsed_s)
            filtered_translation = (
                translation_alpha * translation
                + (1.0 - translation_alpha)
                * self.previous_filtered_translation
            )

            raw_rotation_step = _rotation_log(
                self.previous_raw_rotation.T @ rotation
            )
            raw_angular_velocity = raw_rotation_step / elapsed_s
            filtered_angular_velocity = (
                derivative_alpha * raw_angular_velocity
                + (1.0 - derivative_alpha) * self.previous_angular_velocity
            )
            rotation_cutoff = (
                self.config.rotation_min_cutoff_hz
                + self.config.rotation_beta
                * np.linalg.norm(filtered_angular_velocity)
            )
            rotation_alpha = _low_pass_alpha(rotation_cutoff, elapsed_s)
            filtered_rotation_step = _rotation_log(
                self.previous_filtered_rotation.T @ rotation
            )
            filtered_rotation = (
                self.previous_filtered_rotation
                @ _rotation_exp(rotation_alpha * filtered_rotation_step)
            )
            self.previous_translation_velocity = filtered_velocity
            self.previous_angular_velocity = filtered_angular_velocity

        self.previous_timestamp = timestamp
        self.previous_raw_translation = translation.copy()
        self.previous_filtered_translation = filtered_translation.copy()
        self.previous_raw_rotation = rotation.copy()
        self.previous_filtered_rotation = filtered_rotation.copy()

        filtered_transform = np.eye(4, dtype=float)
        filtered_transform[:3, :3] = filtered_rotation
        filtered_transform[:3, 3] = filtered_translation
        return _validate_rigid_transform(filtered_transform, "filtered_pose")


class PoseQualityGate:
    """Reject bad poses and cautiously reacquire tracking after a gap.

    During normal tracking, every measurement is compared with the previous
    accepted cleaning-head pose. After a gap, several consecutive candidate
    poses must agree with each other and with the last reliable pose before
    tracking is restarted. Candidate poses never enter the trajectory or pose
    filter until reacquisition succeeds.
    """

    def __init__(
        self,
        T_marker_cleaning_head,
        config: PoseQualityConfig | None = None,
    ) -> None:
        self.T_marker_cleaning_head = _validate_rigid_transform(
            T_marker_cleaning_head, "T_marker_cleaning_head"
        )
        self.config = config or PoseQualityConfig()
        self.config.validate()
        self.previous_timestamp = None
        self.previous_T_camera_cleaning_head = None
        self._clear_reacquisition()

    @staticmethod
    def _rotation_difference_deg(first_rotation, second_rotation) -> float:
        relative_rotation = first_rotation.T @ second_rotation
        cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.rad2deg(np.arccos(cosine)))

    def _clear_reacquisition(self) -> None:
        self.reacquisition_count = 0
        self.reacquisition_timestamp = None
        self.reacquisition_T_camera_cleaning_head = None

    def reject(
        self,
        reason: str,
        keep_reacquisition: bool = False,
        **metrics,
    ) -> PoseQualityResult:
        if not keep_reacquisition:
            self._clear_reacquisition()
        return PoseQualityResult(False, reason, **metrics)

    def _evaluate_reacquisition(
        self,
        T_camera_marker: np.ndarray,
        T_camera_cleaning_head: np.ndarray,
        timestamp: float,
        timestamp_gap: float,
    ) -> PoseQualityResult:
        """Withhold candidates until a stable, plausible pose is confirmed."""
        candidate_number = self.reacquisition_count + 1

        if self.reacquisition_count > 0:
            candidate_dt = timestamp - self.reacquisition_timestamp
            candidate_translation = float(
                np.linalg.norm(
                    T_camera_cleaning_head[:3, 3]
                    - self.reacquisition_T_camera_cleaning_head[:3, 3]
                )
            )
            candidate_rotation = self._rotation_difference_deg(
                self.reacquisition_T_camera_cleaning_head[:3, :3],
                T_camera_cleaning_head[:3, :3],
            )
            candidates_disagree = (
                candidate_dt <= 0.0
                or candidate_dt > self.config.max_timestamp_gap_s
                or candidate_translation
                > self.config.max_reacquisition_step_m
                or candidate_rotation
                > self.config.max_reacquisition_step_deg
            )
            if candidates_disagree:
                # The current pose becomes candidate 1 of a new sequence.
                self.reacquisition_count = 1
                self.reacquisition_timestamp = timestamp
                self.reacquisition_T_camera_cleaning_head = (
                    T_camera_cleaning_head.copy()
                )
                return self.reject(
                    "reacquisition_inconsistent",
                    keep_reacquisition=True,
                    timestamp_gap_s=timestamp_gap,
                    translation_jump_m=candidate_translation,
                    rotation_jump_deg=candidate_rotation,
                )

        self.reacquisition_count = candidate_number
        self.reacquisition_timestamp = timestamp
        self.reacquisition_T_camera_cleaning_head = (
            T_camera_cleaning_head.copy()
        )

        if self.reacquisition_count < self.config.reacquisition_samples:
            return self.reject(
                "reacquiring_"
                f"{self.reacquisition_count}_of_"
                f"{self.config.reacquisition_samples}",
                keep_reacquisition=True,
                timestamp_gap_s=timestamp_gap,
            )

        # A stable planar solution can still be the wrong pose branch. Compare
        # the confirmed candidate with the last pose trusted before dropout.
        reference_translation = float(
            np.linalg.norm(
                T_camera_cleaning_head[:3, 3]
                - self.previous_T_camera_cleaning_head[:3, 3]
            )
        )
        reference_rotation = self._rotation_difference_deg(
            self.previous_T_camera_cleaning_head[:3, :3],
            T_camera_cleaning_head[:3, :3],
        )
        if reference_translation > self.config.max_reacquisition_translation_m:
            return self.reject(
                "reacquisition_reference_translation",
                timestamp_gap_s=timestamp_gap,
                translation_jump_m=reference_translation,
                rotation_jump_deg=reference_rotation,
            )
        if reference_rotation > self.config.max_reacquisition_rotation_deg:
            return self.reject(
                "reacquisition_reference_rotation",
                timestamp_gap_s=timestamp_gap,
                translation_jump_m=reference_translation,
                rotation_jump_deg=reference_rotation,
            )

        self._clear_reacquisition()
        self.previous_timestamp = timestamp
        self.previous_T_camera_cleaning_head = T_camera_cleaning_head.copy()
        return PoseQualityResult(
            True,
            "reacquisition_confirmed",
            starts_new_segment=True,
            T_camera_marker=T_camera_marker,
            T_camera_cleaning_head=T_camera_cleaning_head,
            timestamp_gap_s=timestamp_gap,
            translation_jump_m=reference_translation,
            rotation_jump_deg=reference_rotation,
        )

    def evaluate(self, detection, pose, timestamp: float) -> PoseQualityResult:
        """Validate one measurement and update history only when accepted."""
        if not np.isfinite(timestamp):
            return self.reject("invalid_timestamp")
        if detection.num_markers < self.config.min_markers:
            return self.reject("insufficient_markers")
        if detection.num_charuco_corners < self.config.min_charuco_corners:
            return self.reject("insufficient_charuco_corners")
        if detection.corners_are_collinear:
            return self.reject("collinear_charuco_corners")
        if pose is None:
            return self.reject("pose_unavailable")
        if (
            not np.isfinite(pose.mean_reprojection_error_px)
            or pose.mean_reprojection_error_px
            > self.config.max_mean_reprojection_error_px
        ):
            return self.reject("mean_reprojection_error")
        if (
            not np.isfinite(pose.max_reprojection_error_px)
            or pose.max_reprojection_error_px
            > self.config.max_corner_reprojection_error_px
        ):
            return self.reject("corner_reprojection_error")

        try:
            T_camera_marker = _make_marker_z_point_into_board(
                pose.T_camera_board
            )
            if T_camera_marker[2, 3] <= 0.0:
                return self.reject("marker_behind_camera")
            inward_score = float(
                np.dot(T_camera_marker[:3, 2], T_camera_marker[:3, 3])
            )
            if inward_score <= 0.0:
                return self.reject("marker_z_not_inward")
            T_camera_cleaning_head = _validate_rigid_transform(
                T_camera_marker @ self.T_marker_cleaning_head,
                "T_camera_cleaning_head",
            )
        except (ValueError, RuntimeError):
            return self.reject("invalid_rigid_transform")

        if self.previous_timestamp is None:
            self.previous_timestamp = timestamp
            self.previous_T_camera_cleaning_head = (
                T_camera_cleaning_head.copy()
            )
            return PoseQualityResult(
                True,
                "accepted_initial_pose",
                starts_new_segment=True,
                T_camera_marker=T_camera_marker,
                T_camera_cleaning_head=T_camera_cleaning_head,
            )
        else:
            timestamp_gap = timestamp - self.previous_timestamp
            if timestamp_gap <= 0.0:
                return self.reject(
                    "non_monotonic_timestamp",
                    timestamp_gap_s=timestamp_gap,
                )
            if timestamp_gap > self.config.max_timestamp_gap_s:
                return self._evaluate_reacquisition(
                    T_camera_marker,
                    T_camera_cleaning_head,
                    timestamp,
                    timestamp_gap,
                )

        metrics = {"timestamp_gap_s": timestamp_gap}
        previous = self.previous_T_camera_cleaning_head
        translation_jump = float(
            np.linalg.norm(
                T_camera_cleaning_head[:3, 3] - previous[:3, 3]
            )
        )
        rotation_jump = self._rotation_difference_deg(
            previous[:3, :3], T_camera_cleaning_head[:3, :3]
        )
        linear_speed = translation_jump / timestamp_gap
        angular_speed = rotation_jump / timestamp_gap
        metrics.update(
            translation_jump_m=translation_jump,
            rotation_jump_deg=rotation_jump,
            linear_speed_m_s=linear_speed,
            angular_speed_deg_s=angular_speed,
        )
        if translation_jump > self.config.max_translation_jump_m:
            return self.reject("translation_jump", **metrics)
        if rotation_jump > self.config.max_rotation_jump_deg:
            return self.reject("rotation_jump", **metrics)
        if linear_speed > self.config.max_linear_speed_m_s:
            return self.reject("linear_speed", **metrics)
        if angular_speed > self.config.max_angular_speed_deg_s:
            return self.reject("angular_speed", **metrics)

        self.previous_timestamp = timestamp
        self.previous_T_camera_cleaning_head = T_camera_cleaning_head.copy()
        return PoseQualityResult(
            True,
            "accepted",
            starts_new_segment=False,
            T_camera_marker=T_camera_marker,
            T_camera_cleaning_head=T_camera_cleaning_head,
            **metrics,
        )


def _validate_rigid_transform(transform, name: str) -> np.ndarray:
    """Return a validated copy of one 4x4 rigid transformation."""
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} has an invalid last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
        raise ValueError(f"{name} rotation determinant is not +1")
    return transform.copy()


def get_T_marker_cleaning_head(
    path: Path = MARKER_TO_CLEANING_HEAD_FILE,
) -> np.ndarray:
    """Load the fixed pose of the cleaning head in the marker frame.

    The returned matrix maps cleaning-head-frame coordinates into the marker
    frame: ``p_marker = T_marker_cleaning_head @ p_cleaning_head``.
    """
    import json

    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError(f"{path} does not contain a data matrix")
    return _validate_rigid_transform(
        payload["data"], "T_marker_cleaning_head"
    )


def _estimate_camera_marker(
    image_bgr,
    board,
    detector,
    camera_matrix,
    dist_coeffs=None,
):
    """Return the ChArUco detection and pose used by the public getter."""
    from calibration import detect_charuco_board, estimate_charuco_pose

    detection = detect_charuco_board(image_bgr, board, detector)
    pose = estimate_charuco_pose(
        detection,
        board,
        camera_matrix,
        dist_coeffs,
    )
    return detection, pose


def _make_marker_z_point_into_board(T_camera_marker) -> np.ndarray:
    """Return the equivalent marker frame whose +Z points into the board.

    For the visible printed face, inward points generally away from the camera.
    Thus the camera-frame marker Z axis must have a positive dot product with
    the camera-to-marker translation vector. If it does not, rotate the marker
    frame 180 degrees about its X axis, flipping Y and Z together.
    """
    transform = _validate_rigid_transform(
        T_camera_marker, "T_camera_marker_raw"
    )
    camera_to_marker = transform[:3, 3]
    distance = np.linalg.norm(camera_to_marker)
    if distance < 1e-9:
        raise ValueError("Marker position is too close to the camera origin")

    if np.dot(transform[:3, 2], camera_to_marker) <= 0.0:
        transform = transform @ T_MARKER_Z_FLIP

    transform = _validate_rigid_transform(transform, "T_camera_marker")
    inward_score = float(np.dot(transform[:3, 2], transform[:3, 3]))
    if inward_score <= 0.0:
        raise RuntimeError("Marker +Z axis does not point into the board")
    return transform


def get_T_camera_marker(
    image_bgr,
    board,
    detector,
    camera_matrix,
    dist_coeffs=None,
) -> np.ndarray | None:
    """Estimate the marker pose in the camera frame from one BGR image.

    Returns ``None`` when the ChArUco observation cannot support a pose.
    Otherwise the returned matrix maps marker-frame coordinates into the
    camera frame: ``p_camera = T_camera_marker @ p_marker``.
    """
    _, pose = _estimate_camera_marker(
        image_bgr,
        board,
        detector,
        camera_matrix,
        dist_coeffs,
    )
    if pose is None:
        return None
    return _make_marker_z_point_into_board(pose.T_camera_board)


def save_marker_to_cleaning_head_transform(
    x_mm: float,
    y_mm: float,
    z_mm: float,
    rotation_deg: float,
) -> Path:
    """Build and save the adjustable cleaning-head transform once."""
    import json

    from robot import Rx, inv_SE3

    translation_mm = np.asarray([x_mm, y_mm, z_mm], dtype=float)
    if not np.all(np.isfinite(translation_mm)):
        raise ValueError("Cleaning-head XYZ offsets must be finite")
    if not np.isfinite(rotation_deg):
        raise ValueError("Cleaning-head rotation must be finite")

    # In the base orientation, +X is right, +Y is back, and +Z is down.
    # Flipping both Y and Z with Rx(pi) preserves a proper right-handed SO(3)
    # frame. Apply the
    # additional cleaning-to-marker rotation about marker X. Positive rotation
    # is counterclockwise when viewed from marker +X toward the marker origin.
    rotation_cleaning_head_marker = Rx(
        np.deg2rad(rotation_deg)
    ) @ Rx(np.pi)

    if not np.allclose(
        rotation_cleaning_head_marker.T @ rotation_cleaning_head_marker,
        np.eye(3),
        atol=1e-8,
    ):
        raise RuntimeError("Cleaning-head rotation is not orthonormal")
    if not np.isclose(
        np.linalg.det(rotation_cleaning_head_marker), 1.0, atol=1e-8
    ):
        raise RuntimeError("Cleaning-head rotation determinant is not +1")

    # The CAD translation locates the marker origin in the cleaning-head
    # frame, so first construct T_cleaning_head_marker. The runtime transform
    # needed by the camera chain is its inverse, T_marker_cleaning_head.
    T_cleaning_head_marker = np.eye(4, dtype=float)
    T_cleaning_head_marker[:3, :3] = rotation_cleaning_head_marker
    T_cleaning_head_marker[:3, 3] = translation_mm / 1000.0
    transform = inv_SE3(T_cleaning_head_marker)

    payload = {
        "description": {
            "summary": (
                "Fixed pose of the cleaning-head center in the detected "
                "ChArUco marker frame."
            ),
            "transform": "T_marker_cleaning_head",
            "mapping": "p_marker = T_marker_cleaning_head @ p_cleaning_head",
            "matrix_layout": "4x4 homogeneous transform",
            "translation_units": "meters",
            "axis_definition": (
                "Cleaning +X right, with +Y back and +Z down before the "
                "additional X rotation; positive X rotation is counterclockwise "
                "when viewed from marker +X toward the marker origin."
            ),
            "construction": (
                "T_marker_cleaning_head = inverse([Rx(rotation_deg) @ "
                "Rx(pi), translation_cleaning_head_to_marker])"
            ),
        },
        "data": transform.tolist(),
    }

    MARKER_TO_CLEANING_HEAD_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = MARKER_TO_CLEANING_HEAD_FILE.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(MARKER_TO_CLEANING_HEAD_FILE)
    print(f"Saved marker-to-cleaning-head transform: {MARKER_TO_CLEANING_HEAD_FILE}")
    print(
        "Cleaning-head-to-marker translation [mm]: "
        f"{x_mm:.3f}, {y_mm:.3f}, {z_mm:.3f}\n"
        f"Counterclockwise X rotation [deg]: {rotation_deg:.3f}"
    )
    return MARKER_TO_CLEANING_HEAD_FILE


def configure_marker_to_cleaning_head_transform(arguments: list[str]) -> None:
    """Parse adjustable CAD offsets for the one-time transform writer."""
    parser = argparse.ArgumentParser(
        prog="python3 Integration/human.py set-tool-transform",
        description=(
            "Save T_marker_cleaning_head by inverting the adjustable CAD "
            "cleaning-head-to-marker transform."
        ),
    )
    parser.add_argument("--x-mm", type=float, default=-50.0)
    parser.add_argument("--y-mm", type=float, default=-60.0)
    parser.add_argument("--z-mm", type=float, default=-109.0)
    parser.add_argument(
        "--rotation-deg",
        type=float,
        default=150.0,
        help=(
            "Counterclockwise X rotation viewed from marker +X toward the origin, "
            "default: %(default)s"
        ),
    )
    args = parser.parse_args(arguments)
    save_marker_to_cleaning_head_transform(
        args.x_mm,
        args.y_mm,
        args.z_mm,
        args.rotation_deg,
    )


def plot_marker_to_cleaning_head_transform() -> None:
    """Plot the marker relative to the cleaning-head frame at the origin."""
    import matplotlib.pyplot as plt

    T_marker_cleaning_head = get_T_marker_cleaning_head()
    rotation = T_marker_cleaning_head[:3, :3]
    T_cleaning_head_marker = np.eye(4, dtype=float)
    T_cleaning_head_marker[:3, :3] = rotation.T
    T_cleaning_head_marker[:3, 3] = (
        -rotation.T @ T_marker_cleaning_head[:3, 3]
    )
    cleaning_head_transform = np.eye(4, dtype=float)

    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(111, projection="3d")

    def draw_frame(frame: np.ndarray, label: str, axis_length: float) -> None:
        origin = frame[:3, 3]
        colors = ("red", "green", "blue")
        names = ("X", "Y", "Z")
        for index, (color, name) in enumerate(zip(colors, names)):
            endpoint = origin + axis_length * frame[:3, index]
            axis.plot(
                [origin[0], endpoint[0]],
                [origin[1], endpoint[1]],
                [origin[2], endpoint[2]],
                color=color,
                linewidth=2.5,
            )
            axis.text(*endpoint, f"{label} {name}", color=color)
        axis.scatter(*origin, color="black", s=35)
        axis.text(*origin, label, color="black")

    draw_frame(cleaning_head_transform, "Cleaning head", 0.04)
    draw_frame(T_cleaning_head_marker, "Marker", 0.04)

    cleaning_head_origin = cleaning_head_transform[:3, 3]
    marker_origin = T_cleaning_head_marker[:3, 3]
    axis.plot(
        [marker_origin[0], cleaning_head_origin[0]],
        [marker_origin[1], cleaning_head_origin[1]],
        [marker_origin[2], cleaning_head_origin[2]],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Cleaning-head-to-marker offset",
    )

    plotted_points = np.vstack(
        [
            marker_origin,
            cleaning_head_origin,
            marker_origin + 0.04 * T_cleaning_head_marker[:3, :3].T,
            cleaning_head_origin + 0.04 * cleaning_head_transform[:3, :3].T,
        ]
    )
    center = plotted_points.mean(axis=0)
    half_range = max(np.ptp(plotted_points, axis=0).max() / 2.0, 0.06)
    axis.set_xlim(center[0] - half_range, center[0] + half_range)
    axis.set_ylim(center[1] - half_range, center[1] + half_range)
    axis.set_zlim(center[2] - half_range, center[2] + half_range)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("Cleaning-head X (m)")
    axis.set_ylabel("Cleaning-head Y (m)")
    axis.set_zlabel("Cleaning-head Z (m)")
    axis.set_title("Marker pose in the cleaning-head frame")
    axis.legend()
    axis.grid(True, alpha=0.35)
    figure.tight_layout()
    plt.show()


def test_cleaning_device_marker() -> None:
    """Display and report the cleaning-device ChArUco pose from the ZED."""
    import cv2

    from calibration import (
        HUMAN_TOOL_CHARUCO_CONFIG,
        create_charuco_detector,
    )
    from camera import (
        get_image,
        get_zed_left_intrinsics_rectified,
        open_zed,
    )

    window_name = "Cleaning-device ChArUco detection"
    zed = None
    try:
        T_marker_cleaning_head = get_T_marker_cleaning_head()
        zed, runtime_params, image_zed = open_zed()
        camera_matrix, dist_coeffs = get_zed_left_intrinsics_rectified(zed)
        board, detector = create_charuco_detector(
            HUMAN_TOOL_CHARUCO_CONFIG,
            camera_matrix,
            dist_coeffs,
        )
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print("Press Q or Escape to stop.\n")

        next_print_time = 0.0
        while True:
            image = get_image(zed, runtime_params, image_zed)
            if image is None:
                continue

            detection, pose = _estimate_camera_marker(
                image,
                board,
                detector,
                camera_matrix,
                dist_coeffs,
            )
            display = image.copy()

            if detection.marker_ids is not None:
                cv2.aruco.drawDetectedMarkers(
                    display,
                    detection.marker_corners,
                    detection.marker_ids,
                )
            if detection.charuco_ids is not None:
                cv2.aruco.drawDetectedCornersCharuco(
                    display,
                    detection.charuco_corners,
                    detection.charuco_ids,
                )

            if pose is None:
                status = (
                    f"No pose | markers {detection.num_markers}/4 | "
                    f"corners {detection.num_charuco_corners}/4"
                )
                color = (0, 0, 255)
            else:
                T_camera_marker = _make_marker_z_point_into_board(
                    pose.T_camera_board
                )
                T_camera_cleaning_head = (
                    T_camera_marker @ T_marker_cleaning_head
                )
                cleaning_head_rvec, _ = cv2.Rodrigues(
                    T_camera_cleaning_head[:3, :3]
                )
                marker_rvec, _ = cv2.Rodrigues(
                    T_camera_marker[:3, :3]
                )
                cv2.drawFrameAxes(
                    display,
                    camera_matrix,
                    dist_coeffs,
                    marker_rvec,
                    T_camera_marker[:3, 3],
                    0.025,
                )
                cv2.drawFrameAxes(
                    display,
                    camera_matrix,
                    dist_coeffs,
                    cleaning_head_rvec,
                    T_camera_cleaning_head[:3, 3],
                    0.040,
                    3,
                )
                status = (
                    f"Tracking | markers {detection.num_markers}/4 | "
                    f"corners {detection.num_charuco_corners}/4 | "
                    f"error {pose.mean_reprojection_error_px:.2f} px"
                )
                color = (0, 255, 0)

                now = time.monotonic()
                if now >= next_print_time:
                    print(
                        "T_camera_marker:\n"
                        + "\n".join(
                            "  " + " ".join(f"{value: .6f}" for value in row)
                            for row in T_camera_marker
                        )
                        + f"\nMean reprojection error: "
                        f"{pose.mean_reprojection_error_px:.3f} px\n"
                        + "T_camera_cleaning_head:\n"
                        + "\n".join(
                            "  " + " ".join(f"{value: .6f}" for value in row)
                            for row in T_camera_cleaning_head
                        )
                        + "\n"
                    )
                    next_print_time = now + 1.0

            cv2.putText(
                display,
                status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, display)
            key = cv2.waitKeyEx(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        if zed is not None:
            zed.close()


def plot_camera_frame_trajectories(
    T_camera_marker_samples,
    T_camera_cleaning_head_samples,
    segment_start_flags=None,
) -> None:
    """Plot marker and cleaning-head paths expressed in the camera frame."""
    import matplotlib.pyplot as plt

    marker_transforms = np.asarray(T_camera_marker_samples, dtype=float)
    cleaning_transforms = np.asarray(
        T_camera_cleaning_head_samples, dtype=float
    )
    if marker_transforms.ndim != 3 or marker_transforms.shape[1:] != (4, 4):
        raise ValueError("Marker trajectory must have shape (N, 4, 4)")
    if cleaning_transforms.shape != marker_transforms.shape:
        raise ValueError("Marker and cleaning-head trajectories must match")
    if len(marker_transforms) == 0:
        raise ValueError("At least one valid trajectory sample is required")

    marker_positions = marker_transforms[:, :3, 3]
    cleaning_positions = cleaning_transforms[:, :3, 3]
    if segment_start_flags is None:
        segment_start_flags = np.zeros(len(marker_transforms), dtype=bool)
        segment_start_flags[0] = True
    else:
        segment_start_flags = np.asarray(segment_start_flags, dtype=bool)
        if segment_start_flags.shape != (len(marker_transforms),):
            raise ValueError("Segment flags must have shape (N,)")
        segment_start_flags[0] = True

    # NaNs break Matplotlib lines, preventing a false straight connection
    # across a long detection/timestamp gap.
    def break_path_at_segments(positions):
        plotting_positions = []
        for index, position in enumerate(positions):
            if index > 0 and segment_start_flags[index]:
                plotting_positions.append(np.full(3, np.nan))
            plotting_positions.append(position)
        return np.asarray(plotting_positions)

    marker_plot_positions = break_path_at_segments(marker_positions)
    cleaning_plot_positions = break_path_at_segments(cleaning_positions)

    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    plotted_points = [np.zeros(3), marker_positions, cleaning_positions]

    def draw_frame(
        transform,
        label,
        axis_length,
        linestyle="-",
        alpha=1.0,
    ) -> None:
        transform = _validate_rigid_transform(transform, label)
        origin = transform[:3, 3]
        colors = ("red", "green", "blue")
        axis_names = ("X", "Y", "Z")
        endpoints = []
        for index, (color, axis_name) in enumerate(
            zip(colors, axis_names)
        ):
            endpoint = origin + axis_length * transform[:3, index]
            endpoints.append(endpoint)
            axis.plot(
                [origin[0], endpoint[0]],
                [origin[1], endpoint[1]],
                [origin[2], endpoint[2]],
                color=color,
                linestyle=linestyle,
                linewidth=2.2,
                alpha=alpha,
            )
            axis.text(
                *endpoint,
                f"{label} {axis_name}",
                color=color,
                fontsize=8,
                alpha=alpha,
            )
        axis.scatter(*origin, color="black", s=25, alpha=alpha)
        plotted_points.append(np.asarray(endpoints))

    camera_transform = np.eye(4, dtype=float)
    draw_frame(camera_transform, "Camera", 0.06)

    axis.plot(
        marker_plot_positions[:, 0],
        marker_plot_positions[:, 1],
        marker_plot_positions[:, 2],
        color="tab:blue",
        linewidth=2.0,
        label="Marker path",
    )
    axis.plot(
        cleaning_plot_positions[:, 0],
        cleaning_plot_positions[:, 1],
        cleaning_plot_positions[:, 2],
        color="tab:orange",
        linewidth=2.0,
        label="Cleaning-head path",
    )

    draw_frame(
        marker_transforms[0],
        "Marker start",
        0.035,
        linestyle="--",
        alpha=0.75,
    )
    draw_frame(marker_transforms[-1], "Marker end", 0.035)
    draw_frame(
        cleaning_transforms[0],
        "Head start",
        0.035,
        linestyle="--",
        alpha=0.75,
    )
    draw_frame(cleaning_transforms[-1], "Head end", 0.035)

    for sample_index, label in ((0, "Start offset"), (-1, "End offset")):
        marker_position = marker_positions[sample_index]
        cleaning_position = cleaning_positions[sample_index]
        axis.plot(
            [marker_position[0], cleaning_position[0]],
            [marker_position[1], cleaning_position[1]],
            [marker_position[2], cleaning_position[2]],
            color="gray",
            linestyle=":" if sample_index == 0 else "--",
            linewidth=1.3,
            label=label,
        )

    all_points = np.vstack(plotted_points)
    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    center = (minimum + maximum) / 2.0
    half_range = max(np.max(maximum - minimum) / 2.0, 0.08)
    axis.set_xlim(center[0] - half_range, center[0] + half_range)
    axis.set_ylim(center[1] - half_range, center[1] + half_range)
    axis.set_zlim(center[2] - half_range, center[2] + half_range)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("Camera X (m)")
    axis.set_ylabel("Camera Y (m)")
    axis.set_zlabel("Camera Z (m)")
    axis.set_title(
        "Marker and cleaning-head trajectories in the camera frame"
    )
    axis.legend(loc="best")
    axis.grid(True, alpha=0.35)
    figure.tight_layout()
    plt.show()


def track_cleaning_device_trajectory(
    quality_config: PoseQualityConfig | None = None,
    filter_config: PoseFilterConfig | None = None,
    interpolation_frequency_hz: float = 30.0,
    max_interpolation_gap_s: float = 0.100,
) -> None:
    """Track marker/head motion live and plot both camera-frame paths."""
    import cv2

    from calibration import (
        HUMAN_TOOL_CHARUCO_CONFIG,
        create_charuco_detector,
    )
    from camera import (
        get_image,
        get_zed_left_intrinsics_rectified,
        open_zed,
    )

    T_marker_cleaning_head = get_T_marker_cleaning_head()
    marker_samples = []
    cleaning_head_samples = []
    segment_start_flags = []
    measured_trajectory_samples = []
    rejection_counts = Counter()
    quality_gate = PoseQualityGate(
        T_marker_cleaning_head,
        quality_config,
    )
    pose_filter = PoseOneEuroFilter(filter_config)
    window_name = "Cleaning-device trajectory tracking"
    zed = None
    try:
        zed, runtime_params, image_zed = open_zed()
        camera_matrix, dist_coeffs = get_zed_left_intrinsics_rectified(zed)
        board, detector = create_charuco_detector(
            HUMAN_TOOL_CHARUCO_CONFIG,
            camera_matrix,
            dist_coeffs,
        )
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print(
            "Move the cleaning device through the demonstration. "
            "Press Q or Escape to stop and plot the trajectories.\n"
        )

        while True:
            image = get_image(zed, runtime_params, image_zed)
            if image is None:
                continue
            detection, pose = _estimate_camera_marker(
                image,
                board,
                detector,
                camera_matrix,
                dist_coeffs,
            )
            measurement_time = time.monotonic()
            display = image.copy()
            if detection.marker_ids is not None:
                cv2.aruco.drawDetectedMarkers(
                    display,
                    detection.marker_corners,
                    detection.marker_ids,
                )
            if detection.charuco_ids is not None:
                cv2.aruco.drawDetectedCornersCharuco(
                    display,
                    detection.charuco_corners,
                    detection.charuco_ids,
                )

            quality = quality_gate.evaluate(
                detection,
                pose,
                measurement_time,
            )
            if not quality.accepted:
                if quality.reason.startswith("reacquiring_"):
                    status = (
                        f"Confirming pose: {quality.reason} | "
                        f"accepted {len(marker_samples)}"
                    )
                    color = (0, 255, 255)
                else:
                    rejection_counts[quality.reason] += 1
                    status = (
                        f"Rejected: {quality.reason} | "
                        f"accepted {len(marker_samples)} | "
                        f"rejected {sum(rejection_counts.values())}"
                    )
                    color = (0, 0, 255)
            else:
                if quality.starts_new_segment:
                    pose_filter.reset()
                T_camera_marker = pose_filter.update(
                    quality.T_camera_marker,
                    measurement_time,
                )
                # Compose after filtering so the calibrated rigid marker/head
                # relationship remains exact at every trajectory sample.
                T_camera_cleaning_head = _validate_rigid_transform(
                    T_camera_marker @ T_marker_cleaning_head,
                    "T_camera_cleaning_head_filtered",
                )
                marker_samples.append(T_camera_marker)
                cleaning_head_samples.append(T_camera_cleaning_head)
                segment_start_flags.append(quality.starts_new_segment)
                measured_trajectory_samples.append(
                    PoseTrajectorySample(
                        measurement_time,
                        T_camera_marker,
                        "measured",
                        quality.starts_new_segment,
                    )
                )

                cleaning_head_rvec, _ = cv2.Rodrigues(
                    T_camera_cleaning_head[:3, :3]
                )
                marker_rvec, _ = cv2.Rodrigues(
                    T_camera_marker[:3, :3]
                )
                cv2.drawFrameAxes(
                    display,
                    camera_matrix,
                    dist_coeffs,
                    marker_rvec,
                    T_camera_marker[:3, 3],
                    0.025,
                )
                cv2.drawFrameAxes(
                    display,
                    camera_matrix,
                    dist_coeffs,
                    cleaning_head_rvec,
                    T_camera_cleaning_head[:3, 3],
                    0.040,
                    3,
                )
                status = (
                    f"Accepted + filtered | samples {len(marker_samples)} | "
                    f"rejected {sum(rejection_counts.values())} | "
                    f"error {pose.mean_reprojection_error_px:.2f} px"
                )
                color = (0, 255, 0)

            cv2.putText(
                display,
                status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, display)
            key = cv2.waitKeyEx(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        if zed is not None:
            zed.close()

    if not marker_samples:
        print("No valid marker poses were collected; no trajectory to plot.")
        return

    completed_samples = fill_short_pose_gaps(
        measured_trajectory_samples,
        interpolation_frequency_hz,
        max_interpolation_gap_s,
    )
    marker_samples = [sample.T_camera_marker for sample in completed_samples]
    cleaning_head_samples = [
        _validate_rigid_transform(
            sample.T_camera_marker @ T_marker_cleaning_head,
            "completed T_camera_cleaning_head",
        )
        for sample in completed_samples
    ]
    segment_start_flags = [
        sample.starts_new_segment for sample in completed_samples
    ]
    interpolated_count = sum(
        sample.status == "interpolated" for sample in completed_samples
    )

    marker_positions = np.asarray(marker_samples)[:, :3, 3]
    cleaning_positions = np.asarray(cleaning_head_samples)[:, :3, 3]

    def segmented_path_length(positions) -> float:
        if len(positions) < 2:
            return 0.0
        step_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        connected_steps = ~np.asarray(segment_start_flags[1:], dtype=bool)
        return float(step_lengths[connected_steps].sum())

    marker_distance = segmented_path_length(marker_positions)
    cleaning_distance = segmented_path_length(cleaning_positions)
    print(
        f"Collected {len(measured_trajectory_samples)} measured poses and "
        f"inserted {interpolated_count} short-gap poses; "
        f"rejected {sum(rejection_counts.values())} measurements.\n"
        f"Marker path length: {marker_distance:.3f} m\n"
        f"Cleaning-head path length: {cleaning_distance:.3f} m"
    )
    if rejection_counts:
        print("Rejection summary:")
        for reason, count in rejection_counts.most_common():
            print(f"  {reason}: {count}")
    plot_camera_frame_trajectories(
        marker_samples,
        cleaning_head_samples,
        segment_start_flags,
    )


def configure_marker_trajectory(arguments: list[str]) -> None:
    """Parse tunable measurement-quality thresholds and start tracking."""
    defaults = PoseQualityConfig()
    filter_defaults = PoseFilterConfig()
    parser = argparse.ArgumentParser(
        prog="python3 Integration/human.py marker-trajectory",
        description=(
            "Track marker and cleaning-head trajectories after rejecting "
            "low-quality or physically implausible poses."
        ),
    )
    parser.add_argument("--min-markers", type=int, default=defaults.min_markers)
    parser.add_argument(
        "--min-corners",
        type=int,
        default=defaults.min_charuco_corners,
    )
    parser.add_argument(
        "--max-mean-error-px",
        type=float,
        default=defaults.max_mean_reprojection_error_px,
    )
    parser.add_argument(
        "--max-corner-error-px",
        type=float,
        default=defaults.max_corner_reprojection_error_px,
    )
    parser.add_argument(
        "--max-translation-jump-m",
        type=float,
        default=defaults.max_translation_jump_m,
    )
    parser.add_argument(
        "--max-rotation-jump-deg",
        type=float,
        default=defaults.max_rotation_jump_deg,
    )
    parser.add_argument(
        "--max-linear-speed-m-s",
        type=float,
        default=defaults.max_linear_speed_m_s,
    )
    parser.add_argument(
        "--max-angular-speed-deg-s",
        type=float,
        default=defaults.max_angular_speed_deg_s,
    )
    parser.add_argument(
        "--max-timestamp-gap-s",
        type=float,
        default=defaults.max_timestamp_gap_s,
    )
    parser.add_argument(
        "--reacquisition-samples",
        type=int,
        default=defaults.reacquisition_samples,
        help="Consecutive consistent poses required after a gap (default: %(default)s)",
    )
    parser.add_argument(
        "--max-reacquisition-step-m",
        type=float,
        default=defaults.max_reacquisition_step_m,
        help="Maximum translation between confirmation candidates (default: %(default)s)",
    )
    parser.add_argument(
        "--max-reacquisition-step-deg",
        type=float,
        default=defaults.max_reacquisition_step_deg,
        help="Maximum rotation between confirmation candidates (default: %(default)s)",
    )
    parser.add_argument(
        "--max-reacquisition-translation-m",
        type=float,
        default=defaults.max_reacquisition_translation_m,
        help="Maximum displacement from the last reliable pose (default: %(default)s)",
    )
    parser.add_argument(
        "--max-reacquisition-rotation-deg",
        type=float,
        default=defaults.max_reacquisition_rotation_deg,
        help="Maximum rotation from the last reliable pose (default: %(default)s)",
    )
    parser.add_argument(
        "--translation-min-cutoff-hz",
        type=float,
        default=filter_defaults.translation_min_cutoff_hz,
        help="Lower values smooth translation more (default: %(default)s)",
    )
    parser.add_argument(
        "--translation-beta",
        type=float,
        default=filter_defaults.translation_beta,
        help="Translation response to faster motion (default: %(default)s)",
    )
    parser.add_argument(
        "--rotation-min-cutoff-hz",
        type=float,
        default=filter_defaults.rotation_min_cutoff_hz,
        help="Lower values smooth rotation more (default: %(default)s)",
    )
    parser.add_argument(
        "--rotation-beta",
        type=float,
        default=filter_defaults.rotation_beta,
        help="Rotation response to faster motion (default: %(default)s)",
    )
    parser.add_argument(
        "--derivative-cutoff-hz",
        type=float,
        default=filter_defaults.derivative_cutoff_hz,
        help="Motion-speed smoothing cutoff (default: %(default)s)",
    )
    parser.add_argument(
        "--interpolation-frequency-hz",
        type=float,
        default=30.0,
        help="Output rate used to identify missing frames (default: %(default)s)",
    )
    parser.add_argument(
        "--max-interpolation-gap-s",
        type=float,
        default=0.100,
        help="Only gaps at or below this duration are filled (default: %(default)s)",
    )
    args = parser.parse_args(arguments)
    config = PoseQualityConfig(
        min_markers=args.min_markers,
        min_charuco_corners=args.min_corners,
        max_mean_reprojection_error_px=args.max_mean_error_px,
        max_corner_reprojection_error_px=args.max_corner_error_px,
        max_translation_jump_m=args.max_translation_jump_m,
        max_rotation_jump_deg=args.max_rotation_jump_deg,
        max_linear_speed_m_s=args.max_linear_speed_m_s,
        max_angular_speed_deg_s=args.max_angular_speed_deg_s,
        max_timestamp_gap_s=args.max_timestamp_gap_s,
        reacquisition_samples=args.reacquisition_samples,
        max_reacquisition_step_m=args.max_reacquisition_step_m,
        max_reacquisition_step_deg=args.max_reacquisition_step_deg,
        max_reacquisition_translation_m=(
            args.max_reacquisition_translation_m
        ),
        max_reacquisition_rotation_deg=(
            args.max_reacquisition_rotation_deg
        ),
    )
    config.validate()
    filter_config = PoseFilterConfig(
        translation_min_cutoff_hz=args.translation_min_cutoff_hz,
        translation_beta=args.translation_beta,
        rotation_min_cutoff_hz=args.rotation_min_cutoff_hz,
        rotation_beta=args.rotation_beta,
        derivative_cutoff_hz=args.derivative_cutoff_hz,
    )
    filter_config.validate()
    if (
        not np.isfinite(args.interpolation_frequency_hz)
        or args.interpolation_frequency_hz <= 0.0
    ):
        raise ValueError("Interpolation frequency must be finite and positive")
    if (
        not np.isfinite(args.max_interpolation_gap_s)
        or args.max_interpolation_gap_s <= 0.0
    ):
        raise ValueError("Maximum interpolation gap must be finite and positive")
    track_cleaning_device_trajectory(
        config,
        filter_config,
        args.interpolation_frequency_hz,
        args.max_interpolation_gap_s,
    )


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {
        "calibrate",
        "read",
        "marker-test",
        "marker-trajectory",
        "transform-plot",
        "set-tool-transform",
    }:
        raise SystemExit(
            "Usage: python3 Integration/human.py "
            "{marker-test|marker-trajectory|transform-plot|"
            "set-tool-transform|calibrate|read} [options]"
        )

    if sys.argv[1] == "set-tool-transform":
        configure_marker_to_cleaning_head_transform(sys.argv[2:])
        return

    if sys.argv[1] == "transform-plot":
        if len(sys.argv) != 2:
            raise SystemExit("transform-plot does not accept additional arguments")
        plot_marker_to_cleaning_head_transform()
        return

    if sys.argv[1] == "marker-test":
        if len(sys.argv) != 2:
            raise SystemExit("marker-test does not accept additional arguments")
        test_cleaning_device_marker()
        return

    if sys.argv[1] == "marker-trajectory":
        configure_marker_trajectory(sys.argv[2:])
        return

    # force_sensor.py uses nested commands to distinguish robot from human.
    sys.argv.insert(2, "human")
    from force_sensor import main as force_sensor_main

    force_sensor_main()


if __name__ == "__main__":
    main()
