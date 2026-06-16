"""
body34.py

Utility functions for extracting and processing arm data from
the ZED BODY_34 human skeleton model.

This module provides:

    - BODY_34 joint index definitions
    - Body tracking initialization
    - Human selection from detected bodies
    - Arm keypoint extraction
    - Arm visualization utilities
    - Arm length estimation

The primary purpose of this file is to provide a clean interface
between the ZED body-tracking system and downstream robot
teleoperation algorithms.

Coordinate Data
---------------
The ZED SDK returns 3D keypoints in the camera coordinate frame:

    X : Right
    Y : Down
    Z : Forward

Units are determined by the ZED camera configuration
(e.g. millimeters or meters).

BODY_34 Arm Structure
---------------------
Right Arm:

    Shoulder (12)
        |
    Elbow (13)
        |
    Wrist (14)
        |
    Hand (15)

Left Arm:

    Shoulder (5)
        |
    Elbow (6)
        |
    Wrist (7)
        |
    Hand (8)

Typical Workflow
----------------

    1. Initialize body tracking:
           body_runtime = setup_body_tracking(zed)

    2. Retrieve bodies:
           zed.retrieve_bodies(bodies, body_runtime)

    3. Select one person:
           body = get_single_body(bodies)

    4. Extract arm points:
           arm_data = get_arm_points(body, arm="right")

    5. Compute arm lengths:
           lengths = estimate_arm_length(arm_data)

    6. Use the resulting 3D positions for:
           - Human motion tracking
           - Velocity estimation
           - Intent prediction
           - Human-to-robot teleoperation
           - Data collection and analysis

This module is intended to be reused by both real-time
teleoperation scripts and offline data-processing pipelines.
"""

import pyzed.sl as sl
import cv2 as cv
import numpy as np

BODY_FORMAT = sl.BODY_FORMAT.BODY_34

# Right arm BODY_34 indices
RIGHT_SHOULDER = 12
RIGHT_ELBOW = 13
RIGHT_WRIST = 14
RIGHT_HAND = 15

# Left arm BODY_34 indices
LEFT_SHOULDER = 5
LEFT_ELBOW = 6
LEFT_WRIST = 7
LEFT_HAND = 8


def setup_body_tracking(zed):
    """
    Enable BODY_34 body tracking on an already-opened ZED camera.

    Parameters
    ----------
    zed : sl.Camera
        Already opened ZED camera object.

    Returns
    -------
    body_runtime : sl.BodyTrackingRuntimeParameters
        Runtime parameters used by zed.retrieve_bodies(...).
    """

    # -------------------------------------------------
    # Create body tracking parameter object
    # -------------------------------------------------

    body_params = sl.BodyTrackingParameters()

    # -------------------------------------------------
    # Use the accurate human body tracking model
    # -------------------------------------------------

    body_params.detection_model = (
        sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    )

    # -------------------------------------------------
    # Enable tracking so the ZED can keep person IDs
    # consistent between frames
    # -------------------------------------------------

    body_params.enable_tracking = True

    # -------------------------------------------------
    # Enable body fitting for smoother skeleton points
    # -------------------------------------------------

    body_params.enable_body_fitting = True

    # -------------------------------------------------
    # Use the BODY_34 skeleton format
    # -------------------------------------------------

    body_params.body_format = BODY_FORMAT

    # -------------------------------------------------
    # Positional tracking is required for ZED body tracking
    # -------------------------------------------------

    positional_params = sl.PositionalTrackingParameters()

    err = zed.enable_positional_tracking(positional_params)

    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(
            f"Failed to enable positional tracking: {err}"
        )

    # -------------------------------------------------
    # Enable body tracking
    # -------------------------------------------------

    err = zed.enable_body_tracking(body_params)

    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(
            f"Failed to enable body tracking: {err}"
        )

    # -------------------------------------------------
    # Create runtime body tracking parameters
    # -------------------------------------------------

    body_runtime = sl.BodyTrackingRuntimeParameters()

    # -------------------------------------------------
    # Ignore low-confidence body detections
    # -------------------------------------------------

    body_runtime.detection_confidence_threshold = 40

    print("Body tracking enabled.")

    return body_runtime

def get_single_body(bodies, mode="closest"):
    """
    Select one person from all detected people.

    Parameters
    ----------
    bodies : sl.Bodies
        Body container returned by zed.retrieve_bodies(...).

    mode : str
        Selection method.

        "closest":
            Select the detected person closest to the camera.

        "first":
            Select the first detected person.

    Returns
    -------
    body : sl.BodyData or None
        One selected body.

        Returns None if no bodies are detected.
    """

    # -------------------------------------------------
    # If no human is detected, return None
    # -------------------------------------------------

    if len(bodies.body_list) == 0:
        print("No detected bodies!")
        return None

    # -------------------------------------------------
    # Option 1: use the first detected person
    # -------------------------------------------------

    if mode == "first":
        return bodies.body_list[0]

    # -------------------------------------------------
    # Option 2: use the closest detected person
    # -------------------------------------------------
    #
    # body.position[2] is the depth value.
    #
    # Smaller positive z value usually means the person
    # is closer to the camera.
    #
    # -------------------------------------------------

    if mode == "closest":
        return min(
            bodies.body_list,
            key=lambda b: float(b.position[2])
            if b.position[2] > 0
            else float("inf")
        )

    # -------------------------------------------------
    # Reject invalid selection modes
    # -------------------------------------------------

    raise ValueError("mode must be 'closest' or 'first'")


def get_arm_indices(arm="right"):
    """
    Return BODY_34 indices for the selected arm.

    Parameters
    ----------
    arm : str
        "right" or "left".

    Returns
    -------
    shoulder_idx, elbow_idx, wrist_idx : tuple
        BODY_34 joint indices for the selected arm.
    """

    # -------------------------------------------------
    # Normalize user input
    # -------------------------------------------------

    arm = arm.lower()

    # -------------------------------------------------
    # Return right arm indices
    # -------------------------------------------------

    if arm == "right":
        return RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, RIGHT_HAND

    # -------------------------------------------------
    # Return left arm indices
    # -------------------------------------------------

    if arm == "left":
        return LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_HAND

    # -------------------------------------------------
    # Reject invalid arm names
    # -------------------------------------------------

    raise ValueError("arm must be 'right' or 'left'")

def get_arm_points(body, arm="right"):
    """
    Extract shoulder, elbow, and wrist points for one arm.

    Parameters
    ----------
    body : sl.BodyData
        One detected person from bodies.body_list.

    arm : str
        Arm to extract.
        Options:
            "right"
            "left"

    Returns
    -------
    arm_data : dict or None
        Dictionary containing 2D and 3D points.

        Returns None if the selected arm points are invalid.
    """

    # -------------------------------------------------
    # Get correct BODY_34 indices
    # -------------------------------------------------

    shoulder_idx, elbow_idx, wrist_idx, hand_idx = get_arm_indices(arm)

    # -------------------------------------------------
    # Get 2D pixel keypoints
    # -------------------------------------------------
    #
    # These are used for drawing on the OpenCV image.
    #
    # -------------------------------------------------

    keypoints_2d = body.keypoint_2d

    # -------------------------------------------------
    # Get 3D keypoints
    # -------------------------------------------------
    #
    # These are used for distance calculations and
    # later robot control.
    #
    # -------------------------------------------------

    keypoints_3d = body.keypoint

    # -------------------------------------------------
    # Check that the selected indices exist
    # -------------------------------------------------

    max_idx = max(shoulder_idx, elbow_idx, wrist_idx, hand_idx)

    if len(keypoints_2d) <= max_idx:
        return None

    if len(keypoints_3d) <= max_idx:
        return None

    # -------------------------------------------------
    # Extract 2D points
    # -------------------------------------------------

    shoulder_2d = np.array(keypoints_2d[shoulder_idx], dtype=float)
    elbow_2d = np.array(keypoints_2d[elbow_idx], dtype=float)
    wrist_2d = np.array(keypoints_2d[wrist_idx], dtype=float)
    hand_2d = np.array(keypoints_2d[hand_idx], dtype=float)

    # -------------------------------------------------
    # Extract 3D points
    # -------------------------------------------------

    shoulder_3d = np.array(keypoints_3d[shoulder_idx], dtype=float)
    elbow_3d = np.array(keypoints_3d[elbow_idx], dtype=float)
    wrist_3d = np.array(keypoints_3d[wrist_idx], dtype=float)
    hand_3d = np.array(keypoints_3d[hand_idx], dtype=float)

    # -------------------------------------------------
    # Check 2D points
    # -------------------------------------------------
    #
    # Invalid 2D points are often [0, 0] or negative.
    #
    # -------------------------------------------------

    for point in [shoulder_2d, elbow_2d, wrist_2d]:
        if point[0] <= 0 or point[1] <= 0:
            return None

    # -------------------------------------------------
    # Check 3D points
    # -------------------------------------------------
    #
    # Invalid 3D points can contain nan or inf.
    #
    # -------------------------------------------------

    for point in [shoulder_3d, elbow_3d, wrist_3d]:
        if not np.isfinite(point).all():
            return None

    # -------------------------------------------------
    # Package arm data
    # -------------------------------------------------

    arm_data = {
        "arm": arm,

        "shoulder_2d": shoulder_2d,
        "elbow_2d": elbow_2d,
        "wrist_2d": wrist_2d,
        "hand_2d" : hand_2d,

        "shoulder_3d": shoulder_3d,
        "elbow_3d": elbow_3d,
        "wrist_3d": wrist_3d,
        "hand_3d" : hand_3d
    }

    return arm_data



def draw_arm_points_and_lines(image, arm_data):
    """
    Draw shoulder, elbow, and wrist on an OpenCV image.

    Parameters
    ----------
    image : np.ndarray
        OpenCV image.

    arm_data : dict or None
        Output from get_arm_points(...).

    Returns
    -------
    image : np.ndarray
        Image with arm overlay.
    """

    # -------------------------------------------------
    # If arm data is invalid, return image unchanged
    # -------------------------------------------------

    if arm_data is None:
        return image

    # -------------------------------------------------
    # Extract 2D points
    # -------------------------------------------------

    shoulder = arm_data["shoulder_2d"]
    elbow = arm_data["elbow_2d"]
    wrist = arm_data["wrist_2d"]
    hand = arm_data["hand_2d"]

    # -------------------------------------------------
    # Convert float pixel coordinates to integer pixels
    # -------------------------------------------------

    shoulder = (int(shoulder[0]), int(shoulder[1]))
    elbow = (int(elbow[0]), int(elbow[1]))
    wrist = (int(wrist[0]), int(wrist[1]))
    hand = (int(hand[0]), int(hand[1]))

    # -------------------------------------------------
    # Draw arm links first
    # -------------------------------------------------

    cv.line(image, shoulder, elbow, (0, 255, 255), 3)
    cv.line(image, elbow, wrist, (0, 255, 255), 3)
    cv.line(image, wrist, hand, (0, 255, 255), 3)

    # -------------------------------------------------
    # Draw arm joints
    # -------------------------------------------------

    cv.circle(image, shoulder, 7, (0, 255, 0), -1)
    cv.circle(image, elbow, 7, (0, 255, 0), -1)
    cv.circle(image, wrist, 7, (0, 255, 0), -1)
    cv.circle(image, hand, 7, (0, 255, 0), -1)

    # -------------------------------------------------
    # Label joints
    # -------------------------------------------------

    cv.putText(
        image,
        "Shoulder",
        (shoulder[0] + 10, shoulder[1] - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

    cv.putText(
        image,
        "Elbow",
        (elbow[0] + 10, elbow[1] - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

    cv.putText(
        image,
        "Wrist",
        (wrist[0] + 10, wrist[1] - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

    cv.putText(
        image,
        "Hand",
        (hand[0] + 10, hand[1] - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )

    return image


def estimate_arm_length(arm_data):
    """
    Estimate BODY_34 arm segment lengths from one valid arm frame.

    BODY_34 arm structure:
        shoulder -> elbow -> wrist -> hand

    Parameters
    ----------
    arm_data : dict
        Output from get_arm_points(...).

    Returns
    -------
    lengths : dict
        Arm lengths in the same units as the ZED camera.
    """

    if arm_data is None:
        return None

    shoulder = np.array(arm_data["shoulder_3d"], dtype=float)
    elbow = np.array(arm_data["elbow_3d"], dtype=float)
    wrist = np.array(arm_data["wrist_3d"], dtype=float)
    hand = np.array(arm_data["hand_3d"], dtype=float)

    points = [shoulder, elbow, wrist, hand]

    for p in points:
        if not np.isfinite(p).all():
            return None

    shoulder_to_elbow = np.linalg.norm(elbow - shoulder)
    elbow_to_wrist = np.linalg.norm(wrist - elbow)
    wrist_to_hand = np.linalg.norm(hand - wrist)

    shoulder_to_wrist = np.linalg.norm(wrist - shoulder)
    shoulder_to_hand = np.linalg.norm(hand - shoulder)

    full_arm_chain_length = (
        shoulder_to_elbow
        + elbow_to_wrist
        + wrist_to_hand
    )

    arm_chain_to_wrist = shoulder_to_elbow + elbow_to_wrist

    lengths = {
        "shoulder_to_elbow": float(shoulder_to_elbow),
        "elbow_to_wrist": float(elbow_to_wrist),
        "wrist_to_hand": float(wrist_to_hand),

        "shoulder_to_wrist_direct": float(shoulder_to_wrist),
        "shoulder_to_hand_direct": float(shoulder_to_hand),

        "arm_chain_to_wrist": float(arm_chain_to_wrist),
        "full_arm_chain_length": float(full_arm_chain_length),
    }

    return lengths



def quaternion_to_matrix(q):
    """
    Convert quaternion [w, x, y, z] to a 3x3 rotation matrix.
    """

    q = np.asarray(q, dtype=float)

    if len(q) != 4:
        raise ValueError("Quaternion must be [w, x, y, z]")

    x, y, z, w = q

    x2 = x * x
    y2 = y * y
    z2 = z * z

    xy = x * y
    xz = x * z
    yz = y * z

    wx = w * x
    wy = w * y
    wz = w * z

    R = np.array([
        [1 - 2*y2 - 2*z2, 2*xy - 2*wz,     2*xz + 2*wy],
        [2*xy + 2*wz,     1 - 2*x2 - 2*z2, 2*yz - 2*wx],
        [2*xz - 2*wy,     2*yz + 2*wx,     1 - 2*x2 - 2*y2]
    ])

    return R

def get_hand_orientation(body, arm="right"):
    """
    Extract hand orientation from BODY_34 and return both
    quaternion and rotation matrix.

    Parameters
    ----------
    body : sl.BodyData

    arm : str
        "right" or "left"

    Returns
    -------
    orientation : dict or None

        {
            "quaternion": np.ndarray (4,),
            "rotation_matrix": np.ndarray (3,3)
        }
    """

    _, _, _, hand_idx = get_arm_indices(arm)

    # BODY_34 local joint orientations
    local_orientations = body.global_orientation_per_joint

    if len(local_orientations) <= hand_idx:
        return None

    quat = np.array(
        local_orientations[hand_idx],
        dtype=float
    )

    if not np.isfinite(quat).all():
        return None

    # Normalize quaternion
    norm = np.linalg.norm(quat)

    if norm < 1e-8:
        return None

    quat = quat / norm

    R = quaternion_to_matrix(quat)

    return {
        "quaternion": quat,
        "rotation_matrix": R
    }