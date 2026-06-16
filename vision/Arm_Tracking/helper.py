import cv2
import numpy as np


def get_arm_length(shoulder_pos, hand_pos):
    """
    Compute shoulder-to-hand arm length.
    """

    shoulder_pos = np.asarray(shoulder_pos, dtype=float).reshape(3)
    hand_pos = np.asarray(hand_pos, dtype=float).reshape(3)

    return np.linalg.norm(hand_pos - shoulder_pos)


def euclidean_distance(p1, p2):
    """
    Compute 3D distance between two points.
    """

    if p1 is None or p2 is None:
        return None

    p1 = np.asarray(p1, dtype=float).reshape(3)
    p2 = np.asarray(p2, dtype=float).reshape(3)

    return np.linalg.norm(p1 - p2)


def extract_arm_positions(arm_data, body_model):
    """
    Extract shoulder, wrist, and hand positions from body tracking output.

    BODY_34:
        Uses actual hand keypoint.

    BODY_18:
        Falls back to wrist as hand.
    """

    if arm_data is None:
        return None, None, None

    shoulder_pos = np.asarray(
        arm_data["shoulder_3d"],
        dtype=float
    ).reshape(3)

    wrist_pos = np.asarray(
        arm_data["wrist_3d"],
        dtype=float
    ).reshape(3)

    if body_model == 34:
        hand_pos = np.asarray(
            arm_data["hand_3d"],
            dtype=float
        ).reshape(3)
    else:
        hand_pos = wrist_pos.copy()

    return shoulder_pos, wrist_pos, hand_pos


def build_reference_data(
    shoulder_ref_samples,
    wrist_ref_samples,
    hand_ref_samples,
    num_reference_frames,
    body_model,
    arm_to_track,
):
    """
    Average multiple frames to create reference position.
    """

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
        "num_reference_frames": num_reference_frames,
        "body_model": body_model,
        "arm": arm_to_track,

        "shoulder_position_mm": shoulder_ref,
        "wrist_position_mm": wrist_ref,
        "hand_position_mm": hand_ref,

        "arm_length_mm": reference_arm_length,
        "arm_length_definition": "shoulder_to_hand",
    }

    return reference_data, shoulder_ref, wrist_ref, hand_ref


def build_arm_tracking_record(
    shoulder_pos,
    wrist_pos,
    hand_pos,
    prev_shoulder_pos,
    prev_wrist_pos,
    prev_hand_pos,
):
    """
    Build arm tracking block for JSON.
    """

    current_arm_length = get_arm_length(
        shoulder_pos,
        hand_pos
    )

    wrist_delta_from_previous = wrist_pos - prev_wrist_pos
    hand_delta_from_previous = hand_pos - prev_hand_pos
    shoulder_delta_from_previous = shoulder_pos - prev_shoulder_pos

    return {
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


def draw_distance_info(
    frame,
    distance_mm,
    hand_close_to_object,
    object_name,
):
    """
    Draw hand-to-object distance status.
    """

    frame = np.ascontiguousarray(frame)

    if distance_mm is None:
        text = "Hand-object distance: unavailable"
        color = (0, 0, 255)
    else:
        text = f"Hand-to-{object_name} distance: {distance_mm:.1f} mm"

        if hand_close_to_object:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

    cv2.putText(
        frame,
        text,
        (30, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )

    if hand_close_to_object:
        status_text = f"HAND CLOSE TO {object_name.upper()}"
    else:
        status_text = f"HAND NOT CLOSE TO {object_name.upper()}"

    cv2.putText(
        frame,
        status_text,
        (30, 245),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )

    return frame


def draw_main_status(
    frame,
    body_model,
    arm_to_track,
    collecting_reference,
    reference_collected,
):
    """
    Draw main arm tracking status.
    """

    frame = np.ascontiguousarray(frame)

    if collecting_reference:
        status_text = "Collecting reference | Keep arm still"
        status_color = (0, 255, 255)

    elif reference_collected:
        status_text = "Reference collected | Recording all data"
        status_color = (0, 255, 0)

    else:
        status_text = "Press ENTER to collect reference"
        status_color = (0, 255, 255)

    cv2.putText(
        frame,
        status_text,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"BODY_{body_model} | Tracking {arm_to_track} arm",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "ENTER: collect reference | q: quit",
        (30, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame