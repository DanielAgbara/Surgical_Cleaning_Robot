#!/usr/bin/env python3

"""
Create or edit T_marker_cleaning_head.

Inputs describe the CAD transform from the cleaning-head
origin to the ChArUco marker origin:

    x_mm
    y_mm
    z_mm
    rotation_deg

The saved transform is:

    T_marker_cleaning_head

such that:

    p_marker = T_marker_cleaning_head @ p_cleaning_head

Translation is stored in meters in the JSON file to remain
compatible with the original project data.
"""

import argparse
import json
from pathlib import Path

import numpy as np


# ============================================================
# Rotation
# ============================================================

def Rx(angle_rad):
    """
    Rotation about X.
    """

    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


# ============================================================
# Rigid Transform Inverse
# ============================================================

def invert_transform(T):
    """
    Invert a rigid 4x4 transform.
    """

    T = np.asarray(
        T,
        dtype=np.float64,
    ).reshape(4, 4)

    inverse = np.eye(
        4,
        dtype=np.float64,
    )

    R = T[:3, :3]
    t = T[:3, 3]

    inverse[:3, :3] = R.T
    inverse[:3, 3] = -R.T @ t

    return inverse


# ============================================================
# Validation
# ============================================================

def validate_transform(T):
    """
    Validate that T is a proper SE(3) transform.
    """

    T = np.asarray(
        T,
        dtype=np.float64,
    )

    if T.shape != (4, 4):
        raise ValueError(
            "Transform must be 4x4"
        )

    if not np.all(
        np.isfinite(T)
    ):
        raise ValueError(
            "Transform contains non-finite values"
        )

    if not np.allclose(
        T[3],
        [0.0, 0.0, 0.0, 1.0],
        atol=1e-9,
    ):
        raise ValueError(
            "Invalid homogeneous last row"
        )

    R = T[:3, :3]

    if not np.allclose(
        R.T @ R,
        np.eye(3),
        atol=1e-8,
    ):
        raise ValueError(
            "Rotation matrix is not orthonormal"
        )

    if not np.isclose(
        np.linalg.det(R),
        1.0,
        atol=1e-8,
    ):
        raise ValueError(
            "Rotation determinant must be +1"
        )

    return T


# ============================================================
# Build Transform
# ============================================================

def build_marker_cleaning_transform(
    x_mm,
    y_mm,
    z_mm,
    rotation_deg,
):
    """
    Build T_marker_cleaning_head.

    CAD measurements describe:

        T_cleaning_head_marker

    We invert that transform because runtime tracking requires:

        T_marker_cleaning_head
    """

    translation_mm = np.array(
        [
            x_mm,
            y_mm,
            z_mm,
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Same axis convention as the old human.py.
    #
    # Base cleaning-head orientation:
    #
    #   +X right
    #   +Y back
    #   +Z down
    #
    # Rx(pi) flips Y and Z while maintaining a proper
    # right-handed coordinate frame.
    #
    # Then apply the adjustable X rotation.
    # --------------------------------------------------------

    rotation_cleaning_head_marker = (
        Rx(
            np.deg2rad(
                rotation_deg
            )
        )
        @ Rx(np.pi)
    )

    # --------------------------------------------------------
    # T_cleaning_head_marker
    # --------------------------------------------------------

    T_cleaning_head_marker = np.eye(
        4,
        dtype=np.float64,
    )

    T_cleaning_head_marker[
        :3,
        :3,
    ] = rotation_cleaning_head_marker

    # Store internally in meters.
    T_cleaning_head_marker[
        :3,
        3,
    ] = (
        translation_mm
        / 1000.0
    )

    validate_transform(
        T_cleaning_head_marker
    )

    # --------------------------------------------------------
    # Runtime transform
    # --------------------------------------------------------

    T_marker_cleaning_head = (
        invert_transform(
            T_cleaning_head_marker
        )
    )

    validate_transform(
        T_marker_cleaning_head
    )

    return (
        T_cleaning_head_marker,
        T_marker_cleaning_head,
    )


# ============================================================
# Save JSON
# ============================================================

def save_transform(
    output_path,
    x_mm,
    y_mm,
    z_mm,
    rotation_deg,
):

    (
        T_cleaning_head_marker,
        T_marker_cleaning_head,
    ) = build_marker_cleaning_transform(
        x_mm,
        y_mm,
        z_mm,
        rotation_deg,
    )

    output_path = Path(
        output_path
    ).expanduser().resolve()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "description": {
            "summary": (
                "Fixed pose of the cleaning-head center "
                "in the detected ChArUco marker frame."
            ),
            "transform": (
                "T_marker_cleaning_head"
            ),
            "mapping": (
                "p_marker = "
                "T_marker_cleaning_head "
                "@ p_cleaning_head"
            ),
            "matrix_layout": (
                "4x4 homogeneous transform"
            ),
            "translation_units": (
                "meters"
            ),
            "cad_input_translation_mm": [
                float(x_mm),
                float(y_mm),
                float(z_mm),
            ],
            "cad_input_rotation_deg": (
                float(rotation_deg)
            ),
            "construction": (
                "T_marker_cleaning_head = "
                "inverse([Rx(rotation_deg) @ "
                "Rx(pi), "
                "translation_cleaning_head_to_marker])"
            ),
        },

        "data": (
            T_marker_cleaning_head.tolist()
        ),
    }

    temporary_path = (
        output_path.with_suffix(
            output_path.suffix + ".tmp"
        )
    )

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
        )

        file.write("\n")

    temporary_path.replace(
        output_path
    )

    # ========================================================
    # Print Result
    # ========================================================

    print(
        "\nCleaning-head -> marker CAD input:"
    )

    print(
        f"  X: {x_mm:.3f} mm"
    )

    print(
        f"  Y: {y_mm:.3f} mm"
    )

    print(
        f"  Z: {z_mm:.3f} mm"
    )

    print(
        f"  X rotation: {rotation_deg:.3f} deg"
    )

    print(
        "\nT_cleaning_head_marker:"
    )

    print(
        T_cleaning_head_marker
    )

    print(
        "\nT_marker_cleaning_head:"
    )

    print(
        T_marker_cleaning_head
    )

    print(
        "\nT_marker_cleaning_head translation [mm]:"
    )

    print(
        T_marker_cleaning_head[
            :3,
            3,
        ] * 1000.0
    )

    print(
        f"\nSaved to:\n{output_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create/edit the fixed ChArUco "
            "marker-to-cleaning-head transform."
        )
    )

    parser.add_argument(
        "--x-mm",
        type=float,
        default=-50.0,
        help=(
            "Marker X position measured from "
            "cleaning-head origin [mm]"
        ),
    )

    parser.add_argument(
        "--y-mm",
        type=float,
        default=-60.0,
        help=(
            "Marker Y position measured from "
            "cleaning-head origin [mm]"
        ),
    )

    parser.add_argument(
        "--z-mm",
        type=float,
        default=-109.0,
        help=(
            "Marker Z position measured from "
            "cleaning-head origin [mm]"
        ),
    )

    parser.add_argument(
        "--rotation-deg",
        type=float,
        default=150.0,
        help=(
            "Additional rotation about marker/cleaning "
            "X axis [deg]"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Path to marker_to_cleaning_head.json"
        ),
    )

    args = parser.parse_args()

    save_transform(
        args.output,
        args.x_mm,
        args.y_mm,
        args.z_mm,
        args.rotation_deg,
    )


if __name__ == "__main__":

    main()