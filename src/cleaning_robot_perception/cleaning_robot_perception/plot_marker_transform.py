#!/usr/bin/env python3

"""
Plot the fixed transform between the ChArUco marker and the
human cleaning head.

The cleaning-head frame is placed at the origin.

Units displayed:
    millimeters
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ament_index_python.packages import (
    get_package_share_directory,
)


PACKAGE_NAME = "cleaning_robot_perception"


# ============================================================
# Load Transform
# ============================================================

def load_marker_to_cleaning_head_transform():
    """
    Load:

        T_marker_cleaning_head

    from the installed JSON file.

    The JSON stores translation in meters. Convert only the
    translation column to millimeters.
    """

    package_share = get_package_share_directory(
        PACKAGE_NAME
    )

    transform_path = Path(
        package_share
    ) / "config" / "marker_to_cleaning_head.json"

    if not transform_path.exists():

        raise FileNotFoundError(
            f"Transform file not found: {transform_path}"
        )

    with open(
        transform_path,
        "r",
        encoding="utf-8",
    ) as file:

        payload = json.load(file)

    T_marker_cleaning_head = np.asarray(
        payload["data"],
        dtype=np.float64,
    )

    if T_marker_cleaning_head.shape != (4, 4):

        raise ValueError(
            "T_marker_cleaning_head must be 4x4"
        )

    # JSON translation is stored in meters.
    #
    # Convert translation only:
    #
    # m -> mm
    #
    T_marker_cleaning_head = (
        T_marker_cleaning_head.copy()
    )

    T_marker_cleaning_head[
        :3,
        3,
    ] *= 1000.0

    return T_marker_cleaning_head


# ============================================================
# Invert Rigid Transform
# ============================================================

def invert_transform(T):
    """
    Rigid transform inverse.

        T = [R t]
            [0 1]

        T^-1 = [R^T  -R^T t]
               [ 0      1  ]
    """

    T = np.asarray(
        T,
        dtype=np.float64,
    ).reshape(
        4,
        4,
    )

    inverse = np.eye(
        4,
        dtype=np.float64,
    )

    R = T[
        :3,
        :3,
    ]

    t = T[
        :3,
        3,
    ]

    inverse[
        :3,
        :3,
    ] = R.T

    inverse[
        :3,
        3,
    ] = -R.T @ t

    return inverse


# ============================================================
# Draw Coordinate Frame
# ============================================================

def draw_frame(
    axis,
    transform,
    label,
    axis_length_mm=40.0,
):
    """
    Draw X, Y and Z axes of a 4x4 transform.
    """

    origin = transform[
        :3,
        3,
    ]

    colors = (
        "red",
        "green",
        "blue",
    )

    names = (
        "X",
        "Y",
        "Z",
    )

    for index, (
        color,
        name,
    ) in enumerate(
        zip(
            colors,
            names,
        )
    ):

        direction = (
            transform[
                :3,
                index,
            ]
            * axis_length_mm
        )

        endpoint = (
            origin
            + direction
        )

        axis.plot(
            [
                origin[0],
                endpoint[0],
            ],
            [
                origin[1],
                endpoint[1],
            ],
            [
                origin[2],
                endpoint[2],
            ],
            color=color,
            linewidth=2.5,
        )

        axis.text(
            endpoint[0],
            endpoint[1],
            endpoint[2],
            f"{label} {name}",
            color=color,
        )

    axis.scatter(
        origin[0],
        origin[1],
        origin[2],
        color="black",
        s=35,
    )

    axis.text(
        origin[0],
        origin[1],
        origin[2],
        f"  {label}",
        color="black",
    )


# ============================================================
# Equal 3D Axis Scaling
# ============================================================

def set_axes_equal(
    axis,
    points,
):
    """
    Give X/Y/Z equal physical scaling.
    """

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    minimum = np.min(
        points,
        axis=0,
    )

    maximum = np.max(
        points,
        axis=0,
    )

    center = (
        minimum + maximum
    ) / 2.0

    half_range = max(
        np.max(
            maximum - minimum
        ) / 2.0,
        60.0,
    )

    half_range *= 1.25

    axis.set_xlim(
        center[0] - half_range,
        center[0] + half_range,
    )

    axis.set_ylim(
        center[1] - half_range,
        center[1] + half_range,
    )

    axis.set_zlim(
        center[2] - half_range,
        center[2] + half_range,
    )

    axis.set_box_aspect(
        (1, 1, 1)
    )


# ============================================================
# Plot
# ============================================================

def plot_marker_transform():

    # --------------------------------------------------------
    # Stored transform:
    #
    # T_marker_cleaning_head
    #
    # maps:
    #
    # cleaning head -> marker
    # --------------------------------------------------------

    T_marker_cleaning_head = (
        load_marker_to_cleaning_head_transform()
    )

    # --------------------------------------------------------
    # For visualization we want:
    #
    # cleaning head = origin
    #
    # therefore obtain:
    #
    # T_cleaning_head_marker
    # --------------------------------------------------------

    T_cleaning_head_marker = (
        invert_transform(
            T_marker_cleaning_head
        )
    )

    T_cleaning_head = np.eye(
        4,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Print transforms
    # --------------------------------------------------------

    print(
        "\nT_marker_cleaning_head [mm translation]:"
    )

    print(
        T_marker_cleaning_head
    )

    print(
        "\nT_cleaning_head_marker [mm translation]:"
    )

    print(
        T_cleaning_head_marker
    )

    print(
        "\nMarker position relative to cleaning head [mm]:"
    )

    print(
        T_cleaning_head_marker[
            :3,
            3,
        ]
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    figure = plt.figure(
        figsize=(9, 8)
    )

    axis = figure.add_subplot(
        111,
        projection="3d",
    )

    draw_frame(
        axis,
        T_cleaning_head,
        "Cleaning head",
        axis_length_mm=40.0,
    )

    draw_frame(
        axis,
        T_cleaning_head_marker,
        "Marker",
        axis_length_mm=40.0,
    )

    # --------------------------------------------------------
    # Draw offset between origins
    # --------------------------------------------------------

    cleaning_origin = (
        T_cleaning_head[
            :3,
            3,
        ]
    )

    marker_origin = (
        T_cleaning_head_marker[
            :3,
            3,
        ]
    )

    axis.plot(
        [
            cleaning_origin[0],
            marker_origin[0],
        ],
        [
            cleaning_origin[1],
            marker_origin[1],
        ],
        [
            cleaning_origin[2],
            marker_origin[2],
        ],
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Marker offset",
    )

    # --------------------------------------------------------
    # Equal axis scaling
    # --------------------------------------------------------

    extra_points = [
        cleaning_origin,
        marker_origin,
    ]

    for transform in (
        T_cleaning_head,
        T_cleaning_head_marker,
    ):

        origin = transform[
            :3,
            3,
        ]

        for index in range(3):

            extra_points.append(
                origin
                + 40.0
                * transform[
                    :3,
                    index,
                ]
            )

    set_axes_equal(
        axis,
        extra_points,
    )

    # --------------------------------------------------------
    # Plot labels
    # --------------------------------------------------------

    axis.set_xlabel(
        "Cleaning-head X [mm]"
    )

    axis.set_ylabel(
        "Cleaning-head Y [mm]"
    )

    axis.set_zlabel(
        "Cleaning-head Z [mm]"
    )

    axis.set_title(
        "ChArUco marker relative to cleaning head"
    )

    axis.legend()

    axis.grid(
        True,
        alpha=0.35,
    )

    figure.tight_layout()

    plt.show()


# ============================================================
# Main
# ============================================================

def main():

    plot_marker_transform()


if __name__ == "__main__":

    main()