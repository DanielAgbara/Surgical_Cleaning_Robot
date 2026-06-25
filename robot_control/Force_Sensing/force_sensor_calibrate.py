#!/usr/bin/env python3
"""
force_sensor_calibrate.py

Calibration script for force sensors used in the Surgical Cleaning Robot project.

This script uses the ForceSensor class from:

    robot_control/Force_Sensing/force_sensor.py

Supported profiles:

    qlmh41_human
        Human-side force sensor.

    qlmh25_robot
        Robot-side force sensor.

Example commands:

    # Full calibration for human QLMH-41
    python robot_control/Force_Sensing/force_sensor_calibrate.py \
        --profile qlmh41_human \
        --action full \
        --port /dev/ttyUSB0 \
        --mass 0.2

    # Full calibration for robot QLMH-25
    python robot_control/Force_Sensing/force_sensor_calibrate.py \
        --profile qlmh25_robot \
        --action full \
        --port /dev/ttyUSB0 \
        --mass 0.2

    # Read calibrated human force
    python robot_control/Force_Sensing/force_sensor_calibrate.py \
        --profile qlmh41_human \
        --action read \
        --port /dev/ttyUSB0

    # Read raw ADC
    python robot_control/Force_Sensing/force_sensor_calibrate.py \
        --profile qlmh41_human \
        --action raw \
        --port /dev/ttyUSB0
"""

import argparse
import sys
from pathlib import Path


# --------------------------------------------------
# Project path setup
# --------------------------------------------------

# This file should be located at:
#
# Surgical_Cleaning_Robot/
#     robot_control/
#         Force_Sensing/
#             force_sensor.py
#             force_sensor_calibrate.py
#
THIS_FILE = Path(__file__).resolve()
FORCE_SENSING_DIR = THIS_FILE.parent

# Add this folder to Python path so we can import force_sensor.py
sys.path.insert(0, str(FORCE_SENSING_DIR))


# --------------------------------------------------
# Force sensor import
# --------------------------------------------------

from force_sensor import ForceSensor, SENSOR_PROFILES


def build_arg_parser():
    """
    Create command-line arguments for calibration and reading.
    """

    parser = argparse.ArgumentParser(
        description="Calibrate and read QLMH force sensors."
    )

    parser.add_argument(
        "--profile",
        choices=list(SENSOR_PROFILES.keys()),
        required=True,
        help=(
            "Force sensor profile to use. "
            "Use qlmh41_human for the human sensor. "
            "Use qlmh25_robot for the robot sensor."
        ),
    )

    parser.add_argument(
        "--action",
        choices=["tare", "calibrate", "full", "read", "raw"],
        required=True,
        help=(
            "Action to perform. "
            "tare = zero the unloaded sensor. "
            "calibrate = calibrate with known mass. "
            "full = tare then calibrate. "
            "read = live calibrated force reading. "
            "raw = live raw ADC reading."
        ),
    )

    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="Arduino serial port. Default: /dev/ttyUSB0",
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Arduino baud rate. Default: 115200",
    )

    parser.add_argument(
        "--mass",
        type=float,
        default=0.2,
        help=(
            "Known calibration mass in kilograms. "
            "Example: 0.2 means 200 g. Default: 0.2 kg."
        ),
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help=(
            "Number of samples to average during tare/calibration. "
            "Higher is smoother but slower. Default: 200."
        ),
    )

    parser.add_argument(
        "--hz",
        type=float,
        default=100.0,
        help="Live reading rate in Hz. Default: 100 Hz.",
    )

    parser.add_argument(
        "--force-sign",
        type=float,
        choices=[-1.0, 1.0],
        default=1.0,
        help=(
            "Force direction sign. "
            "Use 1 if force increases correctly. "
            "Use -1 if force has the opposite sign."
        ),
    )

    return parser


def main():
    """
    Main calibration/read entry point.
    """

    parser = build_arg_parser()
    args = parser.parse_args()

    print("")
    print("--------------------------------------------------")
    print("FORCE SENSOR TOOL")
    print("--------------------------------------------------")
    print(f"Profile: {args.profile}")
    print(f"Action: {args.action}")
    print(f"Port: {args.port}")
    print(f"Baud: {args.baud}")
    print("--------------------------------------------------")
    print("")

    sensor = ForceSensor(
        profile_name=args.profile,
        port=args.port,
        baud=args.baud,
        force_sign=args.force_sign,
    )

    try:
        if args.action == "tare":
            sensor.tare(
                samples=args.samples,
                interactive=True,
            )

        elif args.action == "calibrate":
            sensor.calibrate_with_multiple_loads(
                masses_kg=None,
                samples=args.samples,
                interactive=True,
            )

        elif args.action == "full":
            sensor.full_calibration(
                known_mass_kg=args.mass,
                samples=args.samples,
            )

        elif args.action == "read":
            sensor.live_read_force(
                hz=args.hz,
            )

        elif args.action == "raw":
            sensor.live_read_raw(
                hz=args.hz,
            )

    except KeyboardInterrupt:
        print("")
        print("Stopped by user.")

    finally:
        sensor.disconnect()
        print("Force sensor disconnected.")


if __name__ == "__main__":
    main()