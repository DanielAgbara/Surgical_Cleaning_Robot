#!/usr/bin/env python3
"""
force_reader.py

Runtime force sensor reader.

This script uses the reusable ForceSensor class from:

    robot_control/Force_Sensing/force_sensor.py

Use this script after calibration.

Examples:

    # Read human QLMH-41 calibrated force
    python force_reader.py --profile qlmh41_human --port /dev/ttyUSB0

    # Read robot QLMH-25 calibrated force
    python force_reader.py --profile qlmh25_robot --port /dev/ttyUSB0

    # Read only raw ADC data
    python force_reader.py --profile qlmh41_human --port /dev/ttyUSB0 --raw
"""

import argparse
import sys
import time
from pathlib import Path


# --------------------------------------------------
# Path setup
# --------------------------------------------------
# This lets this file import force_sensor.py from the same folder.
# Folder:
#   Surgical_Cleaning_Robot/robot_control/Force_Sensing/
# Files:
#   force_sensor.py
#   force_reader.py
#   force_sensor_calibrate.py
# --------------------------------------------------

THIS_FILE = Path(__file__).resolve()
FORCE_SENSING_DIR = THIS_FILE.parent

sys.path.insert(0, str(FORCE_SENSING_DIR))


# --------------------------------------------------
# Import reusable force sensor class
# --------------------------------------------------

from force_sensor import ForceSensor, SENSOR_PROFILES


def build_arg_parser():
    """
    Build command-line arguments for live force reading.
    """

    parser = argparse.ArgumentParser(
        description="Read raw or calibrated force sensor data."
    )

    parser.add_argument(
        "--profile",
        choices=list(SENSOR_PROFILES.keys()),
        required=True,
        help=(
            "Sensor profile to use. "
            "Use qlmh41_human for human sensor. "
            "Use qlmh25_robot for robot sensor."
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
        "--hz",
        type=float,
        default=100.0,
        help="Reading frequency in Hz. Default: 100 Hz.",
    )

    parser.add_argument(
        "--raw",
        action="store_true",
        help="Read raw ADC data only instead of calibrated force.",
    )

    parser.add_argument(
        "--force-sign",
        type=float,
        choices=[-1.0, 1.0],
        default=1.0,
        help="Use -1 if force direction is reversed. Default: 1.",
    )

    return parser


def main():
    """
    Main live reading loop.
    """

    args = build_arg_parser().parse_args()

    sensor = ForceSensor(
        profile_name=args.profile,
        port=args.port,
        baud=args.baud,
        force_sign=args.force_sign,
    )

    dt = 1.0 / args.hz

    print("")
    print("--------------------------------------------------")
    print("FORCE SENSOR READER")
    print("--------------------------------------------------")
    print(f"Profile: {args.profile}")
    print(f"Port: {args.port}")
    print(f"Baud: {args.baud}")
    print(f"Rate: {args.hz} Hz")
    print(f"Mode: {'RAW ADC' if args.raw else 'CALIBRATED FORCE'}")
    print("Press CTRL+C to stop.")
    print("--------------------------------------------------")
    print("")

    try:
        while True:
            if args.raw:
                data = sensor.read_raw()

                if data is not None:
                    print(
                        f"Raw ADC: {data['raw_adc']:.0f} | "
                        f"Arduino Time: {data['arduino_time_ms']:.0f} ms | "
                        f"PC Time: {data['pc_time_s']:.3f}"
                    )

            else:
                data = sensor.read_force()

                if data is not None:
                    print(
                        f"Raw: {data['raw_adc']:.0f} | "
                        f"Tared: {data['tared_adc']:.0f} | "
                        f"Force: {data['force_N']:.4f} N | "
                        f"Arduino Time: {data['arduino_time_ms']:.0f} ms | "
                        f"PC Time: {data['pc_time_s']:.3f}"
                    )

            time.sleep(dt)

    except KeyboardInterrupt:
        print("")
        print("Shutting down...")

    finally:
        sensor.disconnect()
        print("Force sensor disconnected.")


if __name__ == "__main__":
    main()