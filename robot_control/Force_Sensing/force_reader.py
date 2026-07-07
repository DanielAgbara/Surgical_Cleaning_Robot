#!/usr/bin/env python3

"""
force_reader.py

Runtime force sensor reader.

This script uses calibration JSON files created by calibrate_force_sensor.py.

Example calibration files:

    calibration/human_sensor_calibration.json
    calibration/robot_sensor_calibration.json

Example usage:

   

    python force_reader.py --profile robot_sensor --port /dev/ttyACM1

    python force_reader.py --profile human_sensor --port /dev/ttyACM0 --raw

Notes:
    - This assumes Arduino sends raw numeric sensor values.
    - Python loads offset and newtons_per_unit from the calibration JSON file.
    - The reader sends a tare command before reading.
"""

import argparse
import sys
import time
from pathlib import Path


# --------------------------------------------------
# Path setup
# --------------------------------------------------

THIS_FILE = Path(__file__).resolve()
FORCE_SENSING_DIR = THIS_FILE.parent

sys.path.insert(0, str(FORCE_SENSING_DIR))


# --------------------------------------------------
# Import reusable force sensor class
# --------------------------------------------------

from force_sensor import ForceSensor


def build_arg_parser():
    """
    Build command-line arguments for live force reading.
    """

    parser = argparse.ArgumentParser(
        description="Read raw or calibrated force sensor data."
    )

    parser.add_argument(
        "--profile",
        type=str,
        required=True,
        help=(
            "Calibration profile name. "
            "Example: human_sensor or robot_sensor. "
            "This loads calibration/<profile>_calibration.json"
        ),
    )

    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM0",
        help="Arduino serial port. Default: /dev/ttyACM0",
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=9600,
        help="Arduino baud rate. Default: 9600",
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
        help="Print raw sensor readings only.",
    )

    parser.add_argument(
        "--force-sign",
        type=float,
        choices=[-1.0, 1.0],
        default=1.0,
        help="Use -1 if force direction is reversed. Default: 1.",
    )

    parser.add_argument(
        "--calibration-dir",
        type=str,
        default="calibration",
        help="Folder containing calibration JSON files.",
    )

    parser.add_argument(
        "--no-tare",
        action="store_true",
        help="Do not send tare command before reading.",
    )

    return parser


def main():
    """
    Main live reading loop.
    """

    args = build_arg_parser().parse_args()

    dt = 1.0 / args.hz

    sensor = ForceSensor(
        port=args.port,
        baud_rate=args.baud,
        print_data=False,
        input_mode="raw_units",
        profile_name=args.profile,
        calibration_dir=args.calibration_dir,
    )

    print("")
    print("--------------------------------------------------")
    print("FORCE SENSOR READER")
    print("--------------------------------------------------")
    print(f"Profile: {args.profile}")
    print(f"Port: {args.port}")
    print(f"Baud: {args.baud}")
    print(f"Rate: {args.hz} Hz")
    print(f"Calibration dir: {args.calibration_dir}")
    print(f"Mode: {'RAW SENSOR VALUES' if args.raw else 'CALIBRATED FORCE'}")
    print(f"Force sign: {args.force_sign}")
    print("Press CTRL+C to stop.")
    print("--------------------------------------------------")
    print("")

    try:
        # --------------------------------------------------
        # Tare before reading
        # --------------------------------------------------

        if not args.no_tare:
            input("Remove all load from the sensor, then press ENTER to tare...")

            print("Sending tare command...")
            sensor.tare()

            # Give Arduino time to apply tare.
            time.sleep(1.0)

            # Clear old readings after tare.
            if sensor.serial_connection is not None:
                sensor.serial_connection.reset_input_buffer()

            print("Tare complete. Starting live readings...")
            print("")

        # --------------------------------------------------
        # Live reading loop
        # --------------------------------------------------

        while True:
            if args.raw:
                raw_value = sensor.read_numeric_line()

                if raw_value is not None:
                    print(f"Raw value: {raw_value:.3f}")

            else:
                data = sensor.read()

                if data is not None:
                    raw_value = data["raw_value"]
                    force_n = data["force_n"] * args.force_sign
                    mass_kg = force_n / sensor.g
                    weight_g = mass_kg * 1000.0

                    tared_value = raw_value - sensor.offset

                    print(
                        f"Raw: {raw_value:.3f} | "
                        f"Tared: {tared_value:.3f} | "
                        f"Weight: {weight_g:.2f} g | "
                        f"Force: {force_n:.4f} N | "
                        f"Time: {data['timestamp']:.3f}"
                    )

            time.sleep(dt)

    except KeyboardInterrupt:
        print("")
        print("Shutting down...")

    finally:
        sensor.close()
        print("Force sensor disconnected.")


if __name__ == "__main__":
    main()