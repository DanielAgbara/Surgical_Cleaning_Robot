#!/usr/bin/env python3

"""
calibrate_force_sensor.py

Multi-weight force sensor calibration.

This uses ForceSensor from force_sensor.py.

Instead of calibrating from only one known mass, this script collects
multiple known masses and fits:

    force_N = slope * raw_value + intercept

This is better because it uses all calibration points together.

Example:
    python calibrate_force_sensor.py --profile human_sensor --port /dev/ttyACM0
    python calibrate_force_sensor.py --profile robot_sensor --port /dev/ttyACM1
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from force_sensor import ForceSensor


def ask_yes_no(prompt):
    while True:
        ans = input(prompt).strip().lower()

        if ans in ["y", "yes"]:
            return True

        if ans in ["n", "no"]:
            return False

        print("Please enter y or n.")


def ask_known_mass_g():
    while True:
        ans = input(
            "Enter known mass in grams, e.g. 0, 100, 200, 500: "
        ).strip()

        try:
            mass_g = float(ans)

            if mass_g < 0:
                print("Mass cannot be negative.")
                continue

            return mass_g

        except ValueError:
            print("Please enter a valid number.")


def save_linear_calibration(
    calibration_dir,
    profile_name,
    port,
    baud_rate,
    gravity,
    slope,
    intercept,
    calibration_points,
    samples,
):
    """
    Save linear calibration JSON.

    Formula:
        force_N = slope * raw_value + intercept
    """

    calibration_dir = Path(calibration_dir)
    calibration_dir.mkdir(parents=True, exist_ok=True)

    calibration_file = calibration_dir / f"{profile_name}_calibration.json"

    data = {
        "profile_name": profile_name,
        "calibration_type": "linear_regression",
        "port": port,
        "baud_rate": baud_rate,
        "gravity": gravity,
        "slope": float(slope),
        "intercept": float(intercept),
        "formula": "force_N = slope * raw_value + intercept",
        "samples_per_mass": samples,
        "calibration_points": calibration_points,
        "timestamp": time.time(),
    }

    with open(calibration_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Saved calibration to: {calibration_file}")


def show_calibrated_readings(sensor, slope, intercept, num_readings=50):
    """
    Show live readings using the fitted line.
    """

    print("")
    print("--------------------------------------------------")
    print(f"SHOWING {num_readings} CALIBRATED READINGS")
    print("--------------------------------------------------")
    print("Using:")
    print("    force_N = slope * raw_value + intercept")
    print("")

    count = 0

    while count < num_readings:
        raw_value = sensor.read_numeric_line()

        if raw_value is None:
            time.sleep(0.005)
            continue

        force_n = slope * raw_value + intercept
        mass_kg = force_n / sensor.g
        weight_g = mass_kg * 1000.0

        print(
            f"{count + 1:02d} | "
            f"Raw: {raw_value:.3f} | "
            f"Weight: {weight_g:.2f} g | "
            f"Force: {force_n:.4f} N"
        )

        count += 1
        time.sleep(0.02)

    print("--------------------------------------------------")
    print("")


def run_calibration(
    profile_name,
    port,
    baud_rate,
    samples,
    calibration_dir,
    check_readings,
):
    sensor = ForceSensor(
        port=port,
        baud_rate=baud_rate,
        print_data=False,
        input_mode="raw_units",
        profile_name=None,
        calibration_dir=calibration_dir,
    )

    calibration_points = []

    try:
        print("")
        print("--------------------------------------------------")
        print("FORCE SENSOR CALIBRATION")
        print("--------------------------------------------------")
        print(f"Profile: {profile_name}")
        print(f"Port: {port}")
        print(f"Baud: {baud_rate}")
        print(f"Samples per mass: {samples}")
        print("--------------------------------------------------")
        print("")

        input("Remove all load from the sensor, then press ENTER to tare...")

        sensor.tare()
        time.sleep(1.0)

        if sensor.serial_connection is not None:
            sensor.serial_connection.reset_input_buffer()

        print("Tare complete.")
        print("")

        print("First, collect the zero-load point.")
        input("Make sure there is 0 g on the sensor, then press ENTER...")

        zero_samples = sensor.collect_raw_samples(num_samples=samples)
        zero_raw_mean = float(np.mean(zero_samples))
        zero_raw_std = float(np.std(zero_samples))

        calibration_points.append(
            {
                "mass_g": 0.0,
                "force_N": 0.0,
                "raw_mean": zero_raw_mean,
                "raw_std": zero_raw_std,
            }
        )

        print("")
        print("Zero point collected:")
        print(f"Raw mean: {zero_raw_mean:.6f}")
        print(f"Raw std:  {zero_raw_std:.6f}")
        print("")

        while True:
            mass_g = ask_known_mass_g()

            if mass_g == 0:
                print("Zero point already collected. Use a nonzero mass.")
                continue

            mass_kg = mass_g / 1000.0
            force_n = mass_kg * sensor.g

            input(f"Place {mass_g:.1f} g on the sensor, then press ENTER...")

            if sensor.serial_connection is not None:
                sensor.serial_connection.reset_input_buffer()

            raw_samples = sensor.collect_raw_samples(num_samples=samples)
            raw_mean = float(np.mean(raw_samples))
            raw_std = float(np.std(raw_samples))

            calibration_points.append(
                {
                    "mass_g": float(mass_g),
                    "force_N": float(force_n),
                    "raw_mean": raw_mean,
                    "raw_std": raw_std,
                }
            )

            print("")
            print("Point collected:")
            print(f"Mass:      {mass_g:.2f} g")
            print(f"Force:     {force_n:.6f} N")
            print(f"Raw mean:  {raw_mean:.6f}")
            print(f"Raw std:   {raw_std:.6f}")
            print("")

            if len(calibration_points) >= 3:
                done = ask_yes_no(
                    "Do you want to finish and fit calibration now? [y/n]: "
                )

                if done:
                    break
            else:
                print("Collect at least 2 nonzero masses if possible.")
                print("Example: 100 g, 200 g, 500 g.")
                print("")

        raw_values = np.array(
            [p["raw_mean"] for p in calibration_points],
            dtype=float,
        )

        force_values = np.array(
            [p["force_N"] for p in calibration_points],
            dtype=float,
        )

        # Fit force_N = slope * raw_value + intercept
        slope = np.sum(raw_values * force_values) / np.sum(raw_values ** 2)
        intercept = 0.0

        predicted_force = slope * raw_values + intercept
        residuals = force_values - predicted_force
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        save_linear_calibration(
            calibration_dir=calibration_dir,
            profile_name=profile_name,
            port=port,
            baud_rate=baud_rate,
            gravity=sensor.g,
            slope=slope,
            intercept=intercept,
            calibration_points=calibration_points,
            samples=samples,
        )

        print("")
        print("--------------------------------------------------")
        print("LINEAR CALIBRATION COMPLETE")
        print("--------------------------------------------------")
        print("Formula:")
        print("    force_N = slope * raw_value + intercept")
        print("")
        print(f"Slope:     {slope:.12f}")
        print(f"Intercept: {intercept:.12f}")
        print(f"RMSE:      {rmse:.6f} N")
        print("--------------------------------------------------")
        print("")

        print("Calibration points:")
        for p, pred, err in zip(calibration_points, predicted_force, residuals):
            print(
                f"Mass: {p['mass_g']:8.2f} g | "
                f"Raw mean: {p['raw_mean']:12.3f} | "
                f"Actual: {p['force_N']:9.5f} N | "
                f"Predicted: {pred:9.5f} N | "
                f"Error: {err:9.5f} N"
            )

        print("")

        if check_readings > 0:
            input(
                "Keep any known mass on the sensor and press ENTER "
                f"to show {check_readings} calibrated readings..."
            )

            show_calibrated_readings(
                sensor=sensor,
                slope=slope,
                intercept=intercept,
                num_readings=check_readings,
            )

    finally:
        sensor.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--profile",
        type=str,
        required=True,
        help="Profile name, e.g. human_sensor or robot_sensor",
    )

    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port, e.g. /dev/ttyACM0 or /dev/ttyACM1",
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=9600,
        help="Baud rate. Must match Arduino Serial.begin().",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of samples to average for each known mass.",
    )

    parser.add_argument(
        "--calibration-dir",
        type=str,
        default="calibration",
        help="Folder where calibration JSON files are saved.",
    )

    parser.add_argument(
        "--check-readings",
        type=int,
        default=50,
        help="Number of calibrated readings to display after calibration.",
    )

    args = parser.parse_args()

    run_calibration(
        profile_name=args.profile,
        port=args.port,
        baud_rate=args.baud,
        samples=args.samples,
        calibration_dir=args.calibration_dir,
        check_readings=args.check_readings,
    )


if __name__ == "__main__":
    main()