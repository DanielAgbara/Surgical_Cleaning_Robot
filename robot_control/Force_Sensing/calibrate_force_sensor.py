#!/usr/bin/env python3

"""
calibrate_force_sensor.py

Interactive calibration script for force_sensor.py.

This script uses the ForceSensor class, so all serial communication stays
inside force_sensor.py.

Works for:
    - human force sensor
    - robot force sensor

Example usage:
    python calibrate_force_sensor.py --profile human_sensor --port /dev/ttyACM0

    python calibrate_force_sensor.py --profile robot_sensor --port /dev/ttyACM1

Assumption:
    Arduino sends one numeric raw sensor value per line.

Calibration formula:
    known_mass_kg = known_mass_g / 1000
    known_force_N = known_mass_kg * 9.80665

    offset = average zero-load reading
    delta = loaded_mean - offset

    newtons_per_unit = known_force_N / delta

    force_N = (raw_reading - offset) * newtons_per_unit
"""

import argparse
import time

import numpy as np

from force_sensor import ForceSensor


def ask_known_mass_g():
    """
    Ask user for the known calibration mass in grams.

    Example valid inputs:
        100
        200
        500
    """

    while True:
        user_input = input(
            "Enter known calibration mass in grams, e.g. 100, 200, 500: "
        ).strip()

        try:
            mass_g = float(user_input)

            if mass_g <= 0:
                print("Mass must be greater than zero.")
                continue

            return mass_g

        except ValueError:
            print("Invalid input. Please enter a number like 100, 200, or 500.")


def ask_yes_no(prompt):
    """
    Ask a yes/no question.

    Returns
    -------
    bool
        True for yes.
        False for no.
    """

    while True:
        user_input = input(prompt).strip().lower()

        if user_input in ["y", "yes"]:
            return True

        if user_input in ["n", "no"]:
            return False

        print("Please enter y or n.")


def show_calibrated_readings(
    sensor,
    offset,
    newtons_per_unit,
    num_readings=50,
):
    """
    Show converted readings after calibration.

    This lets you check if the calibration looks reasonable.

    If you keep a 200 g mass on the sensor, the displayed weight should be
    close to 200 g.
    """

    print("")
    print("--------------------------------------------------")
    print(f"SHOWING {num_readings} CALIBRATED READINGS")
    print("--------------------------------------------------")
    print("Using:")
    print("    force_N = (raw_reading - offset) * newtons_per_unit")
    print("")

    count = 0

    while count < num_readings:
        raw_value = sensor.read_numeric_line()

        if raw_value is None:
            time.sleep(0.005)
            continue

        force_n = (raw_value - offset) * newtons_per_unit
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
    """
    Run interactive force sensor calibration.

    Steps:
        1. Connect to force sensor.
        2. Ask user to remove load.
        3. Send Arduino tare command.
        4. Collect zero-load readings.
        5. Ask user for known mass in grams.
        6. Collect loaded readings.
        7. Compute calibration scale.
        8. Save calibration JSON.
        9. Show calibrated readings for validation.
        10. Ask if user wants to try another known mass.
    """

    sensor = ForceSensor(
        port=port,
        baud_rate=baud_rate,
        print_data=False,

        # During calibration, we want the raw numeric values.
        input_mode="raw_units",

        # Do not load a profile because we are creating/updating one.
        profile_name=None,

        calibration_dir=calibration_dir,
    )

    try:
        print("")
        print("--------------------------------------------------")
        print("FORCE SENSOR CALIBRATION")
        print("--------------------------------------------------")
        print(f"Profile name: {profile_name}")
        print(f"Port: {port}")
        print(f"Baud rate: {baud_rate}")
        print(f"Samples per step: {samples}")
        print("--------------------------------------------------")
        print("")

        # --------------------------------------------------
        # Step 1: Tare / zero the sensor first
        # --------------------------------------------------

        input("Remove all load from the sensor, then press ENTER to tare...")

        print("Sending tare command to Arduino...")
        sensor.tare()

        # Give Arduino time to apply its internal tare.
        time.sleep(1.0)

        # Clear any old readings or Arduino messages after tare.
        if sensor.serial_connection is not None:
            sensor.serial_connection.reset_input_buffer()

        print("Tare command complete.")
        print("Collecting zero-load readings...")

        zero_samples = sensor.collect_raw_samples(num_samples=samples)

        offset = float(np.mean(zero_samples))
        zero_std = float(np.std(zero_samples))

        print("")
        print("Zero measurement after tare:")
        print(f"Offset mean: {offset:.6f}")
        print(f"Offset std:  {zero_std:.6f}")
        print("")

        # --------------------------------------------------
        # Step 2: Calibration loop
        # --------------------------------------------------

        while True:
            known_mass_g = ask_known_mass_g()
            known_mass_kg = known_mass_g / 1000.0

            input(
                f"Place {known_mass_g:.1f} g on the sensor, then press ENTER..."
            )

            # Clear transition readings while the user is placing the weight.
            if sensor.serial_connection is not None:
                sensor.serial_connection.reset_input_buffer()

            loaded_samples = sensor.collect_raw_samples(num_samples=samples)

            loaded_mean = float(np.mean(loaded_samples))
            loaded_std = float(np.std(loaded_samples))

            known_force_n = known_mass_kg * sensor.g
            delta = loaded_mean - offset

            if abs(delta) < 1e-9:
                print("")
                print("Calibration failed.")
                print("Loaded reading is too close to zero reading.")
                print("Try again with a heavier known mass.")
                print("")
                continue

            newtons_per_unit = known_force_n / delta

            # Save calibration profile.
            sensor.save_calibration(
                profile_name=profile_name,
                offset=offset,
                newtons_per_unit=newtons_per_unit,
                known_mass_kg=known_mass_kg,
                known_force_n=known_force_n,
                zero_mean=offset,
                loaded_mean=loaded_mean,
                zero_std=zero_std,
                loaded_std=loaded_std,
                samples=samples,
            )

            print("")
            print("--------------------------------------------------")
            print("CALIBRATION COMPLETE")
            print("--------------------------------------------------")
            print(f"Known mass:       {known_mass_g:.2f} g")
            print(f"Known force:      {known_force_n:.6f} N")
            print(f"Zero offset:      {offset:.6f}")
            print(f"Loaded mean:      {loaded_mean:.6f}")
            print(f"Loaded std:       {loaded_std:.6f}")
            print(f"Delta reading:    {delta:.6f}")
            print(f"Newtons per unit: {newtons_per_unit:.9f}")
            print("")
            print("Formula:")
            print("    force_N = (raw_reading - offset) * newtons_per_unit")
            print("--------------------------------------------------")
            print("")

            # --------------------------------------------------
            # Step 3: Show calibrated readings for validation
            # --------------------------------------------------

            if check_readings > 0:
                input(
                    f"Keep the {known_mass_g:.1f} g mass on the sensor and "
                    f"press ENTER to show {check_readings} calibrated readings..."
                )

                show_calibrated_readings(
                    sensor=sensor,
                    offset=offset,
                    newtons_per_unit=newtons_per_unit,
                    num_readings=check_readings,
                )

            # --------------------------------------------------
            # Step 4: Ask if user wants another known weight
            # --------------------------------------------------

            repeat = ask_yes_no(
                "Do you want to calibrate/test another known weight? [y/n]: "
            )

            if not repeat:
                print("Calibration finished.")
                break

            print("")
            input(
                "Remove the weight from the sensor, then press ENTER "
                "to continue using the same zero offset..."
            )

            if sensor.serial_connection is not None:
                sensor.serial_connection.reset_input_buffer()

    finally:
        sensor.close()


def main():
    """
    Parse command-line arguments and start calibration.
    """

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
        help="Number of samples to average at each calibration step.",
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