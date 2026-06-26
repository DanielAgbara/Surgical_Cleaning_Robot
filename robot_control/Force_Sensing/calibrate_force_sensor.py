#!/usr/bin/env python3

"""
calibrate_force_sensor.py

Calibration script that uses ForceSensor from force_sensor.py.

This is adaptable to:
    - human force sensor
    - robot force sensor

Example usage:

Human sensor:
    python calibrate_force_sensor.py --profile human_sensor --port /dev/ttyACM0 --mass 0.2

Robot sensor:
    python calibrate_force_sensor.py --profile robot_sensor --port /dev/ttyACM1 --mass 0.2

Important:
    This script is mainly useful when Arduino sends RAW sensor units.

If your Arduino already sends calibrated grams, then you may not need this script.
"""

import argparse
import numpy as np

from force_sensor import ForceSensor


def run_calibration(
    profile_name,
    port,
    baud_rate,
    known_mass_kg,
    samples,
    calibration_dir,
):
    """
    Run two-point calibration.

    Step 1:
        Read sensor with no load.
        Average value becomes offset.

    Step 2:
        Read sensor with known mass.
        Difference between loaded average and offset gives scale.

    Calculation:
        known_force_N = known_mass_kg * 9.80665

        delta = loaded_mean - zero_mean

        newtons_per_unit = known_force_N / delta

        force_N = (raw_reading - offset) * newtons_per_unit
    """

    sensor = ForceSensor(
        port=port,
        baud_rate=baud_rate,
        print_data=False,

        # Important:
        # We want raw values during calibration.
        input_mode="raw_units",

        # Do not load profile yet because we are creating one.
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
        print(f"Known mass: {known_mass_kg} kg")
        print(f"Samples per step: {samples}")
        print("--------------------------------------------------")
        print("")

        input("Remove all load from the sensor, then press ENTER...")

        zero_samples = sensor.collect_raw_samples(num_samples=samples)

        zero_mean = float(np.mean(zero_samples))
        zero_std = float(np.std(zero_samples))

        print("")
        print("Zero measurement:")
        print(f"Mean: {zero_mean:.6f}")
        print(f"Std:  {zero_std:.6f}")
        print("")

        input(f"Place {known_mass_kg} kg on the sensor, then press ENTER...")

        loaded_samples = sensor.collect_raw_samples(num_samples=samples)

        loaded_mean = float(np.mean(loaded_samples))
        loaded_std = float(np.std(loaded_samples))

        print("")
        print("Loaded measurement:")
        print(f"Mean: {loaded_mean:.6f}")
        print(f"Std:  {loaded_std:.6f}")
        print("")

        known_force_n = known_mass_kg * sensor.g
        delta = loaded_mean - zero_mean

        if abs(delta) < 1e-9:
            raise RuntimeError(
                "Calibration failed. Loaded reading is too close to zero reading."
            )

        newtons_per_unit = known_force_n / delta

        sensor.save_calibration(
            profile_name=profile_name,
            offset=zero_mean,
            newtons_per_unit=newtons_per_unit,
            known_mass_kg=known_mass_kg,
            known_force_n=known_force_n,
            zero_mean=zero_mean,
            loaded_mean=loaded_mean,
            zero_std=zero_std,
            loaded_std=loaded_std,
            samples=samples,
        )

        print("")
        print("--------------------------------------------------")
        print("CALIBRATION COMPLETE")
        print("--------------------------------------------------")
        print(f"Offset:           {zero_mean:.6f}")
        print(f"Known force:      {known_force_n:.6f} N")
        print(f"Delta reading:    {delta:.6f}")
        print(f"Newtons per unit: {newtons_per_unit:.9f}")
        print("")
        print("Use this formula:")
        print("    force_N = (raw_reading - offset) * newtons_per_unit")
        print("")

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
        help="Serial port, e.g. /dev/ttyACM0",
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=9600,
        help="Baud rate",
    )

    parser.add_argument(
        "--mass",
        type=float,
        default=0.2,
        help="Known calibration mass in kg. Example: 0.2 for 200 g",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of samples to average at each calibration step",
    )

    parser.add_argument(
        "--calibration-dir",
        type=str,
        default="calibration",
        help="Folder where calibration JSON files are saved",
    )

    args = parser.parse_args()

    run_calibration(
        profile_name=args.profile,
        port=args.port,
        baud_rate=args.baud,
        known_mass_kg=args.mass,
        samples=args.samples,
        calibration_dir=args.calibration_dir,
    )


if __name__ == "__main__":
    main()