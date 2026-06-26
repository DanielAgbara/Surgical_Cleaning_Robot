#!/usr/bin/env python3

"""
force_sensor.py

Reusable force sensor driver using direct serial communication.

This version does NOT use Haplink.

It assumes the Arduino sends one numeric value per line over serial.

The Arduino value can be either:
    1. raw ADC / HX711 counts
    2. already-calibrated grams

This class supports both.

Recommended project structure:

Force_Sensing/
│
├── force_sensor.py
├── calibrate_force_sensor.py
└── calibration/
    ├── human_sensor_calibration.json
    └── robot_sensor_calibration.json
"""

import serial
import time
import json
from pathlib import Path


class ForceSensor:
    def __init__(
        self,
        port="/dev/ttyACM0",
        baud_rate=9600,
        timeout=0.1,
        gravity=9.80665,
        print_data=False,
        profile_name=None,
        calibration_dir="calibration",
        input_mode="calibrated_grams",
    ):
        """
        Initialize force sensor.

        Parameters
        ----------
        port : str
            Serial port.
            Linux example: "/dev/ttyACM0"
            Windows example: "COM5"

        baud_rate : int
            Must match Arduino Serial.begin().

        timeout : float
            Serial timeout in seconds.

        gravity : float
            Gravity used for force conversion.

        print_data : bool
            Print readings if True.

        profile_name : str or None
            Calibration profile to load.
            Example: "human_sensor" or "robot_sensor"

        calibration_dir : str
            Folder where calibration JSON files are stored.

        input_mode : str
            "calibrated_grams":
                Arduino already sends grams.

            "raw_units":
                Arduino sends raw ADC/count values.
                Python uses calibration profile to convert to Newtons.
        """

        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.g = gravity
        self.print_data = print_data
        self.input_mode = input_mode

        self.serial_connection = None

        # Calibration data
        self.profile_name = profile_name
        self.calibration_dir = Path(calibration_dir)

        self.offset = 0.0
        self.newtons_per_unit = None

        # Latest values
        self.latest_raw_value = None
        self.latest_weight_g = None
        self.latest_mass_kg = None
        self.latest_force_n = None
        self.latest_time = None

        self.connect()

        if self.profile_name is not None:
            self.load_calibration(self.profile_name)

    def connect(self):
        """
        Open serial connection to Arduino.
        """

        try:
            print(f"Opening force sensor serial port: {self.port}")

            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout,
            )

            # Arduino often resets when serial opens.
            time.sleep(2.0)

            # Remove old startup text or stale data.
            self.serial_connection.reset_input_buffer()

            print("Force sensor connected.")

        except Exception as e:
            raise RuntimeError(
                f"Error opening force sensor port {self.port}: {e}"
            )

    def close(self):
        """
        Close serial connection.
        """

        if self.serial_connection is not None:
            try:
                self.serial_connection.close()
                print("Force sensor serial connection closed.")
            except Exception:
                pass

            self.serial_connection = None

    def tare(self):
        """
        Send tare command to Arduino.

        This assumes Arduino listens for character 't'.
        """

        if self.serial_connection is None:
            print("Cannot tare. Serial connection is not open.")
            return

        try:
            self.serial_connection.write(b"t")
            self.serial_connection.flush()
            print("Tare command sent.")

        except Exception as e:
            print(f"Error sending tare command: {e}")

    def calibrate_arduino(self):
        """
        Send calibration command to Arduino.

        This assumes Arduino listens for character 'c'.
        """

        if self.serial_connection is None:
            print("Cannot calibrate. Serial connection is not open.")
            return

        try:
            self.serial_connection.write(b"c")
            self.serial_connection.flush()
            print("Arduino calibration command sent.")

        except Exception as e:
            print(f"Error sending calibration command: {e}")

    def read_numeric_line(self):
        """
        Read one numeric value from serial.

        Returns
        -------
        float or None
            Returns None if no valid numeric line is available.
        """

        if self.serial_connection is None:
            return None

        try:
            if self.serial_connection.in_waiting <= 0:
                return None

            raw_line = self.serial_connection.readline()

            if not raw_line:
                return None

            line = raw_line.decode("utf-8", errors="ignore").strip()

            if not line:
                return None

            try:
                value = float(line)
                return value
            except ValueError:
                # Ignore Arduino messages like:
                # "Tare complete"
                # "Calibration complete"
                return None

        except Exception as e:
            print(f"Serial read error: {e}")
            return None

    def read(self):
        """
        Read one force sensor sample.

        Returns
        -------
        dict or None

        If valid data is available, returns:

        {
            "raw_value": ...,
            "weight_g": ...,
            "mass_kg": ...,
            "force_n": ...,
            "timestamp": ...
        }
        """

        raw_value = self.read_numeric_line()

        if raw_value is None:
            return None

        timestamp = time.time()

        self.latest_raw_value = raw_value
        self.latest_time = timestamp

        if self.input_mode == "calibrated_grams":
            # Arduino is already sending grams.
            weight_g = raw_value
            mass_kg = weight_g / 1000.0
            force_n = mass_kg * self.g

        elif self.input_mode == "raw_units":
            # Arduino sends raw sensor units.
            # Python calibration converts raw units to Newtons.
            if self.newtons_per_unit is None:
                print("No calibration loaded. Cannot convert raw units to force.")
                return None

            force_n = (raw_value - self.offset) * self.newtons_per_unit
            mass_kg = force_n / self.g
            weight_g = mass_kg * 1000.0

        else:
            raise ValueError(
                "input_mode must be either 'calibrated_grams' or 'raw_units'"
            )

        self.latest_weight_g = weight_g
        self.latest_mass_kg = mass_kg
        self.latest_force_n = force_n

        data = {
            "raw_value": raw_value,
            "weight_g": weight_g,
            "mass_kg": mass_kg,
            "force_n": force_n,
            "timestamp": timestamp,
        }

        if self.print_data:
            print(
                f"Raw: {raw_value:.4f} | "
                f"Weight: {weight_g:.2f} g | "
                f"Force: {force_n:.4f} N"
            )

        return data

    def collect_raw_samples(self, num_samples=200):
        """
        Collect raw numeric samples from Arduino.

        This is used by the calibration script.
        """

        samples = []

        print(f"Collecting {num_samples} valid samples...")

        while len(samples) < num_samples:
            value = self.read_numeric_line()

            if value is not None:
                samples.append(value)
                print(f"\rSamples: {len(samples)}/{num_samples}", end="")

            time.sleep(0.005)

        print("")
        return samples

    def save_calibration(
        self,
        profile_name,
        offset,
        newtons_per_unit,
        known_mass_kg,
        known_force_n,
        zero_mean,
        loaded_mean,
        zero_std,
        loaded_std,
        samples,
    ):
        """
        Save calibration profile as JSON.
        """

        self.calibration_dir.mkdir(parents=True, exist_ok=True)

        calibration_file = self.calibration_dir / f"{profile_name}_calibration.json"

        data = {
            "profile_name": profile_name,
            "port": self.port,
            "baud_rate": self.baud_rate,
            "gravity": self.g,
            "offset": offset,
            "newtons_per_unit": newtons_per_unit,
            "known_mass_kg": known_mass_kg,
            "known_force_n": known_force_n,
            "zero_mean": zero_mean,
            "loaded_mean": loaded_mean,
            "zero_std": zero_std,
            "loaded_std": loaded_std,
            "samples": samples,
            "timestamp": time.time(),
        }

        with open(calibration_file, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Calibration saved to: {calibration_file}")

    def load_calibration(self, profile_name):
        """
        Load calibration profile from JSON.
        """

        calibration_file = self.calibration_dir / f"{profile_name}_calibration.json"

        if not calibration_file.exists():
            raise FileNotFoundError(
                f"Calibration file not found: {calibration_file}"
            )

        with open(calibration_file, "r") as f:
            data = json.load(f)

        self.profile_name = profile_name
        self.offset = float(data["offset"])
        self.newtons_per_unit = float(data["newtons_per_unit"])

        print(f"Loaded calibration profile: {profile_name}")
        print(f"Offset: {self.offset:.6f}")
        print(f"Newtons/unit: {self.newtons_per_unit:.9f}")

    def get_latest_force(self):
        """
        Return latest force in Newtons.
        """

        return self.latest_force_n

    def __del__(self):
        self.close()


if __name__ == "__main__":
    """
    Standalone test mode.

    Controls:
        t  -> tare Arduino
        c  -> Arduino calibration command
        q  -> quit

    No Enter needed.
    """

    import sys
    import tty
    import termios
    import select

    PORT = "/dev/ttyACM0"
    BAUD_RATE = 9600

    def get_key_nonblocking():
        """
        Read one keyboard key without Enter.
        """

        dr, _, _ = select.select([sys.stdin], [], [], 0)

        if dr:
            return sys.stdin.read(1).lower()

        return None

    sensor = ForceSensor(
        port=PORT,
        baud_rate=BAUD_RATE,
        print_data=True,

        # Use this if Arduino sends grams.
        input_mode="calibrated_grams",

        # Use this later if Arduino sends raw values and Python handles calibration.
        # input_mode="raw_units",
        profile_name="human_sensor",
    )

    print("")
    print("--------------------------------------------------")
    print("Force sensor active.")
    print("Press 't' to tare.")
    print("Press 'c' to send Arduino calibration command.")
    print("Press 'q' to quit.")
    print("--------------------------------------------------")
    print("")

    old_terminal_settings = termios.tcgetattr(sys.stdin)

    try:
        tty.setcbreak(sys.stdin.fileno())

        while True:
            sensor.read()

            key = get_key_nonblocking()

            if key == "t":
                sensor.tare()

            elif key == "c":
                sensor.calibrate_arduino()

            elif key == "q":
                print("Quitting...")
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nShutting down...")

    finally:
        termios.tcsetattr(
            sys.stdin,
            termios.TCSADRAIN,
            old_terminal_settings,
        )

        sensor.close()