#!/usr/bin/env python3
"""
force_sensor.py

Reusable force sensor module for the Surgical Cleaning Robot project.

This file contains:

1. ForceSensor class
   - Connects to Arduino/Haplink
   - Reads raw ADC data
   - Converts ADC to force in Newtons
   - Supports tare
   - Supports calibration with known mass
   - Saves/loads calibration JSON files

2. Built-in sensor profiles
   - qlmh41_human
   - qlmh25_robot

Expected project structure:

Surgical_Cleaning_Robot/
│
├── force_sensor/
│   └── force_sensor.py
│
├── data/
│   └── force_sensor/
│       ├── qlmh41_human_calibration.json
│       └── qlmh25_robot_calibration.json
│
├── vision/
├── robot_control/
└── ...

Arduino/Haplink telemetry must match:

Telemetry ID 0: raw_adc       INT32
Telemetry ID 1: arduino_time  INT32
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

from haplink import Haplink, DataType


# --------------------------------------------------
# Project paths
# --------------------------------------------------

# This assumes this file is located at:
#
# Surgical_Cleaning_Robot/force_sensor/force_sensor.py
#
# parent        -> Surgical_Cleaning_Robot/force_sensor
# parent.parent -> Surgical_Cleaning_Robot
#
ROOT = Path(__file__).resolve().parent.parent

# Calibration files will be saved in:
#
# Surgical_Cleaning_Robot/data/force_sensor/
#
FORCE_SENSOR_DATA_DIR = ROOT / "data" / "force_sensor"


# --------------------------------------------------
# Sensor profiles
# --------------------------------------------------
# These profiles do NOT hard-code calibration constants.
# They only define which physical sensor/setup you are using.
#
# Calibration constants are loaded from JSON after calibration.
#
# qlmh41_human:
#   Human-side force sensor.
#
# qlmh25_robot:
#   Robot/end-effector force sensor.
#
SENSOR_PROFILES = {
    "qlmh41_human": {
        "sensor_model": "QLMH-41",
        "usage": "human",
        "calibration_filename": "qlmh41_human_calibration.json",
    },
    "qlmh25_robot": {
        "sensor_model": "QLMH-25",
        "usage": "robot",
        "calibration_filename": "qlmh25_robot_calibration.json",
    },
}


def get_calibration_file(profile_name: str) -> Path:
    """
    Return the calibration file path for a selected sensor profile.
    """

    if profile_name not in SENSOR_PROFILES:
        raise ValueError(
            f"Unknown profile '{profile_name}'. "
            f"Available profiles: {list(SENSOR_PROFILES.keys())}"
        )

    FORCE_SENSOR_DATA_DIR.mkdir(parents=True, exist_ok=True)

    filename = SENSOR_PROFILES[profile_name]["calibration_filename"]

    return FORCE_SENSOR_DATA_DIR / filename


class ForceSensor:
    """
    Force sensor interface using Haplink telemetry.

    Calibration model:

        tared_adc = raw_adc - offset

        scale = ADC counts per Newton

        force_N = tared_adc / scale

    If the force comes out negative, that usually means the ADC decreases
    when load is applied. That is not automatically wrong. It tells you
    direction.

    You can flip the sign by using force_sign=-1.
    """

    def __init__(
        self,
        profile_name: str,
        port: str = "/dev/ttyUSB0",
        baud: int = 115200,
        timeout: float = 0.001,
        force_sign: float = 1.0,
        auto_connect: bool = True,
    ):
        """
        Initialize force sensor.

        Parameters
        ----------
        profile_name : str
            Sensor profile:
                "qlmh41_human"
                "qlmh25_robot"

        port : str
            Arduino serial port.
            ELEGOO Uno R3 with CH340 usually appears as /dev/ttyUSB0.

        baud : int
            Serial baud rate. Must match Arduino firmware.

        timeout : float
            Serial timeout for Haplink.

        force_sign : float
            Use 1.0 for normal sign.
            Use -1.0 if you want to flip force direction.

        auto_connect : bool
            If True, connect immediately.
        """

        if profile_name not in SENSOR_PROFILES:
            raise ValueError(
                f"Unknown profile '{profile_name}'. "
                f"Available profiles: {list(SENSOR_PROFILES.keys())}"
            )

        self.profile_name = profile_name
        self.profile = SENSOR_PROFILES[profile_name]

        self.sensor_model = self.profile["sensor_model"]
        self.usage = self.profile["usage"]

        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.force_sign = force_sign

        self.g = 9.80665

        self.calibration_file = get_calibration_file(profile_name)

        # Default values before calibration.
        # These will be overwritten when calibration JSON exists.
        self.offset = 0.0
        self.scale = 1.0

        self.connected = False
        self.haplink = None

        self.load_calibration()

        if auto_connect:
            self.connect()

    def connect(self):
        """
        Connect to Arduino/Haplink device.
        """

        if self.connected:
            return

        self.haplink = Haplink(
            self.port,
            baudrate=self.baud,
            timeout=self.timeout,
        )

        if not self.haplink.connect():
            raise RuntimeError(f"Haplink connection failed on {self.port}")

        self.haplink.register_telemetry(0, "raw_adc", DataType.INT32)
        self.haplink.register_telemetry(1, "arduino_time", DataType.INT32)

        self.connected = True

        print("")
        print("Force sensor connected.")
        print(f"Profile: {self.profile_name}")
        print(f"Sensor model: {self.sensor_model}")
        print(f"Usage: {self.usage}")
        print(f"Port: {self.port}")
        print(f"Baud: {self.baud}")
        print(f"Calibration file: {self.calibration_file}")
        print(f"Offset: {self.offset:.2f}")
        print(f"Scale: {self.scale:.6f} ADC/N")
        print(f"Force sign: {self.force_sign}")
        print("")

    def disconnect(self):
        """
        Disconnect from Haplink.
        """

        if self.haplink is not None:
            try:
                self.haplink.disconnect()
            except Exception:
                pass

        self.connected = False

    def load_calibration(self):
        """
        Load offset and scale from calibration JSON if it exists.
        """

        if not self.calibration_file.exists():
            print(f"No calibration file found yet: {self.calibration_file}")
            print("Using default offset=0.0 and scale=1.0")
            return

        with open(self.calibration_file, "r") as f:
            data = json.load(f)

        self.offset = float(data.get("offset", self.offset))
        self.scale = float(data.get("scale", self.scale))
        self.force_sign = float(data.get("force_sign", self.force_sign))

    def save_calibration(self):
        """
        Save calibration values to JSON.
        """

        self.calibration_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "profile_name": self.profile_name,
            "sensor_model": self.sensor_model,
            "usage": self.usage,
            "offset": self.offset,
            "scale": self.scale,
            "scale_units": "ADC counts per Newton",
            "force_units": "Newtons",
            "force_sign": self.force_sign,
            "gravity_m_per_s2": self.g,
            "port_used": self.port,
            "baud_used": self.baud,
            "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(self.calibration_file, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Saved calibration to: {self.calibration_file}")

    def read_raw(self) -> Optional[Dict[str, float]]:
        """
        Read raw telemetry only.

        Returns
        -------
        dict or None
            {
                "raw_adc": float,
                "arduino_time_ms": float,
                "pc_time_s": float
            }
        """

        if not self.connected:
            raise RuntimeError("ForceSensor is not connected.")

        self.haplink.update()

        raw_adc_val = self.haplink.get_telemetry("raw_adc")
        arduino_time_val = self.haplink.get_telemetry("arduino_time")

        if raw_adc_val is None:
            return None

        raw_adc = float(raw_adc_val)

        if arduino_time_val is not None:
            arduino_time = float(arduino_time_val)
        else:
            arduino_time = 0.0

        return {
            "raw_adc": raw_adc,
            "arduino_time_ms": arduino_time,
            "pc_time_s": time.time(),
        }

    def read_force(self) -> Optional[Dict[str, float]]:
        """
        Read calibrated force data.

        Returns
        -------
        dict or None
            {
                "raw_adc": float,
                "tared_adc": float,
                "force_N": float,
                "arduino_time_ms": float,
                "pc_time_s": float
            }
        """

        raw = self.read_raw()

        if raw is None:
            return None

        raw_adc = raw["raw_adc"]
        tared_adc = raw_adc - self.offset

        if abs(self.scale) < 1e-12:
            raise RuntimeError("Invalid scale value. Calibrate sensor first.")

        force_N = self.force_sign * (tared_adc / self.scale)

        return {
            "raw_adc": raw_adc,
            "tared_adc": tared_adc,
            "force_N": force_N,
            "arduino_time_ms": raw["arduino_time_ms"],
            "pc_time_s": raw["pc_time_s"],
        }

    def collect_raw_average(
        self,
        samples: int = 200,
        delay: float = 0.005,
    ) -> float:
        """
        Collect raw ADC samples and average them.
        """

        values = []

        print(f"Collecting {samples} raw ADC samples...")

        while len(values) < samples:
            data = self.read_raw()

            if data is not None:
                values.append(data["raw_adc"])

            time.sleep(delay)

        avg = sum(values) / len(values)

        print(f"Average raw ADC: {avg:.2f}")

        return avg

    def get_default_calibration_masses_kg(self):
        """
        Return default calibration masses based on the selected profile.

        Human sensor:
            50 g, 100 g, 200 g

        Robot sensor:
            50 g, 100 g
        """

        if self.profile_name == "qlmh41_human":
            return [0.05, 0.10, 0.20]

        if self.profile_name == "qlmh25_robot":
            return [0.05, 0.10]

        # Safe fallback if a new profile is added later.
        return [0.05, 0.10, 0.20]


    def collect_force_debug_average(
        self,
        samples: int = 200,
        delay: float = 0.005,
    ):
        """
        Collect raw ADC samples and also print the current estimated force.

        This is mostly for debugging. During calibration, the force shown here
        uses the previous scale value until the new multi-load calibration is
        computed.
        """

        values = []

        print(f"Collecting {samples} samples...")

        while len(values) < samples:
            raw_data = self.read_raw()

            if raw_data is not None:
                raw_adc = raw_data["raw_adc"]
                values.append(raw_adc)

                # Show estimated force using current calibration.
                # This is useful for debugging whether force is increasing/decreasing.
                if abs(self.scale) > 1e-12:
                    tared_adc = raw_adc - self.offset
                    estimated_force_N = self.force_sign * (tared_adc / self.scale)
                else:
                    estimated_force_N = 0.0

                print(
                    f"Sample {len(values):03d}/{samples} | "
                    f"Estimated Force: {estimated_force_N:.4f} N"
                )

            time.sleep(delay)

        avg_adc = sum(values) / len(values)

        if abs(self.scale) > 1e-12:
            avg_force_N = self.force_sign * ((avg_adc - self.offset) / self.scale)
        else:
            avg_force_N = 0.0

        print(f"Average estimated force: {avg_force_N:.4f} N")

        return avg_adc, avg_force_N


    def tare(
        self,
        samples: int = 200,
        interactive: bool = True,
    ):
        """
        Zero the force sensor.

        This records the no-load ADC value as offset.
        After tare, unloaded force should be close to 0 N.
        """

        print("")
        print("--------------------------------------------------")
        print("TARE")
        print("--------------------------------------------------")
        print(f"Profile: {self.profile_name}")
        print(f"Sensor model: {self.sensor_model}")
        print("Remove all force/load from the sensor.")
        print("Keep the sensor mounted exactly how it will be used.")
        print("")

        if interactive:
            input("Press ENTER when unloaded and stable...")

        # For tare, we care about raw ADC average.
        self.offset = self.collect_raw_average(samples=samples)

        self.save_calibration()

        print("")
        print("Tare complete.")
        print("Unloaded force should now be approximately 0 N.")
        print(f"New offset saved.")
        print("")


    def calibrate_with_multiple_loads(
        self,
        masses_kg=None,
        samples: int = 200,
        interactive: bool = True,
    ):
        """
        Calibrate the ADC-to-force scale using multiple known masses.

        Instead of using only one mass, this uses several loads and fits:

            ADC_delta = scale * Force_N

        where:
            ADC_delta = loaded_adc - offset
            Force_N = mass_kg * 9.80665

        The final scale is found using least-squares through the origin:

            scale = sum(Force_N * ADC_delta) / sum(Force_N^2)

        This is usually better than single-point calibration because it reduces
        the effect of noise from one measurement.
        """

        if masses_kg is None:
            masses_kg = self.get_default_calibration_masses_kg()

        print("")
        print("--------------------------------------------------")
        print("MULTI-LOAD CALIBRATION")
        print("--------------------------------------------------")
        print(f"Profile: {self.profile_name}")
        print(f"Sensor model: {self.sensor_model}")
        print("Calibration masses:")
        for mass in masses_kg:
            print(f"  {mass * 1000:.0f} g -> {mass * self.g:.4f} N")
        print("")
        print("Make sure each load direction matches real use direction.")
        print("")

        force_values = []
        adc_delta_values = []

        for mass_kg in masses_kg:
            known_force_N = mass_kg * self.g

            print("")
            print("--------------------------------------------------")
            print(f"LOAD STEP: {mass_kg * 1000:.0f} g")
            print("--------------------------------------------------")
            print(f"Expected applied force: {known_force_N:.4f} N")
            print("Place this mass/load on the sensor and keep it stable.")
            print("")

            if interactive:
                input("Press ENTER when load is applied and stable...")

            loaded_adc, old_estimated_force_N = self.collect_force_debug_average(
                samples=samples
            )

            adc_delta = loaded_adc - self.offset

            force_values.append(known_force_N)
            adc_delta_values.append(adc_delta)

            print("")
            print("Load result:")
            print(f"  Expected force: {known_force_N:.4f} N")
            print(f"  Estimated force using OLD calibration: {old_estimated_force_N:.4f} N")
            print("")

        numerator = 0.0
        denominator = 0.0

        for force_N, adc_delta in zip(force_values, adc_delta_values):
            numerator += force_N * adc_delta
            denominator += force_N * force_N

        if abs(denominator) < 1e-12:
            raise RuntimeError("Calibration failed: denominator is zero.")

        self.scale = numerator / denominator

        if abs(self.scale) < 1e-12:
            raise RuntimeError("Calibration failed: scale is too close to zero.")

        self.save_calibration()

        print("")
        print("--------------------------------------------------")
        print("MULTI-LOAD CALIBRATION COMPLETE")
        print("--------------------------------------------------")
        print(f"New scale saved.")

        for force_N, adc_delta in zip(force_values, adc_delta_values):
            predicted_force_N = self.force_sign * (adc_delta / self.scale)
            error_N = predicted_force_N - force_N

            print(
                f"Known: {force_N:.4f} N | "
                f"Measured after calibration: {predicted_force_N:.4f} N | "
                f"Error: {error_N:.4f} N"
            )

        print("")

    def full_calibration(
        self,
        known_mass_kg: float = None,
        samples: int = 200,
    ):
        """
        Run full calibration:
            1. Tare unloaded sensor
            2. Calibrate using multiple known loads
        """

        self.tare(samples=samples, interactive=True)

        self.calibrate_with_multiple_loads(
            masses_kg=None,
            samples=samples,
            interactive=True,
        )

    def live_read_raw(
        self,
        hz: float = 100.0,
    ):
        """
        Print live raw ADC data.
        """

        dt = 1.0 / hz

        print("")
        print("LIVE RAW READING")
        print("Press CTRL+C to stop.")
        print("")

        while True:
            data = self.read_raw()

            if data is not None:
                print(
                    f"Raw ADC: {data['raw_adc']:.0f} | "
                    f"Arduino Time: {data['arduino_time_ms']:.0f} ms"
                )

            time.sleep(dt)

    def live_read_force(
        self,
        hz: float = 100.0,
    ):
        """
        Print live calibrated force data.
        """

        dt = 1.0 / hz

        print("")
        print("LIVE FORCE READING")
        print("Press CTRL+C to stop.")
        print("")

        while True:
            data = self.read_force()

            if data is not None:
                print(
                    f"Raw: {data['raw_adc']:.0f} | "
                    f"Tared: {data['tared_adc']:.0f} | "
                    f"Force: {data['force_N']:.4f} N | "
                    f"Arduino Time: {data['arduino_time_ms']:.0f} ms"
                )

            time.sleep(dt)

    def get_latest_force_N(self) -> Optional[float]:
        """
        Convenience function for robot control loops.

        Returns only force in Newtons.
        """

        data = self.read_force()

        if data is None:
            return None

        return data["force_N"]

    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """
        Convenience function for logging/control loops.

        Returns full calibrated data dictionary.
        """

        return self.read_force()


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Force sensor calibration and reading tool."
    )

    parser.add_argument(
        "--profile",
        choices=list(SENSOR_PROFILES.keys()),
        required=True,
        help="Sensor profile to use.",
    )

    parser.add_argument(
        "--action",
        choices=["tare", "calibrate", "full", "read", "raw"],
        required=True,
        help="Action to perform.",
    )

    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="Serial port. Default: /dev/ttyUSB0",
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Serial baud rate. Default: 115200",
    )

    parser.add_argument(
        "--mass",
        type=float,
        default=0.2,
        help="Known calibration mass in kg. Default: 0.2 kg.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of samples for averaging.",
    )

    parser.add_argument(
        "--hz",
        type=float,
        default=100.0,
        help="Live reading frequency.",
    )

    parser.add_argument(
        "--force-sign",
        type=float,
        default=1.0,
        choices=[-1.0, 1.0],
        help="Use -1 to flip force sign, 1 to keep sign.",
    )

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    sensor = ForceSensor(
        profile_name=args.profile,
        port=args.port,
        baud=args.baud,
        force_sign=args.force_sign,
    )

    try:
        if args.action == "tare":
            sensor.tare(samples=args.samples)

        elif args.action == "calibrate":
            sensor.calibrate_with_mass(
                known_mass_kg=args.mass,
                samples=args.samples,
            )

        elif args.action == "full":
            sensor.full_calibration(
                known_mass_kg=args.mass,
                samples=args.samples,
            )

        elif args.action == "read":
            sensor.live_read_force(hz=args.hz)

        elif args.action == "raw":
            sensor.live_read_raw(hz=args.hz)

    except KeyboardInterrupt:
        print("")
        print("Stopped by user.")

    finally:
        sensor.disconnect()
        print("Force sensor disconnected.")


if __name__ == "__main__":
    main()