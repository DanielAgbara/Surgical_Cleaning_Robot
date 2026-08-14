#!/usr/bin/env python3
"""
force_sensor.py

Lite6-integrated calibration and test interface for an Arduino + HX711 force
sensor using Haplink.

What this program does
----------------------
1. Opens /dev/ttyUSB0 at 115200 baud.
2. Receives raw HX711 readings from the Arduino.
3. Runs fixed-pose scale calibration using a known mass.
4. Moves through unloaded Lite6 poses and fits a gravity model.
5. Saves scale, gravity, payload, and fit-quality calibration values.
6. In test mode, streams gravity-compensated contact force until Ctrl+C.

Calibration convention
----------------------
counts_per_newton = (loaded_adc - tare_offset) / known_force_newtons

Therefore:
force_newtons = (raw_adc - tare_offset) / counts_per_newton

Gravity is used once during calibration. It is NOT multiplied again during
normal force conversion.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from haplink import DataType, Haplink

import robot


# ---------------------------------------------------------------------------
# These IDs and types MUST exactly match the Arduino sketch.
# ---------------------------------------------------------------------------

TELEMETRY_RAW_ADC = 0
TELEMETRY_ARDUINO_MILLIS = 1
TELEMETRY_SAMPLE_COUNTER = 2

STANDARD_GRAVITY = 9.80665  # m/s^2


# ---------------------------------------------------------------------------
# Output locations
# ---------------------------------------------------------------------------
#
# The script is intended to live inside:
#
#   Surgical_Cleaning_Robot/Integration/force_sensor.py
#
# All generated files are therefore placed in:
#
#   Surgical_Cleaning_Robot/Integration/data/force_data/
#
# Using __file__ instead of the current working directory means the files go
# to the correct folder even when the script is launched from another folder.

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
FORCE_DATA_DIR = DATA_DIR / "force_data"

DEFAULT_CALIBRATION_FILE = FORCE_DATA_DIR / "force_sensor_calibration.json"
DEFAULT_HUMAN_CALIBRATION_FILE = (
    FORCE_DATA_DIR / "human_force_sensor_calibration.json"
)
HUMAN_GRAVITY_DATA_FILE = FORCE_DATA_DIR / "human_gravity_calibration.json"


@dataclass(frozen=True)
class ForceSample:
    """One fresh force-sensor sample."""

    raw_adc: float
    tared_adc: float
    force_newtons: float
    arduino_millis: int
    sample_counter: int
    host_unix_time: float
    host_monotonic_time: float


@dataclass
class Calibration:
    """Calibration values needed to convert raw counts to force."""

    counts_per_newton: float
    tare_offset: float = 0.0
    known_mass_kg: float | None = None
    saved_unix_time: float | None = None
    adc_intercept: float | None = None
    gravity_coefficients: list[float] | None = None
    estimated_payload_mass_kg: float | None = None
    gravity_fit_rmse_newtons: float | None = None
    gravity_calibration_poses: int | None = None
    orientation_coefficients: list[float] | None = None

    def validate(self) -> None:
        if (
            not np.isfinite(self.counts_per_newton)
            or abs(self.counts_per_newton) < 1e-12
        ):
            raise ValueError("counts_per_newton must be finite and nonzero")
        if self.gravity_coefficients is not None:
            coefficients = np.asarray(self.gravity_coefficients, dtype=float)
            if coefficients.shape != (3,) or not np.all(np.isfinite(coefficients)):
                raise ValueError("gravity_coefficients must contain 3 finite values")
            if self.adc_intercept is None or not np.isfinite(self.adc_intercept):
                raise ValueError("adc_intercept is required by the gravity model")
        if self.orientation_coefficients is not None:
            coefficients = np.asarray(self.orientation_coefficients, dtype=float)
            if coefficients.shape != (9,) or not np.all(np.isfinite(coefficients)):
                raise ValueError(
                    "orientation_coefficients must contain 9 finite values"
                )
            if self.adc_intercept is None or not np.isfinite(self.adc_intercept):
                raise ValueError("adc_intercept is required by the orientation model")

    @property
    def has_gravity_model(self) -> bool:
        return self.adc_intercept is not None and self.gravity_coefficients is not None

    @property
    def has_orientation_model(self) -> bool:
        return (
            self.adc_intercept is not None
            and self.orientation_coefficients is not None
        )

    def predict_unloaded_adc_from_rotation(
        self,
        rotation_camera_marker: np.ndarray,
    ) -> float:
        """Predict unloaded human-tool ADC from its ChArUco orientation."""
        self.validate()
        if not self.has_orientation_model:
            raise RuntimeError("Calibration does not contain an orientation model")
        rotation = np.asarray(rotation_camera_marker, dtype=float).reshape(3, 3)
        return float(
            self.adc_intercept
            + np.asarray(self.orientation_coefficients) @ rotation.reshape(9)
        )

    def predict_unloaded_adc(self, T_base_to_ee: np.ndarray) -> float:
        """Predict the raw ADC value caused by bias and payload gravity."""
        self.validate()
        if not self.has_gravity_model:
            raise RuntimeError("Calibration does not contain a gravity model")
        transform = np.asarray(T_base_to_ee, dtype=float).reshape(4, 4)
        gravity_base = np.array([0.0, 0.0, -STANDARD_GRAVITY])
        gravity_ee = transform[:3, :3].T @ gravity_base
        return float(
            self.adc_intercept
            + np.asarray(self.gravity_coefficients) @ gravity_ee
        )

    def contact_force_from_raw(
        self,
        raw_adc: float,
        T_base_to_ee: np.ndarray,
        runtime_adc_shift: float = 0.0,
    ) -> float:
        """Return gravity-compensated axial contact force in newtons."""
        unloaded_adc = self.predict_unloaded_adc(T_base_to_ee)
        return (
            float(raw_adc) - unloaded_adc - float(runtime_adc_shift)
        ) / self.counts_per_newton


class ForceDataLogger:
    """Write streamed force measurements to a CSV file.

    The file is created inside Integration/data/force_data by default. A header is
    written once, and every fresh force sample is saved immediately. The file
    is periodically flushed so that most data remains available even if the
    program is stopped unexpectedly.
    """

    FIELDNAMES = [
        "host_iso_time",
        "host_unix_time_s",
        "host_monotonic_time_s",
        "arduino_millis",
        "sample_counter",
        "raw_adc",
        "tared_adc",
        "force_newtons",
        "tare_offset",
        "counts_per_newton",
        "predicted_unloaded_adc",
        "calibrated_force_newtons",
    ]

    def __init__(self, path: Path, flush_every: int = 10) -> None:
        if flush_every <= 0:
            raise ValueError("flush_every must be positive")

        self.path = path
        self.flush_every = int(flush_every)
        self._rows_since_flush = 0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open(
            "w",
            newline="",
            encoding="utf-8",
        )
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self.FIELDNAMES,
        )
        self._writer.writeheader()
        self._file.flush()

    def write(
        self,
        sample: ForceSample,
        calibration: Calibration,
        predicted_unloaded_adc: float | None = None,
        calibrated_force_newtons: float | None = None,
    ) -> None:
        """Append one fresh force sample to the CSV file."""
        host_iso_time = datetime.fromtimestamp(
            sample.host_unix_time,
        ).astimezone().isoformat(timespec="milliseconds")

        self._writer.writerow(
            {
                "host_iso_time": host_iso_time,
                "host_unix_time_s": f"{sample.host_unix_time:.6f}",
                "host_monotonic_time_s": (
                    f"{sample.host_monotonic_time:.6f}"
                ),
                "arduino_millis": sample.arduino_millis,
                "sample_counter": sample.sample_counter,
                "raw_adc": f"{sample.raw_adc:.3f}",
                "tared_adc": f"{sample.tared_adc:.3f}",
                "force_newtons": f"{sample.force_newtons:.8f}",
                "tare_offset": f"{calibration.tare_offset:.6f}",
                "counts_per_newton": (
                    f"{calibration.counts_per_newton:.9f}"
                ),
                "predicted_unloaded_adc": (
                    "" if predicted_unloaded_adc is None
                    else f"{predicted_unloaded_adc:.6f}"
                ),
                "calibrated_force_newtons": (
                    "" if calibrated_force_newtons is None
                    else f"{calibrated_force_newtons:.8f}"
                ),
            }
        )

        self._rows_since_flush += 1
        if self._rows_since_flush >= self.flush_every:
            self._file.flush()
            self._rows_since_flush = 0

    def close(self) -> None:
        """Flush and close the CSV file."""
        if self._file.closed:
            return

        self._file.flush()
        self._file.close()

    def __enter__(self) -> "ForceDataLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class ForceSensor:
    """Haplink client for the Arduino/HX711 force-sensor firmware."""

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout_seconds: float = 0.001,
        counts_per_newton: float = 77500.0,
    ) -> None:
        if not port:
            raise ValueError("A serial port is required")

        self.port = port
        self.baudrate = int(baudrate)
        self.timeout_seconds = float(timeout_seconds)

        self.calibration = Calibration(
            counts_per_newton=float(counts_per_newton),
        )
        self.calibration.validate()

        self.haplink = Haplink(
            self.port,
            baudrate=self.baudrate,
            timeout=self.timeout_seconds,
        )

        self._connected = False
        self._last_sample_counter: int | None = None

    @property
    def offset(self) -> float:
        return self.calibration.tare_offset

    @property
    def counts_per_newton(self) -> float:
        return self.calibration.counts_per_newton

    def connect(self) -> None:
        """
        Open the serial device and register all expected telemetry variables.
        """
        if self._connected:
            return

        connected = self.haplink.connect()
        if not connected:
            raise RuntimeError(
                f"Haplink could not connect to {self.port} "
                f"at {self.baudrate} baud"
            )

        # Human-readable Python names can differ from the Arduino variable
        # names, but IDs and binary data types must match exactly.
        self.haplink.register_telemetry(
            TELEMETRY_RAW_ADC,
            "raw_adc",
            DataType.INT32,
        )
        self.haplink.register_telemetry(
            TELEMETRY_ARDUINO_MILLIS,
            "arduino_millis",
            DataType.INT32,
        )
        self.haplink.register_telemetry(
            TELEMETRY_SAMPLE_COUNTER,
            "sample_counter",
            DataType.INT32,
        )

        self._connected = True
        self._last_sample_counter = None

    def disconnect(self) -> None:
        """Close the serial device safely."""
        if not self._connected:
            return

        try:
            self.haplink.disconnect()
        finally:
            self._connected = False

    def read_fresh_raw(
        self,
        timeout_seconds: float = 2.0,
        debug_haplink: bool = False,
    ) -> tuple[float, int, int]:
        """
        Wait for one genuinely new HX711 sample.

        The Arduino sends:
            raw_adc -> arduino_millis -> sample_counter

        Because sample_counter is sent last, seeing a changed counter means
        that the other two cached values correspond to that completed sample.

        Returns
        -------
        (raw_adc, arduino_millis, sample_counter)
        """
        if not self._connected:
            raise RuntimeError("Force sensor is not connected")

        deadline = time.monotonic() + timeout_seconds

        while time.monotonic() < deadline:
            self.haplink.update(debug=debug_haplink)

            raw_value = self.haplink.get_telemetry("raw_adc")
            time_value = self.haplink.get_telemetry("arduino_millis")
            counter_value = self.haplink.get_telemetry("sample_counter")

            if (
                raw_value is None
                or time_value is None
                or counter_value is None
            ):
                time.sleep(0.0005)
                continue

            sample_counter = int(counter_value)

            if sample_counter == self._last_sample_counter:
                # Haplink returns the most recently cached value, so wait
                # until the Arduino transmits a new sample counter.
                time.sleep(0.0005)
                continue

            self._last_sample_counter = sample_counter

            return (
                float(raw_value),
                int(time_value),
                sample_counter,
            )

        raise TimeoutError(
            "No fresh force-sensor sample was received. Check the Arduino "
            "firmware, HX711 wiring, telemetry IDs, baud rate, and serial port."
        )

    def collect_raw_samples(
        self,
        sample_count: int,
        per_sample_timeout_seconds: float = 2.0,
    ) -> list[float]:
        """Collect a requested number of unique raw HX711 samples."""
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")

        readings: list[float] = []

        while len(readings) < sample_count:
            raw_adc, _, _ = self.read_fresh_raw(
                timeout_seconds=per_sample_timeout_seconds,
            )
            readings.append(raw_adc)

        return readings

    def average_raw(self, sample_count: int = 100) -> float:
        """Average several fresh raw samples to reduce measurement noise."""
        return statistics.fmean(
            self.collect_raw_samples(sample_count)
        )

    def tare(self, sample_count: int = 100) -> float:
        """
        Measure the unloaded zero offset.

        The sensor must have no externally applied force while this runs.
        """
        self.calibration.tare_offset = self.average_raw(sample_count)

        print(
            f"Tare complete: "
            f"{self.calibration.tare_offset:.3f} ADC counts"
        )
        return self.calibration.tare_offset

    def calibrate(
        self,
        known_mass_kg: float,
        sample_count: int = 100,
        gravity: float = STANDARD_GRAVITY,
    ) -> float:
        """
        Determine ADC counts per newton from a known calibration mass.

        The sensor must already be tared, and the known mass must remain
        stationary on the sensor during sample collection.
        """
        if known_mass_kg <= 0.0:
            raise ValueError("known_mass_kg must be positive")
        if gravity <= 0.0:
            raise ValueError("gravity must be positive")

        loaded_adc = self.average_raw(sample_count)
        tared_counts = loaded_adc - self.calibration.tare_offset
        known_force_newtons = known_mass_kg * gravity

        if abs(tared_counts) < 1.0:
            raise RuntimeError(
                "Calibration reading is too close to the tare reading. "
                "Check that the known mass is installed and loading the cell."
            )

        # The sign is intentionally preserved. If force in the desired loading
        # direction should be positive but the result is negative, reverse the
        # mechanical loading direction or swap A+ and A- on the HX711.
        self.calibration.counts_per_newton = (
            tared_counts / known_force_newtons
        )
        self.calibration.known_mass_kg = known_mass_kg
        self.calibration.saved_unix_time = time.time()
        self.calibration.validate()

        print(
            "\nCalibration complete:\n"
            f"  Tare offset:       "
            f"{self.calibration.tare_offset:.3f} counts\n"
            f"  Loaded average:    {loaded_adc:.3f} counts\n"
            f"  Tared difference:  {tared_counts:.3f} counts\n"
            f"  Known force:       {known_force_newtons:.6f} N\n"
            f"  Counts/newton:     "
            f"{self.calibration.counts_per_newton:.6f}\n"
        )

        return self.calibration.counts_per_newton

    def force_from_raw(self, raw_adc: float) -> float:
        """Convert one raw ADC reading directly to force in newtons."""
        self.calibration.validate()
        return (
            float(raw_adc) - self.calibration.tare_offset
        ) / self.calibration.counts_per_newton

    def read(self, timeout_seconds: float = 2.0) -> ForceSample:
        """Receive and convert one fresh sensor sample."""
        raw_adc, arduino_millis, sample_counter = self.read_fresh_raw(
            timeout_seconds=timeout_seconds,
        )

        tared_adc = raw_adc - self.calibration.tare_offset

        return ForceSample(
            raw_adc=raw_adc,
            tared_adc=tared_adc,
            force_newtons=self.force_from_raw(raw_adc),
            arduino_millis=arduino_millis,
            sample_counter=sample_counter,
            host_unix_time=time.time(),
            host_monotonic_time=time.monotonic(),
        )

    def save_calibration(self, path: Path) -> None:
        """Save the current scale and tare offset as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self.calibration), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Saved calibration to: {path}")

    def load_calibration(self, path: Path) -> None:
        """Load a previously saved JSON calibration."""
        data = json.loads(path.read_text(encoding="utf-8"))
        calibration = Calibration(
            counts_per_newton=float(data["counts_per_newton"]),
            tare_offset=float(data.get("tare_offset", 0.0)),
            known_mass_kg=(
                None
                if data.get("known_mass_kg") is None
                else float(data["known_mass_kg"])
            ),
            saved_unix_time=(
                None
                if data.get("saved_unix_time") is None
                else float(data["saved_unix_time"])
            ),
            adc_intercept=(
                None
                if data.get("adc_intercept") is None
                else float(data["adc_intercept"])
            ),
            gravity_coefficients=(
                None
                if data.get("gravity_coefficients") is None
                else [float(value) for value in data["gravity_coefficients"]]
            ),
            estimated_payload_mass_kg=(
                None
                if data.get("estimated_payload_mass_kg") is None
                else float(data["estimated_payload_mass_kg"])
            ),
            gravity_fit_rmse_newtons=(
                None
                if data.get("gravity_fit_rmse_newtons") is None
                else float(data["gravity_fit_rmse_newtons"])
            ),
            gravity_calibration_poses=(
                None
                if data.get("gravity_calibration_poses") is None
                else int(data["gravity_calibration_poses"])
            ),
            orientation_coefficients=(
                None
                if data.get("orientation_coefficients") is None
                else [float(value) for value in data["orientation_coefficients"]]
            ),
        )
        calibration.validate()
        self.calibration = calibration


def parse_args_legacy() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read an Arduino/HX711 force sensor through Haplink."
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="Serial device, default: /dev/ttyUSB0",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Serial baud rate, default: 115200",
    )
    parser.add_argument(
        "--calibration-file",
        type=Path,
        default=DEFAULT_CALIBRATION_FILE,
        help=(
            "Calibration JSON file. Default: "
            "Integration/data/force_data/force_sensor_calibration.json"
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help=(
            "CSV force-data file. By default, a timestamped file is created "
            "inside Integration/data/force_data."
        ),
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=10,
        help="Flush the CSV file after this many samples, default: 10",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Samples used for tare and calibration",
    )
    parser.add_argument(
        "--known-mass-kg",
        type=float,
        default=0.200,
        help="Known calibration mass in kilograms, default: 0.200",
    )
    parser.add_argument(
        "--print-rate",
        type=float,
        default=20.0,
        help="Maximum terminal print rate in Hz, default: 20",
    )
    parser.add_argument(
        "--debug-haplink",
        action="store_true",
        help="Print Haplink packet debugging while waiting for first data",
    )
    return parser.parse_args()


def ask_yes_no(prompt: str, default: bool) -> bool:
    """Read a simple yes/no terminal response."""
    suffix = " [Y/n]: " if default else " [y/N]: "

    while True:
        answer = input(prompt + suffix).strip().lower()

        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False

        print("Please enter y or n.")


def main_legacy() -> None:
    args = parse_args_legacy()

    # Keep all force-sensor calibration and test artifacts together.
    FORCE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Generate a unique session filename when the user does not provide one.
    if args.output_file is None:
        timestamp = datetime.now().astimezone().strftime(
            "%Y%m%d_%H%M%S"
        )
        args.output_file = FORCE_DATA_DIR / f"force_data_{timestamp}.csv"

    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.print_rate <= 0.0:
        raise ValueError("--print-rate must be positive")
    if args.flush_every <= 0:
        raise ValueError("--flush-every must be positive")

    sensor = ForceSensor(
        port=args.port,
        baudrate=args.baud,
    )

    data_logger: ForceDataLogger | None = None

    try:
        print(
            f"Connecting to {args.port} at {args.baud} baud..."
        )
        sensor.connect()
        print("Serial port opened and Haplink telemetry registered.")

        # Confirm that matching firmware is actually sending valid packets.
        print("Waiting for the first complete sensor sample...")
        raw_adc, arduino_ms, sample_number = sensor.read_fresh_raw(
            timeout_seconds=5.0,
            debug_haplink=args.debug_haplink,
        )
        print(
            f"First sample received: raw={raw_adc:.0f}, "
            f"arduino_ms={arduino_ms}, sample={sample_number}"
        )

        if args.calibration_file.exists():
            sensor.load_calibration(args.calibration_file)
            print(
                f"Loaded saved scale: "
                f"{sensor.counts_per_newton:.6f} counts/N"
            )

        # Tare each session because sensor zero can drift with mounting,
        # temperature, preload, and electronics.
        input(
            "\nRemove the calibration mass and all external force. "
            "Let the sensor settle, then press Enter to tare..."
        )
        sensor.tare(sample_count=args.samples)

        recalibrate_default = not args.calibration_file.exists()
        should_calibrate = ask_yes_no(
            f"Calibrate using {args.known_mass_kg * 1000:.1f} g",
            default=recalibrate_default,
        )

        if should_calibrate:
            input(
                f"Place the {args.known_mass_kg * 1000:.1f} g mass in "
                "the intended positive-force direction. Let it settle, "
                "then press Enter..."
            )

            sensor.calibrate(
                known_mass_kg=args.known_mass_kg,
                sample_count=args.samples,
            )
            sensor.save_calibration(args.calibration_file)
        elif not args.calibration_file.exists():
            print(
                "\nWarning: no saved calibration exists. The built-in "
                "placeholder scale will be used and may be inaccurate."
            )

        # Save the current session tare as well. When a previous scale was
        # loaded, this updates the calibration JSON with the new zero offset.
        sensor.save_calibration(args.calibration_file)

        data_logger = ForceDataLogger(
            path=args.output_file,
            flush_every=args.flush_every,
        )

        print(f"Saving force data to: {args.output_file}")

        print(
            "\nStreaming calibrated force. Press Ctrl+C to stop.\n"
        )

        print_period = 1.0 / args.print_rate
        next_print_time = time.monotonic()

        while True:
            sample = sensor.read(timeout_seconds=2.0)

            # Save every fresh sensor sample, not only the samples selected
            # for terminal display.
            data_logger.write(sample, sensor.calibration)

            now = time.monotonic()
            if now < next_print_time:
                continue

            next_print_time = now + print_period

            print(
                f"Raw: {sample.raw_adc:10.0f} | "
                f"Tared: {sample.tared_adc:10.0f} | "
                f"Force: {sample.force_newtons:9.4f} N | "
                f"Arduino: {sample.arduino_millis:11d} ms | "
                f"Sample: {sample.sample_counter:11d}"
            )

    except KeyboardInterrupt:
        print("\nStopping force-sensor reader.")

    finally:
        if data_logger is not None:
            data_logger.close()
            print(f"Force data saved to: {data_logger.path}")

        sensor.disconnect()


PHASE_1_JOINTS_DEG = np.array([0.0, 0.0, 90.0, 0.0, -90.0, 0.0])
GRAVITY_BASE = np.array([0.0, 0.0, -STANDARD_GRAVITY])

# Conservative defaults for gravity-data collection. They are deliberately
# narrower than the controller limits, but the operator must still ensure the
# robot workspace is clear before approving motion.
RANDOM_JOINT_RANGES_DEG = np.array(
    [
        [-50.0, 50.0],
        [-30.0, 60.0],
        [35.0, 120.0],
        [-90.0, 90.0],
        [-90.0, 90.0],
        [-120.0, 120.0],
    ]
)


def add_sensor_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--debug-haplink", action="store_true")


def add_robot_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--robot-ip",
        default=robot.Lite6.DEFAULT_IP_ADDRESS,
        help=f"Lite 6 IP address (default: {robot.Lite6.DEFAULT_IP_ADDRESS})",
    )
    parser.add_argument(
        "--robot-speed-percent",
        type=float,
        default=robot.Lite6.DEFAULT_SPEED_PERCENT,
        help="Automatic joint-motion speed percentage (default: 20)",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate or read robot- and human-mounted Arduino/HX711 force "
            "sensors."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    calibrate_parser = commands.add_parser(
        "calibrate",
        help="Calibrate a robot- or human-mounted sensor",
    )
    calibrate_targets = calibrate_parser.add_subparsers(
        dest="target", required=True
    )
    robot_calibrate = calibrate_targets.add_parser("robot")
    add_sensor_arguments(robot_calibrate)
    add_robot_arguments(robot_calibrate)
    robot_calibrate.add_argument(
        "--calibration-file", type=Path, default=DEFAULT_CALIBRATION_FILE
    )
    robot_calibrate.add_argument("--known-mass-kg", type=float, default=0.200)
    robot_calibrate.add_argument(
        "--sweep-poses",
        type=int,
        choices=(3, 4, 5),
        default=5,
        help="Number of initial J5-only poses from -90 to +90",
    )
    robot_calibrate.add_argument(
        "--random-poses",
        type=int,
        default=15,
        help="Number of bounded random six-joint poses after the J5 sweep",
    )
    robot_calibrate.add_argument("--random-seed", type=int, default=7)
    robot_calibrate.add_argument("--settle-seconds", type=float, default=2.0)
    robot_calibrate.add_argument(
        "--gravity-data-file",
        type=Path,
        default=None,
        help="Optional path for collected pose/ADC data",
    )

    human_calibrate = calibrate_targets.add_parser("human")
    add_sensor_arguments(human_calibrate)
    human_calibrate.add_argument(
        "--calibration-file", type=Path, default=DEFAULT_HUMAN_CALIBRATION_FILE
    )
    human_calibrate.add_argument("--known-mass-kg", type=float, default=0.200)
    human_calibrate.add_argument(
        "--orientations",
        type=int,
        default=15,
        help="Number of manually held unloaded orientations, default: 15",
    )
    human_calibrate.add_argument(
        "--gravity-data-file", type=Path, default=HUMAN_GRAVITY_DATA_FILE
    )

    read_parser = commands.add_parser(
        "read",
        help="Read calibrated force for a robot or human tool",
    )
    read_targets = read_parser.add_subparsers(dest="target", required=True)
    robot_read = read_targets.add_parser("robot")
    add_sensor_arguments(robot_read)
    add_robot_arguments(robot_read)
    robot_read.add_argument(
        "--calibration-file", type=Path, default=DEFAULT_CALIBRATION_FILE
    )
    robot_read.add_argument("--print-rate", type=float, default=20.0)
    robot_read.add_argument("--flush-every", type=int, default=10)
    robot_read.add_argument("--output-file", type=Path, default=None)
    robot_read.add_argument(
        "--session-zero",
        action="store_true",
        help=(
            "Optionally measure and remove current ADC drift while the sensor "
            "is unloaded; the saved calibration is used directly by default"
        ),
    )

    human_read = read_targets.add_parser("human")
    add_sensor_arguments(human_read)
    human_read.add_argument(
        "--calibration-file", type=Path, default=DEFAULT_HUMAN_CALIBRATION_FILE
    )
    human_read.add_argument("--print-rate", type=float, default=20.0)
    human_read.add_argument(
        "--session-zero",
        action="store_true",
        help="Measure an unloaded ADC correction at the current orientation",
    )
    return parser.parse_args()


def connect_sensor(args: argparse.Namespace) -> ForceSensor:
    sensor = ForceSensor(port=args.port, baudrate=args.baud)
    try:
        print(f"Connecting force sensor on {args.port} at {args.baud} baud...")
        sensor.connect()
        raw, arduino_ms, counter = sensor.read_fresh_raw(
            timeout_seconds=5.0,
            debug_haplink=args.debug_haplink,
        )
    except Exception:
        sensor.disconnect()
        raise
    print(
        f"First sample: raw={raw:.0f}, arduino_ms={arduino_ms}, "
        f"sample={counter}"
    )
    return sensor


def gravity_in_ee(T_base_to_ee: np.ndarray) -> np.ndarray:
    transform = np.asarray(T_base_to_ee, dtype=float).reshape(4, 4)
    return transform[:3, :3].T @ GRAVITY_BASE


def build_gravity_poses(
    sweep_pose_count: int,
    random_pose_count: int,
    random_seed: int,
) -> list[np.ndarray]:
    if random_pose_count < 4:
        raise ValueError("--random-poses must be at least 4 for a full-rank fit")

    poses: list[np.ndarray] = []
    for j5 in np.linspace(-90.0, 90.0, sweep_pose_count):
        pose = PHASE_1_JOINTS_DEG.copy()
        pose[4] = j5
        poses.append(pose)

    generator = np.random.default_rng(random_seed)
    lower = RANDOM_JOINT_RANGES_DEG[:, 0]
    upper = RANDOM_JOINT_RANGES_DEG[:, 1]
    for _ in range(random_pose_count):
        poses.append(generator.uniform(lower, upper))
    return poses


def fit_gravity_model(
    calibration: Calibration,
    gravity_vectors_ee: np.ndarray,
    raw_adc_values: np.ndarray,
) -> None:
    gravity_vectors = np.asarray(gravity_vectors_ee, dtype=float).reshape(-1, 3)
    readings = np.asarray(raw_adc_values, dtype=float).reshape(-1)
    design = np.column_stack([np.ones(readings.size), gravity_vectors])
    coefficients, _, rank, _ = np.linalg.lstsq(design, readings, rcond=None)
    if rank < 4:
        raise RuntimeError(
            "Gravity fit is rank deficient; collect more diverse orientations"
        )

    predicted = design @ coefficients
    residual_newtons = (readings - predicted) / calibration.counts_per_newton
    calibration.adc_intercept = float(coefficients[0])
    calibration.gravity_coefficients = coefficients[1:].astype(float).tolist()
    calibration.estimated_payload_mass_kg = float(
        np.linalg.norm(coefficients[1:]) / abs(calibration.counts_per_newton)
    )
    calibration.gravity_fit_rmse_newtons = float(
        np.sqrt(np.mean(np.square(residual_newtons)))
    )
    calibration.gravity_calibration_poses = int(readings.size)
    calibration.saved_unix_time = time.time()
    calibration.validate()


def fit_human_orientation_model(
    calibration: Calibration,
    rotations_camera_marker: np.ndarray,
    raw_adc_values: np.ndarray,
) -> None:
    """Fit unloaded ADC directly from the observed ChArUco orientation."""
    rotations = np.asarray(rotations_camera_marker, dtype=float).reshape(-1, 9)
    readings = np.asarray(raw_adc_values, dtype=float).reshape(-1)
    if rotations.shape[0] != readings.size:
        raise ValueError("Human orientation and ADC sample counts do not match")
    design = np.column_stack([np.ones(readings.size), rotations])
    coefficients, _, rank, _ = np.linalg.lstsq(design, readings, rcond=None)
    if rank < 10:
        raise RuntimeError(
            "Human gravity fit is rank deficient. Repeat calibration with "
            "more varied pitch, roll, and yaw orientations."
        )

    predicted = design @ coefficients
    residual_newtons = (readings - predicted) / calibration.counts_per_newton
    orientation_matrix = coefficients[1:].reshape(3, 3)
    calibration.adc_intercept = float(coefficients[0])
    calibration.gravity_coefficients = None
    calibration.orientation_coefficients = coefficients[1:].tolist()
    calibration.estimated_payload_mass_kg = float(
        np.linalg.svd(orientation_matrix, compute_uv=False)[0]
        / (abs(calibration.counts_per_newton) * STANDARD_GRAVITY)
    )
    calibration.gravity_fit_rmse_newtons = float(
        np.sqrt(np.mean(np.square(residual_newtons)))
    )
    calibration.gravity_calibration_poses = int(readings.size)
    calibration.saved_unix_time = time.time()
    calibration.validate()


def open_human_charuco_tracker():
    """Open the ZED and create the 100 mm, 3x3 human-tool tracker."""
    import cv2

    from calibration import CharucoBoardConfig, create_charuco_detector
    from camera import get_zed_left_intrinsics_rectified, open_zed

    config = CharucoBoardConfig(
        squares_x=3,
        squares_y=3,
        square_length_m=0.100 / 3.0,
        marker_length_m=0.027,
        dictionary_id=cv2.aruco.DICT_4X4_50,
    )
    zed, runtime_params, image_zed = open_zed()
    camera_matrix, dist_coeffs = get_zed_left_intrinsics_rectified(zed)
    board, detector = create_charuco_detector(
        config, camera_matrix, dist_coeffs
    )
    return (
        zed,
        runtime_params,
        image_zed,
        camera_matrix,
        dist_coeffs,
        board,
        detector,
        config,
    )


def detect_human_marker_pose(
    image,
    board,
    detector,
    camera_matrix,
    dist_coeffs,
):
    """Detect and draw the human-tool ChArUco board in one ZED frame."""
    import cv2

    from calibration import detect_charuco_board, estimate_charuco_pose

    detection = detect_charuco_board(image, board, detector)
    pose = estimate_charuco_pose(
        detection, board, camera_matrix, dist_coeffs
    )
    display = image.copy()
    if detection.marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(
            display, detection.marker_corners, detection.marker_ids
        )
    if detection.charuco_ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(
            display, detection.charuco_corners, detection.charuco_ids
        )
    if pose is not None:
        cv2.drawFrameAxes(
            display,
            camera_matrix,
            dist_coeffs,
            pose.rotation_vector,
            pose.translation_vector,
            0.025,
        )
    return pose, detection, display


def run_human_calibration(args: argparse.Namespace) -> None:
    """Calibrate scale and gravity response from manually held tool poses."""
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.known_mass_kg <= 0.0:
        raise ValueError("--known-mass-kg must be positive")
    if args.orientations < 10:
        raise ValueError("--orientations must be at least 10")

    import cv2

    from camera import get_image

    sensor: ForceSensor | None = None
    zed = None
    window_name = "Human force gravity calibration"
    rotations: list[np.ndarray] = []
    raw_values: list[float] = []
    records: list[dict] = []
    try:
        sensor = connect_sensor(args)
        input(
            "\nHold the tool in its normal working orientation with no "
            "external contact. Let it settle, then press Enter to tare..."
        )
        sensor.tare(args.samples)
        input(
            f"Apply the {args.known_mass_kg * 1000.0:.1f} g mass along the "
            "positive sensing axis without changing orientation. Let it "
            "settle, then press Enter..."
        )
        sensor.calibrate(args.known_mass_kg, args.samples)
        input(
            "Remove the calibration mass. The remaining orientation samples "
            "must have no contact force. Press Enter to open the camera..."
        )

        tracker = open_human_charuco_tracker()
        (
            zed,
            runtime_params,
            image_zed,
            camera_matrix,
            dist_coeffs,
            board,
            detector,
            config,
        ) = tracker
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print(
            "\nMove the unloaded tool through varied roll, pitch, and yaw. "
            "At each orientation hold it still and press C. Press Q to cancel."
        )

        while len(rotations) < args.orientations:
            image = get_image(zed, runtime_params, image_zed)
            if image is None:
                continue
            pose, detection, display = detect_human_marker_pose(
                image, board, detector, camera_matrix, dist_coeffs
            )
            status = (
                f"samples {len(rotations)}/{args.orientations} | "
                f"corners {detection.num_charuco_corners}/4 | "
                + ("C capture" if pose is not None else "pose unavailable")
            )
            cv2.putText(
                display,
                status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0) if pose is not None else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, display)
            key = cv2.waitKeyEx(1) & 0xFF
            if key in (ord("q"), 27):
                raise RuntimeError("Human force calibration cancelled")
            if key != ord("c") or pose is None:
                continue

            print(
                f"Hold still: collecting orientation "
                f"{len(rotations) + 1}/{args.orientations}..."
            )
            raw_average = sensor.average_raw(args.samples)
            rotation = pose.rotation_matrix.copy()
            rotations.append(rotation)
            raw_values.append(raw_average)
            records.append(
                {
                    "T_camera_marker": pose.T_camera_board.tolist(),
                    "raw_adc_average": raw_average,
                    "mean_reprojection_error_px": pose.mean_reprojection_error_px,
                    "charuco_corners": detection.num_charuco_corners,
                }
            )
            print(f"Accepted orientation; average raw ADC={raw_average:.1f}")

        fit_human_orientation_model(
            sensor.calibration, np.asarray(rotations), np.asarray(raw_values)
        )
        sensor.save_calibration(args.calibration_file)
        args.gravity_data_file.parent.mkdir(parents=True, exist_ok=True)
        args.gravity_data_file.write_text(
            json.dumps(
                {
                    "calibration": asdict(sensor.calibration),
                    "board": asdict(config),
                    "records": records,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        result = sensor.calibration
        print(
            "\nHuman full calibration finished:\n"
            f"  Counts/newton: {result.counts_per_newton:.6f}\n"
            f"  Estimated supported mass: "
            f"{result.estimated_payload_mass_kg * 1000.0:.1f} g\n"
            f"  Orientation-fit RMSE: {result.gravity_fit_rmse_newtons:.4f} N\n"
            f"  Calibration: {args.calibration_file}\n"
            f"  Orientation data: {args.gravity_data_file}"
        )
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        if zed is not None:
            zed.close()
        if sensor is not None:
            sensor.disconnect()


def require_run_confirmation(message: str) -> None:
    print(message)
    if input("Type RUN to continue: ").strip() != "RUN":
        raise RuntimeError("Operation cancelled by user")


def run_robot_calibration(args: argparse.Namespace) -> None:
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.known_mass_kg <= 0.0:
        raise ValueError("--known-mass-kg must be positive")
    if args.settle_seconds < 0.0:
        raise ValueError("--settle-seconds cannot be negative")

    poses = build_gravity_poses(
        args.sweep_poses,
        args.random_poses,
        args.random_seed,
    )
    speed_deg_s = robot.Lite6.speed_percent_to_deg_s(
        args.robot_speed_percent
    )
    require_run_confirmation(
        "\nCALIBRATION WILL MOVE THE ROBOT AUTOMATICALLY.\n"
        f"Robot: {args.robot_ip}\n"
        f"Speed: {args.robot_speed_percent:.1f}% ({speed_deg_s:.1f} deg/s)\n"
        f"Phase-1 target: {PHASE_1_JOINTS_DEG.tolist()} deg\n"
        f"Phase 2: {args.sweep_poses} J5 sweep poses and "
        f"{args.random_poses} bounded random poses.\n"
        "Clear the workspace, keep the tool unloaded except when prompted, "
        "and remain ready to stop the robot."
    )

    lite6 = robot.Lite6(args.robot_ip, speed_percent=args.robot_speed_percent)
    sensor: ForceSensor | None = None
    records: list[dict] = []
    try:
        lite6.connect()
        sensor = connect_sensor(args)

        print("\nPhase 1: moving to the fixed scale-calibration pose...")
        lite6.move_to_joint_angles(PHASE_1_JOINTS_DEG)
        time.sleep(args.settle_seconds)

        input(
            "Remove the known mass and all external contact. Let the tool "
            "settle, then press Enter to tare..."
        )
        sensor.tare(args.samples)
        input(
            f"Apply the {args.known_mass_kg * 1000.0:.1f} g mass along the "
            "positive sensing axis. Let it settle, then press Enter..."
        )
        sensor.calibrate(args.known_mass_kg, args.samples)
        input(
            "Remove the calibration mass. Phase 2 must be completely "
            "unloaded. Press Enter when the tool is clear..."
        )

        require_run_confirmation(
            "\nPhase 2 will now execute every gravity-calibration pose. "
            "Verify the full workspace is clear."
        )
        gravity_vectors: list[np.ndarray] = []
        raw_values: list[float] = []
        for index, target in enumerate(poses, start=1):
            print(
                f"Pose {index}/{len(poses)}: "
                + ", ".join(
                    f"J{joint + 1}={angle:.1f}"
                    for joint, angle in enumerate(target)
                )
            )
            lite6.move_to_joint_angles(target)
            time.sleep(args.settle_seconds)
            measured_joints = lite6.get_joint_angles_deg()
            transform = lite6.get_T_base_to_ee()
            raw_average = sensor.average_raw(args.samples)
            gravity_vector = gravity_in_ee(transform)
            gravity_vectors.append(gravity_vector)
            raw_values.append(raw_average)
            records.append(
                {
                    "target_joints_deg": target.tolist(),
                    "measured_joints_deg": measured_joints.tolist(),
                    "T_base_to_ee": transform.tolist(),
                    "gravity_ee_m_s2": gravity_vector.tolist(),
                    "raw_adc_average": raw_average,
                }
            )

        fit_gravity_model(
            sensor.calibration,
            np.asarray(gravity_vectors),
            np.asarray(raw_values),
        )
        sensor.save_calibration(args.calibration_file)

        data_path = args.gravity_data_file
        if data_path is None:
            timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
            data_path = (
                FORCE_DATA_DIR
                / f"force_gravity_calibration_{timestamp}.json"
            )
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(
            json.dumps(
                {
                    "calibration": asdict(sensor.calibration),
                    "robot_ip": args.robot_ip,
                    "records": records,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        result = sensor.calibration
        print(
            "\nComplete calibration finished:\n"
            f"  Counts/newton: {result.counts_per_newton:.6f}\n"
            f"  Estimated payload: "
            f"{result.estimated_payload_mass_kg * 1000.0:.1f} g\n"
            f"  Gravity-fit RMSE: {result.gravity_fit_rmse_newtons:.4f} N\n"
            f"  Calibration: {args.calibration_file}\n"
            f"  Pose data: {data_path}"
        )
    finally:
        if sensor is not None:
            sensor.disconnect()
        lite6.disconnect()


def run_robot_read(args: argparse.Namespace) -> None:
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.print_rate <= 0.0:
        raise ValueError("--print-rate must be positive")
    if args.flush_every <= 0:
        raise ValueError("--flush-every must be positive")
    if not args.calibration_file.exists():
        raise FileNotFoundError(f"Calibration not found: {args.calibration_file}")

    FORCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.output_file is None:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        args.output_file = FORCE_DATA_DIR / f"force_test_{timestamp}.csv"

    lite6 = robot.Lite6(args.robot_ip, speed_percent=args.robot_speed_percent)
    sensor: ForceSensor | None = None
    logger: ForceDataLogger | None = None
    try:
        # prepare=False makes testing read-only with respect to robot motion.
        lite6.connect(prepare=False)
        sensor = connect_sensor(args)
        sensor.load_calibration(args.calibration_file)
        if not sensor.calibration.has_gravity_model:
            raise RuntimeError(
                "Calibration file has no gravity model; run the calibrate command"
            )

        # Normally test mode uses the saved calibration exactly as it was
        # produced by the calibration command. ADC electronics can drift with
        # temperature or time, so --session-zero provides an optional, explicit
        # correction. It is not performed automatically because an accidental
        # contact during this measurement would redefine zero incorrectly.
        runtime_adc_shift = 0.0
        if args.session_zero:
            input(
                "Ensure the tool is unloaded and stationary at its current "
                "orientation, then press Enter for the session zero check..."
            )
            transform = lite6.get_T_base_to_ee()
            observed = sensor.average_raw(args.samples)
            predicted = sensor.calibration.predict_unloaded_adc(transform)
            runtime_adc_shift = observed - predicted
            print(
                f"Session zero correction: {runtime_adc_shift:.1f} ADC counts "
                f"({runtime_adc_shift / sensor.counts_per_newton:.4f} N)"
            )

        logger = ForceDataLogger(args.output_file, args.flush_every)
        print(f"Streaming calibrated force to {args.output_file}")
        print("Press Ctrl+C to stop.\n")
        print_period = 1.0 / args.print_rate
        next_print = time.monotonic()
        while True:
            # One iteration performs the complete raw-ADC-to-force conversion:
            #
            # 1. Read a fresh ADC sample from the HX711/Arduino.
            # 2. Read the Lite6 TCP orientation. Test mode only reads robot
            #    state; it never sends a motion command.
            # 3. Use the saved gravity coefficients and the TCP orientation to
            #    predict the ADC reading expected from tool weight when there
            #    is no external force.
            # 4. Subtract that predicted unloaded ADC value from the actual
            #    raw ADC value. This leaves ADC counts caused by external force.
            # 5. Divide the remaining counts by the saved counts-per-newton
            #    scale from Phase 1. The result is the calibrated force in N.
            sample = sensor.read(timeout_seconds=2.0)
            transform = lite6.get_T_base_to_ee()
            predicted_unloaded_adc = (
                sensor.calibration.predict_unloaded_adc(transform)
                + runtime_adc_shift
            )
            calibrated_force_newtons = (
                sample.raw_adc - predicted_unloaded_adc
            ) / sensor.counts_per_newton

            # The CSV retains raw data and supporting values for later
            # analysis, while the terminal intentionally shows only the two
            # values requested by the test interface: Raw ADC and Force.
            logger.write(
                sample,
                sensor.calibration,
                predicted_unloaded_adc=predicted_unloaded_adc,
                calibrated_force_newtons=calibrated_force_newtons,
            )
            now = time.monotonic()
            if now >= next_print:
                next_print = now + print_period
                print(
                    f"Raw ADC: {sample.raw_adc:10.0f} | "
                    f"Force: {calibrated_force_newtons:9.4f} N"
                )
    except KeyboardInterrupt:
        print("\nStopping force-sensor test.")
    finally:
        if logger is not None:
            logger.close()
            print(f"Force data saved to: {logger.path}")
        if sensor is not None:
            sensor.disconnect()
        lite6.disconnect()


def run_human_read(args: argparse.Namespace) -> None:
    """Use ChArUco orientation to print gravity-compensated human force."""
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.print_rate <= 0.0:
        raise ValueError("--print-rate must be positive")
    if not args.calibration_file.exists():
        raise FileNotFoundError(f"Calibration not found: {args.calibration_file}")

    import cv2

    from camera import get_image

    sensor: ForceSensor | None = None
    zed = None
    window_name = "Human force sensor tracking"
    try:
        sensor = connect_sensor(args)
        sensor.load_calibration(args.calibration_file)
        if not sensor.calibration.has_orientation_model:
            raise RuntimeError(
                "Calibration has no human orientation model; run "
                "'calibrate human' first"
            )

        tracker = open_human_charuco_tracker()
        (
            zed,
            runtime_params,
            image_zed,
            camera_matrix,
            dist_coeffs,
            board,
            detector,
            _,
        ) = tracker
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        runtime_adc_shift = 0.0
        if args.session_zero:
            input(
                "Hold the unloaded tool still with the marker visible, then "
                "press Enter for the session zero check..."
            )
            pose = None
            while pose is None:
                image = get_image(zed, runtime_params, image_zed)
                if image is None:
                    continue
                pose, _, display = detect_human_marker_pose(
                    image, board, detector, camera_matrix, dist_coeffs
                )
                cv2.imshow(window_name, display)
                cv2.waitKeyEx(1)
            observed = sensor.average_raw(args.samples)
            predicted = sensor.calibration.predict_unloaded_adc_from_rotation(
                pose.rotation_matrix
            )
            runtime_adc_shift = observed - predicted
            print(
                f"Session zero correction: {runtime_adc_shift:.1f} ADC counts "
                f"({runtime_adc_shift / sensor.counts_per_newton:.4f} N)"
            )

        print("\nRaw ADC | Force\nPress Ctrl+C to stop.\n")
        print_period = 1.0 / args.print_rate
        next_print = time.monotonic()
        while True:
            image = get_image(zed, runtime_params, image_zed)
            if image is None:
                continue
            pose, detection, display = detect_human_marker_pose(
                image, board, detector, camera_matrix, dist_coeffs
            )
            status = (
                f"corners {detection.num_charuco_corners}/4 | "
                + ("tracking" if pose is not None else "pose unavailable")
            )
            cv2.putText(
                display,
                status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0) if pose is not None else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, display)
            key = cv2.waitKeyEx(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if pose is None:
                continue

            sample = sensor.read(timeout_seconds=2.0)
            predicted_unloaded_adc = (
                sensor.calibration.predict_unloaded_adc_from_rotation(
                    pose.rotation_matrix
                )
                + runtime_adc_shift
            )
            force_newtons = (
                sample.raw_adc - predicted_unloaded_adc
            ) / sensor.counts_per_newton
            now = time.monotonic()
            if now >= next_print:
                next_print = now + print_period
                print(
                    f"{sample.raw_adc:10.0f} | {force_newtons:9.4f} N"
                )
    except KeyboardInterrupt:
        print("\nStopping human force-sensor reading.")
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        if zed is not None:
            zed.close()
        if sensor is not None:
            sensor.disconnect()


def main() -> None:
    args = parse_args()
    FORCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.command == "calibrate" and args.target == "robot":
        run_robot_calibration(args)
    elif args.command == "calibrate" and args.target == "human":
        run_human_calibration(args)
    elif args.command == "read" and args.target == "robot":
        run_robot_read(args)
    elif args.command == "read" and args.target == "human":
        run_human_read(args)


if __name__ == "__main__":
    main()
