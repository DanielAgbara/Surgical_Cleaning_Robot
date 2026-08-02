#!/usr/bin/env python3
"""
force_sensor.py

Standalone Python interface for an Arduino + HX711 force sensor using Haplink.

What this program does
----------------------
1. Opens /dev/ttyUSB0 at 115200 baud.
2. Receives raw HX711 readings from the Arduino.
3. Tares the sensor by averaging unloaded samples.
4. Optionally calibrates using a known mass.
5. Converts ADC counts directly to force in newtons.
6. Streams calibrated force readings until Ctrl+C.

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

from haplink import DataType, Haplink


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
#   Surgical_Cleaning_Robot/Integration/force_sensing.py
#
# All generated files are therefore placed in:
#
#   Surgical_Cleaning_Robot/Integration/data/
#
# Using __file__ instead of the current working directory means the files go
# to the correct folder even when the script is launched from another folder.

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

DEFAULT_CALIBRATION_FILE = DATA_DIR / "force_sensor_calibration.json"


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

    def validate(self) -> None:
        if abs(self.counts_per_newton) < 1e-12:
            raise ValueError("counts_per_newton cannot be zero")


class ForceDataLogger:
    """Write streamed force measurements to a CSV file.

    The file is created inside Integration/data by default. A header is
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

    def write(self, sample: ForceSample, calibration: Calibration) -> None:
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
        )
        calibration.validate()
        self.calibration = calibration


def parse_args() -> argparse.Namespace:
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
            "Integration/data/force_sensor_calibration.json"
        ),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help=(
            "CSV force-data file. By default, a timestamped file is created "
            "inside Integration/data."
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


def main() -> None:
    args = parse_args()

    # Create Integration/data before attempting to load or save anything.
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Generate a unique session filename when the user does not provide one.
    if args.output_file is None:
        timestamp = datetime.now().astimezone().strftime(
            "%Y%m%d_%H%M%S"
        )
        args.output_file = DATA_DIR / f"force_data_{timestamp}.csv"

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


if __name__ == "__main__":
    main()
