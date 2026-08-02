/*
  force_sensor_haplink.ino

  Purpose
  -------
  Read raw measurements from an HX711 load-cell amplifier and transmit them
  to a Linux/Python computer using the Haplink binary serial protocol.

  Hardware assumed
  ----------------
  Arduino Uno-compatible board
  HX711 module
  Load cell

  HX711 -> Arduino wiring used by this sketch
  --------------------------------------------
  VCC  -> 5V
  GND  -> GND
  DOUT -> Digital pin 2
  SCK  -> Digital pin 3

  Important
  ---------
  1. Do NOT use Serial.print()/Serial.println() on Serial in this sketch.
     Haplink uses Serial for binary packets; ordinary text would corrupt the
     packet stream.

  2. The telemetry IDs and data types below MUST exactly match Python:
       ID 0: raw_adc         HL_INT32
       ID 1: arduino_millis  HL_INT32
       ID 2: sample_counter  HL_INT32

  3. The sample counter is deliberately transmitted LAST. Python uses it as a
     commit marker: when the counter changes, raw_adc and arduino_millis for
     that sample have already been sent.
*/

#include <Arduino.h>
#include <HX711.h>
#include <haplink.h>

// ---------------------------------------------------------------------------
// Hardware configuration
// ---------------------------------------------------------------------------

constexpr uint8_t HX711_DOUT_PIN = 2;
constexpr uint8_t HX711_SCK_PIN  = 3;
constexpr uint32_t SERIAL_BAUD   = 115200;

// ---------------------------------------------------------------------------
// Haplink telemetry IDs
// These numbers must match the Python program exactly.
// ---------------------------------------------------------------------------

constexpr uint8_t TELEMETRY_RAW_ADC        = 0;
constexpr uint8_t TELEMETRY_ARDUINO_MILLIS = 1;
constexpr uint8_t TELEMETRY_SAMPLE_COUNTER = 2;

// ---------------------------------------------------------------------------
// Global objects
// Haplink stores pointers to registered variables, so the telemetry variables
// must remain alive for the entire program. Globals are appropriate here.
// ---------------------------------------------------------------------------

HX711 load_cell;
Haplink haplink;

// HX711 produces a signed 24-bit reading, which fits safely inside int32_t.
int32_t raw_adc = 0;

// millis() is unsigned long on an Uno. It is cast to int32_t because Haplink
// currently provides a signed 32-bit telemetry type. It will wrap eventually,
// but Python relies on sample_counter—not time—for freshness.
int32_t arduino_millis = 0;

// Incremented once for every genuinely new HX711 conversion.
int32_t sample_counter = 0;


// Turn on the built-in LED permanently if setup registration fails.
void indicate_setup_error()
{
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);

  // Stop here. We cannot safely print an error because Serial carries Haplink.
  while (true)
  {
    delay(1000);
  }
}


void setup()
{
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  // Must match the baud rate used by the Python program.
  Serial.begin(SERIAL_BAUD);

  // Initialize the HX711. The default gain is 128 on channel A.
  load_cell.begin(HX711_DOUT_PIN, HX711_SCK_PIN);

  // Bind Haplink to the same USB serial connection.
  haplink.begin(Serial);

  // Register each variable with its ID and exact binary type.
  const bool raw_ok = haplink.registerTelemetry(
    TELEMETRY_RAW_ADC,
    &raw_adc,
    HL_INT32
  );

  const bool time_ok = haplink.registerTelemetry(
    TELEMETRY_ARDUINO_MILLIS,
    &arduino_millis,
    HL_INT32
  );

  const bool counter_ok = haplink.registerTelemetry(
    TELEMETRY_SAMPLE_COUNTER,
    &sample_counter,
    HL_INT32
  );

  if (!raw_ok || !time_ok || !counter_ok)
  {
    indicate_setup_error();
  }
}


void loop()
{
  // Process any valid host-to-device Haplink packets.
  // This application currently receives no parameters, but calling update()
  // keeps the communication layer operating normally.
  haplink.update();

  // is_ready() is non-blocking. A reading is taken only when the HX711 has
  // completed a new ADC conversion.
  if (!load_cell.is_ready())
  {
    return;
  }

  // Read the new raw ADC conversion. Calibration is intentionally performed
  // in Python, so the Arduino sends unscaled sensor counts.
  raw_adc = static_cast<int32_t>(load_cell.read());
  arduino_millis = static_cast<int32_t>(millis());
  sample_counter++;

  // Send data fields first...
  haplink.sendTelemetry(TELEMETRY_RAW_ADC);
  haplink.sendTelemetry(TELEMETRY_ARDUINO_MILLIS);

  // ...and send the counter last as a "sample complete" marker.
  haplink.sendTelemetry(TELEMETRY_SAMPLE_COUNTER);
}
