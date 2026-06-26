import serial
import time
import threading
import sys

ARDUINO_PORT = 'COM5'
BAUD_RATE = 9600

try:
    # A tiny timeout ensures Python never gets stuck waiting on a character
    arduino = serial.Serial(port=ARDUINO_PORT, baudrate=BAUD_RATE, timeout=0.1)
    print(f"Opening {ARDUINO_PORT}...")
    time.sleep(2)
    arduino.reset_input_buffer()
except Exception as e:
    print(f"Error opening port: {e}")
    sys.exit()


def keyboard_input():
    while True:
        user_input = input().strip().lower()
        if user_input == 't':
            try:
                arduino.write(b't')
                arduino.flush()  # Force the byte down the wire instantly
                print(" -> [TARE COMMAND SENT]")
            except Exception as e:
                print(f"\nWrite error: {e}")


# Run terminal typing tasks on a detached thread
threading.Thread(target=keyboard_input, daemon=True).start()

print("---------------------------------------------------------")
print("Scale Active. Type 't' + Enter to tare.")
print("---------------------------------------------------------")

while True:
    try:
        if arduino.in_waiting > 0:
            raw_line = arduino.readline()
            if raw_line:
                weight_string = raw_line.decode(
                    'utf-8', errors='ignore').strip()
                # Ignore empty drops or corrupt data packets
                if weight_string and not weight_string.isalpha():
                    print(f"Weight: {weight_string} g")
        time.sleep(0.01)
    except KeyboardInterrupt:
        arduino.close()
        sys.exit()
