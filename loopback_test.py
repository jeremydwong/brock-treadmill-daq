#!/usr/bin/env python3
"""
Step loopback test for USB-1608GX-2AO.

Wiring:
  AOUT0 (Pin 13) -> CH0 (Pin 1)   speed command loopback
  AOUT1 (Pin 15) -> CH1 (Pin 4)   incline command loopback
  AGND  (Pin 14) -> AGND (Pin 3)

Writes a series of known voltages to each AO channel and reads them
back on the corresponding AI channel. Reports error at each step.

Usage:
  uv run loopback
"""

import sys
import time

from mcc1608g import USB1608G, AiMode, AiRange, DaqError, find_devices

# --- Configuration ---
CHANNEL_PAIRS = [(0, 0), (1, 1)]     # (AO channel, AI channel)
AI_MODE = AiMode.SINGLE_ENDED
AI_RANGE = AiRange.BIP10V

TEST_VOLTAGES = [-5.0, -2.5, -1.0, 0.0, 1.0, 2.5, 5.0, 9.0]
SETTLE_TIME = 0.01     # seconds between AO write and AI read
MAX_ERROR = 0.05       # volts


def main():
    devices = find_devices()
    if not devices:
        print("ERROR: No MCC DAQ devices found.")
        sys.exit(1)
    print(f"Found {len(devices)} device(s):")
    for i, d in enumerate(devices):
        print(f"  [{i}] product 0x{d.idProduct:04x}  (serial: {d.serial_number})")

    try:
        daq = USB1608G(devices[0])
    except DaqError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    print(f"\nConnected to {daq}\n")

    if not daq.has_ao:
        print("ERROR: Device has no analog outputs. Need the USB-1608GX-2AO variant.")
        daq.close()
        sys.exit(1)

    total_tests = 0
    failures = 0
    with daq:
        for ao_ch, ai_ch in CHANNEL_PAIRS:
            print(f"--- Channel pair: AO{ao_ch} -> AI{ai_ch} ---")
            print(f"{'Set (V)':>10}  {'Read (V)':>10}  {'Error (V)':>10}  {'Status':>8}")
            print("-" * 48)

            for v_set in TEST_VOLTAGES:
                daq.a_out(ao_ch, v_set)
                time.sleep(SETTLE_TIME)
                v_read = daq.a_in(ai_ch, AI_MODE, AI_RANGE)

                error = abs(v_read - v_set)
                status = "PASS" if error < MAX_ERROR else "FAIL"
                total_tests += 1
                failures += error >= MAX_ERROR
                print(f"{v_set:10.3f}  {v_read:10.4f}  {error:10.4f}  {status:>8}")

            daq.a_out(ao_ch, 0.0)
            print()

    print("=" * 48)
    if failures == 0:
        print(f"ALL {total_tests} TESTS PASSED")
    else:
        print(f"{failures} of {total_tests} TESTS FAILED")
    print("=" * 48)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
