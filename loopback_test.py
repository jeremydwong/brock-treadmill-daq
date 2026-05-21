#!/usr/bin/env python3
"""
Loopback test for USB-1608GX-2AO.

Wiring:
  AO0 -> AI0  (speed command loopback)
  AO1 -> AI1  (incline command loopback)
  AO GND -> AI GND

Writes a series of known voltages to each AO channel and reads them
back on the corresponding AI channel. Reports error at each step.

Setup (RT Linux):
  sudo apt install libuldaq-dev
  uv init loopback_test
  cd loopback_test
  uv add uldaq
  uv run loopback_test.py
"""

import time
import sys

from uldaq import (
    get_daq_device_inventory,
    DaqDevice,
    InterfaceType,
    AiInputMode,
    AOutFlag,
    AInFlag,
    Range,
)

# --- Configuration ---
AO_CHANNELS = [0, 1]
AI_CHANNELS = [0, 1]
AI_MODE = AiInputMode.SINGLE_ENDED
AI_RANGE = Range.BIP10VOLTS
AO_RANGE = Range.BIP10VOLTS

# Test voltages to sweep through on each channel
TEST_VOLTAGES = [-5.0, -2.5, -1.0, 0.0, 1.0, 2.5, 5.0, 9.0]

# Settling time after AO write before AI read (seconds)
SETTLE_TIME = 0.01

# Maximum acceptable error (volts)
MAX_ERROR = 0.05


def main():
    # Discover devices
    devices = get_daq_device_inventory(InterfaceType.USB)
    if not devices:
        print("ERROR: No MCC DAQ devices found.")
        sys.exit(1)

    print(f"Found {len(devices)} device(s):")
    for i, d in enumerate(devices):
        print(f"  [{i}] {d.product_name}  (serial: {d.unique_id})")

    # Connect to first device
    daq = DaqDevice(devices[0])
    daq.connect()
    print(f"\nConnected to {devices[0].product_name}\n")

    ai = daq.get_ai_device()
    ao = daq.get_ao_device()

    if ai is None:
        print("ERROR: Device has no analog input subsystem.")
        daq.disconnect()
        daq.release()
        sys.exit(1)

    if ao is None:
        print("ERROR: Device has no analog output subsystem.")
        print("       Make sure you have the USB-1608GX-2AO variant.")
        daq.disconnect()
        daq.release()
        sys.exit(1)

    # Run loopback test
    total_tests = 0
    failures = 0

    for ao_ch, ai_ch in zip(AO_CHANNELS, AI_CHANNELS):
        print(f"--- Channel pair: AO{ao_ch} -> AI{ai_ch} ---")
        print(f"{'Set (V)':>10}  {'Read (V)':>10}  {'Error (V)':>10}  {'Status':>8}")
        print("-" * 48)

        for v_set in TEST_VOLTAGES:
            # Write voltage
            ao.a_out(ao_ch, AO_RANGE, AOutFlag.DEFAULT, v_set)

            # Wait for settling
            time.sleep(SETTLE_TIME)

            # Read voltage
            v_read = ai.a_in(ai_ch, AI_MODE, AI_RANGE, AInFlag.DEFAULT)

            error = abs(v_read - v_set)
            status = "PASS" if error < MAX_ERROR else "FAIL"
            total_tests += 1
            if error >= MAX_ERROR:
                failures += 1

            print(f"{v_set:10.3f}  {v_read:10.4f}  {error:10.4f}  {status:>8}")

        # Return channel to 0V
        ao.a_out(ao_ch, AO_RANGE, AOutFlag.DEFAULT, 0.0)
        print()

    # Summary
    print("=" * 48)
    if failures == 0:
        print(f"ALL {total_tests} TESTS PASSED")
    else:
        print(f"{failures} of {total_tests} TESTS FAILED")
    print("=" * 48)

    # Cleanup
    daq.disconnect()
    daq.release()


if __name__ == "__main__":
    main()
