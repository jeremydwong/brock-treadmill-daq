#!/usr/bin/env python3
"""
Antiphase sinusoid loopback test for USB-1608GX-2AO.

Wiring:
  AOUT0 (Pin 13) -> CH0 (Pin 1)
  AOUT1 (Pin 15) -> CH1 (Pin 4)
  AGND  (Pin 14) -> AGND (Pin 3)

Outputs a sinusoid on AOUT0 and its antiphase (180° shifted)
on AOUT1, reads both back on CH0 and CH1, and plots the result.

Usage:
  uv run sinusoid_loopback.py
"""

import time
import math
import sys

import matplotlib.pyplot as plt
import numpy as np

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
AI_MODE = AiInputMode.SINGLE_ENDED
AI_RANGE = Range.BIP10VOLTS
AO_RANGE = Range.BIP10VOLTS

AMPLITUDE = 5.0       # Volts peak
OFFSET = 0.0          # DC offset (keep output within ±10V)
FREQUENCY = 2.0       # Hz
NUM_CYCLES = 3        # Number of full cycles to capture
SAMPLES_PER_CYCLE = 100
SETTLE_TIME = 0.001   # Seconds between write and read


def main():
    # --- Connect to device ---
    devices = get_daq_device_inventory(InterfaceType.USB)
    if not devices:
        print("ERROR: No MCC DAQ devices found.")
        sys.exit(1)

    daq = DaqDevice(devices[0])
    daq.connect()
    print(f"Connected to {devices[0].product_name}")

    ai = daq.get_ai_device()
    ao = daq.get_ao_device()

    if ai is None or ao is None:
        print("ERROR: Device missing AI or AO subsystem.")
        daq.disconnect()
        daq.release()
        sys.exit(1)

    # --- Generate and sample ---
    total_samples = NUM_CYCLES * SAMPLES_PER_CYCLE
    dt = 1.0 / (FREQUENCY * SAMPLES_PER_CYCLE)

    t_cmd = np.zeros(total_samples)
    v_set_0 = np.zeros(total_samples)
    v_set_1 = np.zeros(total_samples)
    v_read_0 = np.zeros(total_samples)
    v_read_1 = np.zeros(total_samples)
    t_read = np.zeros(total_samples)

    print(f"Outputting {FREQUENCY} Hz sinusoid, {NUM_CYCLES} cycles, "
          f"{total_samples} samples...")
    print(f"  AOUT0: {AMPLITUDE}V amplitude, 0° phase")
    print(f"  AOUT1: {AMPLITUDE}V amplitude, 180° phase")

    t_start = time.monotonic()

    for i in range(total_samples):
        t = i * dt
        phase = 2.0 * math.pi * FREQUENCY * t

        # Antiphase sinusoids
        v0 = OFFSET + AMPLITUDE * math.sin(phase)
        v1 = OFFSET + AMPLITUDE * math.sin(phase + math.pi)

        # Write both channels
        t_cmd[i] = time.monotonic() - t_start
        ao.a_out(0, AO_RANGE, AOutFlag.DEFAULT, v0)
        ao.a_out(1, AO_RANGE, AOutFlag.DEFAULT, v1)

        # Brief settle
        time.sleep(SETTLE_TIME)

        # Read both channels
        v_read_0[i] = ai.a_in(0, AI_MODE, AI_RANGE, AInFlag.DEFAULT)
        v_read_1[i] = ai.a_in(1, AI_MODE, AI_RANGE, AInFlag.DEFAULT)
        t_read[i] = time.monotonic() - t_start

        v_set_0[i] = v0
        v_set_1[i] = v1

    # Return outputs to 0V
    ao.a_out(0, AO_RANGE, AOutFlag.DEFAULT, 0.0)
    ao.a_out(1, AO_RANGE, AOutFlag.DEFAULT, 0.0)

    elapsed = time.monotonic() - t_start
    effective_rate = total_samples / elapsed
    print(f"Done. {total_samples} samples in {elapsed:.2f}s "
          f"({effective_rate:.1f} samples/s)")

    # --- Cleanup DAQ ---
    daq.disconnect()
    daq.release()

    # --- Compute error ---
    err_0 = v_read_0 - v_set_0
    err_1 = v_read_1 - v_set_1
    print(f"\nCH0 error: mean={np.mean(err_0)*1000:.2f} mV, "
          f"max={np.max(np.abs(err_0))*1000:.2f} mV")
    print(f"CH1 error: mean={np.mean(err_1)*1000:.2f} mV, "
          f"max={np.max(np.abs(err_1))*1000:.2f} mV")

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Top: commanded signals
    axes[0].plot(t_cmd, v_set_0, 'b-', linewidth=1, label='AOUT0 (0°)')
    axes[0].plot(t_cmd, v_set_1, 'r-', linewidth=1, label='AOUT1 (180°)')
    axes[0].set_ylabel('Commanded (V)')
    axes[0].set_title('Antiphase Sinusoid Loopback Test')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Middle: read-back signals
    axes[1].plot(t_read, v_read_0, 'b.-', markersize=3, linewidth=0.8,
                 label='CH0 readback')
    axes[1].plot(t_read, v_read_1, 'r.-', markersize=3, linewidth=0.8,
                 label='CH1 readback')
    axes[1].set_ylabel('Measured (V)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Bottom: error
    axes[2].plot(t_read, err_0 * 1000, 'b.-', markersize=3, linewidth=0.8,
                 label='CH0 error')
    axes[2].plot(t_read, err_1 * 1000, 'r.-', markersize=3, linewidth=0.8,
                 label='CH1 error')
    axes[2].set_ylabel('Error (mV)')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sinusoid_loopback.png', dpi=150)
    print("\nPlot saved to sinusoid_loopback.png")
    plt.show()


if __name__ == "__main__":
    main()
