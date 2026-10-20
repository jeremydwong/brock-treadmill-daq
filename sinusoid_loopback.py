#!/usr/bin/env python3
"""
Antiphase sinusoid loopback test for USB-1608GX-2AO (software-timed).

Wiring:
  AOUT0 (Pin 13) -> CH0 (Pin 1)
  AOUT1 (Pin 15) -> CH1 (Pin 4)
  AGND  (Pin 14) -> AGND (Pin 3)

Outputs a sinusoid on AOUT0 and its antiphase (180 deg shifted) on AOUT1,
reads both back on CH0 and CH1, and plots the result.

Every sample here is a separate USB round trip driven by this Python loop,
so timing is only as good as the host. Two things keep that honest:
  * the loop sleeps until each sample's scheduled time, rather than free-running
  * the commanded voltage is computed from the *actual* clock, so a late
    iteration advances the phase instead of stretching the waveform.
For sample-accurate timing use sinusoid_scan.py, which runs on the device's pacer.

Usage:
  uv run sinusoid
"""

import math
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from mcc1608g import USB1608G, AiMode, AiRange, DaqError

# --- Configuration ---
AI_MODE = AiMode.SINGLE_ENDED
AI_RANGE = AiRange.BIP10V

AMPLITUDE = 5.0       # Volts peak
OFFSET = 0.0          # DC offset (keep output within +/-10V)
FREQUENCY = 2.0       # Hz
NUM_CYCLES = 3        # Number of full cycles to capture
SAMPLES_PER_CYCLE = 100
SETTLE_TIME = 0.001   # Seconds between write and read


def wait_until(t_target: float):
    """Sleep until time.monotonic() reaches t_target, spinning for the last bit."""
    while True:
        remaining = t_target - time.monotonic()
        if remaining <= 0:
            return
        if remaining > 0.002:
            time.sleep(remaining - 0.001)


def main():
    try:
        daq = USB1608G.open()
    except DaqError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    print(f"Connected to {daq}")
    if not daq.has_ao:
        print("ERROR: Device has no analog outputs.")
        daq.close()
        sys.exit(1)

    total_samples = NUM_CYCLES * SAMPLES_PER_CYCLE
    dt = 1.0 / (FREQUENCY * SAMPLES_PER_CYCLE)

    t_cmd = np.zeros(total_samples)
    v_set_0 = np.zeros(total_samples)
    v_set_1 = np.zeros(total_samples)
    v_read_0 = np.zeros(total_samples)
    v_read_1 = np.zeros(total_samples)
    t_read = np.zeros(total_samples)

    print(f"Outputting {FREQUENCY} Hz sinusoid, {NUM_CYCLES} cycles, "
          f"{total_samples} samples at {1 / dt:.0f} S/s (software timed)...")
    print(f"  AOUT0: {AMPLITUDE}V amplitude, 0 deg phase")
    print(f"  AOUT1: {AMPLITUDE}V amplitude, 180 deg phase")

    with daq:
        t_start = time.monotonic()
        for i in range(total_samples):
            wait_until(t_start + i * dt)

            # Phase from the real clock, so jitter never distorts the waveform.
            t = time.monotonic() - t_start
            phase = 2.0 * math.pi * FREQUENCY * t
            v0 = OFFSET + AMPLITUDE * math.sin(phase)
            v1 = OFFSET + AMPLITUDE * math.sin(phase + math.pi)

            t_cmd[i] = t
            daq.a_out(0, v0)
            daq.a_out(1, v1)

            time.sleep(SETTLE_TIME)

            v_read_0[i] = daq.a_in(0, AI_MODE, AI_RANGE)
            v_read_1[i] = daq.a_in(1, AI_MODE, AI_RANGE)
            t_read[i] = time.monotonic() - t_start

            v_set_0[i] = v0
            v_set_1[i] = v1

        daq.a_out(0, 0.0)
        daq.a_out(1, 0.0)
        elapsed = time.monotonic() - t_start

    print(f"Done. {total_samples} samples in {elapsed:.2f}s "
          f"({total_samples / elapsed:.1f} samples/s, target {1 / dt:.0f})")
    gaps = np.diff(t_cmd)
    print(f"Sample spacing: mean {gaps.mean() * 1000:.2f} ms, "
          f"max {gaps.max() * 1000:.2f} ms (target {dt * 1000:.2f} ms)")

    # --- Compute error ---
    err_0 = v_read_0 - v_set_0
    err_1 = v_read_1 - v_set_1
    print(f"\nCH0 error: mean={np.mean(err_0) * 1000:.2f} mV, "
          f"max={np.max(np.abs(err_0)) * 1000:.2f} mV")
    print(f"CH1 error: mean={np.mean(err_1) * 1000:.2f} mV, "
          f"max={np.max(np.abs(err_1)) * 1000:.2f} mV")

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(t_cmd, v_set_0, 'b-', linewidth=1, label='AOUT0 (0 deg)')
    axes[0].plot(t_cmd, v_set_1, 'r-', linewidth=1, label='AOUT1 (180 deg)')
    axes[0].set_ylabel('Commanded (V)')
    axes[0].set_title('Antiphase Sinusoid Loopback Test (software timed)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_read, v_read_0, 'b.-', markersize=3, linewidth=0.8, label='CH0 readback')
    axes[1].plot(t_read, v_read_1, 'r.-', markersize=3, linewidth=0.8, label='CH1 readback')
    axes[1].set_ylabel('Measured (V)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_read, err_0 * 1000, 'b.-', markersize=3, linewidth=0.8, label='CH0 error')
    axes[2].plot(t_read, err_1 * 1000, 'r.-', markersize=3, linewidth=0.8, label='CH1 error')
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
