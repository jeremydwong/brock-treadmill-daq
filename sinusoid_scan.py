#!/usr/bin/env python3
"""
Antiphase sinusoid loopback using the device's hardware pacers (pure Python).

This is the Python twin of sinusoid_loopback.cpp. The whole waveform is
handed to the DAC pacer and the ADC pacer clocks the readback, so sample
timing is exact regardless of what the host is doing. The host only has to
keep the two FIFOs fed and drained.

Wiring:
  AOUT0 (Pin 13) -> CH0 (Pin 1)
  AOUT1 (Pin 15) -> CH1 (Pin 4)
  AGND  (Pin 14) -> AGND (Pin 3)

Usage:
  uv run sinusoid-scan
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

from mcc1608g import USB1608G, AiMode, AiRange, DaqError

# --- Configuration (matches sinusoid_loopback.cpp) ---
AMPLITUDE = 5.0
OFFSET = 0.0
FREQUENCY = 10.0          # Hz
NUM_CYCLES = 80
SAMPLE_RATE = 10_000.0    # scans per second, per channel
SAMPLES_PER_CHANNEL = int(NUM_CYCLES * SAMPLE_RATE / FREQUENCY)

AI_MODE = AiMode.SINGLE_ENDED
AI_RANGE = AiRange.BIP10V
CSV_PATH = "sinusoid_scan.csv"
PNG_PATH = "sinusoid_scan.png"


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

    # --- Build the waveform: column 0 -> AOUT0, column 1 -> AOUT1 ---
    t = np.arange(SAMPLES_PER_CHANNEL) / SAMPLE_RATE
    phase = 2.0 * np.pi * FREQUENCY * t
    commanded = np.column_stack([
        OFFSET + AMPLITUDE * np.sin(phase),
        OFFSET + AMPLITUDE * np.sin(phase + np.pi),
    ])

    duration = SAMPLES_PER_CHANNEL / SAMPLE_RATE
    print(f"Waveform: {FREQUENCY:.1f} Hz, +/-{AMPLITUDE:.1f}V, {NUM_CYCLES} cycles")
    print(f"Scan:     {SAMPLE_RATE:.0f} S/s/ch, {SAMPLES_PER_CHANNEL} samples/ch, "
          f"{duration:.2f}s duration\n")

    with daq:
        # Start the ADC first so it is already listening when the DAC begins.
        ai = daq.a_in_scan(0, 1, AI_MODE, AI_RANGE, SAMPLES_PER_CHANNEL, SAMPLE_RATE)
        print(f"AI scan started (actual rate: {ai.rate:.1f} S/s)")
        ao = daq.a_out_scan(0, 1, commanded, SAMPLE_RATE)
        print(f"AO scan started (actual rate: {ao.rate:.1f} S/s)")

        measured = ai.read_all()
        print("AI scan complete.")
        ao.wait()
        print("AO scan complete.")
        daq.a_out(0, 0.0)
        daq.a_out(1, 0.0)

    # --- Error statistics ---
    err = measured - commanded
    for ch in range(2):
        e = err[:, ch]
        print(f"CH{ch} error: mean={e.mean() * 1000:.2f} mV, "
              f"max={np.abs(e).max() * 1000:.2f} mV, "
              f"rms={np.sqrt(np.mean(e ** 2)) * 1000:.2f} mV")

    # --- CSV, same columns as the C++ version ---
    t_ms = t * 1000.0
    table = np.column_stack([t_ms, commanded, measured, err * 1000.0])
    np.savetxt(CSV_PATH, table, delimiter=",", fmt="%.6g",
               header="time_ms,ao_ch0,ao_ch1,ai_ch0,ai_ch1,err_ch0_mv,err_ch1_mv", comments="")
    print(f"Data saved to {CSV_PATH}")

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f"Antiphase Sinusoid Loopback - Hardware Pacer, pure Python "
                 f"({ai.rate:.0f} S/s/ch, {FREQUENCY:.0f} Hz)")

    axes[0].plot(t_ms, commanded[:, 0], color='#3366cc', linewidth=1.2, label='AOUT0 (0 deg)')
    axes[0].plot(t_ms, commanded[:, 1], color='#cc3333', linewidth=1.2, label='AOUT1 (180 deg)')
    axes[0].set_ylabel('Commanded (V)')

    axes[1].plot(t_ms, measured[:, 0], color='#3366cc', linewidth=0.8, label='CH0 readback')
    axes[1].plot(t_ms, measured[:, 1], color='#cc3333', linewidth=0.8, label='CH1 readback')
    axes[1].set_ylabel('Measured (V)')

    axes[2].plot(t_ms, err[:, 0] * 1000, color='#3366cc', linewidth=0.6, label='CH0 error')
    axes[2].plot(t_ms, err[:, 1] * 1000, color='#cc3333', linewidth=0.6, label='CH1 error')
    axes[2].set_ylabel('Error (mV)')
    axes[2].set_xlabel('Time (ms)')

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(PNG_PATH, dpi=150)
    print(f"Plot saved to {PNG_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
