#!/usr/bin/env python3
"""
Check that full force-plate captures (12 channels, 5 s at 1 kHz) come off the
DAQ and onto disk intact on this computer.

Runs several captures back to back while another thread keeps Python busy,
the worst case for the DAQ's 0.68 s input buffer (8192 samples / 12 channels),
and saves each one the way the GUI does, reading it back to compare.

Usage:
  uv run capture_check.py              # 5 captures
  uv run capture_check.py --runs 20
"""

import argparse
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

import forceplate as fp
from mcc1608g import USB1608G, AiMode, AiRange, DaqError


def keep_python_busy(stop: threading.Event):
    """Pure-Python work holds the GIL, like a slow redraw on the GUI thread."""
    while not stop.is_set():
        sum(i * i for i in range(200_000))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    samples = int(fp.CAPTURE_SECONDS * fp.SAMPLE_RATE)
    daq = USB1608G.open()
    print(f"{daq}: {args.runs} captures of {samples} scans x 12 channels")
    stop = threading.Event()
    threading.Thread(target=keep_python_busy, args=(stop,), daemon=True).start()
    failures = 0
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for run in range(1, args.runs + 1):
                t0 = time.perf_counter()
                try:
                    ai = daq.a_in_scan(fp.AI_LOW, fp.AI_HIGH, AiMode.SINGLE_ENDED,
                                       AiRange.BIP10V, samples, fp.SAMPLE_RATE)
                    volts = ai.read_all()
                except DaqError as e:
                    failures += 1
                    print(f"  run {run}: FAILED  {e}")
                    continue
                t1 = time.perf_counter()
                path = Path(tmp) / f"trial_{run:03d}.csv"
                fp.save_capture_csv(path, volts, ai.rate)    # raises if it reads back wrong
                t2 = time.perf_counter()
                ok = volts.shape == (samples, 12) and np.isfinite(volts).all()
                failures += not ok
                print(f"  run {run}: {'ok' if ok else 'BAD SHAPE'}  {volts.shape}  "
                      f"capture {t1 - t0:.2f} s  save+verify {t2 - t1:.2f} s  "
                      f"{path.stat().st_size / 1e6:.2f} MB")
    finally:
        stop.set()
        daq.close()
    print("PASS: no overruns, every file read back exactly." if not failures
          else f"FAIL: {failures} of {args.runs} captures had problems.")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
