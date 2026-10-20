#!/usr/bin/env python3
"""
Force-plate calibration: shared definitions, session files and the fit.

The treadmill deck sits on four 3-axis load cells (1-4), each giving X, Y and
Z voltages: 12 signals in all. A calibration session is a folder holding

  session.json     deck size, channel map, and one entry per capture
  trial_001.csv    5 s of all 12 channels at 1 kHz (time_s, 1X, 1Y, ... 4Z)
  trial_002.csv    ...

Captures are either a *tare* (deck unloaded) or a *load* (a known force at a
known point, in a known direction). Each load remembers the most recent tare,
and the fit uses (load mean - tare mean) so slow drift between trials drops out.

The fit finds the 6x12 matrix C that maps the 12 tared voltages to the load
on the deck:

    [Fx, Fy, Fz, Mx, My, Mz] = C @ [1X, 1Y, 1Z, 2X, ..., 4Z]

Forces are in newtons and moments in newton-metres about the deck frame origin.
The first three rows alone are the force-only calibration; the moments are
what let you compute centre of pressure.

Deck frame (seen from above, standing on the belt facing forward; right-handed):
  origin  front-left corner of the deck, on its top surface
  +x      toward the back, along the belt     (0 .. deck length)
  +y      toward the right, across the belt   (0 .. deck width)
  +z      up; points below the deck surface (e.g. pulls on the substructure)
          have negative z
A direction such as "left" is the direction the applied force pushes the deck.

Usage:
  uv run forceplate_gui.py                       # collect a session
  uv run forceplate.py calib_data/<session>      # solve for C
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- What is wired where (single-ended mode) ---
# In single-ended mode the USB-1608G reads CHnH as channel n and CHnL as
# channel n + 8, so scanning channels 0..11 returns the signals in this order.
CHANNELS = [
    # (signal, AI channel, screw terminal)
    ("1X", 0, "CH0H"),
    ("1Y", 1, "CH1H"),
    ("1Z", 2, "CH2H"),
    ("2X", 3, "CH3H"),
    ("2Y", 4, "CH4H"),
    ("2Z", 5, "CH5H"),
    ("3X", 6, "CH6H"),
    ("3Y", 7, "CH7H"),
    ("3Z", 8, "CH0L"),
    ("4X", 9, "CH1L"),
    ("4Y", 10, "CH2L"),
    ("4Z", 11, "CH3L"),
]
SIGNAL_NAMES = [name for name, _, _ in CHANNELS]
AI_LOW, AI_HIGH = 0, 11

# --- Capture settings ---
SAMPLE_RATE = 1000.0      # scans per second, all 12 channels each scan
CAPTURE_SECONDS = 5.0
GRAVITY = 9.80665         # m/s^2, to turn calibration masses into newtons

# --- Load directions: unit vector of the force applied to the deck ---
DIRECTIONS = {
    "down":    (0.0, 0.0, -1.0),
    "left":    (0.0, -1.0, 0.0),
    "right":   (0.0, 1.0, 0.0),
    "forward": (-1.0, 0.0, 0.0),
    "back":    (1.0, 0.0, 0.0),
}

LOAD_NAMES = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
LOAD_UNITS = ["N", "N", "N", "N·m", "N·m", "N·m"]


# ---------------------------------------------------------------------------
# Session files
# ---------------------------------------------------------------------------

def known_load(direction: str, force_n: float, x: float, y: float, z: float) -> np.ndarray:
    """[Fx, Fy, Fz, Mx, My, Mz] for a force of `force_n` newtons applied at (x, y, z)."""
    f = force_n * np.asarray(DIRECTIONS[direction])
    m = np.cross([x, y, z], f)
    return np.concatenate([f, m])


class Session:
    """A folder of trial CSVs plus session.json describing them."""

    def __init__(self, folder: Path, meta: dict):
        self.folder = Path(folder)
        self.meta = meta

    @classmethod
    def create(cls, folder: Path, deck_length_m: float, deck_width_m: float,
               daq_info: dict | None = None) -> "Session":
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        meta = {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "deck": {
                "length_m": deck_length_m,
                "width_m": deck_width_m,
                "frame": "origin at front-left corner on top surface; +x back, +y right, +z up",
            },
            "capture": {"sample_rate_hz": SAMPLE_RATE, "seconds": CAPTURE_SECONDS,
                        "mode": "SINGLE_ENDED", "range": "BIP10V"},
            "daq": daq_info or {},
            "channels": [{"signal": s, "ai_channel": c, "terminal": t} for s, c, t in CHANNELS],
            "directions": {k: list(v) for k, v in DIRECTIONS.items()},
            "trials": [],
        }
        session = cls(folder, meta)
        session.save()
        return session

    @classmethod
    def load(cls, folder: Path) -> "Session":
        folder = Path(folder)
        return cls(folder, json.loads((folder / "session.json").read_text()))

    @property
    def trials(self) -> list[dict]:
        return self.meta["trials"]

    def save(self):
        # Write then rename, so a crash mid-write never leaves a half-written file.
        tmp = self.folder / "session.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=2))
        tmp.replace(self.folder / "session.json")

    def last_tare_id(self) -> int | None:
        tares = [t["id"] for t in self.trials if t["kind"] == "tare"]
        return tares[-1] if tares else None

    def add_trial(self, volts: np.ndarray, rate: float, kind: str, **fields) -> dict:
        """Save one capture (samples x 12 volts) as a CSV and record it in session.json."""
        trial_id = max((t["id"] for t in self.trials), default=0) + 1
        filename = f"trial_{trial_id:03d}.csv"
        t = np.arange(len(volts)) / rate
        np.savetxt(self.folder / filename, np.column_stack([t, volts]), delimiter=",",
                   fmt="%.6f", header="time_s," + ",".join(SIGNAL_NAMES), comments="")

        entry = {
            "id": trial_id,
            "kind": kind,
            "file": filename,
            "time": dt.datetime.now().isoformat(timespec="seconds"),
            "rate_hz": rate,
            "mean_v": [round(v, 6) for v in volts.mean(axis=0)],
            "std_v": [round(v, 6) for v in volts.std(axis=0)],
        }
        if kind == "load":
            entry["tare_id"] = self.last_tare_id()
        entry.update(fields)
        self.trials.append(entry)
        self.save()
        return entry

    def remove_trial(self, trial_id: int):
        """Drop a trial from session.json; its CSV is renamed, not deleted."""
        entry = next(t for t in self.trials if t["id"] == trial_id)
        csv = self.folder / entry["file"]
        if csv.exists():
            csv.rename(csv.with_suffix(".csv.removed"))
        self.trials.remove(entry)
        self.save()

    def trial_mean(self, entry: dict) -> np.ndarray:
        """Mean of each of the 12 channels, read back from the trial's CSV."""
        data = np.loadtxt(self.folder / entry["file"], delimiter=",", skiprows=1)
        return data[:, 1:].mean(axis=0)


def load_trial_entry(direction: str, x: float, y: float, z: float, mass_kg: float) -> dict:
    """The fields GUI stores for a load capture."""
    force_n = mass_kg * GRAVITY
    return {
        "direction": direction,
        "unit_vector": list(DIRECTIONS[direction]),
        "x_m": x, "y_m": y, "z_m": z,
        "mass_kg": mass_kg,
        "force_n": force_n,
        "known_load": [round(v, 6) for v in known_load(direction, force_n, x, y, z)],
    }


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

@dataclass
class Fit:
    matrix: np.ndarray          # (6, 12): loads = matrix @ tared volts
    trial_ids: list[int]
    signals: np.ndarray         # (K, 12) tared mean volts per load trial
    known: np.ndarray           # (K, 6)
    rank: int
    singular_values: np.ndarray

    @property
    def predicted(self) -> np.ndarray:
        return self.signals @ self.matrix.T

    @property
    def rms_error(self) -> np.ndarray:
        return np.sqrt(np.mean((self.predicted - self.known) ** 2, axis=0))


def gather(session: Session) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Tared signal means and known loads for every load trial that has a tare."""
    by_id = {t["id"]: t for t in session.trials}
    tare_means: dict[int, np.ndarray] = {}
    ids, rows, known = [], [], []
    for entry in session.trials:
        if entry["kind"] != "load":
            continue
        tare_id = entry.get("tare_id")
        if tare_id is None or tare_id not in by_id:
            print(f"  skipping trial {entry['id']}: no tare recorded before it")
            continue
        if tare_id not in tare_means:
            tare_means[tare_id] = session.trial_mean(by_id[tare_id])
        ids.append(entry["id"])
        rows.append(session.trial_mean(entry) - tare_means[tare_id])
        known.append(known_load(entry["direction"], entry["force_n"],
                                entry["x_m"], entry["y_m"], entry["z_m"]))
    return ids, np.array(rows).reshape(-1, 12), np.array(known).reshape(-1, 6)


def fit_calibration(trial_ids: list[int], signals: np.ndarray, known: np.ndarray) -> Fit:
    """
    Least-squares C such that known ~= signals @ C.T.

    Each row of C is solved independently from the same K x 12 signal matrix, so
    every load component needs at least 12 trials that move the 12 signals in
    independent ways. Below that the solution is the minimum-norm one, and the
    reported rank says how far short it is.
    """
    ct, _, rank, sv = np.linalg.lstsq(signals, known, rcond=None)
    return Fit(ct.T, trial_ids, signals, known, int(rank), sv)


def save_fit(fit: Fit, session: Session) -> tuple[Path, Path]:
    json_path = session.folder / "calibration.json"
    csv_path = session.folder / "calibration_matrix.csv"
    json_path.write_text(json.dumps({
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "session": str(session.folder),
        "equation": "[Fx,Fy,Fz,Mx,My,Mz] = matrix @ (volts - tare_volts)",
        "frame": session.meta["deck"]["frame"],
        "rows": LOAD_NAMES,
        "row_units": LOAD_UNITS,
        "columns": SIGNAL_NAMES,
        "matrix": fit.matrix.tolist(),
        "trial_ids": fit.trial_ids,
        "rank": fit.rank,
        "rms_error": dict(zip(LOAD_NAMES, fit.rms_error.tolist())),
    }, indent=2))
    header = "load," + ",".join(SIGNAL_NAMES)
    lines = [header] + [f"{name}," + ",".join(f"{v:.6g}" for v in row)
                        for name, row in zip(LOAD_NAMES, fit.matrix)]
    csv_path.write_text("\n".join(lines) + "\n")
    return json_path, csv_path


def centre_of_pressure(load: np.ndarray) -> tuple[float, float]:
    """(x, y) on the deck surface where a pure force would give these moments."""
    fz, mx, my = load[2], load[3], load[4]
    return -my / fz, mx / fz


def report(fit: Fit):
    k = len(fit.trial_ids)
    print(f"\nLoad trials used: {k}   signal-matrix rank: {fit.rank} of 12")
    if k < 12 or fit.rank < 12:
        print("  WARNING: fewer than 12 independent trials; the matrix is not fully"
              " determined. Add trials at more locations and in more directions.")
    sv = fit.singular_values
    if len(sv) and sv[-1] > 0:
        print(f"  condition number: {sv[0] / sv[-1]:.3g}")

    print("\nCalibration matrix (rows: load, columns: tared volts)")
    print("       " + "".join(f"{s:>9}" for s in SIGNAL_NAMES))
    for name, row in zip(LOAD_NAMES, fit.matrix):
        print(f"  {name:4} " + "".join(f"{v:9.3g}" for v in row))

    print("\nRMS error over the trials: " + ", ".join(
        f"{n} {e:.3g} {u}" for n, e, u in zip(LOAD_NAMES, fit.rms_error, LOAD_UNITS)))

    print("\nPer trial (known -> predicted)")
    for tid, known, pred in zip(fit.trial_ids, fit.known, fit.predicted):
        parts = [f"{n} {a:7.1f}->{b:7.1f}" for n, a, b in zip(LOAD_NAMES[:3], known, pred)]
        line = f"  trial {tid:3d}: " + "  ".join(parts)
        if abs(known[2]) > 1e-9 and abs(pred[2]) > 1e-9:
            kx, ky = centre_of_pressure(known)
            px, py = centre_of_pressure(pred)
            line += f"   COP ({kx:+.3f},{ky:+.3f})->({px:+.3f},{py:+.3f}) m"
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Fit the 6x12 force-plate calibration matrix.")
    parser.add_argument("session", type=Path, help="session folder containing session.json")
    args = parser.parse_args()

    session = Session.load(args.session)
    print(f"Session {session.folder}: {len(session.trials)} captures")
    ids, signals, known = gather(session)
    if not ids:
        print("No usable load trials (each needs a tare captured before it).")
        sys.exit(1)

    fit = fit_calibration(ids, signals, known)
    report(fit)
    json_path, csv_path = save_fit(fit, session)
    print(f"\nSaved {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
