#!/usr/bin/env python3
"""
Force-plate calibration: shared definitions, session files and the fit.

The treadmill deck sits on four 3-axis load cells (1-4), each giving X, Y and
Z voltages: 12 signals in all. A calibration session is a folder holding

  session.json     deck size, channel map, and one entry per capture
  trial_001.csv    5 s of all 12 channels at 1 kHz (time_s, 1X, 1Y, ... 4Z)
  trial_002.csv    ...
  captures_log.csv one row per capture: date, time, kind, load, where it was
                   applied (as entered and in metres), deck size and notes.
                   Rewritten from session.json after every change; open it in
                   Excel, but session.json is the record the fit reads.

Captures are either a *tare* (deck unloaded) or a *load* (a known force at a
known point, in a known direction). Each load remembers the most recent tare,
and the fit uses (load mean - tare mean) so slow drift between trials drops out.

The fit finds the 6x12 matrix C that maps the 12 tared voltages to the load
on the deck:

    [Fx, Fy, Fz, Mx, My, Mz] = C @ [1X, 1Y, 1Z, 2X, ..., 4Z]

Forces are in newtons and moments in newton-metres about the deck frame origin.
The first three rows alone are the force-only calibration; the moments are
what let you compute centre of pressure.

Deck frame (seen from above, standing on the belt facing forward; right-handed,
the same convention as AMTI and Bertec force plates):
  origin  front-left corner of the deck, on its top surface
  +x      to the right, across the belt          (0 .. deck width)
  +y      toward the back, along the belt        (0 .. deck length)
  +z      down, into the deck; points below the deck surface (e.g. pulls on
          the substructure) have positive z. A load pressing down is +Fz.
So a load "1.2 m back from the front edge, 0.3 m right of the left edge,
0.1 m below the deck" sits at (x, y, z) = (0.3, 1.2, 0.1).
A direction such as "left" is the direction the applied force pushes the deck.

Usage:
  uv run forceplate_gui.py                       # collect a session
  uv run forceplate.py "<data folder>/<session>" # solve for C
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
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
    "down":    (0.0, 0.0, 1.0),
    "left":    (-1.0, 0.0, 0.0),
    "right":   (1.0, 0.0, 0.0),
    "forward": (0.0, -1.0, 0.0),
    "back":    (0.0, 1.0, 0.0),
}
FRAME = "origin at front-left corner on top surface; +x right, +y back, +z down"
FRAME_VERSION = 2         # version 1 was +x back, +y right, +z up

# --- Units people can type in; everything is stored in SI ---
LENGTH_UNITS = {"m": 1.0, "ft": 0.3048}                       # metres per unit
FORCE_UNITS = {"kg": GRAVITY, "lb": 0.45359237 * GRAVITY,     # newtons per unit
               "N": 1.0, "lbf": 4.4482216152605}
MASS_UNITS = {"kg": 1.0, "lb": 0.45359237}                    # kg per unit

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
        self.log_warning: str | None = None     # set by save() if the log was locked

    @classmethod
    def create(cls, folder: Path, deck_length_m: float, deck_width_m: float,
               daq_info: dict | None = None) -> "Session":
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        meta = {
            "created": dt.datetime.now().isoformat(timespec="seconds"),
            "frame_version": FRAME_VERSION,
            "deck": {
                "length_m": deck_length_m,      # front to back, along +y
                "width_m": deck_width_m,        # side to side, along +x
                "frame": FRAME,
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
        meta = json.loads((folder / "session.json").read_text(encoding="utf-8"))
        if meta.get("frame_version") != FRAME_VERSION:
            raise ValueError(f"{folder} uses an older deck frame ({meta['deck']['frame']}); "
                             "start a new session.")
        return cls(folder, meta)

    @property
    def trials(self) -> list[dict]:
        return self.meta["trials"]

    def save(self):
        """
        Write session.json, then the captures log. session.json must succeed or
        raise. The log may fail (on Windows, Excel locks a CSV it has open); then
        `log_warning` says so and the next save tries again.
        """
        _write_atomic(self.folder / "session.json", json.dumps(self.meta, indent=2))
        try:
            self.write_log()
            self.log_warning = None
        except OSError as e:
            self.log_warning = (f"Could not update {LOG_NAME} ({e.strerror or e}); is it open "
                                "in Excel? It is rewritten after the next capture.")

    def write_log(self):
        deck = self.meta["deck"]
        rows = []
        for t in self.trials:
            when = dt.datetime.fromisoformat(t["time"])
            entered = t.get("entered", {})
            rows.append({
                "id": t["id"], "date": when.date().isoformat(), "time": when.time().isoformat(),
                "kind": t["kind"], "file": t["file"],
                "direction": t.get("direction", ""),
                "load": entered.get("load", ""), "load_unit": entered.get("load_unit", ""),
                "force_N": _fmt(t.get("force_n")),
                "right_of_left_edge": entered.get("right", ""),
                "back_from_front_edge": entered.get("back", ""),
                "below_deck": entered.get("below", ""),
                "length_unit": entered.get("length_unit", ""),
                "x_m": _fmt(t.get("x_m")), "y_m": _fmt(t.get("y_m")), "z_m": _fmt(t.get("z_m")),
                "deck_side_to_side_m": deck["width_m"], "deck_front_to_back_m": deck["length_m"],
                "tare_id": t.get("tare_id") or "",
                "noisiest_std_mV": _fmt(1000 * max(t["std_v"])),
                "notes": t.get("notes", ""),
            })
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=LOG_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        # utf-8-sig: the byte-order mark makes Excel read accents in notes correctly.
        _write_atomic(self.folder / LOG_NAME, out.getvalue(), encoding="utf-8-sig")

    def last_tare_id(self) -> int | None:
        tares = [t["id"] for t in self.trials if t["kind"] == "tare"]
        return tares[-1] if tares else None

    def add_trial(self, volts: np.ndarray, rate: float, kind: str, **fields) -> dict:
        """Save one capture (samples x 12 volts) as a CSV and record it in session.json."""
        trial_id = max((t["id"] for t in self.trials), default=0) + 1
        filename = f"trial_{trial_id:03d}.csv"
        save_capture_csv(self.folder / filename, volts, rate)

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
        try:
            self.save()
        except Exception:
            self.trials.remove(entry)     # keep memory in step with session.json on disk
            raise
        return entry

    def remove_trial(self, trial_id: int):
        """Drop a trial from session.json; its CSV is renamed, not deleted."""
        entry = next(t for t in self.trials if t["id"] == trial_id)
        path = self.folder / entry["file"]
        try:
            path.replace(path.with_suffix(".csv.removed"))
        except OSError:
            pass    # missing, or open in another program: it just stays as it is
        self.trials.remove(entry)
        self.save()

    def trial_mean(self, entry: dict) -> np.ndarray:
        """Mean of each of the 12 channels, read back from the trial's CSV."""
        data = np.loadtxt(self.folder / entry["file"], delimiter=",", skiprows=1)
        return data[:, 1:].mean(axis=0)


LOG_NAME = "captures_log.csv"
LOG_COLUMNS = ["id", "date", "time", "kind", "file", "direction", "load", "load_unit",
               "force_N", "right_of_left_edge", "back_from_front_edge", "below_deck",
               "length_unit", "x_m", "y_m", "z_m", "deck_side_to_side_m",
               "deck_front_to_back_m", "tare_id", "noisiest_std_mV", "notes"]


def _fmt(v) -> str:
    return "" if v is None else f"{v:.6g}"


def _write_atomic(path: Path, text: str, encoding: str = "utf-8"):
    """Write then rename, so a crash mid-write never leaves a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


def save_capture_csv(path: Path, volts: np.ndarray, rate: float):
    """
    Write one capture (samples x 12 volts) and read it back to check every row
    landed. Written to a .tmp file and renamed, so a full disk or a crash never
    leaves a short CSV under the real name.
    """
    t = np.arange(len(volts)) / rate
    tmp = path.with_name(path.name + ".tmp")
    np.savetxt(tmp, np.column_stack([t, volts]), delimiter=",", fmt="%.6f",
               header="time_s," + ",".join(SIGNAL_NAMES), comments="")
    back = np.loadtxt(tmp, delimiter=",", skiprows=1, ndmin=2)
    if back.shape != (len(volts), 13) or not np.allclose(back[:, 1:], volts, atol=1e-6):
        tmp.unlink(missing_ok=True)
        raise OSError(f"{path.name} did not read back correctly; nothing was saved.")
    tmp.replace(path)


def load_trial_entry(direction: str, x: float, y: float, z: float, force_n: float,
                     entered: dict | None = None) -> dict:
    """
    The fields the GUI stores for a load capture. (x, y, z) in metres in the deck
    frame; `entered` keeps what the person typed (values and units) for the log.
    """
    entry = {
        "direction": direction,
        "unit_vector": list(DIRECTIONS[direction]),
        "x_m": x, "y_m": y, "z_m": z,
        "force_n": force_n,
        "known_load": [round(v, 6) for v in known_load(direction, force_n, x, y, z)],
    }
    if entered:
        entry["entered"] = entered
        if entered.get("load_unit") in MASS_UNITS:
            entry["mass_kg"] = float(entered["load"]) * MASS_UNITS[entered["load_unit"]]
    return entry


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
    _write_atomic(json_path, json.dumps({
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
    _write_atomic(csv_path, "\n".join(lines) + "\n")
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
