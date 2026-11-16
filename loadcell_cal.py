#!/usr/bin/env python3
"""
Calibration of the four load cells, in one place: from volts to newtons.

There are two levels.

1. Per load cell, from the Michigan Scientific calibration sheets (SHEETS).
   Each TR3D-A-3K cell gives X, Y and Z bridge outputs. The SGA12 amplifier
   applies EXCITATION_V of excitation and a gain of AMP_GAIN, so a sheet line
   "1 mV/V = L lbf" means

       newtons_per_volt = L * LBF_N / (EXCITATION_V * AMP_GAIN / 1000)

   Each axis also leaks a little into the other two. The sheet's crosstalk
   matrix K (rows: what the X, Y, Z channels read; columns: load applied on
   X, Y, Z; as a fraction) gives reading = K @ true, so

       true = inv(K) @ (tared volts * newtons_per_volt)        per cell

2. For the whole deck, a 6x12 matrix C fitted to known loads at known points
   (fit_calibration):

       [Fx, Fy, Fz, Mx, My, Mz] = C @ [1X, 1Y, 1Z, 2X, ..., 4Z]   (tared volts)

Which serial is on which amplifier input. forceplate_layout.json says what we
expect: input N carries the cell at a given corner, with a given serial. A
cable swap breaks that silently, so each cell's own zero offset, also on its
sheet ("unbalance"), works as a fingerprint: unloaded, the X and Y outputs sit
near that offset (Z doesn't: it carries the deck's weight). identify_serials
matches the four inputs' resting X/Y volts to the four sheets, trying all 24
assignments. A position belongs to the serial (405 is rear right whatever
input it is plugged into), so a session can be re-labelled after the fact.

Every capture CSV holds raw volts, so any of this can be redone later.

Usage:
  uv run loadcell_cal.py "<data folder>/<session>"          # wiring, scales, known-load check
  uv run loadcell_cal.py "<data folder>/<session>" --fit    # also fit the 6x12 deck matrix
  uv run loadcell_cal.py --check-sheets                     # SHEETS against the PDFs
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import itertools
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

import forceplate as fp

# ---------------------------------------------------------------------------
# The calibration sheets
# ---------------------------------------------------------------------------

LBF_N = 4.4482216152605            # newtons per pound-force
AMP_GAIN = 201.0                   # SGA12, V/V (manual: +/-0.1%)
EXCITATION_V = 10.0                # SGA12, +/-5 V bipolar (manual: +/-0.1%)
CAL_GAIN_EXCITATION = 399.3 * 5.0  # Michigan Scientific's gain x excitation for the sheets
SHEET_DIR = Path.home() / "Dropbox/Treadmill Brock 2025/Instrumentation design/michsci calibration"


class Sheet(NamedTuple):
    """One TR3D-A-3K calibration sheet. Tuples are (X, Y, Z)."""
    date: str
    lbf_per_mvv: tuple[float, float, float]     # "Sensitivity: 1 mV/V = ... lbs"
    shunt_lbf: tuple[float, float, float]       # load equivalent to the 100 kOhm shunt
    unbalance_v: tuple[float, float, float]     # zero offset, volts at CAL_GAIN_EXCITATION
    crosstalk: tuple[tuple[float, ...], ...]    # rows read X, Y, Z; columns applied X, Y, Z


SHEETS = {
    "401": Sheet("2026-03-20", lbf_per_mvv=(830.0, 784.0, 787.1), shunt_lbf=(728.3, 688.5, 691.4),
                 unbalance_v=(0.031, 0.255, -0.002),
                 crosstalk=((1.0, -0.00584, 0.00745),
                            (0.00187, 1.0, -0.00723),
                            (-0.00507, -0.00116, 1.0))),
    "402": Sheet("2026-05-20", lbf_per_mvv=(832.1, 787.2, 779.5), shunt_lbf=(730.6, 691.4, 684.2),
                 unbalance_v=(-0.144, 0.162, 0.528),
                 crosstalk=((1.0, -0.00013, 0.00286),
                            (0.00177, 1.0, 0.00154),
                            (-0.00051, 0.00132, 1.0))),
    "405": Sheet("2026-02-19", lbf_per_mvv=(828.0, 779.6, 781.2), shunt_lbf=(728.3, 684.9, 686.4),
                 unbalance_v=(0.365, -0.328, 0.175),
                 crosstalk=((1.0, 0.00207, 0.00308),
                            (0.00079, 1.0, 0.00293),
                            (-0.00124, -0.00061, 1.0))),
    "406": Sheet("2026-06-13", lbf_per_mvv=(837.2, 790.2, 785.9), shunt_lbf=(735.2, 693.7, 689.8),
                 unbalance_v=(-0.158, 0.232, 0.187),
                 crosstalk=((1.0, -0.00522, 0.00265),
                            (0.00774, 1.0, -0.00249),
                            (-0.00517, 0.00057, 1.0))),
}


def serial_key(serial: str) -> str:
    """'TR3D-A-3K 401' -> '401' (the SHEETS key)."""
    return serial.split()[-1] if serial else ""


def volts_per_mvv() -> float:
    """Volts out of our amplifier for 1 mV/V of bridge output."""
    return EXCITATION_V * AMP_GAIN / 1000.0


def newtons_per_volt(serial: str) -> np.ndarray:
    """(3,) X, Y, Z newtons per volt for this serial on our amplifier."""
    return np.array(SHEETS[serial_key(serial)].lbf_per_mvv) * LBF_N / volts_per_mvv()


def expected_shunt_step_v(serial: str) -> np.ndarray:
    """(3,) volts each channel should step by with the SGA12's SHUNT switch at POS."""
    s = SHEETS[serial_key(serial)]
    return np.array(s.shunt_lbf) / np.array(s.lbf_per_mvv) * volts_per_mvv()


def expected_unbalance_v(serial: str) -> np.ndarray:
    """(3,) unloaded output on our amplifier, scaled from the sheet's gain x excitation."""
    return np.array(SHEETS[serial_key(serial)].unbalance_v) * EXCITATION_V * AMP_GAIN / CAL_GAIN_EXCITATION


def read_sheet(pdf: Path) -> Sheet:
    """
    Read a Michigan Scientific calibration PDF (needs pypdf: `uv sync --extra sheets`).
    Crosstalk is the slope, through zero, of each off-axis 'Output (lb)' against
    the applied load, over all 21 points of that axis's table.
    """
    import pypdf
    pages = [p.extract_text() for p in pypdf.PdfReader(pdf).pages]

    def per_axis(pattern: str, text: str) -> tuple[float, float, float]:
        found = {k: float(v.replace(",", "")) for k, v in re.findall(pattern, text)}
        return tuple(found[k] for k in ("Fx", "Fy", "Fz"))

    m, d, y = re.search(r"Current (\d\d)/(\d\d)/(\d{4})", pages[2]).groups()
    k = np.eye(3)
    for j, page in enumerate((4, 6, 8)):                       # Fx, Fy, Fz crosstalk tables
        rows = re.findall(r"^([-\d,.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+)% ([-\d.]+) ([-\d.]+) ([-\d.]+)%$",
                          pages[page], re.M)
        a = np.array([[float(x.replace(",", "")) for x in r] for r in rows])
        applied = a[:, 0]
        for i, out in zip([i for i in range(3) if i != j], (a[:, 2], a[:, 5])):
            k[i, j] = round(float(applied @ out / (applied @ applied)), 5)
    return Sheet(f"{y}-{m}-{d}",
                 lbf_per_mvv=per_axis(r"(F[xyz]) 1 mV/V = ([\d.,]+)\s+lbs", pages[0]),
                 shunt_lbf=per_axis(r"(F[xyz]) 100K Shunt = ([\d.,]+)\s+lbs", pages[0]),
                 unbalance_v=per_axis(r"(F[xyz]) 100K [\d.]+ [\d,.]+ ([-\d.]+) 350", pages[-1]),
                 crosstalk=tuple(tuple(r) for r in k.tolist()))


def check_sheets(folder: Path = SHEET_DIR) -> bool:
    """Re-read every PDF and compare with SHEETS. True when they all match."""
    ok = True
    for key, sheet in SHEETS.items():
        pdf = folder / f"TR3D-A-3K-{key}.pdf"
        got = read_sheet(pdf)
        same = (got.date == sheet.date
                and all(np.allclose(getattr(got, f), getattr(sheet, f), atol=1e-6)
                        for f in ("lbf_per_mvv", "shunt_lbf", "unbalance_v"))
                and np.allclose(got.crosstalk, sheet.crosstalk, atol=1e-5))
        ok &= same
        print(f"  {key}: {'matches' if same else 'DIFFERS: ' + repr(got)}  ({pdf.name})")
    return ok


# ---------------------------------------------------------------------------
# Layout: corners, expected wiring, and the scales that follow from the serials
# ---------------------------------------------------------------------------

LAYOUT_FILE = Path(__file__).with_name("forceplate_layout.json")
CELLS = ["1", "2", "3", "4"]


@dataclass
class Layout:
    deck_length_m: float
    deck_width_m: float
    where: dict[str, str]           # input "1".."4" -> "front left", ...
    positions: np.ndarray           # (4, 2) x, y in metres of the cell on inputs 1..4
    newtons_per_volt: np.ndarray    # (12,) in SIGNAL_NAMES order; negative flips that axis
    placeholder: bool               # True while the scales are not the real ones
    path: Path
    serials: dict[str, str] = field(default_factory=dict)    # input -> "TR3D-A-3K 401"
    # (4, 3, 3): per input, the inverse of its cell's crosstalk matrix (identity if none)
    uncross: np.ndarray = field(default_factory=lambda: np.tile(np.eye(3), (4, 1, 1)))
    # The layout with every number filled in: what a session stores, so it
    # still opens the same way after the sheets or the wiring change.
    raw: dict = field(default_factory=dict)

    @property
    def wiring(self) -> dict[str, str]:
        """Input "1".."4" -> serial key ("401", ...)."""
        return {c: serial_key(s) for c, s in self.serials.items()}


def resolve_cell(cell: dict, name: str) -> dict:
    """
    One load_cells entry with its scale and crosstalk filled in: from the
    entry's own numbers if it has them (an override, or a stored copy), else
    from its serial's sheet. "flip": ["X"] makes that axis negative.
    """
    cell = copy.deepcopy(cell)
    key = serial_key(cell.get("serial", ""))
    if "newtons_per_volt" not in cell:
        if key not in SHEETS:
            raise ValueError(f"load cell {name}: serial {cell.get('serial')!r} has no calibration "
                             "sheet in loadcell_cal.SHEETS and no newtons_per_volt")
        npv = newtons_per_volt(key)
        cell["newtons_per_volt"] = {a: round(float(v), 2) for a, v in zip("XYZ", npv)}
    for axis in cell.pop("flip", []):
        cell["newtons_per_volt"][axis] = -abs(cell["newtons_per_volt"][axis])
    if "crosstalk" not in cell and key in SHEETS:
        cell["crosstalk"] = [list(r) for r in SHEETS[key].crosstalk]
    return cell


def parse_layout(raw: dict, path: Path) -> Layout:
    """A layout from the contents of forceplate_layout.json (or a session's copy of it)."""
    raw = copy.deepcopy(raw)
    raw["load_cells"] = {c: resolve_cell(raw["load_cells"][c], c) for c in CELLS}
    cells = raw["load_cells"]
    return Layout(
        deck_length_m=float(raw["deck"]["length_m"]),
        deck_width_m=float(raw["deck"]["width_m"]),
        where={c: cells[c]["where"] for c in CELLS},
        positions=np.array([[cells[c]["x_m"], cells[c]["y_m"]] for c in CELLS], dtype=float),
        newtons_per_volt=np.array([cells[s[0]]["newtons_per_volt"][s[1]] for s in fp.SIGNAL_NAMES],
                                  dtype=float),
        placeholder=bool(raw.get("scale_is_placeholder", False)),
        path=Path(path),
        serials={c: cells[c].get("serial", "") for c in CELLS},
        uncross=np.array([np.linalg.inv(np.array(cells[c].get("crosstalk", np.eye(3)), dtype=float))
                          for c in CELLS]),
        raw=raw,
    )


def load_layout(path: Path = LAYOUT_FILE) -> Layout:
    """Read forceplate_layout.json (see the "about" lines in it)."""
    return parse_layout(json.loads(Path(path).read_text(encoding="utf-8")), path)


def layout_for_wiring(layout: Layout, wiring: dict[str, str]) -> Layout:
    """
    The same deck with serials on different inputs: each input takes the corner,
    position, scale and crosstalk of the serial now plugged into it.
    """
    by_serial = {serial_key(c["serial"]): c for c in layout.raw["load_cells"].values()}
    raw = copy.deepcopy(layout.raw)
    raw["load_cells"] = {c: copy.deepcopy(by_serial[wiring[c]]) for c in CELLS}
    return parse_layout(raw, layout.path)


def session_layout(folder: Path, default: Layout) -> Layout:
    """
    The layout a session was recorded with: the copy in its session.json, so a
    later rewiring or new scale doesn't silently change old data. Sessions from
    before that copy was kept get `default`.
    """
    try:
        raw = json.loads((Path(folder) / "session.json").read_text(encoding="utf-8")).get("layout")
    except (OSError, ValueError):
        raw = None
    return parse_layout(raw, Path(folder) / "session.json") if raw else default


# ---------------------------------------------------------------------------
# Which serial is on which input, from resting volts
# ---------------------------------------------------------------------------

IDENTIFY_MAX_V = 0.15     # mean X/Y mismatch per input above which we don't trust a match


@dataclass
class Wiring:
    assignment: dict[str, str]    # input "1".."4" -> serial key
    cost_v: float                 # sum over inputs of |rest XY - sheet XY|, volts
    runner_up_v: float            # the same for the second-best assignment

    @property
    def trusted(self) -> bool:
        return self.cost_v / len(self.assignment) <= IDENTIFY_MAX_V


def identify_serials(rest_volts: np.ndarray, serials: list[str] | None = None) -> Wiring:
    """
    Which serial is on which input: match each input's resting X and Y volts
    (12,) to the sheets' zero offsets, over every assignment. Z is left out:
    it carries the deck's weight.
    """
    serials = serials or list(SHEETS)
    rest = np.asarray(rest_volts, dtype=float).reshape(4, 3)[:, :2]
    ref = {s: expected_unbalance_v(s)[:2] for s in serials}
    ranked = sorted((sum(float(np.linalg.norm(rest[i] - ref[s])) for i, s in enumerate(perm)), perm)
                    for perm in itertools.permutations(serials, 4))
    (cost, best), (runner_up, _) = ranked[0], ranked[1]
    return Wiring(dict(zip(CELLS, best)), cost, runner_up)


def rest_volts_for_session(folder: Path) -> np.ndarray | None:
    """
    (12,) resting volts for a session: the median of its tares' means if it has
    tares, else the median over its captures of each one's first 200 samples.
    """
    folder = Path(folder)
    try:
        trials = json.loads((folder / "session.json").read_text(encoding="utf-8")).get("trials", [])
    except (OSError, ValueError):
        trials = []
    tares = [t["mean_v"] for t in trials if t.get("kind") == "tare" and "mean_v" in t]
    if tares:
        return np.median(np.array(tares), axis=0)
    starts = []
    for csv_path in sorted(folder.glob("trial_*.csv")):
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1, max_rows=200, ndmin=2)
        if data.shape[1] == 13:
            starts.append(np.median(data[:, 1:], axis=0))
    return np.median(np.array(starts), axis=0) if starts else None


def calibration_for_session(folder: Path, default: Layout) -> tuple[Layout, str | None]:
    """
    The layout to read a session with, and a note when it isn't `default`:
    the session's stored copy if it has one; otherwise the wiring identified
    from its resting volts (when that match is clear), else `default`.
    """
    stored = session_layout(folder, default)
    if stored is not default:
        return stored, None
    rest = rest_volts_for_session(folder)
    if rest is None:
        return default, None
    found = identify_serials(rest, sorted(set(default.wiring.values())))
    if not found.trusted:
        return default, (f"wiring not identified: rest volts are {found.cost_v / 4:.2f} V "
                         "per input from the sheets")
    if found.assignment == default.wiring:
        return default, None
    moved = [c for c in CELLS if found.assignment[c] != default.wiring[c]]
    return layout_for_wiring(default, found.assignment), (
        "rest volts show inputs " + ",".join(moved) + " carrying serials "
        + ",".join(found.assignment[c] for c in moved) + f", not "
        + ",".join(default.wiring[c] for c in moved) + f" as in {default.path.name}; using that wiring")


# ---------------------------------------------------------------------------
# Volts to newtons with a layout
# ---------------------------------------------------------------------------

def cell_forces(volts: np.ndarray, zero: np.ndarray | None, layout: Layout) -> np.ndarray:
    """
    Volts (..., 12) -> newtons (..., 12), each cell in its own axes. No zero: untared.
    Each channel is scaled by its own N/V, then each cell's three readings are
    corrected for crosstalk with the inverse of its sheet's matrix (worked on
    magnitudes, so a negative N/V only flips the axis afterwards).
    """
    npv = layout.newtons_per_volt
    read = (volts - (0.0 if zero is None else zero)) * np.abs(npv)
    read = read.reshape(*read.shape[:-1], 4, 3)
    true = np.einsum("cij,...cj->...ci", layout.uncross, read)
    return true.reshape(*true.shape[:-2], 12) * np.sign(npv)


def cop_from_cells(forces: np.ndarray, layout: Layout,
                   min_n: float = 20.0) -> tuple[float, float, float] | None:
    """
    Centre of pressure from geometry alone: the four vertical (Z) forces, (12,)
    newtons, weighted by where the cells sit. Returns (x, y, total Fz), or None
    when the total is under `min_n` and the ratio would only be noise.
    """
    fz = forces[[fp.SIGNAL_NAMES.index(f"{c}Z") for c in CELLS]]
    total = fz.sum()
    if abs(total) < min_n:
        return None
    x, y = fz @ layout.positions / total
    return float(x), float(y), float(total)


# ---------------------------------------------------------------------------
# The 6x12 deck fit
# ---------------------------------------------------------------------------

LOAD_NAMES = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
LOAD_UNITS = ["N", "N", "N", "N·m", "N·m", "N·m"]


def known_load(direction: str, force_n: float, x: float, y: float, z: float) -> np.ndarray:
    """[Fx, Fy, Fz, Mx, My, Mz] for a force of `force_n` newtons applied at (x, y, z)."""
    f = force_n * np.asarray(fp.DIRECTIONS[direction])
    m = np.cross([x, y, z], f)
    return np.concatenate([f, m])


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


def gather(session: fp.Session) -> tuple[list[int], np.ndarray, np.ndarray]:
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


def save_fit(fit: Fit, session: fp.Session) -> tuple[Path, Path]:
    json_path = session.folder / "calibration.json"
    csv_path = session.folder / "calibration_matrix.csv"
    fp._write_atomic(json_path, json.dumps({
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "session": str(session.folder),
        "equation": "[Fx,Fy,Fz,Mx,My,Mz] = matrix @ (volts - tare_volts)",
        "frame": session.meta["deck"]["frame"],
        "rows": LOAD_NAMES,
        "row_units": LOAD_UNITS,
        "columns": fp.SIGNAL_NAMES,
        "matrix": fit.matrix.tolist(),
        "trial_ids": fit.trial_ids,
        "rank": fit.rank,
        "rms_error": dict(zip(LOAD_NAMES, fit.rms_error.tolist())),
    }, indent=2))
    header = "load," + ",".join(fp.SIGNAL_NAMES)
    lines = [header] + [f"{name}," + ",".join(f"{v:.6g}" for v in row)
                        for name, row in zip(LOAD_NAMES, fit.matrix)]
    fp._write_atomic(csv_path, "\n".join(lines) + "\n")
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
    print("       " + "".join(f"{s:>9}" for s in fp.SIGNAL_NAMES))
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


def run_fit(folder: Path) -> int:
    """Fit the 6x12 matrix for a session, print it and save it beside the session."""
    session = fp.Session.load(folder)
    print(f"Session {session.folder}: {len(session.trials)} captures")
    ids, signals, known = gather(session)
    if not ids:
        print("No usable load trials (each needs a tare captured before it).")
        return 1
    fit = fit_calibration(ids, signals, known)
    report(fit)
    json_path, csv_path = save_fit(fit, session)
    print(f"\nSaved {json_path} and {csv_path}")
    return 0


# ---------------------------------------------------------------------------
# Session report
# ---------------------------------------------------------------------------

def known_load_check(folder: Path, layout: Layout) -> list[tuple[dict, float]]:
    """
    For each load trial with a tare: the total measured force along the
    entered direction (middle 60% of the capture, tared), in newtons.
    """
    meta = json.loads((Path(folder) / "session.json").read_text(encoding="utf-8"))
    by_id = {t["id"]: t for t in meta["trials"]}
    out = []
    for t in meta["trials"]:
        tare = by_id.get(t.get("tare_id"))
        if t.get("kind") != "load" or tare is None or "direction" not in t:
            continue
        volts = np.loadtxt(Path(folder) / t["file"], delimiter=",", skiprows=1, ndmin=2)[:, 1:]
        n = len(volts)
        f = cell_forces(volts[n // 5: 4 * n // 5].mean(axis=0), np.array(tare["mean_v"]), layout)
        total = f.reshape(4, 3).sum(axis=0) @ np.asarray(fp.DIRECTIONS[t["direction"]])
        out.append((t, float(total)))
    return out


def session_report(folder: Path, default: Layout):
    folder = Path(folder)
    print(f"Session {folder}")
    rest = rest_volts_for_session(folder)
    if rest is not None:
        found = identify_serials(rest, sorted(set(default.wiring.values())))
        print("\nResting X / Y per input (V) against the sheets' zero offsets on our amplifier:")
        for c in CELLS:
            x, y = rest.reshape(4, 3)[int(c) - 1, :2]
            s = found.assignment[c]
            ex, ey = expected_unbalance_v(s)[:2]
            print(f"  input {c}: {x:+.3f} / {y:+.3f}   closest {s}: {ex:+.3f} / {ey:+.3f}"
                  f"   (expected {default.wiring[c]})")
        print(f"  identified {found.assignment}  mismatch {found.cost_v:.3f} V, "
              f"next best {found.runner_up_v:.3f} V" + ("" if found.trusted else "  NOT TRUSTED"))
    layout, note = calibration_for_session(folder, default)
    if layout is default:
        print(f"\nUsing {default.path.name}" + (f" ({note})" if note else ""))
    else:
        print(f"\nUsing: {note or 'the layout stored with the session'}")
    print("Newtons per volt in use (X / Y / Z), crosstalk corrected per cell:")
    for i, c in enumerate(CELLS):
        npv = layout.newtons_per_volt[3 * i:3 * i + 3]
        print(f"  input {c} {layout.serials[c]:14} {layout.where[c]:12} "
              + " / ".join(f"{v:7.1f}" for v in npv))
    checks = known_load_check(folder, layout) if (folder / "session.json").exists() else []
    if checks:
        print("\nKnown loads (tared, middle 60% of each capture):")
        for t, total in checks:
            entered = t["force_n"]
            print(f"  trial {t['id']:3d}: {total:7.1f} N along '{t['direction']}' vs "
                  f"{entered:6.1f} N entered ({100 * (total / entered - 1):+.1f}%)  {t.get('notes', '')}")


def main():
    parser = argparse.ArgumentParser(description="Load-cell calibration: wiring, scales, deck fit.")
    parser.add_argument("session", type=Path, nargs="?", help="session folder containing session.json")
    parser.add_argument("--fit", action="store_true", help="also fit and save the 6x12 deck matrix")
    parser.add_argument("--check-sheets", action="store_true",
                        help="re-read the calibration PDFs and compare them with SHEETS")
    args = parser.parse_args()
    if args.check_sheets:
        print(f"Calibration sheets in {SHEET_DIR}:")
        if not check_sheets():
            sys.exit(1)
    if args.session is None:
        if not args.check_sheets:
            parser.error("give a session folder, or --check-sheets")
        return
    session_report(args.session, load_layout())
    if args.fit:
        print()
        sys.exit(run_fit(args.session))


if __name__ == "__main__":
    main()
