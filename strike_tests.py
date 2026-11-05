#!/usr/bin/env python3
"""
Strike (tap) test analysis: resonant frequencies of the deck.

Strike the deck once during a normal 5 s capture. For each trial this finds the
strike, takes the spectrum of the ~1 s ring-down, labels the strongest modes,
gives each load cell's response as the length (norm) of its X/Y/Z vector,
splits each mode by axis, and compares the phase of the four Z cells to show
the mode's shape (bounce, pitch, roll, twist). It writes

  strike_analysis/trial_009_resonance.png   one figure per trial
  strike_analysis/strike_report.md          tables for all trials, figures linked

into the session folder, and prints the same tables.

Units: each channel is turned into newtons with the per-axis newtons_per_volt
in forceplate_layout.json before anything else, so the X/Y/Z norms add like
forces. While that file says scale_is_placeholder, every channel has the same
scale and the numbers are only relative.

Method: the strike is the first sample where the summed deviation from the
quiet start exceeds its mean + 10 std. The 1.024 s after it (mean and step
removed, Hann window, zero-padded FFT) gives the spectrum; the 1 s before it is
the noise floor. All 12 channels are combined by summing power. Peak positions
are good to about +/-0.5 Hz (the window's real resolution is ~1 Hz).

Per-cell norms: in the spectra, a cell's response at each frequency is
sqrt(|X(f)|^2 + |Y(f)|^2 + |Z(f)|^2). Taking the norm sample by sample first
would rectify the signal and move every mode to twice its frequency. The time
plot shows the sample-by-sample norm of each cell's deviation from rest.
Mode shapes still use the Z phases: a norm has no sign, so it cannot say
which corners move together.

Usage:
  uv run strike_tests.py "<data folder>/20261104-163731" 5 8 9 10
  uv run strike_tests.py "<session>"                # every trial whose notes say "strike"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import forceplate as fp

WINDOW = 1024             # samples analysed after the strike (1.024 s at 1 kHz)
NFFT = 16384              # zero-padded FFT length
N_PEAKS = 5
PEAK_SEP_HZ = 3.0         # peaks closer than this count as one
F_MIN, F_MAX = 5.0, 480.0
Z = [2, 5, 8, 11]         # 1Z 2Z 3Z 4Z
BANDS = [(1, 20), (20, 30), (30, 45), (45, 60), (60, 200)]   # Hz, for the per-cell comparison
CELL_NAMES = {1: "front left", 2: "front right", 3: "right rear", 4: "left rear"}
# Same rule as the GUI: colour = component (here always |F|, the norm, dark
# grey); the four cells share a plot, so they differ by line style.
NORM_COLOR = "#333333"
CELL_LINESTYLES = {"1": "-", "2": "--", "3": ":", "4": "-."}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def find_strike(forces: np.ndarray, quiet: int = 300) -> int:
    dev = np.abs(forces - np.median(forces[:quiet], 0)).sum(1)
    above = dev > dev[:quiet].mean() + 10 * dev[:quiet].std()
    if not above.any():
        raise ValueError("no strike found")
    return int(np.argmax(above))


def spectrum(seg: np.ndarray, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Complex amplitude spectrum of each column, mean and step removed."""
    seg = seg - seg.mean(0)
    seg = seg - np.linspace(0, 1, len(seg))[:, None] * (seg[-1] - seg[0])
    w = np.hanning(len(seg))[:, None]
    return np.fft.rfftfreq(NFFT, 1 / rate), np.fft.rfft(seg * w, NFFT, axis=0) * 2 / w.sum()


def pick_peaks(f: np.ndarray, p: np.ndarray, n: int = N_PEAKS) -> list[int]:
    """Indices of the n largest local maxima in [F_MIN, F_MAX], at least PEAK_SEP_HZ apart."""
    local = np.r_[False, (p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:]), False]
    local &= (f > F_MIN) & (f < F_MAX)
    picked: list[int] = []
    for i in np.argsort(p)[::-1]:
        if local[i] and all(abs(f[i] - f[j]) > PEAK_SEP_HZ for j in picked):
            picked.append(int(i))
            if len(picked) == n:
                break
    return picked


def cell_norms(spec: np.ndarray) -> np.ndarray:
    """(freqs, 12) complex spectra -> (freqs, 4) per-cell vector norms."""
    return np.sqrt((np.abs(spec.reshape(len(spec), 4, 3)) ** 2).sum(2))


def mode_shape(phases: np.ndarray, amps: np.ndarray) -> str:
    """
    Name the motion from the Z phase of cells 1-4 (degrees, relative to the
    largest). Cells too weak to read (< 10% of the largest) are left out.
    """
    live = amps > 0.1 * amps.max()
    # A standing mode moves its corners in phase or in antiphase. Phases near
    # +/-90 deg mean two nearby modes are mixed in the window: don't name it.
    if any(60 < abs(phases[c]) < 120 for c in range(4) if live[c]):
        return "mixed (phases near 90°)"
    sign = {c + 1: np.cos(np.radians(phases[c])) > 0 for c in range(4) if live[c]}
    cells = sorted(sign)
    if all(sign[c] == sign[cells[0]] for c in cells):
        return "all in phase: bounce"
    up = [c for c in cells if sign[c]]
    down = [c for c in cells if not sign[c]]
    front, rear, left, right = {1, 2}, {3, 4}, {1, 4}, {2, 3}
    for a, b, name in ((front, rear, "pitch (front vs rear)"),
                       (left, right, "roll (left vs right)"),
                       ({1, 3}, {2, 4}, "twist (diagonals)")):
        if (set(up) <= a and set(down) <= b) or (set(up) <= b and set(down) <= a):
            return name
    return "cells " + ",".join(map(str, up)) + " against " + ",".join(map(str, down))


def analyse(path: Path, layout: fp.Layout) -> dict:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    t = data[:, 0]
    forces = fp.cell_forces(data[:, 1:], None, layout)    # newtons; the mean is removed below
    rate = 1 / np.median(np.diff(t))
    k0 = find_strike(forces)
    if k0 + WINDOW > len(forces) or k0 < WINDOW + 50:
        raise ValueError(f"strike at {t[k0]:.2f} s is too close to the edge of the capture")
    f, post = spectrum(forces[k0:k0 + WINDOW], rate)
    _, pre = spectrum(forces[k0 - WINDOW - 50:k0 - 50], rate)
    deck_post = np.sqrt((np.abs(post) ** 2).sum(1))
    deck_pre = np.sqrt((np.abs(pre) ** 2).sum(1))

    modes = []
    for i in sorted(pick_peaks(f, deck_post), key=lambda i: f[i]):
        a = post[i]
        axis = {name: float(np.sqrt((np.abs(a[k::3]) ** 2).sum())) for k, name in enumerate("XYZ")}
        z = a[Z]
        ref = int(np.argmax(np.abs(z)))
        phases = np.degrees(np.angle(z / z[ref]))
        modes.append({"hz": float(f[i]), "amp_n": float(deck_post[i]), "axis_n": axis,
                      "cell_norm_n": cell_norms(a[None])[0].tolist(),
                      "z_amp_n": np.abs(z).tolist(), "z_phase_deg": phases.tolist(),
                      "shape": mode_shape(phases, np.abs(z))})
    strongest = max(modes, key=lambda m: m["amp_n"])
    # Each cell's norm in each band, as a fraction of the other three cells' mean
    norms = cell_norms(post)
    bands = {}
    for lo, hi in BANDS:
        e = np.sqrt((norms[(f >= lo) & (f < hi)] ** 2).sum(0))
        bands[f"{lo}-{hi}"] = [float(e[c] / np.delete(e, c).mean()) for c in range(4)]
    return {"file": path.name, "strike_s": float(t[k0]), "k0": k0, "rate": rate,
            "t": t, "forces": forces, "f": f, "post": post, "deck_post": deck_post,
            "deck_pre": deck_pre, "modes": modes, "strongest_hz": strongest["hz"],
            "cell_norm_spec": norms, "bands": bands}


# ---------------------------------------------------------------------------
# Figure and report
# ---------------------------------------------------------------------------

def plot(r: dict, title: str, out: Path, unit: str = "N"):
    t, v, k0, f = r["t"], r["forces"], r["k0"], r["f"]
    f0 = r["strongest_hz"]
    fig = plt.figure(figsize=(12, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.3, 1])

    ax = fig.add_subplot(gs[0, :])
    rest = v[k0 - 300:k0 - 50].mean(0)
    dev = np.linalg.norm((v - rest).reshape(len(v), 4, 3), axis=2)
    for cell in "1234":
        ax.plot(t - t[k0], dev[:, int(cell) - 1], color=NORM_COLOR, ls=CELL_LINESTYLES[cell],
                lw=0.9, label=f"cell {cell} |ΔF|")
    ax.axvspan(0, WINDOW / r["rate"], color="#888", alpha=0.08)
    ax.set_xlim(-0.2, 1.5)
    ax.set_xlabel("time from strike (s)")
    ax.set_ylabel(f"|ΔF| ({unit})")
    ax.set_title(f"{title}: each cell's X/Y/Z vector, length (shaded = window analysed)")
    ax.legend(ncol=4, loc="upper right")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, :])
    ax.semilogy(f, r["deck_post"], color="#222", lw=1.2, label="after strike (all 12 channels)")
    ax.semilogy(f, r["deck_pre"], color="#aaa", lw=1, label="before strike (noise floor)")
    for m in r["modes"]:
        main = m["hz"] == f0
        ax.annotate(f"{m['hz']:.1f} Hz", (m["hz"], m["amp_n"]), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=10,
                    fontweight="bold" if main else "normal", color="#c22" if main else "#333")
    ax.set_xlim(0, 150)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(f"amplitude ({unit})")
    ax.set_title(f"Resonance: {f0:.1f} Hz  (strongest peak; others labelled)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, which="both")

    ax = fig.add_subplot(gs[2, 0])
    for cell in "1234":
        ax.semilogy(f, r["cell_norm_spec"][:, int(cell) - 1], color=NORM_COLOR,
                    ls=CELL_LINESTYLES[cell], lw=1, label=f"cell {cell}")
    ax.axvline(f0, color="#c22", lw=0.8, ls="--")
    ax.set_xlim(5, 100)
    ax.set_ylim(r["cell_norm_spec"][(f > 5) & (f < 100)].max() * 1e-4, None)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel(f"amplitude ({unit})")
    ax.set_title("Per load cell (norm of X, Y, Z)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    # Spectrogram of the summed power of all 12 channels: which modes ring longest.
    ax = fig.add_subplot(gs[2, 1])
    win, hop, nf = 256, 8, 2048
    starts = np.arange(k0 - 100, min(len(v) - win, k0 + 1500), hop)
    hw = np.hanning(win)[:, None]
    frames = []
    for k in starts:
        blk = v[k:k + win]
        blk = blk - blk.mean(0)
        frames.append(np.sqrt((np.abs(np.fft.rfft(blk * hw, nf, axis=0)) ** 2).sum(1)))
    S = 20 * np.log10(np.array(frames).T + 1e-9)
    fs = np.fft.rfftfreq(nf, 1 / r["rate"])
    keep = fs <= 100
    im = ax.pcolormesh((starts + win / 2 - k0) / r["rate"], fs[keep], S[keep], shading="auto",
                       cmap="magma", vmin=S[keep].max() - 50, vmax=S[keep].max())
    ax.axhline(f0, color="w", lw=0.6, ls="--")
    ax.set_xlabel("time from strike (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("All channels over time (dB)")
    fig.colorbar(im, ax=ax, label="dB")

    fig.savefig(out, dpi=110)
    plt.close(fig)


def mode_table(r: dict) -> list[str]:
    lines = ["| Mode (Hz) | Deck (N) | X (N) | Y (N) | Z (N) | Cell norm, cells 1-4 (N) | Z phase, cells 1-4 (deg) | Motion |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m in r["modes"]:
        ax = m["axis_n"]
        lines.append(f"| {m['hz']:.1f} | {m['amp_n']:.1f} | {ax['X']:.2f} | {ax['Y']:.2f} | "
                     f"{ax['Z']:.2f} | " + " ".join(f"{a:.2f}" for a in m["cell_norm_n"]) + " | "
                     + " ".join(f"{p:+.0f}" for p in m["z_phase_deg"]) + f" | {m['shape']} |")
    return lines


def band_table(results: list[tuple[dict, dict]]) -> list[str]:
    """Each cell's norm per band, divided by the mean of the other three cells."""
    head = " | ".join(f"{b} Hz" for b in results[0][1]["bands"])
    lines = [f"| Trial | Cell | {head} |", "| --- | --- |" + " --- |" * len(BANDS)]
    for entry, r in results:
        for c in range(4):
            lines.append(f"| {entry['id']} | {c + 1} | "
                         + " | ".join(f"{v[c]:.2f}" for v in r["bands"].values()) + " |")
    return lines


def write_report(session: Path, results: list[tuple[dict, dict]], out_dir: Path,
                 placeholder: bool) -> Path:
    lines = [f"# Strike tests: {session.name}", "",
             "| Trial | Notes | Strike (s) | Strongest mode (Hz) | Other modes (Hz) |",
             "| --- | --- | --- | --- | --- |"]
    for entry, r in results:
        others = sorted((m for m in r["modes"] if m["hz"] != r["strongest_hz"]),
                        key=lambda m: -m["amp_n"])
        lines.append(f"| {entry['id']} | {entry.get('notes', '').strip()} | {r['strike_s']:.3f} | "
                     f"{r['strongest_hz']:.1f} | " + ", ".join(f"{m['hz']:.1f}" for m in others) + " |")
    lines += ["", "Z cells: 1 front left, 2 front right, 3 right rear, 4 left rear. Phases are "
              "relative to the cell with the largest response; cells under 10% of it are not "
              "used to name the motion. Amplitudes are newtons from forceplate_layout.json"
              + (" (PLACEHOLDER scale: equal for every channel, so only relative)." if placeholder
                 else "."), "",
              "## Each cell against the others",
              "",
              "A cell's norm of X, Y and Z in each frequency band, after the strike, divided "
              "by the mean of the other three cells. 1 = the same as the rest.", "",
              *band_table(results), ""]
    for entry, r in results:
        lines += [f"## Trial {entry['id']}", "", *mode_table(r), "",
                  f"![trial {entry['id']}]({Path(r['file']).stem}_resonance.png)", ""]
    path = out_dir / "strike_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Resonant frequencies from strike-test captures.")
    parser.add_argument("session", type=Path, help="session folder containing session.json")
    parser.add_argument("trials", type=int, nargs="*",
                        help="trial ids (default: every trial whose notes mention 'strike')")
    parser.add_argument("--out", type=Path, help="output folder (default: <session>/strike_analysis)")
    args = parser.parse_args()

    meta = json.loads((args.session / "session.json").read_text(encoding="utf-8"))
    by_id = {t["id"]: t for t in meta["trials"]}
    ids = args.trials or [t["id"] for t in meta["trials"] if "strike" in t.get("notes", "").lower()]
    if not ids:
        parser.error("no trials given and none have 'strike' in their notes")
    out_dir = args.out or args.session / "strike_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    layout = fp.load_layout()
    unit = "N, placeholder scale" if layout.placeholder else "N"
    if layout.placeholder:
        print(f"Note: {layout.path.name} still has placeholder scales; newtons are only relative.")

    results = []
    for tid in ids:
        entry = by_id[tid]
        try:
            r = analyse(args.session / entry["file"], layout)
        except ValueError as e:
            print(f"trial {tid}: skipped, {e}")
            continue
        plot(r, f"{args.session.name}/{entry['file']}", out_dir / f"{Path(entry['file']).stem}_resonance.png", unit)
        results.append((entry, r))
        print(f"\ntrial {tid} ({entry.get('notes', '').strip()}): strike at {r['strike_s']:.3f} s, "
              f"strongest {r['strongest_hz']:.1f} Hz")
        print("\n".join(mode_table(r)))

    if results:
        print("\nEach cell's norm per band, / mean of the other three cells:")
        print("\n".join(band_table(results)))
        print(f"\nWrote {write_report(args.session, results, out_dir, layout.placeholder)} and "
              f"{len(results)} figure(s) in {out_dir}")


if __name__ == "__main__":
    main()
