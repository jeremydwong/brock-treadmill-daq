#!/usr/bin/env python3
"""
How the whole treadmill moves: strike (hammer) tests read as deck motions.

The twelve channels are four vertical forces and eight horizontal ones. Four
vertical forces can only do four things, so every vertical reading is a mix of

  Fz total   all four corners together          (the deck bouncing; what gait analysis uses)
  pitch      front pair minus back pair         (rocking nose up / nose down)
  roll       left pair minus right pair         (rocking side to side)
  twist      one diagonal minus the other       (racking: carries no net load)

and these four add up to exactly the same power as the four Z channels. Each
resonance is then named by how its power splits between them, with its damping.

This file is the arithmetic (numpy only); modes_gui.py is the window.

  uv run deck_modes.py "<session folder>"            # strike report (PDF) of every hit
  uv run deck_modes.py "<session folder>" 4 5 6      # only these trials
  uv run deck_modes.py "<session folder>" --out report.pdf

Corners are found from the cell positions in the layout, not from the input
numbers, so a session recorded with two cables swapped still reads correctly.

Method. The hit is the first sample where the summed deviation of all twelve
forces passes 20% of its largest value. The ring-down is the second after it,
skipping the first 25 ms (the blow itself, which holds every frequency). Its
spectrum: mean and trend removed, Hann window, zero-padded FFT; peak positions
are good to about +/-0.5 Hz. Damping: each peak is band-passed (+/-2 Hz) and a
straight line fitted to the log of its envelope; two modes inside that band
beat, the line fits badly, and the damping is then left blank.

Filters here have a Butterworth-shaped gain applied in the frequency domain,
so they shift nothing in time (zero phase) and need no extra library.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import forceplate as fp
import loadcell_cal as lc

RING_SKIP_S = 0.025       # leave out the blow itself
RING_SECONDS = 1.0        # length of ring-down analysed
QUIET_GAP_S = 0.05        # the "before the hit" window ends this long before it
STRIKE_RATIO = 15.0       # a hit: largest deviation at least this many times the median
STRIKE_FRACTION = 0.2     # ... and it starts where the deviation first passes this much of it
NFFT = 16384              # zero-padded FFT length (0.06 Hz steps at 1 kHz)
F_MIN, F_MAX = 5.0, 100.0 # peaks are looked for in this range, Hz
PEAK_SEP_HZ = 2.5         # peaks closer than this count as one
N_PEAKS = 8
PEAK_FLOOR = 0.05         # ignore peaks under this fraction of the strongest
DAMPING_BAND_HZ = 2.0     # half-width of the band-pass used for the decay fit
BANDS = [(0, 10), (10, 25), (25, 35), (35, 45), (45, 60), (60, 100)]   # Hz, for power fractions

# Same colour rule as the calibration GUI: colour = component.
AXIS_COLORS = {"X": "#3366cc", "Y": "#dd8800", "Z": "#33aa55", "|F|": "#333333"}
MOTION_COLORS = {"pitch": "#8e44ad", "roll": "#c0392b", "twist": "#008b8b"}
CELL_LINESTYLES = ["-", "--", ":", "-."]        # inputs 1-4, as in the Plot Results window
# Solid greys rather than transparency, which EPS cannot store
SHADE_COLOR, MARK_COLOR, GRID_COLOR = "#eeeeee", "#cccccc", "#e6e6e6"

CORNERS = ["front left", "front right", "back right", "back left"]
PATTERNS = {              # weights on the corners above
    "Fz total": [1, 1, 1, 1],
    "pitch": [1, 1, -1, -1],
    "roll": [1, -1, -1, 1],
    "twist": [1, -1, 1, -1],
}
CORNER_SHORT = ["FL", "FR", "BR", "BL"]         # CORNERS, as written in the formulas


def pattern_formulas() -> list[str]:
    """Each pattern written out from PATTERNS, e.g. 'pitch = FL + FR − BR − BL'."""
    lines = []
    for name, weights in PATTERNS.items():
        terms = "".join(f" {'+' if w > 0 else '−'} {c}" for c, w in zip(CORNER_SHORT, weights))
        lines.append(f"{name:8} = {terms[3:]}")
    return lines + ["each the Z (vertical) force at that corner:",
                    "F front, B back, L left, R right"]


MOTION_NAMES = {"Fz total": "bounce (all corners together)", "pitch": "pitch (front against back)",
                "roll": "roll (left against right)", "twist": "twist (diagonals)"}


# ---------------------------------------------------------------------------
# Signals: sums and differences of the twelve channels
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Signal:
    """One plotted line: a weighted sum of the 12 forces, or one cell's |F|."""
    name: str
    weights: tuple[float, ...] | None = None    # (12,) for a weighted sum
    cell: int | None = None                     # input 0-3, for the norm of its X, Y, Z
    color: str = "#333333"
    ls: str = "-"

    def series(self, forces: np.ndarray) -> np.ndarray:
        """(samples, 12) -> (samples,). |F| is the length of the cell's force vector."""
        if self.cell is None:
            return forces @ np.array(self.weights)
        return np.sqrt((forces[:, 3 * self.cell:3 * self.cell + 3] ** 2).sum(1))

    def amplitude(self, spec: np.ndarray) -> np.ndarray:
        """
        (freqs, 12) complex spectra -> (freqs,) amplitude. For |F| the norm is
        taken per frequency: a norm of the time signal rectifies it and moves
        every mode to twice its frequency.
        """
        if self.cell is None:
            return np.abs(spec @ np.array(self.weights))
        return np.sqrt((np.abs(spec[:, 3 * self.cell:3 * self.cell + 3]) ** 2).sum(1))


def corner_inputs(layout: lc.Layout) -> list[int]:
    """Input (0-3) at each of CORNERS, from where the cells sit (+x right, +y back)."""
    x, y = layout.positions[:, 0], layout.positions[:, 1]
    front, left = y < y.mean(), x < x.mean()
    found = []
    for want_front, want_left in ((True, True), (True, False), (False, False), (False, True)):
        match = np.flatnonzero((front == want_front) & (left == want_left))
        if len(match) != 1:
            raise ValueError("the layout's cell positions do not make one cell per corner")
        found.append(int(match[0]))
    return found


def z_weights(layout: lc.Layout, corner_weights) -> tuple[float, ...]:
    """Weights on CORNERS -> (12,) weights on the channels' Z columns."""
    w = np.zeros(12)
    for corner, weight in zip(corner_inputs(layout), corner_weights):
        w[3 * corner + 2] = weight
    return tuple(w)


def axis_weights(layout: lc.Layout, corner_weights, axis: str) -> np.ndarray:
    """Weights on CORNERS -> (12,) weights on that axis (X, Y or Z) of the channels."""
    w = np.zeros(12)
    for corner, weight in zip(corner_inputs(layout), corner_weights):
        w[3 * corner + "XYZ".index(axis)] = weight
    return w


def yaw_weights(layout: lc.Layout) -> np.ndarray:
    """
    (12,) weights giving the turning moment about the vertical through the
    middle of the four cells, divided by the cells' mean distance from it: the
    sideways force at the corners that would turn the treadmill, in newtons.
    """
    arm = layout.positions - layout.positions.mean(0)       # (4, 2) x, y from the middle
    reach = np.linalg.norm(arm, axis=1).mean()
    w = np.zeros(12)
    w[0::3] = -arm[:, 1] / reach         # Mz = sum(x Fy - y Fx)
    w[1::3] = arm[:, 0] / reach
    return w


def horizontal_partners(layout: lc.Layout) -> list[tuple[str, str, np.ndarray]]:
    """
    For pitch, roll and twist in turn, the sideways signal that goes with it:
    (its partner's name, its own name, its (12,) weights).
      pitch: front Y minus back Y   the frame stretched or squeezed along its length
      roll:  left X minus right X   the same across its width
      twist: yaw                    the frame turned about the vertical
    """
    return [("pitch", "Y: front − back", axis_weights(layout, PATTERNS["pitch"], "Y")),
            ("roll", "X: left − right", axis_weights(layout, PATTERNS["roll"], "X")),
            ("twist", "yaw", yaw_weights(layout))]


def weights_formula(layout: lc.Layout, weights: np.ndarray) -> str:
    """Weights written out by corner: 'FLy + FRy − BRy − BLy', '0.95 (FLx + ...) + 0.32 (−FLy ...)'."""
    parts = []
    for k, axis in enumerate("xyz"):
        used = [(short, weights[3 * c + k]) for short, c in zip(CORNER_SHORT, corner_inputs(layout))
                if abs(weights[3 * c + k]) > 1e-9]
        if not used:
            continue
        size = abs(used[0][1])              # one size per axis in every signal we make
        terms = "".join(f" {'+' if w > 0 else '−'} {short}{axis}" for short, w in used)
        terms = terms[3:] if terms.startswith(" +") else "−" + terms[3:]
        parts.append(terms if abs(size - 1) < 1e-9 else f"{size:.2f} ({terms})")
    return " + ".join(parts)


def horizontal_formulas(layout: lc.Layout) -> list[str]:
    """The sideways partners written out, for the window's 'What is plotted' box."""
    return [f"{name} = {weights_formula(layout, w)}" for _, name, w in horizontal_partners(layout)]


def pair_signal(layout: lc.Layout, cell_a: int, cell_b: int, axis: str, sign: int) -> Signal:
    """Cell a's axis plus (sign=+1) or minus (-1) cell b's: inputs 1-4, axis X, Y or Z."""
    w = np.zeros(12)
    k = "XYZ".index(axis)
    w[3 * (cell_a - 1) + k] += 1
    w[3 * (cell_b - 1) + k] += sign
    return Signal(f"{cell_a}{axis} {'+' if sign > 0 else '−'} {cell_b}{axis}", tuple(w),
                  color=AXIS_COLORS[axis], ls=CELL_LINESTYLES[(cell_a + cell_b) % 4])


def standard_signals(layout: lc.Layout) -> dict[str, list[Signal]]:
    """The signals the window offers, by group."""
    total = lambda k: tuple(np.tile(np.eye(3)[k], 4))       # noqa: E731
    return {
        "Whole treadmill": [
            Signal("Fz total", z_weights(layout, PATTERNS["Fz total"]), color=AXIS_COLORS["Z"]),
            Signal("Fx total", total(0), color=AXIS_COLORS["X"]),
            Signal("Fy total", total(1), color=AXIS_COLORS["Y"]),
        ],
        "Rocking and twisting (Z)": [
            Signal(name, z_weights(layout, PATTERNS[name]), color=MOTION_COLORS[name])
            for name in ("pitch", "roll", "twist")
        ],
        # Dashed, in the colour of the Z motion each one goes with
        "Sideways partners (X, Y)": [
            Signal(name, tuple(w), color=MOTION_COLORS[partner], ls="--")
            for partner, name, w in horizontal_partners(layout)
        ],
        "Pairs (Z)": [
            Signal(name, z_weights(layout, w), color=AXIS_COLORS["Z"], ls=ls)
            for name, w, ls in (("front pair", [1, 1, 0, 0], "--"), ("back pair", [0, 0, 1, 1], ":"),
                                ("left pair", [1, 0, 0, 1], "-."), ("right pair", [0, 1, 1, 0], (0, (1, 3))))
        ],
        "Each cell": [
            *(Signal(f"{c + 1}{axis}", tuple(np.eye(12)[3 * c + k]), color=AXIS_COLORS[axis],
                     ls=CELL_LINESTYLES[c]) for c in range(4) for k, axis in enumerate("XYZ")),
            *(Signal(f"{c + 1}|F|", cell=c, color=AXIS_COLORS["|F|"], ls=CELL_LINESTYLES[c])
              for c in range(4)),
        ],
    }


# ---------------------------------------------------------------------------
# Filters, spectra, the hit
# ---------------------------------------------------------------------------

FILTERS = ["none", "low-pass", "high-pass", "band-pass", "notch"]


def filter_gain(f: np.ndarray, kind: str, lo: float, hi: float, order: int = 4) -> np.ndarray:
    """
    Gain at each frequency. low-pass uses `hi`, high-pass uses `lo`, band-pass
    and notch use both edges. Butterworth-shaped: 0.707 at an edge.
    """
    if kind == "none":
        return np.ones_like(f)
    f = np.maximum(f, 1e-12)
    low = 1 / np.sqrt(1 + (f / max(hi, 1e-9)) ** (2 * order))
    high = 1 / np.sqrt(1 + (max(lo, 1e-9) / f) ** (2 * order))
    if kind == "low-pass":
        return low
    if kind == "high-pass":
        return high
    if kind == "band-pass":
        return low * high
    if kind == "notch":
        with np.errstate(divide="ignore"):
            return 1 / np.sqrt(1 + (f * (hi - lo) / (f ** 2 - lo * hi)) ** (2 * order))
    raise ValueError(f"unknown filter {kind!r}")


def zero_phase_filter(x: np.ndarray, rate: float, kind: str, lo: float = 0.0, hi: float = 0.0,
                      order: int = 4) -> np.ndarray:
    """
    Filter each column of x (samples, ...) without shifting it in time: the
    gain above is applied to its FFT. The ends are mirrored first (up to 1 s)
    so the filter does not see a jump where the capture starts and stops.
    """
    if kind == "none":
        return x
    if kind in ("band-pass", "notch") and not 0 < lo < hi:
        raise ValueError("the band needs 0 < low edge < high edge")
    if kind in ("low-pass", "high-pass") and not (hi if kind == "low-pass" else lo) > 0:
        raise ValueError("the cut-off must be above 0 Hz")
    n = len(x)
    pad = min(n - 1, int(rate))
    padded = np.concatenate([2 * x[0] - x[pad:0:-1], x, 2 * x[-1] - x[-2:-pad - 2:-1]])
    gain = filter_gain(np.fft.rfftfreq(len(padded), 1 / rate), kind, lo, hi, order)
    gain = gain.reshape(-1, *[1] * (x.ndim - 1))
    return np.fft.irfft(np.fft.rfft(padded, axis=0) * gain, len(padded), axis=0)[pad:pad + n]


def find_strike(forces: np.ndarray) -> int:
    """
    Row where the hit starts. Raises ValueError when the capture has no hit
    (nothing stands STRIKE_RATIO above the usual wobble).
    """
    dev = np.abs(forces - np.median(forces, axis=0)).sum(1)
    usual = np.median(dev)
    if dev.max() < STRIKE_RATIO * max(usual, 1e-12):
        raise ValueError("no hit found")
    k = int(np.argmax(dev > STRIKE_FRACTION * dev.max()))
    while k > 0 and dev[k - 1] > 3 * usual:     # walk back to where it left the noise
        k -= 1
    return k


def spectrum(seg: np.ndarray, rate: float, nfft: int = NFFT) -> tuple[np.ndarray, np.ndarray]:
    """Complex amplitude spectrum of each column: straight-line trend removed, Hann window."""
    ramp = np.linspace(-1, 1, len(seg))
    seg = seg - seg.mean(0)
    seg = seg - ramp[:, None] * (ramp @ seg) / (ramp @ ramp)      # least-squares slope
    w = np.hanning(len(seg))[:, None]
    nfft = max(nfft, 1 << (len(seg) - 1).bit_length())
    return np.fft.rfftfreq(nfft, 1 / rate), np.fft.rfft(seg * w, nfft, axis=0) * 2 / w.sum()


def spectrogram(x: np.ndarray, rate: float, window: int = 256, max_columns: int = 600
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding-window spectrum of one signal: (times, freqs, amplitude[freq, time]).
    Each column is a Hann-windowed stretch of `window` samples (0.256 s at
    1 kHz, so about 4 Hz wide), stepped so there are at most `max_columns`.
    """
    window = min(window, len(x))
    step = max(8, (len(x) - window) // max_columns + 1)
    starts = np.arange(0, len(x) - window + 1, step)
    w = np.hanning(window)
    frames = np.stack([x[s:s + window] for s in starts])
    frames = (frames - frames.mean(1, keepdims=True)) * w
    amp = np.abs(np.fft.rfft(frames, 4 * window, axis=1)) * 2 / w.sum()
    return (starts + window / 2) / rate, np.fft.rfftfreq(4 * window, 1 / rate), amp.T


def pick_peaks(f: np.ndarray, p: np.ndarray, n: int = N_PEAKS) -> list[int]:
    """Indices of up to n local maxima in [F_MIN, F_MAX], strongest first kept, sorted by Hz."""
    local = np.r_[False, (p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:]), False]
    local &= (f > F_MIN) & (f < F_MAX)
    if not local.any():
        return []
    floor = PEAK_FLOOR * p[local].max()
    picked: list[int] = []
    for i in np.argsort(p)[::-1]:
        if p[i] < floor or len(picked) == n:
            break
        if local[i] and all(abs(f[i] - f[j]) > PEAK_SEP_HZ for j in picked):
            picked.append(int(i))
    return sorted(picked)


def damping_ratio(ring: np.ndarray, rate: float, hz: float) -> float | None:
    """
    Fraction of critical damping of the mode at `hz`, from how fast it dies
    away in `ring` (samples, 12): band-pass, envelope of all channels together,
    straight line through its log. None when the line fits badly (two modes
    beating inside the band) or the mode is too small to follow.
    """
    n = len(ring)
    spec = np.fft.fft(ring - ring.mean(0), axis=0)
    f = np.fft.fftfreq(n, 1 / rate)
    keep = filter_gain(np.abs(f), "band-pass", hz - DAMPING_BAND_HZ, hz + DAMPING_BAND_HZ, 6)
    keep = keep * (f > 0) * 2                    # one-sided: the analytic signal
    env = np.sqrt((np.abs(np.fft.ifft(spec * keep[:, None], axis=0)) ** 2).sum(1))
    edge = int(0.08 * rate)                      # the filter rings at both ends
    env, t = env[edge:n - edge], np.arange(edge, n - edge) / rate
    if len(env) < 50:
        return None
    use = env > 0.1 * env.max()
    use &= np.arange(len(env)) <= (np.flatnonzero(use)[-1] if use.any() else 0)
    if use.sum() < 50:
        return None
    slope, intercept = np.polyfit(t[use], np.log(env[use]), 1)
    resid = np.log(env[use]) - (slope * t[use] + intercept)
    if slope >= 0 or resid.std() > 0.35:
        return None
    return float(-slope / (2 * np.pi * hz))


# ---------------------------------------------------------------------------
# One hit, read as deck motions
# ---------------------------------------------------------------------------

def ring_window(forces: np.ndarray, rate: float) -> tuple[int, int, int]:
    """(hit row, first row, last row) of the ring-down after the hit."""
    k = find_strike(forces)
    a = k + int(RING_SKIP_S * rate)
    b = min(len(forces), a + int(RING_SECONDS * rate))
    if b - a < 0.3 * rate:
        raise ValueError(f"the hit at {k / rate:.2f} s leaves under 0.3 s of ring-down")
    return k, a, b


def describe_motion(shares: dict[str, float]) -> str:
    """'pitch (front against back)', or the two largest when none has 60%."""
    ranked = sorted(shares, key=shares.get, reverse=True)
    if shares[ranked[0]] >= 0.6:
        return MOTION_NAMES[ranked[0]]
    short = {"Fz total": "bounce"}
    return "mixed: " + " + ".join(f"{short.get(k, k)} {shares[k] * 100:.0f}%" for k in ranked[:2])


def modes_in(seg: np.ndarray, rate: float, layout: lc.Layout) -> list[dict]:
    """
    The resonances in a stretch of forces (samples, 12), each as a dict:
    hz, the amplitude (N) of Fz total / pitch / roll / twist and of the
    horizontal totals, the share of the vertical power in each pattern, each
    corner's Z amplitude and phase (degrees, from the largest), damping.
    """
    f, spec = spectrum(seg, rate)
    deck = np.sqrt((np.abs(spec) ** 2).sum(1))
    corners = [3 * c + 2 for c in corner_inputs(layout)]
    modes = []
    for i in pick_peaks(f, deck):
        z = spec[i, corners]
        amps = {name: complex(z @ np.array(w)) for name, w in PATTERNS.items()}
        power = {name: abs(a) ** 2 / 4 for name, a in amps.items()}     # /4: unit-length patterns
        total = sum(power.values()) or 1.0
        shares = {name: p / total for name, p in power.items()}
        ref = int(np.argmax(np.abs(z)))
        modes.append({
            "hz": float(f[i]), "deck_n": float(deck[i]),
            "amp_n": {name: abs(a) for name, a in amps.items()},
            "fx_n": float(abs(spec[i, 0::3].sum())), "fy_n": float(abs(spec[i, 1::3].sum())),
            "vertical_share": float((np.abs(z) ** 2).sum() / deck[i] ** 2),
            "shares": shares, "motion": describe_motion(shares),
            "corner_z_n": np.abs(z).tolist(),
            "corner_phase_deg": np.degrees(np.angle(z / z[ref])).tolist(),
            "damping": damping_ratio(seg, rate, float(f[i])),
        })
    return modes


def band_fractions(seg: np.ndarray, rate: float, signal: Signal) -> list[float]:
    """Fraction of the signal's power (mean removed) in each of BANDS; the rest is above them."""
    f, spec = spectrum(seg, rate)
    p = signal.amplitude(spec) ** 2
    total = p.sum() or 1.0
    return [float(p[(f >= lo) & (f < hi)].sum() / total) for lo, hi in BANDS]


def mode_table(modes: list[dict]) -> list[str]:
    """
    Fixed-width lines, for the window and the report. The last four columns
    are each corner's Z: amplitude in newtons @ phase in degrees from the
    largest corner (0 moves with it, 180 against it).
    """
    corners = "".join(f"{name:>13}" for name in CORNERS)
    lines = [f"   Hz   Fz total  pitch   roll  twist   Fx    Fy   damping   {'motion':31}{corners}",
             "           N        N      N      N      N     N      %      " + " " * 31
             + "      N @ deg" * 4]
    for m in modes:
        a = m["amp_n"]
        damp = f"{m['damping'] * 100:5.1f}" if m["damping"] is not None else "    –"
        phases = "".join(f"{n:7.0f} @{p:+5.0f}" for n, p in zip(m["corner_z_n"], m["corner_phase_deg"]))
        lines.append(f"{m['hz']:6.1f} {a['Fz total']:8.0f} {a['pitch']:6.0f} {a['roll']:6.0f} "
                     f"{a['twist']:6.0f} {m['fx_n']:5.0f} {m['fy_n']:5.0f}   {damp}     "
                     f"{m['motion']:31}{phases}")
    return lines if modes else ["no peaks between %g and %g Hz" % (F_MIN, F_MAX)]


PHASE_COLORS = {"with": "#33aa55", "against": "#c0392b", "between": "#888888"}
PHASE_NOTE = ("Small decks: seen from above, front at the top. A disc at each corner: its area "
              "(not its radius) follows that corner's Z amplitude, scaled to the largest corner "
              "of that resonance, with a small minimum so every corner shows. Green + moves with "
              "the largest corner, red − against it, grey (with its degrees) is in between: two "
              "modes overlapping.")


def draw_mode_shapes(axes, modes: list[dict]):
    """
    One small deck per mode, seen from above with the front at the top. At each
    corner a disc whose area is in proportion to that corner's Z amplitude, on
    top of a small minimum so a quiet corner still shows; the largest corner of
    each mode gets the full size, so sizes compare corners within a deck, not
    one deck with another. (Power goes as amplitude squared, so the areas
    understate differences in power.) Green with a + moves in step with the largest corner, red
    with a − moves against it; grey, labelled with its phase in degrees, is
    neither (within 60 degrees of a quarter cycle).
    """
    spots = [(-0.5, 0.8), (0.5, 0.8), (0.5, -0.8), (-0.5, -0.8)]       # CORNERS order
    for ax, m in zip(axes, list(modes) + [None] * len(axes)):
        ax.clear()
        ax.set_axis_off()
        if m is None:
            continue
        ax.set(xlim=(-1.5, 1.5), ylim=(-1.6, 1.6), aspect="equal")
        ax.plot([-0.5, 0.5, 0.5, -0.5, -0.5], [0.8, 0.8, -0.8, -0.8, 0.8], color="#999999",
                lw=0.8, zorder=1)
        ax.set_title(f"{m['hz']:.1f} Hz", fontsize=8, pad=1)
        largest = max(m["corner_z_n"]) or 1.0
        for (x, y), amp, deg in zip(spots, m["corner_z_n"], m["corner_phase_deg"]):
            kind = "with" if abs(deg) < 60 else "against" if abs(deg) > 120 else "between"
            ax.scatter([x], [y], s=12 + 200 * amp / largest, color=PHASE_COLORS[kind], zorder=2,
                       linewidths=0)
            mark = {"with": "+", "against": "−", "between": f"{deg:+.0f}°"}[kind]
            if kind == "between":
                ax.text(x + (0.3 if x > 0 else -0.3), y, mark, fontsize=5, va="center",
                        ha="left" if x > 0 else "right", color="#555555")
            elif amp > 0.25 * largest:
                ax.text(x, y, mark, fontsize=7, ha="center", va="center", color="#ffffff",
                        fontweight="bold", zorder=3)
    if modes:
        axes[0].text(-1.45, 0.8, "front", fontsize=6, ha="right", va="center", color="#666666",
                     clip_on=False)


# ---------------------------------------------------------------------------
# A session
# ---------------------------------------------------------------------------

def session_trials(folder: Path) -> list[dict]:
    """The session's captures (session.json; without one, every trial_*.csv), tares included."""
    folder = Path(folder)
    meta = folder / "session.json"
    if meta.exists():
        trials = json.loads(meta.read_text(encoding="utf-8")).get("trials", [])
        return [t for t in trials if t.get("kind") != "loopback" and (folder / t["file"]).exists()]
    return [{"id": i, "kind": "load", "file": p.name, "notes": ""}
            for i, p in enumerate(sorted(folder.glob("trial_*.csv")), start=1)]


def load_forces(path: Path, layout: lc.Layout) -> tuple[np.ndarray, np.ndarray, float]:
    """A capture as (time, forces in newtons with each channel's median removed, rate)."""
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if data.shape[1] != 13 or len(data) < 16:
        raise ValueError(f"expected time + 12 channels, found {data.shape[1]} columns "
                         f"and {len(data)} rows")
    forces = lc.cell_forces(data[:, 1:], None, layout)
    return data[:, 0], forces - np.median(forces, axis=0), 1 / float(np.median(np.diff(data[:, 0])))


def captures_with_a_hit(folder: Path, layout: lc.Layout) -> list[dict]:
    """The session's captures that hold a strike; tares, standing loads and the like are left out."""
    hits = []
    for entry in session_trials(folder):
        try:
            _, forces, _ = load_forces(Path(folder) / entry["file"], layout)
            find_strike(forces)
        except (OSError, ValueError):
            continue
        hits.append(entry)
    return hits


def resting_load_lines(static: dict[str, float] | None) -> list[str]:
    """The resting load on each corner in words, for the window and the report."""
    if not static:
        return []
    total = sum(static.values())
    return ["   ".join(f"{name} {static[name]:.0f} N" for name in CORNERS),
            f"total {total:.0f} N ({total / fp.GRAVITY:.0f} kg);  front pair "
            f"{(static[CORNERS[0]] + static[CORNERS[1]]) / total * 100:.0f}%;  front left + back right "
            f"{(static[CORNERS[0]] + static[CORNERS[2]]) / total * 100:.0f}% (50% when the two "
            "diagonals share evenly)"]


def resting_load_short(static: dict[str, float] | None) -> str:
    """The Z loads in two short lines, front row over back row, for the window."""
    if not static:
        return "not known for this session"
    fl, fr, br, bl = (static[name] for name in CORNERS)
    total = fl + fr + br + bl
    return (f"FL {fl:5.0f}   FR {fr:5.0f} N    total {total / fp.GRAVITY:.0f} kg\n"
            f"BL {bl:5.0f}   BR {br:5.0f} N    front {(fl + fr) / total * 100:.0f}%, "
            f"FL+BR {(fl + br) / total * 100:.0f}%")


def default_report_name(folder: Path) -> str:
    return f"{Path(folder).name}_strike_report.pdf"


def static_channel_loads(folder: Path, layout: lc.Layout) -> np.ndarray | None:
    """
    (4, 3) newtons resting on each of CORNERS in X, Y and Z: the session's
    resting volts minus each cell's unloaded output from its calibration sheet.
    Only as good as those sheet offsets still are once the cells are bolted in.
    X and Y are in each cell's own axes.
    """
    rest = lc.rest_volts_for_session(folder)
    if rest is None or not all(lc.serial_key(layout.serials.get(c, "")) in lc.SHEETS for c in lc.CELLS):
        return None
    unloaded = np.concatenate([lc.expected_unbalance_v(layout.serials[c]) for c in lc.CELLS])
    forces = lc.cell_forces(rest, unloaded, layout).reshape(4, 3)
    return forces[corner_inputs(layout)]


def static_corner_loads(folder: Path, layout: lc.Layout) -> dict[str, float] | None:
    """The Z column of static_channel_loads: newtons resting on each corner."""
    loads = static_channel_loads(folder, layout)
    return None if loads is None else dict(zip(CORNERS, loads[:, 2].tolist()))


def resting_channel_lines(loads: np.ndarray | None) -> list[str]:
    """
    The resting load in every channel, laid out as the deck is seen from
    above: front row over back row, left beside right.
    """
    if loads is None:
        return []
    fl, fr, br, bl = loads                      # CORNERS order
    cell = lambda v: "".join(f"{x:7.0f}" for x in v)        # noqa: E731
    head = f"{'X':>7}{'Y':>7}{'Z':>7}"
    return [f"  {'':6}{'left':^21}    {'right':^21}",
            f"  {'':6}{head}    {head}",
            f"  {'front':6}{cell(fl)}    {cell(fr)}",
            f"  {'back':6}{cell(bl)}    {cell(br)}",
            f"  {'total':6}{cell(loads.sum(0))}"]


def analyse_session(folder: Path, trial_ids: list[int] | None = None,
                    default: lc.Layout | None = None) -> dict:
    """
    Every hit in the session (or in the trials named): its window, modes and
    band fractions. Captures without a strike (tares, standing loads) are
    listed in "skipped" and take no further part.
    """
    folder = Path(folder)
    layout, note = lc.calibration_for_session(folder, default or lc.load_layout())
    fz = standard_signals(layout)["Whole treadmill"][0]
    hits, skipped = [], []
    for entry in session_trials(folder):
        if trial_ids and entry["id"] not in trial_ids:
            continue
        try:
            t, forces, rate = load_forces(folder / entry["file"], layout)
            k, a, b = ring_window(forces, rate)
        except (OSError, ValueError) as e:
            skipped.append((entry, str(e)))
            continue
        hits.append({"entry": entry, "t": t, "forces": forces, "rate": rate, "hit": k, "ring": (a, b),
                     "modes": modes_in(forces[a:b], rate, layout),
                     "bands": band_fractions(forces[a:b], rate, fz)})
    return {"folder": folder, "layout": layout, "note": note, "hits": hits, "skipped": skipped,
            "static": static_corner_loads(folder, layout),
            "static_xyz": static_channel_loads(folder, layout)}


def modes_across(hits: list[dict], tolerance_hz: float = 1.2) -> list[dict]:
    """
    Group the hits' modes by frequency: each group's median Hz, how many hits
    show it, median damping and pattern shares, and the largest Fz total.
    """
    every = sorted((m for h in hits for m in h["modes"]), key=lambda m: m["hz"])
    groups: list[list[dict]] = []
    for m in every:
        if groups and m["hz"] - np.median([g["hz"] for g in groups[-1]]) <= tolerance_hz:
            groups[-1].append(m)
        else:
            groups.append([m])
    out = []
    for g in groups:
        damp = [m["damping"] for m in g if m["damping"] is not None]
        shares = {k: float(np.median([m["shares"][k] for m in g])) for k in PATTERNS}
        out.append({"hz": float(np.median([m["hz"] for m in g])), "hits": len(g),
                    "damping": float(np.median(damp)) if damp else None, "shares": shares,
                    "motion": describe_motion(shares),
                    "fz_n": max(m["amp_n"]["Fz total"] for m in g),
                    "deck_n": max(m["deck_n"] for m in g)})
    return out


# ---------------------------------------------------------------------------
# The strike report (PDF)
# ---------------------------------------------------------------------------

def summary_lines(result: dict) -> list[str]:
    hits = result["hits"]
    lines = [f"Strike report: {result['folder'].name}", ""]
    if result["note"]:
        lines += [f"Note: {result['note']}", ""]
    if result["static"]:
        lines += ["Resting load in each channel, newtons (resting volts minus the sheets' unloaded "
                  "output):",
                  *resting_channel_lines(result["static_xyz"]),
                  "  Z: " + resting_load_lines(result["static"])[1],
                  "  X and Y are in each cell's own axes; their directions against the deck are "
                  "not yet checked.", ""]
    lines += [f"Hits analysed: {len(hits)}"]
    for h in hits:
        e = h["entry"]
        lines.append(f"  trial {e['id']:>2}  hit at {h['hit'] / h['rate']:.2f} s   {e.get('notes', '').strip()}")
    if result["skipped"]:
        lines.append(f"Left out, no hit in them: trials "
                     + ", ".join(str(e["id"]) for e, _ in result["skipped"]))
    lines += ["", "Resonances found in more than one hit (medians):",
              "    Hz   hits  damping   bounce pitch  roll twist   largest Fz total   motion",
              "                   %        %     %     %     %          N"]
    for g in modes_across(hits):
        if g["hits"] < min(2, len(hits)):
            continue
        s = g["shares"]
        damp = f"{g['damping'] * 100:5.1f}" if g["damping"] is not None else "    –"
        lines.append(f"  {g['hz']:5.1f}  {g['hits']:3d}    {damp}     {s['Fz total'] * 100:4.0f}  "
                     f"{s['pitch'] * 100:4.0f}  {s['roll'] * 100:4.0f}  {s['twist'] * 100:4.0f}"
                     f"        {g['fz_n']:6.0f}          {g['motion']}")
    lines += ["", "Share of Fz total's ring-down power in each band (Hz):",
              "  trial  " + "  ".join(f"{lo:>3g}-{hi:<3g}" for lo, hi in BANDS)]
    for h in hits:
        lines.append(f"  {h['entry']['id']:>5}  " + "  ".join(f"{x * 100:6.0f}%" for x in h["bands"]))
    lines += ["", "What is plotted:", *("  " + line for line in pattern_formulas()), "",
              "bounce = all four corners together (Fz total), pitch = front against back,",
              "roll = left against right, twist = one diagonal against the other.",
              "Damping is the fraction of critical; – means two modes beat inside the band.",
              "",
              "Phase. At each resonance every corner's Z has an amplitude and a phase (where in",
              "its cycle it is). Nothing is shifted or searched for: pitch is simply front pair",
              "minus back pair with those phases kept, so corners in step add and corners",
              "opposed cancel. Each hit's page shows them: the table gives every corner as",
              "newtons @ degrees from the largest corner (0 = with it, 180 = against it, near",
              "90 = two modes overlapping), and the small decks (seen from above, front at the",
              "top) draw the same as a disc per corner: green + = moves with the largest corner,",
              "red − = against it, grey = in between. A disc's area (not its radius) follows the",
              "corner's amplitude, scaled to the largest corner of that resonance, with a small",
              "minimum size; so compare discs within one deck, not between decks.",
              "Fz total is what a gait analysis reads; pitch and roll move the centre of pressure."]
    return lines


def draw_hit(figure, hit: dict, layout: lc.Layout, title: str):
    """One hit on a figure: motions against time, a spectrogram, their spectra, the mode table."""
    groups = standard_signals(layout)
    signals = groups["Whole treadmill"][:1] + groups["Rocking and twisting (Z)"]
    t, forces, rate = hit["t"], hit["forces"], hit["rate"]
    a, b = hit["ring"]
    grid = figure.add_gridspec(5, 1, height_ratios=[1.1, 1.0, 1.3, 0.8, 0.9])
    ax_t = figure.add_subplot(grid[0])
    ax_s = figure.add_subplot(grid[1], sharex=ax_t)        # same time axis as the traces
    ax_f = figure.add_subplot(grid[2])
    shapes = grid[3].subgridspec(1, N_PEAKS)
    draw_mode_shapes([figure.add_subplot(shapes[i]) for i in range(N_PEAKS)], hit["modes"])
    ax_txt = figure.add_subplot(grid[4])
    shown = slice(max(0, hit["hit"] - int(0.3 * rate)), min(len(t), b + int(0.3 * rate)))
    for s in signals:
        ax_t.plot(t[shown], s.series(forces)[shown], color=s.color, ls=s.ls, lw=0.7, label=s.name)
    ax_t.axvspan(t[a], t[b - 1], color=SHADE_COLOR, lw=0, zorder=0)
    ax_t.set(ylabel="N", title=title)
    ax_t.tick_params(labelbottom=False)
    ax_t.legend(loc="upper right", ncols=4, fontsize=7, frameon=False)

    f, spec = spectrum(forces[a:b], rate)
    band = (f >= 2) & (f <= F_MAX)
    for s in signals:
        ax_f.semilogy(f[band], s.amplitude(spec)[band], color=s.color, ls=s.ls, lw=0.9, label=s.name)
    quiet_b = hit["hit"] - int(QUIET_GAP_S * rate)
    if quiet_b > 0.25 * rate:
        _, before = spectrum(forces[max(0, quiet_b - (b - a)):quiet_b], rate)
        ax_f.semilogy(f[band], signals[0].amplitude(before)[band], color="#aaaaaa", lw=0.7,
                      label="Fz total before the hit")
    for m in hit["modes"]:
        ax_f.axvline(m["hz"], color=MARK_COLOR, lw=0.6, zorder=0)
        ax_f.annotate(f"{m['hz']:.1f}", (m["hz"], 1.0), xycoords=("data", "axes fraction"),
                      fontsize=6, ha="center", va="bottom")
    ax_f.set(xlabel="Hz", ylabel="N", xlim=(2, F_MAX))
    ax_f.legend(loc="upper right", ncols=3, fontsize=7, frameon=False)
    ax_f.grid(True, which="both", color=GRID_COLOR, lw=0.5)
    ax_f.set_axisbelow(True)

    ts, fs, amp = spectrogram(signals[0].series(forces)[shown], rate)
    keep = fs <= F_MAX
    db = 20 * np.log10(np.maximum(amp[keep], 1e-9))
    ax_s.pcolormesh(ts + t[shown][0], fs[keep], db, vmin=db.max() - 50, vmax=db.max(),
                    shading="auto", cmap="magma", rasterized=True)
    ax_s.set(xlabel="time (s)", ylabel="Hz, Fz total", xlim=(t[shown][0], t[shown][-1]))

    ax_txt.axis("off")
    ax_txt.text(0, 1, "\n".join(mode_table(hit["modes"])), family="monospace", fontsize=5.5,
                va="top", transform=ax_txt.transAxes)
    # What is plotted, on every page
    formulas = [" ".join(line.split()) for line in pattern_formulas()[:4]]
    figure.text(0.06, 0.012, "      ".join(formulas) + "\nEach term is the Z (vertical) force at "
                "that corner: F front, B back, L left, R right.", family="monospace", fontsize=6,
                va="bottom")

def write_report(result: dict, out: Path) -> Path:
    """The summary page and one page per hit, as one PDF."""
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        page = Figure(figsize=(8.5, 11))
        page.text(0.06, 0.96, "\n".join(summary_lines(result)), family="monospace", fontsize=7.5,
                  va="top")
        pdf.savefig(page)
        for h in result["hits"]:
            e = h["entry"]
            page = Figure(figsize=(8.5, 11), layout="constrained")
            draw_hit(page, h, result["layout"],
                     f"{result['folder'].name}  trial {e['id']}  {e.get('notes', '').strip()}")
            pdf.savefig(page)
    return out


def main():
    parser = argparse.ArgumentParser(description="Strike report: resonances as deck motions.")
    parser.add_argument("session", type=Path, help="session folder (with session.json)")
    parser.add_argument("trials", type=int, nargs="*", help="trial ids (default: every hit)")
    parser.add_argument("--out", type=Path,
                        help="PDF to write (default: <session>/strike_analysis/<session>_strike_report.pdf)")
    args = parser.parse_args()
    result = analyse_session(args.session, args.trials or None)
    print("\n".join(summary_lines(result)))
    for h in result["hits"]:
        print(f"\ntrial {h['entry']['id']}  {h['entry'].get('notes', '').strip()}")
        print("\n".join(mode_table(h["modes"])))
    if not result["hits"]:
        raise SystemExit("No hits found in this session.")
    out = args.out or args.session / "strike_analysis" / default_report_name(args.session)
    print(f"\nWrote {write_report(result, out)}")


if __name__ == "__main__":
    main()
