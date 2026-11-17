#!/usr/bin/env python3
"""
Strike analysis window: a playground for the frequencies in a capture.

Opens from the calibration GUI (File > Strike analysis) or by itself:

  uv run modes_gui.py                        # opens the session it was closed on
  uv run modes_gui.py "<session folder>"

Pick a capture on the left. Tick what to plot:

  Whole treadmill   Fz total (what a gait analysis reads), Fx total, Fy total
  Rocking, twisting pitch (front pair minus back pair), roll (left minus right),
                    twist (one diagonal minus the other), all from the four Z
  Pairs             the front, back, left or right two Z added
  Each cell         X, Y, Z and |F| of each
  Your own pair     any two cells' X, Y or Z, added or subtracted

The plots are the signals against time, under it on the same time axis a
sliding-window spectrum (spectrogram) of the first signal ticked, and then
their spectrum over the shaded time window. Drag across either of the top two
plots to move the window; "ring-down after the hit" puts it back.

A filter, if chosen, is applied before everything else and shifts nothing in
time (so a narrow band rings before the hit as well as after it): low-pass
keeps below "to", high-pass keeps above "from", band-pass keeps between them
and notch removes between them.

The table under the plots lists the resonances in the window: how much of each
is bounce, pitch, roll and twist, its damping, and each corner's Z as newtons
@ phase. The row of small decks (seen from above) draws them as a disc per
corner: its area follows the amplitude (scaled to the largest corner of that
resonance), green + moves with the largest corner, red − against it.

Only captures with a strike in them are listed (untick "only hits" to see the
rest), and the resting load on each corner is shown under the list. The
controls scroll if the screen is too short for them.

"Save figure" writes the plots as EPS, PDF or SVG (vector) or PNG. "Strike
report" analyses every hit in the session and writes a PDF.

The arithmetic is in deck_modes.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import tk_env  # noqa: F401  (must come before tkinter)
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

import deck_modes as dm
import loadcell_cal as lc

MONO = "TkFixedFont"
PLOT_DPI = 90
# The session that was open when the window was last closed, so it opens there
# next time. It sits beside this file and is not committed (.gitignore): it
# holds a folder on this computer.
STATE_FILE = Path(__file__).with_name(".modes_gui_state.json")
MIN_WINDOW = 64           # samples: a shorter drag is ignored
SPECTROGRAM_DB = 50       # the spectrogram shows this far below its loudest point
DEFAULT_CHECKED = ["Fz total", "pitch", "roll", "twist", "Y: front − back", "X: left − right", "yaw"]
FIGURE_TYPES = [("EPS (vector)", "*.eps"), ("PDF (vector)", "*.pdf"), ("SVG (vector)", "*.svg"),
                ("PNG (image)", "*.png")]


def last_session() -> Path | None:
    """The session folder in STATE_FILE, if it is still there."""
    try:
        folder = Path(json.loads(STATE_FILE.read_text(encoding="utf-8"))["session"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return folder if folder.is_dir() else None


def remember_session(folder: Path | None):
    """Write the open session to STATE_FILE; a failure to write is not worth a message."""
    if folder is None:
        return
    try:
        STATE_FILE.write_text(json.dumps({"session": str(folder)}, indent=2) + "\n",
                              encoding="utf-8")
    except OSError:
        pass


def open_path(path: Path):
    """Open a file or folder with whatever the computer uses for it."""
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class ModesWindow(tk.Toplevel):
    def __init__(self, master, layout: lc.Layout, folder: Path | None = None, on_close=None):
        super().__init__(master)
        self.title("Strike analysis")
        self.geometry(f"1320x{min(980, self.winfo_screenheight() - 80)}")
        self.default_layout = layout
        self.layout = layout
        self._on_close_cb = on_close
        self.protocol("WM_DELETE_WINDOW", self.close)

        self.folder: Path | None = None
        self.trials: list[dict] = []            # the captures listed
        self.every_trial: list[dict] = []
        self.hit_trials: list[dict] = []
        self.note: str | None = None
        self.entry: dict = {}
        self.t: np.ndarray | None = None
        self.forces: np.ndarray | None = None
        self.rate = 0.0
        self.hit: int | None = None
        self.window = (0, 0)                    # rows: the analysis window
        self.modes: list[dict] = []
        self.custom: list[dm.Signal] = []
        self.vars: dict[str, tk.BooleanVar] = {}
        self._selectors: list[SpanSelector] = []

        # Text a little smaller than the system's (never under 10 pt), titles larger and bold
        self.body_font = tkfont.nametofont("TkDefaultFont").copy()
        self.body_font.configure(size=max(10, abs(self.body_font.cget("size")) - 2))
        self.title_font = self.body_font.copy()
        self.title_font.configure(size=self.body_font.cget("size") + 2, weight="bold")
        self.bold_font = self.body_font.copy()
        self.bold_font.configure(weight="bold")
        self.mono_font = tkfont.nametofont(MONO).copy()
        self.mono_font.configure(size=self.body_font.cget("size"))
        style = ttk.Style(self)
        for name in ("TLabel", "TCheckbutton", "TButton"):
            style.configure(f"Modes.{name}", font=self.body_font)

        # The controls scroll when the screen is too short for them all
        side = ttk.Frame(self)
        side.pack(side="left", fill="y", padx=(8, 0), pady=8)
        self.left_canvas = tk.Canvas(side, highlightthickness=0, borderwidth=0,
                                     background=style.lookup("TFrame", "background"))
        self.left_scroll = ttk.Scrollbar(side, command=self.left_canvas.yview)
        self.left_canvas.config(yscrollcommand=self.left_scroll.set)
        self.left_canvas.pack(side="left", fill="y")
        self.left = ttk.Frame(self.left_canvas)
        self.left_canvas.create_window((0, 0), window=self.left, anchor="nw")
        self.left.bind("<Configure>", lambda _e: self._fit_left())
        self.left_canvas.bind("<Configure>", lambda _e: self._fit_left())
        self._build_session()
        self._build_signals()
        self._build_filter()
        self._build_window()
        self._build_output()

        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=8)
        table = self._box(right, "Resonances in the window. Corners: newtons @ degrees "
                                           "from the largest corner (0 with it, 180 against it)")
        table.pack(side="bottom", fill="x", pady=(6, 0))
        self.table = tk.Text(table, height=10, font=self.mono_font, wrap="none", state="disabled",
                             relief="flat")
        self.table.pack(fill="x", padx=4, pady=2)
        self._label(table, dm.PHASE_NOTE, grey=True, wraplength=880, justify="left"
                    ).pack(anchor="w", padx=4, pady=(0, 2))
        self.figure = Figure(figsize=(9, 7), dpi=PLOT_DPI, layout="constrained")
        # Time trace and spectrogram on top, sharing the time axis; the spectrum under them
        grid = self.figure.add_gridspec(4, 1, height_ratios=[1, 1, 1.25, 0.8])
        self.ax_t = self.figure.add_subplot(grid[0])
        self.ax_s = self.figure.add_subplot(grid[1], sharex=self.ax_t)
        self.ax_f = self.figure.add_subplot(grid[2])
        # One small deck per resonance: a disc at each corner for its Z amplitude and phase
        shapes = grid[3].subgridspec(1, dm.N_PEAKS)
        self.shape_axes = [self.figure.add_subplot(shapes[i]) for i in range(dm.N_PEAKS)]
        for ax in self.shape_axes:
            ax.set_axis_off()
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.ax_t.set_title("Open a session and pick a capture", fontsize=10)
        self.canvas.draw()

        # The session asked for, or else the one open when the window was last closed
        folder = folder if folder is not None else last_session()
        if folder is not None:
            self.open_session(folder)

    def _fit_left(self):
        """Size the controls' canvas to them; show its scroll bar only when they don't fit."""
        need = self.left.winfo_reqheight()
        self.left_canvas.config(width=self.left.winfo_reqwidth(), scrollregion=(0, 0, 0, need))
        if need > self.left_canvas.winfo_height() > 1:
            self.left_scroll.pack(side="left", fill="y")
        else:
            self.left_scroll.pack_forget()

    # -- small widget helpers: this window's fonts on everything -------------
    def _box(self, parent, title: str) -> ttk.LabelFrame:
        """A titled section."""
        box = ttk.LabelFrame(parent, labelwidget=ttk.Label(parent, text=title, font=self.title_font))
        box.pack(fill="x", pady=(0, 3))
        return box

    def _label(self, parent, text: str = "", grey: bool = False, **kw) -> ttk.Label:
        if grey:
            kw["foreground"] = "#666"
        return ttk.Label(parent, text=text, style="Modes.TLabel", **kw)

    def _button(self, parent, text: str, command, **kw) -> ttk.Button:
        return ttk.Button(parent, text=text, command=command, style="Modes.TButton", **kw)

    def _tick(self, parent, text: str, variable, command) -> ttk.Checkbutton:
        return ttk.Checkbutton(parent, text=text, variable=variable, command=command,
                               style="Modes.TCheckbutton")

    def _row(self, parent, **pack) -> ttk.Frame:
        row = ttk.Frame(parent)
        row.pack(anchor="w", padx=4, **pack)
        return row

    def _entry(self, parent, variable, width: int = 5) -> ttk.Entry:
        """A number box that redraws on Return or when left."""
        entry = ttk.Entry(parent, textvariable=variable, width=width, font=self.body_font)
        entry.bind("<Return>", lambda _e: self._redraw())
        entry.bind("<FocusOut>", lambda _e: self._redraw())
        return entry

    # -- left panel -------------------------------------------------------------
    def _build_session(self):
        box = self._box(self.left, "Session")
        row = self._row(box, pady=(2, 0))
        self._button(row, "Open...", self._ask_session, width=7).pack(side="left")
        self.session_label = self._label(row, "none open", grey=True)
        self.session_label.pack(side="left", padx=6)
        self.only_hits = tk.BooleanVar(value=True)
        self._tick(row, "only hits", self.only_hits, self._list_trials).pack(side="left", padx=6)
        frame = ttk.Frame(box)
        frame.pack(fill="x", padx=4, pady=3)
        # Usually four hits: room for five, the rest by scrolling
        self.trial_list = tk.Listbox(frame, height=5, width=46, exportselection=False,
                                     font=self.body_font)
        scroll = ttk.Scrollbar(frame, command=self.trial_list.yview)
        self.trial_list.config(yscrollcommand=scroll.set)
        self.trial_list.pack(side="left", fill="x", expand=True)
        scroll.pack(side="left", fill="y")
        self.trial_list.bind("<<ListboxSelect>>", lambda _e: self._on_trial_select())
        self.status = self._label(box, grey=True, wraplength=400, justify="left")
        self.status.pack(fill="x", padx=4, pady=(0, 2))
        rest = self._box(self.left, "Resting load on each corner")
        self.rest_label = self._label(rest, justify="left", font=self.mono_font)   # columns line up
        self.rest_label.pack(fill="x", padx=4, pady=1)

    def _build_signals(self):
        self.signal_box = self._box(self.left, "Show")
        self._fill_signals()

    def _fill_signals(self):
        """(Re)build the tick boxes: the layout decides which input is which corner."""
        checked = [k for k, v in self.vars.items() if v.get()] or DEFAULT_CHECKED
        for child in self.signal_box.winfo_children():
            child.destroy()
        self.signals: dict[str, dm.Signal] = {}
        self.vars = {}
        groups = dm.standard_signals(self.layout)

        def row_of(title: str, group: str):
            """A bold name, then that group's tick boxes on the same line."""
            row = self._row(self.signal_box, pady=(2, 0))
            self._label(row, title, font=self.bold_font).pack(side="left", padx=(0, 6))
            for s in groups[group]:
                self._add_check(row, s, s.name.removesuffix(" pair"), s.name in checked
                                ).pack(side="left", padx=(0, 4))

        row_of("Whole", "Whole treadmill")
        # Each Z motion beside the sideways signal that goes with it (dashed, same colour)
        pairs = self._row(self.signal_box, pady=(2, 0))
        self._label(pairs, "Rock, twist (Z)", font=self.bold_font).grid(row=0, column=0, sticky="w")
        self._label(pairs, "sideways partner (dashed)", font=self.bold_font
                    ).grid(row=0, column=1, sticky="w", padx=(24, 0))
        for r, (z, side) in enumerate(zip(groups["Rocking and twisting (Z)"],
                                          groups["Sideways partners (X, Y)"]), start=1):
            self._add_check(pairs, z, z.name, z.name in checked).grid(row=r, column=0, sticky="w")
            self._add_check(pairs, side, side.name, side.name in checked
                            ).grid(row=r, column=1, sticky="w", padx=(24, 0))
        row_of("Z pairs", "Pairs (Z)")

        cells = self._row(self.signal_box, pady=(2, 0))
        self._label(cells, "Each cell", font=self.bold_font).grid(row=0, column=0, sticky="w")
        for c, comp in enumerate(("X", "Y", "Z", "|F|")):
            tk.Label(cells, text=comp, fg=dm.AXIS_COLORS[comp],
                     font=self.bold_font).grid(row=0, column=c + 1)
        by_name = {s.name: s for s in groups["Each cell"]}
        for r, cell in enumerate(lc.CELLS, start=1):
            self._label(cells, f"{cell}  {self.layout.where[cell]}"
                        ).grid(row=r, column=0, sticky="w", padx=(0, 6))
            for c, comp in enumerate(("X", "Y", "Z", "|F|")):
                s = by_name[f"{cell}{comp}"]
                self._add_check(cells, s, "", s.name in checked).grid(row=r, column=c + 1)

        # Your own pair: cell, axis, plus or minus, cell
        row = self._row(self.signal_box, pady=(2, 0))
        self._label(row, "Own pair", font=self.bold_font).pack(side="left", padx=(0, 6))
        self.pair_a, self.pair_b = tk.StringVar(value="1"), tk.StringVar(value="2")
        self.pair_sign, self.pair_axis = tk.StringVar(value="−"), tk.StringVar(value="Z")
        for var, values in ((self.pair_a, lc.CELLS), (self.pair_axis, ["X", "Y", "Z"]),
                            (self.pair_sign, ["+", "−"]), (self.pair_b, lc.CELLS)):
            ttk.Combobox(row, textvariable=var, values=values, width=2, state="readonly",
                         font=self.body_font).pack(side="left", padx=1)
        self._button(row, "Add", self._add_pair, width=4).pack(side="left", padx=4)
        self.custom_row = self._row(self.signal_box, pady=(0, 2))
        for s in self.custom:
            self._add_check(self.custom_row, s, s.name, s.name in checked).pack(side="left")

    def _add_check(self, parent, signal: dm.Signal, text: str, on: bool) -> ttk.Checkbutton:
        self.signals[signal.name] = signal
        self.vars[signal.name] = tk.BooleanVar(value=on)
        return self._tick(parent, text, self.vars[signal.name], self._redraw)

    def _add_pair(self):
        a, b = int(self.pair_a.get()), int(self.pair_b.get())
        if a == b:
            self._say("Pick two different cells for a pair.")
            return
        s = dm.pair_signal(self.layout, a, b, self.pair_axis.get(),
                           +1 if self.pair_sign.get() == "+" else -1)
        if s.name in self.signals:
            self.vars[s.name].set(True)
        else:
            self.custom.append(s)
            self._add_check(self.custom_row, s, s.name, True).pack(side="left")
        self._redraw()

    def _build_filter(self):
        box = self._box(self.left, "Filter (no time shift)")
        row = self._row(box, pady=2)
        self.filter_kind = tk.StringVar(value="none")
        kind = ttk.Combobox(row, textvariable=self.filter_kind, values=dm.FILTERS, width=9,
                            state="readonly", font=self.body_font)
        kind.pack(side="left")
        kind.bind("<<ComboboxSelected>>", lambda _e: self._redraw())
        self.filter_lo, self.filter_hi = tk.StringVar(value="25"), tk.StringVar(value="35")
        for label, var in (("from", self.filter_lo), ("to", self.filter_hi)):
            self._label(row, label).pack(side="left", padx=(6, 2))
            self._entry(row, var).pack(side="left")
        self._label(row, "Hz").pack(side="left", padx=2)

    def _build_window(self):
        box = self._box(self.left, "Time window")
        row = self._row(box, pady=(2, 0))
        self.window_label = self._label(row)
        self.window_label.pack(side="left")
        self._label(row, "   drag on a top plot to change", grey=True).pack(side="left")
        row = self._row(box, pady=2)
        self._button(row, "ring-down after the hit", self._window_ring).pack(side="left")
        self._button(row, "whole capture", self._window_all).pack(side="left", padx=4)
        row = self._row(box, pady=(0, 2))
        self.log_scale = tk.BooleanVar(value=True)
        self._tick(row, "log scale", self.log_scale, self._redraw).pack(side="left")
        self._label(row, "   spectrum up to").pack(side="left")
        self.f_max = tk.StringVar(value="100")
        self._entry(row, self.f_max).pack(side="left", padx=2)
        self._label(row, "Hz").pack(side="left")

    def _build_output(self):
        row = self._row(self.left, pady=(2, 5))
        self._button(row, "Save figure...", self._save_figure).pack(side="left")
        self._button(row, "Strike report (PDF)...", self._report).pack(side="left", padx=6)
        box = self._box(self.left, "What is plotted")
        self.formula_label = self._label(box, font=self.mono_font, justify="left")
        self.formula_label.pack(anchor="w", padx=4, pady=2)
        self._show_formulas()

    def _show_formulas(self):
        """The sums behind the motions; the yaw's numbers depend on where the cells sit."""
        sideways = [line if len(line) < 48 else line.replace(") + ", ")\n    + ")
                    for line in dm.horizontal_formulas(self.layout)]      # keep the panel narrow
        self.formula_label.config(text="\n".join(
            [*dm.pattern_formulas()[:4], *sideways,
             "z, x, y: that force at the corner (F front, B back)"]))

    def _say(self, text: str):
        self.status.config(text=text)

    # -- session and capture ----------------------------------------------------
    def _ask_session(self):
        start = self.folder.parent if self.folder else Path.home()
        name = filedialog.askdirectory(parent=self, title="Open a session folder",
                                       initialdir=str(start))
        if name:
            self.open_session(Path(name))

    def open_session(self, folder: Path):
        folder = Path(folder)
        every = dm.session_trials(folder)
        if not every:
            self._say(f"No captures found in {folder.name}.")
            return
        self.folder, self.every_trial = folder, every
        self.layout, self.note = lc.calibration_for_session(folder, self.default_layout)
        self._fill_signals()
        self._show_formulas()
        self.session_label.config(text=folder.name)
        self._say("Reading the captures...")
        self.update_idletasks()
        self.hit_trials = dm.captures_with_a_hit(folder, self.layout)
        self.rest_label.config(text=dm.resting_load_short(dm.static_corner_loads(folder, self.layout)))
        self._list_trials()

    def _list_trials(self):
        """Fill the list: only the strikes, unless the box is unticked or there are none."""
        if self.folder is None:
            return
        self.trials = self.hit_trials if self.only_hits.get() and self.hit_trials else self.every_trial
        self.trial_list.delete(0, "end")
        for t in self.trials:
            self.trial_list.insert("end", f"#{t['id']:<3} {t.get('kind', ''):5} {t.get('notes', '')}")
        first = next((i for i, t in enumerate(self.trials) if t.get("kind") != "tare"), 0)
        self.trial_list.selection_set(first)
        self._on_trial_select()
        left_out = len(self.every_trial) - len(self.hit_trials)
        said = [self.status.cget("text")]
        if not self.hit_trials:
            said.append("No capture in this session has a hit: showing them all.")
        elif left_out and self.only_hits.get():
            said.append(f"{left_out} capture{' without a hit is' if left_out == 1 else 's without a hit are'}"
                        " not listed.")
        if self.note:
            said.append(f"Note: {self.note}")
        self._say("  ".join(said))

    def _on_trial_select(self):
        sel = self.trial_list.curselection()
        if not sel or self.folder is None:
            return
        entry = self.trials[sel[0]]
        try:
            self.t, self.forces, self.rate = dm.load_forces(self.folder / entry["file"], self.layout)
        except (OSError, ValueError) as e:
            self._say(f"Could not read {entry['file']}: {e}")
            return
        self.entry = entry
        self._window_ring(redraw=False)
        self._redraw(new_capture=True)

    # -- the time window --------------------------------------------------------
    def _window_ring(self, redraw: bool = True):
        if self.forces is None:
            return
        try:
            self.hit, a, b = dm.ring_window(self.forces, self.rate)
            self.window = (a, b)
            self._say(f"Hit at {self.hit / self.rate:.2f} s.")
        except ValueError as e:
            self.hit, self.window = None, (0, len(self.forces))
            self._say(f"{str(e).capitalize()}: using the whole capture.")
        if redraw:
            self._redraw()

    def _window_all(self):
        if self.forces is not None:
            self.window = (0, len(self.forces))
            self._redraw()

    def _on_span(self, t0: float, t1: float):
        if self.t is None:
            return
        a, b = np.searchsorted(self.t, [t0, t1])
        if b - a >= MIN_WINDOW:
            self.window = (int(a), int(b))
        self._redraw()

    # -- drawing ----------------------------------------------------------------
    def checked(self) -> list[dm.Signal]:
        return [self.signals[k] for k, v in self.vars.items() if v.get()]

    def _number(self, var: tk.StringVar, fallback: float) -> float:
        try:
            return float(var.get())
        except ValueError:
            return fallback

    def filtered(self) -> np.ndarray:
        """The capture's forces through the chosen filter (unfiltered if its numbers are bad)."""
        try:
            return dm.zero_phase_filter(self.forces, self.rate, self.filter_kind.get(),
                                        self._number(self.filter_lo, 0), self._number(self.filter_hi, 0))
        except ValueError as e:
            self._say(f"Filter not applied: {e}.")
            return self.forces

    def _redraw(self, new_capture: bool = False):
        if self.forces is None:
            return
        forces, t, (a, b) = self.filtered(), self.t, self.window
        signals = self.checked()
        f_max = max(5.0, min(self._number(self.f_max, 100.0), self.rate / 2))
        xlim = None if new_capture else self.ax_t.get_xlim()
        for ax in (self.ax_t, self.ax_f, self.ax_s):
            ax.clear()

        for s in signals:
            self.ax_t.plot(t, s.series(forces), color=s.color, ls=s.ls, lw=0.7, label=s.name)
        title = f"{self.folder.name}/{self.entry['file']}"
        if self.entry.get("notes"):
            title += f"   ({self.entry['notes']})"
        if self.filter_kind.get() != "none":
            title += f"   [{self.filter_kind.get()}]"
        self.ax_t.set_title(title, fontsize=10)
        self.ax_t.set_ylabel("N")
        self.ax_t.tick_params(labelbottom=False)
        if signals:
            self.ax_t.legend(loc="upper right", ncols=6, fontsize=7, frameon=False)

        f, spec = dm.spectrum(forces[a:b], self.rate)
        band = (f >= 1) & (f <= f_max)
        for s in signals:
            self.ax_f.plot(f[band], s.amplitude(spec)[band], color=s.color, ls=s.ls, lw=0.9)
        # What the first signal looked like before the hit: the noise floor
        quiet_b = (self.hit or 0) - int(dm.QUIET_GAP_S * self.rate)
        if signals and a >= quiet_b > 0.25 * self.rate:
            _, before = dm.spectrum(forces[max(0, quiet_b - (b - a)):quiet_b], self.rate)
            self.ax_f.plot(f[band], signals[0].amplitude(before)[band], color="#aaaaaa", lw=0.7,
                           label=f"{signals[0].name} before the hit")
            self.ax_f.legend(loc="upper right", fontsize=7, frameon=False)
        self.modes = dm.modes_in(forces[a:b], self.rate, self.layout)
        for m in self.modes:
            if m["hz"] <= f_max:
                self.ax_f.axvline(m["hz"], color=dm.MARK_COLOR, lw=0.6, zorder=0)
                self.ax_f.annotate(f"{m['hz']:.1f}", (m["hz"], 1.0), xycoords=("data", "axes fraction"),
                                   fontsize=7, ha="center", va="bottom")
        self.ax_f.set(xlabel="Hz", ylabel=f"N, {t[a]:.2f} to {t[b - 1]:.2f} s", xlim=(1, f_max))
        if self.log_scale.get() and any(s.amplitude(spec)[band].max() > 0 for s in signals):
            self.ax_f.set_yscale("log")
        self.ax_f.grid(True, which="both", color=dm.GRID_COLOR, lw=0.5)
        self.ax_f.set_axisbelow(True)

        if signals:
            ts, fs, amp = dm.spectrogram(signals[0].series(forces), self.rate)
            keep = fs <= f_max
            db = 20 * np.log10(np.maximum(amp[keep], 1e-9))
            self.ax_s.pcolormesh(ts + t[0], fs[keep], db, vmin=db.max() - SPECTROGRAM_DB,
                                 vmax=db.max(), shading="auto", cmap="magma", rasterized=True)
            self.ax_s.set_ylabel(f"Hz, {signals[0].name}")
        self.ax_s.set_xlabel("time (s)")
        self.ax_s.set_ylim(0, f_max)
        # Solid colours, no transparency: EPS cannot store it
        for ax, color in ((self.ax_t, "#888888"), (self.ax_s, "#ffffff")):
            ax.axvline(t[a], color=color, lw=0.8)
            ax.axvline(t[b - 1], color=color, lw=0.8)
        self.ax_t.axvspan(t[a], t[b - 1], color=dm.SHADE_COLOR, lw=0, zorder=0)
        self.ax_t.set_xlim(xlim if xlim else (t[0], t[-1]))

        # clear() dropped the drag selectors' artists: make new ones
        self._selectors = [SpanSelector(ax, self._on_span, "horizontal", useblit=False,
                                        props=dict(facecolor="#1f6fd1", alpha=0.2))
                           for ax in (self.ax_t, self.ax_s)]
        self.window_label.config(text=f"{t[a]:.2f} to {t[b - 1]:.2f} s ({(b - a) / self.rate:.2f} s)")
        dm.draw_mode_shapes(self.shape_axes, self.modes)
        self._show_table()
        self.canvas.draw_idle()

    def _show_table(self):
        self.table.config(state="normal")
        self.table.delete("1.0", "end")
        self.table.insert("end", "\n".join(dm.mode_table(self.modes)))
        self.table.config(state="disabled")

    # -- output -----------------------------------------------------------------
    def _save_figure(self):
        if self.forces is None:
            return
        name = filedialog.asksaveasfilename(
            parent=self, title="Save the plots", initialdir=str(self.folder),
            initialfile=f"{self.folder.name}_{Path(self.entry['file']).stem}_spectra.eps",
            defaultextension=".eps",
            filetypes=FIGURE_TYPES)
        if not name:
            return
        try:
            self.save_figure(Path(name))
        except (OSError, ValueError) as e:
            messagebox.showerror("Save figure", f"Could not save {name}:\n{e}", parent=self)

    def save_figure(self, path: Path):
        """The three plots as they are on screen; the type follows the file's ending."""
        self.figure.savefig(path, dpi=200)
        self._say(f"Saved {path.name}.")

    def _report(self):
        if self.folder is None:
            return
        out = self.folder / "strike_analysis"
        name = filedialog.asksaveasfilename(
            parent=self, title="Save the strike report", initialdir=str(out if out.is_dir() else self.folder),
            initialfile=dm.default_report_name(self.folder), defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")])
        if not name:
            return
        try:
            path = self.write_report(Path(name))
        except (OSError, ValueError) as e:
            messagebox.showerror("Strike report", f"Could not write the report:\n{e}", parent=self)
            return
        if path is not None:
            try:
                open_path(path)
            except OSError:
                pass

    def write_report(self, path: Path) -> Path | None:
        """Analyse every hit in the session and write the PDF; None if it has no hits."""
        self._say("Writing the strike report...")
        self.update_idletasks()
        result = dm.analyse_session(self.folder, default=self.default_layout)
        if not result["hits"]:
            self._say("No hits found in this session: no report written.")
            return None
        dm.write_report(result, path)
        self._say(f"Strike report of {len(result['hits'])} hits saved as {path.name}.")
        return path

    def close(self):
        remember_session(self.folder)
        self.destroy()
        if self._on_close_cb:
            self._on_close_cb()


def main():
    parser = argparse.ArgumentParser(description="Strike analysis window.")
    parser.add_argument("session", type=Path, nargs="?", help="session folder to open")
    args = parser.parse_args()
    root = tk.Tk()
    root.withdraw()
    ModesWindow(root, lc.load_layout(), args.session, on_close=root.destroy)
    root.mainloop()


if __name__ == "__main__":
    main()
