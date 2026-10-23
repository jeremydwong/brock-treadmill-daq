#!/usr/bin/env python3
"""
Force-plate calibration GUI.

Left: the deck from above, with each load cell's X/Y/Z force plotted beside
the corner it sits under, and the deck size below. Right: the few things set
for every capture, the Tare and Log buttons, and a Details section for the
rest (data folder, session list, fit, scale, loopback test).

Click on the deck where the load is applied, or type how far it is right of
the left edge and back from the front edge; pick its direction, enter the mass
or force, and press "Log 5 s of data". Each capture records all 12 load-cell
channels for 5 s at 1 kHz into a CSV, and its location, direction, load and
notes go into session.json and captures_log.csv (see forceplate.py for the
file layout and the fit).

Load-cell positions and newtons per volt come from forceplate_layout.json.
Forces are tared with the latest Tare; with one, the deck also shows the
centre of pressure worked out from the four Z forces.

Sessions are saved under the data folder shown in the window (by default
Documents/ForcePlate data); "Change..." picks another and it is remembered.

Workflow for one session:
  1. Deck unloaded -> "Tare (no load)".
  2. Put a known mass on (or pull with a known force) -> click or type its
     location, pick the direction -> "Log 5 s of data".
  3. Repeat 2 at many locations and in every direction; re-tare now and then.
     Each load is paired with the latest tare.
  4. "Fit" (or `uv run forceplate.py <session>`) solves for the matrix.

Loopback mode drives AOUT0 or AOUT1 with a fixed voltage or a sinusoid, so
you can wire an output into one input and check the whole chain without the
load cells. Unhook that channel's load-cell lead first: two outputs on one
input fight each other.

Usage:
  uv run forceplate_gui.py                                # new session in the data folder
  uv run forceplate_gui.py --session "<data folder>/2026..."   # keep adding to one
  uv run forceplate_gui.py --length 1.8 --width 0.7       # deck size in metres
                                                          # (default: forceplate_layout.json)

Coordinates are in the deck frame of forceplate.py: origin at the front-left
corner (top left of the drawing), +x right, +y back, +z down (below the deck).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import os
import queue
import shutil
import subprocess
import sys
import threading

import tk_env  # noqa: F401  (must come before tkinter)
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import forceplate as fp
from mcc1608g import USB1608G, AiMode, AiRange, DaqError

# Deck size and load-cell positions: forceplate_layout.json (or --length/--width,
# or set in the GUI).

AI_MODE = AiMode.SINGLE_ENDED
AI_RANGE = AiRange.BIP10V
# Where sessions go unless the person picks another folder (remembered in SETTINGS).
DEFAULT_DATA_DIR = Path.home() / "Documents" / "ForcePlate data"
SETTINGS = Path.home() / ".forceplate_gui.json"
MIN_FREE_MB = 50          # refuse to capture with less disk space than this (a capture is ~0.7 MB)

MONITOR_BLOCK = 100       # samples per live-view scan (0.1 s at 1 kHz)
MONITOR_WINDOW_S = 3.0    # seconds of history in the live view
LIVE_REDRAW_S = 0.25      # redraw the four live plots at most this often (each takes ~25 ms)
LIVE_POINTS = 600         # points per line in the live plots: about 2 per pixel
AO_RATE = 1000.0          # DAC updates per second for the loopback sine
AO_SINE_SECONDS = 30 * 60 # a sine runs this long, then the output holds

AXIS_COLORS = {"X": "#3366cc", "Y": "#dd8800", "Z": "#33aa55"}
MONO = "TkFixedFont"      # the platform's monospace font (Menlo, Consolas, ...)
DECK_PX = 520             # longest side of the deck drawing, on a tall enough screen
CELL_PLOT_PX = 290        # width of each load cell's plot beside the deck
PLOT_DPI = 90
COP_COLOR = "#1f6fd1"


# ---------------------------------------------------------------------------
# DAQ thread
# ---------------------------------------------------------------------------

class DaqWorker(threading.Thread):
    """
    Owns the USB device. The GUI puts commands on `commands`; results come back
    on `results` as (kind, payload) tuples. Every USB transfer happens on this
    thread, one at a time, so the GUI never blocks and transfers never interleave.
    """

    def __init__(self, daq: USB1608G):
        super().__init__(daemon=True, name="daq")
        self.daq = daq
        self.commands: queue.Queue = queue.Queue()
        self.results: queue.Queue = queue.Queue()
        self.monitoring = False
        self._ao_scan = None
        self._running = True

    def send(self, command: str, **kwargs):
        self.commands.put((command, kwargs))

    def run(self):
        while self._running:
            try:
                if self.monitoring:
                    command, kwargs = self.commands.get_nowait()
                else:
                    command, kwargs = self.commands.get(timeout=0.1)
            except queue.Empty:
                command, kwargs = None, {}
            try:
                if command is not None:
                    getattr(self, f"_do_{command}")(**kwargs)
                elif self.monitoring:
                    volts, rate = self._scan(MONITOR_BLOCK)
                    self.results.put(("monitor", volts))
            except Exception as e:
                self.monitoring = False
                self.results.put(("error", f"{command or 'live view'}: {e}"))

    def _scan(self, samples: int) -> tuple[np.ndarray, float]:
        ai = self.daq.a_in_scan(fp.AI_LOW, fp.AI_HIGH, AI_MODE, AI_RANGE,
                                samples, fp.SAMPLE_RATE)
        return ai.read_all(), ai.rate

    # -- commands -----------------------------------------------------------
    def _do_monitor(self, on: bool):
        self.monitoring = on

    def _do_capture(self, kind: str, fields: dict):
        samples = int(fp.CAPTURE_SECONDS * fp.SAMPLE_RATE)
        volts, rate = self._scan(samples)
        self.results.put(("capture", (volts, rate, kind, fields)))

    def _stop_ao(self):
        if self._ao_scan is not None:
            self._ao_scan.stop()
            self._ao_scan = None
        if self.daq.has_ao:
            self.daq.a_out(0, 0.0)
            self.daq.a_out(1, 0.0)

    def _do_ao_off(self, quiet: bool = False):
        self._stop_ao()
        if not quiet:
            self.results.put(("status", "Analog outputs off (0 V)."))

    def _do_ao_dc(self, channel: int, volts: float):
        self._stop_ao()
        self.daq.a_out(channel, volts)
        self.results.put(("status", f"AOUT{channel} holding {volts:+.3f} V."))

    def _do_ao_sine(self, channel: int, amplitude: float, offset: float, freq: float):
        self._stop_ao()
        t = np.arange(int(AO_SINE_SECONDS * AO_RATE)) / AO_RATE
        wave = offset + amplitude * np.sin(2 * np.pi * freq * t)
        self._ao_scan = self.daq.a_out_scan(channel, channel, wave, AO_RATE)
        self.results.put(("status", f"AOUT{channel} playing {amplitude:.3f} V sine at "
                                    f"{freq:g} Hz (offset {offset:+.3f} V)."))

    def _do_quit(self):
        try:
            self._stop_ao()
        finally:
            self.daq.close()
            self._running = False


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self, daq: USB1608G | None, daq_error: str = "",
                 session_dir: Path | None = None,
                 deck_length: float | None = None, deck_width: float | None = None,
                 layout: fp.Layout | None = None):
        super().__init__()
        self.title("Force-plate calibration")
        self.layout = layout or fp.load_layout()
        self.deck_length = deck_length or self.layout.deck_length_m   # metres, front to back (y)
        self.deck_width = deck_width or self.layout.deck_width_m      # metres, side to side (x)
        self.settings = load_settings()
        self.data_dir = Path(self.settings.get("data_dir", DEFAULT_DATA_DIR))
        self.session: fp.Session | None = fp.Session.load(session_dir) if session_dir else None
        # Mean volts of the latest tare: the zero for forces and centre of pressure.
        self.zero_v: np.ndarray | None = None
        if self.session:
            deck = self.session.meta["deck"]
            self.deck_length, self.deck_width = deck["length_m"], deck["width_m"]
            tare = self.session.last_tare_id()
            if tare is not None:
                self.zero_v = self._trial_mean_v(tare)
        self.cop: tuple[float, float] | None = None     # (x, y) metres, drawn on the deck

        self.daq = daq
        self.worker: DaqWorker | None = None
        if daq is not None:
            self.worker = DaqWorker(daq)
            self.worker.start()

        self.location: tuple[float, float] | None = None   # (x, y) metres, from the boxes
        self.busy = False
        n_window = int(MONITOR_WINDOW_S * fp.SAMPLE_RATE)
        self.live_buffer = np.full((n_window, 12), np.nan)
        self._last_live_draw = dt.datetime.min

        # Fit small laptop screens: the deck drawing sets the height of the window.
        self.deck_px = max(280, min(DECK_PX, self.winfo_screenheight() - 300))
        self._build_visuals()
        self._build_controls()
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._refresh_trials()
        self._redraw_deck()
        self._on_loopback_toggle(quiet=True)    # outputs to 0 V, keeping "Connected" in view

        if daq is None:
            self._set_status(f"No DAQ: {daq_error}  (capture disabled)")
        else:
            self._set_status(f"Connected to {daq}.")
        self._update_buttons()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(50, self._poll)
        self.update_idletasks()
        max_h = self.winfo_screenheight() - 80
        if self.winfo_reqheight() > max_h:
            self.geometry(f"{self.winfo_reqwidth()}x{max_h}+0+0")

    # -- layout: everything visual on the left -------------------------------
    def _build_visuals(self):
        frame = ttk.LabelFrame(self, text="Deck from above, front at the top: "
                                          "click where the load is applied")
        frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        grid = ttk.Frame(frame)
        grid.pack(padx=4, pady=4)

        # Room around the deck for the rulers (top, left) and labels.
        self.margin_left, self.margin_top, self.margin_right, self.margin_bottom = 80, 56, 24, 30
        self.canvas = tk.Canvas(grid, background="white", highlightthickness=0)
        self.canvas.grid(row=0, column=1, rowspan=2)
        self.canvas.bind("<Button-1>", self._on_deck_click)
        self._resize_canvas()
        self._build_cell_plots(grid, int(float(self.canvas.cget("height")) / 2))

        # Deck size, below the drawing
        dims = ttk.Frame(frame)
        dims.pack(padx=6, pady=(0, 6))
        ttk.Label(dims, text="Deck size:").pack(side="left")
        self.width_var = tk.StringVar()
        self.length_var = tk.StringVar()
        ttk.Label(dims, text="side to side").pack(side="left", padx=(6, 2))
        ttk.Entry(dims, textvariable=self.width_var, width=6).pack(side="left")
        ttk.Label(dims, text="front to back").pack(side="left", padx=(10, 2))
        ttk.Entry(dims, textvariable=self.length_var, width=6).pack(side="left")
        self.length_unit = tk.StringVar(value=self.settings.get("length_unit", "m"))
        if self.length_unit.get() not in fp.LENGTH_UNITS:
            self.length_unit.set("m")
        units = ttk.Combobox(dims, textvariable=self.length_unit, width=4, state="readonly",
                             values=list(fp.LENGTH_UNITS))
        units.pack(side="left", padx=6)
        units.bind("<<ComboboxSelected>>", lambda _e: self._on_units_changed())
        ttk.Button(dims, text="Set size", command=self._set_deck_size).pack(side="left", padx=4)
        self._show_deck_size()

    def _build_cell_plots(self, grid: ttk.Frame, height_px: int):
        """One small plot per load cell, beside the deck corner the cell sits under."""
        free = [(0, 0), (0, 2), (1, 2), (1, 0)]
        self.axes, self.lines, self.plot_canvases = [], [], []
        for i, cell in enumerate("1234"):
            x, y = self.layout.positions[i]
            want = (0 if y < self.deck_length / 2 else 1, 0 if x < self.deck_width / 2 else 2)
            slot = want if want in free else free[0]     # two cells in one corner: next free
            free.remove(slot)
            fig = Figure(figsize=(CELL_PLOT_PX / PLOT_DPI, height_px / PLOT_DPI), dpi=PLOT_DPI)
            ax = fig.add_subplot()
            ax.set_title(f"load cell {cell}: {self.layout.where[cell]}", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("s", fontsize=8)
            for axis in "XYZ":
                (line,) = ax.plot([], [], color=AXIS_COLORS[axis], linewidth=0.9,
                                  label=f"{cell}{axis}")
                self.lines.append(line)         # SIGNAL_NAMES order: 1X 1Y 1Z 2X ...
            ax.legend(loc="upper left", fontsize=7, ncol=3)
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=grid)
            canvas.get_tk_widget().grid(row=slot[0], column=slot[1], padx=4, pady=2)
            self.axes.append(ax)
            self.plot_canvases.append(canvas)

    def _resize_canvas(self):
        self.scale = self.deck_px / max(self.deck_length, self.deck_width)   # px per metre
        self.canvas.config(
            width=self.deck_width * self.scale + self.margin_left + self.margin_right,
            height=self.deck_length * self.scale + self.margin_top + self.margin_bottom)

    # -- data folder --------------------------------------------------------------
    def _save_settings(self, **changes):
        self.settings.update(changes)
        try:
            SETTINGS.write_text(json.dumps(self.settings, indent=2), encoding="utf-8")
        except OSError:
            pass    # only a convenience; the defaults still work

    def _choose_data_dir(self):
        folder = filedialog.askdirectory(title="Save sessions in...", mustexist=False,
                                         initialdir=str(self.data_dir))
        if not folder:
            return
        self.data_dir = Path(folder)
        self._save_settings(data_dir=str(self.data_dir))
        if self.session is not None and self.session.trials:
            self.session = None     # the next capture starts a session in the new folder
        self._refresh_trials()
        self._redraw_deck()

    def _open_data_folder(self):
        folder = self.session.folder if self.session else self.data_dir
        try:
            folder.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(folder)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except OSError as e:
            messagebox.showerror("Open folder", f"Could not open {folder}:\n{e}")

    # -- units: the boxes show self.length_unit; everything stored is metres --
    def _unit(self) -> float:
        """Metres per displayed length unit."""
        return fp.LENGTH_UNITS[self.length_unit.get()]

    def _show_len(self, metres: float) -> str:
        return f"{round(metres / self._unit(), 3):g}"

    def _show_deck_size(self):
        self.width_var.set(self._show_len(self.deck_width))
        self.length_var.set(self._show_len(self.deck_length))

    def _on_units_changed(self):
        self._show_deck_size()
        if self.location is not None:
            self.x_var.set(self._show_len(self.location[0]))
            self.y_var.set(self._show_len(self.location[1]))
        if self.z_var.get().strip():
            try:
                self.z_var.set(self._show_len(float(self.z_var.get()) * self._last_unit))
            except ValueError:
                pass
        self._last_unit = self._unit()
        for label in self.unit_labels:
            label.config(text=self.length_unit.get())
        self._save_settings(length_unit=self.length_unit.get())
        self._redraw_deck()

    def _set_deck_size(self):
        try:
            width = float(self.width_var.get()) * self._unit()
            length = float(self.length_var.get()) * self._unit()
        except ValueError:
            messagebox.showerror("Deck size", "Side to side and front to back must be numbers.")
            return
        if length <= 0 or width <= 0:
            messagebox.showerror("Deck size", "Side to side and front to back must be positive.")
            return
        if self.session is not None and self.session.trials:
            messagebox.showerror("Deck size", "This session already has captures at the old "
                                              "size. Start a New session to change it.")
            self._show_deck_size()
            return
        self.deck_length, self.deck_width = length, width
        if self.session is not None:
            self.session.meta["deck"].update(length_m=length, width_m=width)
            self.session.save()
        self.x_var.set("")
        self.y_var.set("")
        self._resize_canvas()
        self._redraw_deck()
        self._update_buttons()

    # -- layout: text and controls on the right ---------------------------------
    def _build_controls(self):
        side = ttk.Frame(self)
        side.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=8)

        # The five things set for every capture
        main = ttk.LabelFrame(side, text="Each capture")
        main.pack(fill="x", pady=(0, 6))
        form = ttk.Frame(main)
        form.pack(fill="x", padx=6, pady=4)

        ttk.Label(form, text="direction").grid(row=0, column=0, sticky="w")
        self.direction = tk.StringVar(value="down")
        dirs = ttk.Frame(form)
        dirs.grid(row=0, column=1, columnspan=3, sticky="w", pady=2)
        for i, name in enumerate(fp.DIRECTIONS):
            ttk.Radiobutton(dirs, text=name, value=name, variable=self.direction,
                            command=self._redraw_deck).grid(row=0, column=i, padx=(0, 4))

        ttk.Label(form, text="load").grid(row=1, column=0, sticky="w")
        self.load_amount = tk.StringVar(value="20.0")
        ttk.Entry(form, textvariable=self.load_amount, width=8).grid(row=1, column=1, pady=2)
        self.load_unit = tk.StringVar(value=self.settings.get("load_unit", "kg"))
        if self.load_unit.get() not in fp.FORCE_UNITS:
            self.load_unit.set("kg")
        w = ttk.Combobox(form, textvariable=self.load_unit, width=4, state="readonly",
                         values=list(fp.FORCE_UNITS))
        w.grid(row=1, column=2, padx=(2, 4), sticky="w")
        w.bind("<<ComboboxSelected>>",
               lambda _e: self._save_settings(load_unit=self.load_unit.get()))
        self.force_label = ttk.Label(form, text="")
        self.force_label.grid(row=1, column=3, sticky="w")
        self.load_amount.trace_add("write", lambda *_: self._update_force_label())
        self.load_unit.trace_add("write", lambda *_: self._update_force_label())

        # Where: click the drawing, or type it. Points off the deck (on the
        # frame) are fine; depth is optional, blank meaning the deck surface.
        self.x_var, self.y_var, self.z_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.unit_labels = []
        for r, (text, var, hint) in enumerate((
                ("right of left edge", self.x_var, ""),
                ("back from front edge", self.y_var, ""),
                ("below deck surface", self.z_var, "optional: blank = on the deck")), start=2):
            ttk.Label(form, text=text).grid(row=r, column=0, sticky="w", padx=(0, 6))
            ttk.Entry(form, textvariable=var, width=8).grid(row=r, column=1, pady=2)
            unit = ttk.Label(form, text=self.length_unit.get())
            unit.grid(row=r, column=2, padx=(2, 4), sticky="w")
            self.unit_labels.append(unit)
            if hint:
                ttk.Label(form, text=hint, foreground="#666").grid(row=r, column=3, sticky="w")
        self._last_unit = self._unit()

        ttk.Label(form, text="notes").grid(row=5, column=0, sticky="w")
        self.notes = tk.StringVar()
        ttk.Entry(form, textvariable=self.notes).grid(row=5, column=1, columnspan=3,
                                                      sticky="ew", pady=2)
        form.columnconfigure(3, weight=1)
        self.location_label = ttk.Label(main, text="", foreground="#444", wraplength=400)
        self.location_label.pack(anchor="w", padx=6)

        buttons = ttk.Frame(main)
        buttons.pack(fill="x", padx=6, pady=6)
        self.tare_button = ttk.Button(buttons, text="Tare (no load)",
                                      command=lambda: self._capture("tare"))
        self.tare_button.pack(side="left")
        self.capture_button = ttk.Button(buttons, text=self._log_text(),
                                         command=lambda: self._capture("load"))
        self.capture_button.pack(side="left", padx=6)
        self.live = tk.BooleanVar(value=False)
        self.live_check = ttk.Checkbutton(buttons, text="Live view", variable=self.live,
                                          command=self._on_live_toggle)
        self.live_check.pack(side="right")
        self.progress = ttk.Progressbar(main, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=6, pady=(0, 4))
        self.status = ttk.Label(main, text="", wraplength=400, justify="left")
        self.status.pack(fill="x", padx=6, pady=(0, 4))
        self.showing_label = ttk.Label(main, text="", foreground="#444", wraplength=400)
        self.showing_label.pack(anchor="w", padx=6)
        self.cop_label = ttk.Label(main, text="", foreground=COP_COLOR, wraplength=400)
        self.cop_label.pack(anchor="w", padx=6, pady=(0, 6))

        # Everything else, folded away
        self.details_open = bool(self.settings.get("details_open", False))
        self.details_button = ttk.Button(side, command=self._toggle_details)
        self.details_button.pack(anchor="w")
        self.details = ttk.Notebook(side)
        self._build_session_tab(self.details)
        self._build_more_tab(self.details)
        self._build_loopback_tab(self.details)
        self._show_details()

        self.x_var.trace_add("write", lambda *_: self._on_location_typed())
        self.y_var.trace_add("write", lambda *_: self._on_location_typed())
        self.z_var.trace_add("write", lambda *_: self._on_location_typed())
        self._update_force_label()

    @staticmethod
    def _log_text(loopback: bool = False) -> str:
        what = " (loopback)" if loopback else ""
        return f"Log {fp.CAPTURE_SECONDS:g} s of data{what}"

    def _toggle_details(self):
        self.details_open = not self.details_open
        self._save_settings(details_open=self.details_open)
        self._show_details()

    def _show_details(self):
        if self.details_open:
            self.details_button.config(text="▾ Details")
            self.details.pack(fill="both", expand=True, pady=(4, 0))
        else:
            self.details_button.config(text="▸ Details: data folder, session list, fit, "
                                            "scale, loopback")
            self.details.pack_forget()

    def _build_session_tab(self, tabs: ttk.Notebook):
        sess = ttk.Frame(tabs)
        tabs.add(sess, text="Session")
        data = ttk.Frame(sess)
        data.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Label(data, text="Data folder:").pack(side="left")
        ttk.Button(data, text="Open", command=self._open_data_folder).pack(side="right")
        ttk.Button(data, text="Change...", command=self._choose_data_dir).pack(side="right", padx=4)
        self.data_label = ttk.Label(sess, text="", wraplength=400, foreground="#444")
        self.data_label.pack(anchor="w", padx=6)
        self.session_label = ttk.Label(sess, text="", wraplength=400)
        self.session_label.pack(anchor="w", padx=6, pady=4)
        self.trial_list = tk.Listbox(sess, height=7, font=MONO)
        self.trial_list.pack(fill="both", expand=True, padx=6)
        self.trial_list.bind("<<ListboxSelect>>", self._on_trial_select)
        row = ttk.Frame(sess)
        row.pack(fill="x", padx=6, pady=4)
        ttk.Button(row, text="New session", command=self._new_session).pack(side="left")
        ttk.Button(row, text="Remove selected", command=self._remove_trial).pack(side="left", padx=6)
        ttk.Button(row, text="Fit", command=self._fit).pack(side="right")

    def _build_more_tab(self, tabs: ttk.Notebook):
        more = ttk.Frame(tabs)
        tabs.add(more, text="Scale")
        lay = self.layout
        lines = [f"Load-cell positions and force scale: {lay.path.name}"]
        for i, cell in enumerate("1234"):
            x, y = lay.positions[i]
            npv = lay.newtons_per_volt[3 * i:3 * i + 3]
            lines.append(f"  cell {cell} ({lay.where[cell]}) on amplifier input {cell}: "
                         f"{x:g} m right, {y:g} m back; "
                         + ", ".join(f"{a} {v:g}" for a, v in zip("XYZ", npv)) + " N/V")
        if lay.placeholder:
            lines.append("The N/V numbers are PLACEHOLDERS: copy each load cell's calibration "
                         "sheet into that file (instructions inside) to get real newtons.")
        ttk.Label(more, text="\n".join(lines), wraplength=400, justify="left"
                  ).pack(anchor="w", padx=6, pady=(10, 6))

    def _build_loopback_tab(self, tabs: ttk.Notebook):
        loop = ttk.Frame(tabs)
        tabs.add(loop, text="Loopback test")
        self.loopback = tk.BooleanVar(value=False)
        ttk.Checkbutton(loop, text="Loopback mode", variable=self.loopback,
                        command=self._on_loopback_toggle).pack(anchor="w", padx=6, pady=(4, 0))
        ttk.Label(loop, text="Captures are saved as 'loopback' and not used by the fit; "
                             "the plots show volts.", wraplength=380, foreground="#666"
                  ).pack(anchor="w", padx=6, pady=(0, 4))
        self.loop_widgets = []
        row = ttk.Frame(loop)
        row.pack(fill="x", padx=6)
        ttk.Label(row, text="drive").pack(side="left")
        self.ao_channel = tk.StringVar(value="AOUT0 (pin 13)")
        w = ttk.Combobox(row, textvariable=self.ao_channel, width=14, state="readonly",
                         values=["AOUT0 (pin 13)", "AOUT1 (pin 15)"])
        w.pack(side="left", padx=4)
        self.loop_widgets.append(w)
        ttk.Label(row, text="wired to").pack(side="left")
        self.wired_to = tk.StringVar(value=self._channel_label(0))
        w = ttk.Combobox(row, textvariable=self.wired_to, width=14, state="readonly",
                         values=[self._channel_label(i) for i in range(12)])
        w.pack(side="left", padx=4)
        w.bind("<<ComboboxSelected>>", lambda _e: self._highlight_channel())
        self.loop_widgets.append(w)

        self.ao_mode = tk.StringVar(value="fixed")
        row = ttk.Frame(loop)
        row.pack(fill="x", padx=6, pady=4)
        for mode in ("fixed", "sine"):
            w = ttk.Radiobutton(row, text=mode, value=mode, variable=self.ao_mode)
            w.pack(side="left", padx=(0, 6))
            self.loop_widgets.append(w)
        row = ttk.Frame(loop)
        row.pack(fill="x", padx=6, pady=(0, 4))
        self.ao_volts = tk.StringVar(value="1.0")
        self.ao_freq = tk.StringVar(value="2.0")
        self.ao_offset = tk.StringVar(value="0.0")
        for label, var in (("volts / amplitude", self.ao_volts), ("Hz", self.ao_freq),
                           ("offset", self.ao_offset)):
            ttk.Label(row, text=label).pack(side="left")
            w = ttk.Entry(row, textvariable=var, width=6)
            w.pack(side="left", padx=(2, 8))
            self.loop_widgets.append(w)
        row = ttk.Frame(loop)
        row.pack(fill="x", padx=6, pady=(0, 6))
        for text, cmd in (("Start output", self._ao_start), ("Outputs off", self._ao_off)):
            w = ttk.Button(row, text=text, command=cmd)
            w.pack(side="left", padx=(0, 6))
            self.loop_widgets.append(w)
        ttk.Label(loop, text="Wire the AOUT pin to the chosen CH terminal and AGND "
                             "(pin 14) to an AGND. Unhook that load-cell lead first.",
                  wraplength=380, foreground="#666").pack(anchor="w", padx=6, pady=(0, 6))

    # -- deck drawing ---------------------------------------------------------
    def _to_screen(self, x: float, y: float) -> tuple[float, float]:
        """Deck metres -> canvas pixels. Origin top left, +x (right) right, +y (back) down."""
        return self.margin_left + x * self.scale, self.margin_top + y * self.scale

    def _on_canvas(self, sx: float, sy: float, pad: float = 12) -> tuple[float, float]:
        """Pull a point beyond the drawing back to its edge, so its marker stays visible."""
        w, h = float(self.canvas.cget("width")), float(self.canvas.cget("height"))
        return min(max(sx, pad), w - pad), min(max(sy, pad), h - pad)

    def _to_deck(self, sx: float, sy: float) -> tuple[float, float]:
        return (sx - self.margin_left) / self.scale, (sy - self.margin_top) / self.scale

    def _redraw_deck(self):
        c = self.canvas
        c.delete("all")
        L, W = self.deck_length, self.deck_width
        x0, y0 = self._to_screen(0, 0)         # front-left corner, the origin
        x1, y1 = self._to_screen(W, L)         # back-right corner
        c.create_rectangle(x0, y0, x1, y1, fill="#f4f1ea", outline="#444", width=2)

        # Grid in the chosen unit (10 cm or 6 in), with rulers along the top
        # (right of the left edge) and down the left (back from the front edge).
        unit, name = self._unit(), self.length_unit.get()
        step = 0.1 if name == "m" else 0.5
        span = max(L, W) / unit
        label_every = step if span <= 10 * step else 2 * step
        small = ("Helvetica", 9)
        for g in np.arange(0, W / unit + 1e-9, step):
            a, b = self._to_screen(g * unit, 0), self._to_screen(g * unit, L)
            c.create_line(*a, *b, fill="#e0dbd0")
            c.create_line(a[0], a[1] - 5, a[0], a[1], fill="#444")
            if abs(g / label_every - round(g / label_every)) < 1e-6:
                c.create_text(a[0], a[1] - 8, text=f"{g:g}", anchor="s", font=small)
        for g in np.arange(0, L / unit + 1e-9, step):
            a, b = self._to_screen(0, g * unit), self._to_screen(W, g * unit)
            c.create_line(*a, *b, fill="#e0dbd0")
            c.create_line(a[0] - 5, a[1], a[0], a[1], fill="#444")
            if abs(g / label_every - round(g / label_every)) < 1e-6:
                c.create_text(a[0] - 8, a[1], text=f"{g:g}", anchor="e", font=small)
        c.create_text(x0, y0 - 34, text=f"0   right of left edge ({name}) →", anchor="w",
                      font=("Helvetica", 10))
        c.create_text(x0 - 58, y0 + 40, text=f"back\nfrom\nfront\n({name})\n↓", anchor="n",
                      font=("Helvetica", 10), justify="center")
        c.create_text((x0 + x1) / 2, y0 + 14, text="FRONT", fill="#998",
                      font=("Helvetica", 11, "bold"))
        c.create_text((x0 + x1) / 2, y1 - 14, text="BACK", fill="#998",
                      font=("Helvetica", 11, "bold"))

        for cell, (cx, cy) in zip("1234", self.layout.positions):
            sx, sy = self._to_screen(cx, cy)
            c.create_oval(sx - 11, sy - 11, sx + 11, sy + 11, fill="#556", outline="")
            c.create_text(sx, sy, text=cell, fill="white", font=("Helvetica", 10, "bold"))

        # Earlier load trials, so you can see how well the deck is covered
        if self.session:
            for t in self.session.trials:
                if t["kind"] == "load":
                    sx, sy = self._on_canvas(*self._to_screen(t["x_m"], t["y_m"]))
                    c.create_oval(sx - 3, sy - 3, sx + 3, sy + 3, fill="#aaa", outline="")

        if self.location is not None:
            sx, sy = self._on_canvas(*self._to_screen(*self.location))
            self._draw_force(sx, sy, self.direction.get())
        self._draw_cop()

    def _draw_cop(self):
        """Blue target where the four Z forces balance; redrawn alone in live view."""
        c = self.canvas
        c.delete("cop")
        if self.cop is None:
            return
        sx, sy = self._on_canvas(*self._to_screen(*self.cop))
        c.create_oval(sx - 8, sy - 8, sx + 8, sy + 8, outline=COP_COLOR, width=2, tags="cop")
        c.create_line(sx - 13, sy, sx + 13, sy, fill=COP_COLOR, width=2, tags="cop")
        c.create_line(sx, sy - 13, sx, sy + 13, fill=COP_COLOR, width=2, tags="cop")

    def _draw_force(self, sx: float, sy: float, direction: str):
        c = self.canvas
        red = "#cc2222"
        if direction == "down":
            # Into the page: circle with a cross
            r = 10
            c.create_oval(sx - r, sy - r, sx + r, sy + r, outline=red, width=2)
            c.create_line(sx - 7, sy - 7, sx + 7, sy + 7, fill=red, width=2)
            c.create_line(sx - 7, sy + 7, sx + 7, sy - 7, fill=red, width=2)
            return
        fx, fy, _ = fp.DIRECTIONS[direction]
        ex, ey = sx + fx * 0.15 * self.scale, sy + fy * 0.15 * self.scale
        c.create_oval(sx - 4, sy - 4, sx + 4, sy + 4, fill=red, outline="")
        c.create_line(sx, sy, ex, ey, fill=red, width=3, arrow="last", arrowshape=(12, 14, 5))

    def _on_deck_click(self, event):
        # Anywhere on the drawing: around the deck is the frame, also a place
        # to load. Fill the boxes; _on_location_typed then moves the marker.
        x, y = self._to_deck(event.x, event.y)
        self.x_var.set(self._show_len(x))
        self.y_var.set(self._show_len(y))

    def _on_location_typed(self):
        """The X and Y boxes are the location; parse them whenever they change."""
        self.location = None
        try:
            x = float(self.x_var.get()) * self._unit()
            y = float(self.y_var.get()) * self._unit()
        except ValueError:
            self.location_label.config(text="No location chosen: click the deck or type it")
        else:
            self.location = (x, y)
            name = self.length_unit.get()
            text = (f"Load at {self.x_var.get()} {name} right of the left edge, "
                    f"{self.y_var.get()} {name} back from the front")
            if self.z_var.get().strip():
                text += f", {self.z_var.get()} {name} below the deck surface"
            tol = 1e-6
            if not (-tol <= x <= self.deck_width + tol and -tol <= y <= self.deck_length + tol):
                text += "  (off the deck: on the frame)"
            self.location_label.config(text=text)
        self._redraw_deck()
        self._update_buttons()

    # -- capture --------------------------------------------------------------
    def _force_n(self) -> float:
        return float(self.load_amount.get()) * fp.FORCE_UNITS[self.load_unit.get()]

    def _update_force_label(self):
        try:
            n = self._force_n()
            self.force_label.config(text=f"= {n:.1f} N = {n / fp.FORCE_UNITS['lbf']:.1f} lbf")
        except (ValueError, KeyError):
            self.force_label.config(text="(enter a number)")

    def _update_buttons(self):
        can = self.worker is not None and not self.busy
        loop = self.loopback.get()
        self.tare_button.config(state="normal" if can and not loop else "disabled")
        load_ok = loop or self.location is not None
        self.capture_button.config(state="normal" if can and load_ok else "disabled",
                                   text=self._log_text(loop))
        self.live_check.config(state="normal" if self.worker else "disabled")

    def _capture(self, kind: str):
        if self.loopback.get():
            kind = "loopback"
        fields = {}
        if kind == "load":
            try:
                force_n, below = self._force_n(), float(self.z_var.get().strip() or 0)
                z = below * self._unit()
            except ValueError:
                messagebox.showerror("Capture", "Load and depth below the deck must be numbers.")
                return
            if force_n <= 0:
                messagebox.showerror("Capture", "Load must be positive.")
                return
            if z < 0:
                messagebox.showerror("Capture", "Depth is 0 on the deck surface and positive "
                                                "below it (e.g. a pull on the substructure).")
                return
            if self.session is None or self.session.last_tare_id() is None:
                if not messagebox.askyesno(
                        "No tare yet",
                        "There is no tare in this session, so the fit will skip this load.\n"
                        "Capture anyway?"):
                    return
            entered = {"load": float(self.load_amount.get()), "load_unit": self.load_unit.get(),
                       "right": float(self.x_var.get()), "back": float(self.y_var.get()),
                       "below": below, "length_unit": self.length_unit.get()}
            fields = fp.load_trial_entry(self.direction.get(), *self.location, z, force_n, entered)
        elif kind == "loopback":
            fields = {"loopback": self._loopback_settings()}
        fields["notes"] = self.notes.get().strip()

        target = self.session.folder if self.session else self.data_dir
        free_mb = free_space_mb(target)
        if free_mb is not None and free_mb < MIN_FREE_MB:
            messagebox.showerror("Capture", f"Only {free_mb:.0f} MB free where the data are "
                                            f"saved ({target}). Free some space or change the "
                                            "data folder.")
            return

        self.busy = True
        self._update_buttons()
        self._set_status(f"Capturing {kind}... hold still.")
        self.progress["value"] = 0
        self._capture_started = dt.datetime.now()
        self.worker.send("capture", kind=kind, fields=fields)
        self._tick_progress()

    def _tick_progress(self):
        if not self.busy:
            return
        elapsed = (dt.datetime.now() - self._capture_started).total_seconds()
        self.progress["value"] = min(100, 100 * elapsed / fp.CAPTURE_SECONDS)
        self.after(100, self._tick_progress)

    def _on_captured(self, volts: np.ndarray, rate: float, kind: str, fields: dict):
        self.busy = False
        self.progress["value"] = 100
        expected = int(fp.CAPTURE_SECONDS * fp.SAMPLE_RATE)
        try:
            if volts.shape != (expected, 12) or not np.isfinite(volts).all():
                raise ValueError(f"capture came back as {volts.shape}, expected ({expected}, 12)")
            if self.session is None:
                folder = self.data_dir / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
                info = {"product": self.daq.product_name, "serial": self.daq.serial}
                self.session = fp.Session.create(folder, self.deck_length, self.deck_width, info)
            entry = self.session.add_trial(volts, rate, kind, **fields)
        except Exception as e:
            self._save_failed(e, volts, rate)
            return
        if kind == "tare":
            self.zero_v = volts.mean(axis=0)
        self._refresh_trials()
        self.live.set(False)
        self._on_live_toggle()
        self._show_trial(entry, volts)
        self._redraw_deck()

        msg = f"Saved {entry['file']} ({kind})."
        if kind == "loopback":
            i = self._wired_index()
            v = volts[:, i]
            msg += (f"  {self._channel_label(i)}: mean {v.mean():+.4f} V, "
                    f"min {v.min():+.4f} V, max {v.max():+.4f} V")
        else:
            worst = int(np.argmax(volts.std(axis=0)))
            msg += (f"  Noisiest channel {fp.SIGNAL_NAMES[worst]}: "
                    f"std {volts[:, worst].std() * 1000:.2f} mV")
        if self.session.log_warning:
            msg += "\n" + self.session.log_warning
        self._set_status(msg)
        self._update_buttons()

    def _save_failed(self, error: Exception, volts: np.ndarray, rate: float):
        """Never lose a capture: if the session can't take it, put it in the home folder."""
        rescue = Path.home() / f"forceplate_unsaved_{dt.datetime.now():%Y%m%d-%H%M%S}.csv"
        try:
            fp.save_capture_csv(rescue, volts, rate)
            where = f"The raw data were saved to {rescue} instead."
        except Exception as e2:
            where = f"The data could not be saved anywhere ({e2})."
        self._set_status(f"ERROR saving capture: {error}. {where}")
        self._update_buttons()
        messagebox.showerror("Save failed", f"{error}\n\n{where}")

    # -- live view and plotting ----------------------------------------------
    def _on_live_toggle(self):
        if self.worker is None:
            return
        on = self.live.get()
        if on:
            self.live_buffer[:] = np.nan
        self.worker.send("monitor", on=on)

    def _on_monitor(self, blocks: list[np.ndarray]):
        if not self.live.get():
            return
        volts = np.concatenate(blocks)[-len(self.live_buffer):]
        n = len(volts)
        self.live_buffer = np.roll(self.live_buffer, -n, axis=0)
        self.live_buffer[-n:] = volts
        now = dt.datetime.now()
        if (now - self._last_live_draw).total_seconds() < LIVE_REDRAW_S:
            return
        self._last_live_draw = now
        self._plot(self.live_buffer, fp.SAMPLE_RATE, "live", self.zero_v,
                   t0=-MONITOR_WINDOW_S, points=LIVE_POINTS)
        self._show_cop(volts[-MONITOR_BLOCK:].mean(axis=0), self.zero_v, "Live")

    def _plot(self, volts: np.ndarray, rate: float, title: str, zero: np.ndarray | None,
              t0: float = 0.0, points: int | None = None):
        """
        Newtons per cell (tared by `zero`), or volts in loopback mode. `points`
        thins each line to about that many samples, for a quick live redraw.
        """
        loop = self.loopback.get()
        data = volts if loop else fp.cell_forces(volts, zero, self.layout)
        t = t0 + np.arange(len(data)) / rate
        step = max(1, len(data) // points) if points else 1
        for i, line in enumerate(self.lines):
            line.set_data(t[::step], data[::step, i])
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()
            ax.set_ylabel("volts" if loop else "N", fontsize=8)
        if loop:
            note = "volts (loopback mode)"
        else:
            note = "newtons" + ("" if zero is not None else ", NOT TARED")
            if self.layout.placeholder:
                note += f", placeholder scale (edit {self.layout.path.name})"
        self.showing_label.config(text=f"Plots: {title}, in {note}")
        self._highlight_channel(draw=False)
        # Draw now, not when idle: queued redraws that outlast the 50 ms poll
        # would never let the event loop go idle.
        for canvas in self.plot_canvases:
            canvas.draw()

    def _show_cop(self, mean_v: np.ndarray, zero: np.ndarray | None, what: str):
        """Centre of pressure from the four Z forces, on the deck and as text."""
        self.cop = None
        if self.loopback.get():
            text = ""
        elif zero is None:
            text = f"{what}: tare first (no load, Tare) to see the centre of pressure."
        else:
            forces = fp.cell_forces(mean_v, zero, self.layout)
            cop = fp.cop_from_cells(forces, self.layout)
            if cop is None:
                text = f"{what}: almost no load on the deck."
            else:
                x, y, total = cop
                self.cop = (x, y)
                u = self.length_unit.get()
                text = (f"{what}: {total:.0f} N down; centre of pressure (blue) "
                        f"{self._show_len(x)} {u} right, {self._show_len(y)} {u} back")
        self.cop_label.config(text=text)
        self._draw_cop()

    def _highlight_channel(self, draw: bool = True):
        target = self._wired_index() if self.loopback.get() else None
        for i, line in enumerate(self.lines):
            line.set_linewidth(2.2 if i == target else 0.9)
        if draw:
            for canvas in self.plot_canvases:
                canvas.draw_idle()

    def _trial_mean_v(self, trial_id: int | None) -> np.ndarray | None:
        t = next((t for t in self.session.trials if t["id"] == trial_id), None) \
            if self.session and trial_id is not None else None
        return np.array(t["mean_v"]) if t else None

    def _show_trial(self, t: dict, volts: np.ndarray):
        """Plot a saved capture, tared by its own tare (a load) or the latest one."""
        zero = self._trial_mean_v(t.get("tare_id")) if t["kind"] == "load" else None
        if zero is None:
            zero = self.zero_v
        self._plot(volts, t["rate_hz"], f"#{t['id']} {self._describe(t)}", zero)
        self._show_cop(volts.mean(axis=0), zero, f"#{t['id']} average")

    # -- session list ---------------------------------------------------------
    def _describe(self, t: dict) -> str:
        if t["kind"] == "tare":
            return "tare"
        if t["kind"] == "loopback":
            lb = t.get("loopback", {})
            return f"loopback {lb.get('mode', '')} AOUT{lb.get('ao_channel', '?')}->{lb.get('wired_to', '?')}"
        tare = t.get("tare_id")
        e = t.get("entered")
        if e:
            u = e["length_unit"]
            where = f"R {e['right']:g}{u} B {e['back']:g}{u}" + (f" D {e['below']:g}{u}" if e["below"] else "")
            amount = f"{e['load']:g} {e['load_unit']}"
        else:
            where = f"x {t['x_m']:.2f} y {t['y_m']:.2f} z {t['z_m']:.2f} m"
            amount = f"{t['force_n']:.1f} N"
        return f"load {t['direction']:7} {amount:>9} @ {where}  tare #{tare if tare else '-'}"

    def _refresh_trials(self):
        self.trial_list.delete(0, "end")
        self.data_label.config(text=str(self.data_dir))
        if self.session is None:
            self.session_label.config(text="A new session folder is created in the data "
                                           "folder on the first capture.")
            return
        loads = sum(t["kind"] == "load" for t in self.session.trials)
        self.session_label.config(text=f"Session {self.session.folder.name}   ({loads} load trials)")
        for t in self.session.trials:
            self.trial_list.insert("end", f"#{t['id']:<3} {self._describe(t)}")
        self.trial_list.see("end")

    def _selected_trial(self) -> dict | None:
        sel = self.trial_list.curselection()
        if not sel or self.session is None:
            return None
        return self.session.trials[sel[0]]

    def _on_trial_select(self, _event):
        t = self._selected_trial()
        if t is None:
            return
        data = np.loadtxt(self.session.folder / t["file"], delimiter=",", skiprows=1)
        self.live.set(False)
        self._on_live_toggle()
        self._show_trial(t, data[:, 1:])
        if t.get("notes"):
            self._set_status(f"#{t['id']} notes: {t['notes']}")

    def _remove_trial(self):
        t = self._selected_trial()
        if t is None:
            return
        if messagebox.askyesno("Remove trial", f"Remove trial #{t['id']} from the session?\n"
                                               "(its CSV is renamed to .csv.removed)"):
            self.session.remove_trial(t["id"])
            if self.session.log_warning:
                self._set_status(self.session.log_warning)
            self._refresh_trials()
            self._redraw_deck()

    def _new_session(self):
        self.session = None
        self._refresh_trials()
        self._redraw_deck()

    def _fit(self):
        if self.session is None:
            return
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ids, signals, known = fp.gather(self.session)
            if ids:
                fit = fp.fit_calibration(ids, signals, known)
                fp.report(fit)
                paths = fp.save_fit(fit, self.session)
                print(f"\nSaved {paths[0]} and {paths[1]}")
            else:
                print("No usable load trials (each needs a tare captured before it).")
        win = tk.Toplevel(self)
        win.title("Calibration fit")
        text = tk.Text(win, width=130, height=40, font=MONO)
        text.insert("1.0", out.getvalue())
        text.config(state="disabled")
        text.pack(fill="both", expand=True)

    # -- loopback ---------------------------------------------------------------
    @staticmethod
    def _channel_label(i: int) -> str:
        name, _, terminal = fp.CHANNELS[i]
        return f"{terminal} ({name})"

    def _wired_index(self) -> int:
        return [self._channel_label(i) for i in range(12)].index(self.wired_to.get())

    def _ao_index(self) -> int:
        return 0 if self.ao_channel.get().startswith("AOUT0") else 1

    def _loopback_settings(self) -> dict:
        return {"ao_channel": self._ao_index(), "wired_to": self.wired_to.get(),
                "mode": self.ao_mode.get(), "volts": self.ao_volts.get(),
                "freq_hz": self.ao_freq.get(), "offset": self.ao_offset.get()}

    def _on_loopback_toggle(self, quiet: bool = False):
        on = self.loopback.get()
        usable = on and self.daq is not None and self.daq.has_ao
        for w in self.loop_widgets:
            w.config(state=("readonly" if isinstance(w, ttk.Combobox) else "normal")
                     if usable else "disabled")
        if not on and self.worker is not None:
            self.worker.send("ao_off", quiet=quiet)
        self._highlight_channel()
        self._update_buttons()

    def _ao_start(self):
        try:
            volts = float(self.ao_volts.get())
            freq = float(self.ao_freq.get())
            offset = float(self.ao_offset.get())
        except ValueError:
            messagebox.showerror("Loopback", "Volts, Hz and offset must be numbers.")
            return
        ch = self._ao_index()
        if self.ao_mode.get() == "fixed":
            if abs(volts) > 10:
                messagebox.showerror("Loopback", "AOUT range is +/-10 V.")
                return
            self.worker.send("ao_dc", channel=ch, volts=volts)
        else:
            if abs(offset) + abs(volts) > 10:
                messagebox.showerror("Loopback", "offset + amplitude must stay within 10 V.")
                return
            if not 0 < freq <= AO_RATE / 10:
                messagebox.showerror("Loopback", f"Frequency must be in (0, {AO_RATE / 10:g}] Hz.")
                return
            self.worker.send("ao_sine", channel=ch, amplitude=volts, offset=offset, freq=freq)

    def _ao_off(self):
        self.worker.send("ao_off")

    # -- plumbing ---------------------------------------------------------------
    def _set_status(self, text: str):
        self.status.config(text=text)

    def _poll(self):
        """Drain results from the DAQ thread. Runs every 50 ms on the Tk thread."""
        if self.worker is not None:
            # Handle only what is queued now, and redraw the live view once for
            # all of it, so a fast producer can never keep this loop busy.
            monitor_blocks = []
            try:
                for _ in range(self.worker.results.qsize()):
                    kind, payload = self.worker.results.get_nowait()
                    if kind == "monitor":
                        monitor_blocks.append(payload)
                    elif kind == "capture":
                        self._on_captured(*payload)
                    elif kind == "status":
                        self._set_status(payload)
                    elif kind == "error":
                        self.busy = False
                        self.live.set(False)
                        self._set_status(f"ERROR {payload}")
                        self._update_buttons()
            except queue.Empty:
                pass
            if monitor_blocks:
                self._on_monitor(monitor_blocks)
        self.after(50, self._poll)

    def _on_close(self):
        if self.worker is not None:
            self.worker.monitoring = False
            self.worker.send("quit")
            self.worker.join(timeout=8.0)
        self.destroy()


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def free_space_mb(folder: Path) -> float | None:
    """Free space on the drive holding `folder` (or its nearest existing parent)."""
    for path in (folder, *folder.parents):
        if path.exists():
            try:
                return shutil.disk_usage(path).free / 1e6
            except OSError:
                return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Force-plate calibration GUI.")
    parser.add_argument("--session", type=Path, help="existing session folder to add to")
    parser.add_argument("--length", type=float,
                        help="deck length front-to-back, metres (default: forceplate_layout.json)")
    parser.add_argument("--width", type=float,
                        help="deck width side-to-side, metres (default: forceplate_layout.json)")
    args = parser.parse_args()

    daq, error = None, ""
    try:
        daq = USB1608G.open()
    except DaqError as e:
        error = str(e)
    App(daq, error, args.session, args.length, args.width).mainloop()


if __name__ == "__main__":
    main()
