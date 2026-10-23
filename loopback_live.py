#!/usr/bin/env python3
"""
Live loopback plot: drive an analog output, watch all 12 load-cell inputs.

Default wiring (through the load-cell D26 harness):
  AOUT0  -> D26 pin 1  (1X+)  -> DAQ CH0H  = AI channel 0
  AGND   -> D26 pin 10 (1X-)  -> DAQ AGND
Unplug the load-cell amplifier from that D26 first, or AOUT0 and the amp
drive the same wire.

Two views, switched with the button at the bottom left, both showing 10 s on
a +/-3 V scale:
  12 channels    one small plot per channel, each with its mean and
                 peak-to-peak (green when it sits at 0 V within 20 mV)
  main + rest    the main channel on top, the other 11 together underneath
The main channel (drawn thick, checked against the output) starts as --ai;
change it with the arrow buttons after moving the AOUT wire to another pin.

Inputs with nothing connected will NOT sit at 0: the DAQ reads its inputs one
after another through a single converter, and an unconnected input keeps some
charge from the input read just before it, so it echoes that one. They read
0 V once something drives them (a load-cell amp, or a jumper to a "-" pin).

Unplugging the DAQ doesn't stop the program: the plot freezes, says so, and
picks up again (output restarted) when the DAQ is plugged back in.

Usage:
  uv run loopback_live.py                         # 3 V sine at 1 Hz on AOUT0, wired to 1X
  uv run loopback_live.py --ai 3                  # AOUT0 moved to D26 pin 3 (2X)
  uv run loopback_live.py --dc 2.5                # hold 2.5 V instead of a sine
  uv run loopback_live.py --amp 3 --freq 5        # bigger, faster sine
Close the window (or Ctrl-C) to stop; the output returns to 0 V.
"""

import argparse
import signal
import sys
import time

import tk_env  # noqa: F401  (must come before matplotlib opens a Tk window)

import matplotlib.pyplot as plt
import numpy as np
import usb.core
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

import forceplate as fp
from mcc1608g import USB1608G, AiMode, AiRange, DaqError, find_devices

RATE = 1000.0           # AI scans per second (all 12 channels each scan)
BLOCK = 200             # scans per plot update (0.2 s)
WINDOW_S = 10.0         # seconds of history shown
DRAW_EVERY = 5          # plot every 5th sample; the stats use all of them
AO_RATE = 1000.0        # DAC updates per second for the sine
AO_SECONDS = 60 * 60    # sine buffer length; the output holds after this
RETRY_S = 1.0           # how often to look for an unplugged DAQ
ZERO_TOL = 0.02         # volts: "sitting at 0" for the other channels
Y_LIM = 3.0             # volts, +/- on every plot (widened if the output goes further)

# "+" pin on the load-cell D26 for each signal (see d26wiring.png).
D26_PIN = {"1X": 1, "1Y": 2, "1Z": 19, "2X": 3, "2Y": 20, "2Z": 21,
           "3X": 5, "3Y": 6, "3Z": 23, "4X": 7, "4Y": 8, "4Z": 9}

# Errors that mean "the DAQ went away" rather than a bug in this script.
USB_GONE = (DaqError, usb.core.USBError, ValueError, OSError)


class Rig:
    """The DAQ and the output playing on it. Can be dropped and reconnected."""

    def __init__(self, ao: int, dc: float | None, amp: float, freq: float, offset: float):
        self.ao, self.dc, self.amp, self.freq, self.offset = ao, dc, amp, freq, offset
        self.daq: USB1608G | None = None
        self.ao_scan = None

    def connect(self):
        daq = USB1608G.open()
        try:
            daq.stop_scans()
            if self.dc is not None:
                daq.a_out(self.ao, self.dc)
            else:
                t = np.arange(int(AO_SECONDS * AO_RATE)) / AO_RATE
                wave = self.offset + self.amp * np.sin(2 * np.pi * self.freq * t)
                self.ao_scan = daq.a_out_scan(self.ao, self.ao, wave, AO_RATE)
        except BaseException:
            daq.close()
            raise
        self.daq = daq

    def read_block(self) -> np.ndarray:
        ai = self.daq.a_in_scan(fp.AI_LOW, fp.AI_HIGH, AiMode.SINGLE_ENDED, AiRange.BIP10V,
                                BLOCK, RATE)
        return ai.read_all()

    def drop(self):
        """Forget a DAQ that stopped answering, without touching it further."""
        if self.ao_scan is not None:
            self.ao_scan._stopping.set()    # let the feeder thread exit quietly
        self.ao_scan = None
        if self.daq is not None:
            try:
                self.daq.close()
            except Exception:
                pass
        self.daq = None

    def shutdown(self):
        """Stop the output and set it to 0 V, if the DAQ is still there."""
        if self.daq is None:
            return
        try:
            if self.ao_scan is not None:
                self.ao_scan.stop()
            self.daq.a_out(self.ao, 0.0)
            print("Output set to 0 V.")
        except Exception as e:
            print(f"Could not reset the output ({e}); unplug the DAQ to be sure.")
        self.drop()


def main():
    parser = argparse.ArgumentParser(description="Live AO -> AI loopback plot, all 12 inputs.")
    parser.add_argument("--ao", type=int, default=0, choices=(0, 1), help="output channel")
    parser.add_argument("--ai", type=int, default=0, choices=range(12),
                        help="input channel wired to it (0 = 1X = D26 pin 1)")
    parser.add_argument("--dc", type=float, help="hold this voltage instead of a sine")
    parser.add_argument("--amp", type=float, default=3.0, help="sine amplitude, volts")
    parser.add_argument("--freq", type=float, default=1.0, help="sine frequency, Hz")
    parser.add_argument("--offset", type=float, default=0.0, help="sine offset, volts")
    args = parser.parse_args()

    if args.dc is not None:
        if abs(args.dc) > 10:
            sys.exit("--dc must be within +/-10 V")
        expect_lo = expect_hi = args.dc
        command = f"AOUT{args.ao} = {args.dc:+.3f} V (DC)"
    else:
        if abs(args.offset) + abs(args.amp) > 10 or not 0 < args.freq <= AO_RATE / 10:
            sys.exit("sine must stay within +/-10 V and 0 < freq <= 100 Hz")
        expect_lo, expect_hi = args.offset - args.amp, args.offset + args.amp
        command = (f"AOUT{args.ao} = {args.amp:.2f} V sine, {args.freq:g} Hz, "
                   f"offset {args.offset:+.2f} V")

    # Ctrl-C or a kill still runs the cleanup at the bottom.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    rig = Rig(args.ao, args.dc, args.amp, args.freq, args.offset)
    try:
        rig.connect()
    except USB_GONE as e:
        sys.exit(f"Could not open the DAQ: {e}")
    print(f"Connected to {rig.daq}\nOutput: {command}\nClose the plot window to stop.")

    try:
        run_plot(rig, args.ai, command, expect_lo, expect_hi, args.dc)
    except KeyboardInterrupt:
        pass
    finally:
        rig.shutdown()
        print("Done.")


def run_plot(rig: Rig, wired_ai: int, command: str,
             expect_lo: float, expect_hi: float, dc: float | None):
    n = int(WINDOW_S * RATE)
    buffer = np.full((n, 12), np.nan)
    t_axis = (np.arange(n) / RATE - WINDOW_S)[::DRAW_EVERY]
    ylim = 1.08 * max(Y_LIM, abs(expect_lo), abs(expect_hi))
    view = {"main": wired_ai, "grid": True}

    fig = plt.figure(figsize=(13, 9))
    box = dict(left=0.06, right=0.98, top=0.93, bottom=0.13)

    # View 1: one plot per channel, 4 rows (load cells) x 3 columns (X, Y, Z).
    gs = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.12, **box)
    grid_axes, grid_lines, grid_stats = [], [], []
    for i in range(12):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.axhline(0, color="#999", linewidth=0.6)
        (line,) = ax.plot(t_axis, buffer[::DRAW_EVERY, i])
        grid_axes.append(ax)
        grid_lines.append(line)
        grid_stats.append(ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top",
                                  family="monospace", fontsize=8))
        if i // 3 == 3:
            ax.set_xlabel("seconds")
        else:
            ax.tick_params(labelbottom=False)
        if i % 3 == 0:
            ax.set_ylabel("V")
        else:
            ax.tick_params(labelleft=False)

    # View 2: the main channel on top, the other 11 together underneath.
    gs2 = fig.add_gridspec(2, 1, hspace=0.25, **box)
    main_ax = fig.add_subplot(gs2[0])
    rest_ax = fig.add_subplot(gs2[1], sharex=main_ax)
    (main_line,) = main_ax.plot(t_axis, buffer[::DRAW_EVERY, 0], color="#2255bb", linewidth=1.4)
    main_ax.set_facecolor("#eef3ff")
    main_stats = main_ax.text(0.01, 0.95, "", transform=main_ax.transAxes, va="top",
                              family="monospace", fontsize=10)
    rest_lines = [rest_ax.plot(t_axis, buffer[::DRAW_EVERY, i], linewidth=0.8,
                               label=fp.SIGNAL_NAMES[i])[0] for i in range(12)]
    rest_stats = rest_ax.text(0.01, 0.95, "", transform=rest_ax.transAxes, va="top",
                              family="monospace", fontsize=10)
    rest_ax.set_xlabel("seconds")
    for ax in (main_ax, rest_ax):
        ax.axhline(0, color="#999", linewidth=0.6)
        ax.set_ylabel("V")

    expect_lines = []
    for ax in grid_axes + [main_ax, rest_ax]:
        ax.set_xlim(-WINDOW_S, 0)
        ax.set_ylim(-ylim, ylim)
        ax.grid(True, alpha=0.3)

    title = fig.suptitle("", fontsize=11)
    fig.text(0.5, 0.005, "Unconnected inputs echo the channel read before them; "
                         "they read 0 V only when something drives them.",
             ha="center", fontsize=8, color="#666")

    def headline() -> str:
        name = fp.SIGNAL_NAMES[view["main"]]
        return f"{command}  ->  main: {name} (D26 pin {D26_PIN[name]})"

    def restyle():
        """Apply the current view and main channel to every plot."""
        m = view["main"]
        for line in expect_lines:
            line.remove()
        expect_lines.clear()
        for i, (ax, line) in enumerate(zip(grid_axes, grid_lines)):
            name = fp.SIGNAL_NAMES[i]
            ax.set_title(f"{name}   {fp.CHANNELS[i][2]}   D26 pin {D26_PIN[name]}",
                         fontsize=9, fontweight="bold" if i == m else "normal")
            ax.set_facecolor("#eef3ff" if i == m else "white")
            line.set_color("#2255bb" if i == m else "#557799")
            line.set_linewidth(1.4 if i == m else 0.8)
        for ax in (grid_axes[m], main_ax):
            for v in sorted({expect_lo, expect_hi}):
                expect_lines.append(ax.axhline(v, color="#cc3333", linestyle="--",
                                               linewidth=0.8))
        name = fp.SIGNAL_NAMES[m]
        main_ax.set_title(f"main: {name}   {fp.CHANNELS[m][2]}   D26 pin {D26_PIN[name]}",
                          fontsize=10, fontweight="bold")
        rest_ax.set_title("the other 11 inputs", fontsize=10)
        for i, line in enumerate(rest_lines):
            line.set_visible(i != m)
        rest_ax.legend([l for i, l in enumerate(rest_lines) if i != m],
                       [s for i, s in enumerate(fp.SIGNAL_NAMES) if i != m],
                       loc="upper right", ncol=6, fontsize=8)
        for ax in grid_axes:
            ax.set_visible(view["grid"])
        main_ax.set_visible(not view["grid"])
        rest_ax.set_visible(not view["grid"])
        view_button.label.set_text("Show: main + rest" if view["grid"] else "Show: 12 channels")
        main_label.set_text(f"main: {name}")
        title.set_text(headline())
        fig.canvas.draw_idle()

    # Buttons along the bottom
    view_button = Button(fig.add_axes((0.06, 0.035, 0.16, 0.045)), "")
    prev_button = Button(fig.add_axes((0.25, 0.035, 0.04, 0.045)), "\u25c0")
    next_button = Button(fig.add_axes((0.38, 0.035, 0.04, 0.045)), "\u25b6")
    main_label = fig.text(0.335, 0.057, "", ha="center", va="center", fontsize=10)

    def toggle_view(_event):
        view["grid"] = not view["grid"]
        restyle()

    def step_main(delta):
        view["main"] = (view["main"] + delta) % 12
        restyle()

    view_button.on_clicked(toggle_view)
    prev_button.on_clicked(lambda _e: step_main(-1))
    next_button.on_clicked(lambda _e: step_main(+1))
    restyle()

    state = {"last_try": 0.0, "lost_at": None}

    def update(_frame):
        nonlocal buffer
        if rig.daq is None:
            reconnect()
            return []
        try:
            block = rig.read_block()
        except USB_GONE as e:
            print(f"DAQ lost: {e}")
            rig.drop()
            state["lost_at"] = time.monotonic()
            title.set_text("DAQ DISCONNECTED: plug it back in (plot paused)")
            title.set_color("#cc2222")
            return []

        buffer = np.roll(buffer, -BLOCK, axis=0)
        buffer[-BLOCK:] = block
        lo, hi = np.nanmin(buffer, axis=0), np.nanmax(buffer, axis=0)
        mean, ptp = np.nanmean(buffer, axis=0), hi - lo
        m = view["main"]
        for i in range(12):
            y = buffer[::DRAW_EVERY, i]
            grid_lines[i].set_ydata(y)
            rest_lines[i].set_ydata(y)
            grid_stats[i].set_text(f"mean {mean[i]:+.3f}\np-p  {ptp[i]:.3f}")
            if i != m:
                at_zero = abs(mean[i]) < ZERO_TOL and ptp[i] < ZERO_TOL
                grid_stats[i].set_color("#227722" if at_zero else "#cc7700")
        main_line.set_ydata(buffer[::DRAW_EVERY, m])

        # The main channel: does it match what we're sending?
        if dc is not None:
            ok = abs(mean[m] - dc) < 0.05
            text = f"mean {mean[m]:+.4f}  (expect {dc:+.4f})"
        else:
            ok = abs(lo[m] - expect_lo) < 0.1 and abs(hi[m] - expect_hi) < 0.1
            text = f"min {lo[m]:+.3f}  max {hi[m]:+.3f}  (expect {expect_lo:+.3f} .. {expect_hi:+.3f})"
        color = "#227722" if ok else "#cc2222"
        grid_stats[m].set_text(f"mean {mean[m]:+.3f}\np-p  {ptp[m]:.3f}")
        grid_stats[m].set_color(color)
        main_stats.set_text(text + ("  OK" if ok else "  MISMATCH"))
        main_stats.set_color(color)
        others = [i for i in range(12) if i != m]
        worst = max(others, key=lambda i: ptp[i])
        rest_stats.set_text(f"largest swing: {fp.SIGNAL_NAMES[worst]} p-p {ptp[worst]:.3f} V")
        title.set_text(f"{headline()}    {'OK' if ok else 'MISMATCH'}")
        title.set_color(color)
        return []

    def reconnect():
        now = time.monotonic()
        if now - state["last_try"] < RETRY_S:
            return
        state["last_try"] = now
        gone_for = now - state["lost_at"]
        if not find_devices():
            title.set_text(f"DAQ DISCONNECTED for {gone_for:.0f} s: plug it back in")
            return
        try:
            rig.connect()
        except USB_GONE as e:
            title.set_text(f"DAQ found, still starting up ({e}); retrying")
            rig.drop()
            return
        print(f"Reconnected to {rig.daq}; output restarted.")
        buffer[:] = np.nan
        title.set_text(headline())
        title.set_color("black")

    anim = FuncAnimation(fig, update, interval=10, cache_frame_data=False)  # noqa: F841
    plt.show()
    return update, fig, view, restyle   # handles for testing


if __name__ == "__main__":
    main()
