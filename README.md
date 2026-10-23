# Brock treadmill DAQ

Software for the treadmill force plate, which is read through an MCC USB-1608GX-2AO
data-acquisition box (the "DAQ"). The main program is a calibration app: you put known
weights on the treadmill deck, click where you put them, and the app records the load
cells and works out the calibration.

Works on Mac, Windows and Linux.

---

# Quick start

You only do steps 1–4 once per computer. After that, skip to step 5.

## 1. Open a terminal

The terminal is a window where you type commands. Type (or paste) each command
exactly as shown, then press **Enter**.

- **Mac:** press `Cmd + Space`, type `Terminal`, press Enter.
- **Windows:** click Start, type `PowerShell`, press Enter.
- **Linux:** press `Ctrl + Alt + T`.

## 2. Install uv

uv is a small tool that installs Python and everything else this project needs.
You don't need to install Python yourself.

**Mac or Linux**, paste this into the terminal:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows**, paste this into PowerShell:
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

When it finishes, **close the terminal window and open a new one** (step 1 again).
To check it worked, type `uv --version`. You should see something like
`uv 0.9.27`. If you see "command not found", close the terminal and open a new
one again.

## 3. Download this project

**Easiest: download a ZIP.**
1. On this project's GitHub page, click the green **Code** button, then **Download ZIP**.
2. Unzip it. Move the unzipped folder somewhere easy to find, for example
   your Documents folder, and rename it to `brock-treadmill-daq`.

**Or, if you have git:**
```
cd ~/Documents
git clone https://github.com/jeremydwong/brock-treadmill-daq.git
```

## 4. Set up the project

In the terminal, go into the project folder. If you put it in Documents:
```
cd ~/Documents/brock-treadmill-daq
```
(On Windows the same command works in PowerShell.)

> **Tip:** instead of typing the path, type `cd ` (with a space), drag the folder
> from Finder or Explorer into the terminal window, and press Enter.

Then run:
```
uv sync
```
The first time, this downloads Python and the libraries. It takes a minute or
two and needs internet. When it's done you'll get your prompt back.

### Let the computer talk to the DAQ (once per computer)

- **Mac:** nothing to do.
- **Windows:** Windows needs a different USB driver for the DAQ. Do this once:
  1. Plug the DAQ into USB.
  2. Download **Zadig** from <https://zadig.akeo.ie/> and run it.
  3. In Zadig's menu, click **Options → List All Devices**.
  4. In the drop-down, pick **USB-1608GX-2AO**.
  5. To the right of the green arrow, choose **WinUSB**, then click **Replace Driver**
     (or **Install Driver**). Wait for it to finish.

  This means MCC's own software (InstaCal, DAQami) will no longer see the
  device on this computer. To undo it, run Zadig again and put the original driver back.
- **Linux:** run these two commands (they ask for your password), then unplug
  and replug the DAQ:
  ```
  echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="09db", MODE="0666"' | sudo tee /etc/udev/rules.d/99-mcc.rules
  sudo udevadm control --reload-rules
  ```

## 5. Run the app

Plug in the DAQ. In the terminal, go to the project folder and start the app:
```
cd ~/Documents/brock-treadmill-daq
uv run forceplate_gui.py
```

A window opens. The status line under the **Tare** and **Log** buttons should say
`Connected to USB-1608GX-2AO ...`. The first time after the DAQ is powered up, this takes
about a second while the app loads the DAQ's firmware.

If it says `No DAQ: ...` instead, see [Troubleshooting](#troubleshooting).

To quit, close the window.

---

# Using the calibration app

## What you're looking at

- **Left: the pictures.**
  - **Centre: the deck, seen from above.** The front of the treadmill is at the top.
    The rulers measure from the **top-left corner**, which is the front-left
    corner of the deck: across the top is "right of the left edge", down the
    side is "back from the front edge". The numbered dark circles are the four
    load cells, where `forceplate_layout.json` says they are (see
    [Load-cell layout](#load-cell-layout-and-wiring)).
  - **Around it: one plot per load cell**, beside the corner that cell sits
    under, showing its X, Y and Z forces in **newtons**. Forces are measured
    from the latest **Tare**; until you tare, the plots say "NOT TARED".
  - **Below it: the deck size.**
- **Right: the words and buttons.** At the top, the five things you set for
  each capture (direction, load, where it is, notes) and the **Tare** and
  **Log 5 s of data** buttons. Under them, a blue line gives the total down
  force and the **centre of pressure**, which is also drawn on the deck as a
  blue target.
- **Details** (click to open): the data folder, the list of captures, **Fit**,
  depth below the deck, the load-cell scale, and the loopback test.

## First: set the deck size and units

Below the deck drawing, pick **m** or **ft**, type the deck's **side to side**
and **front to back** size, and click **Set size**. Do this before you capture
anything; a session's deck size can't be changed later. Switching between m and
ft only changes what's shown; you can switch any time.

## Taking a calibration session

1. **Take the weight off the deck** and click **Tare (no load)**. The app records 5
   seconds of "nothing on the deck" as the zero point.
2. **Put a known weight on the deck** (or pull with a known force).
3. **Say where it is.** Either click on the deck drawing (a red marker appears
   and the boxes fill in), or type into the two boxes:
   - how far **right of the left edge**
   - how far **back from the front edge**

   If you're pulling on the frame underneath instead of the deck surface, also
   open **Details → Depth and scale** and type how far **below the deck
   surface** it is (e.g. `0.12` m for 12 cm below; always positive). Set it
   back to `0` afterwards. While it isn't 0, the line under the boxes says so.
4. **Choose the direction** the force pushes the deck:
   - **down**: a weight sitting on the deck
   - **forward / back / left / right**: a sideways pull (for example a rope over a pulley)
5. **Type the load** and pick its unit. **kg** or **lb** means a mass (a weight
   plate); the app works out its weight. **N** or **lbf** means a force you
   measured (for example with a force gauge). The force in N and lbf is shown
   next to it.
6. **Notes** (optional): anything worth remembering, e.g. "2 × 10 kg plates,
   belt off". Notes are saved with every capture until you change them.
7. Click **Log 5 s of data** and **keep everything still** for 5 seconds.
8. Repeat steps 2–7 with weights in many different places and in all five
   directions. Grey dots on the drawing show where you've already measured.
   Re-tare (step 1) every so often. Each capture uses the most recent tare.

You need **at least 12** load captures, and more is better, spread over the whole
deck and all directions.

**Live view** (tick box) shows the forces continuously, which helps you
check that everything is plugged in and that the weight has stopped swinging.
After a tare it also moves the blue centre-of-pressure target as the load
moves. It is worked out only from the four Z forces and where the cells sit,
before any calibration, so it's a sanity check: a weight placed on the deck
should land close to its red marker.

**Made a mistake?** Open **Details**, click the capture in the **Session** list
and click **Remove selected**. Clicking a capture in the list also shows its
plots and its average centre of pressure.

## Getting the calibration

Open **Details** and click **Fit**. A window shows the calibration matrix and how well it matches
each capture. It also saves the result in the session folder. You can also
run the fit later from the terminal:
```
uv run forceplate.py "<session folder>"
```

## Where your data goes

Everything is saved in the **data folder** shown under **Details → Session**.
By default that's **`ForcePlate data` inside your Documents folder**. Click
**Change...** to pick another folder (the app remembers it next time) and
**Open** to see the files.

Each session gets its own folder named with the date and time it started, e.g.
`ForcePlate data/20261019-142957/`:

| File | What's in it |
|---|---|
| `captures_log.csv` | **Open this in Excel.** One row per capture: date, time, tare or load, the load and where it was applied (as you typed it, and in metres), the deck size, and your notes |
| `trial_001.csv`, `trial_002.csv`, … | One per capture: 5 s × 1000 samples/s of all 12 channels, in volts |
| `session.json` | The same information as the log, which the fit reads |
| `calibration.json`, `calibration_matrix.csv` | Made by **Fit**: the 6 × 12 calibration matrix |

If `captures_log.csv` is open in Excel when you capture, Windows won't let the
app update it. The capture is still saved, the status line says so, and the
log catches up on the next capture after you close Excel.

If saving ever fails (folder on an unplugged drive, disk full), the capture is
saved as `forceplate_unsaved_<date>.csv` in your home folder instead, and the
app tells you.

To add more captures to an old session instead of starting a new one:
```
uv run forceplate_gui.py --session "<session folder>"
```

The calibration matrix turns the 12 tared voltages into the forces and moments on the deck:
```
[Fx, Fy, Fz, Mx, My, Mz] = matrix × (volts − tare volts)
```
Forces are in newtons and moments in newton-metres, about the deck's front-left
corner. Axes: **x to the right, y toward the back, z down** (into the deck). This is the
usual force-plate convention (AMTI, Bertec): someone standing on the deck gives a positive Fz.

---

# Testing the wiring (loopback mode)

You can check the DAQ and cabling without the load cells. The DAQ makes a
voltage on one of its outputs, you connect that output to one input, and you
see it appear in the app.

1. **Unplug that input's load-cell wire first.** Never connect an output to an
   input that a load cell is also connected to.
2. Using the screw-terminal board, connect a wire from **AOUT0** (pin 13) or
   **AOUT1** (pin 15) to the input you want to test (e.g. **CH0H**), and a
   second wire from **AGND** (pin 14) to an **AGND** terminal.
3. In the app, open **Details → Loopback test** and tick **Loopback mode**.
4. Choose which output you're driving (**drive**) and which input it's wired to
   (**wired to**). That input's trace is drawn thicker.
5. Choose **fixed** (a steady voltage) or **sine** (a wave). Set the voltage, plus
   frequency and offset for a sine. Click **Start output**.
6. Tick **Live view**. You should see the voltage on the input you chose.
7. Click **Outputs off** when you're done, and reconnect the load-cell wire.

While loopback mode is on, the plots show volts and the **Log** button saves loopback captures.
These are kept, but the calibration fit ignores them.

---

# Load-cell layout and wiring

`forceplate_layout.json` (in this folder) says where each load cell is and how
to turn its volts into newtons. Edit it in any text editor.

**Plug-in order.** Load cell *N* goes into the amplifier's load-cell input *N*
(the four M12 connectors on the Michigan Scientific SGA12), so its signals come
out on wire bundle *N* as NX, NY, NZ:

| Load cell / amplifier input | Where it sits under the deck |
|---|---|
| 1 | front left |
| 2 | front right |
| 3 | back right |
| 4 | back left |

(Seen from above, standing on the belt facing forward: clockwise from the front left.)

**Positions.** For each cell, `x_m` is how far its centre is **right of the
left edge** and `y_m` how far **back from the front edge**, in metres, measured
from the deck's front-left corner. The deck size there is the app's default.

**Scale.** `newtons_per_volt` is per axis and comes from each load cell's
calibration sheet. The amplifier has gain 201 and 10 V excitation, so if a sheet
says *S* mV/V at a rated load of *F* newtons:
```
newtons_per_volt = F / (2.01 × S)
```
A load pressing down should read as positive Z; if it reads negative, make that
cell's Z number negative. The numbers in the file now are **placeholders**, and
the line under the **Log** button says "placeholder scale" until you set `"scale_is_placeholder": false`.

# Wiring reference

All 12 load-cell signals are read in single-ended mode (each signal measured against AGND).

| Load cell signal | DAQ terminal | | Load cell signal | DAQ terminal |
|---|---|---|---|---|
| 1X | CH0H | | 3X | CH6H |
| 1Y | CH1H | | 3Y | CH7H |
| 1Z | CH2H | | 3Z | CH0L |
| 2X | CH3H | | 4X | CH1L |
| 2Y | CH4H | | 4Y | CH2L |
| 2Z | CH5H | | 4Z | CH3L |

Each signal's ground wire goes to an AGND terminal. Outputs for loopback testing: AOUT0
(pin 13) and AOUT1 (pin 15), with AGND on pin 14.

---

# Troubleshooting

**`uv: command not found` (or "not recognized")**
Close the terminal and open a new one. If that doesn't help, repeat step 2.

**Status says `No DAQ: ...`**
- Check the USB cable is plugged in and the DAQ's light is on.
- **Windows:** did you do the Zadig step? It has to be done on every computer.
- **Linux:** did you run the two udev commands and then unplug and replug the DAQ?
- Close any other program that uses the DAQ (InstaCal, DAQami, another copy of this app).
- Unplug the DAQ, wait 5 seconds, plug it back in, and start the app again.

**`ERROR ...` in the status line during a capture**
Unplug and replug the DAQ, then restart the app. Your earlier captures are already saved.

**Checking a new computer can keep up with the DAQ**
With the DAQ plugged in (and the app closed), run
```
uv run capture_check.py
```
It takes a few full captures while deliberately keeping the computer busy, saves
each one and reads it back. It should finish with `PASS`.

**Something else went wrong in `uv sync`**
Make sure you're in the project folder. Type `ls` (Mac/Linux) or `dir` (Windows) and
you should see `forceplate_gui.py` in the list. Then run `uv sync` again.

## Getting the latest version
- **If you used git:** `git pull`, then `uv sync`.
- **If you downloaded a ZIP:** download the new ZIP and replace the old project
  folder. Your data are in the data folder (Documents/ForcePlate data), not in
  the project folder, so they're safe.

---

# For developers

Other programs in this repository (run them the same way, from the project folder):
- `uv run loopback_live.py`: live plot of all 12 inputs while AOUT0 plays a sine (`--help` for options)
- `uv run capture_check.py`: back-to-back full captures under load, to check a computer keeps up
- `uv run loopback`: step voltages on AOUT0/1, read back on CH0/1
- `uv run sinusoid`: software-timed antiphase sinusoid loopback
- `uv run sinusoid-scan`: hardware-paced antiphase sinusoid loopback
- `uv run compare-uldaq`: cross-check our driver against MCC's uldaq (optional, needs `uv sync --extra compare`)
- `./sinusoid_loopback`: hardware-paced loopback via libuldaq (C++, see below)

Wiring for these loopback tests: AOUT0 (pin 13) → CH0 (pin 1), AOUT1 (pin 15) → CH1 (pin 4),
AGND (pin 14) → AGND (pin 3).

## The driver
The Python code talks to the DAQ directly over USB with its own driver (`mcc1608g/`).
It needs no MCC libraries, kernel drivers or InstaCal. On macOS, libusb ships inside the `libusb-package` wheel.

```python
from mcc1608g import USB1608G, AiMode, AiRange

with USB1608G.open() as daq:
    daq.a_out(0, 2.5)                                   # AOUT0 = 2.5 V
    v = daq.a_in(0, AiMode.SINGLE_ENDED, AiRange.BIP10V)

    # Hardware paced: the device's own clocks time every sample.
    ai = daq.a_in_scan(0, 1, AiMode.SINGLE_ENDED, AiRange.BIP10V,
                       samples_per_channel=10000, rate=10000)
    ao = daq.a_out_scan(0, 1, waveform, rate=10000)      # waveform: (N, 2) volts
    data = ai.read_all()                                 # (10000, 2) volts
    ao.wait()
```

`mcc1608g/protocol.py` holds every command code, status bit and EEPROM
address. `mcc1608g/device.py` is the driver itself, in five short sections
(connect and firmware upload, control/EEPROM transfers, calibration, single
samples, scans). The FPGA bitstream in `mcc1608g/firmware/` is the one MCC
ships in libuldaq. The device loses it at power-off, so the driver uploads
it on the first connection after power-up, which takes about a second.

The calibration app is `forceplate_gui.py`, and the session files and fit are in
`forceplate.py`. In single-ended mode CHnH is AI channel n and CHnL is n + 8, so one
scan of channels 0–11 returns 1X 1Y 1Z 2X … 4Z in order.

## Testing
Without hardware, the driver and the calibration app are exercised against a
software model of the device that speaks the same USB protocol:
```
uv run --with pytest pytest -q
```
With hardware and MCC's library installed, `uv sync --extra compare && uv run compare-uldaq`
runs the same voltage sweep through uldaq and through our driver and checks
that identity, calibration and readings agree.

<img src="images/pyfig-sinusoid_loopback.png" width="600">

## C++ quick start (linux)
Run the `install.txt` commands to install the ULDAQ drivers.

Build with
```
g++ -std=c++17 -O2 -o sinusoid_loopback sinusoid_loopback.cpp -luldaq -lm
```

Run with
```
./sinusoid_loopback
```

<img src="images/sinusoid_loopback.png" width="600">

## Todos
1. Testing on Windows.
2. GUI using imgui for Linux/Windows treadmill control.
3. Bullet-proofing design for VFD interaction.
