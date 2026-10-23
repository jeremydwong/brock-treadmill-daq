"""
Force-plate calibration: the fit on synthetic data, and the GUI end to end
against the fake device from test_driver_fake_device.py (AOUT0/1 looped back
to CH0/1, every other input at 0 V).

Run:  uv run --with pytest pytest -q
"""

import csv
import sys
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
import usb.util

import forceplate as fp
from mcc1608g import USB1608G
from test_driver_fake_device import FakeUsbDevice


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def synthetic_session(tmp_path, true_matrix, n_locations=8, noise_v=0.0, seed=0):
    """Build a session whose CSVs are what a plate with `true_matrix` would read."""
    rng = np.random.default_rng(seed)
    sensitivity = np.linalg.pinv(true_matrix)          # volts per unit load, (12, 6)
    session = fp.Session.create(tmp_path / "s", 1.6, 0.6)
    n = 200
    offset = rng.normal(0, 0.05, 12)                   # amplifier zero offsets

    def capture(load):
        v = offset + sensitivity @ load
        return v + rng.normal(0, noise_v, (n, 12)) if noise_v else np.tile(v, (n, 1))

    session.add_trial(capture(np.zeros(6)), 1000.0, "tare")
    for _ in range(n_locations):
        x, y = rng.uniform(0, 0.6), rng.uniform(0, 1.6)
        for direction in fp.DIRECTIONS:
            fields = fp.load_trial_entry(direction, x, y, 0.0, 20.0 * fp.GRAVITY)
            session.add_trial(capture(np.array(fields["known_load"])), 1000.0, "load", **fields)
    return session


def test_known_load_moments():
    # Frame: +x right, +y back, +z down. 100 N pressing down, 0.2 m right, 0.5 m back.
    load = fp.known_load("down", 100.0, 0.2, 0.5, 0.0)
    np.testing.assert_allclose(load[:3], [0, 0, 100])
    np.testing.assert_allclose(load[3:], [50.0, -20.0, 0.0])   # r x F
    assert fp.centre_of_pressure(load) == pytest.approx((0.2, 0.5))
    # Forward is -y: a forward pull 0.2 m right of the left edge gives Mz = x*Fy.
    np.testing.assert_allclose(fp.known_load("forward", 10.0, 0.2, 0.3, 0.0),
                               [0, -10, 0, 0, 0, -2.0])
    # A rightward pull on the substructure, 0.1 m below the deck surface.
    np.testing.assert_allclose(fp.known_load("right", 10.0, 0.3, 1.0, 0.1),
                               [10, 0, 0, 0, 1.0, -10.0])


def test_frame_is_right_handed():
    right, back, down = (np.array(fp.DIRECTIONS[d]) for d in ("right", "back", "down"))
    np.testing.assert_allclose(np.cross(right, back), down)


def test_unit_conversions_in_entry():
    e = fp.load_trial_entry("down", 0.3048, 0.6096, 0.0, 10 * fp.FORCE_UNITS["lb"],
                            {"load": 10, "load_unit": "lb", "right": 1, "back": 2,
                             "below": 0, "length_unit": "ft"})
    assert e["mass_kg"] == pytest.approx(4.5359237)
    assert e["force_n"] == pytest.approx(44.48, abs=0.01)
    assert fp.FORCE_UNITS["lbf"] == pytest.approx(fp.FORCE_UNITS["lb"], rel=1e-6)


def test_old_frame_session_is_refused(tmp_path):
    session = fp.Session.create(tmp_path / "s", 1.6, 0.6)
    meta = json.loads((session.folder / "session.json").read_text())
    del meta["frame_version"]
    (session.folder / "session.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="older deck frame"):
        fp.Session.load(session.folder)


def test_log_csv_and_locked_log(tmp_path, monkeypatch):
    session = fp.Session.create(tmp_path / "s", 1.6, 0.6)
    session.add_trial(np.zeros((10, 12)), 1000.0, "tare", notes="empty deck")
    fields = fp.load_trial_entry("down", 0.3, 1.2, 0.0, 196.1,
                                 {"load": 20, "load_unit": "kg", "right": 0.3, "back": 1.2,
                                  "below": 0, "length_unit": "m"})
    session.add_trial(np.ones((10, 12)), 1000.0, "load", notes="café, plates 1+2", **fields)
    with open(session.folder / fp.LOG_NAME, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["kind"] for r in rows] == ["tare", "load"]
    assert rows[0]["notes"] == "empty deck"
    assert rows[1]["notes"] == "café, plates 1+2"
    assert rows[1]["right_of_left_edge"] == "0.3" and rows[1]["back_from_front_edge"] == "1.2"
    assert rows[1]["deck_front_to_back_m"] == "1.6" and rows[1]["load_unit"] == "kg"
    assert rows[1]["date"] and rows[1]["time"]

    # Excel on Windows locks an open CSV: the capture must still be saved.
    def locked():
        raise PermissionError(13, "Permission denied")
    monkeypatch.setattr(session, "write_log", locked)
    session.add_trial(np.zeros((10, 12)), 1000.0, "tare")
    assert session.log_warning and "Excel" in session.log_warning
    assert len(fp.Session.load(session.folder).trials) == 3


def test_capture_csv_round_trip(tmp_path):
    volts = np.random.default_rng(0).uniform(-10, 10, (5000, 12))
    fp.save_capture_csv(tmp_path / "t.csv", volts, 1000.0)
    data = np.loadtxt(tmp_path / "t.csv", delimiter=",", skiprows=1)
    assert data.shape == (5000, 13)
    np.testing.assert_allclose(data[:, 1:], volts, atol=1e-6)
    assert not list(tmp_path.glob("*.tmp"))


def test_fit_recovers_true_matrix(tmp_path):
    true_matrix = np.random.default_rng(1).normal(0, 50, (6, 12))
    session = synthetic_session(tmp_path, true_matrix)
    ids, signals, known = fp.gather(session)
    assert len(ids) == 8 * len(fp.DIRECTIONS)
    fit = fp.fit_calibration(ids, signals, known)
    # CSVs hold volts to 1 uV, so the fit is exact to about that.
    np.testing.assert_allclose(fit.predicted, fit.known, atol=0.01)

    json_path, csv_path = fp.save_fit(fit, session)
    saved = json.loads(json_path.read_text())
    assert saved["rows"] == fp.LOAD_NAMES and saved["columns"] == fp.SIGNAL_NAMES
    assert np.array(saved["matrix"]).shape == (6, 12)
    assert csv_path.read_text().splitlines()[0] == "load," + ",".join(fp.SIGNAL_NAMES)


def test_fit_with_noise_still_predicts_loads(tmp_path):
    true_matrix = np.random.default_rng(2).normal(0, 50, (6, 12))
    session = synthetic_session(tmp_path, true_matrix, n_locations=10, noise_v=0.002)
    fit = fp.fit_calibration(*fp.gather(session))
    assert fit.rms_error[:3].max() < 1.0      # newtons, on ~196 N loads
    assert fit.rms_error[3:].max() < 0.5      # newton-metres


def test_loads_without_tare_are_skipped(tmp_path):
    session = fp.Session.create(tmp_path / "s", 1.6, 0.6)
    fields = fp.load_trial_entry("down", 0, 0, 0, 98.0)
    session.add_trial(np.zeros((10, 12)), 1000.0, "load", **fields)
    assert session.trials[0]["tare_id"] is None
    ids, _, _ = fp.gather(session)
    assert ids == []


def test_layout_file_matches_channel_order():
    layout = fp.load_layout()
    assert layout.positions.shape == (4, 2) and layout.newtons_per_volt.shape == (12,)
    # Every cell sits on the deck it describes.
    assert (layout.positions >= 0).all()
    assert (layout.positions[:, 0] <= layout.deck_width_m).all()
    assert (layout.positions[:, 1] <= layout.deck_length_m).all()


def test_cop_from_cell_geometry():
    layout = fp.load_layout()
    zero = np.full(12, 0.3)
    z = [fp.SIGNAL_NAMES.index(f"{c}Z") for c in "1234"]
    # Equal volts on all four Z channels: the middle of the four cells.
    volts = zero.copy()
    volts[z] += 0.1
    x, y, total = fp.cop_from_cells(fp.cell_forces(volts, zero, layout), layout)
    assert (x, y) == pytest.approx(layout.positions.mean(axis=0))
    assert total == pytest.approx(0.1 * layout.newtons_per_volt[z].sum())
    # All of it on cell 3: right on top of cell 3.
    volts = zero.copy()
    volts[z[2]] += 0.2
    x, y, _ = fp.cop_from_cells(fp.cell_forces(volts, zero, layout), layout)
    assert (x, y) == pytest.approx(layout.positions[2])
    # X and Y forces don't move it, and too little load gives no answer.
    volts[[0, 1, 9, 10]] += 1.0
    assert fp.cop_from_cells(fp.cell_forces(volts, zero, layout), layout)[:2] \
        == pytest.approx(layout.positions[2])
    assert fp.cop_from_cells(np.zeros(12), layout) is None


def test_remove_trial_keeps_csv(tmp_path):
    session = fp.Session.create(tmp_path / "s", 1.6, 0.6)
    session.add_trial(np.zeros((10, 12)), 1000.0, "tare")
    session.remove_trial(1)
    assert session.trials == []
    assert (session.folder / "trial_001.csv.removed").exists()
    assert fp.Session.load(session.folder).trials == []


# ---------------------------------------------------------------------------
# GUI end to end with the fake device
# ---------------------------------------------------------------------------

@pytest.fixture
def app(monkeypatch, tmp_path):
    tk = pytest.importorskip("tkinter")
    import forceplate_gui as gui

    fake = FakeUsbDevice()
    monkeypatch.setattr(usb.util, "claim_interface", lambda d, i: None)
    monkeypatch.setattr(usb.util, "release_interface", lambda d, i: None)
    monkeypatch.setattr(usb.util, "dispose_resources", lambda d: None)
    monkeypatch.setattr(usb.util, "get_string", lambda d, i: d.serial_number)
    monkeypatch.setattr(gui, "DEFAULT_DATA_DIR", tmp_path / "calib_data")
    monkeypatch.setattr(gui, "SETTINGS", tmp_path / "settings.json")
    # Dialogs would block the test; answer "yes" to every question.
    monkeypatch.setattr(gui.messagebox, "askyesno", lambda *a, **k: True)
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *a, **k: pytest.fail(str(a)))

    try:
        window = gui.App(USB1608G(fake))
    except tk.TclError:
        pytest.skip(f"no display: {sys.exc_info()[1]}")
    window.withdraw()
    yield window
    window._on_close()


def pump(app, until, timeout=20.0):
    deadline = time.monotonic() + timeout
    while not until():
        assert time.monotonic() < deadline, f"timed out; status: {app.status.cget('text')}"
        app.update()
        time.sleep(0.01)


def capture(app, kind):
    app._capture(kind)
    pump(app, lambda: not app.busy)
    assert not app.status.cget("text").startswith("ERROR"), app.status.cget("text")


def test_gui_tare_load_and_fit(app):
    capture(app, "tare")
    assert app.session is not None

    assert app.session.folder.parent == app.data_dir

    sx, sy = app._to_screen(0.1, 0.4)
    app._on_deck_click(SimpleNamespace(x=sx, y=sy))
    assert app.location == pytest.approx((0.1, 0.4), abs=0.005)
    assert float(app.x_var.get()) == pytest.approx(0.1, abs=0.005)    # click fills the boxes
    app._on_deck_click(SimpleNamespace(x=2, y=2))         # outside the deck: ignored
    assert app.location == pytest.approx((0.1, 0.4), abs=0.005)
    app.direction.set("left")
    app.load_amount.set("12.5")
    app.notes.set("first load")
    capture(app, "load")

    meta = json.loads((app.session.folder / "session.json").read_text())
    tare, load = meta["trials"]
    assert tare["kind"] == "tare" and load["kind"] == "load"
    assert load["tare_id"] == tare["id"]
    assert load["direction"] == "left" and load["unit_vector"] == [-1.0, 0.0, 0.0]
    assert load["z_m"] == 0.0 and load["notes"] == "first load"
    assert load["force_n"] == pytest.approx(12.5 * fp.GRAVITY)
    assert load["mass_kg"] == pytest.approx(12.5)
    assert (app.session.folder / fp.LOG_NAME).exists()
    assert [c["signal"] for c in meta["channels"]] == fp.SIGNAL_NAMES

    data = np.loadtxt(app.session.folder / load["file"], delimiter=",", skiprows=1)
    assert data.shape == (int(fp.CAPTURE_SECONDS * fp.SAMPLE_RATE), 13)
    assert app.trial_list.size() == 2


def test_gui_loopback_fixed_and_sine(app):
    app.loopback.set(True)
    app._on_loopback_toggle()
    app.ao_channel.set("AOUT1 (pin 15)")
    app.wired_to.set(app._channel_label(1))      # fake loops AOUT1 -> CH1H (1Y)

    app.ao_mode.set("fixed")
    app.ao_volts.set("1.5")
    app._ao_start()
    pump(app, lambda: "holding" in app.status.cget("text"))
    capture(app, "load")                           # becomes a loopback capture
    entry = app.session.trials[-1]
    assert entry["kind"] == "loopback" and entry["loopback"]["ao_channel"] == 1
    assert entry["mean_v"][1] == pytest.approx(1.5, abs=0.002)

    app.ao_mode.set("sine")
    app.ao_volts.set("2.0")
    app.ao_freq.set("5")
    app._ao_start()
    pump(app, lambda: "sine" in app.status.cget("text"))
    capture(app, "load")
    data = np.loadtxt(app.session.folder / app.session.trials[-1]["file"],
                      delimiter=",", skiprows=1)
    assert data[:, 2].max() == pytest.approx(2.0, abs=0.01)
    assert data[:, 2].min() == pytest.approx(-2.0, abs=0.01)

    app._ao_off()
    pump(app, lambda: "off" in app.status.cget("text"))
    assert fp.gather(app.session)[0] == []        # loopback captures never reach the fit


def test_gui_live_view_fills_buffer(app):
    # The fake returns samples instantly; pace it like the real 1 kHz clock.
    scan = app.worker._scan
    def paced_scan(samples):
        time.sleep(samples / fp.SAMPLE_RATE)
        return scan(samples)
    app.worker._scan = paced_scan

    app.live.set(True)
    app._on_live_toggle()
    pump(app, lambda: not np.isnan(app.live_buffer[-1]).any())
    assert "tare first" in app.cop_label.cget("text")     # no tare yet: no COP
    assert "NOT TARED" in app.showing_label.cget("text")
    app.live.set(False)
    app._on_live_toggle()


def test_gui_cop_from_live_volts(app):
    capture(app, "tare")                                   # fake reads 0 V everywhere
    assert app.zero_v is not None and app.cop is None
    volts = np.zeros(12)
    volts[[fp.SIGNAL_NAMES.index(f"{c}Z") for c in "1234"]] = [0.1, 0.1, 0.3, 0.3]
    app._show_cop(volts, app.zero_v, "Live")
    pos = app.layout.positions
    expected = (pos * np.array([[0.1], [0.1], [0.3], [0.3]])).sum(axis=0) / 0.8
    assert app.cop == pytest.approx(tuple(expected))
    assert app.canvas.find_withtag("cop")                  # drawn on the deck
    app._redraw_deck()
    assert app.canvas.find_withtag("cop")                  # and survives a redraw


def test_gui_cell_plots_sit_at_their_corners(app):
    # Cell 1 is front left: top-left of the grid; cell 3 back right: bottom-right.
    info = [c.get_tk_widget().grid_info() for c in app.plot_canvases]
    assert (int(info[0]["row"]), int(info[0]["column"])) == (0, 0)
    assert (int(info[1]["row"]), int(info[1]["column"])) == (0, 2)
    assert (int(info[2]["row"]), int(info[2]["column"])) == (1, 2)
    assert (int(info[3]["row"]), int(info[3]["column"])) == (1, 0)


def test_gui_deck_size(app, monkeypatch):
    app.length_var.set("2.0")
    app.width_var.set("0.8")
    app._set_deck_size()
    assert (app.deck_length, app.deck_width) == (2.0, 0.8)
    sx, sy = app._to_screen(0.8, 2.0)                   # back-right corner is clickable
    app._on_deck_click(SimpleNamespace(x=sx, y=sy))
    assert app.location == pytest.approx((0.8, 2.0), abs=0.005)

    capture(app, "tare")
    assert app.session.meta["deck"]["length_m"] == 2.0
    errors = []
    import forceplate_gui as gui
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *a, **k: errors.append(a))
    app.length_var.set("3.0")
    app._set_deck_size()                                # refused once captures exist
    assert errors and app.deck_length == 2.0 and app.length_var.get() == "2"


def test_gui_rejects_negative_depth(app, monkeypatch):
    import forceplate_gui as gui
    errors = []
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *a, **k: errors.append(a))
    app.x_var.set("0.3")
    app.y_var.set("0.5")
    app.z_var.set("-0.05")
    app._capture("load")
    assert errors and not app.busy
    app.z_var.set("0.12")
    capture(app, "load")
    assert app.session.trials[-1]["z_m"] == 0.12


def test_gui_typed_location_in_feet_and_lbf(app):
    app.length_unit.set("ft")
    app._on_units_changed()
    assert app.length_var.get() == f"{round(1.6 / 0.3048, 3):g}"    # deck shown in feet
    app.x_var.set("1")
    app.y_var.set("4")
    assert app.location == pytest.approx((0.3048, 1.2192))
    app.y_var.set("99")                                  # off the deck
    assert app.location is None and str(app.capture_button.cget("state")) == "disabled"
    app.y_var.set("4")
    app.z_var.set("0.5")
    app.load_unit.set("lbf")
    app.load_amount.set("50")
    capture(app, "load")
    t = app.session.trials[-1]
    assert (t["x_m"], t["y_m"], t["z_m"]) == pytest.approx((0.3048, 1.2192, 0.1524))
    assert t["force_n"] == pytest.approx(50 * 4.4482216152605)
    assert "mass_kg" not in t
    assert t["entered"] == {"load": 50.0, "load_unit": "lbf", "right": 1.0, "back": 4.0,
                            "below": 0.5, "length_unit": "ft"}
    app._on_units_changed()                              # still metres underneath
    assert app.location == pytest.approx((0.3048, 1.2192))


def test_gui_save_failure_keeps_data(app, monkeypatch, tmp_path):
    import forceplate_gui as gui
    errors = []
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *a, **k: errors.append(a))
    monkeypatch.setattr(gui.Path, "home", classmethod(lambda cls: tmp_path))
    def broken(*a, **k):
        raise OSError("disk unplugged")
    monkeypatch.setattr(fp.Session, "create", broken)
    app._capture("tare")
    pump(app, lambda: not app.busy)
    assert errors and "disk unplugged" in errors[0][1]
    rescued = list(tmp_path.glob("forceplate_unsaved_*.csv"))
    assert len(rescued) == 1
    assert np.loadtxt(rescued[0], delimiter=",", skiprows=1).shape == (5000, 13)
