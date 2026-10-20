"""
Force-plate calibration: the fit on synthetic data, and the GUI end to end
against the fake device from test_driver_fake_device.py (AOUT0/1 looped back
to CH0/1, every other input at 0 V).

Run:  uv run --with pytest pytest -q
"""

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
        x, y = rng.uniform(0, 1.6), rng.uniform(0, 0.6)
        for direction in fp.DIRECTIONS:
            fields = fp.load_trial_entry(direction, x, y, 0.0, 20.0)
            session.add_trial(capture(np.array(fields["known_load"])), 1000.0, "load", **fields)
    return session


def test_known_load_moments():
    load = fp.known_load("down", 100.0, 0.5, 0.2, 0.0)
    np.testing.assert_allclose(load[:3], [0, 0, -100])
    np.testing.assert_allclose(load[3:], [-20.0, 50.0, 0.0])   # r x F
    assert fp.centre_of_pressure(load) == pytest.approx((0.5, 0.2))
    # Forward is -x: a forward pull 0.2 m right of the origin gives Mz = -y*Fx.
    np.testing.assert_allclose(fp.known_load("forward", 10.0, 0.3, 0.2, 0.0),
                               [-10, 0, 0, 0, 0, 2.0])
    # A rightward pull on the substructure, 0.1 m below the deck surface.
    np.testing.assert_allclose(fp.known_load("right", 10.0, 1.0, 0.3, -0.1),
                               [0, 10, 0, 1.0, 0, 10.0])


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
    fields = fp.load_trial_entry("down", 0, 0, 0, 10)
    session.add_trial(np.zeros((10, 12)), 1000.0, "load", **fields)
    assert session.trials[0]["tare_id"] is None
    ids, _, _ = fp.gather(session)
    assert ids == []


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
    monkeypatch.setattr(gui, "OUT_DIR", tmp_path / "calib_data")
    # Dialogs would block the test; answer "yes" to every question.
    monkeypatch.setattr(gui.messagebox, "askyesno", lambda *a, **k: True)
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *a, **k: pytest.fail(str(a)))

    try:
        window = gui.App(USB1608G(fake))
    except tk.TclError:
        pytest.skip("no display")
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

    sx, sy = app._to_screen(0.4, 0.1)
    app._on_deck_click(SimpleNamespace(x=sx, y=sy))
    assert app.location == pytest.approx((0.4, 0.1), abs=0.005)
    app._on_deck_click(SimpleNamespace(x=2, y=2))         # outside the deck: ignored
    assert app.location == pytest.approx((0.4, 0.1), abs=0.005)
    app.direction.set("left")
    app.mass.set("12.5")
    capture(app, "load")

    meta = json.loads((app.session.folder / "session.json").read_text())
    tare, load = meta["trials"]
    assert tare["kind"] == "tare" and load["kind"] == "load"
    assert load["tare_id"] == tare["id"]
    assert load["direction"] == "left" and load["unit_vector"] == [0.0, -1.0, 0.0]
    assert load["z_m"] == 0.0
    assert load["force_n"] == pytest.approx(12.5 * fp.GRAVITY)
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
    app.live.set(False)
    app._on_live_toggle()


def test_gui_deck_size(app, monkeypatch):
    app.length_var.set("2.0")
    app.width_var.set("0.8")
    app._set_deck_size()
    assert (app.deck_length, app.deck_width) == (2.0, 0.8)
    sx, sy = app._to_screen(2.0, 0.8)                   # back-right corner is clickable
    app._on_deck_click(SimpleNamespace(x=sx, y=sy))
    assert app.location == pytest.approx((2.0, 0.8), abs=0.005)

    capture(app, "tare")
    assert app.session.meta["deck"]["length_m"] == 2.0
    errors = []
    import forceplate_gui as gui
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *a, **k: errors.append(a))
    app.length_var.set("3.0")
    app._set_deck_size()                                # refused once captures exist
    assert errors and app.deck_length == 2.0 and app.length_var.get() == "2"


def test_gui_rejects_height_above_deck(app, monkeypatch):
    import forceplate_gui as gui
    errors = []
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *a, **k: errors.append(a))
    sx, sy = app._to_screen(0.5, 0.3)
    app._on_deck_click(SimpleNamespace(x=sx, y=sy))
    app.height.set("0.05")
    app._capture("load")
    assert errors and not app.busy
    app.height.set("-0.12")
    capture(app, "load")
    assert app.session.trials[-1]["z_m"] == -0.12
