"""
deck_modes.py (strike tests read as deck motions) and its window, modes_gui.py,
on made-up captures: a hit that sets off a pitch mode and a bounce mode.
"""

import json
import sys

import numpy as np
import pytest

import deck_modes as dm
import forceplate as fp
import loadcell_cal as lc

RATE = 1000.0


def ringing_forces(seconds=5.0, hit_s=1.5, modes=((19.0, 0.012, "pitch", 60.0),
                                                  (29.0, 0.010, "Fz total", 200.0))):
    """(samples, 12) newtons: quiet, then each (Hz, damping, pattern, N per corner) rings."""
    layout = lc.load_layout()
    t = np.arange(int(seconds * RATE)) / RATE
    rng = np.random.default_rng(0)
    forces = rng.normal(0, 0.3, (len(t), 12))
    after = np.clip(t - hit_s, 0, None)
    for hz, zeta, pattern, amp in modes:
        ring = amp * np.exp(-zeta * 2 * np.pi * hz * after) * np.sin(2 * np.pi * hz * after)
        forces += np.outer(ring * (t >= hit_s), dm.z_weights(layout, dm.PATTERNS[pattern]))
    return t, forces, layout


def write_session(folder, forces, layout, notes="mallet back left"):
    """A session folder holding one tare and one hit, stored as volts like the GUI does."""
    folder.mkdir()
    volts = forces / layout.newtons_per_volt          # crosstalk left out: close enough here
    fp.save_capture_csv(folder / "trial_001.csv", volts[:1000] * 0, RATE)
    fp.save_capture_csv(folder / "trial_002.csv", volts, RATE)
    trials = [{"id": 1, "kind": "tare", "file": "trial_001.csv", "mean_v": [0.0] * 12, "notes": ""},
              {"id": 2, "kind": "load", "file": "trial_002.csv", "notes": notes}]
    (folder / "session.json").write_text(json.dumps({"trials": trials, "layout": layout.raw}))
    return folder


def test_corners_follow_positions_not_input_numbers():
    layout = lc.load_layout()
    assert dm.corner_inputs(layout) == [0, 1, 2, 3]
    swapped = lc.layout_for_wiring(layout, {"1": "401", "2": "402", "3": "406", "4": "405"})
    assert dm.corner_inputs(swapped) == [0, 1, 3, 2]       # back right is now input 4
    pitch = np.array(dm.z_weights(swapped, dm.PATTERNS["roll"]))
    assert pitch[[2, 5, 8, 11]].tolist() == [1, -1, 1, -1]


def test_patterns_keep_the_power_of_the_four_z():
    z = np.random.default_rng(1).normal(size=4)
    parts = [(z @ np.array(w)) ** 2 / 4 for w in dm.PATTERNS.values()]
    assert sum(parts) == pytest.approx((z ** 2).sum())


def test_formulas_are_written_from_the_patterns():
    assert dm.pattern_formulas()[:4] == ["Fz total = FL + FR + BR + BL", "pitch    = FL + FR − BR − BL",
                                         "roll     = FL − FR − BR + BL", "twist    = FL − FR + BR − BL"]


def test_sideways_partners_pick_out_stretch_and_yaw():
    layout = lc.load_layout()
    signals = {s.name: s for s in dm.standard_signals(layout)["Sideways partners (X, Y)"]}
    arm = layout.positions - layout.positions.mean(0)
    turn, along, across, push = (np.zeros((1, 12)) for _ in range(4))
    turn[0, 0::3], turn[0, 1::3] = -arm[:, 1], arm[:, 0]        # every corner pushed round the middle
    along[0, 1::3] = np.sign(arm[:, 1]) * -10                   # front pushed back, back pushed forward
    across[0, 0::3] = np.sign(arm[:, 0]) * -10
    push[0, 0::3], push[0, 1::3] = 7.0, -3.0                    # the whole treadmill pushed sideways
    read = lambda f: {k: round(float(s.series(f)[0]), 6) for k, s in signals.items()}   # noqa: E731
    assert read(push) == {"Y: front − back": 0, "X: left − right": 0, "yaw": 0}
    assert read(along) == {"Y: front − back": 40, "X: left − right": 0, "yaw": 0}
    assert read(across) == {"Y: front − back": 0, "X: left − right": 40, "yaw": 0}
    assert read(turn)["yaw"] > 0 and read(turn)["Y: front − back"] == 0
    assert dm.horizontal_formulas(layout)[:2] == ["Y: front − back = FLy + FRy − BRy − BLy",
                                                  "X: left − right = FLx − FRx − BRx + BLx"]


def test_filters_pass_and_stop_without_a_time_shift():
    t = np.arange(4000) / RATE
    slow, fast = np.sin(2 * np.pi * 5 * t), np.sin(2 * np.pi * 60 * t)
    x = (slow + fast)[:, None]
    mid = slice(500, 3500)
    # A 4th-order edge still takes a few percent off a signal an octave or less away
    assert np.abs(dm.zero_phase_filter(x, RATE, "low-pass", hi=20)[mid, 0] - slow[mid]).max() < 0.02
    assert np.abs(dm.zero_phase_filter(x, RATE, "high-pass", lo=20)[mid, 0] - fast[mid]).max() < 0.01
    assert np.abs(dm.zero_phase_filter(x, RATE, "band-pass", 40, 80)[mid, 0] - fast[mid]).max() < 0.08
    assert np.abs(dm.zero_phase_filter(x, RATE, "notch", 55, 65)[mid, 0] - slow[mid]).max() < 0.01
    assert dm.zero_phase_filter(x, RATE, "none") is x
    with pytest.raises(ValueError):
        dm.zero_phase_filter(x, RATE, "band-pass", 80, 40)


def test_find_strike_ignores_a_small_bump_and_a_quiet_capture():
    t, forces, _ = ringing_forces()
    forces[300:310, 8] += 20.0                 # someone brushes the deck first
    assert abs(dm.find_strike(forces) - 1500) <= 3
    quiet = np.random.default_rng(2).normal(0, 0.3, (5000, 12))
    with pytest.raises(ValueError):
        dm.find_strike(quiet)


def test_modes_are_named_with_their_damping():
    t, forces, layout = ringing_forces()
    k, a, b = dm.ring_window(forces, RATE)
    modes = dm.modes_in(forces[a:b], RATE, layout)
    assert [round(m["hz"]) for m in modes] == [19, 29]
    pitch, bounce = modes
    assert pitch["motion"].startswith("pitch") and pitch["shares"]["pitch"] > 0.95
    assert bounce["motion"].startswith("bounce") and bounce["shares"]["Fz total"] > 0.95
    assert pitch["damping"] == pytest.approx(0.012, rel=0.25)
    assert bounce["damping"] == pytest.approx(0.010, rel=0.25)
    # In phase at every corner for the bounce; front against back for the pitch
    assert np.abs(bounce["corner_phase_deg"]).max() < 5
    assert np.abs(np.abs(pitch["corner_phase_deg"][2:]) - 180).max() < 5
    assert len(dm.mode_table(modes)) == 4


def test_fz_band_fractions_put_the_bounce_in_its_band():
    t, forces, layout = ringing_forces()
    k, a, b = dm.ring_window(forces, RATE)
    fz = dm.standard_signals(layout)["Whole treadmill"][0]
    fractions = dict(zip(dm.BANDS, dm.band_fractions(forces[a:b], RATE, fz)))
    assert fractions[(25, 35)] > 0.95          # the pitch mode cancels in the total


def test_cell_norm_keeps_modes_at_their_own_frequency():
    t, forces, layout = ringing_forces(modes=((29.0, 0.01, "Fz total", 200.0),))
    norm = next(s for s in dm.standard_signals(layout)["Each cell"] if s.name == "1|F|")
    f, spec = dm.spectrum(forces[1525:2525], RATE)
    assert f[np.argmax(norm.amplitude(spec))] == pytest.approx(29.0, abs=0.5)
    pair = dm.pair_signal(layout, 3, 4, "Z", -1)
    assert pair.name == "3Z − 4Z" and np.abs(pair.series(forces)).max() < 5    # they move together


def test_session_report(tmp_path):
    t, forces, layout = ringing_forces()
    folder = write_session(tmp_path / "20260101-000000", forces, layout)
    result = dm.analyse_session(folder)
    assert [h["entry"]["id"] for h in result["hits"]] == [2]
    assert [e["id"] for e, _ in result["skipped"]] == [1]
    assert dm.default_report_name(folder) == "20260101-000000_strike_report.pdf"
    assert [round(g["hz"]) for g in dm.modes_across(result["hits"])] == [19, 29]
    assert any("bounce" in line for line in dm.summary_lines(result))
    assert result["static_xyz"].shape == (4, 3)
    assert result["static_xyz"][:, 2].tolist() == list(result["static"].values())
    table = dm.resting_channel_lines(np.arange(12.0).reshape(4, 3))      # FL, FR, BR, BL rows
    assert table[2].split() == ["front", "0", "1", "2", "3", "4", "5"]   # front left, front right
    assert table[3].split() == ["back", "9", "10", "11", "6", "7", "8"]  # back left, back right
    assert table[4].split() == ["total", "18", "22", "26"]
    assert dm.resting_load_short(dict(zip(dm.CORNERS, [1, 2, 3, 4]))).splitlines()[1].startswith("BL     4   BR     3")
    out = dm.write_report(result, tmp_path / "out" / "report.pdf")
    assert out.read_bytes().startswith(b"%PDF") and out.stat().st_size > 20_000


@pytest.fixture
def window(tmp_path, monkeypatch):
    tk = pytest.importorskip("tkinter")
    import modes_gui

    monkeypatch.setattr(modes_gui, "STATE_FILE", tmp_path / "state.json")

    t, forces, layout = ringing_forces()
    folder = write_session(tmp_path / "20260101-000000", forces, layout)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip(f"no display: {sys.exc_info()[1]}")
    root.withdraw()
    win = modes_gui.ModesWindow(root, layout, folder)
    win.withdraw()
    yield win
    root.destroy()


def test_window_opens_on_the_hit_and_lists_its_modes(window):
    assert [t["id"] for t in window.trials] == [2]       # the tare has no hit: not listed
    assert "1 capture without a hit is not listed" in window.status.cget("text")
    assert "kg" in window.rest_label.cget("text")
    window.only_hits.set(False)
    window._list_trials()
    assert [t["id"] for t in window.trials] == [1, 2] and window.entry["id"] == 2
    assert abs(window.hit - 1500) <= 3
    assert [round(m["hz"]) for m in window.modes] == [19, 29]
    assert "bounce" in window.table.get("1.0", "end")
    assert sorted(s.name for s in window.checked()) == sorted(
        ["Fz total", "pitch", "roll", "twist", "Y: front − back", "X: left − right", "yaw"])


def test_window_filter_pair_and_dragged_window(window, tmp_path):
    window.filter_kind.set("notch")
    window.filter_lo.set("27")
    window.filter_hi.set("31")
    window._redraw()
    left = {round(m["hz"]): m["amp_n"]["Fz total"] for m in window.modes}
    assert 19 in left and 29 not in left and max(left.values()) < 80   # the 800 N bounce is gone
    window.filter_hi.set("oops")
    window._redraw()                                             # a bad number: unfiltered, no crash
    assert "not applied" in window.status.cget("text")
    window.filter_kind.set("none")

    window.pair_a.set("3")
    window.pair_b.set("4")
    window.pair_sign.set("+")
    window._add_pair()
    assert "3Z + 4Z" in [s.name for s in window.checked()]

    window._on_span(2.0, 2.8)
    assert window.window == (2000, 2800)
    window._on_span(2.0, 2.01)                                   # too short a drag: kept as it was
    assert window.window == (2000, 2800)
    window._window_all()
    assert window.window == (0, 5000)

    for ending in ("eps", "pdf", "svg", "png"):
        window.save_figure(tmp_path / f"plots.{ending}")
        assert (tmp_path / f"plots.{ending}").stat().st_size > 5000
    assert window.write_report(tmp_path / "report.pdf").exists()


def test_last_session_is_remembered(tmp_path, monkeypatch):
    import modes_gui

    monkeypatch.setattr(modes_gui, "STATE_FILE", tmp_path / "state.json")
    assert modes_gui.last_session() is None                 # no file yet
    modes_gui.remember_session(None)                        # nothing was open: nothing written
    assert not modes_gui.STATE_FILE.exists()
    modes_gui.remember_session(tmp_path)
    assert modes_gui.last_session() == tmp_path
    modes_gui.remember_session(tmp_path / "deleted since")
    assert modes_gui.last_session() is None                 # the folder has gone
    modes_gui.STATE_FILE.write_text("not json")
    assert modes_gui.last_session() is None


def test_window_reopens_on_the_session_it_was_closed_on(window):
    import modes_gui

    folder, root, layout = window.folder, window.master, window.layout
    window.close()
    assert modes_gui.last_session() == folder
    again = modes_gui.ModesWindow(root, layout)             # no folder given
    again.withdraw()
    assert again.folder == folder and again.entry["id"] == 2
