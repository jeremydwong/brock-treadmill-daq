"""
Exercise the driver against a software model of the USB-1608GX-2AO.

The fake speaks the same control-transfer and bulk-endpoint protocol the real
device does (see mcc1608g/protocol.py), with AOUT0/1 looped back to CH0/1
through a made-up calibration. This checks the whole path from volts in to
volts out without hardware: FPGA upload, EEPROM parsing, count/volt math,
scan configuration packing and bulk streaming.

Run:  uv run --with pytest pytest -q
"""

import struct
import threading

import numpy as np
import pytest
import usb.core
import usb.util

from mcc1608g import USB1608G, AiMode, AiRange, DaqError
from mcc1608g import protocol as P
from mcc1608g.device import FIRMWARE_DIR
from mcc1608g.protocol import Cmd

# Calibration the fake stores in its EEPROM and applies to its own converters.
FAKE_AI_CAL = {r: (1.0 + 0.001 * i, -12.0 - i) for i, r in enumerate(AiRange)}
FAKE_AO_CAL = [(0.998, 5.0), (1.002, -7.0)]
AI_CAL_DATE = (2024, 3, 15, 10, 30, 0)
AO_CAL_DATE = (2024, 3, 15, 11, 0, 0)


class FakeEndpoint:
    def __init__(self, address, max_packet):
        self.bEndpointAddress = address
        self.wMaxPacketSize = max_packet


class FakeUsbDevice:
    """Just enough of pyusb's Device for mcc1608g.device to drive."""

    idVendor = P.MCC_VENDOR_ID
    idProduct = 0x0136
    iSerialNumber = 3
    serial_number = "FAKE001"

    def __init__(self):
        self.fpga_loaded = False
        self.fpga_bytes = bytearray()
        self.config_mode = False
        self.eeprom = bytearray(0x8000)
        self.mem_addr = 0
        self.ai_config = bytes(16)
        self.ao_volts = [0.0, 0.0]
        self.ai_scan = None       # dict while a scan is configured
        self.ao_scan = None
        self.ao_fifo = bytearray()
        self.ao_fifo_lock = threading.Condition()
        self.configured = True
        self.stopped_commands = []
        self._fill_eeprom()

    # -- EEPROM contents --------------------------------------------------
    def _fill_eeprom(self):
        for i, rng in enumerate(AiRange):
            struct.pack_into("<ff", self.eeprom, P.EEPROM_AI_CAL_ADDR + 8 * i, *FAKE_AI_CAL[rng])
        for ch in range(2):
            struct.pack_into("<ff", self.eeprom, P.EEPROM_AO_CAL_ADDR + 8 * ch, *FAKE_AO_CAL[ch])
        struct.pack_into(">6H", self.eeprom, P.EEPROM_AI_CAL_DATE_ADDR, *AI_CAL_DATE)
        struct.pack_into(">6H", self.eeprom, P.EEPROM_AO_CAL_DATE_ADDR, *AO_CAL_DATE)

    # -- physics: what the ADC sees for a channel -------------------------
    def _ai_count(self, slot_byte):
        chan = slot_byte & 7
        rng = AiRange((slot_byte >> 3) & 3)
        mode = (slot_byte >> 5) & 3
        if mode == 2:
            chan += 8
        volts = self.ao_volts[chan] if chan < 2 else 0.0
        lsb = rng.span / 65536
        ideal = (volts - rng.minimum) / lsb
        slope, offset = FAKE_AI_CAL[rng]
        # Device reports raw counts; host undoes this with the EEPROM coefficients.
        raw = (ideal - offset) / slope
        return int(round(min(max(raw, 0), 65535)))

    def _ao_apply(self, channel, count):
        slope, offset = FAKE_AO_CAL[channel]
        ideal = (count - offset) / slope
        self.ao_volts[channel] = ideal * (20.0 / 65536) - 10.0

    # -- pyusb Device surface ---------------------------------------------
    def get_active_configuration(self):
        if not self.configured:
            raise usb.core.USBError("not configured")
        return [[FakeEndpoint(P.AI_SCAN_ENDPOINT, 512), FakeEndpoint(P.AO_SCAN_ENDPOINT, 512)]]

    def set_configuration(self):
        self.configured = True

    def clear_halt(self, ep):
        pass

    def ctrl_transfer(self, bmRequestType, bRequest, wValue=0, wIndex=0,
                      data_or_wLength=None, timeout=None):
        cmd = Cmd(bRequest)
        if bmRequestType == P.REQ_OUT:
            data = bytes(data_or_wLength or b"")
            self._handle_out(cmd, wValue, wIndex, data)
            return len(data)
        assert bmRequestType == P.REQ_IN
        return self._handle_in(cmd, wValue, wIndex, data_or_wLength)

    def _handle_out(self, cmd, w_value, w_index, data):
        if cmd == Cmd.FPGA_CONFIG:
            assert data == bytes([P.FPGA_UNLOCK])
            self.config_mode = True
        elif cmd == Cmd.FPGA_DATA:
            assert self.config_mode and len(data) <= 64
            self.fpga_bytes += data
            self.fpga_loaded = len(self.fpga_bytes) >= 73722
        elif cmd == Cmd.MEMORY_ADDRESS:
            (self.mem_addr,) = struct.unpack("<H", data)
        elif cmd == Cmd.AIN_CONFIG:
            assert len(data) == 16
            self.ai_config = data
        elif cmd == Cmd.AOUT:
            self._ao_apply(w_index, w_value)
        elif cmd == Cmd.AIN_SCAN_START:
            n, retrig, period, pkt, opts, _ = P.AinScanConfig.unpack(data)
            self.ai_scan = {"count": n, "period": period, "packet": pkt + 1, "sent": 0}
        elif cmd == Cmd.AOUT_SCAN_START:
            n, retrig, period, opts = P.AoutScanConfig.unpack(data)
            self.ao_scan = {"count": n, "period": period, "mask": opts & 3, "played": 0}
        elif cmd in (Cmd.AIN_SCAN_STOP, Cmd.AOUT_SCAN_STOP, Cmd.AIN_CLEAR_FIFO,
                     Cmd.AOUT_CLEAR_FIFO):
            self.stopped_commands.append(cmd)
            if cmd == Cmd.AOUT_CLEAR_FIFO:
                self.ao_fifo.clear()
        elif cmd == Cmd.BLINK_LED:
            pass
        else:
            raise AssertionError(f"unexpected OUT command {cmd}")

    def _handle_in(self, cmd, w_value, w_index, length):
        if cmd == Cmd.STATUS:
            s = P.STATUS_FPGA_LOADED if self.fpga_loaded else 0
            if self.ao_scan and self.ao_scan["played"] < self.ao_scan["count"]:
                s |= P.STATUS_AO_SCAN_RUNNING
            return struct.pack("<H", s)
        if not self.fpga_loaded:
            raise AssertionError(f"{cmd} before FPGA load")
        if cmd == Cmd.FPGA_VERSION:
            return struct.pack("<H", 0x0107)
        if cmd == Cmd.MEMORY:
            assert length <= 64
            out = bytes(self.eeprom[self.mem_addr:self.mem_addr + length])
            self.mem_addr += length
            return out
        if cmd == Cmd.AIN:
            slot = self.ai_config[0]
            assert slot & 7 == w_value % 8 and slot & 0x80, "single read must be a 1-slot list"
            return struct.pack("<H", self._ai_count(slot))
        raise AssertionError(f"unexpected IN command {cmd}")

    # -- bulk endpoints ----------------------------------------------------
    def write(self, ep, data, timeout=None):
        assert ep == P.AO_SCAN_ENDPOINT
        with self.ao_fifo_lock:
            self.ao_fifo += bytes(data)
            self.ao_fifo_lock.notify_all()
        return len(data)

    def read(self, ep, size, timeout=None):
        """Emulate the ADC: each scan samples the *current* AO values, which
        advance one DAC scan at a time while an AO scan is running."""
        assert ep == P.AI_SCAN_ENDPOINT and self.ai_scan
        slots = [b for b in self.ai_config[:16]]
        n_chan = next(i for i, b in enumerate(slots) if b & 0x80) + 1
        out = bytearray()
        while len(out) + 2 * n_chan <= size and self.ai_scan["sent"] < self.ai_scan["count"]:
            self._step_ao_scan()
            for b in slots[:n_chan]:
                out += struct.pack("<H", self._ai_count(b))
            self.ai_scan["sent"] += 1
            if len(out) >= 2 * self.ai_scan["packet"] * 1 and self.ai_scan["packet"] < 256:
                break   # slow mode: one packet per scan
        if not out:
            raise usb.core.USBTimeoutError("no data", 110, 110)
        return out

    def _step_ao_scan(self):
        if not self.ao_scan or self.ao_scan["played"] >= self.ao_scan["count"]:
            return
        chans = [c for c in range(2) if self.ao_scan["mask"] & (1 << c)]
        need = 2 * len(chans)
        with self.ao_fifo_lock:
            while len(self.ao_fifo) < need:
                assert self.ao_fifo_lock.wait(timeout=2.0), "AO feeder starved the fake FIFO"
            frame = bytes(self.ao_fifo[:need])
            del self.ao_fifo[:need]
        for c, count in zip(chans, struct.unpack(f"<{len(chans)}H", frame)):
            self._ao_apply(c, count)
        self.ao_scan["played"] += 1


@pytest.fixture
def daq(monkeypatch):
    fake = FakeUsbDevice()
    monkeypatch.setattr(usb.util, "claim_interface", lambda d, i: None)
    monkeypatch.setattr(usb.util, "release_interface", lambda d, i: None)
    monkeypatch.setattr(usb.util, "dispose_resources", lambda d: None)
    monkeypatch.setattr(usb.util, "get_string", lambda d, i: d.serial_number)
    dev = USB1608G(fake)
    dev.fake = fake
    yield dev
    dev.close()


def test_connect_uploads_firmware_and_reads_identity(daq):
    fake = daq.fake
    assert fake.fpga_loaded
    assert fake.fpga_bytes == (FIRMWARE_DIR / "USB_1608G_2.rbf").read_bytes()
    assert daq.product_name == "USB-1608GX-2AO"
    assert daq.has_ao
    assert daq.serial == "FAKE001"
    assert daq.fpga_version == 0x0107
    assert daq.ai_cal_date.year == 2024 and daq.ai_cal_date.hour == 10
    assert daq.ao_cal_date.minute == 0
    for rng in AiRange:
        assert daq.ai_calibration[rng].slope == pytest.approx(FAKE_AI_CAL[rng][0])
        assert daq.ai_calibration[rng].offset == pytest.approx(FAKE_AI_CAL[rng][1])
    assert daq.ao_calibration[1].offset == pytest.approx(-7.0)


def test_connect_skips_upload_when_fpga_already_loaded(monkeypatch):
    fake = FakeUsbDevice()
    fake.fpga_loaded = True
    monkeypatch.setattr(usb.util, "claim_interface", lambda d, i: None)
    monkeypatch.setattr(usb.util, "get_string", lambda d, i: d.serial_number)
    USB1608G(fake)
    assert fake.fpga_bytes == b""


@pytest.mark.parametrize("volts", [-9.0, -2.5, 0.0, 0.123, 4.99, 9.0])
def test_single_sample_loopback(daq, volts):
    for ch in (0, 1):
        daq.a_out(ch, volts)
        read = daq.a_in(ch, AiMode.SINGLE_ENDED, AiRange.BIP10V)
        # One LSB is 0.3 mV on the +/-10 V range; allow a couple.
        assert read == pytest.approx(volts, abs=0.001)


def test_a_in_other_ranges_use_their_own_calibration(daq):
    daq.a_out(0, 0.75)
    assert daq.a_in(0, AiMode.SINGLE_ENDED, AiRange.BIP1V) == pytest.approx(0.75, abs=1e-4)
    assert daq.a_in(0, AiMode.SINGLE_ENDED, AiRange.BIP2V) == pytest.approx(0.75, abs=2e-4)


def test_a_out_clips_to_range(daq):
    daq.a_out(0, 15.0)
    assert daq.a_in(0) == pytest.approx(10.0, abs=0.01)
    daq.a_out(0, -15.0)
    assert daq.a_in(0) == pytest.approx(-10.0, abs=0.01)


def test_ai_config_bytes_for_high_channels(daq):
    daq.a_in(9, AiMode.SINGLE_ENDED, AiRange.BIP5V)
    slot = daq.fake.ai_config[0]
    assert slot & 7 == 1            # 9 % 8
    assert (slot >> 3) & 3 == 1     # BIP5V
    assert (slot >> 5) & 3 == 2     # single-ended bank 8-15
    assert slot & 0x80              # last
    assert daq.fake.ai_config[1] == 0


def test_differential_channel_range_is_checked(daq):
    with pytest.raises(DaqError):
        daq.a_in(8, AiMode.DIFFERENTIAL, AiRange.BIP10V)


def test_scan_config_packing(daq):
    ai = daq.a_in_scan(0, 1, AiMode.SINGLE_ENDED, AiRange.BIP10V, 500, 10_000)
    cfg = daq.fake.ai_scan
    assert cfg["count"] == 500
    assert cfg["period"] == 6399                     # 64 MHz / 10 kHz - 1
    assert cfg["packet"] == 256                      # block mode: 512-byte packets
    assert ai.rate == pytest.approx(10_000)
    ai.stop()

    daq.a_in_scan(0, 3, AiMode.SINGLE_ENDED, AiRange.BIP10V, 10, 100)
    assert daq.fake.ai_scan["packet"] == 4           # slow mode: one scan per packet

    with pytest.raises(DaqError):
        daq.a_in_scan(0, 1, AiMode.SINGLE_ENDED, AiRange.BIP10V, 10, 400_000)


def test_hardware_paced_loopback_matches_commanded(daq):
    n = 6000                                          # bigger than the 4k AO FIFO
    t = np.arange(n) / 10_000
    commanded = np.column_stack([5 * np.sin(2 * np.pi * 10 * t),
                                 5 * np.sin(2 * np.pi * 10 * t + np.pi)])

    ai = daq.a_in_scan(0, 1, AiMode.SINGLE_ENDED, AiRange.BIP10V, n, 10_000)
    ao = daq.a_out_scan(0, 1, commanded, 10_000)
    assert daq.fake.ao_scan["mask"] == 0b11
    assert daq.fake.ao_scan["count"] == n

    measured = ai.read_all()
    ao.wait()

    assert measured.shape == (n, 2)
    np.testing.assert_allclose(measured, commanded, atol=0.001)
    assert daq.fake.ao_scan["played"] == n
    assert Cmd.AIN_SCAN_STOP in daq.fake.stopped_commands
    assert Cmd.AOUT_SCAN_STOP in daq.fake.stopped_commands


def test_single_channel_ao_scan_accepts_1d(daq):
    ramp = np.linspace(-1, 1, 100)
    ai = daq.a_in_scan(1, 1, AiMode.SINGLE_ENDED, AiRange.BIP10V, 100, 500)
    ao = daq.a_out_scan(1, 1, ramp, 500)
    measured = ai.read_all()
    ao.wait()
    np.testing.assert_allclose(measured[:, 0], ramp, atol=0.001)


def test_ao_scan_rejects_wrong_channel_count(daq):
    with pytest.raises(DaqError):
        daq.a_out_scan(0, 1, np.zeros((10, 3)), 1000)
