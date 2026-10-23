"""
Pure-Python driver for the MCC USB-1608GX-2AO (and its AO-less siblings).

Talks to the device directly over libusb via pyusb. No MCC libraries needed.

Typical use::

    from mcc1608g import USB1608G, AiMode, AiRange

    with USB1608G.open() as daq:
        daq.a_out(0, 2.5)                              # set AOUT0 to 2.5 V
        volts = daq.a_in(0, AiMode.SINGLE_ENDED, AiRange.BIP10V)

        ai = daq.a_in_scan(0, 1, AiMode.SINGLE_ENDED, AiRange.BIP10V,
                           samples_per_channel=1000, rate=10_000)
        ao = daq.a_out_scan(0, 1, waveform, rate=10_000)  # waveform: (N, 2) volts
        data = ai.read_all()                            # (1000, 2) volts
        ao.wait()

Layout of this file, top to bottom:
  1. finding and opening the device (firmware upload happens here)
  2. the two primitive transports: control transfers and EEPROM reads
  3. calibration coefficients
  4. single-sample analog in / out
  5. hardware-paced scans (AiScan / AoScan helpers)
"""

from __future__ import annotations

import datetime as _dt
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import usb.core
import usb.util

from . import protocol as P
from .protocol import AiMode, AiRange, Cmd

CONTROL_TIMEOUT_MS = 1000
FIRMWARE_DIR = Path(__file__).parent / "firmware"


class DaqError(Exception):
    """Anything the device or the USB stack refused to do."""


@dataclass(frozen=True)
class CalCoef:
    """Linear correction stored by the factory: corrected = slope * x + offset."""
    slope: float
    offset: float


# ---------------------------------------------------------------------------
# 1. Discovery and connection
# ---------------------------------------------------------------------------

def _usb_backend():
    """Find a libusb-1.0 backend, preferring the wheel-bundled copy."""
    try:
        import libusb_package
        return libusb_package.get_libusb1_backend()
    except ImportError:
        pass
    import usb.backend.libusb1 as libusb1
    for candidate in (None, "/opt/homebrew/lib/libusb-1.0.dylib",
                      "/usr/local/lib/libusb-1.0.dylib"):
        backend = libusb1.get_backend(
            find_library=(lambda _n, c=candidate: c) if candidate else None)
        if backend is not None:
            return backend
    raise DaqError("libusb-1.0 not found. Install it (brew/apt) or `uv add libusb-package`.")


def find_devices() -> list[usb.core.Device]:
    """All MCC USB-1608G-family devices currently plugged in."""
    backend = _usb_backend()
    found = usb.core.find(find_all=True, idVendor=P.MCC_VENDOR_ID, backend=backend)
    return [d for d in found if d.idProduct in P.PRODUCT_NAMES]


class USB1608G:
    """One open USB-1608G-family device."""

    # ---- lifecycle ----------------------------------------------------------

    def __init__(self, usb_device: usb.core.Device):
        self._dev = usb_device
        self.product_id = usb_device.idProduct
        self.product_name = P.PRODUCT_NAMES[self.product_id]
        self.has_ao = self.product_id in P.PRODUCTS_WITH_AO
        self.ai_max_rate = P.AI_MAX_SAMPLE_RATE.get(
            self.product_id, P.AI_DEFAULT_MAX_SAMPLE_RATE)

        self._io_lock = threading.RLock()   # serialises control transfers
        self._ai_cal: dict[AiRange, CalCoef] = {}
        self._ao_cal: list[CalCoef] = []

        self._connect()

    @classmethod
    def open(cls, index: int = 0) -> "USB1608G":
        devices = find_devices()
        if not devices:
            raise DaqError("No MCC USB-1608G-family device found.")
        if index >= len(devices):
            raise DaqError(f"Only {len(devices)} device(s) found; index {index} out of range.")
        return cls(devices[index])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _connect(self):
        dev = self._dev
        try:
            dev.get_active_configuration()
        except usb.core.USBError:
            dev.set_configuration()
        try:
            usb.util.claim_interface(dev, 0)
        except usb.core.USBError as e:
            raise DaqError(
                f"Could not claim USB interface: {e}. On Linux add a udev rule for "
                f"VID 09db; on Windows bind WinUSB with Zadig; make sure nothing else "
                f"(libuldaq, InstaCal) has the device open.") from e

        self.serial = usb.util.get_string(dev, dev.iSerialNumber)

        # The FPGA is volatile; load it after every power-up.
        if not self._fpga_loaded():
            self._upload_fpga()
            if not self._fpga_loaded():
                raise DaqError("FPGA upload finished but the device reports it not loaded.")
        self.fpga_version = self._query(Cmd.FPGA_VERSION, length=2, fmt="<H")[0]

        # Make sure nothing is left running from a previous crashed session.
        self._send(Cmd.AIN_SCAN_STOP)
        if self.has_ao:
            self._send(Cmd.AOUT_SCAN_STOP)

        self._load_calibration()

        # Per MCC: the very first ADC reading after an FPGA load is garbage.
        self._set_ai_channels(AiMode.DIFFERENTIAL, AiRange.BIP10V, 0, 0)
        self._query(Cmd.AIN, w_value=0, length=2)

    def close(self):
        if self._dev is None:
            return
        try:
            self._send(Cmd.AIN_SCAN_STOP)
            if self.has_ao:
                self._send(Cmd.AOUT_SCAN_STOP)
        except (usb.core.USBError, DaqError):
            pass
        usb.util.release_interface(self._dev, 0)
        usb.util.dispose_resources(self._dev)
        self._dev = None

    def __repr__(self):
        return f"<{self.product_name} serial={self.serial} fpga=0x{self.fpga_version:04x}>"

    # ---- FPGA ----------------------------------------------------------------

    def _fpga_loaded(self) -> bool:
        return bool(self.status() & P.STATUS_FPGA_LOADED)

    def _upload_fpga(self):
        image = (FIRMWARE_DIR / P.FPGA_IMAGE[self.product_id]).read_bytes()
        self._send(Cmd.FPGA_CONFIG, data=bytes([P.FPGA_UNLOCK]))
        for start in range(0, len(image), P.FPGA_CHUNK):
            self._send(Cmd.FPGA_DATA, data=image[start:start + P.FPGA_CHUNK])

    # ---------------------------------------------------------------------------
    # 2. Primitive transports
    # ---------------------------------------------------------------------------

    def _send(self, cmd: Cmd, w_value: int = 0, w_index: int = 0, data: bytes = b""):
        """Vendor control transfer, host -> device."""
        with self._io_lock:
            try:
                sent = self._dev.ctrl_transfer(P.REQ_OUT, cmd, w_value, w_index,
                                               data, CONTROL_TIMEOUT_MS)
            except usb.core.USBError as e:
                raise DaqError(f"{cmd.name} failed: {e}") from e
        if sent != len(data):
            raise DaqError(f"{cmd.name}: sent {sent} of {len(data)} bytes")

    def _query(self, cmd: Cmd, w_value: int = 0, w_index: int = 0,
               length: int = 0, fmt: str | None = None):
        """Vendor control transfer, device -> host. Returns bytes, or unpacked `fmt`."""
        with self._io_lock:
            try:
                raw = self._dev.ctrl_transfer(P.REQ_IN, cmd, w_value, w_index,
                                              length, CONTROL_TIMEOUT_MS)
            except usb.core.USBError as e:
                raise DaqError(f"{cmd.name} failed: {e}") from e
        raw = bytes(raw)
        if len(raw) != length:
            raise DaqError(f"{cmd.name}: got {len(raw)} bytes, expected {length}")
        return struct.unpack(fmt, raw) if fmt else raw

    def status(self) -> int:
        return self._query(Cmd.STATUS, length=2, fmt="<H")[0]

    def blink_led(self, count: int = 3):
        self._send(Cmd.BLINK_LED, data=bytes([count]))

    def read_memory(self, address: int, count: int) -> bytes:
        """Read `count` bytes of EEPROM starting at `address`."""
        with self._io_lock:
            self._send(Cmd.MEMORY_ADDRESS, data=struct.pack("<H", address))
            out = bytearray()
            while len(out) < count:
                n = min(P.EEPROM_MAX_TRANSFER, count - len(out))
                out += self._query(Cmd.MEMORY, length=n)
        return bytes(out)

    # ---------------------------------------------------------------------------
    # 3. Calibration
    # ---------------------------------------------------------------------------

    def _load_calibration(self):
        raw = self.read_memory(P.EEPROM_AI_CAL_ADDR, P.CalPair.size * len(AiRange))
        for i, rng in enumerate(AiRange):
            self._ai_cal[rng] = CalCoef(*P.CalPair.unpack_from(raw, i * P.CalPair.size))
        self.ai_cal_date = self._read_cal_date(P.EEPROM_AI_CAL_DATE_ADDR)

        if self.has_ao:
            raw = self.read_memory(P.EEPROM_AO_CAL_ADDR, P.CalPair.size * 2)
            self._ao_cal = [CalCoef(*P.CalPair.unpack_from(raw, i * P.CalPair.size))
                            for i in range(2)]
            self.ao_cal_date = self._read_cal_date(P.EEPROM_AO_CAL_DATE_ADDR)

    def _read_cal_date(self, address: int) -> _dt.datetime | None:
        y, mo, d, h, mi, s = struct.unpack(">6H", self.read_memory(address, 12))
        try:
            return _dt.datetime(y, mo, d, h, mi, s)
        except ValueError:
            return None

    @property
    def ai_calibration(self) -> dict[AiRange, CalCoef]:
        return dict(self._ai_cal)

    @property
    def ao_calibration(self) -> list[CalCoef]:
        return list(self._ao_cal)

    # Count <-> volts. The factory coefficients correct the *count*, and the
    # ideal transfer function then maps counts onto the range.

    def _ai_counts_to_volts(self, counts, rng: AiRange):
        cal = self._ai_cal[rng]
        lsb = rng.span / 2 ** P.AI_RESOLUTION_BITS
        return (cal.slope * np.asarray(counts, dtype=np.float64) + cal.offset) * lsb + rng.minimum

    def _ao_volts_to_counts(self, volts, channel: int):
        cal = self._ao_cal[channel]
        lsb = P.AO_SPAN_VOLTS / 2 ** P.AO_RESOLUTION_BITS
        ideal = (np.asarray(volts, dtype=np.float64) - P.AO_MIN_VOLTS) / lsb
        counts = np.floor(cal.slope * ideal + cal.offset + 0.5)
        return np.clip(counts, 0, 2 ** P.AO_RESOLUTION_BITS - 1).astype(np.uint16)

    # ---------------------------------------------------------------------------
    # 4. Single samples
    # ---------------------------------------------------------------------------

    def _set_ai_channels(self, mode: AiMode, rng: AiRange, low: int, high: int):
        """Program the scan list: channels low..high, all with the same mode/range."""
        max_chan = 8 if mode == AiMode.DIFFERENTIAL else 16
        if not 0 <= low <= high < max_chan:
            raise DaqError(f"Channels {low}..{high} invalid for {mode.name}")
        cfg = bytearray(P.AI_CONFIG_SLOTS)
        for slot, ch in enumerate(range(low, high + 1)):
            cfg[slot] = P.ai_config_byte(ch, mode, rng, last=(ch == high))
        self._send(Cmd.AIN_CONFIG, data=bytes(cfg))

    def a_in(self, channel: int, mode: AiMode = AiMode.SINGLE_ENDED,
             rng: AiRange = AiRange.BIP10V) -> float:
        """Read one voltage from one channel (software-timed)."""
        with self._io_lock:
            self._set_ai_channels(mode, rng, channel, channel)
            (count,) = self._query(Cmd.AIN, w_value=channel, length=2, fmt="<H")
        return float(self._ai_counts_to_volts(count, rng))

    def a_in_raw(self, channel: int, mode: AiMode = AiMode.SINGLE_ENDED,
                 rng: AiRange = AiRange.BIP10V) -> int:
        """Same as a_in but returns the uncalibrated 16-bit count."""
        with self._io_lock:
            self._set_ai_channels(mode, rng, channel, channel)
            return self._query(Cmd.AIN, w_value=channel, length=2, fmt="<H")[0]

    def a_out(self, channel: int, volts: float):
        """Set one output channel to `volts` (software-timed)."""
        self._check_ao(channel)
        count = int(self._ao_volts_to_counts(volts, channel))
        self._send(Cmd.AOUT, w_value=count, w_index=channel)

    def _check_ao(self, channel: int):
        if not self.has_ao:
            raise DaqError(f"{self.product_name} has no analog outputs.")
        if channel not in (0, 1):
            raise DaqError("AO channel must be 0 or 1.")

    # ---------------------------------------------------------------------------
    # 5. Hardware-paced scans
    # ---------------------------------------------------------------------------

    def _bulk_max_packet(self, endpoint: int) -> int:
        for intf in self._dev.get_active_configuration():
            for ep in intf:
                if ep.bEndpointAddress == endpoint:
                    return ep.wMaxPacketSize
        raise DaqError(f"Endpoint 0x{endpoint:02x} not found in descriptor.")

    def a_in_scan(self, low: int, high: int, mode: AiMode, rng: AiRange,
                  samples_per_channel: int, rate: float) -> "AiScan":
        """
        Start the ADC pacer sampling channels low..high at `rate` scans/second.

        Returns immediately with an AiScan; call `.read_all()` to collect the
        data. The device buffers 8k samples, so at high rates you must start
        reading promptly or the FIFO overruns.
        """
        n_chan = high - low + 1
        if not 0 < rate <= self.ai_max_rate / n_chan:
            raise DaqError(f"Rate {rate} exceeds {self.ai_max_rate / n_chan:.0f} S/s "
                           f"for {n_chan} channel(s).")
        period, actual_rate = P.pacer_period(rate)

        with self._io_lock:
            self._set_ai_channels(mode, rng, low, high)
            self._dev.clear_halt(P.AI_SCAN_ENDPOINT)
            self._send(Cmd.AIN_CLEAR_FIFO)

            # Slow scans: pack whole scans into each USB packet, as many as fit
            # and arrive within ~50 ms, picking a count that divides the scan so
            # the last packet is full. One scan per packet would mean one USB
            # read per scan on the host (1000/s for 12 channels at 1 kHz), and a
            # busy GUI thread could then fall behind and overrun the FIFO.
            # Fast scans: fill 512-byte packets for throughput.
            max_samples = self._bulk_max_packet(P.AI_SCAN_ENDPOINT) // 2
            if rate <= 1000:
                most = max(1, min(max_samples // n_chan, int(rate * 0.05)))
                scans = max(k for k in range(1, most + 1) if samples_per_channel % k == 0)
                packet_samples = scans * n_chan
            else:
                packet_samples = max_samples
            cfg = P.AinScanConfig.pack(samples_per_channel, 0, period,
                                       packet_samples - 1, 0, 0)
            self._send(Cmd.AIN_SCAN_START, data=cfg)

        return AiScan(self, n_chan, samples_per_channel, rng, actual_rate)

    def a_out_scan(self, low: int, high: int, volts, rate: float) -> "AoScan":
        """
        Start the DAC pacer playing `volts` on channels low..high at `rate`.

        `volts` is shaped (samples, channels) for several channels, or 1-D for
        one. The FIFO is pre-filled before the pacer starts; the rest streams
        from a background thread. Call `.wait()` on the returned AoScan.
        """
        self._check_ao(low)
        self._check_ao(high)
        n_chan = high - low + 1
        if not 0 < rate <= P.AO_MAX_SAMPLE_RATE:
            raise DaqError(f"Rate {rate} exceeds {P.AO_MAX_SAMPLE_RATE} S/s.")

        volts = np.asarray(volts, dtype=np.float64)
        if volts.ndim == 1:
            volts = volts[:, None]
        if volts.shape[1] != n_chan:
            raise DaqError(f"Expected {n_chan} column(s), got {volts.shape[1]}.")
        counts = np.empty(volts.shape, dtype="<u2")
        for i, ch in enumerate(range(low, high + 1)):
            counts[:, i] = self._ao_volts_to_counts(volts[:, i], ch)
        payload = counts.tobytes()   # interleaved, little-endian, one scan per row

        period, actual_rate = P.pacer_period(rate)
        channel_mask = sum(1 << ch for ch in range(low, high + 1))
        cfg = P.AoutScanConfig.pack(len(volts), 0, period, channel_mask)

        scan = AoScan(self, payload, n_chan, len(volts), actual_rate)
        with self._io_lock:
            self._dev.clear_halt(P.AO_SCAN_ENDPOINT)
            self._send(Cmd.AOUT_CLEAR_FIFO)
            scan._prefill()
            self._send(Cmd.AOUT_SCAN_START, data=cfg)
        scan._start_feeder()
        return scan

    def stop_scans(self):
        self._send(Cmd.AIN_SCAN_STOP)
        if self.has_ao:
            self._send(Cmd.AOUT_SCAN_STOP)


class AiScan:
    """A running analog-input scan. Produced by USB1608G.a_in_scan()."""

    READ_CHUNK_BYTES = 16 * 1024
    BYTES_PER_SAMPLE = 2

    def __init__(self, daq: USB1608G, n_chan: int, samples_per_channel: int,
                 rng: AiRange, rate: float):
        self._daq = daq
        self.n_chan = n_chan
        self.samples_per_channel = samples_per_channel
        self.rng = rng
        self.rate = rate                 # actual pacer rate, scans/second
        self._raw = bytearray()

    @property
    def total_bytes(self) -> int:
        return self.n_chan * self.samples_per_channel * self.BYTES_PER_SAMPLE

    def read_all(self) -> np.ndarray:
        """Block until every sample has arrived. Returns volts, shape (samples, channels)."""
        dev = self._daq._dev
        while len(self._raw) < self.total_bytes:
            want = min(self.READ_CHUNK_BYTES, self.total_bytes - len(self._raw))
            # Time for this many bytes to be produced, plus slack.
            timeout_ms = int(1000 * want / (self.BYTES_PER_SAMPLE * self.n_chan * self.rate)) + 2000
            try:
                chunk = dev.read(P.AI_SCAN_ENDPOINT, want, timeout_ms)
            except usb.core.USBTimeoutError as e:
                self._daq._send(Cmd.AIN_SCAN_STOP)
                overrun = self._daq.status() & P.STATUS_AI_OVERRUN
                raise DaqError(
                    f"AI scan timed out after {len(self._raw)} of {self.total_bytes} bytes"
                    + (" (FIFO overrun)" if overrun else "")) from e
            self._raw += bytes(chunk)

        if self._daq.status() & P.STATUS_AI_OVERRUN:
            raise DaqError("AI FIFO overrun during scan; data is incomplete.")
        self._daq._send(Cmd.AIN_SCAN_STOP)

        counts = np.frombuffer(bytes(self._raw[:self.total_bytes]), dtype="<u2")
        counts = counts.reshape(self.samples_per_channel, self.n_chan)
        return self._daq._ai_counts_to_volts(counts, self.rng)

    def stop(self):
        self._daq._send(Cmd.AIN_SCAN_STOP)


class AoScan:
    """A running analog-output scan. Produced by USB1608G.a_out_scan()."""

    WRITE_CHUNK_SAMPLES = 1024

    def __init__(self, daq: USB1608G, payload: bytes, n_chan: int,
                 samples_per_channel: int, rate: float):
        self._daq = daq
        self._payload = payload
        self._sent = 0
        self.n_chan = n_chan
        self.samples_per_channel = samples_per_channel
        self.rate = rate
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self.error: Exception | None = None

    @property
    def duration(self) -> float:
        return self.samples_per_channel / self.rate

    def _write(self, n_bytes: int):
        chunk = self._payload[self._sent:self._sent + n_bytes]
        # The device NAKs while its FIFO is full, so this blocks until room appears.
        samples = len(chunk) // 2
        timeout_ms = int(1000 * samples / (self.n_chan * self.rate)) + 2000
        written = self._daq._dev.write(P.AO_SCAN_ENDPOINT, chunk, timeout_ms)
        if written != len(chunk):
            raise DaqError(f"AO write sent {written} of {len(chunk)} bytes before timing out.")
        self._sent += written

    def _prefill(self):
        """Load what the FIFO accepts before the pacer starts; the feeder sends the rest."""
        self._write(min(len(self._payload), P.AO_PREFILL_SAMPLES * 2))

    def _start_feeder(self):
        if self._sent >= len(self._payload):
            return
        self._thread = threading.Thread(target=self._feed, name="ao-feeder", daemon=True)
        self._thread.start()

    def _feed(self):
        try:
            while self._sent < len(self._payload) and not self._stopping.is_set():
                self._write(self.WRITE_CHUNK_SAMPLES * 2)
        except Exception as e:      # surfaced by wait()
            if not self._stopping.is_set():
                self.error = e

    def wait(self, extra_timeout: float = 2.0):
        """Block until all samples were sent and the pacer reports idle."""
        if self._thread is not None:
            self._thread.join()
        if self.error is not None:
            raise DaqError(f"AO feeder failed: {self.error}") from self.error

        deadline = time.monotonic() + self.duration + extra_timeout
        while time.monotonic() < deadline:
            status = self._daq.status()
            if status & P.STATUS_AO_UNDERRUN:
                self._daq._send(Cmd.AOUT_SCAN_STOP)
                raise DaqError("AO FIFO underrun: host could not keep the DAC fed.")
            if not status & P.STATUS_AO_SCAN_RUNNING:
                break
            time.sleep(0.005)
        self._daq._send(Cmd.AOUT_SCAN_STOP)

    def stop(self):
        """Stop early. The feeder may be blocked in a write for up to its timeout."""
        self._stopping.set()
        self._daq._send(Cmd.AOUT_SCAN_STOP)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
