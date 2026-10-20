"""
Wire protocol for the MCC USB-1608G family (USB-1608G, USB-1608GX, USB-1608GX-2AO).

Everything the host says to the device is a USB *vendor control transfer*:
an 8-byte setup packet (request code, wValue, wIndex, length) optionally
followed by a small payload. Streaming scan data moves over two *bulk*
endpoints instead. This module is the single place where those numbers live.

Sources: MCC's open-source libuldaq (src/usb/Usb1608g.cpp, ai/AiUsb1608g.cpp,
ao/AoUsb1208hs.cpp, UsbFpgaDevice.cpp) and Warren Jasper's usb-1608G driver.
"""

import struct
from enum import IntEnum

# --- USB identification -----------------------------------------------------

MCC_VENDOR_ID = 0x09DB

PRODUCT_NAMES = {
    0x0110: "USB-1608G",
    0x0111: "USB-1608GX",
    0x0112: "USB-1608GX-2AO",
    0x0134: "USB-1608G",        # second hardware revision
    0x0135: "USB-1608GX",
    0x0136: "USB-1608GX-2AO",
}

# Products with the two analog-output channels.
PRODUCTS_WITH_AO = {0x0112, 0x0136}

# Which FPGA bitstream a product needs. The device forgets it on power loss,
# so the host uploads it on every first connection after power-up.
FPGA_IMAGE = {
    0x0110: "USB_1608G.rbf",
    0x0111: "USB_1608G.rbf",
    0x0112: "USB_1608G.rbf",
    0x0134: "USB_1608G_2.rbf",
    0x0135: "USB_1608G_2.rbf",
    0x0136: "USB_1608G_2.rbf",
}

# bmRequestType for vendor requests aimed at the device.
REQ_OUT = 0x40   # host -> device
REQ_IN = 0xC0    # device -> host

# Bulk endpoints used by hardware-paced scans.
AI_SCAN_ENDPOINT = 0x86   # IN:  ADC samples stream out of the device here
AO_SCAN_ENDPOINT = 0x02   # OUT: DAC samples stream into the device here

# --- Command codes (bRequest) -----------------------------------------------


class Cmd(IntEnum):
    # Analog input
    AIN = 0x10                 # IN,  wValue=channel -> uint16 LE count
    AIN_SCAN_START = 0x12      # OUT, payload = AinScanConfig
    AIN_SCAN_STOP = 0x13       # OUT, no payload
    AIN_CONFIG = 0x14          # OUT, payload = 16 channel-config bytes
    AIN_CLEAR_FIFO = 0x15      # OUT, no payload
    # Analog output
    AOUT = 0x18                # OUT, wValue=count, wIndex=channel
    AOUT_SCAN_START = 0x1A     # OUT, payload = AoutScanConfig
    AOUT_SCAN_STOP = 0x1B      # OUT, no payload
    AOUT_CLEAR_FIFO = 0x1C     # OUT, no payload
    # Non-volatile memory (EEPROM)
    MEMORY = 0x30              # IN/OUT, <=64 bytes at the current address
    MEMORY_ADDRESS = 0x31      # OUT, payload = uint16 LE address
    # Housekeeping
    STATUS = 0x40              # IN,  -> uint16 LE status bits
    BLINK_LED = 0x41           # OUT, payload = uint8 blink count
    RESET = 0x42               # OUT, no payload
    TRIGGER_CONFIG = 0x43      # OUT, payload = uint8 (external trigger, unused here)
    # FPGA
    FPGA_CONFIG = 0x50         # OUT, payload = 0xAD to enter configuration mode
    FPGA_DATA = 0x51           # OUT, payload = <=64 bytes of bitstream
    FPGA_VERSION = 0x52        # IN,  -> uint16 LE version


# Byte the device expects before it will accept bitstream data.
FPGA_UNLOCK = 0xAD
# Bitstream is fed in control transfers of this size.
FPGA_CHUNK = 64

# --- STATUS bits ------------------------------------------------------------

STATUS_AI_SCAN_RUNNING = 0x0002
STATUS_AI_OVERRUN = 0x0004      # ADC FIFO filled before the host drained it
STATUS_AO_SCAN_RUNNING = 0x0008
STATUS_AO_UNDERRUN = 0x0010     # DAC FIFO ran dry before the host refilled it
STATUS_FPGA_LOADED = 0x0100

# --- Timing -----------------------------------------------------------------

# Both pacers divide this clock. Sample rate = CLOCK_HZ / (pacer_period + 1).
CLOCK_HZ = 64_000_000

AI_FIFO_SAMPLES = 8 * 1024
AO_FIFO_SAMPLES = 4 * 1024

# --- Analog input configuration --------------------------------------------


class AiMode(IntEnum):
    DIFFERENTIAL = 0    # channels 0-7, measured between CHn HI and CHn LO
    SINGLE_ENDED = 1    # channels 0-15, measured against AGND


class AiRange(IntEnum):
    """Input range codes as the hardware wants them, plus their span in volts."""
    BIP10V = 0
    BIP5V = 1
    BIP2V = 2
    BIP1V = 3

    @property
    def span(self) -> float:
        return {0: 20.0, 1: 10.0, 2: 4.0, 3: 2.0}[self.value]

    @property
    def minimum(self) -> float:
        return -self.span / 2


AI_RESOLUTION_BITS = 16
AI_MAX_SAMPLE_RATE = {0x0110: 250_000, 0x0134: 250_000}   # everything else 500 kS/s
AI_DEFAULT_MAX_SAMPLE_RATE = 500_000

# Each AIN_CONFIG byte describes one slot of the channel scan list:
#   bits 0-2  channel within its bank (channel % 8)
#   bits 3-4  range code
#   bits 5-6  mode: 0 = differential, 1 = single-ended CH0-7, 2 = single-ended CH8-15
#   bit  7    set on the last slot in the list
AI_CONFIG_SLOTS = 16


def ai_config_byte(channel: int, mode: AiMode, rng: AiRange, last: bool) -> int:
    if mode == AiMode.DIFFERENTIAL:
        mode_code = 0
    else:
        mode_code = 1 if channel < 8 else 2
    return (channel % 8) | (rng << 3) | (mode_code << 5) | (int(last) << 7)


# AIN_SCAN_START payload (little-endian, packed):
#   uint32 scan_count      number of scans (all channels once); 0 = continuous
#   uint32 retrig_count    only used with external retriggering
#   uint32 pacer_period    CLOCK_HZ / rate - 1
#   uint8  packet_size     samples per USB packet minus one
#   uint8  options         bit0 burst, bit3 ext trigger, bit6 retrigger
#   uint8  reserved
AinScanConfig = struct.Struct("<IIIBBB")

# --- Analog output configuration -------------------------------------------

AO_RESOLUTION_BITS = 16
AO_SPAN_VOLTS = 20.0          # DAC is fixed at +/-10 V
AO_MIN_VOLTS = -10.0
AO_MAX_SAMPLE_RATE = 500_000

# AOUT_SCAN_START payload (little-endian, packed):
#   uint32 scan_count      0 = continuous
#   uint32 retrig_count
#   uint32 pacer_period
#   uint8  options         bits 0-1 channel mask, bit4 ext trigger, bit5 retrigger
AoutScanConfig = struct.Struct("<IIIB")

# --- EEPROM layout ----------------------------------------------------------
#
# Calibration coefficients are pairs of little-endian float32 (slope, offset)
# that map raw ADC counts to corrected counts, or ideal DAC counts to the
# counts that actually produce the requested voltage.

EEPROM_AI_CAL_ADDR = 0x7000         # 4 pairs, one per AiRange, in enum order
EEPROM_AI_CAL_DATE_ADDR = 0x7098    # 6 x uint16 big-endian: Y M D h m s
EEPROM_AO_CAL_ADDR = 0x7080         # 2 pairs, one per AO channel
EEPROM_AO_CAL_DATE_ADDR = 0x70A4
EEPROM_USER_ADDR = 0x7410           # 3048 bytes free for the user
EEPROM_MAX_TRANSFER = 64

CalPair = struct.Struct("<ff")


def pacer_period(rate_hz: float) -> tuple[int, float]:
    """Convert a requested sample rate to the pacer divider and the rate it yields."""
    period = int(CLOCK_HZ / rate_hz - 1)
    period = max(0, min(period, 0xFFFFFFFF))
    return period, CLOCK_HZ / (period + 1)
