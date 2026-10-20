#!/usr/bin/env python3
"""
Cross-check our pure-Python driver against MCC's uldaq library.

Both libraries want exclusive access to the USB device, so the test runs in
two passes: first uldaq writes a set of voltages and reads them back, then
our driver does the same thing. The two result tables should agree to within
ADC noise (a millivolt or so). Device identity and calibration dates are
compared too.

Wiring (same as the loopback tests):
  AOUT0 -> CH0, AOUT1 -> CH1, AGND -> AGND

Usage:
  uv sync --extra compare
  uv run compare-uldaq
"""

import sys
import time

TEST_VOLTAGES = [-9.0, -5.0, -2.5, -1.0, 0.0, 1.0, 2.5, 5.0, 9.0]
SETTLE_TIME = 0.01
CHANNEL_PAIRS = [(0, 0), (1, 1)]   # (AO channel, AI channel)
AGREEMENT_LIMIT_V = 0.005          # ours vs uldaq must agree to this


def run_with_uldaq():
    from uldaq import (AInFlag, AiInputMode, AOutFlag, DaqDevice, InterfaceType,
                       Range, get_daq_device_inventory)

    devices = get_daq_device_inventory(InterfaceType.USB)
    if not devices:
        sys.exit("uldaq: no device found")
    daq = DaqDevice(devices[0])
    daq.connect()
    info = {
        "product": devices[0].product_name,
        "serial": devices[0].unique_id,
    }
    ai, ao = daq.get_ai_device(), daq.get_ao_device()
    readings = {}
    try:
        for ao_ch, ai_ch in CHANNEL_PAIRS:
            for v in TEST_VOLTAGES:
                ao.a_out(ao_ch, Range.BIP10VOLTS, AOutFlag.DEFAULT, v)
                time.sleep(SETTLE_TIME)
                readings[(ao_ch, v)] = ai.a_in(ai_ch, AiInputMode.SINGLE_ENDED,
                                               Range.BIP10VOLTS, AInFlag.DEFAULT)
            ao.a_out(ao_ch, Range.BIP10VOLTS, AOutFlag.DEFAULT, 0.0)
    finally:
        daq.disconnect()
        daq.release()
    return info, readings


def run_with_ours():
    from mcc1608g import USB1608G, AiMode, AiRange

    with USB1608G.open() as daq:
        info = {
            "product": daq.product_name,
            "serial": daq.serial,
            "fpga": f"0x{daq.fpga_version:04x}",
            "ai_cal_date": daq.ai_cal_date,
            "ao_cal_date": daq.ao_cal_date,
            "ai_cal": daq.ai_calibration,
            "ao_cal": daq.ao_calibration,
        }
        readings = {}
        for ao_ch, ai_ch in CHANNEL_PAIRS:
            for v in TEST_VOLTAGES:
                daq.a_out(ao_ch, v)
                time.sleep(SETTLE_TIME)
                readings[(ao_ch, v)] = daq.a_in(ai_ch, AiMode.SINGLE_ENDED, AiRange.BIP10V)
            daq.a_out(ao_ch, 0.0)
    return info, readings


def main():
    print("Pass 1: MCC uldaq")
    ul_info, ul_read = run_with_uldaq()
    print(f"  {ul_info['product']}  serial {ul_info['serial']}")

    print("Pass 2: mcc1608g (ours)")
    our_info, our_read = run_with_ours()
    print(f"  {our_info['product']}  serial {our_info['serial']}  FPGA {our_info['fpga']}")
    print(f"  AI calibrated {our_info['ai_cal_date']}, AO calibrated {our_info['ao_cal_date']}")
    for rng, coef in our_info["ai_cal"].items():
        print(f"  AI cal {rng.name:7s} slope={coef.slope:.6f} offset={coef.offset:+.3f}")
    for ch, coef in enumerate(our_info["ao_cal"]):
        print(f"  AO cal ch{ch}     slope={coef.slope:.6f} offset={coef.offset:+.3f}")

    print()
    identity_ok = (ul_info["product"] == our_info["product"]
                   and ul_info["serial"] == our_info["serial"])
    print(f"Identity match: {'yes' if identity_ok else 'NO'}")

    print(f"\n{'AO':>3} {'Set (V)':>9} {'uldaq (V)':>11} {'ours (V)':>11} {'diff (mV)':>10}  ok")
    worst = 0.0
    failures = 0
    for key in ul_read:
        ao_ch, v = key
        diff = our_read[key] - ul_read[key]
        worst = max(worst, abs(diff))
        ok = abs(diff) < AGREEMENT_LIMIT_V
        failures += not ok
        print(f"{ao_ch:>3} {v:9.3f} {ul_read[key]:11.4f} {our_read[key]:11.4f} "
              f"{diff * 1000:10.2f}  {'yes' if ok else 'NO'}")

    print(f"\nWorst disagreement: {worst * 1000:.2f} mV")
    if failures or not identity_ok:
        print(f"{failures} reading(s) disagree beyond {AGREEMENT_LIMIT_V * 1000:.0f} mV")
        sys.exit(1)
    print("Drivers agree.")


if __name__ == "__main__":
    main()
