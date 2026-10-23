"""
Import this before tkinter (or a matplotlib Tk window) is first used.

uv's standalone Pythons look for Tcl/Tk's script library relative to the venv,
where it isn't, and Tk() then fails with "Can't find a usable init.tcl".
This points them at the base install (lib/ on macOS and Linux, tcl/ on
Windows). No effect if the variables are already set or nothing is found.
"""

import glob
import os
import sys

for _pattern, _marker, _var in (("tcl[0-9]*", "init.tcl", "TCL_LIBRARY"),
                                ("tk[0-9]*", "tk.tcl", "TK_LIBRARY")):
    _found = sorted(d for sub in ("lib", "tcl")
                    for d in glob.glob(os.path.join(sys.base_prefix, sub, _pattern))
                    if os.path.isfile(os.path.join(d, _marker)))
    if _found:
        os.environ.setdefault(_var, _found[-1])
