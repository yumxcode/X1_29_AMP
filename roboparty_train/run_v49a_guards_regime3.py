#!/usr/bin/env python3
"""v49a launcher: v48's SOLVED guards (drift/back05) back in the 13/13
kernel regime — audit item 3's direct request.

v48 proved the guard recipe (yaw_bias -1.0 + hip/knee RMS -1.0 under
disturbance): drift +0.75/-0.73 both under gate, back05 0.967/0.929
toe_first=0, robustness 52/60. But its dense ±0.65 regime crushed the
platform kernels (8/13) and walk05 (0.724). Regime 3 (±0.5 @5-10s) is
the domain that delivered 13/13 in v43/v44/v45/v46 AND walk05 PASS
(v44 0.978 / v45 0.994 / v46 0.951) — v49a runs the v48 guards THERE.

Dose +600 (kernel restores by +200; the guards hold form in this
regime — v44/v46 evidence). Everything else identical to v48 (arm
smoothing quartered, lerp 0.68, style 2.5, disc intact from v45).

Accept (single checkpoint): platform 13/13 AND P7 8/8 AND G2 walk05 +
walk10 hip AND knee >= 0.85 AND back05 hip >= 0.85 toe_first == 0 AND
|drift| <= 1.0 m AND robustness >= 48 AND arms >= 20 deg. push1.0
remains the documented open trade.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "3"        # the 13/13 kernel regime
os.environ["X1_FORM_GUARDS"] = "1"         # full guard set (v48 recipe)
os.environ["X1_FINE_TUNE_ITERS"] = "600"   # v45 7395 -> total 7995

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_7395_v45.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(p for p in _HERE.parent.glob("model_7395*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v45 base checkpoint not found")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
