#!/usr/bin/env python3
"""v50 launcher: the audit's named lever — disc VELOCITY channel x arm
smoothing EIGHTHED. Single run targeting all four audit items.

Evidence chain: v40 proved each halving of arm smoothing lifts swing
(~+36%: 18.7 -> 25.5 deg); v49b proved the velocity disc channel is
LIVE (form held, drift best-in-class) but at quartered smoothing the
22-25 deg plateau persisted — the smoothing penalty is the remaining
suppressor. v50 pairs the channel with EIGHTHED smoothing for the
>=35 deg target (audit items 1/3), runs from the disc-intact v45 base
in regime 3 (the 13/13 domain, audit items 2/4), full guards.

Accept (single checkpoint): platform 13/13 AND P7 8/8 AND arms >= 30
deg AND G2 walk05 + walk10 hip AND knee >= 0.85 AND |drift| <= 1.0 m
AND robustness >= 45. (30 deg = the stretch between the 22-25 plateau
and the 35 target; anything >= 30 with all gates green is a strict
improvement worth releasing.)
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"            # velocity channel ON
os.environ["X1_FINE_TUNE_ITERS"] = "1000"  # v45 7395 -> total 8395

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
