#!/usr/bin/env python3
"""v52 launcher: DIRECT arm-swing amplitude prior — the audit's structural
lever (route 3 of 3), now implemented after v50/v51 falsified the other two.

arm_swing_amplitude_prior: positive prior on the HIGH-PASSED shoulder RMS
swing amplitude, full reward in the [30, 45] deg band (proportional push
below 30, decay above 45; min(L,R) anti-farming; DC-removed so frozen
offsets score zero). Weight 0.3 (coupling-term scale). Paired with: v48
guard recipe (yaw -1.0 + hip/knee -1.0: the drift/back05 solution), full
guards, velocity disc channel, arm smoothing 1/4 (peak band), regime 3.

Base: v45 model_7395 (disc intact). +1000 iters. PRIMARY: arms >= 30 deg
on the local battery. Secondary (the full-gate sweep): 13/13 AND P7 8/8
AND walk05+10 hip/knee >= 0.85 AND back05 hip >= 0.85 toe_first == 0 AND
|drift| <= 1.0 AND robustness >= 45.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
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
