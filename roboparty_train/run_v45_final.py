#!/usr/bin/env python3
"""v45 launcher: the final 0.012 — one notch more of the working lever.

v44 verdict (TASK_20260915_008): single-checkpoint convergence is ONE
metric away — platform 13/13, P7 8/8 (v43's P7h miss fixed), walk10 G2
0.955/0.985, back05 G2 0.890/0.941 (was 0.697 FAIL), drift +0.44,
robustness 48/60. Sole miss: walk05 knee 0.838 vs 0.85 (0.012 = 1.4%).

The knee-pair guard took that metric 0.745 -> 0.838 in one dose at
weight -0.8 with EVERY other axis improving or holding — the cleanest
Pareto improvement of the arc. v45: same lever, one notch (-1.2),
short dose (+300 iters from v44 model_7097) to close the last gap
without giving the regime-switch erosion (onset ~+600) time to act.

Accept (single checkpoint): platform 13/13 AND P7 8/8 AND G2 walk05 +
walk10 hip AND knee >= 0.85 AND |drift| <= 1.0 m AND robustness >= 48/60.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "3"        # kernel regime (held in v43/v44)
os.environ["X1_FORM_GUARDS"] = "1"         # full guard set
os.environ["X1_FINE_TUNE_ITERS"] = "300"   # v44 7097 -> total 7397

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_7097_v44.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(p for p in _HERE.parent.glob("model_7097*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v44 base checkpoint not found")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
