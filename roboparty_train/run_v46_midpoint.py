#!/usr/bin/env python3
"""v46 launcher: bracket closed — midpoint guard between the complementary
one-metric misses.

The guard-dose bracket is now measured on both ends:
  v44 @ -0.8: walk10 hip 0.955 / knee 0.985 PASS;  walk05 knee 0.838 (1.4% short)
  v45 @ -1.2: walk05 hip 0.994 / knee 0.939 PASS; walk10 hip 0.843 (0.8% short)
Both checkpoints are otherwise fully green (platform 13/13 each, walkturn
+ back05 PASS, drift 0.31-0.44, arms 21-24 deg). The two metrics respond
in opposite directions to the guard — the midpoint (-1.0) is expected to
land both ratios inside [0.85, 0.99].

v46 = v45 model_7396 base + guard -1.0 + 300 iters (short dose; the
erosion onset under regime switch is ~+600, and this base has been in
the SAME regime since v44 — no switch here at all).

Accept (single checkpoint): platform 13/13 AND P7 8/8 AND G2 walk05 +
walk10 hip AND knee >= 0.85 AND |drift| <= 1.0 m AND robustness >= 48/60.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "3"        # kernel regime (held since v43)
os.environ["X1_FORM_GUARDS"] = "1"         # full guard set
os.environ["X1_FINE_TUNE_ITERS"] = "300"   # v45 7396 -> total 7696

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_7396_v45.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(p for p in _HERE.parent.glob("model_7396*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v45 base checkpoint not found")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
