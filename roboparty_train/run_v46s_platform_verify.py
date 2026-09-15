#!/usr/bin/env python3
"""Platform-side validation of the v46s soup release (+50 iters).

The soup (0.5*v44 + 0.5*v45) passed the ENTIRE local battery (G1-G3 all
walk scenarios, P7 8/8, drift, robustness 51/60) — but check_amp P1-P6
metrics are training-run measurements, and the soup has no training run.
This tiny run resumes from the merged soup .pt for +50 iterations in the
v44/v45/v46 regime to produce a fresh platform VERDICT on the soup
lineage. 50 iters barely moves the policy (kernel plateau is stable);
the metrics answer 'does the platform gate hold at the soup point'.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_FINE_TUNE_ITERS"] = "50"

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_soup_v46.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    raise FileNotFoundError("soup checkpoint missing")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
