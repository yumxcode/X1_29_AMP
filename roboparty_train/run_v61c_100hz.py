#!/usr/bin/env python3
"""v61c launcher: v61b + AMP_NUM_STEPS 3->6 (restore the 60 ms disc window).

v61b verdict (TASK_20260920_032, first 100 Hz run):
  PASS: platform 13/13 + T1/T2/T3 (kernel 0.8607 / ang 0.6354 / style 1.857),
        sim2sim stability 5/5 x 11.5 s, drift 0.61/-0.23 m (gate 1.0),
        robustness 52/60 (>= soup59c 51), arm antiphase -0.95.
  FAIL: arm amplitude 14.5/10.8 deg (gate >= 24) and swing height < 12 mm
        (drag walk; soup59c had 54 mm) — the disc window halved to 30 ms
        with AMP_NUM_STEPS=3 at 100 Hz, matching the pre-registered
        falsification condition of the fine-control hypothesis.

Fix (pre-registered): X1_AMP_NUM_STEPS=6 -> 6 x 10 ms = 60 ms window,
identical TIME coverage to the 50 Hz champion recipe. Disc obs expand on
resume is handled by the amp_runner shape-mismatch re-init path (v38)."""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_CONTROL_HZ"] = "100"
os.environ["X1_AMP_NUM_STEPS"] = "6"
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_FINE_TUNE_ITERS"] = "1200"
os.environ["X1_YAW_GUARD"] = "1.0"

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_9393_v56b.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(_HERE.parent.glob("model_9393_v56b*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v56b base checkpoint not found")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
