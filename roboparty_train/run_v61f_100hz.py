#!/usr/bin/env python3
"""v61f launcher: v61e + task_style_lerp 0.68 -> 0.0 (style channel removed).

Corrected forensics across v61b/c/d2/e (all 100 Hz fine-tunes collapsed
amplitude — arms 12-19 deg vs champion 34.7, swing 10-24 mm vs 54):
  - Per-SECOND style income of the micro-gait is ~2.7x the 50 Hz champion's
    (5.4e-3/s vs 2.0e-3/s): the disc equilibrium at 100 Hz does not merely
    saturate — it ACTIVELY REWARDS micro-stepping (the earlier "dead
    channel" reading was an arithmetic slip).
  - Direct prior doubling (v61e, 0.06->0.12) could not out-pull it.
Single-variable step: remove the style term from the reward mix entirely
(X1_TASK_LERP=0). Task rewards + form guards + amplitude prior 0.12 remain.
Accept: arms >= 24 deg, swing >= 30 mm, and all non-regress gates (kernels,
drift, robustness, K1/K2). The disc still trains and logs (its score is
diagnostic), it just contributes zero reward."""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_CONTROL_HZ"] = "100"
os.environ["X1_AMP_NUM_STEPS"] = "6"
os.environ["X1_ARM_PRIOR"] = "0.12"
os.environ["X1_TASK_LERP"] = "0.0"
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
