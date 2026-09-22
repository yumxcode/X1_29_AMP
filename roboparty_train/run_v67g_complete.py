#!/usr/bin/env python3
"""v67g launcher: COMPLETE the v67f data-route scratch to convergence.

v67f (TASK_20260922_090, 2799/2800): trio removal CONFIRMED as convergence
impairment (ep_len 703-850 vs v67d 599 at same iters) AND the rhythm moved
further: hip cadence 0.87 Hz at m2799 — INSIDE the human gate band
[0.55, 1.20] for the first time (3.83 Hz old-lib -> 1.92 Hz v67d trio ->
0.87 Hz v67f no-trio). Remaining gap = SWING AMPLITUDE (airtime 0.7-1.0%,
human-cadence shuffle). ep_len < 900 -> pre-registered continuation pod.

+1200 iters from m2799 -> 4000 total (standard convergence budget).
Env = v67f frozen (overground library + VMATCH + macro disc, NO trio)."""
import os
import runpy
import sys
from pathlib import Path

# --- v67d recipe (frozen) --------------------------------------------------
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_YAW_GUARD"] = "1.0"
os.environ["X1_AMP_NUM_STEPS"] = "30"
os.environ["X1_DISC_LINVEL"] = "1"
os.environ["X1_DISC_BUFFER"] = "24"
os.environ["X1_DISC_VMATCH"] = "1"
os.environ["X1_MOTION_DIR"] = "x1_lab_v67"
os.environ["X1_CADENCE_PRIOR"] = "0"
os.environ["X1_SWING_PRIOR"] = "0"
os.environ["X1_HIP_PHASE"] = "0"

# --- continuation -----------------------------------------------------------
os.environ["X1_FINE_TUNE_ITERS"] = "1200"

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_2799_v67f.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(_HERE.parent.glob("model_2799_v67d*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v67f m2799 base checkpoint not found")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
