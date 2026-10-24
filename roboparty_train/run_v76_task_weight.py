#!/usr/bin/env python3
"""v76 launcher: TASK-WEIGHT dose ladder (track_lin_vel_xy_exp 1.25 -> 2.0).

v75 verdict (TASK_20261024_036): H7 largely falsified — the tol-0.15 demo
stream moved walk10 only 0.85 -> 0.86 m/s (walk05 0.26 -> 0.34). Structure:
the clock pins cadence 0.91 Hz, the style mixture pins step at 0.48-0.50 m
(three in-band clips walk 0.48 m steps, two walk 0.56 m), so
0.91 x 2 x 0.49 = 0.86 m/s is the style-task equilibrium. v74b falsified
"more iterations"; v75 falsified "fetch tightening". The remaining lever on
the tradeoff is the TASK side of the equilibrium.

v76 = v75 recipe with track_lin_vel_xy_exp.weight 1.25 -> 2.0 (single
variable; the FIRST controlled reward-side dose added on top of the initial
table per the user's add-one-at-a-time protocol). Everything else
byte-identical: initial-15+clock reward, tol 0.15 VMATCH, v72 dataset.

Expected side effect (positive): as speed rises, per-step progression
grows and the over-long swing (0.62-0.72 s) should fall back toward the
0.43 s schedule window.

Base: embedded v75 m5000 (all-gates-pass, arc-best v-err 0.162) +1500.

PRE-REGISTERED acceptance:
  walk10 distance >= 10.7 m (>= 0.93 m/s) with ALL rhythm gates PASS
  walk05 >= 0.40 m/s
Decision tree:
  met                    -> delivery candidate; v77 arm-form chain
  speed up but gate broke -> mid-dose 1.6 re-launch from m5000
  no change              -> task-weight falsified for the plateau; next =
                            PRIME weights 1.2->3.0 (style-mass lever)
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- THE v76 VARIABLE: task weight dose --------------------------------------
os.environ["X1_TRACK_W"] = "2.0"

# --- the v75 recipe, byte-identical ------------------------------------------
os.environ["X1_VMATCH_TOL"] = "0.15"
os.environ["X1_INITIAL_REWARD"] = "1"
os.environ["X1_GAIT_CLOCK"] = "1"
os.environ["X1_FOOT_CLOCK_W"] = "1.0"
os.environ["X1_CONTACT_CLOCK_W"] = "0.5"

os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_AMP_NUM_STEPS"] = "30"
os.environ["X1_DISC_LINVEL"] = "1"
os.environ["X1_DISC_BUFFER"] = "24"
os.environ["X1_MOTION_DIR"] = "x1_lab_v72"
os.environ["X1_DISC_VMATCH"] = "1"

# --- resume from the EMBEDDED v75 m5000 (all-pass, best v-err) ---------------
_EMBEDDED = _here / "checkpoints" / "model_5000_v75.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    raise FileNotFoundError(f"v76 embedded base not found: {_EMBEDDED}")
os.environ["X1_FINE_TUNE_ITERS"] = "1500"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
