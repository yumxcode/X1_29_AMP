#!/usr/bin/env python3
"""v72 launcher: v71 gait-clock recipe + THE NEW LOW-CADENCE DATASET.

Single variable vs v71 (TASK_20261017_051, from-scratch 4000): the demo
library. X1_MOTION_DIR=x1_lab_v67 -> x1_lab_v72.

Dataset (audit 2026-10-22 fix order, evidence acceptance/evidence/):
  x1_lab_v72 = 29 CMU low-cadence clips (robot scale 0.80-1.09 Hz/foot,
  step 0.38-0.59 m, planted 75-96%, stance drift <0.05 m/s — the
  stance-anchored root_xy rebuild replaced the human-scale translation)
  + 7 anchored legacy walks (0000/0002/0005/0007/0008/0026/103_07);
  jogs + CMU 36_xx stepping-platform clips REMOVED; J-gate PASS 0 fail.
  Weights: speed x cadence 2D buckets (PRIME 1.2 / MID 0.7 / FAST 0.5).

Mechanism stack is byte-identical to v71 (clock obs +4, foot_clock 1.0,
contact_clock -0.5, hip_phase 0.4, 600ms disc window, VMATCH, v68
identified dynamics, v69/v70 guard doses, 50Hz).

PRE-REGISTERED accept = v71's A (non-regression battery) + B (R1-R6+L1
binding on walk10+walk05, acceptance/rhythm_gates.py).
Decision tree:
  A+B pass     -> deliver (soup candidate per v69 flow);
  A pass, B fail with R2/R3 stuck as in v71 -> data route falsified for
                 the swing-duration dimension (report honestly, next arc
                 = continuous-duration swing constraint / T* curriculum);
  A fail       -> v72b continuation round (iterations-only, v71b lesson:
                 recipe byte-identical, no dose change while ladder
                 improves monotonically).
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- from-scratch pre-flight: no resume checkpoint anywhere near -----------
os.environ.pop("X1_RESUME_CKPT", None)
os.environ.pop("X1_FINE_TUNE_ITERS", None)
_stray = sorted((_here.parent).glob("model_*.pt"))
if _stray:
    raise FileNotFoundError(
        f"FROM-SCRATCH pre-flight: repo root holds mounted checkpoints "
        f"{[p.name for p in _stray]} — run_x1_amp_train would auto-resume; "
        f"remove/rename them first")

# --- soup59c validated base -------------------------------------------------
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_YAW_GUARD"] = "1.0"

# --- disc observability + data side -----------------------------------------
os.environ["X1_AMP_NUM_STEPS"] = "30"        # 600 ms disc window @50Hz
os.environ["X1_DISC_LINVEL"] = "1"
os.environ["X1_DISC_BUFFER"] = "24"
os.environ["X1_MOTION_DIR"] = "x1_lab_v72"   # THE v72 VARIABLE: low-cadence lib
os.environ["X1_DISC_VMATCH"] = "1"           # speed-matched demo fetch

# --- v69/v70 delivery-dose posture guards ----------------------------------
os.environ["X1_TOE_GUARD"] = "0.6"
os.environ["X1_ABD_GUARD"] = "0.8"
os.environ["X1_SWAY_GUARD"] = "0.4"
os.environ["X1_STANCE_MAX"] = "0.38"
os.environ["X1_SY_GUARD"] = "0.5"
os.environ["X1_LEAN_GUARD"] = "0.5"

# --- the v71 mechanism: exogenous gait clock (unchanged) --------------------
os.environ["X1_GAIT_CLOCK"] = "1"
os.environ["X1_FOOT_CLOCK_W"] = "1.0"
os.environ["X1_CONTACT_CLOCK_W"] = "0.5"
os.environ["X1_HIP_PHASE"] = "0.4"
os.environ["X1_CADENCE_PRIOR"] = "0"
os.environ["X1_SWING_PRIOR"] = "0"

# --- from-scratch budget ----------------------------------------------------
os.environ["X1_FRESH_ITERS"] = "4000"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
