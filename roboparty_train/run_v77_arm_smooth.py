#!/usr/bin/env python3
"""v77 launcher: ARM-CHANNEL SMOOTHING SPLIT (X1_ARM_SMOOTH=1).

v76 verdict (TASK_20261024_046, evaluated under the operator's UPDATED
acceptance: walking speed >= 0.85 m/s sufficient):
  m5500 = DELIVERY CANDIDATE — 0.87 m/s, ALL walk10 rhythm gates PASS,
  v-err 0.152 (best among all-pass points), 4/4 P7 survival, back05 walks
  0.45 m/s, walk05 0.33 m/s. m6000/m6499 traded gates for speed (R2 0.66 /
  R3 0.41) — discarded; per the operator directive NO further speed doses.

Remaining delivery gaps (other sim2sim quality gates unchanged):
  arm form — P7b elbow p95>70 in all scenarios, P7g/P7h lean asym/shared,
  P7a anti on walk05/back05/walkturn (H5: caused by the uniform full-
  strength smoothing on the arm channels making arm motion expensive —
  the v73-v76 clean design dropped the v40 arm split along with the
  guards); robustness 46/60 vs gate 48; mesh clearance unverified.

v77 = v76 recipe (X1_TRACK_W=2.0 kept — it is now part of the delivery
recipe) + THE v40 SPLIT ONLY: base acc/rate terms narrowed to legs+torso,
arm terms at the v51 delivery doses (-6.25e-8 / -0.0025). Second
controlled addition from the initial table. Base: embedded v76 m5500
+1500.

PRE-REGISTERED acceptance:
  arm form: P7b elbow p95 <= 70 AND P7a anti <= -0.5 recovered in >= 2/4
    scenarios (H5 test)
  non-regression: walk10 rhythm gates ALL PASS at speed >= 0.85 m/s
    (distance >= 9.8 m); 4/4 survival; walk05 >= 0.30 m/s
Decision tree:
  arm recovered + rhythm held -> v78: sy/lean guards 0.5 if P7g/P7h
    residual, else DELIVERY (full battery + report)
  arm unchanged               -> H5 falsified -> v78: sy/lean guards 0.5
    directly (P7g/P7h exact statistics)
  rhythm regressed             -> fall back to m5500 delivery + guards-only
    fine-tune
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- THE v77 VARIABLE: v40 arm smoothing split -------------------------------
os.environ["X1_ARM_SMOOTH"] = "1"

# --- the v76 delivery recipe, byte-identical ---------------------------------
os.environ["X1_TRACK_W"] = "2.0"
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

# --- resume from the EMBEDDED v76 m5500 (delivery candidate) -----------------
_EMBEDDED = _here / "checkpoints" / "model_5500_v76.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    raise FileNotFoundError(f"v77 embedded base not found: {_EMBEDDED}")
os.environ["X1_FINE_TUNE_ITERS"] = "1500"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
