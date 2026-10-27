#!/usr/bin/env python3
"""v83b launcher: iterations-only continuation (RECORD_BOOK section 11).

v83 verdict (TASK_20261027_212, +1500 from embedded v81-m2500, eval
2026-10-27): STANCE DEFECT FIXED -- walk10 sep p50 447 -> 308-309 mm
(demo 35_xx band 281-318) / p90 527 -> 371-373, F5 gates PASS on
m3500/m3999; rhythm ALL-PASS m3999 (0.909 Hz x 0.489 m); survival 4/4.
REMAINING: speed 0.826 m/s < 0.85 line (the narrow-stance cost), walk05
drag (apex 1.8 mm, v80-era family residual), F3 hip_yaw / A1 arms
(known non-binding families).
Soup midpoint branch EXECUTED AND FALSIFIED: 50/75/85 mixes all restore
sep 331-370 (stance not linearly separable in weight space) with NO
speed recovery (0.822-0.823) -- soup dead end.

v83b single variable = iterations only (+1000 from embedded v3999).
Speed trend active (m3500 0.778 -> m3999 0.826); hinge at band edge
(-0.026/step) so remaining optimization budget flows to task reward.

PRE-REGISTERED acceptance (binding):
  speed >= 0.85 m/s AND F5 non-regression (p50 <= 0.33 / p90 <= 0.38)
  AND rhythm all-pass AND survival 4/4.
Decision tree:
  all pass             -> delivery candidate + full battery (P7/collision/
                          robustness) + platform; then the 1 m/s stretch
                          arc per the task contract
  speed in, F5 regresses-> ladder midpoint; if none holds both, v83-m3999
                          stands as the stance-fix delivery and the speed
                          trade goes to a task-weight round (2.0 -> 2.5)
  speed still < 0.85   -> structural narrow-stance/speed trade confirmed:
                          task weight 2.0 -> 2.5 single-variable round from
                          the best ladder point (v76 precedent moved
                          0.86 -> 0.92; rhythm broke THEN -- retest under
                          the stance hinge + clock anchors)
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- v83 recipe (byte-identical) -------------------------------------------
os.environ["X1_STANCE_MAX"] = "0.30"
os.environ["X1_STANCE_HINGE_W"] = "3.0"
os.environ["X1_STANCE_HINGE_MAX"] = "0.32"
os.environ["X1_FORM_PACKAGE"] = "1"
os.environ["X1_TOE_GUARD"] = "0.6"
os.environ["X1_ABD_GUARD"] = "0.8"
os.environ["X1_SOFLAT_W"] = "1.5"
os.environ["X1_ARM_DC_W"] = "0.9"
os.environ["X1_HIPYAW_W"] = "0.8"
os.environ["X1_INITIAL_REWARD"] = "1"
os.environ["X1_GAIT_CLOCK"] = "1"
os.environ["X1_FOOT_CLOCK_W"] = "1.0"
os.environ["X1_CONTACT_CLOCK_W"] = "0.5"
os.environ["X1_TRACK_W"] = "2.0"
os.environ["X1_VMATCH_TOL"] = "0.15"
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_AMP_NUM_STEPS"] = "30"
os.environ["X1_DISC_LINVEL"] = "1"
os.environ["X1_DISC_BUFFER"] = "24"
os.environ["X1_MOTION_DIR"] = "x1_lab_v72"
os.environ["X1_DISC_VMATCH"] = "1"

_EMBEDDED = _here / "checkpoints" / "model_3999_v83.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    raise FileNotFoundError(f"v83b embedded base not found: {_EMBEDDED}")
os.environ["X1_FINE_TUNE_ITERS"] = "1000"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
