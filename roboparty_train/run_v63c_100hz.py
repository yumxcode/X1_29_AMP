#!/usr/bin/env python3
"""v63c launcher: SHORT repair continuation from v63's m10500 (contract rev4).

v63b verdict (TASK_20260921_064): yaw guard 1.0->2.5 BACKFIRED — arms
40/40 -> 24.8/20.6 (m11000) -> 17.1/14.2 (m11299), arm-DC asym exploded to
P7g 48 deg, and the final point FALLS on-pod. The late-training drift was
NOT a yaw-guard-strength problem.

v63 residual at the honest 100 Hz + LPF evaluation (local, correct rate —
the on-pod P7 ran the 50 Hz default, a bug fixed in this commit):
  m10500: arms 40.0/40.0 deg, drift -0.31/-0.52 m, 5/5 scenarios survived,
          K1 11.1, robustness 47/60, P7 7/8 — ONLY P7g arm-lean-asym
          12.8 deg vs gate 12 (R arm frozen ~11 deg back; family trait,
          guard-carried per the v32b attribution).
  m10592: P7g 13.7. m11000: P7g 48 (v63b-corrupted).

v63c single calibrated lever on a SHORT run:
  X1_ARM_ASYM=-2.4   arm_asym_lean -1.2 -> -2.4 (v53 precedent: at this
                     strength the guard pulled 35-54 deg DC asymmetries
                     back under the gate within one run) — targets P7g.
  X1_YAW_GUARD=1.0   v63 value (v63b's 2.5 reverted — it hurt arms AND
                     left the yaw oscillation: m11000 -60 deg / m11299
                     +54 deg heading swings either way).
  +600 iters         TWO candidate checkpoints (m11000 at the save
                     interval + m11100 final): the late yaw oscillation
                     phase is not controllable, so the deliverable point
                     is picked by local battery between the two.
  P7 control-dt fix  the on-pod P7 now rolls at the policy's trained
                     rate (X1_CONTROL_HZ -> --control-dt) — the honest
                     reading this time.

PRE-REGISTERED: VALID iff the best of {m11000, m11100} holds m10500's
quality AND closes P7g: arms >= 24 both sides, drift <= 1.0, P7 8/8 at
100 Hz (P7g <= 12), platform 13/13 (short window -> P4 flat), robustness
>= 48/60. Otherwise m10500 (v63) stands as the route-7 deliverable and
P7g joins the reported residual list."""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_CONTROL_HZ"] = "100"
os.environ["X1_MOTION_DIR"] = "x1_lab_v32_200"
os.environ["X1_AMP_NUM_STEPS"] = "60"
os.environ["X1_DISC_STRIDE"] = "2"
os.environ["X1_DISC_BUFFER"] = "24"
os.environ["X1_DISC_LINVEL"] = "1"
os.environ["X1_ACT_LPF_HZ"] = "10"
os.environ["X1_ACT_LPF_ORDER"] = "2"
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_FINE_TUNE_ITERS"] = "600"
os.environ["X1_YAW_GUARD"] = "1.0"      # v63 value (v63b's 2.5 reverted)
os.environ["X1_ARM_ASYM"] = "-2.4"      # v63c: P7g repair lever (v53 strength)

# X1_RESUME_CKPT unset -> auto-detect the platform-mounted model_*.pt at
# repo root (the m10500 from TASK_20260921_038).

_HERE = Path(__file__).resolve().parent
sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
