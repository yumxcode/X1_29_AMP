#!/usr/bin/env python3
"""v95b launcher: ARM_QUIET burn-in (RECORD_BOOK 14.1, exp_fe1b6795 tree
branch: 'arms improve >= 50% but gate miss -> zero-new-vars same-dose
burn-in +400..600 from the best ckpt').

v95 (TASK_20261030_081, X1_ARM_QUIET_W=-0.5 from x7398 +1000) read-out:
  arm gate failures 26 (x7398) -> 20/13/12 at y7500/y8000/y8397
  steady-state (2 s+ of the MuJoCo rollout) arms NEARLY QUIET:
    elbP p2p 52-91 -> 9-10 deg (std 3), shoY 45-102 -> 6-7, elbY 59-111
    -> 18-33 full-window; shoR DC -26..-39 -> -18..-21; DC asym 17-37 ->
    6-8 deg; corr(phiL,phiR) A4 PASS; hand lat p2p 0.56 -> <=0.26 ALL A5
    PASS. Remaining full-window p2p (elbP 51-64) is the COLD-START
    transient (first ~2 s settle from standing init), not steady pumping.
  guard income during training fell -1.22 -> -0.72 (not converged).
  Non-regression at y8397: C1 PASS, speed 0.873 >= 0.86, walk05 0.424,
    F5 278/319 PASS, no falls; R4 0.441 vs x7398 0.447 (already-failing
    band edge, family jitter scale).

v95b = ZERO new variables: same recipe byte-identical, embedded v95-y8397
+600. Tree after this: all gates -> DELIVERY; still marginal -> +400 then
adjudicate; regression -> fall back to y8397/x7398 adjudication.
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- v95 VARIABLE (unchanged): quiet-hands guard ------------------------------
os.environ["X1_ARM_QUIET_W"] = "-0.5"

# --- v94d recipe (byte-identical) ---------------------------------------------
os.environ["X1_STEP_DRIVE_W"] = "2.5"
os.environ["X1_CHEST_LEAN_W"] = "-2.0"
os.environ.setdefault("X1_CHEST_LEAN_LO", "0.0")
os.environ.setdefault("X1_CHEST_LEAN_HI", "0.35")
os.environ["X1_ROBUST_TRAIN"] = "1"
os.environ["X1_CLOCK_T_A"] = "1.61"
os.environ["X1_CLOCK_T_B"] = "0.66"
os.environ["X1_TRACK_STD"] = "0.25"
os.environ["X1_TRACK_W"] = "2.25"
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
os.environ["X1_VMATCH_TOL"] = "0.15"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_AMP_NUM_STEPS"] = "30"
os.environ["X1_DISC_LINVEL"] = "1"
os.environ["X1_DISC_BUFFER"] = "24"
os.environ["X1_MOTION_DIR"] = "x1_lab_v72"
os.environ["X1_DISC_VMATCH"] = "1"

os.environ["X1_RESUME_CKPT"] = str(_here / "checkpoints" / "model_8397_v95.pt")
os.environ["X1_FINE_TUNE_ITERS"] = "600"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
