#!/usr/bin/env python3
"""X1-side DISC-DIAG control (drift-review adopted, 2026-11-12).

Purpose: the F1w collapse attribution chain closed with "the remaining top
disc-separating channels are X1-parity items" (fnd_339a00f7) — a channel-
IDENTITY inference, never measured with the same tool on the X1 side. This
control runs the HEALTHY X1 recipe FRESH for 60 updates with DISC_DIAG_AT=25
so the shared v81b diagnostic dumps X1-side per-channel |dmu|/pooled-sigma at
the SAME TRAINING AGE as the F1w v2j/v2k dumps (update 25, fresh start).

Comparison decides the next-phase axis:
  X1 sigma ~= F1w sigma  -> demo-source switch loses weight; the separator
                            depth difference lives in training dynamics /
                            reward coupling (tier-6 channel dropout line).
  X1 sigma << F1w sigma  -> the demo family really is more separable on
                            F1w; GMR solver repair becomes first-choice.

Recipe: byte-identical to run_v89_scratch8000.py (the healthy X1 fresh
recipe, x1_lab_v72 library) except X1_FRESH_ITERS 8000 -> 60 and
DISC_DIAG_AT=25.
"""
import os
import runpy
from pathlib import Path

_here = Path(__file__).resolve().parent

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

# --- CONTROL VARIABLES ------------------------------------------------------
os.environ["X1_FRESH_ITERS"] = "60"      # just past the DIAG trigger
os.environ["DISC_DIAG_AT"] = "25"        # same update age as F1w v2j/v2k dumps
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys_path = Path(__file__).resolve().parent
import sys
sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
