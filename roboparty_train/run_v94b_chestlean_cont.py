#!/usr/bin/env python3
"""v94b launcher: CHEST LEAN GUARD continuation (RECORD_BOOK 13.3).

v94 (TASK_20261029_280, X1_CHEST_LEAN_W -2.0 from q4499 +1500, killed at
iter ~5956 with 5000/5500 mirrored) read-out:

  t5000: chest walk10 p50 -2.67 / t5500: -2.34 (from q4499's -9.9);
         walk05 -1.93 PASS / back05 +1.01 PASS (C1 two of three)
  rhythm: R1/R2/R3/L1 all-pass both; R4 step 0.432 -> 0.445 RECOVERING
         (q4499 0.453; the posture transition cost ~8mm step, giving back)
  speed 0.886/0.881 (binding >=0.86 PASS; q4499 0.936 local measure)
  F5 301/360 PASS (better than q4499 317/373) / 4/4 survival / penetration
  all-clear (62.6-139mm)

Both remaining misses (C1 walk10 -2.34 vs -2.0; R4 0.445 vs 0.45) are
<0.5-unit marginal AND on a live improving trend over 5000->5500. The
pre-registered tree branch 'chest moves but < -2 -> dose -4.0' assumed a
STUCK equilibrium; the observed state is a moving one, so the cheapest
single variable is CONTINUATION at the same dose (not a second dose that
would re-perturb the recovering R4/step).

v94b = same recipe byte-identical (chest -2.0, step-drive 2.0, robust 1),
base embedded v94-t5500 +1000 (ladder read 6000/6500).

PRE-REGISTERED acceptance (binding): C1 walk10 p50 >= -2.0 AND rhythm
R1-R4+L1 all-pass AND F5 AND 4/4 AND walk10 speed >= 0.86 AND walk05 >=
0.30 AND robustness >= 48 AND penetration clear.

Decision tree:
  all binding gates     -> v94b delivery + full battery (robustness sweep
                           re-run on the final ckpt) + RECORD_BOOK verdict
  chest stalls <= -2.2 OR R4 < 0.44 at 6500
                         -> v94c dose escalation X1_CHEST_LEAN_W -3.0 from
                           best v94b ckpt +500 (chest push with the margin
                           R4 recovered); if R4 breaks instead, band-shift
                           X1_CHEST_LEAN_LO 0.0 -> -3.0 (accept -3 chest)
                           as the operator-adjudication fallback
  chest over-corrects > +20 (upper hinge holds by construction) or any
  OTHER gate breaks -> dose halve -1.0 (v94c-low) from t5500
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- THE v94 VARIABLE (unchanged): chest lean guard --------------------------
os.environ["X1_CHEST_LEAN_W"] = "-2.0"
os.environ.setdefault("X1_CHEST_LEAN_LO", "0.0")
os.environ.setdefault("X1_CHEST_LEAN_HI", "0.35")

# --- v91b recipe (byte-identical) --------------------------------------------
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

os.environ["X1_RESUME_CKPT"] = str(_here / "checkpoints" / "model_5500_v94.pt")
os.environ["X1_FINE_TUNE_ITERS"] = "1000"
os.environ["X1_STEP_DRIVE_W"] = "2.0"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
