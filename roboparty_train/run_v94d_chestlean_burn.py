#!/usr/bin/env python3
"""v94d launcher: TREND BURN-IN at the v94c recipe (RECORD_BOOK 13.5).

v94c (TASK_20261029_350, STEP_DRIVE 2.5 + chest -2.0 from u6000 +800) read-out:
  w6500 mid-transition dip (chest -3.80 / speed 0.767 / step 0.441)
  w6799 ENDPOINT, all four gates within noise and ALL trending up at kill:
    chest -2.02 (band >= -2.0, short 0.02 deg) / speed 0.859 (band 0.86,
    short 0.001) / step 0.448 (R4 0.45, short 0.002) / F5 p90 384 (380,
    +4mm); walk05 0.443 BEST-EVER (q4499 0.374 / u6000 0.409).
  The 800-iter budget ended mid-recovery (max_iterations cap, not basin).
v94d = ZERO new variables: same recipe byte-identical (chest -2.0,
step-drive 2.5), base embedded v94c-w6799 +600. Binding gates unchanged
(C1 + R1-R4+L1 + F5 + 4/4 + speed >= 0.86 + walk05 >= 0.30 + robust >= 48
+ penetration). Tree: all-pass -> DELIVERY_v94d + full battery; any gate
still marginally short (< 0.03 units) -> ONE more +400 burn-in then
adjudicate; regression -> fall back to adjudication u6000 vs q4499.

v94c launcher docstring (kept for the record):

v94b (TASK_20261029_317, chest -2.0 continuation from t5500 +1000) ladder:

  u6000: C1 ALL-PASS (walk10 -1.59 / walk05 -0.70 / back05 +0.70) — the
         腰后仰 defect is FIXED at median level, chest oscillates
         -4.8..+5.7 with the gait phase (demo-like); F5 308/358 PASS,
         4/4, penetration clear, walk05 0.409 (BETTER than q4499 0.374)
  u6499: chest drifted back (-3.87) — u6000 is the sweet spot of the run
  BOTH: walk10 speed 0.836/0.826 (binding 0.86 FAIL) and R4 step 0.439
        (< 0.45) — SAME root cause: -0.05..-0.10 m/s speed at the upright
        chest equilibrium. NOTE the chest guard's ongoing cost at u6000 is
        only -0.02/step — the speed loss is the POSTURE's balance economy,
        not the penalty pressure; relaxing the dose/band cannot recover it
        (dose-response w0 -> -9.9 chest, w2 -> -1.6: halving the dose just
        re-opens the chest).

The v94b tree branch 'other gate breaks -> dose halve -1.0' assumed the
CHEST gate would break; instead the chest PASSED and SPEED broke. Dose
halving would trade C1 back away (predicted -4..-6 chest). The lever must
be on the SPEED side at the fixed chest:

v94c single variable = X1_STEP_DRIVE_W 2.0 -> 2.5 (the v91-proven lever's
mid-dose; v92's w3.0 regression was measured from the -10 chest base,
the trade is not transferable). Base: embedded v94b-u6000 +800
(ladder 6400/6799 final).

PRE-REGISTERED acceptance (binding): C1 (walk10 >= -2.0, walk05/back05
>= -3.0) AND rhythm R1-R4+L1 all-pass AND F5 AND 4/4 AND walk10 speed
>= 0.86 AND walk05 >= 0.30 AND robustness >= 48 AND penetration clear.

Decision tree:
  all binding gates -> DELIVERY_v94c + full battery + RECORD_BOOK 13.5
  step recovers (>=0.45) but chest breaks (< -2.0)
                       -> the C1-vs-speed frontier is dose-dependent:
                          try w2.25 from u6000 (+500) as the midpoint
  speed recovers but step still < 0.45
                       -> R4's floor is speed-derived at 1 m/s design
                          point; adjudicate u6000/v94c on the C1+speed
                          trade (operator note in RECORD_BOOK)
  nothing recovers (speed < 0.86, step < 0.44, chest < -2)
                       -> the upright-chest equilibrium caps ~0.83-0.84:
                          ADJUDICATION — u6000 (C1 fixed, -0.06 speed vs
                          q4499) as the defect-fix delivery with the trade
                          registered, vs keeping q4499. No further rounds
                          without operator input.
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- v94c VARIABLE: step-drive second dose -----------------------------------
os.environ["X1_STEP_DRIVE_W"] = "2.5"

# --- v94/v94b recipe (unchanged) ----------------------------------------------
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

os.environ["X1_RESUME_CKPT"] = str(_here / "checkpoints" / "model_6799_v94c.pt")
os.environ["X1_FINE_TUNE_ITERS"] = "600"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
