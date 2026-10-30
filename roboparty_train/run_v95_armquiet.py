#!/usr/bin/env python3
"""v95 launcher: ARM_QUIET — quiet-hands guard (user defect 2026-10-30:
手臂摆动别扭), single variable = X1_ARM_QUIET_W -0.5 on top of the
DELIVERY_v94d_x7398 recipe (RECORD_BOOK 14).

DIAGNOSIS (acceptance/arm_swing_audit.py, same-caliber FK/joint stats on
x1_lab_v72 demos vs x7398 MuJoCo rollouts):
  demo arm = RIGID shoP PENDULUM: shoP carries ALL the swing (p2p 92-133
  deg; FK corr(phiL,phiR) -0.97..-0.98, clean contralateral coupling);
  shoR 2-7 deg RMS / shoY p2p <= 0.5 / elbow frozen exactly +20.0 /
  elbY+wrist frozen in 36/36 clips; hands held lat 0.13-0.25 m.
  x7398 defect = UNDER-swinging shoP (gait-band RMS 12-17 vs demo 26-47
  deg) + DISTAL FLAILING at gait frequency: shoY p2p 48-112 deg, elbP
  p2p 71-90 (RMS 20.5), elbY p2p 59-111, wriP/R p2p 35-67; corr(phiL,
  phiR) -0.15..-0.52 (discoordinated); hand lateral sweep p2p 0.56 m,
  vertical bob 0.53 m; L/R amplitude asym 46 deg. FFT: >=90% of the
  flail power sits in 0.4-3 Hz (POLICY-driven pumping, not PD osc).
WHY UNPUNISHED: arm_dc_band anchors EMA DC only (never swing); the disc
  sees the distal channels but they are CONSTANT in every demo (zero
  variance -> zero adversarial gradient).
LEVER: arm_distal_amp_guard (rewards.py) — swing-RMS bands on the 12
  distal channels [shoR/shoY/elbP/elbY/wriP/wriR x L/R], bands
  [8.0/2.9/2.9/2.9/9.2/7.4] deg (cover max demo RMS); shoP EXCLUDED.
  Demos pay ~0 (test_v95: 8 clips incl. PRIME = 0.000-0.009 rad);
  x7398 pays 1.32 rad walk10 / 1.75 rad walk05 -> -0.66/-0.87 per step
  at w=-0.5 (chest-guard scale, the biggest arm defect dose ever).
BASE: embedded v94d-x7398 (checkpoints/model_7398_v94d.pt) +1000,
everything else byte-identical to run_v94d_chestlean_burn.py.

PRE-REGISTERED acceptance (binding), on MuJoCo rollouts (walk10/walk05/
back05, arm_swing_audit.py):
  A1 shoY p2p <= 20 deg all scenarios        (defect 48-112)
  A2 elbP p2p <= 30 deg                      (defect 71-90)
  A3 wriP & wriR p2p <= 35 deg               (defect 35-67)
  A4 corr(phiL,phiR) <= -0.6 walk10 & walk05 (defect -0.33/-0.52)
  A5 hand lateral p2p <= 0.45 m              (defect up to 0.56)
NON-REGRESSION (binding): C1 chest (walk10 >= -2.0, walk05/back05 >=
-3.0) AND rhythm R1-R4+L1 AND F5 AND 4/4 AND walk10 speed >= 0.86 AND
walk05 >= 0.30 AND robustness >= 48 AND penetration clear.

Decision tree:
  all binding -> DELIVERY_v95 + full battery
  arms improve >= 50% but gate miss -> zero-new-vars same-dose burn-in
      +400..600 from the best ckpt (v94d precedent)
  arms barely move (< 30% summed distal p2p reduction) -> dose -0.9
  rhythm/speed/C1 breaks -> dose -0.25 restart from x7398
  hands quiet but shoP RMS < 20 deg (stiff look) -> next arc adds
      arm_amp_phase (v55 prior, pre-registered lever)
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- v95 VARIABLE: quiet-hands guard -----------------------------------------
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

os.environ["X1_RESUME_CKPT"] = str(_here / "checkpoints" / "model_7398_v94d.pt")
os.environ["X1_FINE_TUNE_ITERS"] = "1000"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
