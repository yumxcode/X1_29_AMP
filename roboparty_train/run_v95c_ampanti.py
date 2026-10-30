#!/usr/bin/env python3
"""v95c launcher: AMP-under-ANTIPHASE (audit item 2, RECORD_BOOK 14.5).
Single variable = X1_AMP_ANTI_W 0.25 (v55 dose scale) on top of the v95
recipe. Base: embedded v95-y8397 (model_8397_v95.pt) +800.

WHY (audit 2026-10-30): the A4 gate corr(phiL,phiR) <= -0.6 was never
actually evaluated (arm_gates.py name-prefix bug + flipped op), and the
corrected measurement shows the L/R antiphase dimension UNRESOLVED:
  y8397 walk10/-0.352 walk05/-0.100 (demo -0.98; x7398 -0.522/-0.329 —
  the v95 delivery is WORSE than baseline on both gated scenarios).
Mechanism (measured on the reward's own channels): the residual is the
  L/R SYNC component RMS(hpL+hpR) = A*sqrt(2(1+rho)): demos 5-13 deg,
  policies 17-25 deg, plus a walk05 amplitude asymmetry (29/11 deg).
v55's arm_amp_phase_prior is structurally broken for this job: its
  arm-leg product sums a +0.98 and a -0.91 pair -> ~0 on the NATURAL
  demos (the v33 negation was never ported; gate active on 9-29% of demo
  frames) — verified in test_v95c_ampanti.py docstring.
LEVER: arm_amp_antiphase_prior (rewards.py) — v52/v55 amplitude band on
  min(L,R) shoP RMS [0.52, 0.79] rad x sync gate (cap 0.24 rad = demo
  max + margin). Demos 0.62-1.00, y8397 0.07-0.24 (test ALL PASS).
  Farming it requires BOTH arms swinging demo-large AND killing the sync
  component -> corr(phiL,phiR) toward the demo -0.98.

PRE-REGISTERED acceptance (binding): A4 CORRECTED corr(phiL,phiR)
  <= -0.6 in walk10 AND walk05 (arm_gates.py fixed); arms A1/A2/A3/A5
  no worse than y8397's counts; NON-REGRESSION: C1 (walk10 >= -2.0,
  walk05/back05 >= -3.0), rhythm R1-R3+L1, F5, 4/4, walk10 >= 0.86,
  walk05 >= 0.30, penetration clear, robustness >= 48. R4 tracked
  (pre-existing fail at x7398 0.447 / y8397 0.441).

Decision tree:
  A4 + non-regression all pass -> DELIVERY_v95c (replaces y8397) + battery
  A4 improves (<= -0.45 both) but short -> +400-600 same-dose burn-in
  A4 barely moves / A-gates regress -> dose 0.5 one round
  non-regression breaks (C1/speed/F5) -> adjudication with the corrected
      A4 data: arms-vs-chest-vs-speed frontier + coherence dimension
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- v95c VARIABLE: amp-under-antiphase prior ---------------------------------
os.environ["X1_AMP_ANTI_W"] = "0.25"

# --- v95/v94d recipe (byte-identical) ------------------------------------------
os.environ["X1_ARM_QUIET_W"] = "-0.5"
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
os.environ["X1_FINE_TUNE_ITERS"] = "800"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
