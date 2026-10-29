#!/usr/bin/env python3
"""v94 launcher: CHEST LEAN GUARD — the 腰后仰 fix (RECORD_BOOK 13).

DIAGNOSIS (2026-10-29, acceptance/torso_lean_stats.py + demo_lumbar_survey.py
+ test_v94_chestlean.py): the v91b-q4499 delivery pins the chest at -9.9 deg
(walk10 p50; walk05 -8.5 / back05 -4.2) — chest BACKWARD, the user-observed
腰后仰. Every demo in the loaded stream holds the chest at 0..+6 deg
(nearly vertical; FK medians 07_01 +2.3 / 08_03 +0.1 / 38_02 +5.9 / 35_02
+0.8 / 82_12 +0.2). Three structural facts let the defect persist:

  1. the discriminator CANNOT see the chest — KEY_BODY_NAMES (knees,
     ankles, elbows, shoulders, wrists, all base-relative) has no torso
     body, and the arm-joint space can compensate the visible shoulder
     positions (the v33 attribution probe: with all form guards off only
     arm-DC and TORSO regressed — both guard-carried, and lumbar_posture
     was ZEROED in the v73+ initial-reward reset);
  2. the lumbar state is BISTABLE: continuation lineage v83-v86/v88/v89m3000
     sit at lumbar +26..+28 (chest +13, inside band) while the v87-scratch
     lineage (j/m5k+/n/q/r/s incl. q4499) pin lumbar -10 (chest -10);
     v89 flipped +26 -> -10 INSIDE one run (m3000 -> m5000, +0.05 speed) —
     the -10 basin pays lumbar_posture 0.174/step yet wins on net reward;
  3. nothing else references the chest angle (flat_orientation acts on the
     BASE; lumbar_posture is joint-space and compensation-blind: policy
     parks lumbar -10 with pelvis -3 and the chest stays -10).

v94 single variable = X1_CHEST_LEAN_W -2.0: hinge on the heading-frame
base->shoulder-mid tilt, FREE in [0, +20.05] deg (0 = chest vertical; the
demo stream 0..+6 and the +27-lineage chest +13 both untaxed), linear
(radians) outside. At q4499's -9.9 deg the pull is -0.173 x 2.0 =
-0.35/step (~13% of the 2.65/step task stream — strong enough to move a
bistable equilibrium, ~2x the historical lumbar_posture dose).

Base: embedded v91b-q4499 +1500 (ladder 5000/5500/5998), else v91b recipe
byte-identical (STEP_DRIVE 2.0, ROBUST=1, clock 1.61/0.66, track
2.25/std 0.25, STANCE + FORM, v72 base data).

PRE-REGISTERED acceptance (binding):
  C1 (the user defect): chest_tilt_p50 walk10 in [-2, +20] deg, walk05 and
    back05 >= -3 deg (acceptance/chest_lean_gates.py)
  AND non-regression: rhythm R1-R4+L1 all-pass, F5 p50<=0.33/p90<=0.38,
  survival 4/4, walk10 speed >= 0.86, walk05 >= 0.30, robustness >= 48/60,
  no-penetration.

Decision tree:
  C1 + all gates        -> v94 delivery + full battery + RECORD_BOOK 13
  chest moves but p50 walk10 < -2 (e.g. -10 -> -5)
                         -> dose escalation -2.0 -> -4.0 (+500, v94b)
  chest fixed (>= -2) but a gate breaks
                         -> dose halve -1.0 (v94c); if still broken at -1.0,
                           band-shift diagnostic (lower edge +4) before
                           structural levers
  chest flips over-forward > +20 -> upper hinge holds by construction;
                           verify walk05/back05 too, else dose halve
  chest does not move (< 3 deg change at -2.0)
                         -> the -10 basin is not reward-reachable at this
                           dose: register the CHEST-OBS structural lever
                           (add a torso key body to the disc obs + dataset
                           FK rebuild) — bigger arc, operator adjudication
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- THE v94 VARIABLE: chest lean guard --------------------------------------
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

os.environ["X1_RESUME_CKPT"] = str(_here / "checkpoints" / "model_4499_v91b.pt")
os.environ["X1_FINE_TUNE_ITERS"] = "1500"
os.environ["X1_STEP_DRIVE_W"] = "2.0"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
