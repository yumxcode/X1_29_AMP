#!/usr/bin/env python3
"""v83 launcher: STANCE PACKAGE (user defect spec 2026-10-27: 劈腿走).

v80-v82 arc wrap (RECORD_BOOK section 10.8): delivery candidate v81-m2500
(rhythm all-pass 0.865 m/s, defect-1 toe-out FIXED, arms substantially
improved). NEW user defect report 2026-10-27: the two feet walk severely
wider than shoulder width (劈着腿走).

v83 audit (local, acceptance/v83_eval/):
  v81-m2500 rollout root-frame foot separation p50 447 mm (walk10) /
  477 mm (walk05), p90 527 -- vs X1 shoulder-joint width 292 mm, hip
  separation 268 mm, demo library median 306 mm (35_xx 281-318, 08_xx
  434-448, legacy treadmill 352-406).
  FK wipe attribution: hip_roll DC+AC ~157 mm + hip_yaw external ~103
  + hip_pitch interaction ~81 mm; hip_yaw+roll wiped to zero -> 198 mm
  (demo level).
  Mechanism: the initial feet_distance_y exp bonus band [0.2, 0.5] is
  FLAT at the defect (gradient 1e-7 at 0.15 m excess); the v73 reset
  restored max 0.50 (v69-v72 ran 0.38); demo mixture + ROBUST=3 favor
  wide support. v69 precedent: band 0.38 + abd guard took 460 -> 330.

v83 single variable = STANCE PACKAGE (two knobs on the SAME statistic):
  X1_STANCE_MAX      0.50 -> 0.30  (exp bonus band, applied AFTER the
                                    initial-reward reset -- ordering)
  X1_STANCE_HINGE_W  3.0           (NEW linear hinge beyond 0.32 m on a
                                    1 s EMA of root-frame separation;
                                    pays 0.13*3 = 0.39/step at the
                                    m2500 defect level, 0 inside band)
Base: embedded v81-m2500 (arc optimum) +1500 iters. All other knobs
byte-identical to run_v81_form_dose2.py.

PRE-REGISTERED acceptance (BINDING):
  F5 stance gates walk10: sep p50 <= 0.33 m AND p90 <= 0.38 m
    (sim2sim mujoco_rollout, sole-corner centers, walking-heading
     projection; demo 35_xx p50 281-318 / p90 338-395 envelope)
  constraints (non-regression, binding): rhythm R1-R4+L1 all-pass,
  speed >= 0.85 m/s (stretch 1.0), survival 4/4, walk05 >= 0.30 m/s,
  FORM F1/F2 (toe-out, sole-tilt) non-regression.
Decision tree:
  all pass          -> delivery candidate + full battery; then the 1 m/s
                       speed arc (separate variable)
  stance in, rhythm/speed regress
                    -> soup midpoint (v83-best, v81-m2500); if still
                       broken, hinge dose 3.0 -> 1.5 re-run
  stance stuck > 0.38 (Isaac-side; check hinge income in ep_reward
                       logs -- if ~0 the statistic is not computing, if
                       large but unmoving the wide stance is a structural
                       attractor)
                    -> v84 DATA round: demote 08_xx (0.5->0.15) + legacy
                       walks (0.7->0.3), the wide-stance demo families
  stance in-band Isaac but MuJoCo > 0.38
                    -> sim2sim-gap family (hip_yaw F3): deliver with F5
                       residual note; hip_yaw MuJoCo drift is registered
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- THE v83 VARIABLES: stance package -------------------------------------
os.environ["X1_STANCE_MAX"] = "0.30"        # exp bonus band 0.50 -> 0.30
os.environ["X1_STANCE_HINGE_W"] = "3.0"     # NEW linear hinge, far-field pull
os.environ["X1_STANCE_HINGE_MAX"] = "0.32"  # demo PRIME p50 envelope + margin

# --- v81 recipe (byte-identical) --------------------------------------------
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

# --- resume from the embedded v81 m2500 (arc optimum) ----------------------
_EMBEDDED = _here / "checkpoints" / "model_2500_v81.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    raise FileNotFoundError(f"v83 embedded base not found: {_EMBEDDED}")
os.environ["X1_FINE_TUNE_ITERS"] = "1500"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
