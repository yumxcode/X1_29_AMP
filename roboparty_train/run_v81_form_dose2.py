#!/usr/bin/env python3
"""v81 launcher: failing-family dose escalation (RECORD_BOOK section 10).

v80 verdict (TASK_20261026_227, scratch 4000, eval 2026-10-26):
  DEFECT 1 (everted/toe-out feet) FIXED: toe-out med 54-57 -> 0.5 deg,
  stance sole-tilt med 9-14 -> 1-9 / p95 65-67 -> 13-25 deg, hip_roll DC
  -23.6/+21.7 -> -3.8/+3.3 deg, lift pitch-dominance x3.0 -> x4.2.
  Rhythm R1-R4+L1 ALL PASS on m2000/3000/3999 (0.91 Hz x 0.49-0.52 m);
  m2000 speed 0.89 m/s >= 0.85 line, 4/4 survival, walk05 0.36 >= 0.30.
  REMAINING FORM fails: hip_yaw DC +-16 deg (feet compensated forward via
  ankle chain -- F3), arms shoR -36..-49 (gate 20), elbow p95 75-94 (70),
  shoP R -27..-38 (25). Known residual: walk05 drag (swing-height p50
  0.5-4 mm; the FORM terms made lifting costlier at low speed) -- walk05
  rhythm gates NOT binding this round, registered.

v81 single variable (per the pre-registered decision tree: escalate the
FAILING families only) -- ARM_DC dose -0.3 -> -0.9 + NEW hip_yaw DC band
(hip_yaw_neutral, the generic EMA-band func, band (0, 8 deg), weight -0.8).
Base: embedded v80 m2000 (all rhythm PASS, 0.89 m/s, best arms) +1500 iters.

PRE-REGISTERED acceptance (binding, same gates as v80):
  FORM gates F1-F4 + A1 ALL PASS walk10; constraints: rhythm all-pass,
  speed >= 0.85, survival 4/4, walk05 >= 0.30.
Decision tree:
  all pass            -> delivery candidate + full battery (P7, collision,
                         robustness, platform)
  arms in, hip_yaw in, but rhythm/speed regress -> soup midpoint with v80-m2000
  arms still out      -> arm bands are being outweighed: inspect episode
                         incomes; if arm_dc income ~= equilibrium at -0.9,
                         the style channel pays for the pose -- next single
                         variable = tighten shoR band 20 -> 12 deg (make the
                         same pose cost more) instead of another dose x2
  hip_yaw still out   -> escalate X1_HIPYAW_W 0.8 -> 1.6
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- THE v81 VARIABLES: failing-family doses -------------------------------
os.environ["X1_FORM_PACKAGE"] = "1"
os.environ["X1_TOE_GUARD"] = "0.6"
os.environ["X1_ABD_GUARD"] = "0.8"
os.environ["X1_SOFLAT_W"] = "1.5"
os.environ["X1_ARM_DC_W"] = "0.9"      # v80: 0.3 -> 0.9
os.environ["X1_HIPYAW_W"] = "0.8"      # v81 NEW: hip_yaw DC band 8 deg

# --- v80 recipe (byte-identical) --------------------------------------------
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

# --- resume from the embedded v80 m2000 (all-rhythm-PASS best point) -------
_EMBEDDED = _here / "checkpoints" / "model_2000_v80.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    raise FileNotFoundError(f"v81 embedded base not found: {_EMBEDDED}")
os.environ["X1_FINE_TUNE_ITERS"] = "1500"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
