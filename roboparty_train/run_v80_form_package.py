#!/usr/bin/env python3
"""v80 launcher: v76 recipe + FORM PACKAGE, FROM SCRATCH (RECORD_BOOK §10).

User defect spec 2026-10-26 (rendered v76-m5500, the v73-v79 delivery
candidate):
  1. 双脚外翻行走 -- legs lift/land via roll, feet everted/toe-out
  2. 手臂往上举着走 -- arms held up/out

v80 form audit (acceptance/form_audit.py, FK calibrated against the sole
corner geometry; baseline JSON acceptance/v80_eval/form_audit_baseline.json):

  metric (walk10)          m5500            demos (healthy)
  toe-out med              +57 / +54 deg    -8..+20 deg
  hip_yaw DC / p95         +-31 / 64-66     +-5..18 / <=24
  hip_roll DC              -23.6 / +21.7    +-4..9
  stance sole tilt p95     65-67 deg        9-24 deg
  shoulder roll DC / p95   -74 / 94         -9 / 15
  shoulder pitch DC        -54 / -37        -21..+16
  elbow p95                78-91            ~20

=> the DEMO DATA is healthy; the defect is 100% reward/obs-side:
  (a) the AMP discriminator sees only 10 key-body POSITIONS (invariant
      under hip yaw / ankle roll -> structurally blind to eversion);
  (b) the initial 15-term table has ZERO foot/leg-orientation and arm
      posture terms;
  (c) the vendor DEFAULT_Q standing pose itself carries hip_yaw
      -+19.4 deg (toed-out action origin) and the policy amplified it.

v80 single variable (on top of the full v76 recipe) = X1_FORM_PACKAGE=1:
  foot_yaw_neutral      -0.6  (band 6 deg,  v69 calibration)
  hip_roll_abduction    -0.8  (band 4 deg EMA, v69 calibration)
  stance_sole_flat      -1.5  (phased, v27 dose)
  arm_dc_band           -0.3  (NEW complete 4-joint x 2-side DC anchor)
FROM SCRATCH (no resume): the m5500 attractor (toe-out + raised arms) is
a local optimum the added penalties would have to climb out of; scratch
lets the form terms act from iteration 0 (v73/v74 precedent: scratch
converges and passes all rhythm gates by m3000-3999).

PRE-REGISTERED acceptance (binding FORM gates -- acceptance/form_audit.py
--gate, thresholds calibrated on the demo library):
  FORM: F1 stance sole-tilt med<=10/p95<=30; F2 toe-out med<=15/p95<=35;
        F3 hip_yaw |DC|<=12/p95<=30; F4 hip_roll |DC|<=10;
        A1 shoP DC<=25, shoR DC<=20, shoY DC<=15, elbow p95<=70  (per foot/side)
  CONSTRAINTS (non-regression): R1-R4+L1 all-pass walk10; speed >=0.85 m/s;
        P7 survival 4/4; walk05 >=0.30 m/s.
Decision tree:
  FORM+constraints pass  -> delivery candidate + platform battery
  FORM pass, rhythm/speed regress -> dose halving re-launch (single-variable
                           confirmation), keep best checkpoint
  FORM partial (legs fixed, arms not / vice versa) -> escalate the failing
                           family dose ONLY (TOE/ABD vs ARM_DC), one round
  FORM no move -> check [FORM-PACKAGE] print + per-term income in the pod
                           log (RECORD_BOOK §6 effectiveness checklist)
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- from-scratch pre-flight (v74 precedent) ---------------------------------
os.environ.pop("X1_RESUME_CKPT", None)
os.environ.pop("X1_FINE_TUNE_ITERS", None)
_stray = sorted((_here.parent).glob("model_*.pt"))
if _stray:
    raise FileNotFoundError(
        f"FROM-SCRATCH pre-flight: repo root holds mounted checkpoints "
        f"{[p.name for p in _stray]} — run_x1_amp_train would auto-resume; "
        f"remove/rename them first")

# --- THE v80 VARIABLE: FORM PACKAGE ------------------------------------------
os.environ["X1_FORM_PACKAGE"] = "1"
os.environ["X1_TOE_GUARD"] = "0.6"
os.environ["X1_ABD_GUARD"] = "0.8"
os.environ["X1_SOFLAT_W"] = "1.5"
os.environ["X1_ARM_DC_W"] = "0.3"

# --- v76 delivery recipe (byte-identical infrastructure) --------------------
os.environ["X1_INITIAL_REWARD"] = "1"
os.environ["X1_GAIT_CLOCK"] = "1"
os.environ["X1_FOOT_CLOCK_W"] = "1.0"
os.environ["X1_CONTACT_CLOCK_W"] = "0.5"
os.environ["X1_TRACK_W"] = "2.0"
os.environ["X1_VMATCH_TOL"] = "0.15"

os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_AMP_NUM_STEPS"] = "30"        # 600 ms disc window @50Hz
os.environ["X1_DISC_LINVEL"] = "1"
os.environ["X1_DISC_BUFFER"] = "24"
os.environ["X1_MOTION_DIR"] = "x1_lab_v72"   # 36 sources, dual-gate PASS
os.environ["X1_DISC_VMATCH"] = "1"

os.environ["X1_FRESH_ITERS"] = "4000"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
