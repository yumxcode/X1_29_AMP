#!/usr/bin/env python3
"""v81b launcher: COMPLETE the v81 dose run after the account-49 billing kill.

v81 (TASK_20261026_284) died silently at iter 2741/3500 (no traceback,
ep_len 965-988 healthy — the family billing-kill pattern; 3rd account so).
At +500 iters (m2500) the escalation is working: elbow p95 66.5/66.7
IN-BAND (v80: 94/81), shoR -28/-28 (v80 -36..-49, gate 20), shoP -6/-26,
hip_yaw -10.5/+18.0, rhythm ALL-PASS, 0.865 m/s.

v81b = the SAME v81 recipe/doses (ARM_DC -0.9 + hip_yaw band -0.8),
resumed from the embedded v81-m2500 +1000 iters (completing the
pre-registered 3500 total). Pre-registered acceptance unchanged
(RECORD_BOOK 10.5): FORM F1-F4+A1 all-pass + rhythm + >=0.85 m/s.
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- THE v81 VARIABLES (unchanged from v81): failing-family doses ---------
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

# --- resume from the embedded v81 m2500 (the billing-kill survivor) -------
_EMBEDDED = _here / "checkpoints" / "model_2500_v81.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    raise FileNotFoundError(f"v81b embedded base not found: {_EMBEDDED}")
os.environ["X1_FINE_TUNE_ITERS"] = "1000"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
