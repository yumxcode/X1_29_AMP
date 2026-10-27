#!/usr/bin/env python3
"""v83e launcher: final shot — iterations-only from v83c-c4998 (RECORD_BOOK 11.7).

v83b verdict (TASK_20261027_221, iterations-only +1000 from v83-m3999):
speed STUCK 0.81-0.83 m/s (<0.85 line) across the ladder; F5 held
(b4000 311/385, b4500 307/371, b4998 319/387 -- p90 creeping +5-7mm);
rhythm broke at b4500/b4998 (L 1.85 Hz micro-step asymmetry, the
continuation-drift family) -- the pre-registered branch fires:
STRUCTURAL narrow-stance/speed trade -> task weight round.

v83c single variable = X1_TRACK_W 2.0 -> 2.5 (v76 precedent: 2.0 moved
0.86 -> 0.92 but broke rhythm THEN; retested under the stance hinge +
clock anchors). Base: embedded v83-m3999 (F5 308/373 PASS + rhythm
ALL-PASS, the best all-gates point) +1000.

PRE-REGISTERED acceptance (binding): speed >= 0.85 m/s AND F5
non-regression (p50<=0.33/p90<=0.38) AND rhythm all-pass AND survival
4/4 AND walk05 >= 0.30.
Decision tree:
  all pass      -> delivery candidate + full battery + platform; the
                   1 m/s contract stretch continues only if a further
                   single variable exists without gate breakage
  speed in, F5/rhythm break
                -> ladder midpoints; none holds -> intermediate dose
                   2.25 from m3999; still none -> report the Pareto
                   frontier to the operator (speed vs stance/rhythm)
  speed <0.85   -> task-side saturated: candidates = clock-bell speed
                   coupling / VMATCH PRIME reweight / scratch 4000 with
                   the stance package; operator adjudicates (0.85 line
                   vs 1.0 contract stretch)
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- THE v83c VARIABLE ------------------------------------------------------
os.environ["X1_TRACK_W"] = "2.5"             # v81/v83 recipe: 2.0

# --- v83 recipe (byte-identical) -------------------------------------------
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
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_AMP_NUM_STEPS"] = "30"
os.environ["X1_DISC_LINVEL"] = "1"
os.environ["X1_DISC_BUFFER"] = "24"
os.environ["X1_MOTION_DIR"] = "x1_lab_v72"
os.environ["X1_DISC_VMATCH"] = "1"

_EMBEDDED = _here / "checkpoints" / "model_4998_v83c.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    raise FileNotFoundError(f"v83c embedded base not found: {_EMBEDDED}")
os.environ["X1_FINE_TUNE_ITERS"] = "1000"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
