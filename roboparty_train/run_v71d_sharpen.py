#!/usr/bin/env python3
"""v71d launcher: KERNEL SHARPENING sigma_z 0.03 -> 0.02, resume m4500.

Dose-ladder verdict (v71c, w=1.5/-1.0 from m4500, +2000 iters):
  m6499c REGRESSED phase-lock — cadence back to 1.43/1.82Hz, step 0.27/0.16m
  (duty 0.72-0.76 ok, lift 46-49mm ok): doubling the hinge price broke the
  clock-cadence equilibrium instead of lengthening swing. m4500 (w=1.0/-0.5)
  remains the arc optimum (cadence 0.85/1.06Hz in-band, lift 39/33mm,
  v-err 0.188) with the single structural defect: swing 0.145s < 0.28 gate
  (brief apex pass, early touchdown).

v71d single variable (back at the m4500 doses 1.0/-0.5):
  X1_FOOT_SIGMA_Z 0.03 -> 0.02
Mechanism: at sigma=0.03 a brief ~45mm apex pass collects partial credit on
the bell flanks; at 0.02 the flanks (h* 30-50mm region rising/falling over
the 0.43s desired swing) become expensive unless the foot REMAINS airborne
through them — a shape-tracking pull toward the full swing span, not a
price hike (the falsified v71c lever).

Resume: EMBEDDED model_4500_v71.pt, +1500 iters.
Accept: unchanged (R1-R6+L1 walk10+walk05 binding + non-regression).
Decision: pass -> delivery battery. Swing still <0.28 -> arc wrap with
honest structural verdict (cadence+lift solved; swing duration is the
remaining 1-gate defect; next-arc candidates: demo-schedule conditioning
of T* curriculum, or beyondmimic deviation-termination family).
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_YAW_GUARD"] = "1.0"
os.environ["X1_AMP_NUM_STEPS"] = "30"
os.environ["X1_DISC_LINVEL"] = "1"
os.environ["X1_DISC_BUFFER"] = "24"
os.environ["X1_MOTION_DIR"] = "x1_lab_v67"
os.environ["X1_DISC_VMATCH"] = "1"
os.environ["X1_TOE_GUARD"] = "0.6"
os.environ["X1_ABD_GUARD"] = "0.8"
os.environ["X1_SWAY_GUARD"] = "0.4"
os.environ["X1_STANCE_MAX"] = "0.38"
os.environ["X1_SY_GUARD"] = "0.5"
os.environ["X1_LEAN_GUARD"] = "0.5"

os.environ["X1_GAIT_CLOCK"] = "1"
os.environ["X1_FOOT_CLOCK_W"] = "1.0"      # back at the m4500 dose
os.environ["X1_CONTACT_CLOCK_W"] = "0.5"   # back at the m4500 dose
os.environ["X1_FOOT_SIGMA_Z"] = "0.02"     # THE v71d single variable
os.environ["X1_HIP_PHASE"] = "0.4"
os.environ["X1_CADENCE_PRIOR"] = "0"
os.environ["X1_SWING_PRIOR"] = "0"

_EMBEDDED = _here / "checkpoints" / "model_4500_v71.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    raise FileNotFoundError(f"v71d embedded base not found: {_EMBEDDED}")
os.environ["X1_FINE_TUNE_ITERS"] = "1500"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
