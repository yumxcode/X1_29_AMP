#!/usr/bin/env python3
"""v72b launcher: CONTINUATION of v72 (TASK_20261023_111, from-scratch gait
clock + low-cadence dataset) — +2000 iters from m3999.

Ladder evidence at walk10 (acceptance/v72_eval/):
  m1000: v-err 0.752 (2.97 m) — no rhythm verdicts (too few swings)
  m2000: v-err 0.409, cadence 0.91 Hz OK, swing 0.24 s (ladder's longest),
         apex 41 mm L1 PASS, duty 0.796
  m3000: v-err 0.244 (8.90 m), cadence 0.98 OK, swing 0.145, duty 0.888
  m3999: v-err 0.200 (9.37 m = 0.815 m/s), cadence L0.60/R0.89 OK,
         step 0.72/0.42 R4 OK, apex 31/30 mm — R2 swing 0.14 s and
         R3 duty 0.886 STILL STRUCTURALLY STUCK at the v71 values.
  walk05 m3000/m3999: drag persists (0.28 m/s, insufficient swings).
Platform AMP: P1/P2/P6 PASS (ep_len 995, timeout 0.991, zero base contact),
P3 kernels 0.499/0.475 (v71 was 0.496/0.477 — same clock-vs-speed
equilibrium), P4 late-window dip, P5a style 0.119. lin_vel_track 0.624 at
4000 and STILL CLIMBING — equilibrium not reached.

Pre-registered decision (v72 launcher tree, "A fail -> v72b continuation
round, iterations-only"): same recipe, same doses (the v71b lesson — no
dose change while the ladder improves monotonically), +2000 iters from the
EMBEDDED m3999 (cross-account determinism, v71c pattern).
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- soup59c validated base (identical to v72) ------------------------------
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_YAW_GUARD"] = "1.0"

# --- disc observability + data side (identical to v72) ----------------------
os.environ["X1_AMP_NUM_STEPS"] = "30"
os.environ["X1_DISC_LINVEL"] = "1"
os.environ["X1_DISC_BUFFER"] = "24"
os.environ["X1_MOTION_DIR"] = "x1_lab_v72"
os.environ["X1_DISC_VMATCH"] = "1"

# --- v69/v70 delivery-dose posture guards (identical to v72) ----------------
os.environ["X1_TOE_GUARD"] = "0.6"
os.environ["X1_ABD_GUARD"] = "0.8"
os.environ["X1_SWAY_GUARD"] = "0.4"
os.environ["X1_STANCE_MAX"] = "0.38"
os.environ["X1_SY_GUARD"] = "0.5"
os.environ["X1_LEAN_GUARD"] = "0.5"

# --- the v71 mechanism: exogenous gait clock (identical doses) --------------
os.environ["X1_GAIT_CLOCK"] = "1"
os.environ["X1_FOOT_CLOCK_W"] = "1.0"
os.environ["X1_CONTACT_CLOCK_W"] = "0.5"
os.environ["X1_HIP_PHASE"] = "0.4"
os.environ["X1_CADENCE_PRIOR"] = "0"
os.environ["X1_SWING_PRIOR"] = "0"

# --- continuation: +2000 iters from the EMBEDDED m3999 ----------------------
_EMBEDDED = _here / "checkpoints" / "model_3999_v72.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    raise FileNotFoundError(f"v72b embedded base not found: {_EMBEDDED}")
os.environ["X1_FINE_TUNE_ITERS"] = "2000"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
