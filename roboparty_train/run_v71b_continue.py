#!/usr/bin/env python3
"""v71b launcher: CONTINUATION of v71 (TASK_20261017_051, from-scratch gait
clock) — +2000 iters from m3999.

Ladder evidence at walk10 (acceptance/v71_eval/):
  m1000: cadence 0.91Hz LOCKED + lift 37mm but 0.18 m/s (v-err 0.82)
  m2000: 0.66 m/s but lift degraded (L p95 26.7mm, R drag)
  m3500: v-err 0.243, 8.72m, cadence 0.48/0.93, lift 36/32mm
  m3999: v-err 0.217, 9.27m (0.81 m/s), cadence 0.81/1.04Hz BOTH in R1 band,
         lift median 39/37mm L1 PASS — remaining FAILs R2 swing 0.16s<0.28
         + R3 duty 0.88/0.81>0.75 (same root cause: over-long double support /
         early touchdown); walk05 still drag (0.26 m/s, <25mm lifts).
Platform AMP 8/13 (P2 all PASS, style/disc PASS; P3 kernels lin 0.496<0.82,
P4 late-window dip). lin_vel_track curve STILL CLIMBING at 4000
(0.49→0.62 reward over last 1300 iters) — the equilibrium is not reached.

Pre-registered decision (v71 launcher tree, "A 不过 → 延续轮补收敛后再判 B"):
same recipe, same doses (single-variable = iterations only; the ladder is
monotone improving — no dose-change justification), +2000 iters from the
platform-mounted model_3999.pt (run_x1_amp_train auto-detects it at the repo
root; the v71 from-scratch pre-flight guard is deliberately REMOVED here).
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- soup59c validated base (identical to v71) -----------------------------
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_YAW_GUARD"] = "1.0"

# --- disc observability + data side (identical to v71) ---------------------
os.environ["X1_AMP_NUM_STEPS"] = "30"
os.environ["X1_DISC_LINVEL"] = "1"
os.environ["X1_DISC_BUFFER"] = "24"
os.environ["X1_MOTION_DIR"] = "x1_lab_v67"
os.environ["X1_DISC_VMATCH"] = "1"

# --- v69/v70 delivery-dose posture guards (identical to v71) ---------------
os.environ["X1_TOE_GUARD"] = "0.6"
os.environ["X1_ABD_GUARD"] = "0.8"
os.environ["X1_SWAY_GUARD"] = "0.4"
os.environ["X1_STANCE_MAX"] = "0.38"
os.environ["X1_SY_GUARD"] = "0.5"
os.environ["X1_LEAN_GUARD"] = "0.5"

# --- the v71 mechanism: exogenous gait clock (identical doses) -------------
os.environ["X1_GAIT_CLOCK"] = "1"
os.environ["X1_FOOT_CLOCK_W"] = "1.0"
os.environ["X1_CONTACT_CLOCK_W"] = "0.5"
os.environ["X1_HIP_PHASE"] = "0.4"
os.environ["X1_CADENCE_PRIOR"] = "0"
os.environ["X1_SWING_PRIOR"] = "0"

# --- continuation: +2000 iters from the platform-mounted m3999 -------------
# (X1_RESUME_CKPT left unset -> run_x1_amp_train auto-detects the mounted
#  model_3999.pt at the repo root; explicit override if run locally)
os.environ.pop("X1_RESUME_CKPT", None)
os.environ["X1_FINE_TUNE_ITERS"] = "2000"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
