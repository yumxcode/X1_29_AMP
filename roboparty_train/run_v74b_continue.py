#!/usr/bin/env python3
"""v74b launcher: iterations-only continuation +2000 from the EMBEDDED v74 m3999.

v74 verdict (TASK_20261024_008, 4000/4000, evaluated 2026-10-24):
  walk10 m3000 AND m3999 PASS ALL rhythm gates R1-R4+L1 — FIRST TIME in
  20+ rounds (m3999: 0.91 Hz/foot BOTH feet, swing 0.50-0.58 s — the
  structurally-stuck 0.12-0.18 s regime BROKEN, duty 0.49-0.53 vs the
  0.88 attractor, step 0.48 m, lift 106-149 mm, v-err 0.188 @ 0.83 m/s).
  walk05 m3999: R1/R2/R3/L1 pass; residual = speed undershoot
  (0.30 m/s vs 0.5 cmd; the R4 miss there is the 1 m/s-calibrated range).

Pre-registered next step (RECORD_BOOK §9 tree: full pass -> delivery
candidate; ladder still improving monotonically: walk10 v-err
0.226 -> 0.221 -> 0.188, walk05 duty 0.78 -> 0.68): CONTINUATION ROUND,
recipe byte-identical, iterations the ONLY variable (v71b/v72b lesson —
no dose change while the ladder improves). Target: v-err toward <0.15
(1.0 m/s neighborhood) + walk05 speed tracking.

Accept (pre-registered): non-regression on walk10 ALL-gates-PASS +
v-err improvement; walk05 speed >= 0.4 m/s. Fallback: keep m3999 as the
delivery candidate (it already passes); soup adjacent ladder points if
both m5999 and m3999 pass (v69 flow).
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- THE v74 recipe, byte-identical -----------------------------------------
os.environ["X1_INITIAL_REWARD"] = "1"
os.environ["X1_GAIT_CLOCK"] = "1"
os.environ["X1_FOOT_CLOCK_W"] = "1.0"
os.environ["X1_CONTACT_CLOCK_W"] = "0.5"

os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_AMP_NUM_STEPS"] = "30"
os.environ["X1_DISC_LINVEL"] = "1"
os.environ["X1_DISC_BUFFER"] = "24"
os.environ["X1_MOTION_DIR"] = "x1_lab_v72"
os.environ["X1_DISC_VMATCH"] = "1"

# --- continuation: +2000 iters from the EMBEDDED v74 m3999 ------------------
_EMBEDDED = _here / "checkpoints" / "model_3999_v74.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    raise FileNotFoundError(f"v74b embedded base not found: {_EMBEDDED}")
os.environ["X1_FINE_TUNE_ITERS"] = "2000"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
