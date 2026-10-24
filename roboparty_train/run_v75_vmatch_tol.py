#!/usr/bin/env python3
"""v75 launcher: SPEED-GATED DEMO STREAM (X1_VMATCH_TOL 0.35 -> 0.15).

v74/v74b verdict (TASK_20261024_008/019): rhythm SOLVED (all walk10 gates
pass at m3000/3999/4500/5500; 0.91 Hz/foot, swing 0.5-0.6 s, duty 0.44-0.53,
step 0.47-0.49 m) — but the SPEED plateau is 0.83-0.85 m/s vs the 1.0
target, and walk05 undershoots 0.23-0.30 vs 0.5. v74b falsified H6 (more
iterations do NOT close the gap: v-err 0.188 -> 0.185/0.179/0.201 across
m3999-5998; walk05 worsened 0.30 -> 0.23).

Full-analysis root cause (RECORD_BOOK §5 protocol): the demo stream is
speed-MISMATCHED at every command. With X1_VMATCH_TOL=0.35, cmd-1.0 envs
see the 35-family (0.66-0.73 m/s, ~68% of original-side mass) as in-band
demos — the disc's style equilibrium pulls gait kinematics toward 0.7 m/s
while the task reward pulls 1.0 -> 0.85 compromise. At cmd 0.5 the
35-family is ALSO in-band (|0.66-0.5| = 0.16 < 0.35) -> low-speed envs
imitate 0.7 m/s gaits under a 0.5 command.

v75 = v74 recipe with tol 0.15: cmd-1.0 demos become PRIME-only
(07_01 0.89 / 08_03 1.10 / 38_02 0.90 / 08_01 1.14 — the actual 1.0 m/s
neighborhood), cmd-0.5 demos become the true slow family (0002 0.35 /
0000 0.54 — the walk05 target style with 0.32 m steps at 0.7 Hz/foot).
Single variable (data-fetch tol); reward table stays EXACTLY the v74
initial-15+clock design; no new terms.

Base: resume from the EMBEDDED v74b m4500 (all-gates-pass point, v-err
0.185) +2000 iters — the disc re-equilibrates under the tightened stream
while the policy re-balances; cheaper than from-scratch and tests the
equilibrium shift directly.

PRE-REGISTERED acceptance:
  A (converge + non-regression): rhythm gates stay PASS at walk10;
     survival both scenarios
  B (speed — THE variable): walk10 distance >= 10.4 m (0.90+ m/s,
     v-err <= 0.14); walk05 distance >= 4.6 m (0.40+ m/s)
Decision tree:
  B met                 -> delivery candidate; arm-form chain (v76:
                           smoothing split) next
  walk10 improves but rhythm regresses -> mid-point soup m4500/m5000
                           (v69 flow) + v75b tol 0.20 interpolation
  no speed change       -> demo-composition hypothesis falsified for
                           the plateau; next single variable = PRIME
                           weights 1.2->3.0 (mass, not fetch) or task
                           weight dose ladder
  A fail                -> one iterations-only continuation
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- THE v75 VARIABLE: demo-stream speed tolerance ---------------------------
os.environ["X1_VMATCH_TOL"] = "0.15"

# --- the v74 recipe, byte-identical ------------------------------------------
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

# --- resume from the EMBEDDED v74b m4500 (all-pass point) --------------------
_EMBEDDED = _here / "checkpoints" / "model_4500_v74b.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    raise FileNotFoundError(f"v75 embedded base not found: {_EMBEDDED}")
os.environ["X1_FINE_TUNE_ITERS"] = "2000"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
