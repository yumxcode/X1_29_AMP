#!/usr/bin/env python3
"""v67f launcher: overground library + data-side package ONLY (trio removed).

v67e verdict (TASK_20260922_077, 3999/3999 iters): UNCONVERGED and
oscillating — hip cadence 1.92 Hz (m2799) -> 0.96 Hz-but-frozen (m3500,
2 events) -> 3.92 Hz tap (m3998); ep_len 612-728 vs historical ~990 at
4000. Assessment per pre-registration (the "still unconverged" branch):
more iters on an oscillating dynamic has weak expected value — the
convergence impairment itself is informative.

CONFOUND IDENTIFIED: both scratch runs (v66 old-lib, v67d/e overground)
bundled the RHYTHM TRIO priors (0.6/0.3/0.4). The reward-side priors are
already falsified for rhythm (v64/v64b/v64c) AND they fight the task
rewards during early learning — the historical recipe that converged at
4000 (v26-v31: ep_len ~990) had NO reward priors beyond the form guards.
Route 1's true variable (audit r2 directive 4) is the DEMO LIBRARY; the
trio was route-3 baggage.

v67f = single-variable isolation:
  KEEP   x1_lab_v67 overground library (69% mass) — the route variable
  KEEP   VMATCH full (motion+time speed gating) — data-side, no reward
  KEEP   600 ms macro disc + linvel + BUFFER 24 — disc SEES the rhythm
  DROP   the rhythm trio (X1_CADENCE_PRIOR/SWING_PRIOR/HIP_PHASE = 0)
  REST   = v59c frozen (regime 3, guards 1, disc_vel 1, yaw 1.0, 50 Hz)

Expected: converges at historical scale (no prior interference) -> the
library effect on rhythm is then cleanly readable at convergence.

PRE-REGISTERED accept (unchanged): P8 R1-R4 walk10+walk05 at 100 Hz
direct-drive + no-regress set. Decision: pass = contract met;
converged-fail = demo-replacement falsified (physics evidence attached);
unconverged again = the macro-disc/VMATCH package itself impairs
convergence (next: standard 3-step disc variant).
"""
import os
import runpy
import sys
from pathlib import Path

# --- v59c recipe (frozen) ------------------------------------------------
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_YAW_GUARD"] = "1.0"

# --- data-side package (KEEP — the route variable) ------------------------
os.environ["X1_MOTION_DIR"] = "x1_lab_v67"
os.environ["X1_DISC_VMATCH"] = "1"
os.environ["X1_AMP_NUM_STEPS"] = "30"
os.environ["X1_DISC_LINVEL"] = "1"
os.environ["X1_DISC_BUFFER"] = "24"

# --- rhythm trio REMOVED (convergence confound; reward side falsified) ----
os.environ["X1_CADENCE_PRIOR"] = "0"
os.environ["X1_SWING_PRIOR"] = "0"
os.environ["X1_HIP_PHASE"] = "0"

# --- fresh run, pod-budget-sized ------------------------------------------
os.environ.pop("X1_RESUME_CKPT", None)
os.environ.pop("X1_FINE_TUNE_ITERS", None)
os.environ["X1_FRESH_ITERS"] = "2800"

_here = Path(__file__).resolve().parent
sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
