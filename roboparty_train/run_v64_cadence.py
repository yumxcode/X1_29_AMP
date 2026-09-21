#!/usr/bin/env python3
"""v64 launcher: RHYTHM round — human cadence prior on the v63 recipe.

GOAL (contract rev2): pass the P8 rhythm gates (RHYTHM_GATES.md — cadence
[0.55,1.20] Hz/foot, swing [0.28,0.60] s, duty [0.45,0.75], step length
[0.45,1.05] m @walk10) at 100 Hz control, no regress on P7/G/K/H gates,
sim2sim validation.

Forensics (P8 baseline, 2026-09-21): the rhythm defect is family-wide —
v63 m10500 (100 Hz) cadence 2.86 Hz / swing 0.19 s / duty 0.95 / step
0.145 m; even the 50 Hz champion fails (1.85 Hz / 0.17 s / 0.27 m). The
disc's 600 ms macro-window fixed the amplitude blind spot but cadence
remains un-anchored (nothing in the reward mix pins WHEN feet should
land). Pre-registered lever (RHYTHM_GATES.md §4): CADENCE PRIOR — this
run.

Single variable vs the v63 recipe (frozen):
  X1_CADENCE_PRIOR=0.6   gait_period_prior — exp-kernel touchdown-interval
                         reward vs T*(v)=clamp(1.46-0.36v, 0.70, 1.75) s
                         (episode-clock based -> control-rate independent;
                         dryrun_reward_v63.py 13/13).

Base: v63 m10500 (TASK_20260921_038 — arms 40.0/40.0, drift -0.31/-0.52,
5/5, platform 12/13, disc macro-window lineage, full disc+optimizer state)
embedded at roboparty_train/checkpoints/model_10500_v63.pt.

+1000 iters @100 Hz (behavioral shift is large: cycle 0.35 -> ~1.0 s,
step 0.15 -> ~0.9 m; mid-flight checkpoints mirrored by the v29e4 monitor).

ACCOUNT (user directive, contract rev2 amendment): the last training
account (dacoy87389) is nearly exhausted — this task is created under the
user-given kobay82044 account.

PRE-REGISTERED accept: best mid/final checkpoint @100 Hz local battery:
  P8: R1+R2+R3+R4 PASS on walk10 AND walk05
  no-regress: P7 8/8, arms >= 24 deg both sides, drift <= 1.0 m, 5/5
  survived, K1 <= 18, robustness >= 48/60, platform 13/13 verdict.
  Dose ladder: 0.6 first; if cadence moves < 30% of the way -> 1.0 (v64b);
  if gait destabilizes (falls / P7 collapse) -> 0.3 (v64b).
"""
import os
import runpy
import sys
from pathlib import Path

# --- v63 recipe (frozen, run_v63c_100hz.py heritage) --------------------
os.environ["X1_CONTROL_HZ"] = "100"
os.environ["X1_MOTION_DIR"] = "x1_lab_v32_200"
os.environ["X1_AMP_NUM_STEPS"] = "60"
os.environ["X1_DISC_STRIDE"] = "2"
os.environ["X1_DISC_BUFFER"] = "24"
os.environ["X1_DISC_LINVEL"] = "1"
os.environ["X1_ACT_LPF_HZ"] = "10"
os.environ["X1_ACT_LPF_ORDER"] = "2"
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_YAW_GUARD"] = "1.0"           # v63 value (v63b's 2.5 reverted)

# --- v64 single new variable --------------------------------------------
os.environ["X1_CADENCE_PRIOR"] = "0.6"
os.environ["X1_FINE_TUNE_ITERS"] = "1000"

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_10500_v63.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(_HERE.parent.glob("model_10500_v63*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v63 m10500 base checkpoint not found")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
