#!/usr/bin/env python3
"""v74 launcher: INITIAL reward + CLOCK ONLY (RECORD_BOOK §9).

v73 verdict (TASK_20261024_005, 4000/4000): the pre-registered decision-tree
branch "R2/R3 improved but R1 regressed" was HIT —
  walk10 m3999: v-err 0.157 (family-best), 0.87 m/s, duty 0.46-0.51 IN-BAND
  (vs the v71/v72 structural stuck value 0.88), lift 46-57 mm (vs 30-35
  marginal) — BUT cadence 3.1 Hz (way out of [0.55,1.20]; even faster than
  the pre-clock 2.1-2.3 Hz family) with 0.15 m micro-steps.
Conclusions: (a) the clock is NECESSARY for cadence (H2 confirmed);
(b) the accumulated v27-v72 stack was causally involved in the
early-touchdown/over-long-double-support attractor (duty 0.88 -> 0.47);
(c) swing duration ~0.15-0.18 s is invariant across all three regimes
(clock / data / clean reward) — a dynamics constant of the balance
strategy, needing an active slowing mechanism.

v74 = v73 recipe + THE CLOCK PACKAGE ONLY (single variable vs v73):
  X1_GAIT_CLOCK=1 (policy/critic obs +[sinL,sinR,cosL,cosR] = 100 d/frame)
  + foot_clock w=1.0 + contact_clock w=-0.5 (v71 delivery doses).
  hip_phase stays 0 (it is a phase-locked PRIOR, not clock infrastructure —
  it re-enters only in a later round if needed, one term at a time).
Question: with the balance tradeoff freed by the clean reward, does the
clock hold cadence ~1 Hz AND keep duty in-band (0.6-0.7) — the synthesis
neither v71 (clock+stack: 1 Hz but duty 0.88) nor v73 (clean: duty 0.47
but 3.1 Hz) reached? Watch R2 swing as the third readout (expect ~0.15 s
unless the clock bell stretches it toward the 0.43 s schedule).

PRE-REGISTERED acceptance:
  A (converge): 4000 iters, ep_len > 900, timeout > 0.9
  B (rhythm walk10+walk05): R1 in-band AND R3 <= 0.75 simultaneously =
     the synthesis gate; R2 >= 0.28 = full pass
Decision tree:
  B full pass            -> delivery candidate + platform battery
  R1 in-band, R3 <= 0.80 -> near-synthesis; v74b continuation +2000
                            (v72b precedent) before any recipe change
  R1 in-band, R3 stuck >= 0.85 -> the stack-vs-duty causal claim isolates
                            further: next single variable = disc window
                            30->60 steps (H3), then swing-duration term
  R1 out-of-band         -> clock-vs-clean-reward interaction; escalate
                            clock doses (FOOT_CLOCK_W 1.5 ladder, v71c
                            precedent) — NOT new reward terms
  A fail                 -> one iterations-only continuation
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- from-scratch pre-flight -------------------------------------------------
os.environ.pop("X1_RESUME_CKPT", None)
os.environ.pop("X1_FINE_TUNE_ITERS", None)
_stray = sorted((_here.parent).glob("model_*.pt"))
if _stray:
    raise FileNotFoundError(
        f"FROM-SCRATCH pre-flight: repo root holds mounted checkpoints "
        f"{[p.name for p in _stray]} — run_x1_amp_train would auto-resume; "
        f"remove/rename them first")

# --- THE v74 VARIABLE: initial reward + clock only ---------------------------
os.environ["X1_INITIAL_REWARD"] = "1"
os.environ["X1_GAIT_CLOCK"] = "1"
os.environ["X1_FOOT_CLOCK_W"] = "1.0"
os.environ["X1_CONTACT_CLOCK_W"] = "0.5"

# --- identical infrastructure to v73 -----------------------------------------
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_AMP_NUM_STEPS"] = "30"        # 600 ms disc window @50Hz
os.environ["X1_DISC_LINVEL"] = "1"
os.environ["X1_DISC_BUFFER"] = "24"
os.environ["X1_MOTION_DIR"] = "x1_lab_v72"   # 38_01 fixed, 36 sources
os.environ["X1_DISC_VMATCH"] = "1"

# --- explicitly NOT set (stays OFF under the initial-reward reset) -----------
# X1_HIP_PHASE / guards / priors / demo-reset / track-hip — all zeroed by
# X1_INITIAL_REWARD=1 (see x1_amp_env_cfg [INITIAL-REWARD] print)

os.environ["X1_FRESH_ITERS"] = "4000"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
