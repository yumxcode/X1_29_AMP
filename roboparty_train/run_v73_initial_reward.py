#!/usr/bin/env python3
"""v73 launcher: INITIAL-reward control experiment on the v72 dataset.

User directive 2026-10-24 (RECORD_BOOK acceptance/RECORD_BOOK.md §7-8):
the reward stack is re-derived FROM THE INITIAL DESIGN (commit 6ea0655,
2026-08-06) instead of stacking on v27-v72 additions. Single experimental
variable vs the v72 recipe (TASK_20261023_111): THE REWARD TABLE.

Kept infrastructure (NOT reward; validated independently):
  v68 identified dynamics, 600 ms disc window (X1_AMP_NUM_STEPS=30) +
  linvel disc obs + key-body vel, VMATCH speed-matched demo fetch,
  ROBUST=3 regime, 50 Hz, mirrored v72 dataset (dual gate PASS, re-verified
  2026-10-24) with the 38_01 weights bug FIXED (36 sources by design).

Removed vs v72 (all v27-v72 reward additions, via X1_INITIAL_REWARD=1):
  gait clock (obs + foot_clock/contact_clock/hip_phase), 6 posture guards,
  10 arm/torso/gait priors, stance_sole_flat, yaw_bias, swing_clearance.
  Policy obs returns to 96 d x 3 = 288 (no clock channels).

Scientific question: is the early-touchdown / over-long-double-support
attractor (R2 0.12-0.14 s, R3 0.87-0.91 — structurally IDENTICAL in v71
clock recipe and v72 data route) an artifact of the accumulated reward
stack's balance tradeoff, or a property of AMP+this-data itself? The v72
falsification was obtained UNDER the full v71 reward stack; it has never
been tested under a clean reward.

PRE-REGISTERED acceptance (RECORD_BOOK §8):
  A (converge): 4000 iters, ep_len > 900, timeout > 0.9, zero base contact
     -> fail => one iterations-only +2000 continuation (v72b precedent)
  B (rhythm, walk10 + walk05): R1-R4 + L1; PRIMARY READOUT = R2/R3
     vs the v71/v72 stuck values (0.14 s / 0.88)
Decision tree:
  B full pass        -> delivery candidate (soup per v69 flow)
  R2 >= 0.22 s or R3 <= 0.80 (improved) but R1 regressed (expected:
                       no clock) -> v74 = initial reward + clock ONLY
                       (single variable re-add, the one proven rhythm
                       mechanism)
  R2/R3 unchanged    -> clean-reward confirmation of the balance-optimum
                       verdict; next arc = disc window 600 ms -> 1.2 s
                       (R3 information blind spot, RECORD_BOOK §7 cand 3)
  A fail             -> inspect reward magnitudes (no alive-side positive
                       beyond task terms); consider 6000 iters from
                       scratch instead of resume
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- from-scratch pre-flight: no resume checkpoint anywhere near -----------
os.environ.pop("X1_RESUME_CKPT", None)
os.environ.pop("X1_FINE_TUNE_ITERS", None)
_stray = sorted((_here.parent).glob("model_*.pt"))
if _stray:
    raise FileNotFoundError(
        f"FROM-SCRATCH pre-flight: repo root holds mounted checkpoints "
        f"{[p.name for p in _stray]} — run_x1_amp_train would auto-resume; "
        f"remove/rename them first")

# --- THE v73 VARIABLE: initial reward table --------------------------------
os.environ["X1_INITIAL_REWARD"] = "1"

# --- kept infrastructure (identical to v72) --------------------------------
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_AMP_NUM_STEPS"] = "30"        # 600 ms disc window @50Hz
os.environ["X1_DISC_LINVEL"] = "1"
os.environ["X1_DISC_BUFFER"] = "24"
os.environ["X1_MOTION_DIR"] = "x1_lab_v72"   # 38_01 weights fixed (36 srcs)
os.environ["X1_DISC_VMATCH"] = "1"

# --- explicitly NOT set (byte-level clarity; defaults cover them) ----------
# X1_GAIT_CLOCK / X1_FOOT_CLOCK_W / X1_CONTACT_CLOCK_W / X1_HIP_PHASE: unset
#   (clock OFF; mutually exclusive with X1_INITIAL_REWARD by hard guard)
# X1_TOE_GUARD / X1_ABD_GUARD / X1_SWAY_GUARD / X1_STANCE_MAX / X1_SY_GUARD /
#   X1_LEAN_GUARD / X1_YAW_GUARD / X1_ARM_PRIOR / X1_ARM_ASYM /
#   X1_CADENCE_PRIOR / X1_SWING_PRIOR / X1_ANKLE_FLICK / X1_SWING_CLEAR /
#   X1_TRACK_HIP / X1_DEMO_RESET_P: unset — and ALL are overridden to zero
#   by X1_INITIAL_REWARD=1 regardless (RECORD_BOOK §6 audit lesson)

# --- from-scratch budget ----------------------------------------------------
os.environ["X1_FRESH_ITERS"] = "4000"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
