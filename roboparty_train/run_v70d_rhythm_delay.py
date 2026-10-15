#!/usr/bin/env python3
"""v70d launcher: RHYTHM round 3 — delay co-training under the LPF governor.

Audit round-1 directive (2026-10-15): execute the report's first-choice
open route. m17000c2 (v70c2 LPF mid-point) proved the mechanism physically
works (cadence 2.13 -> 1.65-1.70 Hz, foot lift 12 -> 26-37 mm, all arm/
gait gates PASS at the frozen checkpoint) but FAILED robustness 36/60 with
exactly lat2 0/5 + lag2 0/5 — the LPF consumes the correction bandwidth
those delay variants need; and by +1500 iters the policy escaped through
the filter (9x amplitude via saturation -> 2.35 Hz).

This round co-trains the EXACT failing sweep axes so the policy re-learns
balance under filter+delay stacking (and the 2.3Hz escape route becomes
destabilizing under delay, so the low-cadence solution should stick):
  X1_ACT_LPF_HZ = 1.6, ORDER 2   (unchanged governor)
  X1_ACT_DELAY  = 2              action delay ring 0/1/2 (sweep --action-lag)
  X1_OBS_DELAY  = 2              obs delay ring 0/1/2 NEW (sweep --latency-steps;
                                 per-env random, policy group only, reset-flushed;
                                 amp_obs_delay.ObsDelayRing, unit-tested)
  X1_HIP_PHASE  = 0.5            (phase anchor, unchanged)
Base m16495b (v70b endpoint: sy~0, lean 2.9, all gates green), +1000 iters
to land MID-adaptation at ~1.5-1.7 Hz before any escape pressure develops.
"""
import os
import runpy
import sys
from pathlib import Path

# v70b recipe (identical) — unchanged
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_YAW_GUARD"] = "1.0"
os.environ["X1_TOE_GUARD"] = "0.6"
os.environ["X1_ABD_GUARD"] = "0.8"
os.environ["X1_SWAY_GUARD"] = "0.4"
os.environ["X1_STANCE_MAX"] = "0.38"
os.environ["X1_SY_GUARD"] = "0.5"
os.environ["X1_LEAN_GUARD"] = "0.5"
# v70c recipe (identical)
os.environ["X1_HIP_PHASE"] = "0.5"
os.environ["X1_ACT_LPF_HZ"] = "1.6"
os.environ["X1_ACT_LPF_ORDER"] = "2"
# v70d delay co-training (the ONLY new experimental variables)
os.environ["X1_ACT_DELAY"] = "2"
os.environ["X1_OBS_DELAY"] = "2"
os.environ["X1_FINE_TUNE_ITERS"] = "1000"
os.environ.setdefault("X1_CONTROL_HZ", "50")

_HERE = Path(__file__).resolve().parent
_CANDIDATES = [
    _HERE / "checkpoints" / "model_16495_v70b.pt",
    _HERE / "checkpoints" / "model_16495.pt",
]
for _c in _CANDIDATES:
    if _c.is_file():
        os.environ["X1_RESUME_CKPT"] = str(_c)
        break
else:
    raise FileNotFoundError("v70d base checkpoint not found (model_16495_v70b.pt)")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
