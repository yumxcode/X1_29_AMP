#!/usr/bin/env python3
"""v70e launcher: RHYTHM round 4 — demo-state phase-locked episode init.

Audit round-1 directive route 2. 16 falsifications precede this (v64/64b/
64c reward engineering, v65/v65b VMATCH resampling, v66/v66b scratch,
v70c1 hip-phase on the new base, v70c2 LPF governor [escape + robustness
36/60], v70d delay co-training [robustness 40/60, lat2 still 0/5]).
Common property: the policy always STARTS episodes from the standing
default pose — the on-policy state distribution never contains human-
cadence walking states, so the action function is never trained inside
that basin no matter which reward/filter/data gradient is applied.

This round changes the SAMPLING DISTRIBUTION itself:
  X1_DEMO_RESET_P = 0.5  walking-command envs (cmd > 0.3 m/s) start ON a
                         retargeted demo frame — root pose/velocities +
                         joint pos/vel — at the SAME motion_id + fetch
                         time the discriminator compares against (phase
                         locked by construction; events.py
                         reset_robot_from_demo).
NO LPF, NO delay rings (their robustness cost is measured: 36-40/60).
Kept: all posture guards (sy/lean/toe/abd/sway/stance) + HIP_PHASE 0.5.
Base m16495b (v70b endpoint), +1500 iters.
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
# v70c phase anchor (kept; the reward channel alone was proven inert —
# v70c1 — but under demo-state resets it names the local attractor)
os.environ["X1_HIP_PHASE"] = "0.5"
# v70e demo-state init (the ONLY new experimental variable)
os.environ["X1_DEMO_RESET_P"] = "0.5"
os.environ["X1_FINE_TUNE_ITERS"] = "1500"
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
    raise FileNotFoundError("v70e base checkpoint not found (model_16495_v70b.pt)")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
