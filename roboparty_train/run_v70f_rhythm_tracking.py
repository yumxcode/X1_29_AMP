#!/usr/bin/env python3
"""v70f launcher: RHYTHM round 5 — frame-synced demo joint tracking.

Audit round-2 directive: the persistent full-episode reference anchor
(PW-IL/beyondmimic family) bridged INTO the AMP env as reward terms.
The 17-round arc verdict: t=0 state resets alone leave the 2.2Hz
attractor within seconds (v70e2); only an anchor that persists THROUGH
the episode can hold the human cadence.

Stack (the three pieces compose; disc untouched):
  X1_DEMO_RESET_P = 1.0   every walking env starts ON its demo frame and
                          stashes (motion_id, t0, duration) — v70e event
  X1_DISC_VMATCH  = 1     speed-matched clip assignment (the tracked
                          clip's speed matches the env's command) + drops
                          the 2x-playback 103_07 clips
  X1_TRACK_HIP    = 1.2   demo_joint_tracking_exp: exp(-sum|q_j -
                          demo_j(t0 + eplen*dt)| / 0.35) over
                          [L,R]hip_pitch + [L,R]knee_pitch, straight-
                          walking gated. The demo swing IS the 0.48-0.94
                          Hz/foot human cadence — tracked in joint space
                          from in-sync starts, anchored every step.

Base m16495b (v70b endpoint), +1500 iters. No LPF, no delay rings, no
hip-phase sinusoid (superseded by the literal demo reference).
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
# v70f tracking stack (the ONLY new experimental variables)
os.environ["X1_DEMO_RESET_P"] = "1.0"
os.environ["X1_DISC_VMATCH"] = "1"
os.environ["X1_TRACK_HIP"] = "1.2"
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
    raise FileNotFoundError("v70f base checkpoint not found (model_16495_v70b.pt)")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
