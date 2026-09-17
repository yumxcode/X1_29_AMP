#!/usr/bin/env python3
"""v58 launcher (human-gait arc, structural round): actuation-robustness envelope.

Course-correction step 3 (2026-09-17 mid-arc review): after the deploy-side
harness probe matrix came back negative (PD recompute 200Hz/1kHz, native
implicitfast servos, ankle damping x4/x10 — flick rate invariant ~110 deg/s),
the remaining channel against the cross-sim terminal ankle flick is
training-side structural randomization WIDENING:
  - actuator gains (0.8,1.2) -> (0.7,1.3)
  - armature (0.8,1.2) -> (0.7,1.3)
  - action delay cap 1 -> 2 (per-env random 0/1/2, p-biased 0.8)
Everything else = the v57e recipe (heel_first 0.40 ramp, heel_down_ready 0.3,
ankle_flick -0.2, knee_extension 0.10, phased sole-flat, yaw -2.0).

Base: v57e model_11500 (the drift-clean flight point: -0.64/+0.97 both
walks under gate 1.0, K1 14.4/12.7, carries the saturated heel learning).

Accept (single checkpoint, +1000 iters from 11500 -> 12500):
  sim2sim H1 >= 60% heel-first (walk10, both feet) AND H3 <= 10%
  AND K1 <= 18 held AND H2 >= 90% held AND |drift| <= 1.0 (both walks)
  AND walk05 swing >= 10 events/foot AND no other gate regressions.
A negative result completes the two-channel evidence chain (deploy-side +
training-side) required for the documented-blocker close-out.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "48"
os.environ["X1_ACT_RAND"] = "2"          # v58 structural envelope
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_FINE_TUNE_ITERS"] = "1000"  # v57e 11500 -> 12500

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_11500_v57e.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(_HERE.parent.glob("model_11500_v57e*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v57e base checkpoint not found "
                                "(roboparty_train/checkpoints/model_11500_v57e.pt)")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
