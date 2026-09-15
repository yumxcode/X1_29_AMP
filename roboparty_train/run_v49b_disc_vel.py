#!/usr/bin/env python3
"""v49b launcher: discriminator AMPLITUDE channel — the direct test of the
audit items 1/2 route (pure-disc-led arm swing via wrist VELOCITY obs).

The v38-v41 attribution proved the position-only wrist obs leave the disc
insensitive to swing amplitude. This run adds key_body_VELOCITY to the
discriminator observation (infra commit 403f1d3: ref side derived by
finite difference — no dataset regen; policy side EMA-smoothed finite
difference; X1_DISC_VEL toggle). The disc re-inits (obs 10x3 -> 10x6 per
step) via the proven amp_runner path.

Config: v45 model_7395 base, regime 3, repo-current guards
(leg_amp_asym -1.0, yaw_bias -1.0, arm smoothing quartered), +1000 iters
(disc re-init needs ~400+ to mature; style dips then recovers).

PRIMARY READOUT (not the standard gates): arm swing joint deg on the
local battery vs the 20-26 deg plateau — if the velocity channel lifts
arms >= 35 deg WITHOUT the arm guards, the disc-led amplitude route is
PROVEN and the guards can be retired next. Secondary: standard axes
(P7/G2/drift/robustness) to place it in the matrix.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"            # THE EXPERIMENT
os.environ["X1_FINE_TUNE_ITERS"] = "1000"  # v45 7395 -> total 8395

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_7395_v45.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(p for p in _HERE.parent.glob("model_7395*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v45 base checkpoint not found")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
