#!/usr/bin/env python3
"""v60 launcher (audit r2 item 2): posture-clearance joint round.

The walk05/back05 R-foot drag family (apex 8-11mm vs healthy 21-24mm)
survives the regime-x-yaw 2x2 matrix + 5 soup ratios — coupled to the
straight-knee stance posture's clearance margin at low speed. v60 adds
ONE variable: swing_clearance_floor w=-0.5 (normalized relu below a
12mm floor on the min sole height while airborne, walk speeds) — the
"posture-clearance joint redesign" the audit asked for.

Base: v59c model_9500 (the drift-clean 3/-1 cell: -0.32/-0.12 both
walks, K1 13.7/12.0, hip 0.848 at gate edge). Regime 3 + yaw -1.0 kept
(the native v53/v54 recipe that holds platform 12/13 kernels).

Accept: walk05 both feet >= 10 stance events with apex >= 12mm AND
back05 drift <= 1.0 AND all v59c-m9500 gates held (drift/K1/G2/G3/H2/
arms/collision) — i.e. the all-gate single point.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_SWING_CLEAR"] = "1"          # v60 single variable
os.environ["X1_FINE_TUNE_ITERS"] = "600"    # v59c 9500 -> 10100
os.environ["X1_YAW_GUARD"] = "1.0"

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_9500_v59c.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(_HERE.parent.glob("model_9500_v59c*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v59c base checkpoint not found "
                                "(roboparty_train/checkpoints/model_9500_v59c.pt)")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
