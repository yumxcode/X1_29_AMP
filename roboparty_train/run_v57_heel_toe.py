#!/usr/bin/env python3
"""v57 launcher (human-gait arc, round 2): HEEL-STRIKE + phased sole-flat.

Base: v56b model_9393 (TASK_20260917_028 dose-0.10 winner — K1 15.6/14.3
walk10 + 9.5/8.8 walk05, walk10 all-green, walk05 swing alive, arms
35-39, no penetration). Carried defect: late-training +5.4 deg/s signed
yaw drift (flight profile: m9000 clean) + walk05 hip ratio 0.813 +
back05 L-drag.

v57 variable (one coupled pair, GOAL §5): heel_first_stance w=0.15
(geometric lead at the stance rising edge; +1.0 heel-first / +0.3 flat /
0 toe-first) + stance_sole_flat_walk PHASED rework (heel-strike &
push-off phases exempt via heel/toe end geometry, foot-flat penalty
kept). Baseline: H1 0% heel-first, H3 65-100% toe-first (fine metric),
H2 toe-off 100% (asset to protect).

Defensive (not experimental; v47 precedent): yaw_bias -1.0 -> -2.0
against the measured late drift.

Accept (single checkpoint, +1000 iters from 9393 -> 10393):
  H1 >= 60% heel-first (walk10, both feet) AND H3 <= 10% toe-first
  AND K1 <= 18 held AND H2 >= 90% held AND |drift| <= 1.0 (both walks)
  AND walk05 swing >= 10 events/foot AND no other gate regressions.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "48"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_FINE_TUNE_ITERS"] = "1000"  # v56b 9393 -> 10393

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_9393_v56b.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(_HERE.parent.glob("model_9393_v56b*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v56b base checkpoint not found "
                                "(roboparty_train/checkpoints/model_9393_v56b.pt)")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
