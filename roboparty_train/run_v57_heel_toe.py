#!/usr/bin/env python3
"""v57 launcher (human-gait arc, round 2): HEEL-STRIKE + phased sole-flat.

Base: v57d model_11391 (TASK_20260917_115 — heel reward saturated
in-domain 0.87/event yet sim2sim still toe-first: the last-100 ms
plantarflexion snap is a cross-sim behavior; ankle_flick never fired
in-domain). v57e adds heel_down_ready — the DENSE low-swing posture
ramp that opposes the snap at every frame it unfolds in.

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
os.environ["X1_FINE_TUNE_ITERS"] = "1000"  # v57d 11391 -> 12391

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_11391_v57d.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(_HERE.parent.glob("model_11391_v57d*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v57c base checkpoint not found "
                                "(roboparty_train/checkpoints/model_11391_v57d.pt)")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
