#!/usr/bin/env python3
"""v59 launcher (audit round 1, item 4): ALL-GATE single-point attempt.

The audit found no single checkpoint holds every non-regress gate:
- v56b m9393 (form champion: K1/K2-emerged/G2 0.983/swing 18-17/arms
  35-39/penetration 156mm/robustness 52) fails ONLY drift (+4.68/+2.24 —
  flight profile: m9000 was clean -0.18/-0.97, the bias grew in the last
  ~400 iters under yaw_bias -1.0).
- v57e m11500 (drift clean) lost walk05 swing + G2 edge.

v59 = the minimal defensive round on the form champion: resume v56b
model_9393 with yaw_bias -1.0 -> -2.0 (the dose with v57e evidence of
holding drift net under regime 48) and NOTHING else changed (v56 recipe:
knee_extension 0.10, no heel terms — those were the v57 lineage's swing
cost). Shorter horizon +600 iters: the drift grows late; the form is
already at the target, so this is a stabilization pass, not a shaping
pass. Accept: |drift| <= 1.0 both walks with ALL v56b form gates held
(K1 <= 18 both walks, G2 walk10 >= 0.85, walk05 swing >= 10/foot, arms
>= 24, no penetration, robustness >= 48).
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "48"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_FINE_TUNE_ITERS"] = "600"   # v56b 9393 -> 9993
os.environ["X1_YAW_GUARD"] = "2.0"         # v59 defensive: -1.0 -> -2.0

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
