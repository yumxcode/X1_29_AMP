#!/usr/bin/env python3
"""v40 launcher: disc-led round 3 — from the BEST-form base (v35).

Release matrix so far: v39 took the PLATFORM gate (13/13, first genuine
pass) but local G2 hip symmetry (0.718/0.698) and arm swing (18.7/20.4
deg) regressed for the third consecutive disc-led run; v35 holds the
LOCAL gates (G all-green, arms 28.5/29.6, robustness 55/60) but platform
8/13. Diagnosis across v38/v39: (a) starting from v37 meant starting
from hip 0.823 (v35 was 0.874); (b) leg_amp_asym at -0.3 is too weak
(+0.024 over 1500 iters); (c) the flat smoothing penalties on all 29
joints outweigh the style gradient toward reference arm swing.

v40 (A/B/C: base + guard + smoothing split):
  - BASE = v35 model_3999 (G2 all-green, best arm swing, 55/60 robust)
  - disc obs 10 bodies + style_reward_scale 2.5 + lerp 0.65 (v39 recipe)
  - leg_amp_asym -0.8 (doubled)
  - joint_acc_l2 / action_rate_l2 SPLIT: legs+torso (15 joints) keep the
    original weights, arm channels (14 joints) at HALF — giving the
    arm-style gradient room it never had
  - X1_ROBUST_TRAIN=3, +1500 iters (v35 3999 -> total 5499)

Accept gates: platform 13/13 held AND P7 8/8 AND G2 walk05+walk10 hip
AND knee >= 0.85 AND arm swing >= 28 deg AND |drift| <= 1.0 m AND
robustness >= 48/60. If arms rise but hip fails (or vice versa) the
disc-led route has a structural conflict — release v39/v35 pair.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "3"       # v38/v39 regime
os.environ["X1_FINE_TUNE_ITERS"] = "1500"  # v35 base 3999 -> total 5499

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_3999_v35.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(p for p in _HERE.parent.glob("model_3999*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v35 base checkpoint not found (embedded or mounted)")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
