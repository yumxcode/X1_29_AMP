#!/usr/bin/env python3
"""v39 launcher: disc-led gait, second attempt — informative style channel
amplified + hip-amplitude symmetry guard.

v38 verdict (TASK_20260912_135): platform 12/13 held (lin 0.842/ang
0.539), style score became INFORMATIVE for arm form (0.796 -> 0.562 as
arms fell 29 -> 22 deg — the disc now sees wrists/shoulders), but the
run net-regressed form: walk10 hip ratio 0.823 -> 0.694, walk05 hip
0.755 FAIL, drift -1.47 m (reversed), arm swing DOWN.

v39 changes (A/B vs v38; base/dose/regime identical — v37 model_4798
+1500 iters, X1_ROBUST_TRAIN=3):
  1. style_reward_scale 1.5 -> 2.5 — the style channel now carries arm
     information; amplify it. (v33b-era saturating style is gone.)
  2. task_style_lerp 0.7 -> 0.65 — 35% style mix for gradient room.
  3. NEW leg_amp_asym guard (w -0.3): |RMS_L - RMS_R| on hip pitch,
     tau 8s EMA of dev^2 — penalizes asymmetric swing amplitudes toward
     the mirrored-dataset average. Calibration in rewards.py docstring.

Accept gates (v37 / v38 baselines in parens):
  platform P3a >= 0.82 (0.843 / 0.842), P6_play PASS (FAIL / FAIL — chain
  now works, r4 verified), P7 8/8 (8/8 / 7/8), walk05 knee >= 0.85
  (0.927 / 0.937), walk05 hip >= 0.85 (? / 0.755 FAIL), walk10 hip+ knee
  >= 0.85 (0.823/0.837 MISS / 0.694/0.959), robustness >= 48/60 (48/48),
  |drift| <= 1.0 m (0.94 / 1.47 FAIL), arm swing >= 29 deg joint
  (29.7/26.8 / 21.7/23.2).

Revert rules: P3a < 0.82 -> lerp back to 0.7; P5a < 0.15 -> style scale
back to 1.5; hip ratio < 0.70 at any eval -> the guard is insufficient,
fall back to v37 release.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "3"       # same as v38 (isolates the 3 knobs)
os.environ["X1_FINE_TUNE_ITERS"] = "1500"  # v37 base 4798 -> total 6298

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_4798_v37.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(p for p in _HERE.parent.glob("model_4798*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v37 base checkpoint not found (embedded or mounted)")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
