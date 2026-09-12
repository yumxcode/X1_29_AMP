#!/usr/bin/env python3
"""v37 joint-schedule fine-tune launcher (Pareto point on the two gate families).

Measured trade (v35 vs v36, TASK_20260912_075 vs _111):
  v35 (randomized train): local form/robustness ALL-PASS (P7 8/8,
    walk05 knee 0.935, drift +0.29 m, robustness 55/60) but platform
    P3 kernels capped at 0.79 (lin) / 0.45 (ang) — 8/13.
  v36 (+600 CLEAN iters from v35): platform P3 ALL-PASS by iter ~4125
    (lin 0.864 / ang 0.593, 12/13, only P6_play env contract fails) but
    the clean phase re-grew form defects: walk05 knee symmetry 0.935 ->
    0.678, walk10 drift +0.29 -> +1.83 m (model_4500 mid-point: walk10
    knee 0.921 PASS, walk05 0.708 FAIL — erosion starts before +500).

Documented lesson (experience exp_mtutvjx9): the two gate families are
anti-correlated on the same policy parameters; sequential single-objective
phases ping-pong. This run tests the joint schedule instead: half-strength
perturbations (X1_ROBUST_TRAIN=2: push ±0.4 @ 6-12 s, 1-step delay kept)
so the form/robustness pressure stays alive while the kernel gradient
climbs. Target: platform P3 >= thresholds AND local form gates >= PASS
(walk05 knee >= 0.85, drift <= 1.0 m, P7 8/8, robustness >= 48/60).

Dose: +800 iters from v35 model_3999 (v36 reached the kernel plateau at
~+200 clean; under perturbation convergence is slower, +800 = margin).
All reward terms unchanged from v35 (yaw guard + lerp 0.7 + v34 arm/torso
statistics).

Accept: platform P3a>=0.82 AND walk05 knee>=0.85 AND P7 8/8 AND
robustness>=48/60 AND drift<=1.0m. Else: release decision by full gate
matrix (v35 = local-all-green fallback; v36 = platform-12/13 fallback).
"""
import os
import runpy
import sys
import urllib.request
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "2"      # half-strength joint schedule
os.environ["X1_FINE_TUNE_ITERS"] = "800"  # +800 iters from v35 model_3999

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_3999_v35.pt"
_OSS_URL = ("https://limx-gradmotion.oss-cn-beijing.aliyuncs.com/"
            "upload%2F2026%2F9%2F12%2Fmodel_3999_20260912130604A323.pt"
            "?OSSAccessKeyId=LTAI5tMec8RQN1nZuRkVMgxz&Expires=1789795155"
            "&Signature=qRe4Nh5Bf%2Fx7cl%2FTCbl0i92rx8Q%3D")

if os.environ.get("X1_RESUME_CKPT", "").strip():
    pass  # explicit override wins
elif _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(p for p in _HERE.parent.glob("model_3999*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        print(f"[RESUME] fetching base checkpoint from OSS ({_OSS_URL[:60]}...)")
        _EMBEDDED.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_OSS_URL, _EMBEDDED)
        print(f"[RESUME] fetched {_EMBEDDED} ({_EMBEDDED.stat().st_size // 1024}KB)")
        os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
