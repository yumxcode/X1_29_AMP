#!/usr/bin/env python3
"""v42 launcher: single-checkpoint convergence — combine every PROVEN lever
on the best near-miss base (v40 model_5498).

Audit gap: no single checkpoint passes BOTH gate families. v40 is the
closest (platform 13/13 + walk05 G2 0.978/0.982 + arms 25.5/27.8 +
walk10 knee 0.972); its three misses share ONE root cause — arm-DC
asymmetry (P7g 33.3 deg L-28.9/R+4.5) coupling into drift (-3.06 m).
Robustness 47/60 misses by one cell (push1.0 0/5 under the ±0.5
training regime).

v42 = v40 + three targeted fixes (every component individually proven):
  1. arm_asym_lean -0.2 -> -0.4 (doubled): the guard that owns exactly
     the v40 defect (v32b calibration: frozen 18.8 deg scored -0.066/step
     at -0.2; v40's 33 deg ran at only -0.05/step — underpowered).
  2. X1_ROBUST_TRAIN=4 (push ±0.65 @4-8s): v39-v41 trained at ±0.5 ->
     push1.0 0-1/5; v35 trained at ±0.8 -> 3/5. Both magnitude AND
     frequency interpolate.
  3. task_style_lerp 0.65 -> 0.68: v38-v41 lin kernel 0.834-0.844 vs
     T1 target 0.85 (v36 clean peak 0.864); style margin ample (0.94-1.07
     at 0.65 vs floor 0.15).

Kept from v40 (all proven this arc): v35 base heritage, disc obs 10
bodies, style_reward_scale 2.5, arm smoothing halved (the arm-swing
lever: 18.7 -> 25.5/27.8 deg), leg_amp_asym -0.8, arm_leg_coupling +
arm_pitch_sync guards ON (v41 proved retiring them collapses coupling).

Base: v40 model_5498 embedded at roboparty_train/checkpoints/model_5498_v40.pt.
+1000 iters (total 6499), FORM_GUARDS=1 (full set, calibrated weights).

Accept gates (single checkpoint): platform 13/13 AND T1 lin >= 0.85 AND
P7 8/8 (P7g/h <= 12) AND G2 walk05+walk10 hip+knee >= 0.85 AND |drift|
<= 1.0 m AND arms >= 25 deg AND robustness >= 48/60 with push1.0 >= 2/5.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "4"        # push ±0.65 @4-8s (strong-mid)
os.environ["X1_FORM_GUARDS"] = "1"         # full guard set (v41 lesson)
os.environ["X1_FINE_TUNE_ITERS"] = "1000"  # v40 5498 -> total 6499

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_5498_v40.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(p for p in _HERE.parent.glob("model_5498*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v40 base checkpoint not found")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
