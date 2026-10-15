#!/usr/bin/env python3
"""v70 launcher: right-forearm inward-fold fix fine-tune from m13497
(user defect spec 2026-10-15, rev2).

Defect (video review of soup69_drift, user): 右臂小臂内折，摆动时和躯干
碰撞 — the right forearm folds INWARD across the chest and collides with
the torso during the swing.

Forensics (acceptance/v70_eval/arm_fold_v70.py, mesh-level FK replay):
  soup69_drift holds shoulder_yaw L -24.6 / R -64.6 deg DC with elbows
  flexed 37/49 deg — shoulder yaw rotates the elbow-flexion axis, so the
  R forearm folds inward (base +y component 0.40) and penetrates the
  torso mesh by -61.3 mm (walk10) / -60.4 mm (walkturn); the L arm stays
  +18 mm clear. The demo library (x1_lab_v32, all 22 clips) walks at
  shoulder_yaw ~0 deg, elbow 20 deg, clearance +22..57 mm. Defect decays
  with training (m12000 -59 -> m13000 +11.6 -> m13497 +16.0 mm) — the
  v69 drift-hedge soup(12000,12500) picked the two worst arm postures.

Recipe = v69 (soup59c cell under identified dynamics + the three v69
posture guards) + ONE new guard:
  X1_SY_GUARD = 0.5  relu(|shoulder_yaw| - 15 deg), both arms,
                    sign-symmetric, instantaneous (AC std ~10 deg is
                    inside the band). m13497 resume point pays
                    0.675 rad -> -0.34/step; demo-level arms pay 0.

Single LONG round (+2000 iters from m13497) per the V68 lesson: multi-
round resumes restart the LR schedule and oscillate drift/waist metrics.
"""
import os
import runpy
import sys
from pathlib import Path

# soup59c cell + v69 guards (identical to the v69 rounds) — unchanged
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_YAW_GUARD"] = "1.0"
os.environ["X1_TOE_GUARD"] = "0.6"
os.environ["X1_ABD_GUARD"] = "0.8"
os.environ["X1_SWAY_GUARD"] = "0.4"
os.environ["X1_STANCE_MAX"] = "0.38"
# v70 forearm inward-fold guard (the ONLY new experimental variable)
os.environ["X1_SY_GUARD"] = "0.5"
os.environ["X1_FINE_TUNE_ITERS"] = "2000"
os.environ.setdefault("X1_CONTROL_HZ", "50")  # contract: 50 Hz

_HERE = Path(__file__).resolve().parent
_CANDIDATES = [
    _HERE / "checkpoints" / "model_13497_v69.pt",    # v69 task end
    _HERE / "checkpoints" / "model_13497.pt",
]
for _c in _CANDIDATES:
    if _c.is_file():
        os.environ["X1_RESUME_CKPT"] = str(_c)
        break
else:
    raise FileNotFoundError(
        "v70 base checkpoint not found (roboparty_train/checkpoints/model_13497_v69.pt)")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
