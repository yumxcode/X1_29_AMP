#!/usr/bin/env python3
"""v69 launcher: gait-posture fix fine-tune from m11498 (user defect spec
2026-10-15).

Defects (video review of m11498, user):
  D1 外八 — whole-leg hip_roll abduction (L+13.1/R+12.3 deg, stance 460mm
     vs demo 220-260mm; user clarified: 整条腿往外展) + foot toe-out
     (heading L+20.1/R-10.8 deg) -> small leg swing amplitude
  D2 腰摆 — lumbar roll AC 27 deg ptp @2Hz, lumbar yaw AC 52 deg, base
     lateral sway 619mm
  D3 腰带手 — corr(waist_roll -> L_shoulder) +0.96, waist leading 0.1s:
     arms carried by the waist pump instead of shoulder-driven

Recipe = v68/m11498 baseline (soup59c cell under identified dynamics)
+ three v69 posture guards (single-variable family, env knobs):
  X1_TOE_GUARD  = 0.6  foot heading relu(|hdg|-6deg), turn-gated
  X1_ABD_GUARD  = 0.8  hip_roll outward DC relu(|dc|-4deg), slow EMA
  X1_SWAY_GUARD = 0.4  high-passed |lumbar_roll|+|lumbar_yaw|+2x|hip_roll|
  X1_STANCE_MAX = 0.38 feet_distance_y band max 0.50 -> 0.38
Unit-calibrated on the m11498 forensics (roboparty_train/test_v69_guards.py):
total initial pressure ~-0.7/step, all ->0 at target gait.

Single LONG round (+2000 iters from m11498) per the V68 lesson: multi-round
resumes restart the LR schedule and oscillate drift/waist metrics.
"""
import os
import runpy
import sys
from pathlib import Path

# soup59c cell (identical to v68 rounds) — unchanged
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_YAW_GUARD"] = "1.0"
# v69 posture guards (the ONLY experimental variables)
os.environ["X1_TOE_GUARD"] = "0.6"
os.environ["X1_ABD_GUARD"] = "0.8"
os.environ["X1_SWAY_GUARD"] = "0.4"
os.environ["X1_STANCE_MAX"] = "0.38"
os.environ["X1_FINE_TUNE_ITERS"] = "2000"
os.environ.setdefault("X1_CONTROL_HZ", "50")  # contract: 50 Hz

_HERE = Path(__file__).resolve().parent
_CANDIDATES = [
    _HERE / "checkpoints" / "model_11498_v68b.pt",   # v68 round-2 delivery
    _HERE / "checkpoints" / "model_11498.pt",
]
for _c in _CANDIDATES:
    if _c.is_file():
        os.environ["X1_RESUME_CKPT"] = str(_c)
        break
else:
    raise FileNotFoundError(
        "v69 base checkpoint not found (roboparty_train/checkpoints/model_11498_v68b.pt)")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
