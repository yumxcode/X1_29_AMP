#!/usr/bin/env python3
"""v41 launcher: minimal-guard discriminator-led gait (attribution-driven).

The attribution probe (TASK_20260914_070, ATTR_PROBE_REPORT.md) answered
the user's goal-4 question precisely:
  - velocity tracking  -> task reward (control held: P3 0.843/0.549)
  - leg gait + arm PHASE -> style discriminator (guards OFF improved
    walk10 hip 0.718 -> 0.941, walk05 0.698 -> 0.900 PASS)
  - arm DC lean + torso pitch -> still guard-carried (P7h 14.2 > 12,
    walk05 arm asym 32.5 deg, lumbar +17.3 vs 14.9 target)

v41 = the first run of the PROVEN attribution structure:
X1_FORM_GUARDS=2 — keep only arm_asym_lean + lumbar_posture (the two
guard-carried channels); retire arm_pitch_sync / arm_leg_coupling /
leg_amp_asym (discriminator-led, proven redundant). Base: v39
model_6297 (P7 8/8 clean form). +1000 iters, ROBUST=3.

Accept gates: platform 13/13 AND P7 8/8 (P7g/P7h <= 12) AND walk05
hip/knee >= 0.85 AND walk10 hip/knee >= 0.85 AND |drift| <= 1.0 m AND
arms >= 18 deg (v39-level) AND robustness >= 48/60. Success = the
goal-4 structure shipped as a training config.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "2"        # minimal guards (probe-driven)
os.environ["X1_FINE_TUNE_ITERS"] = "1000"  # v39 6297 -> total 7297

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_6297_v39.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(p for p in _HERE.parent.glob("model_6297*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v39 base checkpoint not found")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
