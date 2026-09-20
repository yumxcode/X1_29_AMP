#!/usr/bin/env python3
"""v61e launcher: v61d2 + arm_amp_prior 0.06 -> 0.12 (disc amplitude channel
replacement at 100 Hz).

v61d2 forensics (all three 100 Hz fine-tunes: v61b/c/d2):
  - Base m9393 driven at 100 Hz in MuJoCo KEEPS amplitude (arms 28 deg,
    swing 24-28 mm) — the base policy is fine; the FINE-TUNE destroys it.
  - Per-step reward comparison vs the 50 Hz champion (v59c): the disc
    equilibrium saturates at 100 Hz (style rew ≈ 1.0 — the warm disc rates
    the collapsed micro-gait as fully demo-like; champion sat at 0.80 with
    a live gradient), arm_amp_prior credit falls to 1/3 (amplitude leaves
    the band), joint_vel_l2 improves (less motion = less penalty).
  => The style channel's amplitude pressure is DEAD at 100 Hz; the recipe
     needs the direct calibrated lever: X1_ARM_PRIOR=0.12 (dose-response:
     v52 0.3 -> 66-71 deg, v53 0.08 -> 29-32 deg band-edge, 0.06 -> 34.7
     only with a live disc)."""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_CONTROL_HZ"] = "100"
os.environ["X1_AMP_NUM_STEPS"] = "6"
os.environ["X1_ARM_PRIOR"] = "0.12"
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_FINE_TUNE_ITERS"] = "1200"
os.environ["X1_YAW_GUARD"] = "1.0"

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_9393_v56b.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(_HERE.parent.glob("model_9393_v56b*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v56b base checkpoint not found")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
