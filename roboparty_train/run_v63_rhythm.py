#!/usr/bin/env python3
"""v63 launcher: RHYTHM round (GOAL_RHYTHM.md) — human cadence prior.

Single-variable discipline: the v59c recipe FROZEN (regime 3, guards 1,
disc-vel on, yaw -1.0, 50 Hz) + ONE new lever:
  X1_CADENCE_PRIOR=0.6   gait_period_prior — exp-kernel touchdown-interval
                         reward vs speed-conditioned human cycle target
                         T*(v)=clamp(1.46-0.36v, 0.70, 1.75) s.

Base: v59c m9500 (TASK_20260918_134 mid-flight clean-drift point — the
50 Hz-trained parent of the 13/14-gate m9500@100Hz direct-drive deployment
point). Fine-tune +1000 iters (behavioral change is large: cycle 0.4-0.55 s
-> ~1.0-1.1 s, stride 0.46-0.53 -> ~0.95 m; 600 may not be enough —
mid-flight checkpoints mirrored by the v29e4 monitor).

Training stays at 50 Hz on purpose (v61 six falsified 100 Hz fine-tune
routes: the disc equilibrium at 100 Hz actively rewards micro-gait);
rhythm is validated by direct-driving the resulting checkpoint at 100 Hz
(eval_100hz_battery.py --control-dt 0.01, R1/R2/R3/R4 gates included).

Accept criteria (pre-registered, GOAL_RHYTHM.md §2):
  walk10: cycle [0.85,1.35] s, stride >= 0.78 m, cv <= 0.20
  walk05: cycle [1.00,1.65] s, stride >= 0.42 m, >= 4 stance events/foot
  no-regress: G1/G2/G3/K1/K2/H2, drift <= 1.0 m, arms >= 24 deg,
              robustness >= 48/60 (hip 0.803 existing gap must not worsen).
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_YAW_GUARD"] = "1.0"           # v59c native dose (frozen)
os.environ["X1_FINE_TUNE_ITERS"] = "1000"    # large behavioral shift
os.environ["X1_CADENCE_PRIOR"] = "0.6"       # THE single new variable

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_9500_v59c.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(_HERE.parent.glob("model_9500_v59c*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v59c m9500 base checkpoint not found")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
