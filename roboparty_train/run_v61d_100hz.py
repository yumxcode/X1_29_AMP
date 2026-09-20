#!/usr/bin/env python3
"""v61d launcher: v61c + disc ::stride (native-cadence disc window).

Root cause chain (v61b/v61c forensics):
  Both first 100 Hz runs collapsed style-guided AMPLITUDE — arms 14.5-18.8
  deg (champion 34.7), swing < 25 mm (champion 54), drag-walk micro-stepping
  — while ALL other gates passed (13/13, kernels, drift, robustness).
  Diagnosis: demo motions are native 50 Hz; at 100 Hz the animation manager
  fetches demo frames at 10 ms spacing = LINEAR INTERPOLATION between the
  20 ms keys. Inside a native interval the interpolated triple's 2nd
  difference is EXACTLY zero — a "temporal smoothness" artifact the disc
  separates trivially. The style reward then rewards smooth micro-motion
  (drag walking) instead of gait style. This also explains why the fresh
  6-step disc (v61c) did not fix it: interpolation persists at any fetch
  spacing finer than 20 ms.

Fix (v61d, commit-local): AMPDiscriminator.disc_obs_stride = 0.02/step_dt
  = 2 at 100 Hz. Both sides keep a 6x10 ms window but the disc consumes
  every OTHER frame: 3 frames at 20 ms spacing — the exact input semantics
  of the 50 Hz champion (v59c/soup59c lineage), whose demo frames were
  equally 20 ms-spaced samples (of the same piecewise-linear demos).
  First/second-difference parity restored on both sides.

Bonus: disc input 121 x 3 = 363 == m9393's disc Linear(363->1024->512)
  (verified from checkpoint) — the discriminator now WARMS-RESUMES from
  the v56b base instead of re-initializing (v61c's disc was fresh; the
  base's amplitude-competent style signal is preserved from iter 0).
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_CONTROL_HZ"] = "100"
os.environ["X1_AMP_NUM_STEPS"] = "6"   # 6x10ms window; disc strides ::2 -> 3x20ms
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
