#!/usr/bin/env python3
"""v62 launcher: 100 Hz + PCHIP-upsampled 200 fps demos (user-directed,
contract rev3): does native-cadence demo data fix the style collapse?

Config delta vs v61d2 (the last 100 Hz run with the stride machinery):
  X1_MOTION_DIR=x1_lab_v32_200  demo 120 fps -> 200 fps (PCHIP, keys exact,
                                FK-recomputed key bodies; 22/22 validated)
  X1_DISC_STRIDE=1              disc window at the 10 ms cadence of the
                                200 fps grid: phase-coherent fetch (sub-key
                                offset CONSTANT within each window; lerp
                                residual 5.3x smaller than the 120 fps
                                grid). v61d2 sampled 120 fps demos at
                                20 ms (stride 2).
  X1_AMP_NUM_STEPS=6            6x10 ms = 60 ms window (same span as the
                                50 Hz champion's 3x20 ms).
Everything else = the v59c recipe (regime 3, guards, disc-vel, yaw 1.0,
prior 0.06 default, lerp 0.68 default) on the v56b m9393 base, +1200 iters.

PRE-REGISTERED falsification (answer to the user's question):
  Fix VALID  iff sim2sim walk10 arms >= 24 deg AND walking gates green
              (stride events >= 40, swing clearance, drift <= 1.0) AND
              non-regress gates hold (kernels 13/13 platform, K1 <= 18,
              robust >= 48/60).
  Otherwise  the data-side route joins the falsified list (6th) and the
              answer to rev3 is: upsampling alone does NOT fix the collapse
              (root cause sits in the training dynamics, per v61d2's
              distribution-equivalent failure)."""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_CONTROL_HZ"] = "100"
os.environ["X1_MOTION_DIR"] = "x1_lab_v32_200"
os.environ["X1_AMP_NUM_STEPS"] = "6"
os.environ["X1_DISC_STRIDE"] = "1"
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
