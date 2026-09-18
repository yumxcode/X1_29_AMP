#!/usr/bin/env python3
"""v59c launcher: the last cell of the regime-x-yaw 2x2 matrix (3 / -1.0).

Matrix so far (all from the v56b K1 checkpoint, +600 iters):
  48 / -1.0  = v56b m9393: form champion, drift 4.68/2.24 FAIL
  48 / -2.0  = v59  m9992: drift 1.13/0.16 (13% over), G2 PASS, R-foot drag
  3  / -2.0  = v59b: platform 12/13 + kernels PASS + K2 26.1/25.8 (both
               gates!), but walk10 phase went IN-PHASE and drift 2.6-3.3
  3  / -1.0  = THIS RUN — the v53/v54 native recipe (that regime held
               walk05 swing 16/16 AND platform 12/13 kernels; parents of
               the soup lineage). Hypothesis: the K1 checkpoint returns
               to its lineage's native environment and keeps everything.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_FINE_TUNE_ITERS"] = "600"
os.environ["X1_YAW_GUARD"] = "1.0"           # v53/v54 native dose

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
