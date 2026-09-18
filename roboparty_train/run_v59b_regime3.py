#!/usr/bin/env python3
"""v59b launcher (audit r1 item 4+5 close-out): regime 3 all-gate attempt.

v59 (regime 48, yaw -2.0) readout: drift recovered +4.68/+2.24 ->
+1.13/+0.16 (walk10 13% over gate), walk10 G2 PASS 0.948/0.975, K1
13.6/11.8 — but walk05 R-foot swing degraded to near-drag (apex 11mm
vs v56b's 21mm; L foot alive 22 edges / 24mm).

Hypothesis (single variable): REGIME 48's dense ±0.65 pushes @2-6s are
what kills low-speed swing — v53/v54 trained under regime 3 kept walk05
swing 16/16 events AND took platform 12/13 (P3 kernels PASS, unlike
every regime-48 run's 8-9/13 measurement trade). v59b = v59 recipe with
X1_ROBUST_TRAIN=3 only. If it lands: all non-regress gates AND the
direct-eval kernel credential on one checkpoint.

Base: v56b model_9393 (form champion). +600 iters stabilization.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "3"          # v59b single variable (was 48)
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_FINE_TUNE_ITERS"] = "600"     # v56b 9393 -> 9993
os.environ["X1_YAW_GUARD"] = "2.0"           # v59's recovered-drift dose

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_9393_v56b.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(_HERE.parent.glob("model_9393_v56b*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v56b base checkpoint not found "
                                "(roboparty_train/checkpoints/model_9393_v56b.pt)")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
