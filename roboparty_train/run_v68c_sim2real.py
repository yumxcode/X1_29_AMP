#!/usr/bin/env python3
"""v68c round-3 launcher: continue adaptation to CONVERGENCE (+800 iters from m11498).

Round 1 (TASK_20261014_099): +1200 iters from soup59c under the identified
dynamics — platform 13/13 + P7 PASS; local battery stable at 0.5/1.0 m/s,
arms 33-37 deg, antiphase -1.00, robustness 52/60 (champion 51/60). ONE
unhealed artifact: lateral drift 1.5-2.3 m @11.5 s (champion 0.18 m; gate
<=1.0). Drift is non-monotone across the ladder (1.50 / 4.24 / 2.26 m) —
an in-flight adaptation transient, not a converged equilibrium. This round
continues the SAME recipe unchanged (soup59c cell: robust 3 / form guards /
disc vel / yaw guard 1.0) for +800 more iters from m10699 to let heading
control finish adapting. Ladder m11100..m11499 (+ endpoint soups) will be
battery-tested locally; if drift stays >1.0 m we report it honestly as the
known residual.

Resume base: roboparty_train/checkpoints/model_10699_v68.pt (downloaded from
TASK_20261014_099; full resumable ckpt incl. discriminator + optimizer).
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_YAW_GUARD"] = "1.0"           # soup59c native dose (unchanged)
os.environ["X1_FINE_TUNE_ITERS"] = "800"
os.environ.setdefault("X1_CONTROL_HZ", "50") # contract: 50 Hz

_HERE = Path(__file__).resolve().parent
_CANDIDATES = [
    _HERE / "checkpoints" / "model_11498_v68b.pt",     # round-2 download
    _HERE / "checkpoints" / "model_11498.pt",
    ]
for _c in _CANDIDATES:
    if _c.is_file():
        os.environ["X1_RESUME_CKPT"] = str(_c)
        break
else:
    mounted = sorted(_HERE.parent.glob("model_11498*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("round-3 base checkpoint not found "
                                "(roboparty_train/checkpoints/model_11498_v68b.pt)")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
