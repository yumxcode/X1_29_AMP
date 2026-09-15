#!/usr/bin/env python3
"""v48 launcher: joint-convergence regime — drift + back05 + push1.0 in one
run, on the champion base.

Audit items targeted together (they interact through the yaw/asymmetry
axis): (1) v47 drift +1.81/+2.32 m (yaw guard -0.5 -> -1.0), (2) v47
back05 hip 0.812 + toe-first (hip+knee RMS guard already -1.0 — now
trained under disturbances where it must HOLD, not only form), (3)
push1.0 1/5 on every checkpoint since v38 (new regime '48': ±0.65
magnitude at HIGH frequency 2-6 s so ±1.0-magnitude events are dense;
v42 proved ±0.65 recovers push1.0 2/5 at a −0.026 kernel cost, and the
v45->v47 lineage's kernel headroom (clean 0.868, regime 0.843) absorbs
it).

Base: v45 model_7395 (disc state INTACT). Regime 48 is a NEW
randomization envelope (not the v45 regime, not clean) — the knee/hip
guards and arm guards are all live, arm smoothing stays QUARTERED
(v47's arms 24.6/26.0 were the best clean point; keep the gain).
Dose +800 (regime change needs the v37 precedent's convergence margin).

Accept (single checkpoint): platform 13/13 (T1 may MISS at 0.82-gate
level, that is acceptable — v47 stays T1 champion) AND P7 8/8 AND G2
walk05 + walk10 hip AND knee >= 0.85 AND back05 hip >= 0.85 AND G3
back05 toe_first == 0 AND |drift walk10| <= 1.0 m AND robustness >= 48
with push1.0 >= 2/5 AND arms >= 22 deg.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "48"      # joint-convergence regime
os.environ["X1_FORM_GUARDS"] = "1"        # full guard set
os.environ["X1_FINE_TUNE_ITERS"] = "800"  # v45 7395 -> total 8195

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_7395_v45.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(p for p in _HERE.parent.glob("model_7395*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v45 base checkpoint not found")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
