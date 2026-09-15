#!/usr/bin/env python3
"""v44 launcher: single-checkpoint convergence, second attempt — the knee
guard mirrors the proven hip-guard success.

v43 verdict (TASK_20260914_111): platform 13/13 + walk10 G2 0.900/0.933 +
walk05 hip 0.860 + drift 0.32 — the closest ever to dual-family pass;
misses are walk05 knee 0.745 (and P7h 12.09 vs 12 by 0.09 deg). v42 is
the local all-green champion but platform 9/13 at the ±0.65 regime.

The v39->v40 precedent: a targeted RMS-amplitude symmetry guard took
walk05 hip 0.698 -> 0.978. v44 applies the same lever to the remaining
deficient pair: leg_amp_asym generalized to hip+knee pairs (calibration:
v42 symmetric rollouts 0.009-0.012 rad vs v43 knee-deficient 0.022-0.034
— 2-3x separation at weight -0.8).

Config: v42 model_6498 base (local all-green start), ±0.5 kernel regime
(ROBUST=3 — the 0.842-kernel domain), +600 iters (kernel restored by
+200; this dose eroded v43's knee — the NEW knee guard exists exactly
to hold it), full guard set, arm smoothing halved, disc 10 bodies,
style 2.5, lerp 0.68.

Accept (single checkpoint): platform 13/13 AND P7 8/8 AND G2 walk05 +
walk10 hip AND knee >= 0.85 AND |drift| <= 1.0 m AND robustness >= 48/60.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "3"        # kernel regime (0.842 plateau)
os.environ["X1_FORM_GUARDS"] = "1"         # full guard set
os.environ["X1_FINE_TUNE_ITERS"] = "600"   # v42 6498 -> total 7098

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_6498_v42.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(p for p in _HERE.parent.glob("model_6498*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v42 base checkpoint not found")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
