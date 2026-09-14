#!/usr/bin/env python3
"""v43 launcher: the convergence endgame — restore the kernel regime on the
form-locked v42 base.

v42 verdict (TASK_20260914_088): ALL local gates green for the first time
(P7 8/8 coupling +0.93, walk10 G2 0.852/0.983, walk05 G2 0.968/0.999,
drift +0.94) and robustness recovered (51/60, push1.0 2/5 from 0-1/5) —
the only misses are platform P3a/P3c/P3d/P4, ALL measured under the
±0.65 @4-8s push regime which mechanically caps the training-time kernels
at 0.816-0.820 (vs 0.840-0.844 at ±0.5 @5-10s in v39/v41 — the kernel
plateau is regime-set, seen across the whole v42 trajectory).

v43: return to the ±0.5 regime (X1_ROBUST_TRAIN=3) on the v42 base —
the form is already locked in (v42 walk G2 is the strongest profile of
the arc), so a short +600-iter fine-tune should lift the kernel back to
the ~0.84 regime plateau while the form guards + style hold it. Every
ingredient is individually proven; the only question is whether the
form survives the regime change at this short dose (the v36 lesson says
form erodes over ~500+ iters of regime change — the accept gate checks
exactly that).

Accept gates (single checkpoint): platform 13/13 AND P7 8/8 AND G2
walk05+walk10 hip+knee >= 0.85 AND |drift| <= 1.0 m AND arms >= 20 deg
AND robustness >= 48/60 with push1.0 >= 1/5.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "3"        # back to the 0.84-kernel regime
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
