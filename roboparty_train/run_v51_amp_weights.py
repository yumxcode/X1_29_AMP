#!/usr/bin/env python3
"""v51 launcher: ref-side DEMO AMPLITUDE WEIGHTING — the audit's other named
lever, now implemented (no code changes needed: the motion-weights mechanism
exists).

v50's dose-response curve proved the 22-26 deg arm plateau is locked by the
disc's style preference — i.e., by WHAT THE DEMOS ARE. This run changes the
demo distribution itself: the highest wrist-swing clips are upweighted so
the discriminator's "style average" carries more amplitude:
  0003_treadmill_jog 2.0 -> 5.0  (wrist swing 198/244 mm — highest)
  0009_normal_jog1   2.0 -> 5.0  (121/168 mm)
  36_01              1.0 -> 3.0  (197/165 mm)
(mirrors matched — the symmetric-pair property is preserved.)
Plus: velocity disc channel ON (X1_DISC_VEL=1) and arm smoothing back to
the 1/4 peak band (v50 proved 1/8 regresses).

Base v45 model_7395 (disc intact), regime 3, full guards, +1000 iters.
PRIMARY: arms >= 30 deg on the local battery (walk10/walk05 shoulder-pitch
swing). Secondary: standard axes; gates-green variant = new champion.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_FINE_TUNE_ITERS"] = "1000"  # v45 7395 -> total 8395

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
