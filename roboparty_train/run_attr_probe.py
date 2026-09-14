#!/usr/bin/env python3
"""Attribution probe: is gait form carried by the style DISCRIMINATOR or by
the task-side hand-engineered form guards? (user goal: velocity tracking ->
task reward; ALL other gait form -> style discriminator)

Design: short fine-tune (+300 iters) from the v40 final checkpoint with
X1_FORM_GUARDS=0 — the five hand-engineered form guards (arm_pitch_sync /
arm_asym_lean / arm_leg_coupling / lumbar_posture / leg_amp_asym) zeroed.
Everything else unchanged (disc obs 10 bodies, style 2.5, lerp 0.65,
ROBUST=3, task velocity rewards, physics regularization).

Verdict table (evaluated on the local battery, vs the v40 baseline):
  form HOLDS (P7 8/8, walk05/10 hip+knee >= 0.85, arm swing within ~20%
    of v40): discriminator is the form controller -> goal structure
    ACHIEVED; the guards can be retired in the next full run.
  form REGRESSES (any P7 fail / hip < 0.85 / arms -30%): the guards are
    still carrying form -> style not yet dominant; document and keep the
    guards (the disc-led route needs a stronger style mechanism, e.g.
    per-body disc weighting or longer style-only phases).
  Velocity kernels (P3a/P3b) are task-side by construction and must stay
    >= 0.82 in BOTH cases — that channel is never in question.

Checkpoints: v40 final (model_5498, embedded at
roboparty_train/checkpoints/model_5498_v40.pt after the v40 run — this
launcher expects it; if absent, mount or OSS per the established chain).
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "3"        # identical to v40 (no confounds)
os.environ["X1_FORM_GUARDS"] = "0"         # THE ABLATION: form guards OFF
os.environ["X1_FINE_TUNE_ITERS"] = "300"   # short probe (+300)

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_5498_v40.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(p for p in _HERE.parent.glob("model_5498*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError(
            "v40 base checkpoint not found — embed model_5498 at "
            "roboparty_train/checkpoints/model_5498_v40.pt first (v40 run "
            "TASK_20260914_037 output)")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
