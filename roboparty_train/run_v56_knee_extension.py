#!/usr/bin/env python3
"""v56 launcher (human-gait arc, round 1): STRAIGHT-KNEE stance prior.

Base decision (GOAL_HUMAN_GAIT.md §0): v55 phase-locked FAILED its accept
(platform VERDICT 8/13, P3a/P3b kernel gates; TASK_20260916_082) -> this
arc starts from soup534_40 (0.6*v53 + 0.4*v54, merged .pt with v53 disc
states, iter 8394) — the local comprehensive champion holding the FULL
credential set (G2 4/4 + P7 8/8 + drift + robustness 51 + platform
direct-eval 8/8).

Single variable (v38/v41 lesson): knee_extension_stance w=0.15 — per-foot
exp kernel exp(-max(0, knee-0.26)/0.17) gated by stance contact at walk
speeds. Everything else = the v54 recipe (the soup's regime): arm_amp_prior
0.06, arm_asym_lean -1.2, arm_amp_phase 0 (v55 term retired), regime 48.

Accept (single checkpoint, 1000-iter ft from soup534):
  K1 <= 18 deg stance-mid knee flexion (walk10 AND walk05, both feet;
       baseline soup534_40 = 29-37 deg)
  AND no regress: P7 8/8, G2 3-walk + turn, G3 all, |drift| <= 1.0 m,
       robustness >= 48/60, arms >= 24 deg (soup route), 13/13 lineage or
       direct-eval 8/8.
  K2 (>= 25 deg stance knee rhythm) is REPORTED this round, gated from v58
  (the extension pressure could transiently flatten it — dose-response
  reading first).
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "48"       # soup-era regime (v53/v54/v55)
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_FINE_TUNE_ITERS"] = "1000"  # soup 8394 -> 9394

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_soup534.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(_HERE.parent.glob("model_soup534*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("soup534 base checkpoint not found "
                                "(roboparty_train/checkpoints/model_soup534.pt)")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
