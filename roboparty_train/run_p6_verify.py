#!/usr/bin/env python3
"""Cheap P6-chain verification task (50 iters, ~15 min total).

Validates three engineering fixes from the v38 postmortem WITHOUT a full
training run:
  1. isaac_play_dump.py r3 (mirrors play_amp.py exactly — r2 hand-rolled
     the env/runner and crashed silently; failures now print rc+tail to
     the TASK log) + skeleton_render.py GIF -> first-ever P6_play PASS.
  2. final-sweep late-checkpoint mirror (v38's model_6297 was written
     after the monitor's last poll and never registered).
  3. amp_runner disc re-init on obs-shape mismatch (already exercised by
     the v38 run; re-verified here).

The resulting checkpoint is NOT a release candidate (50 iters from v37
with the current cfg); the run exists purely to validate the evidence
chain that every future task depends on.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "3"     # same regime as v38 (cfg unchanged)
os.environ["X1_FINE_TUNE_ITERS"] = "50"  # minimal dose

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_4798_v37.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(p for p in _HERE.parent.glob("model_4798*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v37 base checkpoint not found (embedded or mounted)")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
