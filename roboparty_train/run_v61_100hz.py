#!/usr/bin/env python3
"""v61 launcher: the v59c recipe re-run at 100 Hz control (v61 100 Hz arc).

Goal (this arc): 50 Hz -> 100 Hz control frequency, keep sim2sim pass and
the soup59c gait metric set.

What changes vs run_v59c_matrix.py:
  X1_CONTROL_HZ=100      decimation 4 -> 2 (sim.dt 0.005 kept; policy 10 ms).
                         AmpEnvCfg picks it up; x1_amp_env_cfg re-normalizes
                         EMA alphas (tau = dt/alpha), action-delay steps
                         (20 ms coverage), action-rate weights, and
                         x1_amp_agent_cfg doubles style_reward_scale
                         (predict_style_reward multiplies by dt).
  X1_FINE_TUNE_ITERS 600 -> 1200: same SIMULATED adaptation time (each iter
                         is 24 control steps; at 10 ms that is half the
                         seconds of the 50 Hz runs). Wall time is roughly
                         unchanged: per-iter physics halves (2 substeps vs 4).

Base: v56b m9393 — same as v59/v59b/v59c (the soup59c parent lineage).
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_CONTROL_HZ"] = "100"
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_FINE_TUNE_ITERS"] = "1200"
os.environ["X1_YAW_GUARD"] = "1.0"           # v53/v54/v59c native dose

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
