#!/usr/bin/env python3
"""v29e clean-phase fine-tune launcher.

Resumes from the v29d robust fine-tune checkpoint (mounted model_9497.pt,
TASK_20260909_231) for +1000 iterations with the v29 randomization DISABLED
(exact v28d environment). Hypothesis under test: the policy keeps (most of)
its internalized push/latency recovery while the gait re-converges to the
AMP style — recovering walk05 knee symmetry and the platform kernel gates
that v29d narrowly missed (lin 0.811 vs 0.82, ang 0.472 vs 0.5).
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "0"       # disable push/delay randomization
os.environ["X1_FINE_TUNE_ITERS"] = "1000"  # +1000 clean iterations

HERE = Path(__file__).resolve().parent
sys.argv = [str(HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(HERE / "run_x1_amp_train.py"), run_name="__main__")
