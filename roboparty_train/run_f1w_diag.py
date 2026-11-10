#!/usr/bin/env python3
"""F1w env-build diagnostic (256 envs / 50 iters, console-visible).

TASK_20261110_121 hung 95+ min after Phase 2 with zero iterations at 4096
envs. This script runs the SAME env/task with stdout UNBUFFERED straight
to the console (the platform log) so the blocking stage is visible:
dataset gate -> env import -> gym.make -> motion loading -> first
iterations. Timing stamps every stage."""
import functools
import os
import sys
import time
from pathlib import Path

print = functools.partial(print, flush=True)
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
os.environ.setdefault("X1_INITIAL_REWARD", "1")
os.environ.setdefault("X1_GAIT_CLOCK", "1")
os.environ.setdefault("X1_FOOT_CLOCK_W", "1.0")
os.environ.setdefault("X1_CONTACT_CLOCK_W", "0.5")
os.environ.setdefault("X1_DISC_VEL", "1")
os.environ.setdefault("X1_DISC_LINVEL", "1")
os.environ.setdefault("X1_AMP_NUM_STEPS", "30")
os.environ.setdefault("X1_DISC_BUFFER", "24")
os.environ.setdefault("X1_DISC_VMATCH", "1")
os.environ.setdefault("X1_VMATCH_TOL", "0.15")
os.environ.setdefault("X1_CLOCK_T_A", "1.265")
os.environ.setdefault("X1_CLOCK_T_B", "0.384")
os.environ.setdefault("X1_CLOCK_T_LO", "0.30")
os.environ.setdefault("X1_CLOCK_T_HI", "1.30")
os.environ.setdefault("X1_CLOCK_D_A", "0.72")
os.environ.setdefault("X1_CLOCK_D_B", "0.17")
os.environ.setdefault("X1_CLOCK_D_LO", "0.30")
os.environ.setdefault("X1_CLOCK_D_HI", "0.70")
os.environ.setdefault("X1_ROBUST_TRAIN", "1")
os.environ.setdefault("X1_CONTROL_HZ", "50")
os.environ.setdefault("F1W_MOTION_DIR", "f1w_lab_v1")

t0 = time.time()
def mark(msg):
    print(f"[DIAG +{time.time()-t0:7.1f}s] {msg}")

robolab = REPO_ROOT / "roboparty_train" / "robolab"
rsl_rl = REPO_ROOT / "roboparty_train" / "rsl_rl"
os.environ["PYTHONPATH"] = ":".join([str(rsl_rl), str(robolab),
                                     os.environ.get("PYTHONPATH", "")])
mark("PYTHONPATH set; importing train_cli")
sys.argv = ["train.py", "--task=F1W-AMP", "--headless", "--logger=tensorboard",
            "--num_envs=256", "--max_iterations=50"]
import runpy
try:
    runpy.run_path(str(robolab / "scripts" / "rsl_rl" / "train.py"), run_name="__main__")
except SystemExit as e:
    mark(f"train exited: {e}")
mark("DONE")
