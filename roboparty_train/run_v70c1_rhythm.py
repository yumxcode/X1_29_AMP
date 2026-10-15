#!/usr/bin/env python3
"""v70c1 launcher: RHYTHM round — hip-phase reference alone (user spec
rev2 2026-10-15: 正常抬腿行走, 步频~1Hz; current family walks at
2.1-2.3 Hz hip oscillation with P8 zero swing events = 小步快走/拖步).

v64c tried this mechanism on the 100Hz line from m10500 (dose 0.4):
income froze at partial credit (0.16-0.23 vs 0.8) — a basin problem:
fast stepping is the stability attractor and scalar reward does not
cross it. NEW conditions here: 50Hz line, v68 identified dynamics, all
posture guards converged (toe/abd/sway/sy/lean), physics feasibility
L-side says human cadence needs LESS hip torque than tapping (21 vs 91
Nm p95) — and the parallel arm v70c2 adds an LPF action governor that
physically removes the fast-step equilibrium (A/B on the ceiling).

  X1_HIP_PHASE = 0.5  hip_phase_reference_prior: anti-phased sinusoid
                     hip reference at T*(v)=1.46-0.36v clamped [0.70,1.75]
                     -> at 1.0 m/s T*=1.10s = 0.91 Hz/foot (user target)
"""
import os
import runpy
import sys
from pathlib import Path

# v70b recipe (identical) — unchanged
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_YAW_GUARD"] = "1.0"
os.environ["X1_TOE_GUARD"] = "0.6"
os.environ["X1_ABD_GUARD"] = "0.8"
os.environ["X1_SWAY_GUARD"] = "0.4"
os.environ["X1_STANCE_MAX"] = "0.38"
os.environ["X1_SY_GUARD"] = "0.5"
os.environ["X1_LEAN_GUARD"] = "0.5"
# v70c1 rhythm lever (the ONLY new experimental variable)
os.environ["X1_HIP_PHASE"] = "0.5"
os.environ["X1_FINE_TUNE_ITERS"] = "1500"
os.environ.setdefault("X1_CONTROL_HZ", "50")

_HERE = Path(__file__).resolve().parent
_CANDIDATES = [
    _HERE / "checkpoints" / "model_16495_v70b.pt",
    _HERE / "checkpoints" / "model_16495.pt",
]
for _c in _CANDIDATES:
    if _c.is_file():
        os.environ["X1_RESUME_CKPT"] = str(_c)
        break
else:
    raise FileNotFoundError("v70c1 base checkpoint not found (model_16495_v70b.pt)")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
