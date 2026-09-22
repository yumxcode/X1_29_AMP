#!/usr/bin/env python3
"""v67 launcher: overground-dominant demo library, from scratch (audit r2
directive 4 — the REAL pre-registered route, replacing the near-in-place
library instead of gating it).

Evidence trail that justifies this exact configuration:
  - Physics (acceptance/physics_rhythm_feasibility.py): human-cadence
    swinging needs hip p95 ~6.5-25 Nm vs the 4 Hz tap's 91-126 Nm —
    human rhythm is PHYSICALLY EASIER on X1; the blocker is the learning
    distribution, not the hardware. Gate stays human-calibrated.
  - v65/v65b: VMATCH gating on the entrenched base did not move cadence
    (1.85->1.89 Hz) — composition alone is insufficient once the basin is
    set; the library itself must be replaced AND training restarted.
  - v66/v66b: scratch with the OLD (77% in-place) library produced 4 Hz
    tapping — consistent with the disc's demo stream being dominated by
    in-place horizontal kinematics.
  - Local AMASS deep screen (116 pass -> 3 GOOD): only ONE new usable
    overground clip exists locally (138_18, 93 spm, 1.19 m/s); rescued
    from the v30 dataset (10-body FK 0.0 mm, mirror 0.35 mm).

v67 dataset (roboparty_train/robolab/data/motions/x1_lab_v67):
  overground 36_01/36_11/0026/138_18 (+mirrors) = 74% weight mass;
  in-place family token weights (serve stand under VMATCH); no 103_07.

Levers: the v66 package unchanged otherwise (rhythm trio, full VMATCH,
600 ms macro disc, 50 Hz, regime 3). Pod budget sized for the wall clock
(v66 died at 3463/4000 in 2h51m): X1_FRESH_ITERS=2800 (~2h20m + wrap-up),
continuation pods (v67b...) extend to convergence (ep_len >= 900 proxy).

PRE-REGISTERED accept: same as v66 — P8 R1-R4 walk10+walk05 at 100 Hz
direct-drive (no LPF) + no-regress set. Decision: pass -> contract met;
converged-fail -> demo-replacement route falsified with the physics
evidence attached (structural limit of this AMP stack for rhythm);
unconverged -> continue pods before verdict.
"""
import os
import runpy
import sys
from pathlib import Path

# --- v59c/v66 recipe (frozen) --------------------------------------------
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_YAW_GUARD"] = "1.0"
os.environ["X1_AMP_NUM_STEPS"] = "30"
os.environ["X1_DISC_LINVEL"] = "1"
os.environ["X1_DISC_BUFFER"] = "24"
os.environ["X1_DISC_VMATCH"] = "1"
os.environ["X1_CADENCE_PRIOR"] = "0.6"
os.environ["X1_SWING_PRIOR"] = "0.3"
os.environ["X1_HIP_PHASE"] = "0.4"

# --- v67 THE variable: the overground-dominant demo library --------------
os.environ["X1_MOTION_DIR"] = "x1_lab_v67"

# --- fresh run, pod-budget-sized ------------------------------------------
os.environ.pop("X1_RESUME_CKPT", None)
os.environ.pop("X1_FINE_TUNE_ITERS", None)
os.environ["X1_FRESH_ITERS"] = "2800"

_here = Path(__file__).resolve().parent
sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
