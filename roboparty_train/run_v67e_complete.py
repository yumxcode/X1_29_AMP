#!/usr/bin/env python3
"""v67e launcher: COMPLETE the v67d scratch run to convergence scale.

v67d (TASK_20260922_046) ran its full 2800-iter pod budget (ep_len 599,
unconverged; same trajectory as v66 at comparable iters — 587 @ ~3400).
Local forensics at m2799 (100 Hz direct-drive):
  - hip cadence 1.92 Hz — HALFWAY to human (v66's old-library scratch hit
    3.83 Hz): the overground-dominant library DID shift the equilibrium
    cadence, first quantitative movement of the rhythm arc;
  - but swing durations 20-85 ms (below the 100 ms P8 detector minimum),
    airtime 1.5/6.2% — micro-swings persist;
  - m2000 mid-flight had transient real stepping (f0 22% air, 46 swings)
    that regressed by 2799 — the fast-tap basin is still winning at this
    training stage.
Pre-registered path for unconverged (ep_len < 900): continuation pod.
+1200 iters from m2799 -> 4000 total = the standard fresh-run convergence
budget (v26-v31 series converge ep_len ~990 at 4000).

Fidelity check (audit r2 directive 3): the first logged iterations after
resume must continue from ep_len ~599, not reset (v66b precedent 587->628).

Env = v67d frozen (overground library 69% mass, rhythm trio, full VMATCH,
600 ms macro disc @50 Hz, regime 3, no LPF)."""
import os
import runpy
import sys
from pathlib import Path

# --- v67d recipe (frozen) --------------------------------------------------
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_YAW_GUARD"] = "1.0"
os.environ["X1_AMP_NUM_STEPS"] = "30"
os.environ["X1_DISC_LINVEL"] = "1"
os.environ["X1_DISC_BUFFER"] = "24"
os.environ["X1_DISC_VMATCH"] = "1"
os.environ["X1_MOTION_DIR"] = "x1_lab_v67"
os.environ["X1_CADENCE_PRIOR"] = "0.6"
os.environ["X1_SWING_PRIOR"] = "0.3"
os.environ["X1_HIP_PHASE"] = "0.4"

# --- continuation -----------------------------------------------------------
os.environ["X1_FINE_TUNE_ITERS"] = "1200"

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_2799_v67d.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(_HERE.parent.glob("model_2799_v67d*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v67d m2799 base checkpoint not found")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
