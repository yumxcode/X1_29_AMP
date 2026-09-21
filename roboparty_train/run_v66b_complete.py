#!/usr/bin/env python3
"""v66b launcher: COMPLETE the v66 scratch run (killed by pod wall-clock).

v66 (TASK_20260922_036) was killed at iteration ~3463/4000 (2h51m pod cap;
ep_len 587 = unconverged; mid-flight m3000 saved by the monitor insurance).
Forensics at m3000 (100 Hz direct-drive): REAL stepping emerged (airtime
21.7/14.9%, 46 swing events/foot, lift 27-30 mm — the family's first air
since the rhythm arc began) but at a ~4 Hz tapping cadence (swing ~54 ms <
the P8 detector's 100 ms minimum). The rhythm trio did not prevent a
faster-than-human basin from forming.

Declaring route 3 falsified on an UNCONVERGED checkpoint would be weak
evidence (audit directive 5: evaluate, don't terminate). v66b resumes
model_3000 for +1000 iters to complete the pre-registered 4000-iter
scratch — if the converged policy still fails P8, the route-3 verdict is
airtight; if it converges toward rhythm, the gate decision is live.

Env = v66 frozen (identical levers: rhythm trio + full vmatch + 600 ms
macro disc @50 Hz, BUFFER 24, regime 3, no LPF). Resume semantics:
--max_iterations is ADDITIONAL on resume (base 3000 + 1000 = 4000)."""
import os
import runpy
import sys
from pathlib import Path

# --- v66 recipe (frozen) --------------------------------------------------
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

# --- continuation ----------------------------------------------------------
os.environ["X1_FINE_TUNE_ITERS"] = "1000"

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_3000_v66.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(_HERE.parent.glob("model_3000_v66*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v66 m3000 base checkpoint not found")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
