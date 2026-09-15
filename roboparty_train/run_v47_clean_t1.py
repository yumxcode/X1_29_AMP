#!/usr/bin/env python3
"""v47 launcher: clean-regime T1 strike + arm-amplitude push — the
single-checkpoint endgame, attempt 3.

Audit items targeted: (1) arm amplitude ~23 deg vs ref 92/83 (lever:
arm smoothing halved->QUARTERED; halving alone lifted arms 18.7->25.5
in v40), (3) T1 lin kernel 0.85 never met (randomized-regime peak
0.8492; the CLEAN regime reached 0.864 in v36 within +125 iters),
(4) native platform 13/13 for the release point (if this single
checkpoint passes natively, it supersedes the soup lineage argument).

Critical difference vs the failed r3 soup verify: the base is v45's
model_7395 — a NORMAL training save with the discriminator state
INTACT. Resuming it in the clean regime keeps the trained disc (no
fresh-disc adversarial transient that damaged r3), so the style
channel continues to hold form while the kernels re-converge in the
clean domain. Form erosion (the v36 lesson: walk05 knee fell to 0.678
by +600) is countered by the hip+knee RMS guard at -1.0 and a SHORT
+400 dose.

Config: X1_ROBUST_TRAIN=0 (clean), X1_FORM_GUARDS=1 (full set,
leg_amp_asym -1.0 midpoint), arm smoothing quartered, lerp 0.68,
style 2.5, disc obs 10 bodies — everything else identical to v45.

Accept (single checkpoint, native): platform 13/13 with T1 lin >= 0.85
AND P7 8/8 AND G2 walk05 + walk10 hip AND knee >= 0.85 AND arms >= 25
deg AND |drift| <= 1.0 m. push1.0 robustness is the accepted cost of
the clean regime (documented trade; the soup remains the robustness
champion at 51/60).
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "0"        # CLEAN regime (kernel 0.86 domain)
os.environ["X1_FORM_GUARDS"] = "1"         # full guard set (anti-erosion)
os.environ["X1_FINE_TUNE_ITERS"] = "400"   # v45 7395 -> total 7795

_HERE = Path(__file__).resolve().parent
# v45 model_7395 embedded copy preferred (disc state intact)
_EMBEDDED = _HERE / "checkpoints" / "model_7395_v45.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    # platform resume mount: checkPointMountPath "X1_29_AMP/" lands the
    # model_7395*.pt at repo root — auto-detected by run_x1_amp_train.py
    mounted = sorted(p for p in _HERE.parent.glob("model_7395*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v45 base checkpoint not found "
                                "(embed model_7395_v45.pt or mount model_7395*.pt)")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
