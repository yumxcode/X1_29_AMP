#!/usr/bin/env python3
"""v55 launcher: PHASE-LOCKED amplitude prior — the structural amplitude-form
co-design (audit round 10, items 1/3).

The v52/v53/v54 dose ladder proved a flat amplitude prior trades against
phase coherence and DC symmetry (0.3 -> 66-71 deg with phase lost; 0.08 ->
29-32 with walk05 phase sliding). v55's arm_amp_phase_prior makes amplitude
UN-FARMABLE by in-phase flailing or asymmetric lean: the band reward is
gated by the instantaneous antiphase product (sho_L*hip_R + sho_R*hip_L,
the arm_leg_coupling pairing). Offline calibration on committed npz:
refs gated 0.37/0.13 vs policies 0.01-0.09 — the gradient points exactly
at reference-like coherent swing.

Also targets P4/push1.0 (items 2/3): regime 48 (±0.65 @2-6s dense pushes
— v48's recipe that took robustness to 52/60 and solved drift+back05) at
dose +1000.

Base: v45 model_7395 (disc intact). Accept (single checkpoint): arms >= 30
deg AND P7 8/8 AND G2 walk05+10 hip+knee >= 0.85 AND phase antiphase AND
|drift| <= 1.0 AND platform 13/13 AND robustness >= 48.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_ROBUST_TRAIN"] = "48"       # dense-push regime (v48 recipe)
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_FINE_TUNE_ITERS"] = "1000"  # v45 7395 -> total 8395

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_7395_v45.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(p for p in _HERE.parent.glob("model_7395*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v45 base checkpoint not found")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
