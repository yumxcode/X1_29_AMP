#!/usr/bin/env python3
"""v64b launcher: rhythm-lever PAIR — cadence prior + DENSE swing prior.

v64@0.6 verdict (TASK_20260921_118, m11499/m11000): FAIL pre-registered
accept. Forensics:
  - on-pod P8 identical to m10500 baseline (2.857 Hz / 0.19 s / 0.95 /
    0.145 m — cadence did not move in-domain; NOTE: on-pod P7/P8 harness
    appears to glob model_*.pt alphabetically and may have read the BASE
    model_10500.pt — a harness defect to fix separately);
  - local sim2sim (honest ckpts + LPF): REGRESSED to full drag-gait (both
    feet <12 mm for 100% frames, 0.93 m/s by skating, 0 swing events);
  - structural causes: (1) event-prior CREDIT GAP — the TD-edge reward
    lands 0.35-1.1 s after the actions that set the interval, beyond the
    24-step rollout chunk; (2) DRAG IMMUNITY — no events = no reward but
    no penalty, so ground-skating is a free escape for "long intervals".

v64b = the rhythm-lever FAMILY (v57 heel_first+sole_flat coupled-variable
precedent), from the CLEAN v63 base m10500:
  X1_CADENCE_PRIOR=0.6 (unchanged — anchors the full cycle T*(v))
  X1_SWING_PRIOR=0.3   NEW — swing_airtime_prior: dense per-airborne-step
                       exp(-|a-0.43 s|/0.15); pays ~0.14/step for human
                       swing (0.43 s, P8 reference median) vs ~0.006 for
                       the current micro-gait and exactly 0 for drag —
                       23-47x differential, credit-local (no chunk gap),
                       and it closes the drag escape.

Everything else = the v63 recipe frozen. +1000 iters @100 Hz from
model_10500_v63.pt (NOT from m11499 — the drag-corrupted point must not
be inherited).

PRE-REGISTERED accept: best checkpoint @100 Hz + LPF local battery:
  P8 R1+R2+R3+R4 on walk10 AND walk05; no-regress P7 8/8, arms >= 24 deg,
  drift <= 1.0 m, 5/5 survived, K1 <= 18, robustness >= 48/60.
Dose ladder: swing 0.3 first; if swing duration moves <30% of the way to
0.43 s -> 0.6 (v64c); if gait destabilizes -> 0.15.
"""
import os
import runpy
import sys
from pathlib import Path

# --- v63 recipe (frozen) ------------------------------------------------
os.environ["X1_CONTROL_HZ"] = "100"
os.environ["X1_MOTION_DIR"] = "x1_lab_v32_200"
os.environ["X1_AMP_NUM_STEPS"] = "60"
os.environ["X1_DISC_STRIDE"] = "2"
os.environ["X1_DISC_BUFFER"] = "24"
os.environ["X1_DISC_LINVEL"] = "1"
os.environ["X1_ACT_LPF_HZ"] = "10"
os.environ["X1_ACT_LPF_ORDER"] = "2"
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_YAW_GUARD"] = "1.0"

# --- v64b rhythm-lever pair ----------------------------------------------
os.environ["X1_CADENCE_PRIOR"] = "0.6"
os.environ["X1_SWING_PRIOR"] = "0.3"
os.environ["X1_FINE_TUNE_ITERS"] = "1000"

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_10500_v63.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(_HERE.parent.glob("model_10500_v63*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v63 m10500 base checkpoint not found")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
