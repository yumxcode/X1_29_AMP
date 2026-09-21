#!/usr/bin/env python3
"""v64c launcher: PHASE-LOCKED hip reference — the rhythm ANCHOR.

v64 (event cadence prior @0.6) and v64b (dense swing pair @0.3+0.6)
verdicts: BOTH FAIL (pre-registered criteria).
  - sim2sim regressed toward drag in both (v64b m11499: airtime 0.4-1.1%
    vs base m10500's 3.7%; zero P8-measurable swings);
  - in-domain reward income FLAT over 1000 iters (swing 0.0099->0.0097,
    cadence 0.0034->0.0037) despite a 10-15x achievable differential —
    the policy does not climb scalar reward slopes that require crossing
    the single-support balance basin (contrast: the v52 arm prior moved
    arms 22->66 deg in one run — arms carry no balance risk).

ROUTE DEVIATION (documented, GOAL_RHYTHM.md §5): the pre-registered dose
ladder said swing 0.6 next; the flat-income evidence says the binding
constraint is basin discovery, not reward magnitude (v61e lesson: dose
cannot out-pull an equilibrium). v64c replaces dose escalation with a
KINEMATIC anchor that names the trajectory:

  X1_HIP_PHASE=0.4     hip_phase_reference_prior — per-step dense
                       exp(-|hip_dev - A(v)sin(2pi t/T*(v) + off_i)|/0.15);
                       L/R anti-phased, T*(v) same curve as cadence prior,
                       A(v) = asin(v T*/(4*0.7)) stride geometry. Tracking
                       pays 0.8/step; frozen 0.21; micro-cadence 0.23
                       (dryrun: track 2.00 / frozen 0.52 / inphase 1.12 /
                       2x-period 0.58 of 2.0 max, per foot pair @w=1).
  X1_CADENCE_PRIOR=0.6 kept (touchdown-interval anchor, consistent with
                       the clock lock once tracking succeeds)
  X1_SWING_PRIOR=0.3   kept (anti-drag guard)

Everything else = the v63 recipe frozen. Base = CLEAN m10500 (v64b's
m11499 was drag-corrupted; not inherited). +1000 iters @100 Hz.

PRE-REGISTERED accept: best checkpoint @100 Hz + LPF local battery:
  P8 R1+R2+R3+R4 on walk10 AND walk05; no-regress P7 8/8, arms >= 24,
  drift <= 1.0 m, 5/5 survived, K1 <= 18, robustness >= 48/60.
Success/failure decision rule: if hip_phase income climbs toward
~0.6+/step AND sim2sim swings appear but P8 still fails -> tune sigma or
weight (v64d); if income stays flat again -> the reward-engineering
channel for the rhythm family is EXHAUSTED (three independent mechanisms:
event, airtime scalar, kinematic reference); report recommends structural
routes (demo-phase policy resets / mjlab domain training).
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

# --- v64c rhythm family v2 ----------------------------------------------
os.environ["X1_CADENCE_PRIOR"] = "0.6"
os.environ["X1_SWING_PRIOR"] = "0.3"
os.environ["X1_HIP_PHASE"] = "0.4"
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
