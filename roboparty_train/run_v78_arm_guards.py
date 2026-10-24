#!/usr/bin/env python3
"""v78 launcher: ARM-POSTURE GUARDS (X1_ARM_GUARDS=1, v70b delivery doses).

v77r verdict (TASK_20261024_073, 6999/7000): H5 FALSIFIED — the arm
smoothing split did NOT recover arm form; it made posture WORSE (P7g
lean_asym 24.8 -> 60-63 deg, R arm +62-79 deg forward lean: freed arm
motion without posture anchoring drifts into a forward-lean solution;
P7b elbow 86-91 still >70; P7a anti partially recovered -0.89). Rhythm
REGRESSED on all three checkpoints (R2 swing 0.66-0.68 > 0.60, R3 duty
0.39-0.44 < 0.45 — the familiar continuation drift). The v77 branch is
rejected wholesale; X1_ARM_SMOOTH is reverted.

Per the pre-registered tree ("arm unchanged -> H5 falsified -> v78:
sy/lean guards directly"): the guards ARE the P7g/P7h statistics
(shoulder_yaw_fold = |sy| band 15 deg; arm_shared_lean = shared forward
lean band 6 deg — the v70b package that FIXED the same defects on the
v69/v70 lineage). v78 = v76 recipe (the delivery candidate m5500) +
X1_SY_GUARD=0.5 + X1_LEAN_GUARD=0.5 ONLY. Base: embedded v76 m5500 +1500.

Honest prior (recorded): v72 evidence shows guards alone from scratch
(4-6k iters) did NOT pass P7b elbow in the clock era; but v78 resumes
from the all-pass m5500 (not scratch) and targets P7g/P7h primarily.
If P7b remains >70 after guards, the elbow dimension needs its own arc
(disc demo elbow style / longer training / accept-with-note per the
operator's quality-gate review).

PRE-REGISTERED acceptance:
  primary: P7g lean_asym <= 12 deg AND P7h lean_shared <= 12 deg in
    >= 2/4 scenarios
  secondary (partial credit): P7b elbow p95 moves toward 70; P7a holds
  binding: walk10 rhythm gates ALL PASS with distance >= 9.8 m;
    4/4 survival
Decision tree:
  P7g/P7h recovered + rhythm held  -> delivery battery on the best
    point; if P7b also <= 70 -> FULL delivery; else deliver with the
    elbow noted as residual (operator quality-gate review)
  guards work but rhythm regressed -> mid-ladder point soup (v69 flow)
  no arm change                    -> guard-dose escalation 1.0 (v70
    precedent) OR accept m5500 with arm-form noted
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- THE v78 VARIABLE: v70b arm-posture guard package ------------------------
os.environ["X1_ARM_GUARDS"] = "1"
os.environ["X1_SY_GUARD"] = "0.5"
os.environ["X1_LEAN_GUARD"] = "0.5"

# --- the v76 delivery recipe, byte-identical ---------------------------------
os.environ["X1_TRACK_W"] = "2.0"
os.environ["X1_VMATCH_TOL"] = "0.15"
os.environ["X1_INITIAL_REWARD"] = "1"
os.environ["X1_GAIT_CLOCK"] = "1"
os.environ["X1_FOOT_CLOCK_W"] = "1.0"
os.environ["X1_CONTACT_CLOCK_W"] = "0.5"

os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_AMP_NUM_STEPS"] = "30"
os.environ["X1_DISC_LINVEL"] = "1"
os.environ["X1_DISC_BUFFER"] = "24"
os.environ["X1_MOTION_DIR"] = "x1_lab_v72"
os.environ["X1_DISC_VMATCH"] = "1"

# --- resume from the EMBEDDED v76 m5500 (delivery candidate) -----------------
_EMBEDDED = _here / "checkpoints" / "model_5500_v76.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    raise FileNotFoundError(f"v78 embedded base not found: {_EMBEDDED}")
os.environ["X1_FINE_TUNE_ITERS"] = "1500"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
