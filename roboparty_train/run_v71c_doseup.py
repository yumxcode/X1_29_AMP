#!/usr/bin/env python3
"""v71c launcher: CLOCK DOSE ESCALATION — resume m4500 (v71b ladder point).

Falsification-driven dose step (v52-54 dose-response methodology):

  v71   (w=1.0 / -0.5, 4000 iters from scratch): walk10 cadence 0.81/1.04Hz
        BOTH in R1 band + bilateral lift 39/37mm (L1 PASS) + v-err 0.217 —
        but R2 swing 0.16s < 0.28 and R3 duty 0.88 > 0.75.
  v71b  (+800 iters, same doses, pod-reclaimed at 4801): speed improved
        (v-err 0.188 walk10 / 0.226 walk05) but R2/R3 STRUCTURALLY UNMOVED
        (swing 0.145s, duty 0.86) and walk05 still drag (<25mm lifts).

Diagnosis: the policy matches the bell APEX (~40mm) but briefly — it lands
early and keeps double support (duty 0.86 vs desired 0.61). The -0.5 hinge
prices those wrong-side foot-frames at ~-0.43/step, which the stability
benefit of double support outweighs; the kernel's partial credit pays the
brief apex pass. The equilibrium needs a higher price on wrong-side contact
and a stronger bell pull through the FULL desired swing span.

v71c single variable (dose ladder, everything else byte-identical to v71/b):
  X1_FOOT_CLOCK_W    1.0 -> 1.5
  X1_CONTACT_CLOCK_W 0.5 -> 1.0
Resume: platform-mounted model_4500.pt, +2000 iters (ladder saves every 500;
accept truncation at pod reclaim).

PRE-REGISTERED accept (unchanged, X1_GAIT_SPEC_V71.md §5): walk10+walk05
R1-R6+L1 binding + non-regression (platform AMP P1-P6, 5-scenario survive,
P7, arms, drift, robustness >=48/60).
Decision: pass -> delivery battery (soup candidates on the ladder). R2/R3
still stuck -> sigma_z 0.03->0.02 sharpening round (v71d). Regression on
speed (v-err > 0.35 walk10) -> dose midpoint 1.25/0.75.
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- soup59c validated base (identical to v71/v71b) ------------------------
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_YAW_GUARD"] = "1.0"

# --- disc observability + data side (identical) ----------------------------
os.environ["X1_AMP_NUM_STEPS"] = "30"
os.environ["X1_DISC_LINVEL"] = "1"
os.environ["X1_DISC_BUFFER"] = "24"
os.environ["X1_MOTION_DIR"] = "x1_lab_v67"
os.environ["X1_DISC_VMATCH"] = "1"

# --- v69/v70 delivery-dose posture guards (identical) ----------------------
os.environ["X1_TOE_GUARD"] = "0.6"
os.environ["X1_ABD_GUARD"] = "0.8"
os.environ["X1_SWAY_GUARD"] = "0.4"
os.environ["X1_STANCE_MAX"] = "0.38"
os.environ["X1_SY_GUARD"] = "0.5"
os.environ["X1_LEAN_GUARD"] = "0.5"

# --- the v71c single variable: escalated clock doses -----------------------
os.environ["X1_GAIT_CLOCK"] = "1"
os.environ["X1_FOOT_CLOCK_W"] = "1.5"      # v71/b: 1.0
os.environ["X1_CONTACT_CLOCK_W"] = "1.0"   # v71/b: 0.5
os.environ["X1_HIP_PHASE"] = "0.4"
os.environ["X1_CADENCE_PRIOR"] = "0"
os.environ["X1_SWING_PRIOR"] = "0"

# --- continuation: +2000 iters from the EMBEDDED m4500 ---------------------
# (cross-account resume: the pool account 47 cannot mount account-46's OSS
#  uploads; the checkpoint is committed at roboparty_train/checkpoints/ and
#  pinned explicitly — the v60/v68 embedded-launcher pattern)
_EMBEDDED = _here / "checkpoints" / "model_4500_v71.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    raise FileNotFoundError(f"v71c embedded base not found: {_EMBEDDED}")
os.environ["X1_FINE_TUNE_ITERS"] = "2000"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
