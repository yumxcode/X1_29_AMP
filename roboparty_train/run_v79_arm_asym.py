#!/usr/bin/env python3
"""v79 launcher: ANTISYMMETRIC-DC ARM GUARD (X1_ARM_ASYM, v54 dose -1.2).

v78 verdict (TASK_20261024_093, 6999/7000): P7h shared lean FIXED
(24.2 -> 0.1-0.7 deg, the lean guard worked exactly as designed) BUT the
arms re-solved into a FROZEN ANTIPHASE-OFFSET pose — shoL mean -53 deg /
shoR +33 deg swinging antiphase (corr -0.9) around offset centers, R
elbow frozen at 80 deg: P7g lean_asym 89.6 -> 121-147 (worse), P7b elbow
88-94 (unchanged; coupled — bent elbows shorten the pendulum for the big
offset swing). Rhythm marginal-regressed (R2 swing 0.66 at the best point
m6999; the familiar continuation drift). Speed held 0.82-0.89.

Forensics: the offset pose is EXACTLY the v28b "one-forward-one-back"
defect family; the existing arm_asym_lean term (paired EMA DIFFERENCE
statistic, v54 dose -1.2) was built for it — the difference statistic
pays |ema(devL) - ema(devR)| which for L-53/R+33 is huge. Expected
coupled effect: pulling the offsets toward neutral shrinks the swing
inertia demand -> the elbow may relax toward the demo bend (42 deg).

v79 = v78 recipe + arm_asym_lean ONLY (chain protocol: one new term per
round, previous additions kept). Base: embedded v78 m6999 (P7h-fixed,
R2-marginal best point) +1000 iters (shorter: guards bite fast — v70
precedent; also limits the swing-drift exposure).

PRE-REGISTERED acceptance:
  primary: P7g lean_asym <= 12 deg in >= 2/4 scenarios
  coupled-watch: P7b elbow p95 moving toward 70; P7a holds
  binding: walk10 rhythm gates ALL PASS with distance >= 9.8 m;
    4/4 survival
Decision tree:
  P7g recovered + rhythm all-pass -> full delivery battery (P7b<=70 ->
    FULL delivery; else elbow residual noted for operator review)
  P7g recovered, rhythm still R2-marginal -> soup v79/m5500 points
  P7g not moving -> dose -2.4 (v53 strength) one round; still nothing ->
    deliver m5500-family with arm-form residual noted (operator review)
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- THE v79 VARIABLE: antisymmetric-DC arm guard ----------------------------
os.environ["X1_ARM_ASYM"] = "1"      # exemption flag (weight read below)
os.environ["X1_ARM_ASYM_W"] = "-1.2"  # informational; the weight env is:

# the config reads X1_ARM_ASYM as the WEIGHT env var; override both roles:
# exemption flag semantics live in x1_amp_env_cfg (!= "0"), so set the dose
# explicitly through the same variable the main body reads.
os.environ["X1_ARM_ASYM"] = "-1.2"   # != "0" -> exempted AND weight -1.2

# --- the v78 recipe, byte-identical ------------------------------------------
os.environ["X1_ARM_GUARDS"] = "1"
os.environ["X1_SY_GUARD"] = "0.5"
os.environ["X1_LEAN_GUARD"] = "0.5"
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

# --- resume from the EMBEDDED v78 m6999 (P7h-fixed best point) ---------------
_EMBEDDED = _here / "checkpoints" / "model_6999_v78.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    raise FileNotFoundError(f"v79 embedded base not found: {_EMBEDDED}")
os.environ["X1_FINE_TUNE_ITERS"] = "1000"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
