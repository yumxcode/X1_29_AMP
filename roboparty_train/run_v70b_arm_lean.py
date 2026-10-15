#!/usr/bin/env python3
"""v70b launcher: shared arm-lean fix continuation from m15496.

P7h regression found on the v70 ladder (local sim2sim battery, P7 gate
= acceptance/check_policy_gait.py): the sy band fixed the forearm fold
(syR -64.6 -> ~0 deg, torso penetration -61 -> +16-18 mm) but the policy
re-solved the shoulder/elbow redundancy by adducting the right shoulder
(shoulder_roll -16 -> -33 deg, wrist positions preserved — disc-blind),
leaving both upper arms leaned forward:

    m13497 7.1 / m13500 8.9 deg shared lean (PASS, gate <= 12)
    m14000 18.0 / m14500 18.9 / m15000 19.7 / m15496 19.9 deg (FAIL,
    stable across +1500 iters = equilibrium adaptation, not transient)

Recipe = v70 (soup59c cell + v69 guards + X1_SY_GUARD=0.5) + ONE new
guard:
  X1_LEAN_GUARD = 0.5  relu(|mean(phiL,phiR)| - 6 deg), phi = the EXACT
                    P7h statistic (upper-arm forward angle, FK on body
                    positions, base frame). Walking demos lean 0.1-3.7
                    deg -> pay 0; m15496 pays 0.243 rad -> -0.12/step.

Short continuation (+1000 iters from m15496): the needed adaptation is
smaller than v70's (un-adduct one shoulder while keeping sy ~0; demos
prove sy~0 & lean<=4 coexist).
"""
import os
import runpy
import sys
from pathlib import Path

# v70 recipe (identical) — unchanged
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_YAW_GUARD"] = "1.0"
os.environ["X1_TOE_GUARD"] = "0.6"
os.environ["X1_ABD_GUARD"] = "0.8"
os.environ["X1_SWAY_GUARD"] = "0.4"
os.environ["X1_STANCE_MAX"] = "0.38"
os.environ["X1_SY_GUARD"] = "0.5"
# v70b shared arm-lean guard (the ONLY new experimental variable)
os.environ["X1_LEAN_GUARD"] = "0.5"
os.environ["X1_FINE_TUNE_ITERS"] = "1000"
os.environ.setdefault("X1_CONTROL_HZ", "50")  # contract: 50 Hz

_HERE = Path(__file__).resolve().parent
_CANDIDATES = [
    _HERE / "checkpoints" / "model_15496_v70.pt",    # v70 task end
    _HERE / "checkpoints" / "model_15496.pt",
]
for _c in _CANDIDATES:
    if _c.is_file():
        os.environ["X1_RESUME_CKPT"] = str(_c)
        break
else:
    raise FileNotFoundError(
        "v70b base checkpoint not found (roboparty_train/checkpoints/model_15496_v70.pt)")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
