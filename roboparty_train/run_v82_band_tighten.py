#!/usr/bin/env python3
"""v82 launcher: BAND TIGHTENING of the failing families (RECORD_BOOK 10.6).

v81/v81b verdict (TASK_20261026_284/291):
  Ladder m2500 > m3000 > m3499: m2500 = arc optimum (rhythm ALL-PASS,
  0.865 m/s, elbow p95 66.5/66.7 IN-BAND, shoR -28, toe-out 0.3 deg);
  m3499 rhythm REGRESSED (R foot 1.28 Hz / step 0.37 -- the v77/v78
  continuation-drift family) and elbow p95 re-drifted to 97/79.
  Diagnosis: the remaining FORM fails are the "in-band in-training,
  drifts out in MuJoCo" family (hip_yaw training income -0.02/step =
  excess ~1.4 deg vs rollout DC 20 deg; shoR/elbow similar) -- the
  low-load joints carry the Isaac->MuJoCo dynamics gap. More DOSE has
  flat marginal effect; the v81 tree branch applies: TIGHTEN the bands
  so the same pose costs more (shift the whole distribution inward).

v82 single variable = band tightening (doses unchanged):
  X1_SHOR_BAND_DEG   20 -> 12   (shoR DC equilibrium -24..-40 -> target -15)
  X1_ELBOW_HALF_DEG  40 -> 32   (elbow band [20+-32]; median 43-55 -> in)
  X1_HIPYAW_BAND_DEG  8 -> 5    (R hip_yaw DC ~20 -> target <12)
Base: embedded v81-m2500 (arc optimum) + 600 iters (SHORT: the v77/v78/v81b
continuation-drift precedent -- rhythm degrades with long fine-tunes).

PRE-REGISTERED acceptance (unchanged, binding): FORM F1-F4+A1 walk10
ALL-PASS + rhythm all-pass + >=0.85 m/s + 4/4 survival + walk05 >=0.30.
Decision tree:
  all pass           -> DELIVERY CANDIDATE + full battery (P7/collision/
                        robustness/platform)
  arms in, hip_yaw residual only (sim2sim-gap family) -> escalate
                        X1_HIPYAW_W 0.8 -> 1.6 one round; if hip_yaw still
                        out with arms+rhythm in-band -> deliver with the
                        F3 residual documented (feet straight = user defect
                        fixed; hip_yaw is an internal posture DOF)
  rhythm regress     -> soup(m2500, v82-best) midpoint; if still regressed,
                        deliver m2500 family with residual notes
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- THE v82 VARIABLE: band tightening --------------------------------------
os.environ["X1_FORM_PACKAGE"] = "1"
os.environ["X1_TOE_GUARD"] = "0.6"
os.environ["X1_ABD_GUARD"] = "0.8"
os.environ["X1_SOFLAT_W"] = "1.5"
os.environ["X1_ARM_DC_W"] = "0.9"        # dose unchanged
os.environ["X1_HIPYAW_W"] = "0.8"        # dose unchanged
os.environ["X1_SHOR_BAND_DEG"] = "12"    # v82: 20 -> 12
os.environ["X1_ELBOW_HALF_DEG"] = "32"   # v82: 40 -> 32
os.environ["X1_HIPYAW_BAND_DEG"] = "5"   # v82: 8 -> 5

# --- v80 recipe (byte-identical) --------------------------------------------
os.environ["X1_INITIAL_REWARD"] = "1"
os.environ["X1_GAIT_CLOCK"] = "1"
os.environ["X1_FOOT_CLOCK_W"] = "1.0"
os.environ["X1_CONTACT_CLOCK_W"] = "0.5"
os.environ["X1_TRACK_W"] = "2.0"
os.environ["X1_VMATCH_TOL"] = "0.15"

os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_AMP_NUM_STEPS"] = "30"
os.environ["X1_DISC_LINVEL"] = "1"
os.environ["X1_DISC_BUFFER"] = "24"
os.environ["X1_MOTION_DIR"] = "x1_lab_v72"
os.environ["X1_DISC_VMATCH"] = "1"

# --- resume from the embedded v81 m2500 (arc optimum) ----------------------
_EMBEDDED = _here / "checkpoints" / "model_2500_v81.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    raise FileNotFoundError(f"v82 embedded base not found: {_EMBEDDED}")
os.environ["X1_FINE_TUNE_ITERS"] = "600"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
