#!/usr/bin/env python3
"""v63b launcher: continuation of v63 from m10500 (the good mid-run point).

v63 verdict (TASK_20260921_038, contract rev4 route 7):
  FIXED   arms 37.7-42.0 deg (gate >=24; v62b collapse 7.5/7.8; champion
          34.7/34.1) — the macro-window + linvel + LPF route RESTORED the
          style channel at 100 Hz.
  FIXED   drift at m10500: -0.31/-0.52 m (gate <=1.0) — but UNSTABLE late:
          m10000 1.94 -> m10500 -0.31 -> m10592 +5.42 (yaw DC oscillation,
          +65 deg heading swing). Base m9393+LPF drifts 5.74 for reference.
  PASS    P7 on-pod (first with LPF): antiphase -1.00, coupling +0.95,
          elbow 33.8, survived 11.5 s.
  PASS    platform 12/13 — P4_no_collapse FAIL is the disc-re-init reward
          rebaseline (style income 1.4-1.5 vs v62b's saturated 2.0: the
          NEW disc scores the same gait lower BY DESIGN), not an end-of-
          run spiral (tail reward stable 19.0-19.4, arm_amp_prior steady
          0.018 vs champion 0.0218 / v62b collapse 0.0124).
  OPEN    left-foot shallow swing (max 12-20 mm vs champion 24-34): the
          v61b-v62b family trait; cross-domain (v60 proved zero in-domain
          gradient for swing floors), not v63-caused (m9500+LPF keeps
          24-26 mm swings; v63 policy is LPF-coupled by design).

v63b single-variable stabilizer (from m10500, disc WARM — same obs config,
no re-init, reward baseline continuous):
  X1_YAW_GUARD 1.0 -> 2.5   kill the late yaw oscillation (v57 precedent:
                            the same play at -0.5 -> -1.0 fixed a +5.4
                            deg/s bias; 2.5 sits between v57 and stronger)
  +800 iters (half dose)    checkpoint pick at ~11300
Everything else byte-identical to v63.

PRE-REGISTERED: VALID iff m10500-quality holds (arms >=24 both, drift
<=1.0 on the FINAL checkpoint, platform 13/13 incl. P4 with the warm
baseline, P7 PASS) — the left-swing gate is reported but not blocking
(inherited family trait, cross-domain)."""
import os
import runpy
import sys
from pathlib import Path

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
os.environ["X1_FINE_TUNE_ITERS"] = "800"
os.environ["X1_YAW_GUARD"] = "2.5"      # v63b: 1.0 -> 2.5 (drift stabilizer)

# X1_RESUME_CKPT intentionally UNSET: run_x1_amp_train.py auto-detects the
# highest model_*.pt mounted at repo root by the platform resume mechanism
# (the m9393 embedded fallback in run_v63_100hz.py is bypassed by NOT
# delegating to that launcher).

_HERE = Path(__file__).resolve().parent
sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
