#!/usr/bin/env python3
"""v63 launcher: 100 Hz + disc macro-window + disc root-linvel + action LPF
(user-directed, contract rev4). The 7th route against the 100 Hz style
collapse — the first one aimed at TRAINING DYNAMICS OBSERVABILITY rather
than demo data (rev3 verdict: 200 fps demos do NOT fix it; root cause
confirmed in the training dynamics).

Three coupled levers (user suggestion + AMP-paper-backed):
  X1_AMP_NUM_STEPS=60   disc/animation window 60x10 ms = 600 ms — >= 1 gait
                        cycle (~1.1 s at 1.0 m/s walking, so >= half-cycle
                        with full stride-geometry visibility; the 60 ms
                        champion window sees only leg-swing fragments where
                        micro-stepping and walking are locally identical).
  X1_DISC_STRIDE=2      disc input = 30 frames at 20 ms spacing (3720 dims
                        with linvel on) — the champion's temporal cadence,
                        native 200 fps keys (exact fetch, no interpolation),
                        window span x10. Buffer 24 steps = 2.9 GB (v62b was
                        2.4 GB at 6x121).
  X1_DISC_LINVEL=1      root LINEAR velocity joins the disc obs (121->124).
                        The AMP paper ablation marks velocity features
                        REQUIRED; both canonical implementations include
                        local root lin vel. This repo had it commented out
                        — without it the disc window cannot see displacement
                        rate AT ALL, which is the observation-space basis of
                        micro-gait style income 2.7x the champion's (v61f).
  X1_ACT_LPF_HZ=10      action low-pass (2 cascaded sections, alpha =
  X1_ACT_LPF_ORDER=2    1-exp(-2*pi*10*0.01) = 0.4665/step). Policy action
                        bandwidth capped at the demo band; the 100 Hz PD
                        servo keeps its correction bandwidth. Mirrored
                        flag-for-flag in sim2sim (--action-lpf) and the P7
                        gate / battery / robustness sweep.

Everything else = the v59c recipe on the v56b m9393 base (identical to
v62b): regime 3, guards, disc-vel, yaw 1.0, prior 0.06, lerp 0.68,
+1200 fine-tune iters, PCHIP 200 fps demos.

PRE-REGISTERED falsification (contract rev4):
  Fix VALID  iff platform 13/13 + T1/T2/T3 AND P7 PASS (with the LPF on)
              AND sim2sim walk10 arms >= 24 deg (both sides) AND drift
              <= 1.0 m AND stride events >= 40 AND swing clearance gates
              AND K1 <= 18 deg AND robust >= 48/60 AND kernels 13/13.
  Otherwise  the observability route (window+linvel) joins the falsified
              list; if arms hold but kernels/velocity tracking degrade,
              the linvel term (command-agnostic style pressure toward
              demo-like speeds) is the suspect — single-variable drop
              X1_DISC_LINVEL first.
  Early training-side signal: arm_amp_prior income — champion 0.0218,
              collapsed runs 0.012-0.013; >= 0.02 by iter ~400 predicts
              arms >= 24 deg (v51-v59 calibration)."""
import os
import runpy
import sys
from pathlib import Path

os.environ["X1_CONTROL_HZ"] = "100"
os.environ["X1_MOTION_DIR"] = "x1_lab_v32_200"
os.environ["X1_AMP_NUM_STEPS"] = "60"   # 600 ms disc/animation window
os.environ["X1_DISC_STRIDE"] = "2"      # 30 frames @ 20 ms (champion cadence)
os.environ["X1_DISC_BUFFER"] = "24"     # CircularBuffer depth (= rollout len)
os.environ["X1_DISC_LINVEL"] = "1"      # root lin vel into disc obs (124 dim)
os.environ["X1_ACT_LPF_HZ"] = "10"      # action low-pass cutoff
os.environ["X1_ACT_LPF_ORDER"] = "2"    # cascaded sections (12 dB/oct)
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_FINE_TUNE_ITERS"] = "1200"
os.environ["X1_YAW_GUARD"] = "1.0"

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_9393_v56b.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(_HERE.parent.glob("model_9393_v56b*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v56b base checkpoint not found")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
