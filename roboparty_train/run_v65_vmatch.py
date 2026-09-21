#!/usr/bin/env python3
"""v65 launcher: STRUCTURAL route — speed-matched demo sampling at 50 Hz.

Audit r1 + course-correction directives (2026-09-22): the reward channel
for the rhythm family is falsified (v64/v64b/v64c); this run executes the
pre-registered structural route 1, revised on PAIRING FORENSICS:

  CircularBuffer.mini_batch_generator draws INDEPENDENT randperms for the
  policy and demo buffers — per-window phase pairing between policy and
  demo windows DOES NOT EXIST, so literal phase alignment is unimplementable
  at the batch level. The implementable data-side repair is demo
  COMPOSITION: the disc's demo stream at random fetch is dominated by the
  near-IN-PLACE treadmill family (14/22 clips, world v ~= 0) whose
  horizontal key-body kinematics resemble micro-stepping — the direct
  explanation for the v61-forensics equilibrium (disc prefers micro-gait,
  2.7x style income). 103_07 is a 60Hz source stored at 120fps = 2x
  playback whose 218 spm cadence literally demonstrates micro-gait rhythm.

  X1_DISC_VMATCH=1 (THE variable): MotionDataTerm.sample_times_speed_gated
  — demo fetch times land only in segments whose LOCAL speed matches each
  env's COMMANDED speed (tol 0.35 m/s, window margin kept; uniform
  fallback when a clip has no in-band frames). Walking commands fetch
  overground human-cadence segments (0026/36_01/36_11's walking parts);
  stand commands still see the in-place family. 103_07(+mirror) weight 0.

Control rate per audit directive 3: TRAIN AT 50 Hz (the v61 six-route
falsification was specific to 100 Hz fine-tuning; the 50 Hz disc
equilibrium historically held healthy style gradient), evaluate by
100 Hz DIRECT-DRIVE (proven channel: m9500@100Hz 13/14 gates). Base =
m9500 (v59c champion parent; duty 0.70 already passes P8 R3, cadence
1.85 Hz is the family's closest to the 1.20 gate).

Disc package (v63-validated observability trio, time-span preserved):
  X1_AMP_NUM_STEPS=30   600 ms macro-window at 50 Hz (30 x 20 ms)
  X1_DISC_LINVEL=1      root linvel channel (v63)
  obs shape changes vs m9500's 3-step disc -> AMPRunner disc re-init
  (v38-proven path; fresh disc catches up in a few hundred iters while
  the strong policy anchors the rollout distribution)

Everything else = the v59c recipe frozen (regime 3, yaw 1.0, guards 1,
disc_vel 1, dataset x1_lab_v32 at native 120 fps, NO action LPF — m9500
never had one and this stays 50 Hz).

+1000 iters. PRE-REGISTERED accept (best ckpt, 100 Hz direct-drive local
battery, NO --action-lpf): P8 R1+R2+R3+R4 on walk10 AND walk05; no-regress
P7 8/8, arms >= 24 deg, drift <= 1.0 m, 5/5 survived, K1 <= 18 deg (the
v64c 19.7 watch item), walk05 speed error <= 0.15 m/s, robustness >= 48.
Decision: P8 pass -> done; partial (cadence moves > 30% toward the band)
-> v65b tol 0.25 + overground weight boost; no movement -> route 2 (mjlab)
escalation per audit directive 5.
"""
import os
import runpy
import sys
from pathlib import Path

# --- v59c recipe (frozen, 50 Hz native) ---------------------------------
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_YAW_GUARD"] = "1.0"           # v59c native dose
os.environ["X1_FINE_TUNE_ITERS"] = "1000"

# --- disc observability package (v63-validated, 600 ms at 50 Hz) --------
os.environ["X1_AMP_NUM_STEPS"] = "30"
os.environ["X1_DISC_LINVEL"] = "1"

# --- v65 THE variable ----------------------------------------------------
os.environ["X1_DISC_VMATCH"] = "1"

_HERE = Path(__file__).resolve().parent
_EMBEDDED = _HERE / "checkpoints" / "model_9500_v59c.pt"
if _EMBEDDED.is_file():
    os.environ["X1_RESUME_CKPT"] = str(_EMBEDDED)
else:
    mounted = sorted(_HERE.parent.glob("model_9500_v59c*.pt"))
    if mounted:
        os.environ["X1_RESUME_CKPT"] = str(mounted[0])
    else:
        raise FileNotFoundError("v59c m9500 base checkpoint not found")

sys.argv = [str(_HERE / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_HERE / "run_x1_amp_train.py"), run_name="__main__")
