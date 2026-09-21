#!/usr/bin/env python3
"""v66 launcher: STRUCTURAL route 3 — 50 Hz FROM-SCRATCH + full rhythm package.

Escalation trail (audit r1 directive 5, pre-registered order):
  v64/v64b/v64c  reward channel (3 mechanisms) — EXHAUSTED (flat income,
                 no sim2sim movement; V64_REPORT)
  v65/v65b       demo-composition disc fix on the ENTRENCHED m9500 — no
                 movement (on-pod cadence 1.85->1.89 Hz = noise; sim2sim
                 10499 drag / 10000 2.86 Hz). Also exposed the composition
                 leak fixed in v65.1: motion ASSIGNMENT is now speed-gated
                 too (sample_motions_speed_gated; in-place family 77% of
                 weight mass no longer serves walking commands).
  route 2 mjlab  infra-blocked (probe v3/v4/v5 exhausted: pip egress OK,
                 jax+mjlab un-installable within pod budget)
  -> THIS RUN   from-scratch removes the entrenched-basin variable entirely
                 (the one untested cell in the reward-channel falsification;
                 its applicability note says "50 Hz from-scratch untested").

Package (all levers individually dry-run-verified):
  RHYTHM TRIO from iteration 0 (the policy learns the rhythm as it learns
  to walk — no basin to escape):
    X1_CADENCE_PRIOR=0.6  touchdown-interval exp kernel, T*(v) anchor
    X1_SWING_PRIOR=0.3    dense swing-duration kernel (anti-drag/anti-micro)
    X1_HIP_PHASE=0.4      sinusoidal hip kinematic anchor (L/R anti-phase)
  FULL VMATCH (v65.1): X1_DISC_VMATCH=1 — disc demo stream at walking
    commands = overground human-cadence segments ONLY (motion assignment
    AND time sampling both speed-gated); 103_07 2x-playback dropped.
  DISC package (v63/v65b-validated at 50 Hz): 600 ms macro window
    (X1_AMP_NUM_STEPS=30) + root linvel + X1_DISC_BUFFER=24.
  v59c recipe otherwise frozen (regime 3, yaw 1.0, guards 1, disc_vel 1,
  dataset x1_lab_v32 native 120 fps, NO action LPF).

  X1_RESUME_CKPT deliberately UNSET and repo root verified clean of
  model_*.pt (run_x1_amp_train auto-detects a mounted resume checkpoint —
  from-scratch must not pick one up). 4000 iters (~2h).

PRE-REGISTERED accept (best ckpt, 100 Hz direct-drive local battery, no
--action-lpf): P8 R1+R2+R3+R4 walk10 AND walk05; no-regress P7 8/8, arms
>= 24 deg, drift <= 1.0 m, 5/5 survived, K1 <= 18, walk05 speed error
<= 0.15 m/s, robustness >= 48/60, platform verdict 13/13 (from-scratch
runs historically reach 12-13/13).
Decision: P8 pass -> CONTRACT GOAL MET (report + ship). Partial rhythm
movement but gates fail -> dose adjust (v66b trio x1.5). No movement from
SCRATCH with the full package -> rhythm under this AMP stack is a
structural limit; terminal report with the complete evidence chain.
"""
import os
import runpy
import sys
from pathlib import Path

# --- v59c recipe (frozen) -----------------------------------------------
os.environ["X1_ROBUST_TRAIN"] = "3"
os.environ["X1_FORM_GUARDS"] = "1"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_YAW_GUARD"] = "1.0"

# --- disc observability package (600 ms at 50 Hz) ------------------------
os.environ["X1_AMP_NUM_STEPS"] = "30"
os.environ["X1_DISC_LINVEL"] = "1"
os.environ["X1_DISC_BUFFER"] = "24"

# --- full vmatch (motion assignment + time sampling) --------------------
os.environ["X1_DISC_VMATCH"] = "1"

# --- rhythm trio from iteration 0 ----------------------------------------
os.environ["X1_CADENCE_PRIOR"] = "0.6"
os.environ["X1_SWING_PRIOR"] = "0.3"
os.environ["X1_HIP_PHASE"] = "0.4"

# --- from-scratch: 4000 iters, NO resume ---------------------------------
os.environ.pop("X1_RESUME_CKPT", None)
os.environ.pop("X1_FINE_TUNE_ITERS", None)

_here = Path(__file__).resolve().parent
sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
