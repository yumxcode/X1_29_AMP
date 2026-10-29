#!/usr/bin/env python3
"""v91 launcher: ANALYTIC STEP-LENGTH DRIVE (RECORD_BOOK 12.12,
adjudication lever B2; audit directive 2026-10-29).

The seven-lever chain closed at 0.895 all-pass: cadence tracks the clock
exactly (1.05-1.06 Hz) but step stays style-pinned at 0.446-0.453 m
(speed = 1.86 x f x step = 0.89). B2 sets the step target ANALYTICALLY:
at each foot touchdown, exp kernel on |step - v_cmd/(1.86 x f_clk)|;
L*(1.0) = 0.511 m (R4 band [0.45,1.05]) -> predicted speed ~1.0.

v91 single variable = X1_STEP_DRIVE_W 2.0 on the v87 recipe, base
embedded j3000 (the 0.895 all-pass delivery) +1500.

PRE-REGISTERED acceptance (binding): walk10 >= 0.95 AND F5 AND rhythm
R1-R4+L1 all-pass AND survival 4/4 AND walk05 >= 0.30.
Decision tree: >=1.0 all-pass -> CONTRACT MET + battery + delivery;
0.90-0.95 all-pass -> step drive works but dose short: ladder/soup +
second dose 3.0; rhythm breaks (swing rushed at longer steps) -> sigma
0.05->0.08 / dose 1.0 retry; F5 widens -> stance hinge co-dose 3.0->4.0;
stuck ~0.89 -> B2 falsified, fire B1 staged curriculum.

The speed-arc falsification chain (v83b-v86b2, 8 rounds): all-pass family
ceiling ~0.89 via the CONTINUATION lineage. The v74 precedent: from-scratch
training under a new mechanism package BEAT the accumulated fine-tune
stack. The final recipe (clock 1.61/0.66 + track 2.25/std 0.25 + STANCE
pkg + FORM pkg + ROBUST=1) has NEVER trained from scratch. Single
variable = initialization (scratch vs continuation).

PRE-REGISTERED acceptance (binding): walk10 speed >= 0.95 AND F5 AND
rhythm R1-R4+L1 all-pass AND survival 4/4 AND walk05 >= 0.30.
Read-out: >=1.0 all-pass -> CONTRACT MET; 0.89-0.95 all-pass -> ties/beats
h7499, soup the two lineages; <0.89 or gates broken -> continuation
lineage stands (h7499), basin-level ceiling confirmed, wrap.

v85 verdict (TASK_20261028_002, std 0.25 from v84-f5999 +1000): speed
0.859-0.863 (marginal +0.01 over v84), R4 still 14mm short (step 0.436),
g6998 drifted (R2/R3). TEMPERATURE LEVER FALSIFIED as the binder.

Decisive new evidence: the in-stream style centroid at cmd 1.0 is ALREADY
~0.99 m/s (07_01 0.89 w1.2 / 38_02 0.90 w1.2 / 08_03 1.10 w1.2 / 08_01
1.14 w0.5) yet the policy sits at 0.86 — 0.12 BELOW the centroid. The
style anchor is NOT the binder (data lever deprioritized); the CAPACITY /
risk-aversion hypothesis is: ROBUST=3 randomization (push events, mass/
friction, latency injection) makes high-speed recovery expensive, so the
policy rationally caps cadence x step.

v86 single variable = X1_ROBUST_TRAIN 3 -> 1 (DIAGNOSTIC, NOT a delivery
round — the robustness battery is expected to drop; this round LOCATES
the binder). Base: embedded v85-g6500 +1000, else byte-identical.

READ-OUT (pre-registered):
  speed >= 0.95 with F5/rhythm held  -> capacity CONFIRMED as the binder;
       next arc = robustness-speed co-design (robust ramp AFTER speed, or
       robust-on-legs-only); operator adjudication on the trade
  speed 0.86-0.95                    -> partial; combine with data lever
  speed ~0.86 unchanged              -> neither style nor randomization:
       the binder is the controller/dynamics itself (longer training /
       architecture / sim2sim actuation) — operator adjudication, wrap

v84 verdict (TASK_20261027_267, clock 1.61/0.66 from d4500 +1500): the
policy FOLLOWS the new clock exactly (cadence 1.04-1.06 Hz, swing 0.38 s,
F5 289/332 best-ever) but SPEED IS INVARIANT ~0.85 (0.91Hz x 0.51m = 0.86
vs 1.05Hz x 0.43m = 0.85; step shrank below the R4 floor 0.433 < 0.45) --
the f x step product self-adjusts; the equilibrium is pinned by the
task-style balance, not the clock.

v85 single variable = X1_TRACK_STD 0.5 -> 0.25 (kernel temperature of
track_lin_vel_xy_exp; shape lever, distinct from the v76-falsified weight
amplitude scaling: gradient at the |err|=0.15 operating point rises ~1.5x
and the equilibrium shifts toward the command). Base: embedded v84-f5999
+1000, everything else byte-identical to v84 (incl. the 1.61/0.66 clock).

PRE-REGISTERED acceptance (binding): walk10 speed >= 0.95 AND F5 AND
rhythm R1-R4+L1 all-pass (R4 needs step >= 0.45 at the faster clock) AND
survival 4/4 AND walk05 >= 0.30.
Decision tree: >=1.0 all-pass -> contract MET + full battery; 0.95-1.0
-> soup with v84 genes / second std dose 0.2; rhythm/F5 break -> the
steep kernel over-constrains: revert to std 0.5 + take the 82_xx
data-reanchor lever; speed stuck <0.95 -> the equilibrium is style-side:
82_xx reanchor (long-step 1.0 m/s demos into PRIME) is the remaining
registered lever; operator adjudication if that also fails.

v83 arc wrap (delivery soup_ed75 0.861 m/s all-pass): the 1.0 m/s gap is
the clock identity — T*(v)=clamp(1.46-0.36v,0.70,1.75) pins cadence at
0.909 Hz/foot at cmd 1.0, and the style mixture pins step ~0.50 m, so
speed ~= 1.86 x f x step = 0.86. Every >0.85 point in the arc reached
speed only by breaking the clock lock (micro-cadence 1.33-1.58 Hz,
R1/R4-rejected) — five rounds of dose-response closure.

v84 single variable = THE CLOCK SCHEDULE SLOPE (the exogenous rhythm
command itself; v71's mechanism, first schedule change since):
  X1_CLOCK_T_A 1.46 -> 1.61, X1_CLOCK_T_B 0.36 -> 0.66
  T*(1.0) = 0.95 s = 1.053 Hz/foot (R1 ok), swing (1-d)T = 0.37 s (R2 ok)
  T*(0.5) = 1.28 s UNCHANGED (walk05 untouched — its drag is a separate
  registered lever, observed this round)
  predicted speed at 0.50 m step = 1.86 x 1.053 x 0.50 ~= 0.98 m/s
The sim2sim twin reads the SAME env vars (mujoco_rollout.py v84 mirror).

Base: embedded v83d-d4500 (rhythm+robustness genes: all-pass 0.833,
F5 297/375, robust 50/60) +1500, everything else byte-identical to the
v83d recipe (track 2.25 + STANCE package + FORM package).

PRE-REGISTERED acceptance (binding):
  walk10 speed >= 0.95 m/s AND F5 non-regression (p50<=0.33/p90<=0.38)
  AND rhythm R1-R4+L1 all-pass AND survival 4/4 AND walk05 >= 0.30 m/s.
Decision tree:
  >=1.0 all-pass     -> contract speed MET: delivery + full battery + soup
                        fine-tune if marginal (soup with soup_ed75 genes)
  0.95-1.0 all-pass -> second dose T*(1.0)=0.90 (1.11 Hz) one round
  speed in, rhythm/F5 break
                     -> the 1.05 Hz clock exceeds the learned swing
                        mechanics: ladder midpoints, then report the
                        f-vs-gates frontier (the step must lengthen
                        instead: 82_xx re-anchor data lever)
  speed <0.95       -> policy does not follow the faster clock (phase
                        lock too rigid from continuation): scratch 4000
                        with the v84 schedule, or operator adjudication
"""
import os
import runpy
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# --- v86 VARIABLE: robustness randomization ---------------------------------
os.environ["X1_ROBUST_TRAIN"] = "1"

# --- v84/v85 recipe (kept) ---------------------------------------------------
os.environ["X1_CLOCK_T_A"] = "1.61"
os.environ["X1_CLOCK_T_B"] = "0.66"

# --- THE v85 VARIABLE: track kernel temperature -----------------------------
os.environ["X1_TRACK_STD"] = "0.25"

# --- v83d recipe (byte-identical) -------------------------------------------
os.environ["X1_TRACK_W"] = "2.25"
os.environ["X1_STANCE_MAX"] = "0.30"
os.environ["X1_STANCE_HINGE_W"] = "3.0"
os.environ["X1_STANCE_HINGE_MAX"] = "0.32"
os.environ["X1_FORM_PACKAGE"] = "1"
os.environ["X1_TOE_GUARD"] = "0.6"
os.environ["X1_ABD_GUARD"] = "0.8"
os.environ["X1_SOFLAT_W"] = "1.5"
os.environ["X1_ARM_DC_W"] = "0.9"
os.environ["X1_HIPYAW_W"] = "0.8"
os.environ["X1_INITIAL_REWARD"] = "1"
os.environ["X1_GAIT_CLOCK"] = "1"
os.environ["X1_FOOT_CLOCK_W"] = "1.0"
os.environ["X1_CONTACT_CLOCK_W"] = "0.5"
os.environ["X1_VMATCH_TOL"] = "0.15"
os.environ["X1_DISC_VEL"] = "1"
os.environ["X1_AMP_NUM_STEPS"] = "30"
os.environ["X1_DISC_LINVEL"] = "1"
os.environ["X1_DISC_BUFFER"] = "24"
os.environ["X1_MOTION_DIR"] = "x1_lab_v72"
os.environ["X1_DISC_VMATCH"] = "1"

os.environ["X1_RESUME_CKPT"] = str(_here / "checkpoints" / "model_4499_v91b.pt")
os.environ["X1_FINE_TUNE_ITERS"] = "1000"
os.environ["X1_STEP_DRIVE_W"] = "3.0"
os.environ.setdefault("X1_CONTROL_HZ", "50")

sys.argv = [str(_here / "run_x1_amp_train.py")] + sys.argv[1:]
runpy.run_path(str(_here / "run_x1_amp_train.py"), run_name="__main__")
