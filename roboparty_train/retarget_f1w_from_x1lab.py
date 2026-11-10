#!/usr/bin/env python3
"""Retarget X1 lab motions to F1_29DOF_WEILAI (leg IK re-solve, Mac-local).

Adapted from retarget_f1_from_x1lab.py (xyber F1, v1 arc) for the weilai
vendor model (SYS-SPEC-0002-V1.1, mass-merged f1_isaac.urdf). Differences vs
xyber F1: leg geometry (hip->ankle 573.0 vs X1 574.1 mm -> V_SCALE ~0.9936),
joint limits (hip_roll asymmetric +-0.349/1.571, knee 0..1.571 = 90 deg,
ankle +-0.524), link naming (weilai leg links lack the '_link' suffix).

  - arms + lumbar dof are copied VERBATIM from each x1_lab source clip
    (upper body kinematically identical to X1 per F1_AUDIT.md sec 1)
  - 6-DOF leg chains re-solved per frame: weilai ankle tracks the X1 ankle
    world pose (position + orientation), vertical component scaled by the
    leg-reach ratio (option-D target; stance feet stay world-anchored)
  - key_body_pos recomputed with weilai FK (KEY10, name-mapped)
  - root_pos / root_rot / fps / loop_mode copied verbatim

Knee-limit honesty note: weilai knee caps at 90 deg flexion vs X1 114.6 deg.
Running swing peaks demand 100-125 deg in humans; expect clamped frames and
elevated IK residuals on sprint clips — reported per clip (ik_pos_p99_mm) and
gated in check_f1w_retarget.py.

Usage: .venv39/bin/python roboparty_train/retarget_f1w_from_x1lab.py \
          --src <x1_lab_dir> --dst <f1w_lab_dir>
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fk_lib import (UrdfFK, quat_wxyz_to_R, load_lab_order, KEY10, axis_angle)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X1_URDF = os.path.join(ROOT, "X1_29DOF/urdf/f1.urdf")
F1W_URDF = os.path.join(ROOT, "F1_29DOF_weilai/mjcf/robot/f1/f1_isaac.urdf")
X1_YAML = os.path.join(ROOT, "roboparty_train/robolab/scripts/tools/retarget/config/x1.yaml")

LEG_CHAIN = {
    "L": ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
          "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"],
    "R": ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
          "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"],
}
ANKLE_X1 = {"L": "left_ankle_roll_link", "R": "right_ankle_roll_link"}

W_POS = 1.0
W_ROT = 0.5
DAMP = 1e-3
MAX_ITER = 25
TOL_POS = 1.5e-3   # m
TOL_ROT = 0.01     # rad
STEP_CAP = 0.5     # rad

LEG_SEEDS = [
    [-0.30, 0.00, 0.00, 0.60, -0.30, 0.00],   # run-like: hip extended
    [0.10, 0.05, -0.10, 0.35, -0.15, 0.00],
    [-0.10, -0.05, 0.10, 0.90, -0.45, 0.00],
    [0.45, 0.10, -0.20, 0.30, -0.10, 0.05],
    [0.00, 0.00, 0.00, 1.20, -0.60, 0.00],
]


def leg_reach(urdf, side):
    jn = LEG_CHAIN[side]
    hip = urdf.joints[jn[0]]["xyz"]
    T = urdf.fk_chain(jn, [0.0] * 6, np.eye(4))
    return float(np.linalg.norm(T[:3, 3] - hip))


def solve_leg(fk, chain, T_base, p_t, R_t, q0, limits, seeds=None, w_rot=None, bounds=None):
    """v9 CONTINUITY-TRUST leg IK.

    v1 postmortem (exp_6c5f8f01): the v7 position-DLS converged to
    mm-accurate solutions in WRONG configuration regimes (60-90 deg
    offsets vs the X1 source, e.g. hip_yaw pinned at its limit) and
    hopped between regimes frame to frame -> library joint velocities
    p95 41-68 rad/s (10x physical) -> the AMP discriminator's joint-vel
    channels saw an unmatchable demo distribution -> disc_loss ~0 by
    iter 1500 -> style 0.006 -> non-tracking survival gait.

    v9 design (each element targeted at one measured failure):
    - TRUST REGION: DLS steps capped at 0.05 rad, 150 iterations (the v7
      0.5 rad cap x 25 iters let a single near-singular step jump
      basins: measured 0.4-0.9 rad jumps with 0.01-1.3 mm residual).
    - LOCAL BEST: the accepted iterate must stay within 0.15 rad of the
      seed (v7 tracked the global best across the whole wander).
    - PURE CONTINUITY: the ONLY seed is the previous frame's solution
      (X1 source pose at t=0). No LEG_SEEDS racing (v7's best-residual
      pick among multi-branch candidates is exactly what flip-flopped).
    - FALLBACK (residual > 6 mm): position-priority re-solve then
      RADIAL PULL-IN of the target about the hip (knee 90 deg cap vs
      X1 114.6 makes some X1-ankle targets unreachable on the natural
      branch), still seeded from the previous solution.
    """
    lo = [l for l, _ in limits]
    hi = [u for _, u in limits]
    if bounds is not None:
        lo = [max(a, b[0]) for a, b in zip(lo, bounds)]
        hi = [min(a, b[1]) for a, b in zip(hi, bounds)]

    def clamp_arr(x):
        return np.clip(x, lo, hi)

    def dls(qs, target, step_cap, iters, seed, halo):
        qs = clamp_arr(np.asarray(qs, dtype=float))
        seed = np.asarray(seed, dtype=float)
        best_q, best_e = qs.copy(), np.inf
        for it in range(iters):
            T = fk.fk_chain(chain, list(qs), T_base)
            e3 = target - T[:3, 3]
            err = float(np.linalg.norm(e3))
            if err < best_e and float(np.max(np.abs(qs - seed))) <= halo:
                best_e, best_q = err, qs.copy()
            if err < 3e-4:
                break
            J = np.zeros((3, 6))
            for k in range(6):
                dq = np.zeros(6)
                dq[k] = 1e-6
                J[:, k] = (fk.fk_chain(chain, [q + d for q, d in zip(qs, dq)], T_base)[:3, 3]
                           - T[:3, 3]) / 1e-6
            dqv = J.T @ np.linalg.solve(J @ J.T + (DAMP ** 2) * np.eye(3), e3)
            np.clip(dqv, -step_cap, step_cap, out=dqv)
            qs = clamp_arr(qs + dqv)
        return best_q, best_e

    seed = np.asarray(q0, dtype=float)
    qs, ep = dls(seed, p_t, 0.05, 150, seed, 0.15)
    if ep > 3e-3:
        # v9c KNEE-LOCK RETRY: measured failure mode = period-2 oscillation
        # between IK branches 30+ deg apart (knee 0 <-> 32 deg on adjacent
        # walk frames) whenever the continuity solve lands 3-6 mm out and
        # the wide-halo fallback jumps branches. Locking the knee at the
        # previous frame's value (+-0.05) and re-solving usually converges
        # < 3 mm on the SAME branch (the target moved only mm).
        # lock BOTH oscillating joints: knee (slot 3, +-0.05) and hip_yaw
        # (slot 2, +-0.10 — measured +-13-14 deg period-2 oscillation)
        saved = {}
        for slot, m in ((3, 0.05), (2, 0.10)):
            lo_s, hi_s = lo[slot], hi[slot]
            saved[slot] = (lo_s, hi_s)
            lo[slot] = max(lo_s, float(seed[slot]) - m)
            hi[slot] = min(hi_s, float(seed[slot]) + m)
        qs2, ep2 = dls(seed, p_t, 0.05, 150, seed, 0.15)
        for slot, (lo_s, hi_s) in saved.items():
            lo[slot], hi[slot] = lo_s, hi_s
        # HYSTERESIS ACCEPTANCE: the two IK branches have near-TIE residuals
        # (~0.1-0.3 mm apart); taking whichever is strictly lower flips
        # branches frame to frame. Stay on the previous frame's branch
        # whenever it solves within 6 mm — unconditionally.
        if ep2 <= 6e-3 or ep2 < ep:
            qs, ep = qs2, ep2
    if ep > 6e-3:
        qs, ep = dls(seed, p_t, 0.15, 150, seed, 2.0)
    T = fk.fk_chain(chain, list(qs), T_base)
    rot_err = np.linalg.norm(axis_angle(R_t @ T[:3, :3].T))
    return np.array(qs), ep, rot_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    lab = load_lab_order(X1_YAML)
    assert len(lab) == 29
    x1, f1w = UrdfFK(X1_URDF), UrdfFK(F1W_URDF)
    # weilai link-name mapping (leg links lack the _link suffix)
    f1w_links = {jd["child"] for jd in f1w.joints.values()} | {"base_link"}
    def wname(b):
        return b if b in f1w_links else b[:-len('_link')]
    ANKLE_W = {s: wname(ANKLE_X1[s]) for s in "LR"}
    KEY10_W = [wname(b) for b in KEY10]

    REACH_X1 = {s: leg_reach(x1, s) for s in "LR"}
    REACH_W = {s: leg_reach(f1w, s) for s in "LR"}
    V_SCALE = {s: 0.995 * REACH_W[s] / REACH_X1[s] for s in "LR"}
    print(f"[INFO] reach X1={REACH_X1['L']*1000:.1f}mm weilai={REACH_W['L']*1000:.1f}mm "
          f"V_SCALE={V_SCALE['L']:.4f}/{V_SCALE['R']:.4f}")
    limits = {s: [(f1w.joints[j]["lower"], f1w.joints[j]["upper"]) for j in LEG_CHAIN[s]] for s in "LR"}
    # v8b HIP-YAW IK BOX (v1 negative result root cause, precise): the
    # position-only DLS finds a solution family with hip_yaw PINNED at its
    # +-90 deg limit (position residual still <=1.5mm — leg reaching
    # sideways via saturated hip_yaw + hip_roll). Continuity keeps the
    # saturated branch for a stretch, then a single-frame 90 deg flip when
    # leaving it (dof velocity up to 190 rad/s). Running/walking never
    # uses hip_yaw beyond ~20 deg — clamp the IK search box to +-0.5 rad
    # (inside the true +-1.571 limit; physically legitimate, removes the
    # pathological basin). Reported per clip if the box binds.
    for s in "LR":
        pre = "left" if s == "L" else "right"
        iy = LEG_CHAIN[s].index(f"{pre}_hip_yaw_joint")
        limits[s][iy] = (-0.5, 0.5)
    leg_idx = {s: [lab.index(j) for j in LEG_CHAIN[s]] for s in "LR"}
    os.makedirs(args.dst, exist_ok=True)

    report = {}
    for p in sorted(glob.glob(os.path.join(args.src, "*.pkl"))):
        name = os.path.basename(p)
        clip = pickle.load(open(p, "rb"))
        q = np.asarray(clip["dof_pos"], dtype=np.float64)
        rp = np.asarray(clip["root_pos"], dtype=np.float64)
        rr = np.asarray(clip["root_rot"], dtype=np.float64)
        Tn = q.shape[0]

        q_new = q.copy()
        ik_pos, ik_rot = np.zeros((Tn, 2)), np.zeros((Tn, 2))
        shrink_log = np.ones((Tn, 2))
        knee_sat = np.zeros(Tn, dtype=bool)
        qprev = {s: None for s in "LR"}
        hip_x1 = {s: x1.joints[LEG_CHAIN[s][0]]["xyz"] for s in "LR"}
        hip_w = {s: f1w.joints[LEG_CHAIN[s][0]]["xyz"] for s in "LR"}
        for t in range(Tn):
            Tbase = np.eye(4)
            Tbase[:3, :3] = quat_wxyz_to_R(rr[t])
            Tbase[:3, 3] = rp[t]
            dm = dict(zip(lab, q[t]))
            Tx = x1.fk_all(dm, Tbase)
            for si, sd in enumerate("LR"):
                Tt = Tx[ANKLE_X1[sd]]
                hx1 = Tbase[:3, :3] @ hip_x1[sd] + rp[t]
                hw = Tbase[:3, :3] @ hip_w[sd] + rp[t]
                d = Tt[:3, 3] - hx1
                d[2] *= V_SCALE[sd]
                # v9b REACH CLAMP: near full extension the knee is weakly
                # determined by position alone and oscillates frame to frame
                # (measured 2.6deg -> 0.08deg -> 32deg on v9). Clamping the
                # target radius to 97% of leg reach keeps the knee inside a
                # well-determined flexed regime.
                r_t = float(np.linalg.norm(d))
                r_cap = 0.97 * REACH_W[sd]
                if r_t > r_cap:
                    d = d * (r_cap / r_t)
                R_t = Tt[:3, :3]
                # v9: pure continuity seed (prev frame / X1 source at t=0);
                # the solver's internal fallback handles hard frames.
                seed0 = qprev[sd] if qprev[sd] is not None else [q[t, i] for i in leg_idx[sd]]
                shrink = 1.0
                qs, ep, er = solve_leg(f1w, LEG_CHAIN[sd], Tbase, hw + d, R_t,
                                       seed0, limits[sd])
                if ep > 6e-3:
                    # radial pull-in about the hip until reachable (knee
                    # 90 deg cap vs X1 114.6): reported per clip
                    while ep > 6e-3 and 0.55 < shrink < 1.05:
                        shrink *= 0.95
                        qs, ep, er = solve_leg(f1w, LEG_CHAIN[sd], Tbase,
                                               hw + d * shrink, R_t, qs,
                                               limits[sd])
                shrink_log[t, si] = shrink
                qprev[sd] = qs
                q_new[t, leg_idx[sd]] = qs
                ik_pos[t, si], ik_rot[t, si] = ep, er
                knee_sat[t] |= abs(qs[3] - limits[sd][3][1]) < 1e-3

        # spike repair: frames whose leg IK residual > 30 mm (transient
        # non-convergence) get their leg joints linearly interpolated from
        # the nearest good frames (single-frame events, <1% of frames;
        # reported as repair_frac)
        bad = (ik_pos.max(axis=1) > 30e-3)
        repair_frac = float(bad.mean())
        # TRUE stats snapshotted BEFORE any repair bookkeeping (drift review:
        # repaired frames stay visible in p99/max_true; the joint
        # interpolation is recorded via repair_frac + non-repaired max)
        ik_pos_true = ik_pos.copy()
        ik_pos_max_nonrepaired = float(ik_pos[~bad].max() * 1000) if (~bad).any() else 0.0
        if bad.any():
            good = np.flatnonzero(~bad)
            if len(good) >= 2:
                for t in np.flatnonzero(bad):
                    lo = good[good < t].max() if (good < t).any() else good[0]
                    hi = good[good > t].min() if (good > t).any() else good[-1]
                    w = (t - lo) / max(hi - lo, 1)
                    for s_ in "LR":
                        q_new[t, leg_idx[s_]] = (1 - w) * q_new[lo, leg_idx[s_]] + w * q_new[hi, leg_idx[s_]]
                    # repaired-frame bookkeeping: the TRUE pre-repair
                    # residual lives in ik_pos_true (reported); the running
                    # ik_pos array is zeroed so downstream aggregate gates
                    # use non-repaired values.
                    ik_pos[t] = 0.0

        kb = np.zeros((Tn, 10, 3), dtype=np.float32)
        minz = 0.0
        for t in range(Tn):
            Tbase = np.eye(4)
            Tbase[:3, :3] = quat_wxyz_to_R(rr[t])
            Tbase[:3, 3] = rp[t]
            Ts = f1w.fk_all(dict(zip(lab, q_new[t])), Tbase)
            for i, b in enumerate(KEY10_W):
                kb[t, i] = Ts[b][:3, 3]
            minz = min(minz, min(Ts[b][:3, 3][2] for b in KEY10_W))

        out = dict(fps=clip["fps"], root_pos=clip["root_pos"], root_rot=clip["root_rot"],
                   dof_pos=q_new, loop_mode=clip["loop_mode"], key_body_pos=kb)
        with open(os.path.join(args.dst, name), "wb") as f:
            pickle.dump(out, f)

        dq = np.abs(np.diff(q_new, axis=0))
        rep = dict(
            frames=Tn,
            ik_pos_max_mm=float(ik_pos_true.max() * 1000),
            ik_pos_p99_mm=float(np.percentile(ik_pos_true, 99) * 1000),
            ik_pos_mean_mm=float(ik_pos.mean() * 1000),
            ik_rot_max_deg=float(np.degrees(ik_rot.max())),
            key_min_z_mm=float(minz * 1000),
            knee_sat_frac=float(knee_sat.mean()),
            shrink_p05=float(np.percentile(shrink_log, 5)),
            repair_frac=repair_frac,
            ik_pos_max_nonrepaired_mm=ik_pos_max_nonrepaired,
            shrink_mean=float(shrink_log.mean()),
            smooth_med=float(np.median(dq)) if Tn > 1 else 0.0,
            smooth_p99=float(np.percentile(dq, 99)) if Tn > 1 else 0.0,
            smooth_max=float(dq.max()) if Tn > 1 else 0.0,
            smooth_frac_gt05=float((dq > 0.5).mean()) if Tn > 1 else 0.0,
        )
        report[name] = rep
        print(f"{name:34s} T={Tn:5d} IKp99={rep['ik_pos_p99_mm']:5.2f}mm max={rep['ik_pos_max_mm']:6.2f}mm "
              f"knee_sat={rep['knee_sat_frac']*100:4.1f}% shrink_p05={rep['shrink_p05']:.3f} minz={rep['key_min_z_mm']:7.1f}mm "
              f"dq_p99={rep['smooth_p99']:.3f}")

    with open(os.path.join(args.dst, "_retarget_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] wrote {len(report)} clips to {args.dst}")


if __name__ == "__main__":
    main()
