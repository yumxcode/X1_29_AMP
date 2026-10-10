#!/usr/bin/env python3
"""Retarget X1 lab motions to F1_29DOF (leg IK re-solve, Mac-local, no mujoco).

Contract: F1 vs X1 differ ONLY in the lower-limb 12 DOF (redesigned orthogonal
hip, perfect-mirrored right leg, shorter leg 567.6 vs 584.7 mm, wider hip
290.4 vs 267.9 mm). Upper body (3 lumbar + 14 arm joints) is kinematically
identical, therefore:

  - arms + lumbar dof are copied VERBATIM from each x1_lab_v67 clip
  - the 6-DOF leg chains are re-solved per frame so the F1 ankle_roll_link
    tracks the X1 ankle_roll_link world pose (position + orientation)
  - key_body_pos is recomputed with F1 FK (10 bodies, same names/order)
  - root_pos / root_rot / fps / loop_mode copied verbatim

Acceptance mirrors acceptance/RETARGET_ACCEPTANCE.md (X1 standard):
  A schema | B joint limits (F1 urdf) | C root trajectory (== source)
  D smoothness | E ground clearance (F1 FK) | + IK residual report

Usage: .venv-mac/bin/python roboparty_train/retarget_f1_from_x1lab.py
"""
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fk_lib import (UrdfFK, quat_wxyz_to_R, load_lab_order, KEY10, axis_angle)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "roboparty_train/robolab/data/motions/x1_lab_v67")
DST = os.path.join(ROOT, "roboparty_train/robolab/data/motions/f1_lab_v67")
X1_URDF = os.path.join(ROOT, "X1_29DOF/urdf/f1.urdf")
F1_URDF = os.path.join(ROOT, "F1_29DOF/urdf/f1_29dof.urdf")
X1_YAML = os.path.join(ROOT, "roboparty_train/robolab/scripts/tools/retarget/config/x1.yaml")

LEG_CHAIN = {
    "L": ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
          "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"],
    "R": ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
          "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"],
}
ANKLE = {"L": "left_ankle_roll_link", "R": "right_ankle_roll_link"}

W_POS = 1.0
W_ROT = 0.5
DAMP = 1e-3
MAX_ITER = 25
TOL_POS = 1.5e-3   # m
TOL_ROT = 0.01     # rad
STEP_CAP = 0.5     # rad

# generic multi-start seeds for a humanoid leg (hip p/r/y, knee, ankle p/r)
LEG_SEEDS = [
    [0.30, 0.00, 0.00, 0.60, -0.30, 0.00],
    [0.10, 0.05, -0.10, 0.35, -0.15, 0.00],
    [-0.10, -0.05, 0.10, 0.90, -0.45, 0.00],
    [0.45, 0.10, -0.20, 0.30, -0.10, 0.05],
    [0.00, 0.00, 0.00, 1.20, -0.60, 0.00],
]


def leg_reach(urdf, side):
    """Full-extension |ankle - hip_pitch_center| at q=0 (upper chain zero)."""
    jn = LEG_CHAIN[side]
    hip = np.array([0.0, 0.0, 0.0])
    # hip_pitch joint center in base frame = its origin xyz (parents are all
    # zero-transform here: hip_pitch's parent IS base_link)
    hip = urdf.joints[jn[0]]["xyz"]
    T = urdf.fk_chain(jn, [0.0] * 6, np.eye(4))
    return float(np.linalg.norm(T[:3, 3] - hip))


# vertical scale = F1/X1 leg reach, with 0.5% margin so the boundary case
# (dx=dy=0, X1 knee fully straight) lands at knee ~0.05 rad, not the limit.
REACH_X1 = {s: leg_reach(UrdfFK(os.path.join(ROOT, "X1_29DOF/urdf/f1.urdf")), s) for s in "LR"}
REACH_F1 = {s: leg_reach(UrdfFK(F1_URDF), s) for s in "LR"}
V_SCALE = {s: 0.995 * REACH_F1[s] / REACH_X1[s] for s in "LR"}


def scale_target(p_t, p_hip_x1, p_hip_f1, s):
    """Option-D leg target: keep dx/dy (world-anchored stance, no foot slide,
    identical stride), scale dz by the leg-reach ratio (F1 pelvis sits lower
    on its shorter legs — physically correct). Guarantees reachability:
    |target - hip_f1|^2 = |d|^2 - (1-s^2) dz^2 <= s^2 REACH_X1^2 = REACH_F1^2.
    """
    d = p_t - p_hip_x1
    d[2] *= s
    return p_hip_f1 + d


def solve_leg(fk, chain, T_base, p_t, R_t, q0, limits, seeds=None):
    """Multi-start damped-least-squares IK on a 6-DOF leg chain.

    v2: (a) multi-start seeds (v1 got trapped in local minima at joint
    limits — 39 mm error on the standing pose); (b) clamping moved to the
    update step with margin shrinking over iterations so early large steps
    are not projected onto the limit boundary and stuck there; (c) position
    error dominates the convergence criterion.
    Returns (q, pos_err_m, rot_err_rad).
    """
    def clamp(qs, margin=0.0):
        return [max(l + margin, min(u - margin, v)) for v, (l, u) in zip(qs, limits)]

    def run(qs):
        qs = clamp(list(qs))
        best_q, best_e = list(qs), np.inf
        for it in range(MAX_ITER):
            T = fk.fk_chain(chain, qs, T_base)
            p_c, R_c = T[:3, 3], T[:3, :3]
            e = np.concatenate([W_POS * (p_t - p_c), W_ROT * axis_angle(R_t @ R_c.T)])
            err = np.linalg.norm(e[:3] / W_POS)
            if err < best_e:
                best_e, best_q = err, list(qs)
            if err < TOL_POS and np.linalg.norm(e[3:] / W_ROT) < TOL_ROT:
                break
            J = np.zeros((6, 6))
            for k in range(6):
                dq = np.zeros(6)
                dq[k] = 1e-6
                Tk = fk.fk_chain(chain, [q + d for q, d in zip(qs, dq)], T_base)
                J[:, k] = np.concatenate([(Tk[:3, 3] - p_c) / 1e-6,
                                          axis_angle(Tk[:3, :3] @ R_c.T) / 1e-6])
            JJt = J @ J.T + (DAMP ** 2) * np.eye(6)
            dqv = J.T @ np.linalg.solve(JJt, e)
            np.clip(dqv, -STEP_CAP, STEP_CAP, out=dqv)
            qs = clamp([q + d for q, d in zip(qs, dqv)])
        T = fk.fk_chain(chain, qs, T_base)
        err = np.linalg.norm(T[:3, 3] - p_t)
        if err < best_e:
            best_e, best_q = err, list(qs)
        return best_q, best_e

    candidates = [q0] if not isinstance(q0, list) or True else []
    if seeds:
        candidates += list(seeds)
    best = None
    for cand in candidates:
        q, e = run(cand)
        if best is None or e < best[1]:
            best = (q, e)
        if e < TOL_POS * 0.5:
            break
    q, e = best
    T = fk.fk_chain(chain, q, T_base)
    rot_err = np.linalg.norm(axis_angle(R_t @ T[:3, :3].T))
    return np.array(q), e, rot_err


def main():
    lab = load_lab_order(X1_YAML)
    assert len(lab) == 29
    x1, f1 = UrdfFK(X1_URDF), UrdfFK(F1_URDF)
    print(f"[INFO] leg reach X1 L/R = {REACH_X1['L']*1000:.1f}/{REACH_X1['R']*1000:.1f} mm, "
          f"F1 = {REACH_F1['L']*1000:.1f}/{REACH_F1['R']*1000:.1f} mm, "
          f"V_SCALE = {V_SCALE['L']:.4f}/{V_SCALE['R']:.4f}")
    limits = {s: [(f1.joints[j]["lower"], f1.joints[j]["upper"]) for j in LEG_CHAIN[s]] for s in "LR"}
    leg_idx = {s: [lab.index(j) for j in LEG_CHAIN[s]] for s in "LR"}
    os.makedirs(DST, exist_ok=True)

    report = {}
    for p in sorted(glob.glob(os.path.join(SRC, "*.pkl"))):
        name = os.path.basename(p)
        clip = pickle.load(open(p, "rb"))
        q = np.asarray(clip["dof_pos"], dtype=np.float64)
        rp = np.asarray(clip["root_pos"], dtype=np.float64)
        rr = np.asarray(clip["root_rot"], dtype=np.float64)
        Tn = q.shape[0]

        q_new = q.copy()
        ik_pos, ik_rot = np.zeros((Tn, 2)), np.zeros((Tn, 2))
        qprev = {s: None for s in "LR"}
        hip_x1 = {s: x1.joints[LEG_CHAIN[s][0]]["xyz"] for s in "LR"}
        hip_f1 = {s: f1.joints[LEG_CHAIN[s][0]]["xyz"] for s in "LR"}
        for t in range(Tn):
            Tbase = np.eye(4)
            Tbase[:3, :3] = quat_wxyz_to_R(rr[t])
            Tbase[:3, 3] = rp[t]
            dm = dict(zip(lab, q[t]))
            Tx = x1.fk_all(dm, Tbase)
            for si, s in enumerate("LR"):
                Tt = Tx[ANKLE[s]]
                # hip centers in world (same root for both robots)
                hx1 = Tbase[:3, :3] @ hip_x1[s] + rp[t]
                hf1 = Tbase[:3, :3] @ hip_f1[s] + rp[t]
                p_t = scale_target(Tt[:3, 3].copy(), hx1, hf1, V_SCALE[s])
                R_t = Tt[:3, :3]
                if qprev[s] is None:
                    seed = [q[t, i] for i in leg_idx[s]]
                    qs, ep, er = solve_leg(f1, LEG_CHAIN[s], Tbase, p_t, R_t, seed, limits[s], seeds=LEG_SEEDS)
                else:
                    qs, ep, er = solve_leg(f1, LEG_CHAIN[s], Tbase, p_t, R_t, qprev[s], limits[s], seeds=LEG_SEEDS[:2])
                qprev[s] = qs
                q_new[t, leg_idx[s]] = qs
                ik_pos[t, si], ik_rot[t, si] = ep, er

        # F1 key bodies
        kb = np.zeros((Tn, 10, 3), dtype=np.float32)
        minz = 0.0
        for t in range(Tn):
            Tbase = np.eye(4)
            Tbase[:3, :3] = quat_wxyz_to_R(rr[t])
            Tbase[:3, 3] = rp[t]
            Ts = f1.fk_all(dict(zip(lab, q_new[t])), Tbase)
            for i, b in enumerate(KEY10):
                kb[t, i] = Ts[b][:3, 3]
            minz = min(minz, min(Ts[b][:3, 3][2] for b in KEY10))

        out = dict(fps=clip["fps"], root_pos=clip["root_pos"], root_rot=clip["root_rot"],
                   dof_pos=q_new, loop_mode=clip["loop_mode"], key_body_pos=kb)
        with open(os.path.join(DST, name), "wb") as f:
            pickle.dump(out, f)

        # gates
        dq = np.abs(np.diff(q_new, axis=0))
        dq_max = float(dq.max()) if Tn > 1 else 0.0
        dq_frac = float((dq > 0.5).mean()) if Tn > 1 else 0.0
        med, p99 = (float(np.median(dq)), float(np.percentile(dq, 99))) if Tn > 1 else (0.0, 0.0)
        rep = dict(
            frames=Tn,
            ik_pos_max_mm=float(ik_pos.max() * 1000),
            ik_pos_p99_mm=float(np.percentile(ik_pos, 99) * 1000),
            ik_pos_mean_mm=float(ik_pos.mean() * 1000),
            ik_rot_max_deg=float(np.degrees(ik_rot.max())),
            key_min_z_mm=float(minz * 1000),
            smooth_med=med, smooth_p99=p99, smooth_max=dq_max, smooth_frac_gt05=dq_frac,
        )
        report[name] = rep
        print(f"{name:42s} T={Tn:5d} IKpos p99={rep['ik_pos_p99_mm']:5.2f}mm max={rep['ik_pos_max_mm']:6.2f}mm "
              f"rot_max={rep['ik_rot_max_deg']:5.2f}deg minz={rep['key_min_z_mm']:7.1f}mm "
              f"dq_p99={p99:.3f} dq_max={dq_max:.3f}")

    with open(os.path.join(DST, "_retarget_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] wrote {len(report)} clips to {DST}")


if __name__ == "__main__":
    main()
