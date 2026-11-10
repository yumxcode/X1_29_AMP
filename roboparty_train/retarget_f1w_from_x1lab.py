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


def solve_leg(fk, chain, T_base, p_t, R_t, q0, limits, seeds=None):
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

    candidates = [q0]
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
                R_t = Tt[:3, :3]
                seed0 = qprev[sd] if qprev[sd] is not None else [q[t, i] for i in leg_idx[sd]]
                # target-shrink projection: weilai knee caps at 90 deg (X1
                # 114.6) -> sprint swing-recovery poses are unreachable
                # (p99 43-113mm pre-projection). Pull the target toward the
                # hip in 5% steps until reachable (<=3mm). Preserves the
                # trajectory shape at the largest feasible scale; the shrink
                # factor is reported per clip (honesty: the demo shows a
                # less-extreme heel recovery than the source).
                shrink = 1.0
                for _ in range(8):
                    p_t = hw + d * shrink
                    qs, ep, er = solve_leg(f1w, LEG_CHAIN[sd], Tbase, p_t, R_t, seed0,
                                           limits[sd], seeds=LEG_SEEDS)
                    if ep <= 3e-3:
                        break
                    shrink *= 0.95
                shrink_log[t, si] = shrink
                qprev[sd] = qs
                q_new[t, leg_idx[sd]] = qs
                ik_pos[t, si], ik_rot[t, si] = ep, er
                knee_sat[t] |= abs(qs[3] - limits[sd][3][1]) < 1e-3

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
            ik_pos_max_mm=float(ik_pos.max() * 1000),
            ik_pos_p99_mm=float(np.percentile(ik_pos, 99) * 1000),
            ik_pos_mean_mm=float(ik_pos.mean() * 1000),
            ik_rot_max_deg=float(np.degrees(ik_rot.max())),
            key_min_z_mm=float(minz * 1000),
            knee_sat_frac=float(knee_sat.mean()),
            shrink_p05=float(np.percentile(shrink_log, 5)),
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
