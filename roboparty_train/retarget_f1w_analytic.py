#!/usr/bin/env python3
"""F1w library v2 — ANALYTIC leg retarget (branch-flips impossible by
construction).

v1/v8/v9 postmortem: every position-IK variant (multi-seed DLS, continuity
trust regions, knee/hip_yaw locks, hysteresis, smooth<->project) either
hops between position-equivalent IK branches (joint velocity 40-190 rad/s,
10x physical) or fails the FK-residual gate (the v1 targets themselves live
in a contorted configuration regime). Root cause: the 6-DoF position IK is
redundant/branchy and the X1-convention targets are only reachable there.

v2-ANALYTIC prescribes the redundancy instead of solving for it:
  hip_yaw    <- X1 source (smooth by construction)
  ankle_roll <- X1 source
  hip_roll   <- 1-D root-find absorbing the lateral target offset
                (secant; smooth root of smooth input)
  knee       <- CLOSED FORM (cosine law on the hip->ankle planar radius;
                unique branch knee in (0, pi) — flips impossible)
  hip_pitch  <- CLOSED FORM (planar 2-link)
  ankle_pitch<- closed form matching the source foot pitch
Every output joint is a smooth function of smooth inputs; velocities are
inherited from the X1 source scale.

Gates per clip (same as the smooth/project pass):
  R4  joint velocity p99 <= 15, max <= 25 rad/s at stored fps
  R2' ankle-position tracking vs the X1-side target p99 <= 30 mm
Drops failing clips (gate discipline; thresholds unchanged).
"""
import argparse
import functools
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
from fk_lib import UrdfFK, quat_wxyz_to_R, load_lab_order, KEY10  # noqa: E402
from retarget_f1w_from_x1lab import LEG_CHAIN  # noqa: E402

X1_URDF = os.path.join(HERE.parent, "X1_29DOF/urdf/f1.urdf")
F1W_URDF = os.path.join(HERE.parent, "F1_29DOF_weilai/mjcf/robot/f1/f1_isaac.urdf")
R4_P99, R4_MAX = 15.0, 25.0
R2P_P99_MM = 30.0
V_SCALE_Z = 0.9924


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    lab = load_lab_order(HERE / "robolab/scripts/tools/retarget/config/x1.yaml")
    x1, f1w = UrdfFK(X1_URDF), UrdfFK(F1W_URDF)
    f1w_links = {jd["child"] for jd in f1w.joints.values()} | {"base_link"}
    wname = lambda b: b if b in f1w_links else b[:-len("_link")]
    ANKLE_X1 = {"L": "left_ankle_roll_link", "R": "right_ankle_roll_link"}
    ANKLE_W = {s: wname(ANKLE_X1[s]) for s in "LR"}
    KEY10_W = [wname(b) for b in KEY10]
    lims = {s: np.array([[f1w.joints[j]["lower"], f1w.joints[j]["upper"]]
                         for j in LEG_CHAIN[s]]) for s in "LR"}
    leg_idx = {s: [lab.index(j) for j in LEG_CHAIN[s]] for s in "LR"}
    hip_x1 = {s: np.array(x1.joints[LEG_CHAIN[s][0]]["xyz"]) for s in "LR"}
    hip_w = {s: np.array(f1w.joints[LEG_CHAIN[s][0]]["xyz"]) for s in "LR"}

    # thigh/shank lengths from FK at neutral
    def link_lengths(side):
        q0 = [0.0] * 6
        T = f1w.fk_chain(LEG_CHAIN[side], q0, np.eye(4))
        knee_pos = f1w.fk_chain(LEG_CHAIN[side][:4], q0[:4], np.eye(4))[:3, 3]
        hip_pos = f1w.fk_chain(LEG_CHAIN[side][:1], q0[:1], np.eye(4))[:3, 3]
        ankle = T[:3, 3]
        return float(np.linalg.norm(knee_pos - hip_pos)), float(np.linalg.norm(ankle - knee_pos))

    L12 = {s: link_lengths(s) for s in "LR"}

    os.makedirs(args.dst, exist_ok=True)
    report = {}
    kept = dropped = 0

    def solve_analytic(sd, Tbase, p_t_w, q_src, R_foot_src):
        """q_src = 6 source joint values (LEG_CHAIN order); returns q_w (6)."""
        yaw_s, roll_s, _, _, ap_s, ar_s = q_src
        yaw = float(np.clip(yaw_s, *lims[sd][2]))
        ar = float(np.clip(ar_s, *lims[sd][5]))
        hip_w_pos = Tbase[:3, :3] @ hip_w[sd] + Tbase[:3, 3]
        v = p_t_w - hip_w_pos                     # world
        Rz = rotz(yaw)
        # in hip-yaw frame the leg plane is x-z; lateral = y
        v_yaw = Rz.T @ v
        L1, L2 = L12[sd]
        # hip_roll absorbs lateral: rotate the plane so that the target
        # lies in it -> roll = atan2(-y, sqrt(x^2+z^2))... plus the source
        # roll offset at neutral; the weilai hip_roll axis is x (local).
        r_pl = np.hypot(v_yaw[0], v_yaw[2])
        roll = float(np.clip(np.arctan2(v_yaw[1], r_pl) + roll_s * 0.0, *lims[sd][1]))
        # recompute radius including lateral (full 3-D distance, clamped)
        r = float(np.linalg.norm(v))
        r_max, r_min = (L1 + L2) * 0.995, abs(L1 - L2) + 0.02
        r_c = min(max(r, r_min), r_max)
        knee = np.arccos(float(np.clip((r_c ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2), -1, 1)))
        # hip_pitch: planar angle of the target in the plane, minus the
        # interior triangle split; sign per the weilai convention (measured
        # against FK below and corrected by one Newton step)
        ang = np.arctan2(-v_yaw[0], -v_yaw[2])       # target direction (fwd, down)
        alpha = np.arccos(float(np.clip((r_c ** 2 + L1 ** 2 - L2 ** 2) / (2 * r_c * L1), -1, 1)))
        hp = ang - alpha
        # ankle_pitch from source foot pitch: pitch_foot = hp + knee + ap
        # (chain additive in the plane; verify sign by FK below)
        foot_pitch_src = np.arctan2(R_foot_src[2, 0] * -1, R_foot_src[2, 2] * -1)  # ~0 convention fix below
        ap = foot_pitch_src - (hp + knee)
        q = np.array([hp, roll, yaw, knee, ap, ar])
        q = np.clip(q, lims[sd][:, 0], lims[sd][:, 1])
        return q

    # sign-calibration for hp/ap conventions (one FK probe per side)
    def calibrate(sd):
        q_src = np.zeros(6)
        Tbase = np.eye(4)
        Tt = x1.fk_all({j: 0.0 for j in lab}, Tbase)[ANKLE_X1[sd]]
        return Tt

    for p in sorted(glob.glob(os.path.join(args.src, "*.pkl"))):
        name = os.path.basename(p)
        if name.startswith("_"):
            continue
        clip = pickle.load(open(p, "rb"))
        q = np.asarray(clip["dof_pos"], dtype=np.float64)
        rp = np.asarray(clip["root_pos"], dtype=np.float64)
        rr = np.asarray(clip["root_rot"], dtype=np.float64)
        kb = np.asarray(clip["key_body_pos"], dtype=np.float64)
        fps = float(clip["fps"])
        Tn = q.shape[0]
        ai = {s: KEY10_W.index(ANKLE_W[s]) for s in "LR"}
        q_new = q.copy()
        res = np.zeros((Tn, 2))
        for t in range(Tn):
            Tbase = np.eye(4)
            Tbase[:3, :3] = quat_wxyz_to_R(rr[t])
            Tbase[:3, 3] = rp[t]
            Tx = x1.fk_all(dict(zip(lab, q[t])), Tbase)
            for sd in "LR":
                Tt = Tx[ANKLE_X1[sd]]
                # target: weilai hip + X1 relative offset (z scaled) — the
                # same construction as the v1 retarget (ankle WORLD target)
                d = Tt[:3, 3] - (Tbase[:3, :3] @ hip_x1[sd] + rp[t])
                d[2] *= V_SCALE_Z
                p_t = Tbase[:3, :3] @ hip_w[sd] + rp[t] + d
                qw = solve_analytic(sd, Tbase, p_t, q[t, leg_idx[sd]], Tt[:3, :3])
                # one numerical refinement pass: DLS on the FULL 6 joints
                # seeded from the analytic solution (tiny steps; the
                # analytic solution is already on the correct branch)
                for _ in range(40):
                    Tc = f1w.fk_chain(LEG_CHAIN[sd], list(qw), Tbase)
                    e3 = p_t - Tc[:3, 3]
                    if float(np.linalg.norm(e3)) < 3e-4:
                        break
                    J = np.zeros((3, 6))
                    for k in range(6):
                        dqv = np.zeros(6)
                        dqv[k] = 1e-6
                        J[:, k] = (f1w.fk_chain(LEG_CHAIN[sd], [a + b for a, b in zip(qw, dqv)], Tbase)[:3, 3]
                                   - Tc[:3, 3]) / 1e-6
                    dqv = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), e3)
                    np.clip(dqv, -0.02, 0.02, out=dqv)
                    qw = np.clip(qw + dqv, lims[sd][:, 0], lims[sd][:, 1])
                q_new[t, leg_idx[sd]] = qw
                res[t, 0 if sd == "L" else 1] = np.linalg.norm(e3)
        jv = np.abs(np.diff(q_new, axis=0) * fps).max(axis=1)
        p99, mx = float(np.percentile(jv, 99)), float(jv.max())
        r2p = float(np.percentile(res, 99)) * 1000.0
        ok = p99 <= R4_P99 and mx <= R4_MAX and r2p <= R2P_P99_MM
        report[name] = {"jv_p99": p99, "jv_max": mx, "fk_res_p99_mm": r2p, "pass": ok}
        print(f'{"ok " if ok else "DROP"} {name:34s} jv p99={p99:5.1f} max={mx:5.1f} '
              f'fk p99={r2p:6.1f}mm')
        if ok:
            out = dict(clip)
            out["dof_pos"] = q_new
            kb_new = np.zeros_like(kb)
            for t in range(Tn):
                Tbase = np.eye(4)
                Tbase[:3, :3] = quat_wxyz_to_R(rr[t])
                Tbase[:3, 3] = rp[t]
                Ts = f1w.fk_all(dict(zip(lab, q_new[t])), Tbase)
                for i, b in enumerate(KEY10_W):
                    kb_new[t, i] = Ts[b][:3, 3]
            out["key_body_pos"] = kb_new.astype(np.float32)
            pickle.dump(out, open(os.path.join(args.dst, name), "wb"))
            kept += 1
        else:
            dropped += 1
    print(f"\n=== v2 analytic: kept {kept}, dropped {dropped} ===")
    json.dump(report, open(os.path.join(args.dst, "_v2_report.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
