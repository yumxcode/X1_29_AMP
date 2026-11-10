#!/usr/bin/env python3
"""Stance-pinned leg re-solve for RUN clips on the X1 chain (2026-11-10).

Problem (measured): GMR's auto-IK (walk-tuned, smplx_to_x1_auto.json) leaves
1-2 m/s of REAL foot skating in the retargeted sprint clips (09/35 families:
sole-center stance speed 0.12-0.24 m/s residual AFTER root correction; the
v72 walk library measures 0.04-0.05). Root-side surgery cannot fix joint-
space skating.

Fix: per stance interval, re-solve the stance leg's 6 joints so the ankle
position stays at its touchdown world position (planted by construction);
the swing leg keeps the GMR source joints verbatim (style preserved). Root
xy/rot/z untouched — the SOURCE speed survives (pre_v == post_v).

Details:
  - stance detection: fix_ground_root schmitt on sole geoms (mujoco FK)
  - stance target: ankle position pinned at the interval's first frame;
    ankle ORIENTATION tracked from source (smooth in joint space)
  - liftoff blend: first 4 swing frames ramp from the pinned pose to the
    source trajectory (accumulated skate ~3-6 cm -> no visible jump)
  - joint limits from the X1 URDF; DLS IK, seed = previous frame

Gates (per clip; same numeric bars as the walk library wherever shared):
  G1 stance skate : median stance sole-center speed <= 0.10 m/s
  G2 speed kept   : |post_v - pre_v| / pre_v <= 0.05 (root untouched)
  G3 fidelity     : knee-joint corr vs GMR source >= 0.90 both sides
  G4 smoothness   : |dq| p99 <= 0.35 rad, max <= 3.0 @source fps
  G5 limits       : no frame beyond X1 URDF limits by >0.02 rad
  G6 ground       : key-body min z not lower than source by >5 mm

Usage: .venv39/bin/python roboparty_train/fix_run_stance_x1.py \
          --src .../x1_lab_f1wrun --dst .../x1_lab_f1wrun_st [-j 4]
(pipeline position: replaces the v30 arm-fix source for run clips; the arm
fix then runs on the output as usual)
"""
import argparse
import functools
import json
import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from fk_lib import UrdfFK, quat_wxyz_to_R, axis_angle  # noqa: E402

X1_URDF = ROOT.parent / "X1_29DOF" / "urdf" / "f1.urdf"
YAML = ROOT.parent / "roboparty_train" / "robolab" / "scripts" / "tools" / "retarget" / "config" / "x1.yaml"

LEG_CHAIN = {
    0: ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"],
    1: ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"],
}
ANKLE = {0: "left_ankle_roll_link", 1: "right_ankle_roll_link"}

_G = {}   # per-process globals (fork/spawn-safe via pool initializer)


def init_worker():
    from fk_lib import load_lab_order
    _G["lab"] = load_lab_order(str(YAML))
    _G["fk"] = UrdfFK(str(X1_URDF))
    _G["leg_idx"] = {f: [_G["lab"].index(j) for j in LEG_CHAIN[f]] for f in (0, 1)}
    _G["limits"] = {f: [(_G["fk"].joints[j]["lower"], _G["fk"].joints[j]["upper"])
                        for j in LEG_CHAIN[f]] for f in (0, 1)}

W_POS, W_ROT = 1.0, 0.5
DAMP, MAX_ITER, TOL_POS, TOL_ROT, STEP_CAP = 1e-3, 25, 1.5e-3, 0.01, 0.5


def solve_leg(fk, chain, T_base, p_t, R_t, q0, limits):
    def clamp(qs):
        return [max(l, min(u, v)) for v, (l, u) in zip(qs, limits)]

    qs = clamp(list(q0))
    best_q, best_e = list(qs), np.inf
    for _ in range(MAX_ITER):
        T = fk.fk_chain(chain, qs, T_base)
        p_c, R_c = T[:3, 3], T[:3, :3]
        e = np.concatenate([W_POS * (p_t - p_c), W_ROT * axis_angle(R_t @ R_c.T)])
        err = np.linalg.norm(e[:3])
        if err < best_e:
            best_e, best_q = err, list(qs)
        if err < TOL_POS and np.linalg.norm(e[3:]) < TOL_ROT:
            break
        J = np.zeros((6, 6))
        for k in range(6):
            dq = np.zeros(6)
            dq[k] = 1e-6
            Tk = fk.fk_chain(chain, [q + d for q, d in zip(qs, dq)], T_base)
            J[:, k] = np.concatenate([(Tk[:3, 3] - p_c) / 1e-6,
                                      axis_angle(Tk[:3, :3] @ R_c.T) / 1e-6])
        JJt = J @ J.T + DAMP ** 2 * np.eye(6)
        dqv = J.T @ np.linalg.solve(JJt, e)
        np.clip(dqv, -STEP_CAP, STEP_CAP, out=dqv)
        qs = clamp([q + d for q, d in zip(qs, dqv)])
    T = fk.fk_chain(chain, best_q, T_base)
    if np.linalg.norm(T[:3, 3] - p_t) < best_e:
        pass
    return np.array(best_q)


def stance_intervals(mask):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if j - i >= 3:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def process_clip(job):
    path_str, dst_str = job
    path, dst = Path(path_str), Path(dst_str)
    from fix_ground_root import _setup
    c = _setup()
    mujoco, model, data = c["mujoco"], c["model"], c["data"]
    qadr = c["qadr"]

    x1 = _G["fk"]
    lab = _G["lab"]
    leg_idx = _G["leg_idx"]
    limits = _G["limits"]

    clip = pickle.load(open(path, "rb"))
    q = np.asarray(clip["dof_pos"], float).copy()
    rp = np.asarray(clip["root_pos"], float)
    rr = np.asarray(clip["root_rot"], float)
    T = len(q)
    fps = float(clip["fps"])

    # FK helper: source ankle world poses + sole low-z (mujoco)
    def fk_ankles_soles(qmat):
        ank = np.zeros((T, 2, 3))
        R = np.zeros((T, 2, 3, 3))
        lowz = np.zeros((T, 2))
        for t in range(T):
            Tb = np.eye(4)
            Tb[:3, :3] = quat_wxyz_to_R(rr[t])
            Tb[:3, 3] = rp[t]
            Ts = x1.fk_all(dict(zip(lab, qmat[t])), Tb)
            for f in range(2):
                ank[t, f] = Ts[ANKLE[f]][:3, 3]
                R[t, f] = Ts[ANKLE[f]][:3, :3]
            data.qpos[:] = 0
            data.qpos[0:3] = rp[t]
            data.qpos[3:7] = rr[t] / np.linalg.norm(rr[t])
            data.qpos[qadr] = qmat[t]
            mujoco.mj_forward(model, data)
            for f in range(2):
                pts = np.array([data.geom_xpos[g] for g in c["sole_geoms"][f]])
                lowz[t, f] = pts[:, 2].min()
        return ank, R, lowz

    ank, Rsrc, lowz = fk_ankles_soles(q)
    from fix_ground_root import schmitt_stance
    r = c["sole_r"]
    stance = np.array([schmitt_stance(lowz[:, f] - r) for f in (0, 1)]).T

    q_new = q.copy()
    qprev = {f: None for f in (0, 1)}
    for f in (0, 1):
        for (t0, t1) in stance_intervals(stance[:, f]):
            p_pin = ank[t0, f].copy()
            for t in range(t0, t1):
                Tb = np.eye(4)
                Tb[:3, :3] = quat_wxyz_to_R(rr[t])
                Tb[:3, 3] = rp[t]
                seed = qprev[f] if qprev[f] is not None else [q[t, i] for i in leg_idx[f]]
                qs = solve_leg(x1, LEG_CHAIN[f], Tb, p_pin, Rsrc[t, f], seed, limits[f])
                qprev[f] = qs
                q_new[t, leg_idx[f]] = qs
            # liftoff blend: ramp from pinned pose back to source over 4 frames
            n_b = 4
            qs_pin = qprev[f]
            for k in range(n_b):
                t = t1 + k
                if t >= T:
                    break
                w = (k + 1) / n_b
                p_t = p_pin + w * (ank[t, f] - p_pin)
                Tb = np.eye(4)
                Tb[:3, :3] = quat_wxyz_to_R(rr[t])
                Tb[:3, 3] = rp[t]
                qs = solve_leg(x1, LEG_CHAIN[f], Tb, p_t, Rsrc[t, f], qs_pin, limits[f])
                q_new[t, leg_idx[f]] = qs
            qprev[f] = None

    # recompute key bodies + gates
    KEYBODIES = ["left_knee_pitch_link", "right_knee_pitch_link",
                 "left_ankle_roll_link", "right_ankle_roll_link",
                 "left_elbow_yaw_link", "right_elbow_yaw_link",
                 "left_shoulder_pitch_link", "right_shoulder_pitch_link",
                 "left_wrist_pitch_link", "right_wrist_pitch_link"]
    kb = np.zeros((T, 10, 3))
    for t in range(T):
        Tb = np.eye(4)
        Tb[:3, :3] = quat_wxyz_to_R(rr[t])
        Tb[:3, 3] = rp[t]
        Ts = x1.fk_all(dict(zip(lab, q_new[t])), Tb)
        for i, b in enumerate(KEYBODIES):
            kb[t, i] = Ts[b][:3, 3]

    ank2, _, lowz2 = fk_ankles_soles(q_new)
    stance2 = np.array([schmitt_stance(lowz2[:, f] - r) for f in (0, 1)]).T
    sole2 = np.zeros((T, 2, 3))
    for t in range(T):
        data.qpos[:] = 0
        data.qpos[0:3] = rp[t]
        data.qpos[3:7] = rr[t] / np.linalg.norm(rr[t])
        data.qpos[qadr] = q_new[t]
        mujoco.mj_forward(model, data)
        for f in range(2):
            pts = np.array([data.geom_xpos[g] for g in c["sole_geoms"][f]])
            sole2[t, f] = pts.mean(axis=0)
    v_sole = np.linalg.norm(np.diff(sole2[:, :, :2], axis=0) * fps, axis=2)
    v_sole = np.vstack([v_sole, v_sole[-1]])
    drifts = []
    for f in (0, 1):
        st = stance2[:, f]
        if st.any():
            drifts.append(float(np.median(v_sole[st, f])))
    skate = float(np.median(drifts)) if drifts else 9.9

    pre_v = float(np.median(np.linalg.norm(np.diff(rp[:, :2], axis=0) * fps, axis=1)))
    post_v = pre_v  # root untouched
    knee_corr = {}
    for f, n in ((0, "L"), (1, "R")):
        jn = LEG_CHAIN[f][3]
        a = q[:, lab.index(jn)]
        b = q_new[:, lab.index(jn)]
        knee_corr[n] = 1.0 if a.std() < 1e-6 else float(np.corrcoef(a, b)[0, 1])
    dq = np.abs(np.diff(q_new, axis=0))
    lim_ok = True
    for f in (0, 1):
        for k, jn in enumerate(LEG_CHAIN[f]):
            lo, hi = limits[f][k]
            col = q_new[:, lab.index(jn)]
            if ((col < lo - 0.02) | (col > hi + 0.02)).any():
                lim_ok = False
    kb_src = np.asarray(clip["key_body_pos"], float)
    pen_ok = float(kb[:, :, 2].min()) >= float(kb_src[:, :, 2].min()) - 0.005
    checks = {
        "skate<=0.10": skate <= 0.10,
        "knee_corr>=0.9": min(knee_corr.values()) >= 0.90,
        "smooth_p99<=0.35": float(np.percentile(dq, 99)) <= 0.35,
        "smooth_max<=3.0": float(dq.max()) <= 3.0,
        "limits": lim_ok,
        "ground": pen_ok,
    }
    ok = all(checks.values())
    out = dict(clip)
    out["dof_pos"] = q_new
    out["key_body_pos"] = kb
    with open(dst / path.name, "wb") as f_:
        pickle.dump(out, f_)
    line = (f"{'[OK] ' if ok else '[FAIL]'} {path.stem:24s} skate {skate:5.3f} m/s "
            f"v {pre_v:5.2f} knee_corr {knee_corr['L']:.3f}/{knee_corr['R']:.3f} "
            f"dq_p99 {np.percentile(dq, 99):.3f} "
            + " ".join(k for k, v in checks.items() if not v))
    return ok, line, {path.stem: dict(skate=skate, pre_v=pre_v, post_v=post_v,
                                      knee_corr=knee_corr, ok=ok,
                                      dq_p99=float(np.percentile(dq, 99)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("-j", "--jobs", type=int, default=4)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    init_worker()

    clips = sorted(p for p in src.glob("*.pkl") if not p.stem.endswith("_mirror"))
    print(f"[INFO] {len(clips)} clips {src} -> {dst} (jobs={args.jobs})")
    jobs = [(str(p), str(dst)) for p in clips]
    if args.jobs > 1:
        with mp.Pool(args.jobs, initializer=init_worker) as pool:
            results = pool.map(process_clip, jobs)
    else:
        results = [process_clip(j) for j in jobs]
    report = {}
    n_ok = 0
    for ok, line, rep in results:
        print(line)
        report.update(rep)
        n_ok += ok
    (dst / "_stance_fix_report.json").write_text(json.dumps(report, indent=1))
    print(f"[{'PASS' if n_ok == len(clips) else 'FAIL'}] {n_ok}/{len(clips)} clips")
    sys.exit(0 if n_ok == len(clips) else 1)


if __name__ == "__main__":
    main()
