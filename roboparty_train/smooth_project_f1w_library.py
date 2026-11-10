#!/usr/bin/env python3
"""F1w library v2: alternating smooth<->project trajectory optimization,
then decimation. Fixes the v1 IK branch-flip velocity corruption while
keeping FK tracking.

Loop per clip (at 120 fps):
    repeat R rounds:
        q <- zero-phase 6 Hz low-pass(q)          # kills flip edges
        q <- per-frame ankle-position DLS,        # pulls FK back to the
             seeded from q, step cap 0.01 rad,    # v1 ankle targets
             60 iterations, legs only             # (bounded per-frame
    then decimate x4 -> 30 fps                     #  movement -> bounded
                                                   #  velocity)
Gates per clip (R4 source-calibrated on the X1 delivered library; R2'
uses the v1 R2 threshold family):
    R4  stored-fps joint velocity p99 <= 15, max <= 25 rad/s
    R2' FK(q_final) vs v1 key bodies p99 <= 30 mm
Drops failing clips (gate discipline).
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
from scipy.signal import butter, sosfiltfilt

print = functools.partial(print, flush=True)

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
from fk_lib import UrdfFK, quat_wxyz_to_R, load_lab_order, KEY10  # noqa: E402
from retarget_f1w_from_x1lab import LEG_CHAIN  # noqa: E402

F1W_URDF = os.path.join(HERE.parent, "F1_29DOF_weilai/mjcf/robot/f1/f1_isaac.urdf")
R4_P99, R4_MAX = 15.0, 25.0
R2P_P99_MM = 30.0
FC, DEC, ROUNDS = 5.0, 4, 5
DLS_ITERS, DLS_CAP = 120, 0.01
FRAME_MOVE_CAP = 0.12   # rad/frame total (25 rad/s at 120fps = 0.208; margin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    lab = load_lab_order(HERE / "robolab/scripts/tools/retarget/config/x1.yaml")
    f1w = UrdfFK(F1W_URDF)
    f1w_links = {jd["child"] for jd in f1w.joints.values()} | {"base_link"}
    wname = lambda b: b if b in f1w_links else b[:-len("_link")]
    ANKLE_W = {s: wname(f"{p}_ankle_roll_link") for s, p in (("L", "left"), ("R", "right"))}
    KEY10_W = [wname(b) for b in KEY10]
    limits = {s: [(f1w.joints[j]["lower"], f1w.joints[j]["upper"]) for j in LEG_CHAIN[s]]
              for s in "LR"}
    leg_idx = {s: [lab.index(j) for j in LEG_CHAIN[s]] for s in "LR"}
    sos = butter(4, FC, btype="low", fs=120.0, output="sos")
    os.makedirs(args.dst, exist_ok=True)
    report = {}
    kept = dropped = 0

    def project_leg(qs, chain, T_base, p_t, lims):
        qs0 = np.clip(np.asarray(qs, dtype=float), [l for l, _ in lims], [u for _, u in lims])
        qs = qs0.copy()
        moved = 0.0
        for _ in range(DLS_ITERS):
            if moved >= FRAME_MOVE_CAP:
                break
            T = f1w.fk_chain(chain, list(qs), T_base)
            e3 = p_t - T[:3, 3]
            if float(np.linalg.norm(e3)) < 3e-4:
                break
            J = np.zeros((3, 6))
            for k in range(6):
                dq = np.zeros(6)
                dq[k] = 1e-6
                J[:, k] = (f1w.fk_chain(chain, [q + d for q, d in zip(qs, dq)], T_base)[:3, 3]
                           - T[:3, 3]) / 1e-6
            dqv = J.T @ np.linalg.solve(J @ J.T + (1e-3 ** 2) * np.eye(3), e3)
            np.clip(dqv, -DLS_CAP, DLS_CAP, out=dqv)
            qs = np.clip(qs + dqv, [l for l, _ in lims], [u for _, u in lims])
            moved = float(np.max(np.abs(qs - qs0)))
        return qs

    for p in sorted(glob.glob(os.path.join(args.src, "*.pkl"))):
        name = os.path.basename(p)
        if name.startswith("_"):
            continue
        clip = pickle.load(open(p, "rb"))
        q = np.asarray(clip["dof_pos"], dtype=np.float64).copy()
        rp = np.asarray(clip["root_pos"], dtype=np.float64)
        rr = np.asarray(clip["root_rot"], dtype=np.float64)
        kb_raw = np.asarray(clip["key_body_pos"], dtype=np.float64)
        fps0 = float(clip["fps"])
        Tn = q.shape[0]
        ai = {s: KEY10_W.index(ANKLE_W[s]) for s in "LR"}
        for rnd in range(ROUNDS):
            q = sosfiltfilt(sos, q, axis=0)
            for t in range(Tn):
                Tbase = np.eye(4)
                Tbase[:3, :3] = quat_wxyz_to_R(rr[t])
                Tbase[:3, 3] = rp[t]
                for s in "LR":
                    qleg = q[t, leg_idx[s]]
                    q[t, leg_idx[s]] = project_leg(qleg, LEG_CHAIN[s], Tbase,
                                                   kb_raw[t, ai[s]], limits[s])
        qf = q[::DEC]
        rpf, rrf, kbf = rp[::DEC], rr[::DEC], kb_raw[::DEC]
        fps1 = fps0 / DEC
        jv = np.abs(np.diff(qf, axis=0) * fps1).max(axis=1)
        p99, mx = float(np.percentile(jv, 99)), float(jv.max())
        res = np.zeros((qf.shape[0], 10))
        for t in range(qf.shape[0]):
            Tbase = np.eye(4)
            Tbase[:3, :3] = quat_wxyz_to_R(rrf[t])
            Tbase[:3, 3] = rpf[t]
            Ts = f1w.fk_all(dict(zip(lab, qf[t])), Tbase)
            for i, b in enumerate(KEY10_W):
                res[t, i] = np.linalg.norm(Ts[b][:3, 3] - kbf[t, i])
        r2p = float(np.percentile(res, 99)) * 1000.0
        ok = p99 <= R4_P99 and mx <= R4_MAX and r2p <= R2P_P99_MM
        report[name] = {"jv_p99": p99, "jv_max": mx, "fk_res_p99_mm": r2p, "pass": ok,
                        "fps": fps1, "frames": int(qf.shape[0])}
        print(f'{"ok " if ok else "DROP"} {name:34s} jv p99={p99:5.1f} max={mx:5.1f} '
              f'fk p99={r2p:6.1f}mm T={qf.shape[0]}')
        if ok:
            out = dict(clip)
            out["dof_pos"] = qf
            out["root_pos"] = rpf
            out["root_rot"] = rrf
            out["key_body_pos"] = kbf.astype(np.float32)
            out["fps"] = fps1
            pickle.dump(out, open(os.path.join(args.dst, name), "wb"))
            kept += 1
        else:
            dropped += 1
    print(f"\n=== v2 smooth<->project ({ROUNDS} rounds): kept {kept}, dropped {dropped} ===")
    json.dump(report, open(os.path.join(args.dst, "_v2_report.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
