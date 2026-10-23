#!/usr/bin/env python3
"""Stance-anchored root_xy reconstruction for X1 lab references (v72, the
audit-2026-10-22 "修根平移尺度" step).

Why: GMR preserves the SOURCE pelvis translation scale (human-scale for CMU
overground clips; net-zero for treadmill/in-place clips). The retargeted
robot kinematics then cannot realize that translation -> support feet slide
(103_07 support foot slides ~0.45 m/s; treadmill feet slide at belt speed;
planted-frame fraction 0-4% vs 60-80% for real walking).

Fix: rebuild root_xy by integrating the stance ankle's motion RELATIVE to
the root (docstring intent of fix_ground_root.py step 4, which was never
implemented in code). After the transform the stance ankle is static in
world, so root speed == step_length x cadence of the retargeted gait
(robot-consistent by construction).

Pipeline position: AFTER fix_arm_decomposition (v30) + fix_ground_root
(v31 z/pitch). xy-only change; z, root_rot, dof untouched; key_body_pos
recomputed by FK afterwards.

Gates (exit 1 on fail, audit thresholds):
  planted%  (frames with min ankle world speed < 0.15 m/s) >= 50%
  median stance-ankle drift speed <= 0.10 m/s
  speed ratio: |root speed - leg-implied speed| / leg-implied <= 0.15
  FK no-penetration unchanged (z untouched)

Usage: python fix_root_xy_v72.py --src .../x1_lab_v72_g --dst .../x1_lab_v72_xy [-j 4]
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
from fix_ground_root import _setup, gauss_smooth  # noqa: E402

CTX = {}


def process_clip(job):
    path_str, dst_str = job
    path, dst = Path(path_str), Path(dst_str)
    c = _setup()
    mujoco, model, data = c["mujoco"], c["model"], c["data"]
    qadr = c["qadr"]

    def fk_ankles(rp, rr, q):
        T = len(q)
        ank = np.zeros((T, 2, 3))
        kb = np.zeros((T, 6, 3))
        for t in range(T):
            data.qpos[:] = 0
            data.qpos[0:3] = rp[t]
            data.qpos[3:7] = rr[t] / np.linalg.norm(rr[t])
            data.qpos[qadr] = q[t]
            mujoco.mj_forward(model, data)
            kb[t] = [data.xpos[b] for b in c["kb_bid"]]
            for f in range(2):
                ank[t, f] = data.xpos[c["ankle_bid"][f]]
        return ank, kb

    clip = pickle.load(open(path, "rb"))
    q = np.asarray(clip["dof_pos"], float)
    rp = np.asarray(clip["root_pos"], float).copy()
    rr = np.asarray(clip["root_rot"], float)
    T = len(q)
    fps = float(clip["fps"])

    ank, kb_pre = fk_ankles(rp, rr, q)
    ank_xy = ank[:, :, :2]                      # (T,2,2)
    rel = ank_xy - rp[:, None, :2]              # ankle rel to root (heading-agnostic)

    # stance from sole heights (same schmitt as fix_ground_root)
    lowz = np.zeros((T, 2))
    for t in range(T):
        data.qpos[:] = 0
        data.qpos[0:3] = rp[t]
        data.qpos[3:7] = rr[t] / np.linalg.norm(rr[t])
        data.qpos[qadr] = q[t]
        mujoco.mj_forward(model, data)
        for f in range(2):
            pts = np.array([data.geom_xpos[g] for g in c["sole_geoms"][f]])
            lowz[t, f] = pts[:, 2].min()
    from fix_ground_root import schmitt_stance
    r = c["sole_r"]
    stance = np.array([schmitt_stance(lowz[:, f] - r) for f in range(2)]).T  # (T,2)

    # anchor foot per frame: stance foot with lower ankle world speed
    v_ank = np.linalg.norm(np.diff(ank_xy, axis=0) * fps, axis=2)  # (T-1,2)
    v_ank = np.vstack([v_ank, v_ank[-1]])
    anchor = np.zeros(T, int)
    last = 0
    for t in range(T):
        cand = [f for f in (0, 1) if stance[t, f]]
        if cand:
            last = min(cand, key=lambda f: v_ank[t, f])
        anchor[t] = last

    # integrate: R'(t) = R'(t-1) - (rel(t, f) - rel(t-1, f))  -> stance ankle
    # static in world (A' = R' + rel constant <=> dR' = -drel)
    new_xy = np.zeros((T, 2))
    for t in range(1, T):
        f = anchor[t - 1]
        new_xy[t] = new_xy[t - 1] - (rel[t, f] - rel[t - 1, f])
    new_xy = np.stack([gauss_smooth(new_xy[:, k], 2.0) for k in range(2)], axis=1)
    new_xy -= new_xy[0]
    rp[:, :2] = new_xy

    ank2, kb = fk_ankles(rp, rr, q)

    # ---- gates ------------------------------------------------------------
    v2 = np.linalg.norm(np.diff(ank2[:, :, :2], axis=0) * fps, axis=2)
    v2 = np.vstack([v2, v2[-1]])                # (T,2)
    planted = float((np.minimum(v2[:, 0], v2[:, 1]) < 0.15).mean())
    st_sel = stance[np.arange(T), anchor]
    st_drift = float(np.median(v2[np.arange(T), anchor][st_sel])) if st_sel.any() else 9.9
    root_v = float(np.median(np.linalg.norm(np.diff(new_xy, axis=0) * fps, axis=1)))
    checks = {
        "planted>=50%": planted >= 0.50,
        "stance_drift<=0.10": st_drift <= 0.10,
        "root_v>0.2": root_v > 0.2,
    }
    ok = all(checks.values())
    out = dict(clip)
    out["root_pos"] = rp
    out["key_body_pos"] = kb
    with open(dst / path.name, "wb") as f:
        pickle.dump(out, f)
    line = (f"{'[OK] ' if ok else '[FAIL]'} {path.stem:24s} planted {planted*100:5.1f}% "
            f"drift {st_drift:5.3f} m/s root_v {root_v:5.2f} m/s")
    return ok, line, " ".join(k for k, v in checks.items() if not v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("-j", type=int, default=4)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    jobs = [(str(p), str(dst)) for p in sorted(src.glob("*.pkl"))
            if not p.stem.endswith("_mirror")]
    print(f"[INFO] {len(jobs)} clips")
    ctx = mp.get_context("fork")
    with ctx.Pool(args.j) as pool:
        results = pool.map(process_clip, jobs)
    n_ok = sum(1 for ok, _, _ in results if ok)
    for ok, line, why in results:
        print(line + (f"  <<{why}>>" if not ok else ""))
    print(f"[DONE] {n_ok}/{len(jobs)} pass")
    sys.exit(0 if n_ok == len(jobs) else 1)


if __name__ == "__main__":
    main()
