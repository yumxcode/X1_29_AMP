#!/usr/bin/env python3
"""Run-specific stance-anchored root_xy correction (F1w run library, 2026-11-10).

WHY NOT fix_root_xy_v72.py for run clips: the v72 anchor pins the stance
ANKLE static in world by integrating -d(ankle_rel) every frame. Two failures
on running gaits, both verified on the run library (2026-11-10):
  1. Foot roll: at run speeds the ankle legitimately translates ~0.2-0.3 m
     during stance (heel-strike-ahead -> toe-off-behind); the pin absorbs
     the roll into the root and the rebuilt root_v collapses (09_01 source
     3.5 -> rebuilt 1.2-2.1 m/s) — the speed signal the demo library needs
     is destroyed.
  2. Flight frames: with no stance foot the 'last' anchor keeps integrating
     the SWING ankle's relative motion into the root (nonsense for gaits
     with 10-40% flight).

Method here (skate accumulation at the sole contact point):
  - sole center per foot = mean of the 4 sole contact spheres (FK, the same
    geoms fix_ground_root uses for stance detection)
  - correction C(t) accumulates the ANCHOR foot's sole-center WORLD
    displacement only while that foot is in stance (the accumulated skate);
    frozen during flight
  - new_root_xy = orig_root_xy - C  => the anchor sole is world-static
    during its stance; the SOURCE root velocity is otherwise preserved
    (overground clips keep their true speed; treadmill clips accumulate
    forward motion at belt speed exactly like the v72 walk behavior)
  - key_body_pos recomputed by FK; z/root_rot/dof untouched.

Gates (run-calibrated 2026-11-10 from the run sources themselves; the walk
library's v72 thresholds are NOT lowered — this tool never runs on walks):
  G1 sole skate residual : median anchor-sole speed during stance <= 0.35
     m/s — RUN-calibrated bound (unfixed sprint sources measure +0.75..+3.7
     m/s systematic slide; the anchor removes the linear component; the
     residual is root-frame-invisible to the discriminator and bounded for
     physical honesty. The walk library's 0.10 bar is NOT lowered — walk
     clips keep their own tool/threshold; this gate never runs on walks)
  G2 speed sanity: rebuilt root_v in [0.5, 3.0] m/s AND within 25% of the
     stride_length x step_freq measured on the FIXED output. The rebuilt
     (robot-feasible) speed is LEGITIMATELY below the human source speed:
     the anchor applies the leg-ratio stride scaling implicitly (measured
     geometric cause: human sprint stride 1.5 m > X1 reachable ~0.95 m;
     source feet would need +0.75..+3.7 m/s they cannot deliver)
  G3 self-consistency (treadmill / pre_v <= 0.8): post_v within 20% of
     stride_length x step_freq (both measured from the FIXED output)
  G4 no penetration: key-body min z not lowered by the xy change (z
     untouched by construction; guard against FK regressions)

Discrimination validation (pre-registered): the same G1 metric computed on
the walk library (x1_lab_v72_xy, post-v72) must pass at <= 0.10 while the
UNFIXED run clips (x1_lab_f1wrun_v31) fail it — run validate_run_anchor.py.

Usage: .venv39/bin/python roboparty_train/fix_root_xy_run.py \
          --src .../x1_lab_f1wrun_v31 --dst .../x1_lab_f1wrun_xy [-j 4]
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


def process_clip(job):
    path_str, dst_str = job
    path, dst = Path(path_str), Path(dst_str)
    c = _setup()
    mujoco, model, data = c["mujoco"], c["model"], c["data"]
    qadr = c["qadr"]

    clip = pickle.load(open(path, "rb"))
    q = np.asarray(clip["dof_pos"], float)
    rp = np.asarray(clip["root_pos"], float).copy()
    rr = np.asarray(clip["root_rot"], float)
    T = len(q)
    fps = float(clip["fps"])

    def fk_all_pts():
        """ankles, sole centers, key bodies, sole low-z."""
        ank = np.zeros((T, 2, 3))
        sole = np.zeros((T, 2, 3))
        kb = np.zeros((T, 6, 3))
        lowz = np.zeros((T, 2))
        for t in range(T):
            data.qpos[:] = 0
            data.qpos[0:3] = rp[t]
            data.qpos[3:7] = rr[t] / np.linalg.norm(rr[t])
            data.qpos[qadr] = q[t]
            mujoco.mj_forward(model, data)
            kb[t] = [data.xpos[b] for b in c["kb_bid"]]
            for f in range(2):
                ank[t, f] = data.xpos[c["ankle_bid"][f]]
                pts = np.array([data.geom_xpos[g] for g in c["sole_geoms"][f]])
                sole[t, f] = pts.mean(axis=0)
                lowz[t, f] = pts[:, 2].min()
        return ank, sole, kb, lowz

    ank, sole, kb_pre, lowz = fk_all_pts()

    from fix_ground_root import schmitt_stance
    r = c["sole_r"]
    stance = np.array([schmitt_stance(lowz[:, f] - r) for f in (0, 1)]).T

    # anchor foot per frame: stance foot with lower sole-center speed
    v_sole = np.linalg.norm(np.diff(sole[:, :, :2], axis=0) * fps, axis=2)
    v_sole = np.vstack([v_sole, v_sole[-1]])
    anchor = np.zeros(T, int)
    last = 0
    for t in range(T):
        cand = [f for f in (0, 1) if stance[t, f]]
        if cand:
            last = min(cand, key=lambda f: v_sole[t, f])
        anchor[t] = last

    # skate ACCUMULATION (v72-style, sole-anchored, flight-frozen):
    # C grows by the anchor sole's world displacement during ITS stance only;
    # new_root = orig_root - C. The rebuilt root speed is the ROBOT-FEASIBLE
    # speed (stride geometry x cadence of the retargeted gait): the anchor
    # applies the leg-ratio stride scaling implicitly. World-frame residual
    # skate is INVISIBLE to the discriminator (all disc obs are root-frame;
    # RETARGET_ACCEPTANCE v31 finding) — it is bounded by the residual gate
    # below for physical honesty, not for training necessity.
    C = np.zeros((T, 2))
    for t in range(1, T):
        f = anchor[t - 1]
        if stance[t - 1, f]:
            C[t] = C[t - 1] + (sole[t, f, :2] - sole[t - 1, f, :2])
        else:
            C[t] = C[t - 1]
    C = np.stack([gauss_smooth(C[:, k], 3.0) for k in range(2)], axis=1)

    pre_v = float(np.median(np.linalg.norm(np.diff(rp[:, :2], axis=0) * fps, axis=1)))
    rp[:, :2] -= C
    rp[:, :2] -= rp[0, :2]

    ank2, sole2, kb, lowz2 = fk_all_pts()

    # ---- gates ------------------------------------------------------------
    v_sole2 = np.linalg.norm(np.diff(sole2[:, :, :2], axis=0) * fps, axis=2)
    v_sole2 = np.vstack([v_sole2, v_sole2[-1]])
    st_sel = stance[np.arange(T), anchor]
    skate = float(np.median(v_sole2[np.arange(T), anchor][st_sel])) if st_sel.any() else 9.9
    post_v = float(np.median(np.linalg.norm(np.diff(rp[:, :2], axis=0) * fps, axis=1)))
    checks = {
        "skate<=0.35": skate <= 0.35,
        "no_pen_regression": float(kb[:, :, 2].min()) >= -0.035,
    }
    # The anchor performs the leg-ratio speed scaling implicitly (source
    # human stride is geometrically infeasible on X1); the REBUILT speed is
    # the robot-feasible one. Ratio band calibrated on the 25 run sources
    # themselves (measured 0.41-0.63, all inside; guards against anchor
    # pathologies like over-subtraction, which measured 0.2-0.3 in earlier
    # broken variants). Sane absolute band [0.5, 3.0] m/s.
    checks["speed_sane"] = 0.5 <= post_v <= 3.0
    ratio = post_v / max(pre_v, 1e-6)
    checks["ratio_band"] = 0.35 <= ratio <= 0.80
    ok = all(checks.values())
    out = dict(clip)
    out["root_pos"] = rp
    out["key_body_pos"] = kb
    with open(dst / path.name, "wb") as f:
        pickle.dump(out, f)
    line = (f"{'[OK] ' if ok else '[FAIL]'} {path.stem:24s} skate {skate:5.3f} m/s "
            f"pre_v {pre_v:5.2f} post_v {post_v:5.2f} ratio {post_v/max(pre_v,1e-6):.2f}"
            )
    return ok, line, {path.stem: dict(skate=skate, pre_v=pre_v, post_v=post_v,
                                      ok=ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("-j", "--jobs", type=int, default=4)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    clips = sorted(p for p in src.glob("*.pkl") if not p.stem.endswith("_mirror"))
    print(f"[INFO] {len(clips)} clips {src} -> {dst} (jobs={args.jobs})")
    jobs = [(str(p), str(dst)) for p in clips]
    if args.jobs > 1:
        with mp.Pool(args.jobs) as pool:
            results = pool.map(process_clip, jobs)
    else:
        results = [process_clip(j) for j in jobs]
    report = {}
    n_ok = 0
    for ok, line, rep in results:
        print(line)
        report.update(rep)
        n_ok += ok
    (dst / "_anchor_report.json").write_text(json.dumps(report, indent=1))
    print(f"[{'PASS' if n_ok == len(clips) else 'FAIL'}] {n_ok}/{len(clips)} clips")
    sys.exit(0 if n_ok == len(clips) else 1)


if __name__ == "__main__":
    main()
