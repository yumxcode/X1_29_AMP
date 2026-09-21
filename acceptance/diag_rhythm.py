#!/usr/bin/env python3
"""Rhythm gate calibration (v63 arc): cadence + stride of refs vs policy.

Same Schmitt segmentation code path as gait_metrics.py (policy side) and
ref FK (reference side), so the two are directly comparable.

Per source (ref clip or policy rollout log) we report, per foot and pooled:
  v_med      median horizontal speed of the foot mid-point (m/s)
  cycle_s    period between consecutive touchdowns of the SAME foot (= gait
             cycle; human step cadence = 2 / cycle_s)
  stride_m   horizontal distance between consecutive touchdowns of the SAME
             foot (stride length; human step length = stride_m / 2)
  cad_spm    steps per minute counting BOTH feet (= 120 / cycle_s)
  duty       stance fraction

Reference clips come from roboparty_train/robolab/data/motions/x1_lab/*.pkl
(retargeted AMASS -> X1), FK'd in the SAME x1.xml used by sim2sim.
Policy logs are rollout npz's produced by mujoco_rollout.py --log.

Usage:
  ./.venv39/bin/python acceptance/diag_rhythm.py --ref
  ./.venv39/bin/python acceptance/diag_rhythm.py --log x.npz [y.npz ...]
"""
import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim2sim"))

from sim2sim.gait_metrics import schmitt_contact, stance_windows  # noqa: E402


def summarize(events):
    """events: list of (foot, td_i, lo_i); returns pooled metrics."""
    per = {}
    feet = sorted({f for f, _, _ in events})
    for f in feet:
        tds = [td for ff, td, _ in events if ff == f]
        per[f] = dict(
            n_cycles=max(0, len(tds) - 1),
            cycle_s=float(np.median(np.diff(tds))) if len(tds) > 2 else float("nan"),
        )
    all_tds = sorted([(td, f) for f, td, _ in events])
    return per, all_tds


def rhythm_from_series(bottom_z, dt, feet_names, mid_xy):
    """bottom_z (n,2) lowest sphere bottom; mid_xy (n,2) foot midpoint xy."""
    events = []
    for f, name in enumerate(feet_names):
        contact = schmitt_contact(bottom_z[:, f])
        for td, lo in stance_windows(contact, 2):
            events.append((name, td, lo))
    out = {}
    for name in feet_names:
        tds = [td for n, td, _ in events if n == name]
        f = feet_names.index(name)
        if len(tds) >= 3:
            cyc = np.diff(tds) * dt
            stride = [float(np.linalg.norm(mid_xy[b, f] - mid_xy[a, f]))
                      for a, b in zip(tds[:-1], tds[1:])]
            st = [(lo - td) * dt for n, td, lo in events if n == name]
            out[name] = dict(
                n_cycles=len(tds) - 1,
                cycle_s=float(np.median(cyc)),
                cycle_cv=float(np.std(cyc) / np.mean(cyc)),
                stride_m=float(np.median(stride)),
                cad_spm=float(60.0 / np.median(cyc)),
                duty=float(np.mean(st) / np.median(cyc)) if st else float("nan"),
            )
    return out


def analyze_ref(path):
    import mujoco
    from sim2sim.mujoco_rollout import build_model, find_sole_geoms, parse_yaml_list, DEFAULT_Q
    model, _ = build_model(ROOT / "gmr_x1_assets" / "x1.xml")
    data = mujoco.MjData(model)
    soles, _ = find_sole_geoms(model)
    feet_names = sorted(soles)
    mj_names = [model.joint(i).name for i in range(model.njnt)]
    hinge = [n for n in mj_names if n in DEFAULT_Q]
    jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in hinge}
    qadr = np.array([model.jnt_qposadr[jid[n]] for n in hinge])

    clip = pickle.load(open(path, "rb"), encoding="latin1")
    q = np.asarray(clip["dof_pos"], float)
    rp = np.asarray(clip["root_pos"], float)
    rr = np.asarray(clip["root_rot"], float)
    T, fps = len(q), float(clip["fps"])
    dt = 1.0 / fps
    bottom = np.zeros((T, 2))
    mid_xy = np.zeros((T, 2, 2))
    for i in range(T):
        data.qpos[:] = 0
        data.qpos[0:3] = rp[i]
        data.qpos[3:7] = rr[i] / np.linalg.norm(rr[i])
        data.qpos[qadr] = q[i]
        mujoco.mj_forward(model, data)
        for f, name in enumerate(feet_names):
            pts = np.array([data.geom_xpos[g] for g in soles[name]])
            bottom[i, f] = pts[:, 2].min() - 0.002
            mid_xy[i, f] = pts[:, :2].mean(0)
    v = np.linalg.norm(np.diff(rp[:, :2], axis=0), axis=1) / dt
    r = rhythm_from_series(bottom, dt, feet_names, mid_xy)
    v_med = float(np.median(v))
    for name in r:
        r[name]["v_med"] = v_med
        # consistency: speed ~= stride / cycle
        r[name]["v_impl"] = r[name]["stride_m"] / r[name]["cycle_s"]
    return v_med, r, dt


def analyze_log(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    dt = float(meta["control_dt"])
    feet = list(meta["foot_names"])
    sole = z["sole_xyz"].astype(np.float64)
    bottom = sole[:, :, :, 2].min(2) - 0.002   # (n,2)
    mid_xy = sole.mean(2)[:, :, :2]            # (n,2,2)
    r = rhythm_from_series(bottom, dt, feet, mid_xy)
    bp = z["base_pos"][:, :2]
    start = int(meta["settle_steps"]) + 2
    v = np.linalg.norm(np.diff(bp[start:], axis=0), axis=1) / dt
    v_med = float(np.median(v))
    for name in r:
        r[name]["v_med"] = v_med
        r[name]["v_impl"] = r[name]["stride_m"] / r[name]["cycle_s"]
    return v_med, r, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", action="store_true")
    ap.add_argument("--log", nargs="*", default=[])
    args = ap.parse_args()
    if args.ref:
        print(f"{'clip':28s} {'v':>5s} {'cycle_s':>8s} {'stride_m':>9s} {'cad_spm':>8s} "
              f"{'duty':>5s} {'v_impl':>6s} {'n':>3s}")
        rows = []
        for p in sorted(glob.glob(str(ROOT / "roboparty_train/robolab/data/motions/x1_lab/*.pkl"))):
            if "_mirror" in p:
                continue
            v, r, dt = analyze_ref(p)
            name = Path(p).stem
            vals = list(r.values())
            if not vals:
                print(f"{name:28s} (no events)")
                continue
            cyc = np.mean([x["cycle_s"] for x in vals])
            st = np.mean([x["stride_m"] for x in vals])
            cad = np.mean([x["cad_spm"] for x in vals])
            duty = np.mean([x["duty"] for x in vals])
            n = sum(x["n_cycles"] for x in vals)
            print(f"{name:28s} {v:5.2f} {cyc:8.3f} {st:9.3f} {cad:8.1f} {duty:5.2f} "
                  f"{st/cyc:6.2f} {n:3d}")
            rows.append((name, v, cyc, st, cad, duty))
        # pooled by speed bucket
        print("\n== pooled walk-family reference (0.3 <= v <= 1.3 m/s, walk clips)")
        w = [r for r in rows if "jog" not in r[0] and 0.3 <= r[1] <= 1.3]
        if w:
            print(f"  n={len(w)}  cycle med={np.median([r[2] for r in w]):.3f}s  "
                  f"stride med={np.median([r[3] for r in w]):.3f}m  "
                  f"cad med={np.median([r[4] for r in w]):.1f} spm")
        w1 = [r for r in rows if "jog" not in r[0] and 0.7 <= r[1] <= 1.3]
        if w1:
            print(f"  near-1.0 m/s subset: n={len(w1)}  cycle med={np.median([r[2] for r in w1]):.3f}s  "
                  f"stride med={np.median([r[3] for r in w1]):.3f}m")
        w2 = [r for r in rows if "jog" not in r[0] and r[1] < 0.7]
        if w2:
            print(f"  slow (<0.7) subset:  n={len(w2)}  cycle med={np.median([r[2] for r in w2]):.3f}s  "
                  f"stride med={np.median([r[3] for r in w2]):.3f}m")
    for p in args.log:
        v, r, dt = analyze_log(p)
        print(f"\n== {Path(p).name}  v_med={v:.3f} m/s  dt={dt}")
        for name, d in r.items():
            print(f"   {name:24s} cycle={d['cycle_s']:.3f}s cv={d['cycle_cv']:.3f} "
                  f"stride={d['stride_m']:.3f}m cad={d['cad_spm']:.1f}spm duty={d['duty']:.2f} "
                  f"n={d['n_cycles']}")


if __name__ == "__main__":
    main()
