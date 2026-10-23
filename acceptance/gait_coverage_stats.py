#!/usr/bin/env python3
"""Gait coverage stats for X1 lab demo datasets (cadence x step-length audit).

Reproduces the metrics of the 2026-10-22 X1 gait-data audit (步频×步长覆盖、
支撑脚滑移、根/腿速度比) directly from x1_lab_* pkl files, so the same tool
can produce the before/after comparison report for the new dataset.

Conventions (audit 2026-10-22):
  - touchdown  = local max of ankle forward displacement rel. root (heading frame)
  - step length = fore-aft distance between the two ankles at touchdown
  - cadence    = per-foot cycle freq -> steps/min counts BOTH feet
  - planted    = ankle world horizontal speed < 0.10 m/s
  - key_body_pos columns (v32 10-body): 0/1 knees, 2/3 ankles, 4/5 elbows,
    6/7 shoulders, 8/9 wrists
Usage:
  python gait_coverage_stats.py <motion_dir> [--weights-json F] [--json OUT]
"""
import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

FPS_EPS = 1e-6


def heading_yaw(root_rot):
    """root_rot: (N,4) xyzw quaternions -> yaw per frame."""
    x, y, z, w = root_rot[:, 0], root_rot[:, 1], root_rot[:, 2], root_rot[:, 3]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.unwrap(yaw)


def world_to_heading(pos, yaw):
    """pos (...,3) world -> heading frame (x forward along unwrapped yaw)."""
    c, s = np.cos(yaw), np.sin(yaw)
    c, s = np.cos(yaw), np.sin(yaw)
    if pos.ndim == 3:
        c, s = c[:, None], s[:, None]
    x = c * pos[..., 0] + s * pos[..., 1]
    y = -s * pos[..., 0] + c * pos[..., 1]
    return np.stack([x, y, pos[..., 2]], axis=-1)


def detect_steps(ank_h, ank_fwd, fps, min_swing=0.18):
    """Touchdowns via Schmitt on ankle height (robust vs the fwd-maxima
    variant that double-counts on some clips). Touchdown = swing->stance
    transition (height falls below ON). Returns list of (idx, side)."""
    out = []
    for side in (0, 1):
        h = ank_h[:, side] - ank_h[:, side].min()   # lift above per-foot min
        ON, OFF = 0.010, 0.018                      # X1 lift band 22-63mm
        min_len = max(1, int(min_swing * fps))
        in_swing = False
        start = 0
        td = []
        for i in range(len(h)):
            if not in_swing and h[i] > OFF:
                in_swing, start = True, i
            elif in_swing and h[i] < ON:
                if i - start >= min_len:
                    td.append(i)
                in_swing = False
        out += [(i, side) for i in td]
    out.sort()
    return out


def step_metrics(kb, root_pos, root_rot, fps):
    yaw = heading_yaw(root_rot)
    ank = world_to_heading(kb[:, 2:4], yaw)          # (N,2,3) heading frame
    root_h = world_to_heading(root_pos, yaw)
    ank_fwd = ank[:, :, 0] - root_h[:, 0:1]           # rel. to root
    steps = detect_steps(ank[:, :, 2], ank_fwd, fps)
    res = []
    for k, (i, side) in enumerate(steps):
        others = [j for j, s in steps if s != side]
        if not others:
            continue
        # same-side period -> cadence
        same = [j for j, s in steps if s == side and j != i]
        if not same:
            continue
        prev_same = max([j for j in same if j < i], default=None)
        next_same = min([j for j in same if j > i], default=None)
        if prev_same is None or next_same is None:
            continue
        period = (next_same - prev_same) / fps / 2.0  # full same-side cycle = 2 steps
        cadence_spm = 60.0 / period                   # steps per minute (both feet)
        # step length: fore-aft distance between ankles at touchdown
        other = min(others, key=lambda j: abs(j - i))
        step_len = abs(ank_fwd[i, side] - ank_fwd[other, 1 - side])
        res.append(dict(idx=i, side=side, t=i / fps, cadence=cadence_spm,
                        step_len=float(step_len), period=period))
    return res, ank, root_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motion_dir")
    ap.add_argument("--weights-json", default=None,
                    help="optional dict name->weight (training weights)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--per-clip", action="store_true")
    args = ap.parse_args()

    d = Path(args.motion_dir)
    weights = {}
    if args.weights_json:
        weights = json.loads(Path(args.weights_json).read_text())

    all_steps = []
    clip_rows = []
    for p in sorted(d.glob("*.pkl")):
        if p.stem.endswith("_mirror"):
            continue
        m = pickle.load(open(p, "rb"))
        fps = float(m["fps"])
        kb = np.asarray(m["key_body_pos"])
        rp = np.asarray(m["root_pos"])
        rr = np.asarray(m["root_rot"])
        steps, ank, root_h = step_metrics(kb, rp, rr, fps)
        if not steps:
            clip_rows.append(dict(name=p.stem, n_steps=0))
            continue
        cad = np.array([s["cadence"] for s in steps])
        sl = np.array([s["step_len"] for s in steps])
        # planted fraction: ankle world speed < 0.10
        ank_w = kb[:, 2:4]
        v = np.linalg.norm(np.diff(ank_w[:, :, :2], axis=0) * fps, axis=2)
        planted = float((np.min(v, axis=1) < 0.10).mean())
        # leg speed vs root speed
        leg_v = float(np.median([60.0 * s["step_len"] / (100.0 * 0 + s["cadence"])
                                 for s in steps]))  # step_len * cadence_spm/60
        root_v = float(np.median(np.linalg.norm(np.diff(root_h[:, :2], axis=0) * fps, axis=1)))
        clip_rows.append(dict(
            name=p.stem, n_steps=len(steps),
            cad_med=float(np.median(cad)), cad_p25=float(np.percentile(cad, 2.5)),
            cad_p975=float(np.percentile(cad, 97.5)),
            sl_med=float(np.median(sl)), sl_p25=float(np.percentile(sl, 2.5)),
            sl_p975=float(np.percentile(sl, 97.5)),
            planted_frac=planted, root_v_med=root_v,
            implied_leg_v=leg_v,
            weight=weights.get(p.stem, 1.0)))
        for s in steps:
            all_steps.append(dict(clip=p.stem, weight=weights.get(p.stem, 1.0),
                                  t_frac=s["t"] / (len(kb) / fps), **s))

    # weighted distribution over (cadence, step_len)
    cad_all = np.array([s["cadence"] for s in all_steps])
    sl_all = np.array([s["step_len"] for s in all_steps])
    w_all = np.array([s["weight"] / max(s["t_frac"] * (1 / s["period"]), 1e-9)
                      for s in all_steps])  # weight/duration proxy
    print(f"=== {d.name}: {len(all_steps)} steps across {len(clip_rows)} source clips ===")
    hdr = f"{'clip':28s} {'n':>4s} {'spm med':>8s} {'spm 95%':>14s} {'step med':>9s} {'step 95%':>14s} {'plant%':>7s} {'rootV':>6s}"
    print(hdr)
    for r in clip_rows:
        if r.get("n_steps", 0) == 0:
            print(f"{r['name']:28s}    0")
            continue
        print(f"{r['name']:28s} {r['n_steps']:4d} {r['cad_med']:8.1f} "
              f"[{r['cad_p25']:5.1f},{r['cad_p975']:5.1f}] {r['sl_med']:9.3f} "
              f"[{r['sl_p25']:5.3f},{r['sl_p975']:5.3f}] "
              f"{100*r['planted_frac']:6.1f}% {r['root_v_med']:6.2f}")

    # coverage of the target neighborhood: 1.0 m/s isovelocity band
    v_implied = sl_all * cad_all / 60.0
    in_band = (v_implied > 0.8) & (v_implied < 1.25) & (cad_all < 150)
    print(f"\nsteps in 0.8-1.25 m/s band: {in_band.sum()}/{len(all_steps)}")
    if in_band.any():
        print(f"  cadence range {cad_all[in_band].min():.0f}-{cad_all[in_band].max():.0f} spm, "
              f"step {sl_all[in_band].min():.2f}-{sl_all[in_band].max():.2f} m")
    tgt = (v_implied > 0.8) & (v_implied < 1.25) & (cad_all >= 105) & (cad_all <= 135)
    print(f"steps in TARGET band (0.8-1.25 m/s & 105-135 spm): {tgt.sum()}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            dict(clips=clip_rows, steps=all_steps), indent=1))
        print(f"[saved] {args.json}")


if __name__ == "__main__":
    main()
