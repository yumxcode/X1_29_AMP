#!/usr/bin/env python3
"""CMU SMPL-H RUNNING-clip selection + stageii conversion (F1w run data route).

Adapted from cmu_select_walk.py (v72). Differences:
  - run band targets: speed 1.2-3.5 m/s, cadence 140-200 spm, step 0.55-1.60 m
    (human jog->mid-run; Froude-scaled: F1w L=0.573m, target band >=1.6 m/s,
    demo band must reach ~3.0 with margin)
  - window >= 8 s (assessment doc: prefer >=10 s continuous; report durations)
  - vertical bob amplitude reported (run has large pelvis oscillation; sanity
    against walk-class clips)
  - report -> data/amass/cmu_run_scan_report.json ; converts to the same
    stageii schema consumed by the GMR chain.

Selection metrics identical in spirit to the walk route (heading-frame speed,
net/travel straightness, vertical-oscillation cadence via FFT).
"""
import json
import math
from pathlib import Path

import numpy as np

from cmu_select_walk import (RAW, OUT, ROOT, convert, dominant_freq,
                             velocity_heading, window_metrics)

TGT = dict(cad_lo=140, cad_hi=200, v_lo=1.2, v_hi=3.5, s_lo=0.55, s_hi=1.60,
           min_win=8.0)


def find_best_run_window(wx, wy, y, z, fps):
    N = len(wx)
    w = int(min(10.0 * fps, N))
    best = None
    for i in range(0, N - w + 1, max(1, int(0.5 * fps))):
        j = i + w
        m = window_metrics(wx, wy, y, z, fps, i, j)
        if m is None:
            continue
        ok = (TGT["cad_lo"] <= m["cad"] <= TGT["cad_hi"]
              and TGT["v_lo"] <= m["v"] <= TGT["v_hi"]
              and TGT["s_lo"] <= m["step"] <= TGT["s_hi"]
              and m["straight"] >= 0.85)
        if not ok:
            continue
        bob = float(np.ptp(z[i:j]))
        m["bob"] = bob
        # prefer band center (1.9 m/s, 165 spm), reward longer windows
        score = (abs(m["v"] - 1.9) + 2 * abs(m["cad"] - 165) / 165
                 + abs(m["step"] - 0.95) / 0.95 + (1 - m["straight"])
                 - 0.1 * (m["dur"] - 10.0))
        if best is None or score < best[0]:
            best = (score, i, j, m)
    return best


def main():
    RUN_CANDIDATES = """09_01 09_02 09_03 09_04 09_05 09_06 09_07 09_08 09_09
    09_10 09_11 02_03 16_35 16_36 16_45 16_46 16_55 16_56 35_17 35_18 35_19
    35_20 141_01 141_02 141_03 141_34 143_01 143_42 127_03 127_06 127_07
    127_08""".split()

    rows = []
    for name in RUN_CANDIDATES:
        p = RAW / f"{name}.npz"
        if not p.exists():
            print(f"{name}: MISSING (download failed?)")
            continue
        d = np.load(p, allow_pickle=True)
        try:
            poses = np.asarray(d["poses"], dtype=np.float32)
            trans = np.asarray(d["trans"], dtype=np.float32)
            fps = float(np.asarray(d["mocap_framerate"]).item())
        except Exception as e:
            print(f"{name}: SKIP {e}")
            continue
        if poses.ndim != 2 or poses.shape[1] < 66 or len(poses) < 120:
            continue
        h = velocity_heading(trans, fps)
        y = -np.sin(h) * trans[:, 0] + np.cos(h) * trans[:, 1]
        z = trans[:, 2]
        bw = find_best_run_window(trans[:, 0], trans[:, 1], y, z, fps)
        row = dict(name=name, fps=fps, dur=len(poses) / fps)
        if bw:
            m = bw[3]
            row.update({k: round(v, 3) for k, v in m.items()},
                       i=bw[1], j=bw[2], qualify=True)
        else:
            # diagnostics: best window ignoring bands
            from cmu_select_walk import find_best_window
            any_w = find_best_window(trans[:, 0], trans[:, 1], y, z, fps,
                                     want=False)
            if any_w:
                m = any_w[3]
                row.update({k: round(v, 3) for k, v in m.items()},
                           i=any_w[1], j=any_w[2], qualify=False)
        rows.append(row)

    print(f"{'clip':10s} {'dur':>5s} {'v':>5s} {'spm':>5s} {'step':>5s} "
          f"{'str':>5s} {'bob':>5s} {'qual':>4s}")
    for r in sorted(rows, key=lambda r: (not r.get("qualify"), -r.get("v", 0))):
        print(f"{r['name']:10s} {r['dur']:5.1f} {r.get('v', 0):5.2f} "
              f"{r.get('cad', 0):5.0f} {r.get('step', 0):5.2f} "
              f"{r.get('straight', 0):5.2f} {r.get('bob', -1):5.2f} "
              f"{'Y' if r.get('qualify') else '-':>4s}")

    (ROOT / "data" / "amass" / "cmu_run_scan_report.json").write_text(
        json.dumps(rows, indent=1, default=float))
    n = sum(1 for r in rows if r.get("qualify"))
    print(f"[SCAN] {n}/{len(rows)} qualify in run band "
          f"[{TGT['v_lo']},{TGT['v_hi']}] m/s")


if __name__ == "__main__":
    main()
