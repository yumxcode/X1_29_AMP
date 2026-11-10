#!/usr/bin/env python3
"""F1w run-library stageii conversion: CMU/ACCAD/HUMAN4D -> data/amass/CMU/*_stageii.npz.

Sources (all SMPL-H 156-dim, realdream-ai/AMASS mirror, 2026-11-10):
  - CMU run: 09_01..09_11 (sprint ~3.5, ~1.2s), 35_17..20 (3.2-3.5, 1.3-1.5s),
    02_03 (2.64), 16_35 (2.84)
  - ACCAD (descriptive runs): M/F_C3 straight run, C4/C5 run<->walk
    transitions (mid band 1.0-2.5), F_C2 run->stand, s008 Run1
  - HUMAN4D: s1-s4 treadmill running 13-16.5 s (cadence 135-176 spm,
    net displacement ~0 — treadmill; forward speed materializes after the
    v72 stance-anchor root rebuild in the retarget chain)

Windows:
  overground -> longest straight window (net/travel >= 0.85), any run speed
  treadmill  -> middle 12-16 s with cadence gate 140-190 spm

Output naming (consumed by retarget_v72_local.py from data/amass/CMU/):
  run_<src>_stageii.npz   e.g. run_09_01_stageii.npz, run_accM_C4_stageii.npz
"""
import json
import math
from pathlib import Path

import numpy as np

from cmu_select_walk import (RAW, OUT, ROOT, convert, dominant_freq,
                             velocity_heading, window_metrics)

ACC = ROOT / "data" / "amass" / "ACCAD_raw"
H4D = ROOT / "data" / "amass" / "H4D_raw"

OVERGROUND = {
    # name -> raw file
    **{f"09_{i:02d}": RAW / f"09_{i:02d}.npz" for i in range(1, 12)},
    **{f"35_{i}": RAW / f"35_{i}.npz" for i in (17, 18, 19, 20)},
    "02_03": RAW / "02_03.npz",
    "16_35": RAW / "16_35.npz",
    "accM_C3": ACC / "accM_C3_run.npz",
    "accF_C3": ACC / "accF_C3_run.npz",
    "acc8_run": ACC / "acc8_run1.npz",
    "accM_C4": ACC / "accM_C4_run2walk.npz",
    "accM_C4b": ACC / "accM_C4b_run2walk.npz",
    "accM_C5": ACC / "accM_C5_walk2run.npz",
    "accF_C4": ACC / "accF_C4_run2walk.npz",
    "accF_C5": ACC / "accF_C5_walk2run.npz",
    "accF_C2": ACC / "accF_C2_run2stand.npz",
}
TREADMILL = {
    "h4d_s1": H4D / "h4d_s1_run.npz",
    "h4d_s2": H4D / "h4d_s2_run.npz",
    "h4d_s3": H4D / "h4d_s3_run.npz",
    "h4d_s4": H4D / "h4d_s4_run.npz",
}


def best_overground_window(trans, y, z, fps):
    """Longest straight run window: grow from the best-scoring 4 s seed."""
    N = len(trans)
    w0 = int(min(4.0 * fps, N))
    best = None
    for i in range(0, N - w0 + 1, max(1, int(0.25 * fps))):
        j = i + w0
        m = window_metrics(trans[:, 0], trans[:, 1], y, z, fps, i, j)
        if m is None:
            continue
        if not (1.0 <= m["v"] <= 5.0 and m["straight"] >= 0.9):
            continue
        # extend right while still straight-run
        while j < N:
            m2 = window_metrics(trans[:, 0], trans[:, 1], y, z, fps, i, j + 1)
            if m2 is None or m2["straight"] < 0.85 or not (1.0 <= m2["v"] <= 5.0):
                break
            j += 1
        m = window_metrics(trans[:, 0], trans[:, 1], y, z, fps, i, j)
        if best is None or m["dur"] > best[3]["dur"]:
            best = (i, j, 0, m)
    return best


def treadmill_window(trans, z, fps):
    """Middle segment, cadence-gated."""
    N = len(trans)
    w = int(min(16.0 * fps, N))
    i = (N - w) // 2
    # trim 1 s edges from each side of the full clip
    fz, pz = dominant_freq(z[i:i + w], fps, 1.0, 4.0)
    cad = 60 * fz
    m = dict(v=0.0, cad=cad, step=0.0, straight=0.0, dur=w / fps, bob=float(np.ptp(z[i:i+w])))
    ok = 140 <= cad <= 190 and m["bob"] >= 0.05
    return (i, i + w, 0, m), ok


def main():
    rows = []
    print(f"{'clip':12s} {'src':9s} {'dur':>5s} {'v':>5s} {'spm':>5s} {'str':>5s} {'bob':>5s} {'ok':>3s}")
    for name, path in sorted(OVERGROUND.items()):
        d = np.load(path, allow_pickle=True)
        trans = np.asarray(d["trans"], dtype=np.float32)
        fps = float(np.asarray(d["mocap_framerate"]).item())
        h = velocity_heading(trans, fps)
        y = -np.sin(h) * trans[:, 0] + np.cos(h) * trans[:, 1]
        bw = best_overground_window(trans, y, trans[:, 2], fps)
        if bw is None:
            print(f"{name}: no straight run window")
            rows.append(dict(name=name, ok=False))
            continue
        i, j, _, m = bw
        # trim 0.15 s edges (start/stop transients)
        e = int(0.15 * fps)
        i2, j2 = min(i + e, j), max(j - e, i + e)
        m2 = window_metrics(trans[:, 0], trans[:, 1], y, trans[:, 2], fps, i2, j2)
        dst = OUT / f"run_{name}_stageii.npz"
        convert(path, i2, j2, dst)
        rows.append(dict(name=name, src="over", ok=True, i=i2, j=j2,
                         dur=m2["dur"], v=m2["v"], cad=m2["cad"],
                         straight=m2["straight"], step=m2["step"]))
        print(f"{name:12s} {'over':9s} {m2['dur']:5.1f} {m2['v']:5.2f} "
              f"{m2['cad']:5.0f} {m2['straight']:5.2f} {np.ptp(trans[i2:j2,2]):5.2f} {'Y':>3s}")

    for name, path in sorted(TREADMILL.items()):
        d = np.load(path, allow_pickle=True)
        trans = np.asarray(d["trans"], dtype=np.float32)
        fps = float(np.asarray(d["mocap_framerate"]).item())
        (i, j, _, m), ok = treadmill_window(trans, trans[:, 2], fps)
        if ok:
            dst = OUT / f"run_{name}_stageii.npz"
            convert(path, i, j, dst)
        rows.append(dict(name=name, src="tmill", ok=ok, i=i, j=j, dur=m["dur"],
                         cad=m["cad"], bob=m["bob"]))
        print(f"{name:12s} {'tmill':9s} {m['dur']:5.1f} {0.0:5.2f} "
              f"{m['cad']:5.0f} {0.0:5.2f} {m['bob']:5.2f} {'Y' if ok else 'N':>3s}")

    (ROOT / "data" / "amass" / "cmu_run_stageii_index.json").write_text(
        json.dumps(rows, indent=1, default=float))
    n_ok = sum(1 for r in rows if r.get("ok"))
    d_tot = sum(r.get("dur", 0) for r in rows if r.get("ok"))
    print(f"[DONE] {n_ok} clips, total {d_tot:.1f} s -> {OUT}")


if __name__ == "__main__":
    main()
