#!/usr/bin/env python3
"""Scan the local AMASS/CMU archive (~2k clips) for TRUE OVERGROUND WALKING
windows (v31 source selection, plan A).

BMLrub is entirely treadmill-protocol (all in-place) — plan A source = CMU.
Cheap filter on 'trans' only, incremental cache (.cache_amass_scan.jsonl)
records EVERY file (pass or reject) so re-runs never rescan.

Pass criteria (clip level, generous — deep pass trims windows later):
  dur 3-45 s | median speed 0.6-1.6 m/s | straightness net/travel >= 0.55
  pelvis z mean in [0.90, 1.15], std < 0.08 m

Usage: python scan_amass_walking.py [--procs 8] [--limit N]
"""
import argparse
import functools
import json
import multiprocessing as mp
import zipfile
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache_amass_scan.jsonl"
DIRS = [ROOT / "AMASS/CMU"]


def trans_stats(path_str):
    path = Path(path_str)
    rec = {"file": str(path.relative_to(ROOT))}
    try:
        with zipfile.ZipFile(path) as z:
            with z.open("trans.npy") as f:
                tr = np.load(f, allow_pickle=False)
            with z.open("mocap_frame_rate.npy") as f:
                hz = float(np.load(f, allow_pickle=False))
    except Exception as e:
        rec["err"] = str(e)[:60]
        return rec
    tr = np.asarray(tr, float)
    if tr.ndim != 2 or tr.shape[1] != 3 or len(tr) < 60:
        rec["err"] = "shape"
        return rec
    T = len(tr)
    dur = T / hz
    xy = tr[:, :2]
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    travel = float(seg.sum())
    net = float(np.linalg.norm(xy[-1] - xy[0]))
    v = float(np.median(seg) * hz)
    straight = net / travel if travel > 1e-6 else 0.0
    z = tr[:, 2]
    rec.update(T=T, hz=hz, dur=round(dur, 1), v=round(v, 2),
               net=round(net, 1), straight=round(straight, 3),
               z_mean=round(float(z.mean()), 2), z_std=round(float(z.std()), 3))
    rec["pass"] = bool(3.0 <= dur <= 45.0 and 0.6 <= v <= 1.6 and straight >= 0.55
                       and 0.90 <= rec["z_mean"] <= 1.15 and rec["z_std"] < 0.08)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    done = set()
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            try:
                done.add(json.loads(line)["file"])
            except Exception:
                pass

    files = []
    for d in DIRS:
        files += sorted(str(p) for p in d.rglob("*_stageii.npz"))
    todo = [f for f in files if str(Path(f).relative_to(ROOT)) not in done]
    print(f"[INFO] total {len(files)} | cached {len(done)} | todo {len(todo)}")
    if not todo:
        return
    if args.limit:
        todo = todo[:args.limit]

    with mp.Pool(args.procs) as pool, open(CACHE, "a") as out:
        n_pass = 0
        for i, r in enumerate(pool.imap_unordered(trans_stats, todo, chunksize=16)):
            if r.get("pass"):
                n_pass += 1
            if i % 250 == 0:
                print(f"[INFO] {i}/{len(todo)} scanned, {n_pass} pass so far")
            out.write(json.dumps(r) + "\n")
        out.flush()
        print(f"[INFO] done: {len(todo)} this run, {n_pass} pass")


if __name__ == "__main__":
    main()
