#!/usr/bin/env python3
"""Time-warp F1w milestone clips to the X1-side anchored speed (sec 5.1
time retiming, finally implemented for the WALK band where the speed
mismatch is modest (1.17-1.92x) and geometrically feasible).

For each clip: beta = 1 / rho, rho = v_x1_side / v_f1w_current (< 1).
Slower playback by beta: sample source at t' * rho, T' = T / rho frames,
fps unchanged. Effects (all in the desired direction):
  root speed  -> v * rho = X1-side anchored speed  (C' PASS)
  joint vel   -> jv * rho                           (R4 margin improves)
  cadence     -> cadence * rho                      (F' band [0.7, 1.3])
  stride      -> UNCHANGED (spatial; sec 6 note)
  flight frac -> ~invariant (all durations scale)
After the warp: re-run fix_root_xy_v72_f1w (re-anchor) then C'/R4 audit.
"""
import argparse
import functools
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)
HERE = Path(__file__).parent.resolve()

# X1-side anchored speeds (x1_lab_f1w_final_src, v72 chain) measured 2026-11-11
X1_SPEED = {"07_01": 0.89, "103_07": 0.82, "35_02": 0.73, "38_02": 0.90}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    report = {}
    for p in sorted(Path(args.src).glob("*.pkl")):
        name = p.stem
        d = pickle.load(open(p, "rb"))
        rp = np.asarray(d["root_pos"], float)
        q = np.asarray(d["dof_pos"], float)
        rr = np.asarray(d["root_rot"], float)
        kb = np.asarray(d["key_body_pos"], float)
        fps = float(d["fps"])
        T = len(rp)
        v = float(np.median(np.linalg.norm(np.diff(rp[:, :2], axis=0), axis=1) * fps))
        tgt = X1_SPEED.get(name)
        if tgt is None:
            print(f"[SKIP] {name}: no X1-side speed reference")
            continue
        rho = tgt / v
        if rho > 1.05:
            print(f"[WARN] {name}: target FASTER than current (rho={rho:.3f}) — warp skipped")
            rho = 1.0
        Tw = int(round(T / rho))
        src_t = np.arange(Tw) * rho
        src_t = np.clip(src_t, 0, T - 1)

        def resample(arr):
            flat = arr.reshape(T, -1)
            out = np.stack([np.interp(src_t, np.arange(T), flat[:, k]) for k in range(flat.shape[1])], axis=1)
            return out.reshape(Tw, *arr.shape[1:])

        # quaternions: interpolate then renormalize
        rr_w = resample(rr)
        rr_w /= np.linalg.norm(rr_w, axis=1, keepdims=True)
        out = dict(d)
        out["root_pos"] = resample(rp)
        out["root_rot"] = rr_w.astype(np.float32)
        out["dof_pos"] = resample(q)
        out["key_body_pos"] = resample(kb).astype(np.float32)
        pickle.dump(out, open(os.path.join(args.dst, p.name), "wb"))
        v_new = float(np.median(np.linalg.norm(np.diff(out["root_pos"][:, :2], axis=0), axis=1) * fps))
        jv = np.abs(np.diff(out["dof_pos"], axis=0) * fps).max(axis=1)
        report[name] = {"rho": rho, "v_before": v, "v_after": v_new, "target": tgt,
                        "jv_p99": float(np.percentile(jv, 99)), "T": T, "T_new": Tw}
        print(f"[WARP] {name:10s} rho={rho:.3f}  v {v:.2f} -> {v_new:.2f} (target {tgt:.2f})"
              f"  T {T}->{Tw}  jv p99 {np.percentile(jv, 99):.1f}")
    json.dump(report, open(os.path.join(args.dst, "_warp_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
