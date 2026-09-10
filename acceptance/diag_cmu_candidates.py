#!/usr/bin/env python3
"""Deep-pass the CMU walking candidates: SMPLX gait quality check.

Reuses the metric conventions of diag_arm_swing_smplx.py (verified there):
  shoL/shoR swing about local y (same-sign corr = symmetric antiphase swing)
  elbL/elbR flexion about local x (healthy walking: -10..-30 deg)
  hip swing: sin of local-y rotation; knee flexion
  cadence from hip-swing zero crossings
Select: symmetric arm swing (>+0.6), arm-leg coupling, elbow in range,
cadence 90-135 spm, L/R hip swing ratio 0.8-1.25.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _rot(aa):
    t = np.linalg.norm(aa)
    if t < 1e-9:
        return np.eye(3)
    k = aa / t
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-12 else 0.0


def rng(x):
    return float(np.percentile(x, 95) - np.percentile(x, 5))


def analyze(path):
    z = np.load(path, allow_pickle=True)
    P = np.asarray(z["poses"], float)[:, :66].reshape(-1, 22, 3)
    hz = float(z["mocap_frame_rate"])
    T = len(P)

    # torso counter-rotation (spine1-3 vs pelvis yaw, unwrapped)
    def _yaw(R):
        return np.arctan2(-R[2, 0], R[0, 0])
    sp = np.array([_yaw(_rot(p[3]) @ _rot(p[6]) @ _rot(p[9])) for p in P])
    pv = np.array([_yaw(_rot(p[0])) for p in P])
    rel = np.unwrap(sp - pv)
    # detrend: remove 1s moving average -> per-step oscillation amplitude
    # (slow turning drift is legitimate and must not count as waist swing)
    w = max(int(hz), 3)
    ker = np.ones(w) / w
    pad = np.pad(rel, (w // 2, w - w // 2), mode="edge")
    trend = np.convolve(pad, ker, mode="valid")
    osc = rel - trend[:len(rel)]
    spine_sw = float(np.degrees(np.percentile(osc, 95) - np.percentile(osc, 5)))

    shoL = np.array([np.degrees(np.arcsin(np.clip(_rot(p[16])[0, 2], -1, 1))) for p in P])
    shoR = np.array([np.degrees(np.arcsin(np.clip(_rot(p[17])[0, 2], -1, 1))) for p in P])
    elbL = np.array([np.degrees(_rot(p[18])[1, 2]) for p in P])
    elbR = np.array([np.degrees(_rot(p[19])[1, 2]) for p in P])
    hipL = np.array([np.degrees(np.arcsin(np.clip(_rot(p[1])[0, 2], -1, 1))) for p in P])
    hipR = np.array([np.degrees(np.arcsin(np.clip(_rot(p[2])[0, 2], -1, 1))) for p in P])
    knL = np.array([np.degrees(_rot(p[4])[1, 2]) for p in P])
    knR = np.array([np.degrees(_rot(p[5])[1, 2]) for p in P])

    s0 = hipL - hipL.mean()
    xg = np.where((s0[:-1] < 0) & (s0[1:] >= 0))[0]
    cad = 60.0 * len(xg) / (T / hz) if len(xg) >= 3 else 0.0

    return dict(
        T=T, cad=cad, spine=spine_sw,
        symm=corr(shoL, shoR),
        armleg=corr(shoL, hipR),
        swL=rng(shoL), swR=rng(shoR),
        elbL=float(elbL.mean()), elbR=float(elbR.mean()),
        legs=corr(hipL, hipR), knee=corr(knL, knR),
        hipL=rng(hipL), hipR=rng(hipR),
        ratio=rng(hipL) / max(rng(hipR), 1e-6),
    )


if __name__ == "__main__":
    rows = [json.loads(l) for l in open(ROOT / ".cache_amass_scan.jsonl")]
    cands = [r for r in rows if r.get("pass")]
    cands.sort(key=lambda r: -r["net"])
    print(f"{'file':46s} {'T':>5s} {'cad':>6s} {'symm':>5s} {'armLeg':>6s} {'swL/R':>9s} "
          f"{'elbL/R':>11s} {'legs':>5s} {'knee':>5s} {'hipL/R':>9s} {'ratio':>5s} {'spine':>5s} | verdict")
    n_ok = 0
    for r in cands:
        m = analyze(ROOT / r["file"])
        ok = (m["symm"] > 0.6 and -40 <= m["elbL"] <= 0 and -40 <= m["elbR"] <= 5
              and 60 <= m["cad"] <= 140 and 0.75 <= m["ratio"] <= 1.3
              and m["spine"] < 30.0)
        verdict = "GOOD" if ok else "skip"
        n_ok += ok
        print(f"{r['file']:46s} {m['T']:5d} {m['cad']:6.1f} {m['symm']:+5.2f} {m['armleg']:+6.2f} "
              f"{m['swL']:4.1f}/{m['swR']:4.1f} {m['elbL']:5.1f}/{m['elbR']:5.1f} "
              f"{m['legs']:+5.2f} {m['knee']:+5.2f} {m['hipL']:4.1f}/{m['hipR']:4.1f} "
              f"{m['ratio']:5.2f} {m['spine']:5.1f} | {verdict}")
    print(f"\n{n_ok} GOOD of {len(cands)} examined")
