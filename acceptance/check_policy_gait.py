#!/usr/bin/env python3
"""X1 trained-POLICY gait-quality acceptance gate (P7, v31).

The training gate (check_amp.py P1-P6) verifies process health and
survival/tracking, but NONE of it measures whether the policy's ARM/TORSO
motion matches the human-style reference — v29 passed 13/13 and still had
unnatural arm carriage ("swings the upper body via the waist"). This gate
closes the loop by measuring the POLICY rollout in the same joint space the
AMP discriminator saw.

Input: the npz gait log of sim2sim/mujoco_rollout.py (--log), containing
per-control-step q (hinge order), and meta {hinge_names, settle_steps, fell}.

Checks (window: post-settle frames; FAIL blocks the pipeline):
  P7f rollout validity   fell == False AND survived >= 8 s (else INCONCLUSIVE
                         -> FAIL: metrics on a fallen run are meaningless)
  P7a arm antiphase      corr(shoP_L, shoP_R) <= -0.50   (refs -0.92..-0.99)
  P7b elbow flexion      p95 of max(L,R) elbow_pitch in [8, 70] deg
                         (dead-straight < 8 = locked arms; > 70 = the v29
                         at-limit carriage; healthy walking ~20-60)
  P7c waist restraint    lumbar_yaw swing (p95-p5) <= 40 deg (PRIMARY refs
                         <= 32; 40 = first-shot headroom, tighten on v31 data)
  P7d arm-leg coupling   corr(shoP_L, hipP_R) >= +0.20  (refs +0.8..+0.99;
                         measured v27 -0.12, v29 -0.68; TARGET +0.45)
  P7e leg symmetry       hip swing ratio L/R in [0.7, 1.4]

Usage: python check_policy_gait.py --log rollout.npz [--json out.json]
Exit: 0 PASS, 1 FAIL, 2 usage/setup error.
"""
import argparse
import functools
import json
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

TH = {
    "P7a_anti": -0.50,
    "P7b_lo": 8.0, "P7b_hi": 70.0,
    "P7c_lumsw": 40.0,
    "P7d_coupling": 0.20,
    "P7e_ratio": (0.7, 1.4),
    "P7f_min_survive_s": 8.0,
}
# TARGET lines (informational, non-blocking) — refs measure anti -0.92..-0.99,
# coupling +0.8..+0.99; measured policies so far: v27 anti -0.94/coupling -0.12,
# v29e4b anti +0.91/coupling -0.68. First-shot FAIL bounds are looser than refs
# by design; tighten toward TARGET after the v31 policy is measured.
TG = {"T7a_anti": -0.70, "T7b_coupling": 0.45}
CONTROL_DT = 0.02


def corr(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-12 else 0.0


def rng(x):
    return float(np.percentile(x, 95) - np.percentile(x, 5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="mujoco_rollout.py --log npz")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    p = Path(args.log)
    if not p.exists():
        print(f"[FATAL] log not found: {p}")
        sys.exit(2)
    d = np.load(p, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    hinge = list(meta["hinge_names"])
    fell = bool(meta.get("fell", False))
    settle = int(meta.get("settle_steps", 0))

    q = np.asarray(d["q"], float)                 # (T, 29) hinge order
    T = len(q)
    s0 = min(settle + 25, T // 2)                 # skip settle + 0.5 s ramp
    qs = q[s0:]
    if len(qs) < 200:
        print(f"[FATAL] only {len(qs)} post-settle steps — rollout too short")
        sys.exit(2)
    idx = {n: i for i, n in enumerate(hinge)}     # BY NAME (never assume order)

    survived_s = (T - settle) * CONTROL_DT
    elbL = np.degrees(qs[:, idx["left_elbow_pitch_joint"]])
    elbR = np.degrees(qs[:, idx["right_elbow_pitch_joint"]])
    shoL = np.degrees(qs[:, idx["left_shoulder_pitch_joint"]])
    shoR = np.degrees(qs[:, idx["right_shoulder_pitch_joint"]])
    hipL = np.degrees(qs[:, idx["left_hip_pitch_joint"]])
    hipR = np.degrees(qs[:, idx["right_hip_pitch_joint"]])
    lumY = np.degrees(qs[:, idx["lumbar_yaw_joint"]])

    m = {
        "anti": corr(shoL, shoR),
        "elb_p95": float(np.percentile(np.maximum(elbL, elbR), 95)),
        "elb_mean": float(np.mean([elbL.mean(), elbR.mean()])),
        "lumsw": rng(lumY),
        "coupling": corr(shoL, hipR),
        "ratio": rng(hipL) / max(rng(hipR), 1e-9),
        "survived_s": survived_s,
        "fell": fell,
        "steps_used": len(qs),
    }

    results = {}

    def check(cid, ok, detail):
        results[cid] = {"pass": bool(ok), "detail": detail}
        print(f"  {'ok  ' if ok else 'FAIL'} {cid}: {detail}")

    print("=" * 72)
    cmd_s = " ".join(f"{c:g}" for c in meta.get("cmd", ["?"]))
    print(f"POLICY GAIT GATE  cmd=({cmd_s})  survived {survived_s:.1f}s  "
          f"window={len(qs)} steps @{1/CONTROL_DT:.0f}Hz")
    print("=" * 72)

    check("P7f_valid", (not fell) and survived_s >= TH["P7f_min_survive_s"],
          f"fell={fell}, survived {survived_s:.1f}s >= {TH['P7f_min_survive_s']}s")
    check("P7a_anti", m["anti"] <= TH["P7a_anti"],
          f"arm antiphase {m['anti']:+.2f} <= {TH['P7a_anti']} (refs -0.92..-0.99)")
    check("P7b_elbow", TH["P7b_lo"] <= m["elb_p95"] <= TH["P7b_hi"],
          f"elbow p95 {m['elb_p95']:.1f} in [{TH['P7b_lo']:.0f}, {TH['P7b_hi']:.0f}] deg "
          f"(mean {m['elb_mean']:.1f}; v29 carriage was ~106 at limit)")
    check("P7c_waist", m["lumsw"] <= TH["P7c_lumsw"],
          f"lumY swing {m['lumsw']:.1f} <= {TH['P7c_lumsw']} deg (PRIMARY refs <= 32)")
    check("P7d_coupling", m["coupling"] >= TH["P7d_coupling"],
          f"arm-vs-opposite-leg {m['coupling']:+.2f} >= +{TH['P7d_coupling']} (refs +0.8..+0.99)")
    lo, hi = TH["P7e_ratio"]
    check("P7e_legsym", lo <= m["ratio"] <= hi,
          f"hip swing ratio L/R {m['ratio']:.2f} in [{lo}, {hi}]")

    print("\n--- TARGET lines ---")
    for k, thr in TG.items():
        val = m["anti"] if k == "T7a_anti" else m["coupling"]
        hit = val <= thr if k == "T7a_anti" else val >= thr
        print(f"  {'HIT ' if hit else 'MISS'} {k}: {val:+.2f} vs {thr:+.2f}")

    fails = [k for k, v in results.items() if not v["pass"]]
    verdict = "PASS" if not fails else "FAIL"
    print("=" * 72)
    print(f"VERDICT: {verdict}  fails={fails}")
    print("=" * 72)

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"verdict": verdict, "fails": fails, "metrics": m,
             "results": results, "thresholds": {k: v for k, v in TH.items()}},
            indent=1))
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
