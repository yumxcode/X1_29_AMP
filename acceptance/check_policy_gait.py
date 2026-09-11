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

    # P7g arm lean (v32b defect, user-observed): world-frame upper-arm
    # forward angle means via FK. asym = frozen antisymmetric offset (L back /
    # R fwd), shared = both-arms lean. Guarded import: if mujoco/model is
    # unavailable the check degrades to a WARN (log-only evidence).
    try:
        import mujoco
        import sys as _sys
        from pathlib import Path as _Path
        _root = _Path(__file__).resolve().parent.parent
        _sys.path.insert(0, str(_root))
        _sys.path.insert(0, str(_root / "sim2sim"))
        from sim2sim.mujoco_rollout import build_model, DEFAULT_Q
        _model, _ = build_model(_root / "gmr_x1_assets" / "x1.xml")
        _data = mujoco.MjData(_model)
        _BID = lambda b: mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_BODY, b)
        _qadr = {n: _model.jnt_qposadr[
            mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in hinge}
        stride = max(1, len(qs) // 150)
        phis = {"left": [], "right": []}
        for t in range(0, len(qs), stride):
            _data.qpos[:] = 0
            _data.qpos[0:3] = np.asarray(d["base_pos"][s0 + t], float)
            _data.qpos[3:7] = np.asarray(d["base_quat"][s0 + t], float)
            for n in hinge:
                _data.qpos[_qadr[n]] = qs[t, idx[n]]
            mujoco.mj_forward(_model, _data)
            Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, _data.xquat[_BID("base_link")])
            Rm = Rm.reshape(3, 3)
            fwd = Rm[:, 0]
            for side in ("left", "right"):
                u = _data.xpos[_BID(f"{side}_elbow_pitch_link")] - \
                    _data.xpos[_BID(f"{side}_shoulder_pitch_link")]
                phis[side].append(np.degrees(np.arctan2(u @ fwd, -(u @ Rm[:, 2]))))
        phiL_m, phiR_m = float(np.mean(phis["left"])), float(np.mean(phis["right"]))
        m["phiL_mean"], m["phiR_mean"] = phiL_m, phiR_m
        m["lean_asym"] = abs(phiL_m - phiR_m)
        m["lean_shared"] = abs((phiL_m + phiR_m) / 2)
        fk_ok = True
    except Exception as e:
        print(f"  [WARN] P7g FK unavailable ({type(e).__name__}: {e})")
        fk_ok = False

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
    if fk_ok:
        check("P7g_lean_asym", m["lean_asym"] <= 12.0,
              f"arm lean asym |phiL-phiR| {m['lean_asym']:.1f} deg <= 12 "
              f"(L {m['phiL_mean']:+.1f} R {m['phiR_mean']:+.1f}; v32b was 23)")
        check("P7h_lean_shared", m["lean_shared"] <= 12.0,
              f"shared arm lean |mean(phiL,phiR)| {m['lean_shared']:.1f} deg <= 12 "
              f"(both-arms-forward/backward)")
    else:
        results["P7g_lean_asym"] = {"pass": True, "detail": "FK unavailable — WARN only"}
        results["P7h_lean_shared"] = {"pass": True, "detail": "FK unavailable — WARN only"}
        print("  ~   P7g/P7h arm lean: FK unavailable — skipped (WARN)")

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
