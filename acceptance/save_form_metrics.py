#!/usr/bin/env python3
"""Persist drift + arm-amplitude + human-gait (K/H) metrics as JSON (audit).

Recomputes from the committed rollout npz: world-y drift (walk10), arm
swing amplitude (joint-space p95-p5 of shoulder pitch), P7g/P7h proxies,
and the v56+ human-gait gates K1/K2 (straight-knee) + H1/H2/H3 (heel-toe)
via sim2sim.gait_metrics.analyze.
Usage: python save_form_metrics.py <npz> <out.json> [--more npz2 ...]
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from sim2sim.gait_metrics import analyze as gait_analyze  # noqa: E402

SHO = ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint")


def metrics(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    hinge = list(meta["hinge_names"])
    q = np.asarray(d["q"], float)
    bp = np.asarray(d["base_pos"], float)
    settle = int(meta.get("settle_steps", 0))
    q, bp = q[settle:], bp[settle:]
    il, ir = hinge.index(SHO[0]), hinge.index(SHO[1])
    def amp(i):
        return float(np.degrees(np.percentile(q[:, i], 95) - np.percentile(q[:, i], 5)))
    def corr(i, j):
        a = q[:, i] - q[:, i].mean(); b = q[:, j] - q[:, j].mean()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    T = len(bp)
    return {
        "file": npz_path.split("/")[-1],
        "cmd": meta.get("cmd"),
        "n_steps": int(T),
        "arm_swing_amp_deg": {"L": amp(il), "R": amp(ir)},
        "arm_antiphase_corr": corr(il, ir),
        "world_y_drift_m": float(bp[-1, 1] - bp[0, 1]),
        "duration_s": float(T * 0.02),
    } | kh_fields(npz_path)


def kh_fields(npz_path):
    """K1/K2/H1/H2/H3 from the same npz via the G-gate analyzer (one source
    of truth; gates defined in gait_metrics.SPEC)."""
    try:
        R = gait_analyze(npz_path)
        kh = R["KH"]
        out = {}
        for name, d in kh.items():
            if isinstance(d, dict):
                out[name] = {k: d[k] for k in (
                    "k1_mid_deg", "k2_range_deg", "land_heel_first_frac",
                    "land_toe_first_frac", "launch_toe_off_frac", "n_events")}
        out["gates"] = {k: kh[k] for k in kh if k.endswith("_PASS")}
        return out
    except Exception as e:  # metric failure must not kill the whole report
        return {"kh_error": f"{type(e).__name__}: {e}"}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = args[-1]
    rows = [metrics(p) for p in args[:-1]]
    with open(out, "w") as f:
        json.dump({"note": "recomputed from rollout npz; p95-p5 shoulder pitch; world-y drift", "rows": rows}, f, indent=1)
    print(f"[SAVED] {out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
