"""Regenerate the human-reference rhythm measurement (evidence artifact).

Writes acceptance/evidence/rhythm_reference.json: per-clip per-foot
cadence/swing/duty/step-length from the demo dataset (the SAME clips the
discriminator imitates), plus the walk-family aggregate the R-gates in
acceptance/rhythm_gates.py derive their thresholds from. Re-runnable:
    ./.venv39/bin/python acceptance/rhythm_reference.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "roboparty_train" / "robolab" / "data" / "motions" / "x1_lab_v32"
sys.path.insert(0, str(ROOT / "acceptance"))
from rhythm_gates import measure_foot  # same detection axis as the gate

CLIPS = [
    ("0000_treadmill_norm", False), ("0002_treadmill_slow", False),
    ("0003_treadmill_jog", False), ("0005_normal_walk1", False),
    ("0007_normal_walk3", False), ("0008_normal_walk4", False),
    ("0009_normal_jog1", False), ("0026_circle_walk", False),
    ("36_01", True), ("36_11", True), ("103_07", True),
]

rows = []
for name, overground in CLIPS:
    p = D / f"{name}.pkl"
    if not p.exists():
        continue
    c = pickle.load(open(p, "rb"))
    fps = float(c["fps"]); dt = 1.0 / fps
    kb = np.asarray(c["key_body_pos"])
    root = np.asarray(c["root_pos"])
    v_fwd = float(np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1).mean() / dt)
    row = {"clip": name, "fps": fps, "dur_s": len(root) / fps, "v_fwd": v_fwd}
    for side, idx in (("L", 2), ("R", 3)):
        m = measure_foot(kb[:, idx, 2], dt,
                         xy=kb[:, idx, :2] if overground else None)
        row[side] = m
    rows.append(row)

walk = [r for r in rows if "jog" not in r["clip"]]
agg = {}
for r in walk:
    for E in (r["L"], r["R"]):
        if E:
            for k in ("cadence_hz", "swing_s_med", "duty"):
                agg.setdefault(k, []).append(E[k])
            if E and "step_len_m_med" in E:
                agg.setdefault("step_len", []).append(
                    [r["v_fwd"], E["step_len_m_med"]])

summary = {}
for k, v in agg.items():
    if k == "step_len":
        summary[k] = [ [round(a,2), round(b,2)] for a,b in v ]
    else:
        a = np.array(v)
        summary[k] = {"median": float(np.median(a)),
                      "p15": float(np.percentile(a, 15)),
                      "p85": float(np.percentile(a, 85)), "n": len(a)}

out = {"measured": "2026-09-21", "detection": "lift>25mm above per-clip per-foot min; "
       "ankle-roll origin z (demo key_body); stance gaps<60ms glued; swings<100ms dropped",
       "clips": rows, "walk_family_aggregate": summary}
dst = ROOT / "acceptance" / "evidence" / "rhythm_reference.json"
dst.write_text(json.dumps(out, indent=1))
print(json.dumps(summary, indent=1))
print(f"[JSON] {dst}")
