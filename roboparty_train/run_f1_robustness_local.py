#!/usr/bin/env python3
"""F1 compact robustness sweep (X1-standard variants, local 3.1.1 shim).

Cells: {fwd10, stand, turn} x {nominal, lat1, noise1x, mass+15%, push0.75},
1 seed each = 15 cells. Reports fell/alive per cell + summary (X1 gate
reference: robustness >= 48/60 on the full 60-cell battery; this compact
15-cell view is indicative only).
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim2sim"))
sys.path.insert(0, str(ROOT / "roboparty_train"))

from run_f1_sim2sim_local import local_build_model  # noqa: E402
import mujoco_rollout as MR  # noqa: E402

SCEN = {"fwd10": ["1.0", "0.0", "0.0"], "stand": ["0.0", "0.0", "0.0"],
        "turn": ["1.0", "0.0", "0.8"]}
VAR = {"nominal": [], "lat1": ["--latency-steps", "1"],
       "noise1x": ["--obs-noise", "1.0"], "mass+15%": ["--torso-mass-scale", "1.15"],
       "push0.75": ["--push-mag", "0.75"]}


def run_cell(ckpt, cmd, extra, tag):
    js = str(ROOT / "acceptance/f1_v1" / f"_rob_{tag}.json")
    MR.build_model = local_build_model
    sys.argv = ["mujoco_rollout.py", "--ckpt", ckpt, "--robot", "f1",
                "--cmd"] + cmd + ["--duration", "12", "--settle", "0.5",
                                  "--repo-root", str(ROOT), "--json", js, "--seed", "0"] + extra
    print(f"[CELL] {tag}", flush=True)
    try:
        MR.main()
    except SystemExit:
        pass
    if os.path.exists(js):
        d = json.loads(Path(js).read_text())
        Path(js).unlink()
        return {"fell": d.get("fell", True), "dist": d.get("distance", 0.0)}
    return {"fell": True, "dist": 0.0}


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "acceptance/f1_v1/policy_5498.npz")
    out = {}
    ok = 0
    for sc, cmd in SCEN.items():
        for vn, extra in VAR.items():
            r = run_cell(ckpt, cmd, extra, f"{sc}_{vn}")
            out[f"{sc}/{vn}"] = r
            ok += 0 if r["fell"] else 1
            print(f"  -> {'ALIVE' if not r['fell'] else 'FELL'} dist={r['dist']:.2f}m", flush=True)
    total = len(SCEN) * len(VAR)
    report = {"cells": out, "alive": ok, "total": total}
    p = ROOT / "acceptance/f1_v1/v1h_5498_robustness_compact.json"
    p.write_text(json.dumps(report, indent=2))
    print(f"\n[SUMMARY] {ok}/{total} alive -> {p}")


if __name__ == "__main__":
    main()
