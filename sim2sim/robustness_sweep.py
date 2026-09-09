#!/usr/bin/env python3
"""sim2real readiness sweep: how fragile is the v28d policy to realistic
hardware imperfections (sensor noise, obs/action latency, payload, pushes)?

Matrix (defaults, overridable):
  scenarios : 1.0 fwd / 0.5 fwd / turn 0.8 / stand / -0.5 backward
  variants  : nominal | noise1x | noise2x | lat1 | lat2 | lag1 | lag2 |
              mass+15% | mass-15% | push0.75 | push1.0 | worst-combo
  seeds     : 3 per cell (fall rate over 12 s commanded)

PASS gate (per cell): 0 falls across seeds AND (walking: mean vxy err within
+0.15 m/s of nominal) AND (stand: drift <= 0.75 m).

Usage:
  ./.venv39/bin/python sim2sim/robustness_sweep.py --ckpt <policy.npz> \
      [--out acceptance/v28_eval/robustness_report.json] [--quick]
(env: .venv39 = system py3.9 + mujoco 3.1.6 + numpy<2; .venv_test conda base broke)
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCENARIOS = {
    "fwd10": ["1.0", "0.0", "0.0"],
    "fwd05": ["0.5", "0.0", "0.0"],
    "turn": ["1.0", "0.0", "0.8"],
    "stand": ["0.0", "0.0", "0.0"],
    "back05": ["-0.5", "0.0", "0.0"],
}

VARIANTS = {
    "nominal": [],
    "noise1x": ["--obs-noise", "1.0"],
    "noise2x": ["--obs-noise", "2.0"],
    "lat1": ["--latency-steps", "1"],
    "lat2": ["--latency-steps", "2"],
    "lag1": ["--action-lag", "1"],
    "lag2": ["--action-lag", "2"],
    "mass+15%": ["--torso-mass-scale", "1.15"],
    "mass-15%": ["--torso-mass-scale", "0.85"],
    "push0.75": ["--push-mag", "0.75"],
    "push1.0": ["--push-mag", "1.0"],
    "worst": ["--obs-noise", "1.0", "--latency-steps", "1",
              "--action-lag", "1", "--torso-mass-scale", "1.15",
              "--push-mag", "0.5"],
}


def run_cell(ckpt, cmd, variant_args, seed, duration=12):
    import os, uuid
    # MUST be unique per concurrent worker: threads share pid, so pid+seed
    # still collides across parallel cells (observed clobber + unlink race).
    js = Path(f"/tmp/_rob_{os.getpid()}_{uuid.uuid4().hex[:10]}.json")
    cmdl = [str(ROOT / ".venv39/bin/python"), str(Path(__file__).parent / "mujoco_rollout.py"),
            "--ckpt", ckpt, "--cmd"] + cmd + [
           "--duration", str(duration), "--settle", "0.5",
           "--json", str(js), "--seed", str(seed)] + variant_args
    env = dict(os.environ)
    # 1 thread per worker: parallelism comes from --jobs; without this the
    # N-way process x M-thread BLAS oversubscription collapses throughput ~20x
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[v] = "1"
    r = subprocess.run(cmdl, capture_output=True, text=True, cwd=str(ROOT), env=env)
    if not js.exists():
        return {"error": r.stdout[-400:] + r.stderr[-400:], "fell": True}
    d = json.loads(js.read_text())
    js.unlink()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="acceptance/v28_eval/robustness_report.json")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="seeds=1, nominal+lat2+worst only")
    ap.add_argument("--compact", action="store_true",
                    help="7 variants x 3 scenarios (fwd10/turn/stand)")
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()

    from concurrent.futures import ThreadPoolExecutor
    seeds = list(range(args.seeds))
    if args.quick:
        variants = {k: VARIANTS[k] for k in ("nominal", "lat2", "worst", "noise2x", "push1.0")}
        SCENARIOS.clear(); SCENARIOS.update({"fwd10": ["1.0", "0.0", "0.0"], "stand": ["0.0", "0.0", "0.0"]})
    elif args.compact:
        variants = {k: VARIANTS[k] for k in
                    ("nominal", "noise1x", "lat1", "lag1", "mass+15%", "push1.0", "worst")}
        SCENARIOS.clear(); SCENARIOS.update({"fwd10": ["1.0", "0.0", "0.0"],
                                             "turn": ["1.0", "0.0", "0.8"],
                                             "stand": ["0.0", "0.0", "0.0"]})
    else:
        variants = VARIANTS
    nominal = {}
    results = []
    n_cells = 0
    def one_cell(scen, vn, va):
        cell = {"scenario": scen, "variant": vn, "runs": []}
        for s in seeds:
            d = run_cell(args.ckpt, SCENARIOS[scen], va, s)
            cell["runs"].append({"seed": s, "fell": d.get("fell"),
                                 "survived_s": d.get("survived_s"),
                                 "vxy_err": d.get("mean_vxy_err"),
                                 "distance_m": d.get("distance_m")})
        return cell
    def summarize(cell):
        falls = sum(1 for r in cell["runs"] if r["fell"])
        vxy = [r["vxy_err"] for r in cell["runs"] if r["vxy_err"] is not None]
        dist = [r["distance_m"] for r in cell["runs"] if r["distance_m"] is not None]
        cell["fall_rate"] = falls / len(seeds)
        cell["mean_vxy_err"] = sum(vxy) / len(vxy) if vxy else None
        cell["mean_distance"] = sum(dist) / len(dist) if dist else None
        if cell["variant"] == "nominal":
            nominal[cell["scenario"]] = cell
        return cell

    from concurrent.futures import as_completed
    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {(scen, vn): ex.submit(one_cell, scen, vn, va)
                for scen in SCENARIOS for vn, va in variants.items()}
        cells = {}
        for f in as_completed(futs.values()):
            k = [kk for kk, ff in futs.items() if ff is f][0]
            cell = summarize(f.result())
            cells[k] = cell
            done += 1
            print(f"[{done:3d}/{len(futs)}] {cell['scenario']:7s} {cell['variant']:9s} "
                  f"falls={int(cell['fall_rate']*len(seeds))}/{len(seeds)} "
                  f"vxy={cell['mean_vxy_err']}", flush=True)
    results = [cells[(scen, vn)] for scen in SCENARIOS for vn in variants]
    for scen in SCENARIOS:
        for vn in variants:
            c = cells[(scen, vn)]
            n_cells += 1

    # gating
    for c in results:
        ok = c["fall_rate"] == 0.0
        nom = nominal.get(c["scenario"], {})
        if ok and c["scenario"] != "stand" and c["mean_vxy_err"] is not None \
                and nom.get("mean_vxy_err") is not None:
            ok = c["mean_vxy_err"] <= nom["mean_vxy_err"] + 0.15
        if ok and c["scenario"] == "stand" and c["mean_distance"] is not None:
            ok = c["mean_distance"] <= 0.75
        c["PASS"] = bool(ok)

    total = len(results)
    npass = sum(1 for c in results if c["PASS"])
    by_variant = {}
    for c in results:
        by_variant.setdefault(c["variant"], []).append(c["PASS"])
    summary = {
        "ckpt": args.ckpt, "seeds": seeds, "cells": total, "pass": npass,
        "verdict": "PASS" if npass == total else "FAIL",
        "by_variant_pass": {k: f"{sum(v)}/{len(v)}" for k, v in by_variant.items()},
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "cells": results}, indent=1))
    print(f"\n[VERDICT] {summary['verdict']}  {npass}/{total} cells")
    for k, v in summary["by_variant_pass"].items():
        print(f"    {k:9s} {v}")
    print(f"[JSON] {args.out}")


if __name__ == "__main__":
    main()
