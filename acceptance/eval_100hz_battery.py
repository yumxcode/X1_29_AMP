#!/usr/bin/env python3
"""100 Hz eval battery: one command -> soup59c-comparable metric set.

Runs the SAME local gates the soup59c_50 champion was judged on, at the
policy's TRAINED control rate (--control-dt):
  1. five rollout logs   : walk10 / walk05 / back05 / stand / walkturn
  2. gait_metrics per log: G1 stability / G2 symmetry / G3 landing / K1 K2 H*
  3. form metrics JSON   : drift (walk10+walk05), arm amplitude, P7 proxies
  4. robustness sweep    : 12 variants x 5 scenarios, lat/lag time-matched

Usage:
  ./.venv39/bin/python acceptance/eval_100hz_battery.py \
      --npz acceptance/v61b_eval/model_10593.policy.npz \
      --tag v61b --control-dt 0.01 [--quick]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv39" / "bin" / "python"

SCENARIOS = [
    ("walk10", ["1.0", "0.0", "0.0"]),
    ("walk05", ["0.5", "0.0", "0.0"]),
    ("back05", ["-0.5", "0.0", "0.0"]),
    ("stand", ["0.0", "0.0", "0.0"]),
    ("walkturn", ["1.0", "0.0", "0.8"]),
]


def sh(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), **kw)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--control-dt", type=float, default=0.01)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="skip robustness sweep and back05/stand")
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--action-lpf", type=float, default=0.0,
                    help="v63: action low-pass Hz — must match the policy's "
                         "training X1_ACT_LPF_HZ (mirrored into every rollout "
                         "and the robustness sweep)")
    ap.add_argument("--action-lpf-order", type=int, default=2)
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROOT / "acceptance" / f"{args.tag}_eval"
    out.mkdir(parents=True, exist_ok=True)
    scens = [s for s in SCENARIOS if not args.quick or s[0] in ("walk10", "walk05")]

    logs = {}
    lpf = (["--action-lpf", str(args.action_lpf),
            "--action-lpf-order", str(args.action_lpf_order)]
           if args.action_lpf > 0 else [])
    for name, cmd in scens:
        npz = out / f"{args.tag}_{name}.npz"
        r = sh([str(PY), "sim2sim/mujoco_rollout.py", "--ckpt", args.npz,
                "--cmd"] + cmd + [
            "--duration", str(args.duration), "--settle", "0.5",
            "--control-dt", str(args.control_dt), "--log", str(npz)] + lpf)
        logs[name] = npz if npz.exists() else None
        if r.returncode == 0:
            tail = [l for l in r.stdout.splitlines() if "survived" in l or "v_xy" in l]
            print(f"[{name}] {' | '.join(tail)}", flush=True)

    # gait metrics per log
    gait = {}
    for name, npz in logs.items():
        if npz is None:
            continue
        r = sh([str(PY), "sim2sim/gait_metrics.py", str(npz)])
        gait[name] = r.stdout
        print(r.stdout, flush=True)

    # form metrics (drift/arm/K1K2) on walk logs
    walks = [str(npz) for npz in (logs.get("walk10"), logs.get("walk05")) if npz]
    if walks:
        fm = out / f"{args.tag}_form_metrics.json"
        # save_form_metrics.py CLI: <npz...> <out.json> (positional, in order)
        sh([str(PY), "acceptance/save_form_metrics.py"] + walks + [str(fm)])
        if fm.exists():
            print(f"[FORM] {fm}")

    # robustness sweep (time-matched lat/lag)
    if not args.quick:
        rr = out / f"{args.tag}_robustness_report.json"
        sh([str(PY), "sim2sim/robustness_sweep.py", "--ckpt", args.npz,
            "--out", str(rr), "--control-dt", str(args.control_dt),
            "--action-lpf", str(args.action_lpf),
            "--action-lpf-order", str(args.action_lpf_order)])
        if rr.exists():
            s = json.loads(rr.read_text())["summary"]
            print(f"[ROBUST] {s['pass']}/{s['cells']} cells verdict={s['verdict']}", flush=True)

    print(f"\n[BATTERY DONE] artifacts in {out}")


if __name__ == "__main__":
    main()
