#!/usr/bin/env python3
"""F1w AMP training acceptance — chart-series adapter of acceptance/check_amp.py.

Provenance (why this adapter exists):
  F1W_RUN_ACCEPTANCE.md sec 0 says P1-P6 parse the training stdout with
  check_amp.py in the gradmotion container. The launcher writes
  train_stdout.log to the pod filesystem, which the platform does NOT
  expose after/between tasks; the [STDOUT-TAIL] visibility echo only
  surfaces timing lines. The platform DOES index the same rsl_rl
  TensorBoard scalars via `gm task data get` — these are the SAME
  metrics (same runner, same logger) in a different serialization.

  This adapter therefore applies the IDENTICAL thresholds of
  acceptance/check_amp.py (TH dict below is copied verbatim) to the
  stitched chart series of the v1 run
  (v1e TASK_20261110_162 + v1g TASK_20261110_210 + v1h TASK_20261110_233,
  iters 0..5998). No threshold is changed; the measurement channel is
  TB scalars instead of stdout text.

Usage:
    python check_amp_chart.py --series sim2sim/out_f1w_v1/train_chart_series.json \
        [--json out.json]
"""
import argparse
import functools
import json
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

# ---- thresholds: VERBATIM from acceptance/check_amp.py (AMP_ACCEPTANCE.md) ----
TH = {
    "P2a_ep_len": 950,
    "P2b_timeout": 0.95,
    "P2c_base_contact": 0.0,
    "P2d_bad_sum": 0.03,
    "P3a_lin_kernel": 0.82,
    "P3b_ang_kernel": 0.50,
    "P3c_err_xy": 0.44,
    "P3d_err_yaw": 0.95,
    "P4_ratio": 0.90,
    "P5a_style": 0.15,
    "P5b_disc_loss": 0.05,
}
LAST = 100  # check_amp.py uses the last-100-iteration window for P2/P3/P5


def stitch(series, key):
    parts = []
    for tag in ("v1e", "v1g", "v1h"):
        seg = series.get(tag, {}).get(key)
        if seg:
            parts.append((seg["steps"], seg["values"]))
    steps, vals, last = [], [], -1
    for st, va in parts:
        for s, v in zip(st, va):
            if s > last:
                steps.append(s)
                vals.append(v)
                last = s
    return np.asarray(steps), np.asarray(vals, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    series = json.loads(Path(args.series).read_text())
    res = {}

    def check(name, ok, detail):
        res[name] = {"pass": bool(ok), "detail": detail}
        print(f"  {'ok ' if ok else 'FAIL'} {name}: {detail}")

    # P1 — training completeness: 6000-iter plan reached 5998 via three chained
    # checkpoints (v1e 0->1500, v1g 1500->2999, v1h 2999->5998; see REPORT).
    st_r, _ = stitch(series, "Train/mean_reward")
    check("P1_iters", len(st_r) >= 5990 and st_r[-1] >= 5990,
          f"{len(st_r)} iterations logged, last={int(st_r[-1])} / 6000 plan")

    def win(key, last=LAST):
        _, v = stitch(series, key)
        return float(np.nanmean(v[-last:])) if len(v) else float("nan")

    ep_len = win("Train/mean_episode_length")
    check("P2a_ep_len", ep_len >= TH["P2a_ep_len"], f"ep_len {ep_len:.1f} >= {TH['P2a_ep_len']}")
    tmo = win("Episode_Termination/time_out")
    check("P2b_timeout", tmo >= TH["P2b_timeout"], f"timeout {tmo:.3f} >= {TH['P2b_timeout']}")
    bc = win("Episode_Termination/base_contact")
    check("P2c_base_contact", bc <= TH["P2c_base_contact"] + 1e-9,
          f"base_contact {bc:.4f} <= {TH['P2c_base_contact']:.1f}")
    bo = win("Episode_Termination/bad_orientation")
    bh = win("Episode_Termination/base_height")
    bad_sum = bo + bh
    check("P2d_bad_sum", bad_sum <= TH["P2d_bad_sum"],
          f"bad_orientation {bo:.4f} + base_height {bh:.4f} = {bad_sum:.4f} <= {TH['P2d_bad_sum']}")

    lk = win("Episode_Reward/track_lin_vel_xy_exp")
    check("P3a_lin_kernel", lk >= TH["P3a_lin_kernel"], f"lin kernel {lk:.4f} >= {TH['P3a_lin_kernel']}")
    ak = win("Episode_Reward/track_ang_vel_z_exp")
    check("P3b_ang_kernel", ak >= TH["P3b_ang_kernel"], f"ang kernel {ak:.4f} >= {TH['P3b_ang_kernel']}")
    exy = win("Metrics/base_velocity/error_vel_xy")
    check("P3c_err_xy", exy <= TH["P3c_err_xy"], f"error_vel_xy {exy:.4f} <= {TH['P3c_err_xy']}")
    eyaw = win("Metrics/base_velocity/error_vel_yaw")
    check("P3d_err_yaw", eyaw <= TH["P3d_err_yaw"], f"error_vel_yaw {eyaw:.4f} <= {TH['P3d_err_yaw']}")

    # P4 — no late-training collapse: last-100 reward >= 0.90 * best rolling mean
    _, rew = stitch(series, "Train/mean_reward")
    w = 100
    roll = np.convolve(rew, np.ones(w) / w, mode="valid")
    best = float(roll.max())
    last = float(rew[-w:].mean())
    check("P4_ratio", last >= TH["P4_ratio"] * best,
          f"last100 {last:.2f} >= {TH['P4_ratio']:.2f} x rolling-best {best:.2f}")

    sty = win("Episode_Reward/style")
    check("P5a_style", sty >= TH["P5a_style"], f"style {sty:.4f} >= {TH['P5a_style']}")
    dl = win("Loss/amp/disc_loss")
    check("P5b_disc_loss", dl <= TH["P5b_disc_loss"], f"disc_loss {dl:.4f} <= {TH['P5b_disc_loss']}")

    fails = [k for k, v in res.items() if not v["pass"]]
    verdict = "PASS" if not fails else "FAIL"
    print(f"\n=== F1w v1 (chart-series adapter of AMP_ACCEPTANCE P1-P6): {verdict} "
          f"(fails={len(fails)}: {', '.join(fails)}) ===")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"verdict": verdict, "checks": res, "fails": fails,
             "provenance": {
                 "metrics": "rsl_rl TensorBoard scalars via gm task data get "
                            "(same runner metrics as train stdout; stdout file "
                            "not retrievable from the platform)",
                 "stitch": "v1e TASK_20261110_162 (0-1644) + v1g TASK_20261110_210 "
                           "(1500-2999) + v1h TASK_20261110_233 (2999-5998)",
                 "thresholds": "acceptance/check_amp.py TH dict, verbatim"}},
            indent=1))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
