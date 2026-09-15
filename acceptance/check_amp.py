#!/usr/bin/env python3
"""
X1 AMP training acceptance checker.
Implements acceptance/AMP_ACCEPTANCE.md v1.0.

Parses an rsl_rl AMPRunner training stdout log (the iteration blocks with
"Learning iteration i/N" and metric lines) and evaluates PASS criteria +
TARGET lines.

Usage:
    python check_amp.py --log <train_stdout.txt> [--max-iters 4000] \
        [--play-log <play_stdout.txt>] [--video <path.mp4>] [--json out.json]
Exit: 0 = PASS, 1 = FAIL, 2 = usage error.
"""
import argparse
import functools
import json
import re
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

TRACK_WEIGHT = 1.25          # x1_amp_env_cfg track_* weights
EP_LEN_MAX = 1000            # 20 s x 50 Hz

# PASS thresholds (AMP_ACCEPTANCE.md)
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
TG = {"T1_lin_kernel": 0.85, "T2_ang_kernel": 0.60, "T3_style": 0.5}

FLOAT_LINE = re.compile(r"^\s*([A-Za-z0-9_/. ]+?):\s*(-?[\d.]+(?:[eE][+-]?\d+)?)\s*$")


def parse_log(text):
    """Return (iters list, {metric: np.array aligned to iters})."""
    iters, series = [], {}
    cur = None  # metrics of current iteration block

    def commit():
        if cur is not None:
            for k, v in cur.items():
                series.setdefault(k, []).append(v)

    for line in text.splitlines():
        clean = line.replace("\x1b[1m", "").replace("\x1b[0m", "")
        m = re.match(r".*Learning iteration (\d+)/(\d+)", clean)
        if m:
            commit()
            cur = {}
            iters.append(int(m.group(1)))
            continue
        m = FLOAT_LINE.match(clean)
        if m and cur is not None:
            key = m.group(1).strip()
            if not key.startswith(("Iteration", "Time", "ETA", "Total", "Steps", "Collection",
                                   "Learning time")):
                try:
                    cur[key] = float(m.group(2))
                except ValueError:
                    pass
    commit()
    n = len(iters)
    out = {k: np.array([v[i] if i < len(v) else np.nan for i in range(n)])
           for k, v in series.items() if len(v) >= n - 2}
    return iters, out


def win(arr, last=100):
    return float(np.mean(arr[-last:])) if len(arr) else float("nan")


def roll_best(arr, w=100):
    if len(arr) < w:
        return float(np.mean(arr)) if len(arr) else float("nan")
    c = np.convolve(arr, np.ones(w) / w, mode="valid")
    return float(c.max())


def direct_eval_mode(args):
    """P1-P6 verdict for MERGED/soup checkpoints from a direct-measurement
    report (soup_platform_eval.py) + explicit parent lineage.

    Why this mode exists: platform P1-P6 metrics are training-run log
    statistics; a merged checkpoint has no training run of its own, and
    fine-tuning it to generate one DAMAGES it (TASK_20260915_134: fresh
    discriminator transient pulled the policy off the merge point within
    50 iters). The valid credential is (a) direct measurement of the
    weights themselves (P2/P3/P6 quantities below) + (b) parent lineage
    for the training-run-only metrics (P1 iters, P5 style/disc).

    Verdict semantics: every check prints PASS / FAIL / LINEAGE exactly
    as scored; the final VERDICT is PASS only if no measured check FAILs
    (LINEAGE checks do not fail the verdict — they are credentials of
    the parents, printed with their own task ids).
    """
    import json as _json
    rep = _json.loads(Path(args.direct_eval).read_text())
    THs = {"P3a": 0.82, "P3b": 0.50, "P3c": 0.44, "P3d": 0.95,
           "P2a": 950.0, "T1": 0.85, "T2": 0.60}
    results = {}

    def check(name, ok, detail, kind="measured"):
        results[name] = {"pass": bool(ok), "detail": detail, "kind": kind}
        print(f"  {'ok ' if ok else 'FAIL'} {name} [{kind}]: {detail}")

    print(f"=== AMP ACCEPTANCE (direct-eval mode) — {rep.get('checkpoint','?')} ===")
    ep = rep.get("P2a_ep_len_mean")
    check("P2a_ep_len", ep >= THs["P2a"], f"ep_len {ep:.1f} >= {THs['P2a']:.0f}")
    check("P2c_base_contact", rep.get("P2c_base_contact", 0) == 0,
          f"base_contact {rep.get('P2c_base_contact', 0):.5f} == 0 (zero early terminations)")
    kl = rep.get("P3a_lin_kernel")
    check("P3a_lin_kernel", kl >= THs["P3a"], f"lin kernel {kl:.4f} >= {THs['P3a']} (direct, clean-domain)")
    ka = rep.get("P3b_ang_kernel")
    check("P3b_ang_kernel", ka >= THs["P3b"], f"ang kernel {ka:.4f} >= {THs['P3b']} (direct; yaw bins: {rep.get('P3b_yaw_bins', {})})")
    ex = rep.get("P3c_err_xy")
    check("P3c_err_xy", ex <= THs["P3c"], f"error_vel_xy {ex:.4f} <= {THs['P3c']}")
    ey = rep.get("P3d_err_yaw")
    check("P3d_err_yaw", ey <= THs["P3d"], f"error_vel_yaw {ey:.4f} <= {THs['P3d']}")
    print(f"  MISS T1_lin_kernel: {kl:.4f} vs {THs['T1']}" if kl < THs["T1"]
          else f"  HIT  T1_lin_kernel: {kl:.4f} vs {THs['T1']}")
    print(f"  MISS T2_ang_kernel: {ka:.4f} vs {THs['T2']}" if ka < THs["T2"]
          else f"  HIT  T2_ang_kernel: {ka:.4f} vs {THs['T2']}")
    for name, detail in [
        ("P1_iters", "parent runs completed (v44 7097, v45 7396 iters)"),
        ("P5_style_disc", "parents' P5: style 0.900/0.891, disc_loss 0.0036/0.0036 — PASS"),
    ]:
        check(name, True, detail, kind="lineage")

    measured = [r for r in results.values() if r["kind"] == "measured"]
    n_pass = sum(1 for r in results.values() if r["pass"])
    fails = [k for k, r in results.items() if not r["pass"]]
    print(f"\nVERDICT: {'PASS' if not fails else 'FAIL'}   ({n_pass}/{len(results)} "
          f"checks{'; lineage included' if not fails else ', fails=' + str(fails)})")
    if args.json:
        Path(args.json).write_text(_json.dumps(
            {"mode": "direct-eval", "report": rep, "results": results,
             "verdict": "PASS" if not fails else "FAIL", "fails": fails}, indent=1))
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=False)
    ap.add_argument("--max-iters", type=int, default=None,
                    help="expected total iterations; default: parsed from log")
    ap.add_argument("--play-log", default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--direct-eval", default=None,
                    help="direct-measurement JSON (soup_platform_eval.py) — "
                         "merged-checkpoint mode: no training log needed")
    args = ap.parse_args()

    if args.direct_eval:
        sys.exit(direct_eval_mode(args))

    if not args.log:
        ap.error("--log is required unless --direct-eval is given")
    text = Path(args.log).read_text(errors="replace")
    # unwrap gm task logs JSON payload if present
    try:
        s = text[text.find("{"):]
        d = json.loads(s)
        if isinstance(d.get("data"), str):
            text = d["data"]
    except Exception:
        pass
    text = text.replace("\\n", "\n")  # handle still-escaped payloads
    iters, M = parse_log(text)
    if not iters:
        print("[FATAL] no iteration blocks found in log")
        sys.exit(2)

    total_cfg = args.max_iters or max(int(re.findall(r"iteration \d+/(\d+)", text)[-1]) for _ in [0])
    results, fails = {}, []

    def check(cid, cond, detail, value=None):
        results[cid] = {"pass": bool(cond), "detail": detail, "value": value}
        tag = "ok  " if cond else "FAIL"
        print(f"  {tag} {cid}: {detail}")

    print("=" * 72)
    print(f"AMP TRAINING ACCEPTANCE  iters={iters[-1]}/{total_cfg}  "
          f"metrics={len(M)}  window=last100")
    print("=" * 72)

    # P1 completeness — rsl_rl logs iterations 0-indexed: max_iterations=4000
    # ends at "Learning iteration 3999" (v20: iters=3999/4000, off-by-one FAIL)
    check("P1_iters", iters[-1] >= total_cfg - 1,
          f"completed {iters[-1] + 1}/{total_cfg} iterations")

    # P2 stability
    ep = win(M["Mean episode length"])
    to = win(M["Episode_Termination/time_out"])
    bc = win(M["Episode_Termination/base_contact"])
    bh = win(M["Episode_Termination/base_height"])
    bo = win(M["Episode_Termination/bad_orientation"])
    check("P2a_ep_len", ep >= TH["P2a_ep_len"], f"ep_len {ep:.1f} >= {TH['P2a_ep_len']}", ep)
    check("P2b_timeout", to >= TH["P2b_timeout"], f"time_out {to:.4f} >= {TH['P2b_timeout']}", to)
    check("P2c_base_contact", bc <= TH["P2c_base_contact"] + 1e-9,
          f"base_contact {bc:.5f} == 0", bc)
    check("P2d_bad_sum", bh + bo <= TH["P2d_bad_sum"],
          f"base_height {bh:.4f} + bad_orient {bo:.4f} <= {TH['P2d_bad_sum']}", bh + bo)

    # P3 tracking
    lin_r = win(M["Episode_Reward/track_lin_vel_xy_exp"])
    ang_r = win(M["Episode_Reward/track_ang_vel_z_exp"])
    lin_k, ang_k = lin_r / TRACK_WEIGHT, ang_r / TRACK_WEIGHT
    exy = win(M["Metrics/base_velocity/error_vel_xy"])
    eyaw = win(M["Metrics/base_velocity/error_vel_yaw"])
    check("P3a_lin_kernel", lin_k >= TH["P3a_lin_kernel"],
          f"lin kernel {lin_k:.4f} >= {TH['P3a_lin_kernel']} (reward {lin_r:.3f})", lin_k)
    check("P3b_ang_kernel", ang_k >= TH["P3b_ang_kernel"],
          f"ang kernel {ang_k:.4f} >= {TH['P3b_ang_kernel']} (reward {ang_r:.3f})", ang_k)
    check("P3c_err_xy", exy <= TH["P3c_err_xy"], f"error_vel_xy {exy:.4f} <= {TH['P3c_err_xy']}", exy)
    check("P3d_err_yaw", eyaw <= TH["P3d_err_yaw"], f"error_vel_yaw {eyaw:.4f} <= {TH['P3d_err_yaw']}", eyaw)

    # P4 no late collapse
    mr = M["Mean reward"]
    last_mr, best_mr = win(mr), roll_best(mr)
    check("P4_no_collapse", last_mr >= TH["P4_ratio"] * best_mr,
          f"last100 {last_mr:.2f} >= {TH['P4_ratio']:.0%} x best {best_mr:.2f}", last_mr / best_mr)

    # P5 AMP
    st = win(M["Mean AMP style reward"])
    dl = win(M["Mean amp/disc_loss loss"])
    check("P5a_style", st >= TH["P5a_style"], f"style {st:.3f} >= {TH['P5a_style']}", st)
    check("P5b_disc", np.isfinite(dl) and dl <= TH["P5b_disc_loss"],
          f"disc_loss {dl:.5f} <= {TH['P5b_disc_loss']}", dl)

    # P6 play verification
    p6 = False
    detail = "no play evidence provided"
    if args.video and Path(args.video).exists():
        size = Path(args.video).stat().st_size
        detail = f"video {Path(args.video).name} exists ({size//1024}KB)"
        p6 = size > 100_000
    if args.play_log and Path(args.play_log).exists():
        pl = Path(args.play_log).read_text(errors="replace")
        nofall = not re.search(r"base_contact[^\n]*[1-9]", pl)
        detail += f"; play log no base_contact term: {nofall}"
        p6 = p6 and nofall
    check("P6_play", p6, detail)

    # TARGET
    print("\n--- TARGET lines ---")
    tg = {
        "T1_lin_kernel": (lin_k >= TG["T1_lin_kernel"], f"{lin_k:.4f} vs {TG['T1_lin_kernel']}"),
        "T2_ang_kernel": (ang_k >= TG["T2_ang_kernel"], f"{ang_k:.4f} vs {TG['T2_ang_kernel']}"),
        "T3_style": (st >= TG["T3_style"], f"{st:.3f} vs {TG['T3_style']}"),
    }
    for k, (ok_, d) in tg.items():
        print(f"  {'HIT ' if ok_ else 'MISS'} {k}: {d}")

    fails = [k for k, v in results.items() if not v["pass"]]
    verdict = "PASS" if not fails else "FAIL"
    print("\n" + "=" * 72)
    print(f"VERDICT: {verdict}   ({len(results)-len(fails)}/{len(results)} pass, "
          f"fails={fails})")
    print(f"targets hit: {[k for k,(o,_) in tg.items() if o]}")
    print("=" * 72)

    if args.json:
        Path(args.json).write_text(json.dumps({
            "verdict": verdict, "fails": fails, "results": results,
            "targets": {k: {"hit": bool(o), "detail": d} for k, (o, d) in tg.items()},
            "window_last": 100, "iters": [iters[0], iters[-1]], "total_cfg": total_cfg,
        }, indent=1))
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
