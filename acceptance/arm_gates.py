#!/usr/bin/env python3
"""v95 ARM GATES (pre-registered, RECORD_BOOK 14 / exp_fe1b6795).

Reads arm_swing_audit.py JSON output for walk10/walk05/back05 rollouts
and applies the binding A1-A5 gates (defect = x7398 baseline):

  A1 shoY p2p       <= 20 deg  all scenarios   (defect 48-112, demo <=0.5)
  A2 elbP p2p       <= 30 deg  all scenarios   (defect 71-90,  demo 0)
  A3 wriP/wriR p2p  <= 35 deg  all scenarios   (defect 35-67,  demo 5-29)
  A4 corr(phiL,phiR)<= -0.6    walk10 & walk05 (defect -0.33/-0.52, demo -0.98)
  A5 hand lat p2p   <= 0.45 m  all scenarios   (defect up to 0.56)

Report-only (secondary): shoR DC, phi p2p, elbY p2p, arm-leg coupling.
"""
import argparse
import json
import sys


def eval_run(rep, name):
    M = rep["metrics"]
    checks = []
    for s in "LR":
        checks += [
            ("A1", f"{name} {s} shoY p2p", M[f"{s}_shoulder_yaw_p2p"], 20.0, "le"),
            ("A2", f"{name} {s} elbP p2p", M[f"{s}_elbow_pitch_p2p"], 30.0, "le"),
            ("A3a", f"{name} {s} wriP p2p", M[f"{s}_wrist_pitch_p2p"], 35.0, "le"),
            ("A3b", f"{name} {s} wriR p2p", M[f"{s}_wrist_roll_p2p"], 35.0, "le"),
            ("A5", f"{name} {s} hand lat p2p", M[f"{s}_hand_lat_p2p"], 0.45, "le"),
        ]
    if any(name.endswith(s) for s in ("walk10", "walk05")):
        # pre-registered: corr(phiL,phiR) <= -0.6 (demo -0.98)
        checks.append(("A4", f"{name} corr(phiL,phiR)", M["corr_phiLR"], -0.6, "le"))
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rep = json.load(open(args.json_path))
    fails = 0
    for entry in rep:
        name = entry["log"].split("/")[-1].replace(".npz", "")
        for gid, label, val, bound, op in eval_run(entry, name):
            ok = val <= bound if op == "le" else val >= bound
            tag = "PASS" if ok else "FAIL"
            fails += (not ok)
            print(f"[{tag}] {gid} {label}: {val:.3f} (gate {'<=' if op=='le' else '>='} {bound})")
        M = entry["metrics"]
        print(f"  -- report: shoR DC {M['L_shoulder_roll_mean']:+.1f}/{M['R_shoulder_roll_mean']:+.1f} "
              f"| phi p2p {M['L_phi_p2p']:.0f}/{M['R_phi_p2p']:.0f} "
              f"| elbY p2p {M['L_elbow_yaw_p2p']:.0f}/{M['R_elbow_yaw_p2p']:.0f} "
              f"| DC asym {M['dc_asym']:.1f}")
    print(f"\n{'ALL ARM GATES PASS' if fails == 0 else f'{fails} ARM GATE FAILURES'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
