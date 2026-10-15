#!/usr/bin/env python3
"""v69 gait forensics: quantify the m11498 video defects.

Defects (user, 2026-10-15):
  D1 外八        — feet toes-out (hip_yaw external rotation) -> small leg swing
  D2 腰摆大      — large + frequent lumbar roll oscillation
  D3 腰带手      — arm swing driven by waist sway (not shoulder-driven)

Usage:
  PYTHONPATH=pylibs python acceptance/v69_eval/diag_gait_v69.py <rollout.npz> [...]
"""
import functools
import json
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)


def detrend(x):
    t = np.arange(len(x), dtype=np.float64)
    A = np.stack([t, np.ones_like(t)], 1)
    c, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ c


def best_lag(x, y, dt):
    x, y = detrend(x.astype(np.float64)), detrend(y.astype(np.float64))
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    cc = np.correlate(y, x, mode="full")
    cc = cc / len(x)
    lags = np.arange(-len(x) + 1, len(y))
    k = int(np.argmax(np.abs(cc)))
    return lags[k] * dt, cc[k]


def analyze(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    dt = float(meta["control_dt"])
    hinge = list(meta["hinge_names"])
    q = z["q"].astype(np.float64)
    dq = z["dq"].astype(np.float64) if "dq" in z else None
    start = int(meta["settle_steps"]) + 2
    R = {"file": str(npz_path), "cmd": meta["cmd"], "fell": bool(meta["fell"])}

    def jn(name):
        return q[start:, hinge.index(name)]

    def jnv(name):
        return dq[start:, hinge.index(name)]

    deg = lambda x: np.degrees(x)

    # ---- D1: 外八 ------------------------------------------------------
    # hip_yaw sign convention: need URDF check; report raw + abs stats
    hy = {s: jn(f"{s}_hip_yaw_joint") for s in ("left", "right")}
    # foot yaw direction in world: integrate foot frame heading? Use hip_yaw
    # directly (1:1 transmission) + report sign convention separately.
    R["hip_yaw_deg"] = {
        "left_mean": float(deg(hy["left"].mean())),
        "right_mean": float(deg(hy["right"].mean())),
        "left_p95": float(np.percentile(deg(hy["left"]), 95)),
        "right_p95": float(np.percentile(deg(hy["right"]), 95)),
        "left_min": float(deg(hy["left"]).min()),
        "right_min": float(deg(hy["right"]).min()),
        "left_max": float(deg(hy["left"]).max()),
        "right_max": float(deg(hy["right"]).max()),
    }
    # feet toe-out via sole spheres: cols[0,1] = local+z = TOE (FK-verified
    # 2026-10-15: zero-pose local+z -> world +x forward), cols[2,3] = HEEL.
    # heading in the BASE frame (walk direction = +x_base), unwrapped.
    sole = z["sole_xyz"].astype(np.float64)
    bq = z["base_quat"].astype(np.float64)   # (n,4) MuJoCo wxyz
    base_yaw = 2 * np.arctan2(bq[start:, 3], bq[start:, 0])
    fe = {}
    for f, name in enumerate(meta["foot_names"]):
        toe = sole[start:, f, 0:2].mean(1)
        heel = sole[start:, f, 2:4].mean(1)
        dxy = toe - heel
        hdg = np.unwrap(np.arctan2(dxy[:, 1], dxy[:, 0])) - base_yaw
        hdg = (hdg + np.pi) % (2 * np.pi) - np.pi
        fe[name] = np.degrees(hdg)
    R["foot_heading_deg"] = {k: {"mean": float(v.mean()), "std": float(v.std()),
                                 "p05": float(np.percentile(v, 5)),
                                 "p95": float(np.percentile(v, 95))} for k, v in fe.items()}
    R["foot_heading_deg"] = {k: {"mean": float(v.mean()), "std": float(v.std()),
                                 "p05": float(np.percentile(v, 5)),
                                 "p95": float(np.percentile(v, 95))} for k, v in fe.items()}
    # 外八 in base frame: left foot heading > 0 = toe-out (+y side);
    # right foot heading < 0 = toe-out (-y side).
    lf, rf = fe[meta["foot_names"][0]], fe[meta["foot_names"][1]]
    R["toe_out_deg"] = {
        "left_mean": float(lf.mean()), "right_mean": float(-rf.mean()),
        "sum_mean": float(lf.mean() - rf.mean()),      # + = symmetric toe-out
        "separation_mean": float(lf.mean() - rf.mean()),
    }

    # ---- D2: lumbar roll / waist sway ---------------------------------
    lr = deg(jn("lumbar_roll_joint"))
    hp = {s: deg(jn(f"{s}_hip_pitch_joint")) for s in ("left", "right")}
    R["lumbar_roll_deg"] = {
        "mean": float(lr.mean()), "ptp": float(lr.max() - lr.min()),
        "std": float(lr.std()),
        "swing_band": float(np.percentile(np.abs(detrend(lr)), 95)),
    }
    if dq is not None:
        lrv = deg(jnv("lumbar_roll_joint"))
        # zero crossings per second of high-passed signal
        hp_sig = detrend(lr)
        zc = np.sum(np.diff(np.sign(hp_sig)) != 0) / (len(lr) * dt)
        R["lumbar_roll_deg"]["zero_cross_hz"] = float(zc)
    # base lateral sway
    bp = z["base_pos"].astype(np.float64)[start:]
    by = detrend(bp[:, 1])
    R["base_sway_m"] = {"ptp": float(by.max() - by.min()), "std": float(by.std())}

    # ---- D3: arm swing source (waist vs shoulder) ----------------------
    sp = {s: deg(jn(f"{s}_shoulder_pitch_joint")) for s in ("left", "right")}
    sr = {s: deg(jn(f"{s}_shoulder_roll_joint")) for s in ("left", "right")}
    ly = deg(jn("lumbar_yaw_joint"))
    R["arm_swing_deg"] = {s: {"ptp": float(v.max() - v.min()),
                              "amp": float(np.ptp(detrend(v)))} for s, v in sp.items()}
    R["shoulder_roll_deg"] = {s: {"ptp": float(v.max() - v.min()),
                                  "amp": float(np.ptp(detrend(v)))} for s, v in sr.items()}
    R["lumbar_yaw_deg"] = {"mean": float(ly.mean()), "ptp": float(ly.max() - ly.min()),
                           "amp": float(np.ptp(detrend(ly)))}
    # couplings
    lag, c_waistL = best_lag(lr, sp["left"], dt)
    R["corr_waistroll_vs_Lshoulder"] = {"lag_s": float(lag), "corr": float(c_waistL)}
    lag, c = best_lag(sp["left"], sp["right"], dt)
    R["corr_Lsp_vs_Rsp"] = {"lag_s": float(lag), "corr": float(c)}
    # arm vs opposite hip (the healthy coupling)
    lag, c = best_lag(sp["left"], hp["right"], dt)
    R["corr_Lsp_vs_Rhip"] = {"lag_s": float(lag), "corr": float(c)}
    lag, c = best_lag(sp["right"], hp["left"], dt)
    R["corr_Rsp_vs_Lhip"] = {"lag_s": float(lag), "corr": float(c)}
    # hip pitch swing amplitude (the "small leg swing" claim)
    R["hip_pitch_swing_deg"] = {s: {"ptp": float(v.max() - v.min()),
                                    "amp": float(np.ptp(detrend(v)))}
                                for s, v in hp.items()}
    kn = {s: deg(jn(f"{s}_knee_pitch_joint")) for s in ("left", "right")}
    R["knee_pitch_swing_deg"] = {s: {"ptp": float(v.max() - v.min()),
                                     "amp": float(np.ptp(detrend(v)))} for s, v in kn.items()}
    return R


def main():
    for p in sys.argv[1:]:
        R = analyze(p)
        print(f"\n== {Path(p).name} cmd={R['cmd']}")
        hy = R["hip_yaw_deg"]
        print(f"[D1 外八] hip_yaw deg L mean={hy['left_mean']:+.1f} [{hy['left_min']:+.1f},{hy['left_max']:+.1f}] "
              f"R mean={hy['right_mean']:+.1f} [{hy['right_min']:+.1f},{hy['right_max']:+.1f}]")
        to = R["toe_out_deg"]
        print(f"         foot-heading deg L={to['left_mean']:+.1f} R={to['right_mean']:+.1f} "
              f"sep={to['separation_mean']:+.1f}")
        for k, v in R["foot_heading_deg"].items():
            print(f"         {k:24s} mean={v['mean']:+.1f} std={v['std']:.1f} p05={v['p05']:+.1f} p95={v['p95']:+.1f}")
        lr = R["lumbar_roll_deg"]
        print(f"[D2 腰摆] lumbar_roll deg mean={lr['mean']:+.1f} ptp={lr['ptp']:.1f} "
              f"std={lr['std']:.1f} band95={lr['swing_band']:.1f}"
              + (f" zc={lr['zero_cross_hz']:.2f}Hz" if "zero_cross_hz" in lr else ""))
        bs = R["base_sway_m"]
        print(f"         base_sway y ptp={bs['ptp']*1000:.0f}mm std={bs['std']*1000:.0f}mm")
        arm = R["arm_swing_deg"]
        srd = R["shoulder_roll_deg"]
        lyd = R["lumbar_yaw_deg"]
        print(f"[D3 手臂] shoulder_pitch ptp L={arm['left']['ptp']:.1f} R={arm['right']['ptp']:.1f} deg "
              f"| shoulder_roll ptp L={srd['left']['ptp']:.1f} R={srd['right']['ptp']:.1f} deg "
              f"| lumbar_yaw ptp={lyd['ptp']:.1f} amp={lyd['amp']:.1f} deg")
        cw = R["corr_waistroll_vs_Lshoulder"]
        ca = R["corr_Lsp_vs_Rsp"]
        print(f"         corr(waist_roll, L_shoulder)={cw['corr']:+.2f}@{cw['lag_s']:+.3f}s "
              f"corr(Lsp,Rsp)={ca['corr']:+.2f}@{ca['lag_s']:+.3f}s")
        print(f"         corr(Lsp,Rhip)={R['corr_Lsp_vs_Rhip']['corr']:+.2f} "
              f"corr(Rsp,Lhip)={R['corr_Rsp_vs_Lhip']['corr']:+.2f}")
        hp = R["hip_pitch_swing_deg"]
        kn = R["knee_pitch_swing_deg"]
        print(f"[腿摆幅] hip_pitch ptp L={hp['left']['ptp']:.1f} R={hp['right']['ptp']:.1f} | "
              f"knee ptp L={kn['left']['ptp']:.1f} R={kn['right']['ptp']:.1f} deg")


if __name__ == "__main__":
    main()
