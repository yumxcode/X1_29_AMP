#!/usr/bin/env python3
"""Gait-quality analysis for X1 sim2sim rollouts (strict criteria edition).

Consumes the per-step npz produced by `mujoco_rollout.py --log out.npz` and
scores the policy against the user's strict sim2sim criteria:

  G1 stability        — no falls, base height steady
  G2 swing symmetry   — L/R hip+knee swing amplitude, anti-phase, step length,
                        duty factor, stride regularity
  G3 landing quality  — touchdown foot pitch (flat or mild heel-first OK,
                        toe-first FAIL), roll-to-full-sole latency,
                        heel-up (toe-walk) fraction mid-stance, ground
                        penetration, swing clearance, mid-swing scuffing

Contact segmentation v2: the raw cdist<0 signal chatters every control step
during stance (v1 measured "573 touchdowns in 15 s" on a policy that walked
12.7 m). v2 uses a Schmitt trigger on the lowest sole-sphere height
(on < 4 mm, off > 12 mm vs the floor plane) plus debouncing, which is stable
at 50 Hz sampling.

Usage:
  PYTHONPATH=pylibs python sim2sim/gait_metrics.py log1.npz [log2.npz ...] \
      [--json report.json]

Sole sphere order per foot (from rollout log): [front_l, front_r, back_l, back_r].
pitch>0 => heel touches first (allowed; must roll to flat quickly);
pitch<0 => toe touches first (FAIL: 翘脚面/toe strike).
"""

import argparse
import functools
import json
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

SPEC = {
    # G2 symmetry
    "amp_ratio_min": 0.85,        # min/max of L vs R swing amplitude
    "phase_ref": 0.5, "phase_tol": 0.08,
    "step_len_ratio_min": 0.85,
    "duty_ratio_min": 0.80,
    "stride_cv_max": 0.20,        # stride period coefficient of variation
    # G3 landing
    "flat_deg": 8.0,              # |pitch@TD| <= this => flat landing
    "heel_max_deg": 22.0,         # pitch@TD in (flat_deg, this] => heel-first OK
    "toe_deg": -8.0,              # pitch@TD < this => toe-first FAIL
    "roll_flat_max_s": 0.18,      # TD -> all four spheres within tol of ground
    "ground_tol_m": 0.010,        # "on ground" tolerance for roll-to-flat
    "heelup_frac_max": 0.10,      # mid-stance heel-raised fraction (excl. push-off)
    "pushoff_skip": 0.25,         # trailing fraction of stance exempt (push-off)
    "settle_skip": 0.10,          # leading fraction exempt (weight transfer)
    "pen_max_m": 0.020,           # deepest instantaneous penetration allowed
    "pen_mean_m": 0.006,
    "clear_min_m": 0.015,         # max clearance of lowest sole point mid-swing
    "scuff_frac_max": 0.15,       # contact fraction during middle 60% of swing
    # G1
    "base_z_cv_max": 0.04,
}

# Schmitt trigger thresholds (sole sphere BOTTOM height vs floor z=0)
ON_M, OFF_M, MIN_ON, MIN_OFF = 0.004, 0.012, 4, 4


def detrend(x):
    t = np.arange(len(x), dtype=np.float64)
    A = np.stack([t, np.ones_like(t)], 1)
    c, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ c


def best_lag(x, y, dt):
    """Normalized cross-correlation: returns (lag_s, corr_at_lag, anti)."""
    x, y = detrend(x.astype(np.float64)), detrend(y.astype(np.float64))
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    cc = np.correlate(y, x, mode="full")
    lags = np.arange(-len(x) + 1, len(y))
    k_max, k_min = int(np.argmax(cc)), int(np.argmin(cc))
    if abs(cc[k_min]) > abs(cc[k_max]):
        return lags[k_min] * dt, cc[k_min], True
    return lags[k_max] * dt, cc[k_max], False


def schmitt_contact(min_z):
    """Debounced foot-on-ground signal. Hysteresis kills 50 Hz contact
    chatter; airborne gaps shorter than MIN_OFF samples are glued."""
    n = len(min_z)
    on = np.zeros(n, dtype=bool)
    state = bool(min_z[0] < ON_M)
    i = 0
    while i < n:
        if state:
            j = i
            while j < n and min_z[j] < OFF_M:
                j += 1
            on[i:j] = True
            state = False
            i = j
        else:
            j = i
            while j < n and min_z[j] >= ON_M:
                j += 1
            if j - i < MIN_OFF and j < n and i > 0:  # too-short gap: glue
                on[i:j] = True
            state = True
            i = j
    return on


def stance_windows(contact, start_i):
    """Bool array -> list of (td_i, liftoff_i), td >= start_i, len > MIN_ON."""
    on = contact.astype(int)
    edges = np.diff(on, prepend=0)
    tds = list(np.where(edges == 1)[0])
    los = list(np.where(edges == -1)[0] - 1)
    if tds and los and los[0] < tds[0]:
        los = los[1:]
    if tds and los and tds[-1] > los[-1]:
        tds = tds[:-1]
    return [(t, l) for t, l in zip(tds, los) if t >= start_i and l - t >= MIN_ON]


def analyze(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    dt = float(meta["control_dt"])
    feet = list(meta["foot_names"])
    hinge = list(meta["hinge_names"])
    t = z["t"].astype(np.float64)
    q = z["q"].astype(np.float64)
    sole = z["sole_xyz"].astype(np.float64)        # (n,2,4,3)
    cdist = z["contact_dist"].astype(np.float64)   # (n,2,4); <0 = penetration
    n = len(t)
    start = int(meta["settle_steps"]) + 2
    R = {"file": str(npz_path), "cmd": meta["cmd"], "fell": meta["fell"], "n": n}

    jn = lambda name: q[:, hinge.index(name)]
    hp = [jn("left_hip_pitch_joint"), jn("right_hip_pitch_joint")]
    kn = [jn("left_knee_pitch_joint"), jn("right_knee_pitch_joint")]

    # ---- G1 stability -------------------------------------------------
    base_z = z["base_pos"][:, 2]
    R["G1"] = {
        "fell": bool(meta["fell"]),
        "base_z_mean": float(base_z[start:].mean()),
        "base_z_min": float(base_z[start:].min()),
        "base_z_cv": float(base_z[start:].std() / base_z[start:].mean()),
    }
    R["G1"]["PASS"] = (not meta["fell"]) and R["G1"]["base_z_cv"] < SPEC["base_z_cv_max"]

    # ---- per-foot time series + segmentation ---------------------------
    front, back = slice(0, 2), slice(2, 4)
    bottom = sole[:, :, :, 2] - 0.002              # sphere bottoms, floor z=0
    on_ground = bottom < SPEC["ground_tol_m"]      # (n,2,4) geometric
    st = {}
    for f, name in enumerate(feet):
        P = sole[:, f]
        fm, bm = P[:, front].mean(1), P[:, back].mean(1)
        fv = fm - bm
        pitch = np.degrees(np.arctan2(fv[:, 2], np.linalg.norm(fv[:, :2], axis=1)))
        contact = schmitt_contact(bottom[:, f].min(1))
        strides = stance_windows(contact, start)
        tds = [td for td, _ in strides]
        nxt = {td: (tds[i + 1] if i + 1 < len(tds) else n - 2)
               for i, td in enumerate(tds)}
        st[name] = dict(
            f=f, contact=contact, pitch=pitch,
            cf=on_ground[:, f, front].any(1), cb=on_ground[:, f, back].any(1),
            heel_z=bm[:, 2], toe_z=fm[:, 2],
            midz=0.5 * (fm[:, 2] + bm[:, 2]), mid_xy=0.5 * (fm[:, :2] + bm[:, :2]),
            strides=strides, nxt=nxt,
        )

    # ---- G2 symmetry ---------------------------------------------------
    g2 = {}
    lo, ro = feet[0], feet[1]
    s_lo, s_ro = st[lo]["strides"], st[ro]["strides"]
    amp = {}
    for label, (sa, sb) in (("hip_pitch", hp), ("knee_pitch", kn)):
        a = [float(np.ptp(sa[td:st[lo]["nxt"][td]])) for td, _ in s_lo]
        b = [float(np.ptp(sb[td:st[ro]["nxt"][td]])) for td, _ in s_ro]
        ra = np.mean(a) if a else np.nan
        rb = np.mean(b) if b else np.nan
        ratio = min(ra, rb) / max(ra, rb) if a and b and max(ra, rb) > 0 else np.nan
        amp[label] = (float(np.degrees(ra)), float(np.degrees(rb)), float(ratio))
    g2["amp"] = {k: {"left_deg": v[0], "right_deg": v[1], "ratio": v[2]}
                 for k, v in amp.items()}
    sl = {}
    for name in feet:
        tds = [td for td, _ in st[name]["strides"]]
        xy = st[name]["mid_xy"]
        sl[name] = [float(np.linalg.norm(xy[b] - xy[a])) for a, b in zip(tds[:-1], tds[1:])]
    g2["step_len"] = {name: float(np.mean(v)) if v else np.nan for name, v in sl.items()}
    msl = [np.mean(v) for v in (sl[lo], sl[ro]) if v]
    g2["step_len_ratio"] = float(min(msl) / max(msl)) if len(msl) == 2 and min(msl) > 0 else np.nan
    w0, w1 = start, n - 2
    lag_s, corr, anti = best_lag(hp[0][w0:w1], hp[1][w0:w1], dt)
    per = []
    for name in feet:
        tds = [td for td, _ in st[name]["strides"]]
        per += list(np.diff(tds) * dt)
    med_period = float(np.median(per)) if per else np.nan
    g2.update(phase_lag_s=float(lag_s), xcorr=float(corr), phase_anti=bool(anti),
              phase_frac=float(abs(lag_s) / med_period) if med_period else np.nan,
              stride_period_s=med_period,
              stride_cv=float(np.std(per) / np.mean(per)) if per and np.mean(per) else np.nan,
              n_strides=len(per))
    duty = {}
    for name in feet:
        ds = [(l - td) * dt for td, l in st[name]["strides"]]
        duty[name] = float(np.mean(ds) / med_period) if ds and med_period else np.nan
    g2["duty"] = duty
    dv = [v for v in duty.values() if not np.isnan(v)]
    g2["duty_ratio"] = float(min(dv) / max(dv)) if len(dv) == 2 and min(dv) > 0 else np.nan
    ok = lambda x, thr: (not np.isnan(x)) and x >= thr
    checks = [
        all(amp[k][2] >= SPEC["amp_ratio_min"] for k in amp),
        ok(g2["step_len_ratio"], SPEC["step_len_ratio_min"]),
        ok(g2["phase_frac"], SPEC["phase_ref"] - SPEC["phase_tol"])
        and g2["phase_frac"] <= SPEC["phase_ref"] + SPEC["phase_tol"] and anti,
        ok(SPEC["stride_cv_max"] - g2["stride_cv"], 0) if not np.isnan(g2["stride_cv"]) else False,
        ok(g2["duty_ratio"], SPEC["duty_ratio_min"]),
    ]
    # v27.2 analyzer: turning commands (|wz| > 0.3 rad/s) PHYSICALLY require
    # L/R asymmetry (inner leg shorter stride, phase shift) — a straight-line
    # symmetry gate is invalid there. G2 stays informational for turns.
    turning = abs(float(meta["cmd"][2])) > 0.3
    g2["exempt_turn"] = turning
    g2["PASS"] = (all(checks) and len(per) >= 8) or turning
    R["G2"] = g2

    # ---- G3 landing quality --------------------------------------------
    evs = []
    for name in feet:
        s = st[name]
        for k, (td, lo_i) in enumerate(s["strides"]):
            pre = td - 1
            if pre < start:
                continue
            zH, zT = s["heel_z"], s["toe_z"]
            low = (zH < zH[td:lo_i + 1].min() + SPEC["ground_tol_m"]) & \
                  (zT < zT[td:lo_i + 1].min() + SPEC["ground_tol_m"])
            rtf = next((i for i in range(td, lo_i + 1) if low[i]), None)
            span = lo_i - td + 1
            m0 = td + int(SPEC["settle_skip"] * span)
            m1 = lo_i - int(SPEC["pushoff_skip"] * span)
            heelup = (np.sum(zH[m0:m1 + 1] - zT[m0:m1 + 1] > 0.04)
                      if m1 > m0 else 0)
            nxt_td = s["nxt"][td]
            sw_len = nxt_td - lo_i - 1
            cl, sc = np.nan, np.nan
            if sw_len > 4:
                a0 = lo_i + 1 + sw_len // 5
                a1 = nxt_td - sw_len // 5
                cl = float(bottom[a0:a1, s["f"]].min() * -1)  # placeholder replaced below
                cl = float(bottom[a0:a1, s["f"]].max())       # apex clearance
                sc = float(np.mean(s["contact"][a0:a1]))
            pen = float(-np.minimum(cdist[td:lo_i + 1, s["f"]], 0).min())
            evs.append(dict(
                foot=name, td_t=float(t[td]),
                pitch_td=float(s["pitch"][pre]),
                pitch_min_stance=float(s["pitch"][td:lo_i + 1].min()),
                pitch_max_stance=float(s["pitch"][td:lo_i + 1].max()),
                vz_td=float(-(s["midz"][pre + 1] - s["midz"][pre]) / dt) if pre + 1 < n else np.nan,
                roll_flat_s=float((rtf - td) * dt) if rtf is not None else np.nan,
                heelup_frac=float(heelup / (m1 - m0 + 1)) if m1 > m0 else np.nan,
                pen_m=pen, clearance_m=cl, scuff_frac=sc,
                stance_s=float((lo_i - td) * dt),
            ))
    g3 = {"events": evs}
    if evs:
        pit = np.array([e["pitch_td"] for e in evs])
        cls = {"flat": int(np.sum(np.abs(pit) <= SPEC["flat_deg"])),
               "heel_first": int(np.sum((pit > SPEC["flat_deg"]) & (pit <= SPEC["heel_max_deg"]))),
               "too_high_heel": int(np.sum(pit > SPEC["heel_max_deg"])),
               "toe_first": int(np.sum(pit < SPEC["toe_deg"]))}
        rtf = np.array([e["roll_flat_s"] for e in evs], dtype=np.float64)
        hu = np.array([e["heelup_frac"] for e in evs], dtype=np.float64)
        pen = np.array([e["pen_m"] for e in evs])
        cl = np.array([e["clearance_m"] for e in evs], dtype=np.float64)
        sc = np.array([e["scuff_frac"] for e in evs], dtype=np.float64)
        vz = np.array([e["vz_td"] for e in evs], dtype=np.float64)
        g3.update(n_events=len(evs), touchdown_class=cls,
                  pitch_td_mean=float(pit.mean()), pitch_td_min=float(pit.min()),
                  pitch_td_max=float(pit.max()),
                  roll_flat_med_s=float(np.nanmedian(rtf)),
                  heelup_med=float(np.nanmedian(hu)),
                  pen_max_m=float(pen.max()), pen_mean_m=float(pen.mean()),
                  clearance_med_m=float(np.nanmedian(cl)),
                  scuff_med=float(np.nanmedian(sc)),
                  vz_td_med=float(np.nanmedian(vz)))
        g3["PASS"] = bool(
            cls["toe_first"] == 0 and cls["too_high_heel"] == 0
            and g3["roll_flat_med_s"] <= SPEC["roll_flat_max_s"]
            and g3["heelup_med"] <= SPEC["heelup_frac_max"]
            and g3["pen_max_m"] <= SPEC["pen_max_m"]
            and g3["pen_mean_m"] <= SPEC["pen_mean_m"]
            and g3["clearance_med_m"] >= SPEC["clear_min_m"]
            and g3["scuff_med"] <= SPEC["scuff_frac_max"])
    else:
        g3["PASS"] = False
    R["G3"] = g3
    R["PASS"] = bool(R["G1"]["PASS"] and g2["PASS"] and g3["PASS"])
    return R


def fmt(R):
    lines = [f"== {Path(R['file']).name}  cmd={R['cmd']}  n={R['n']}"]
    g1 = R["G1"]
    lines.append(f"[G1 stability]     PASS={g1['PASS']} fell={g1['fell']} "
                 f"base_z mean={g1['base_z_mean']:.3f} min={g1['base_z_min']:.3f} cv={g1['base_z_cv']:.4f}")
    g2 = R["G2"]
    ex = " [EXEMPT:turn]" if g2.get("exempt_turn") else ""
    lines.append(f"[G2 symmetry]      PASS={g2['PASS']}{ex} strides={g2['n_strides']}")
    for k, v in g2["amp"].items():
        lines.append(f"    {k:12s} L={v['left_deg']:6.2f}° R={v['right_deg']:6.2f}° ratio={v['ratio']:.3f}")
    lines.append(f"    step_len    L={g2['step_len'][list(g2['step_len'])[0]]:.3f}m ratio={g2['step_len_ratio']:.3f} "
                 f"period={g2['stride_period_s']:.3f}s cv={g2['stride_cv']:.3f}")
    lines.append(f"    phase       frac={g2['phase_frac']:.3f} (ref 0.5) anti={g2['phase_anti']} "
                 f"xcorr={g2['xcorr']:+.2f} duty_ratio={g2['duty_ratio']:.3f}")
    g3 = R["G3"]
    if "n_events" in g3:
        c = g3["touchdown_class"]
        lines.append(f"[G3 landing]       PASS={g3['PASS']} events={g3['n_events']} "
                     f"flat={c['flat']} heel_first={c['heel_first']} "
                     f"too_high_heel={c['too_high_heel']} toe_first={c['toe_first']}")
        lines.append(f"    pitch@TD    mean={g3['pitch_td_mean']:6.2f}° range=[{g3['pitch_td_min']:.1f},{g3['pitch_td_max']:.1f}]° "
                     f"vz@TD med={g3['vz_td_med']:.2f} m/s")
        lines.append(f"    roll2flat   med={g3['roll_flat_med_s']*1000:.0f} ms   heelup_med={g3['heelup_med']:.3f}")
        lines.append(f"    penetration max={g3['pen_max_m']*1000:.1f} mm mean={g3['pen_mean_m']*1000:.2f} mm   "
                     f"clearance={g3['clearance_med_m']*1000:.0f} mm scuff={g3['scuff_med']:.3f}")
    else:
        lines.append("[G3 landing]       PASS=False (no touchdown events)")
    lines.append(f"[VERDICT] {'PASS' if R['PASS'] else 'FAIL'}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    out = []
    for p in args.logs:
        R = analyze(p)
        out.append(R)
        print(fmt(R))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=float))
        print(f"[JSON] {args.json}")
    print(f"[TOTAL] {sum(r['PASS'] for r in out)}/{len(out)} scenarios PASS")


if __name__ == "__main__":
    main()
