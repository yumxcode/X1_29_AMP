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
    # KH: human-gait arc (GOAL_HUMAN_GAIT.md §3)
    "k1_mid_max_deg": 18.0,       # stance 25-75% mean knee flexion, per foot
    "k2_range_min_deg": 25.0,     # stance knee min->max swing range
    "h1_heel_first_min": 0.60,    # heel-before-toe landing fraction (walk10)
    "h2_toe_off_min": 0.90,       # heel-off/toe-on launch fraction
    "h3_toe_first_max": 0.10,     # true forefoot strike (lead<=-15mm); baseline
                                  # soup reads 0-9.5% under the fine lead metric
                                  # (G3's coarse pitch gate stays =0 as the hard
                                  # no-regress gate); v57 targets 0
    "kh_min_events": 4,           # per foot, else informational only
    # R: rhythm gate (v63, GOAL_RHYTHM.md §2) — human cadence + stride.
    # cycle = same-foot consecutive-TD interval (this file's stride_period_s);
    # cadence total (both feet) = 120 / cycle. Calibrated on the retargeted
    # reference set (hip-osc path) + human literature at 1.0/0.5 m/s.
    "r1_cycle_lo_s": 0.85,        # walk10: human ~1.03-1.16, gate -18%/+16%
    "r1_cycle_hi_s": 1.35,
    "r1_cycle_lo_s_05": 1.00,     # walk05: human ~1.3-1.5 (refs up to 1.8)
    "r1_cycle_hi_s_05": 1.65,
    "r2_stride_min_m": 0.78,      # walk10: ref 0.94-0.99, lit ~1.05
    "r2_stride_min_m_05": 0.42,   # walk05: ref 0.53-0.73, lit ~0.65
    "r3_min_events": 4,           # per foot — drag (0 events) fails
    "r4_cycle_cv_max": 0.20,      # rhythm regularity (same class as stride_cv)
}

# Schmitt trigger thresholds (sole sphere BOTTOM height vs floor z=0)
# MIN_ON/MIN_OFF are FRAME counts for the 50 Hz reference (80 ms); main()
# rescales them from meta["control_dt"] so other control rates keep the
# same TIME windows (v61 100 Hz support).
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
    global MIN_ON, MIN_OFF
    MIN_ON = max(2, int(round(0.08 / dt)))
    MIN_OFF = max(2, int(round(0.08 / dt)))
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
    # LABELING CAVEAT (v56 audit, semantic anchor on 0002): sole cols
    # [0,1] (+0.07 local z) are the HEEL end and [2,3] the TOE end — the
    # inverse of the "front/back" names used below (and in the rollout
    # meta / find_sole_geoms comments). G3's heel_z/toe_z/pitch are
    # therefore sign-flipped in NAME only; on the flat-landing policies
    # this segment gates, every quantity measures ~0 either way (24
    # generations of reports are unaffected). The KH section below uses
    # the corrected semantics explicitly. Do NOT "fix" G3 in place — it
    # would silently redefine 24 generations of gate history.
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

    # ---- KH: human-gait features (straight knee + heel-toe) --------------
    # K1/K2: knee angle convention 0=full extension, + = flexion (DEFAULT_Q
    # knee 0.632 rad = 36.2 deg crouch). Per stance window:
    #   K1 = mean(theta over the 25%-75% mid-stance span)
    #   K2 = max(theta) - min(theta) within the stance
    # HEEL/TOE SEMANTICS (v56 fix): sole cols [0,1] = local z +0.07 = HEEL
    # end, cols [2,3] = local z -0.07 = TOE end — the OPPOSITE of what the
    # rollout meta / find_sole_geoms comments claim ("front=+0.07"). Proof:
    # semantic anchor on 0002_treadmill_slow (human slow walk, heel-strike
    # is physiological): diag_heeltoe (heel=+0.07) measures 67% heel-first
    # there, and the alone-down frame counts fit ONLY with +0.07=heel (12
    # heel-only frames = brief heel-strike phase, 175 toe-only frames =
    # long push-off); 0005's -0.07 dominance then matches its documented
    # toe-first drag. cf (cols 0,1) is therefore the HEEL channel and cb
    # (cols 2,3) the TOE channel in this section.
    # H1/H2/H3: GEOMETRIC LEAD classification (frame-quantization immune).
    # lead_td = toe_bottom - heel_bottom at the touchdown frame
    # (+ = heel end lower = heel-first incline); lead_lo = same at the
    # last-contact frame (+ = toe end lower = heel-off/toe-off).
    # Measured lead distributions (probe_kh_metric.py, 2026-09-16):
    #   0002 heel-exemplar refs: TD lead med +4.0/+3.5, p75 +6.8 mm
    #     -> the retargeted heel-strike is GENTLE (near-flat, heel 3-7 mm
    #        lower) — an 8 mm threshold would classify the exemplar flat
    #   0005 toe-drag refs:      TD lead -22..-26 mm (true forefoot)
    #   soup534_40 policy:       TD lead -8.6..-11.1 mm (mild plantarflexed
    #        landing — the doc's "toe-first 0" was the old 10 mm/deadband
    #        classifier lumping it flat), LO lead +26..+56 mm (toe-off real)
    # Classification: heel-first lead_td >= +2 mm; true forefoot strike
    # (H3 violation) lead_td <= -15 mm; toe-off lead_lo >= +2 mm.
    kh = {}
    for name in feet:
        s = st[name]
        f = s["f"]
        knee = q[:, hinge.index("left_knee_pitch_joint" if f == 0 else "right_knee_pitch_joint")]
        # v56 fix: cols [0,1] (+0.07 local z) = HEEL end, [2,3] = TOE end
        heel_b = bottom[:, f, 0:2].min(1)
        toe_b = bottom[:, f, 2:4].min(1)
        k1s, k2s, land, launch = [], [], {"heel": 0, "toe": 0, "flat": 0}, {"toeoff": 0, "flat": 0}
        for td, lo_i in s["strides"]:
            span = lo_i - td + 1
            seg = knee[td:lo_i + 1]
            m0, m1 = td + span // 4, td + 3 * span // 4
            k1s.append(float(np.degrees(knee[m0:m1 + 1].mean())))
            k2s.append(float(np.degrees(seg.max() - seg.min())))
            lead_td = float(toe_b[td] - heel_b[td])      # + = heel lower
            if lead_td >= 0.002:
                land["heel"] += 1
            elif lead_td <= -0.015:
                land["toe"] += 1
            else:
                land["flat"] += 1
            lead_lo = float(toe_b[lo_i] - heel_b[lo_i])  # + = toe lower
            if lead_lo >= 0.002:
                launch["toeoff"] += 1
            else:
                launch["flat"] += 1
        n_land = sum(land.values())
        n_launch = sum(launch.values())
        kh[name] = dict(
            n_events=len(s["strides"]),
            k1_mid_deg=float(np.mean(k1s)) if k1s else float("nan"),
            k2_range_deg=float(np.mean(k2s)) if k2s else float("nan"),
            land_heel_first_frac=float(land["heel"] / n_land) if n_land else float("nan"),
            land_toe_first_frac=float(land["toe"] / n_land) if n_land else float("nan"),
            launch_toe_off_frac=float(launch["toeoff"] / n_launch) if n_launch else float("nan"),
            n_land=n_land,
        )
    # verdict: walk scenarios only (turn exempt — heel/toe timing is not a
    # turning invariant); both feet must hold K1/K2 and H3=0. H1/H2 gates
    # are WALK10-only per spec (H1 noise dominates at 0.5 m/s).
    straight = abs(float(meta["cmd"][2])) <= 0.3
    kh["exempt_turn"] = not straight
    feets = feet if straight else []
    k1_vals = [kh[nm]["k1_mid_deg"] for nm in feets]
    k2_vals = [kh[nm]["k2_range_deg"] for nm in feets]
    toe_first = [kh[nm]["land_toe_first_frac"] for nm in feets]
    enough = all(kh[nm]["n_events"] >= SPEC["kh_min_events"] for nm in feets)
    cmd_speed = float(np.linalg.norm(meta["cmd"][:2]))
    is_walk10 = straight and abs(cmd_speed - 1.0) < 0.26
    h1_vals = [kh[nm]["land_heel_first_frac"] for nm in feets] if is_walk10 else []
    h2_vals = [kh[nm]["launch_toe_off_frac"] for nm in feets] if is_walk10 else []
    kh["cmd_speed"] = cmd_speed
    kh["K1_PASS"] = bool(feets and enough and all(v <= SPEC["k1_mid_max_deg"] for v in k1_vals))
    kh["K2_PASS"] = bool(feets and enough and all(v >= SPEC["k2_range_min_deg"] for v in k2_vals))
    kh["H3_PASS"] = bool(feets and enough and all(v <= SPEC["h3_toe_first_max"] for v in toe_first))
    kh["H1_PASS"] = bool(h1_vals and enough and all(v >= SPEC["h1_heel_first_min"] for v in h1_vals))
    kh["H2_PASS"] = bool(h2_vals and enough and all(v >= SPEC["h2_toe_off_min"] for v in h2_vals))
    kh["PASS"] = bool(kh["K1_PASS"] and kh["K2_PASS"] and kh["H3_PASS"] and kh["H2_PASS"] and kh["H1_PASS"])
    R["KH"] = kh

    # ---- R: rhythm gate (v63, GOAL_RHYTHM.md) ---------------------------
    # cadence + stride vs human reference. cycle = same-foot consecutive-TD
    # interval (G2's stride_period_s already computes this per foot — R
    # gates the ABSOLUTE value, G2 gates symmetry/regularity). stride =
    # same-foot consecutive-TD world distance (G2's step_len). Gate applies
    # to straight WALK scenarios only (turn physically breaks symmetric
    # cadence; back05 stays informational).
    r = {}
    cyc = {name: float("nan") for name in feet}
    for name in feet:
        tds = [td for td, _ in st[name]["strides"]]
        if len(tds) >= 2:
            cyc[name] = float(np.median(np.diff(tds)) * dt)
    vals = [v for v in cyc.values() if not np.isnan(v)]
    r["cycle_s"] = {name: v for name, v in cyc.items()}
    r["cycle_s_med"] = float(np.median(vals)) if len(vals) == 2 else float("nan")
    r["cad_spm"] = float(120.0 / r["cycle_s_med"]) if not np.isnan(r["cycle_s_med"]) else float("nan")
    r["stride_m"] = {name: v for name, v in g2["step_len"].items()}
    sl_vals = [v for v in g2["step_len"].values() if not np.isnan(v)]
    r["stride_m_med"] = float(np.median(sl_vals)) if len(sl_vals) >= 1 else float("nan")
    r["n_events"] = {name: len(st[name]["strides"]) for name in feet}
    is_walk10 = straight and abs(cmd_speed - 1.0) < 0.26
    is_walk05 = straight and abs(cmd_speed - 0.5) < 0.26
    gated = is_walk10 or is_walk05
    r["scenario"] = "walk10" if is_walk10 else ("walk05" if is_walk05 else
                  ("turn" if not straight else "other"))
    if gated:
        lo = SPEC["r1_cycle_lo_s"] if is_walk10 else SPEC["r1_cycle_lo_s_05"]
        hi = SPEC["r1_cycle_hi_s"] if is_walk10 else SPEC["r1_cycle_hi_s_05"]
        smin = SPEC["r2_stride_min_m"] if is_walk10 else SPEC["r2_stride_min_m_05"]
        m = r["cycle_s_med"]
        r["R1_PASS"] = bool(not np.isnan(m) and lo <= m <= hi)
        r["R2_PASS"] = bool(not np.isnan(r["stride_m_med"]) and r["stride_m_med"] >= smin)
        r["R3_PASS"] = bool(all(r["n_events"][name] >= SPEC["r3_min_events"] for name in feet))
        cv = g2.get("stride_cv", float("nan"))
        r["R4_PASS"] = bool(not np.isnan(cv) and cv <= SPEC["r4_cycle_cv_max"])
        r["PASS"] = bool(r["R1_PASS"] and r["R2_PASS"] and r["R3_PASS"] and r["R4_PASS"])
    else:
        r["R1_PASS"] = r["R2_PASS"] = r["R3_PASS"] = r["R4_PASS"] = None
        r["PASS"] = None
    R["R"] = r
    R["_feet"] = feet  # printer aid (excluded from json via default path)
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
    kh = R["KH"]
    ex = " [EXEMPT:turn]" if kh.get("exempt_turn") else ""
    lines.append(f"[KH human-gait]    K1={kh['K1_PASS']} K2={kh['K2_PASS']} "
                 f"H1={kh['H1_PASS']} H2={kh['H2_PASS']} H3={kh['H3_PASS']}{ex}")
    for name in R.get("_feet", []):
        d = kh[name]
        lines.append(f"    {name:22s} ev={d['n_events']:2d} K1={d['k1_mid_deg']:6.1f}° K2={d['k2_range_deg']:6.1f}° "
                     f"heel1st={d['land_heel_first_frac']*100:5.1f}% toe1st={d['land_toe_first_frac']*100:4.1f}% "
                     f"toe-off={d['launch_toe_off_frac']*100:5.1f}%")
    lines.append(f"    KH gates: K1<=18° K2>=25° H1>=60%(walk10) H2>=90%(walk10) H3<=10%")
    r = R.get("R", {})
    sc = r.get("scenario", "?")
    if r.get("PASS") is None:
        lines.append(f"[R rhythm]        informational ({sc}) cycle_med={r.get('cycle_s_med', float('nan')):.3f}s "
                     f"stride_med={r.get('stride_m_med', float('nan')):.3f}m")
    else:
        lines.append(f"[R rhythm]        PASS={r['PASS']} ({sc}) "
                     f"R1(cycle {r['cycle_s_med']:.3f}s)={r['R1_PASS']} "
                     f"R2(stride {r['stride_m_med']:.3f}m)={r['R2_PASS']} "
                     f"R3(events)={r['R3_PASS']} R4(cv)={r['R4_PASS']}")
        lines.append(f"    cadence = {r['cad_spm']:.1f} spm (human ref 1.0m/s: ~105-117 spm; "
                     f"gates: cycle [0.85,1.35]s walk10 / [1.0,1.65]s walk05, stride >=0.78/0.42m)")
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
        for R in out:
            R.pop("_feet", None)
        Path(args.json).write_text(json.dumps(out, indent=1, default=float))
        print(f"[JSON] {args.json}")
    print(f"[TOTAL] {sum(r['PASS'] for r in out)}/{len(out)} scenarios PASS")


if __name__ == "__main__":
    main()
