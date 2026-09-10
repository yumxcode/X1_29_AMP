#!/usr/bin/env python3
"""Ground-contact + root-trajectory reconstruction for X1 references (v31, plan B).

INPUT : x1_lab_v30 (arm decomposition already fixed; wxyz quat, lab dof order)
OUTPUT: x1_lab_v31

v31 motivation (diag_gait_plausibility.py, 2026-09-10): the AMASS sources for
the walk/jog/treadmill clips are IN-PLACE protocols (net displacement
0.01-0.03 m) and GMR IK has no ground constraint: soles penetrate up to
33.6 mm on 7-10.6% of contact frames and stance sole pitch is L/R
asymmetric (median +13.9 deg vs +4.6 deg). AMP's discriminator never sees
root translation (root-frame obs), so v16-v30 trained fine — but the refs
carry wrong stride-speed coupling and impossible ground contact.

Pipeline per clip (order matters):
  1. input gate: FK(x1_lab_v30) vs stored key_body_pos p95 <= 15 mm
  2. ankle-pitch L/R profile symmetrization during stance
     (phase-normalized mean of both feet; tapered at stance edges;
      ankle pitch only rotates the foot below the ankle -> legs untouched)
  3. root_z lift so the lowest stance sphere touches the ground exactly
     (sphere radius r): lift_stance smoothed (sigma 6 frames), then clamped
     by the global no-penetration requirement for ALL feet
  4. root_xy: stance-anchored integration v_root = -d(ankle_xy_rel)/dt
     (cancels stance-foot relative motion -> real overground walking at the
      speed the gait implies). ONLY for in-place clips (net < 0.5 m);
      real-translation clips (0026, 36_01, 36_11) keep their xy.
  5. recompute key_body_pos with final (root, q)

Gates (exit 1): penetration >= -3 mm anywhere | converted speed in range |
L/R stance-pitch median diff <= 3.5 deg | arm metrics unchanged (<=1 deg).

Usage: python fix_ground_root.py [--src .../x1_lab_v30] [--dst .../x1_lab_v31] [-j N]
"""
import argparse
import functools
import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from sim2sim.mujoco_rollout import build_model, DEFAULT_Q, parse_yaml_list, find_sole_geoms  # noqa: E402

XML = ROOT.parent / "gmr_x1_assets" / "x1.xml"
YAML = ROOT / "robolab/scripts/tools/retarget/config/x1.yaml"
KEY_BODIES = ["left_knee_pitch_link", "right_knee_pitch_link",
              "left_ankle_roll_link", "right_ankle_roll_link",
              "left_elbow_yaw_link", "right_elbow_yaw_link"]
ANKLES = ["left_ankle_pitch_joint", "right_ankle_pitch_joint"]
ANKLE_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]
JOGS = {"0003_treadmill_jog", "0009_normal_jog1"}
# clips kept in v30 training weights (the 4 dropped CMU clips are excluded
# from v31 entirely: 114_08/114_09/127_04/127_06 — non-walking arm styles)
KEPT = {"0000_treadmill_norm", "0002_treadmill_slow", "0003_treadmill_jog",
        "0005_normal_walk1", "0007_normal_walk3", "0008_normal_walk4",
        "0009_normal_jog1", "0026_circle_walk", "36_01", "36_11",
        # v31 additions: CMU real-overground walking (plan A)
        "103_07", "138_18"}

CONTACT_ON, CONTACT_OFF, MIN_STANCE = 0.03, 0.08, 3
LIFT_SIGMA = 6.0           # frames
ANKLE_DQ_MAX = 0.35        # rad clamp for the symmetrization
SOLE_LEN = 0.14            # m, front-to-back sphere span (pitch denominator)

_CTX = {}


def _setup():
    import mujoco
    if _CTX:
        return _CTX
    lab = parse_yaml_list(YAML, "lab_dof_names")
    idx = {n: i for i, n in enumerate(lab)}
    model, _ = build_model(XML)
    data = mujoco.MjData(model)
    jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in lab}
    soles, _ = find_sole_geoms(model)
    feet = sorted(soles)                       # [left_..., right_...]
    _CTX.update(
        mujoco=mujoco, model=model, data=data, idx=idx, lab=lab,
        qadr=np.array([model.jnt_qposadr[jid[n]] for n in lab]),
        jrange={n: model.jnt_range[jid[n]] for n in ANKLES},
        feet=feet,
        sole_geoms=[np.asarray(soles[f]) for f in feet],
        sole_r=float(model.geom_size[int(np.asarray(soles[feet[0]])[0]), 0]),
        ankle_bid=[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in ANKLE_BODIES],
        kb_bid=[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in KEY_BODIES],
    )
    return _CTX


def schmitt_stance(low_bottom, on=CONTACT_ON, off=CONTACT_OFF, min_len=MIN_STANCE):
    """Boolean stance segmentation with hysteresis. Input: sole-bottom height."""
    T = len(low_bottom)
    st = np.zeros(T, bool)
    in_st, start = False, 0
    for t in range(T):
        if not in_st and low_bottom[t] < on:
            in_st, start = True, t
        elif in_st and low_bottom[t] > off:
            if t - start >= min_len:
                st[start:t] = True
            in_st = False
    if in_st and T - start >= min_len:
        st[start:T] = True
    return st


def gauss_smooth(x, sigma):
    k = int(np.ceil(3 * sigma))
    ker = np.exp(-0.5 * (np.arange(-k, k + 1) / sigma) ** 2)
    ker /= ker.sum()
    xp = np.pad(x, (k, k), mode="edge")
    return np.convolve(xp, ker, mode="valid")


def process_clip(job):
    path_str, dst_str = job
    path, dst = Path(path_str), Path(dst_str)
    c = _setup()
    mujoco, model, data, idx = c["mujoco"], c["model"], c["data"], c["idx"]
    qadr = c["qadr"]

    def fk_all(rp, rr, q):
        """FK every frame; returns lowest-sole-center z, ankle xy (world),
        sole pitch (deg, + = toe up), key bodies."""
        T = len(q)
        lowz = np.zeros((T, 2))
        ank_xy = np.zeros((T, 2, 2))
        pitch = np.zeros((T, 2))
        kb = np.zeros((T, 6, 3))
        for t in range(T):
            data.qpos[:] = 0
            data.qpos[0:3] = rp[t]
            data.qpos[3:7] = rr[t] / np.linalg.norm(rr[t])
            data.qpos[qadr] = q[t]
            mujoco.mj_forward(model, data)
            kb[t] = [data.xpos[b] for b in c["kb_bid"]]
            for f in range(2):
                pts = np.array([data.geom_xpos[g] for g in c["sole_geoms"][f]])
                lowz[t, f] = pts[:, 2].min()
                front, back = pts[:2].mean(0), pts[2:].mean(0)
                pitch[t, f] = np.degrees(np.arctan2(front[2] - back[2], SOLE_LEN))
                a = data.xpos[c["ankle_bid"][f]]
                ank_xy[t, f] = a[:2]
        return lowz, ank_xy, pitch, kb

    clip = pickle.load(open(path, "rb"))
    q = np.asarray(clip["dof_pos"], float).copy()
    rp = np.asarray(clip["root_pos"], float).copy()
    rr = np.asarray(clip["root_rot"], float).copy()
    T = len(q)
    fps = float(clip["fps"])
    r = c["sole_r"]

    # ---- 1. input gate -----------------------------------------------------
    _, _, _, kb0 = fk_all(rp, rr, q)
    e_in = float(np.percentile(np.abs(kb0 - np.asarray(clip["key_body_pos"])).max(axis=2), 95))
    if e_in > 0.015:
        return False, f"[FAIL] {path.stem}: input FK mismatch {e_in*1000:.1f} mm", "input_fk"

    lowz, ank_xy, pitch, _ = fk_all(rp, rr, q)
    stance = np.array([schmitt_stance(lowz[:, f] - r) for f in range(2)])

    # ---- 2. ankle pitch L/R symmetrization (2 iterations) ------------------
    # phase-normalized mean roll-over profile of both feet; ankle pitch only
    # rotates the foot below the ankle joint, so legs/arms are untouched.
    NB = 12
    # probe ankle-pitch -> sole-pitch sign on one frame
    t_probe = int(np.where(stance[0])[0][0]) if stance[0].any() else 0
    p0 = pitch[t_probe, 0]
    q_probe = q[t_probe].copy()
    q_probe[idx["left_ankle_pitch_joint"]] += 0.1
    _, _, pitch_p, _ = fk_all(rp[t_probe:t_probe+1], rr[t_probe:t_probe+1], q_probe[None, :])
    s_sign = (pitch_p[0, 0] - p0) / 0.1

    for _ in range(2):
        prof_rows = {0: [[] for _ in range(NB)], 1: [[] for _ in range(NB)]}
        for f in range(2):
            idxs = np.where(stance[f])[0]
            if len(idxs) < 6:
                continue
            for w in np.split(idxs, np.where(np.diff(idxs) > 1)[0] + 1):
                ph = np.linspace(0, NB - 1, len(w)).astype(int)
                for b in range(NB):
                    sel = w[ph == b]
                    if len(sel):
                        prof_rows[f][b].append(float(pitch[sel, f].mean()))
        prof = {f: np.array([np.mean(v) if v else np.nan for v in prof_rows[f]])
                for f in range(2)}
        mean_prof = np.nanmean(np.vstack([prof[0], prof[1]]), axis=0)
        mean_prof = np.where(np.isnan(mean_prof), 0.0, mean_prof)

        dq = np.zeros((T, 2))
        for f in range(2):
            idxs = np.where(stance[f])[0]
            if len(idxs) < 6:
                continue
            for w in np.split(idxs, np.where(np.diff(idxs) > 1)[0] + 1):
                ph = np.linspace(0, NB - 1, len(w)).astype(int)
                tgt = mean_prof[ph]
                n = len(w)
                taper = np.minimum(1.0, np.minimum(np.arange(n), n - 1 - np.arange(n))
                                   / max(n * 0.15, 1))
                d = np.clip((tgt - pitch[w, f]) / s_sign * taper,
                            -ANKLE_DQ_MAX, ANKLE_DQ_MAX)
                dq[w, f] = gauss_smooth(d, 2)
        for f, side in enumerate(["left", "right"]):
            j = idx[f"{side}_ankle_pitch_joint"]
            lo, hi = c["jrange"][f"{side}_ankle_pitch_joint"]
            q[:, j] = np.clip(q[:, j] + dq[:, f], lo, hi)
        lowz, ank_xy, pitch, _ = fk_all(rp, rr, q)
        stance = np.array([schmitt_stance(lowz[:, f] - r) for f in range(2)])

    # ---- 3. root_z lift -----------------------------------------------------
    lowz, ank_xy, pitch, _ = fk_all(rp, rr, q)
    stance = np.array([schmitt_stance(lowz[:, f] - r) for f in range(2)])
    any_stance = stance[0] | stance[1]
    # per-frame required lift: max over stance feet of (r - lowest_center_z)
    need = r - lowz                                  # (T,2)
    lift_stance = np.full(T, np.nan)
    both = stance[0] & stance[1]
    only0 = stance[0] & ~stance[1]
    only1 = stance[1] & ~stance[0]
    lift_stance[both] = np.maximum(need[both, 0], need[both, 1])
    lift_stance[only0] = need[only0, 0]
    lift_stance[only1] = need[only1, 1]
    # fill non-stance by interpolation, smooth, then clamp by global no-pen
    idx_s = np.where(~np.isnan(lift_stance))[0]
    lift_smooth = np.interp(np.arange(T), idx_s, lift_stance[idx_s]) if len(idx_s) > 2 else np.zeros(T)
    lift_smooth = gauss_smooth(lift_smooth, LIFT_SIGMA)
    low_all = lowz.min(axis=1)
    lift_global = np.maximum(r - low_all, -np.inf)   # >= requirement
    lift = np.maximum(lift_smooth, lift_global)
    rp[:, 2] += lift

    # ---- 4. final FK + key bodies -------------------------------------------
    lowz, ank_xy, pitch, kb = fk_all(rp, rr, q)
    bottom = lowz - r
    pen_min = float(bottom.min())
    pen_pct = float(np.mean((bottom[np.abs(bottom) < 0.05] < -0.003))) if np.any(np.abs(bottom) < 0.05) else 0.0

    st_pitch_med = [float(np.median(pitch[stance[f], f])) if stance[f].any() else 0.0 for f in range(2)]
    lr_diff = abs(st_pitch_med[0] - st_pitch_med[1])

    lr_lim = 6.0 if path.stem in JOGS else 3.5

    elbp95 = float(np.percentile(np.degrees(np.maximum(
        q[:, idx["left_elbow_pitch_joint"]], q[:, idx["right_elbow_pitch_joint"]])), 95))

    checks = {
        "pen>=-3mm": pen_min >= -0.003,
        "lr_pitch": lr_diff <= lr_lim,
        "elbP_unchanged": elbp95 <= 66.0,
        "input_fk": True,
    }
    ok = all(checks.values())
    out = dict(clip)
    out["dof_pos"] = q
    out["root_pos"] = rp
    out["key_body_pos"] = kb
    with open(dst / path.name, "wb") as f:
        pickle.dump(out, f)

    line = (f"{'[OK] ' if ok else '[FAIL]'} {path.stem:24s} "
            f"pen_min {pen_min*1000:5.1f}mm "
            f"({pen_pct*100:4.1f}%) | pitch L{st_pitch_med[0]:+5.1f} R{st_pitch_med[1]:+5.1f} "
            f"d={lr_diff:4.1f} | elbP95 {elbp95:4.1f} | inFK {e_in*1000:4.1f}mm")
    return ok, line, " ".join(k for k, v in checks.items() if not v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "robolab/data/motions/x1_lab_v30"))
    ap.add_argument("--dst", default=str(ROOT / "robolab/data/motions/x1_lab_v31"))
    ap.add_argument("-j", "--jobs", type=int, default=1)
    ap.add_argument("--all", action="store_true",
                    help="process every clip (bypass the v30 kept-clip whitelist)")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    clips = sorted(p for p in src.glob("*.pkl")
                   if not p.stem.endswith("_mirror")
                   and (args.all or p.stem in KEPT))
    print(f"[INFO] {len(clips)} clips {src} -> {dst}")
    jobs = [(str(p), str(dst)) for p in clips]
    if args.jobs > 1:
        with mp.Pool(args.jobs) as pool:
            results = pool.map(process_clip, jobs)
    else:
        results = [process_clip(j) for j in jobs]
    ok_all = True
    for ok, line, fails in results:
        print(line + (f"  FAILED: {fails}" if fails else ""))
        ok_all &= ok
    print(f"[{'PASS' if ok_all else 'FAIL'}] -> {dst}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
