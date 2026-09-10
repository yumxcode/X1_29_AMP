#!/usr/bin/env python3
"""X1 retarget GAIT-QUALITY acceptance gate (v31, mandatory).

Complements check_retarget.py (data integrity, groups A-G) with the posture/
ground/coordination gates developed in the v29-v31 reference rebuild (see
RETARGET_ACCEPTANCE.md groups H & I). Every threshold below is calibrated on
the shipped x1_lab_v31 dataset (measured values in the comments) — a future
dataset must clear them to enter training.

Checks per clip (FAIL blocks training, WARN reported only, --strict upgrades):
  S  structure      file set = N sources + N verified mirrors; pkl schema;
                    quaternion norms (wxyz); finite dof; env-cfg weight table
                    <-> file set bidirectional consistency
  F  FK fidelity    FK(dof, root) vs stored key_body_pos p95 <= 15 mm
                    (column-permutation / silent-fancy-index immune gate)
  A  arms           elbow_pitch p95 <= 65 deg (measured 20) | |shoulder_yaw|
                    mean <= 15 deg (0-2) | |elbow_yaw| mean <= 20 (0-2) |
                    geometric elbow bend p95 <= 45 deg (~41) | shoulder-pitch
                    L/R antiphase corr + arm-opposite-leg coupling: FAIL
                    bound for style-dominant classes (-0.45 / +0.45),
                    (138_18 was REJECTED by this gate on first run: GMR
                    flipped its source-symmetric swing to SAME-PHASE in X1
                    joint space, anti +0.47 / coupling -0.20 — do not re-add
                    without re-running this gate)
  T  torso          lumbar_yaw swing (p95-p5) per clip class:
                    PRIMARY<=32 (19-25) JOG<=65 (57) CMU_OLD<=55 (39-43)
                    CMU_NEW<=75 (62-68, real counter-rotation by design)
  G  ground         sole penetration >= -3 mm everywhere (0.0 after fix) |
                    stance sole-pitch |L-R| median diff <= 3.5/6 deg |
                    stance |pitch| median <= class limit
  C  coordination   hip L/R corr >= +0.3 (anti-aligned axes, mirrored gait) |
                    hip swing ratio L/R per class: PRIMARY/JOG [0.7, 1.4]
                    (retarget-introduced asymmetry; the rejected 127_04 was
                    1.86) | CIRCLE [0.6, 1.6] (curved walking is inherently
                    asymmetric, 0026 measures 1.44) | CMU_OLD [0.5, 2.2] WARN
                    (source-inherent asymmetry, compensated by mirror pairs —
                    36_01 measures 2.05, kept since v16)
  M  mirror         per-source mirror exists; mirror key_body == y-mirror of
                    source with L/R body swap (p95 <= 20 mm)

Usage:
  python check_retarget_gait.py [--dir .../x1_lab_v31] [--json out.json]
                                [--strict] [--repo-root <X1_29_AMP>]
Exit: 0 PASS, 1 FAIL, 2 setup error.
"""
import argparse
import functools
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent.parent

# clip classes (threshold tiers); every source clip MUST be classified
CLASSES = {
    "PRIMARY": {"0000_treadmill_norm", "0002_treadmill_slow", "0005_normal_walk1",
                "0007_normal_walk3", "0008_normal_walk4"},
    "CIRCLE": {"0026_circle_walk"},
    "JOG": {"0003_treadmill_jog", "0009_normal_jog1"},
    "CMU_OLD": {"36_01", "36_11"},
    "CMU_NEW": {"103_07"},
}
LUMY_SWING = {"PRIMARY": 32.0, "CIRCLE": 32.0, "JOG": 65.0,
              "CMU_OLD": 55.0, "CMU_NEW": 75.0}
STANCE_PITCH_ABS = {"PRIMARY": 15.0, "CIRCLE": 15.0, "JOG": 25.0,
                    "CMU_OLD": 20.0, "CMU_NEW": 25.0}
LR_PITCH_DIFF = {"PRIMARY": 3.5, "CIRCLE": 3.5, "JOG": 6.0,
                 "CMU_OLD": 5.0, "CMU_NEW": 6.0}
HIP_RATIO = {"PRIMARY": (0.7, 1.4), "CIRCLE": (0.6, 1.6), "JOG": (0.7, 1.4),
             "CMU_OLD": (0.5, 2.2), "CMU_NEW": (0.6, 1.5)}
HIP_RATIO_WARN_ONLY = {"CMU_OLD"}          # source-inherent, mirrors compensate
# arm coordination: (fail_threshold, warn_threshold) per class
ANTI_BOUNDS = {c: (-0.45, -0.45) for c in
               ("PRIMARY", "CIRCLE", "JOG", "CMU_OLD", "CMU_NEW")}
ARMLEG_BOUNDS = {c: (0.45, 0.45) for c in
                 ("PRIMARY", "CIRCLE", "JOG", "CMU_OLD", "CMU_NEW")}

KEY_BODIES = ["left_knee_pitch_link", "right_knee_pitch_link",
              "left_ankle_roll_link", "right_ankle_roll_link",
              "left_elbow_yaw_link", "right_elbow_yaw_link"]
LR_SWAP = [1, 0, 3, 2, 5, 4]          # L/R body index swap for mirror check
SOLE_LEN = 0.14
CONTACT_ON, CONTACT_OFF, MIN_STANCE = 0.03, 0.08, 3

_CTX = {}


def setup(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "sim2sim"))
    from sim2sim.mujoco_rollout import build_model, DEFAULT_Q, parse_yaml_list, find_sole_geoms
    import mujoco
    yaml = repo_root / "roboparty_train/robolab/scripts/tools/retarget/config/x1.yaml"
    lab = parse_yaml_list(yaml, "lab_dof_names")
    model, _ = build_model(repo_root / "gmr_x1_assets" / "x1.xml")
    data = mujoco.MjData(model)
    jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in lab}
    soles, _ = find_sole_geoms(model)
    feet = sorted(soles)
    _CTX.update(
        mujoco=mujoco, model=model, data=data,
        idx={n: i for i, n in enumerate(lab)},
        qadr=np.array([model.jnt_qposadr[jid[n]] for n in lab]),   # LAB order (by name!)
        kb_bid=[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in KEY_BODIES],
        sole_geoms=[np.asarray(soles[f]) for f in feet],
        sole_r=float(model.geom_size[int(np.asarray(soles[feet[0]])[0]), 0]),
        shp_bid=[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in
                 ["left_shoulder_pitch_link", "right_shoulder_pitch_link",
                  "left_elbow_pitch_link", "right_elbow_pitch_link",
                  "left_wrist_roll_link", "right_wrist_roll_link"]],
    )


def corr(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-12 else 0.0


def rng(x):
    x = np.asarray(x)
    return float(np.percentile(x, 95) - np.percentile(x, 5))


def schmitt_stance(low_bottom):
    T = len(low_bottom)
    st = np.zeros(T, bool)
    in_st, start = False, 0
    for t in range(T):
        if not in_st and low_bottom[t] < CONTACT_ON:
            in_st, start = True, t
        elif in_st and low_bottom[t] > CONTACT_OFF:
            if t - start >= MIN_STANCE:
                st[start:t] = True
            in_st = False
    if in_st and T - start >= MIN_STANCE:
        st[start:T] = True
    return st


def fk_metrics(clip):
    """FK every (downsampled) frame; returns all physical metrics."""
    c = _CTX
    mujoco, model, data = c["mujoco"], c["model"], c["data"]
    q = np.asarray(clip["dof_pos"], float)
    rp = np.asarray(clip["root_pos"], float)
    rr = np.asarray(clip["root_rot"], float)
    kb = np.asarray(clip["key_body_pos"], float)
    T = len(q)
    stride = max(1, T // 400)                    # sample ~400 frames
    ts = list(range(0, T, stride))

    kb_err, low, pitch = [], np.zeros((len(ts), 2)), np.zeros((len(ts), 2))
    kb_pos = np.zeros((len(ts), 6, 3))
    for i, t in enumerate(ts):
        data.qpos[:] = 0
        data.qpos[0:3] = rp[t]
        data.qpos[3:7] = rr[t] / np.linalg.norm(rr[t])
        data.qpos[c["qadr"]] = q[t]
        mujoco.mj_forward(model, data)
        cur = np.array([data.xpos[b] for b in c["kb_bid"]])
        kb_pos[i] = cur
        kb_err.append(np.abs(cur - kb[t]).max())
        for f in range(2):
            pts = np.array([data.geom_xpos[g] for g in c["sole_geoms"][f]])
            low[i, f] = pts[:, 2].min()
            front, back = pts[:2].mean(0), pts[2:].mean(0)
            pitch[i, f] = np.degrees(np.arctan2(front[2] - back[2], SOLE_LEN))
    r = c["sole_r"]
    bottom = low - r
    stance = np.array([schmitt_stance(bottom[:, f]) for f in range(2)])

    # geometric elbow bend (shoulder->elbow->wrist angle) on 60 sampled frames
    bends = []
    for i in range(0, len(ts), max(1, len(ts) // 60)):
        t = ts[i]
        data.qpos[:] = 0
        data.qpos[0:3] = rp[t]
        data.qpos[3:7] = rr[t] / np.linalg.norm(rr[t])
        data.qpos[c["qadr"]] = q[t]
        mujoco.mj_forward(model, data)
        for s_, sp_, eb_, wr_ in [(0, 0, 2, 4), (1, 1, 3, 5)]:
            S, E, W = (data.xpos[c["shp_bid"][k]] for k in (sp_, eb_, wr_))
            u1, u2 = E - S, W - E
            cc = np.dot(u1, u2) / np.linalg.norm(u1) / np.linalg.norm(u2)
            bends.append(np.degrees(np.arccos(np.clip(cc, -1, 1))))

    idx = c["idx"]
    m = dict(
        kb_err_p95=float(np.percentile(kb_err, 95)),
        pen_min=float(bottom.min()),
        elbp95=float(np.percentile(np.degrees(np.maximum(
            q[:, idx["left_elbow_pitch_joint"]], q[:, idx["right_elbow_pitch_joint"]])), 95)),
        shoy=float(np.abs(np.degrees(np.concatenate(
            [q[:, idx["left_shoulder_yaw_joint"]], q[:, idx["right_shoulder_yaw_joint"]]]))).mean()),
        elby=float(np.abs(np.degrees(np.concatenate(
            [q[:, idx["left_elbow_yaw_joint"]], q[:, idx["right_elbow_yaw_joint"]]]))).mean()),
        bend95=float(np.percentile(bends, 95)),
        anti=corr(q[:, idx["left_shoulder_pitch_joint"]], q[:, idx["right_shoulder_pitch_joint"]]),
        lumsw=rng(np.degrees(q[:, idx["lumbar_yaw_joint"]])),
        hipcorr=corr(q[:, idx["left_hip_pitch_joint"]], q[:, idx["right_hip_pitch_joint"]]),
        armleg=corr(q[:, idx["left_shoulder_pitch_joint"]], q[:, idx["right_hip_pitch_joint"]]),
        hipratio=rng(q[:, idx["left_hip_pitch_joint"]]) / max(rng(q[:, idx["right_hip_pitch_joint"]]), 1e-9),
    )
    for f in range(2):
        st = stance[f]
        m[f"pitch_med_{f}"] = float(np.median(pitch[st, f])) if st.any() else float("nan")
        m[f"pitch_abs_{f}"] = abs(m[f"pitch_med_{f}"])
    m["stance_frac"] = float(stance.mean())
    return m, kb_pos


def check_clip(name, cls, clip, mirror_clip):
    """Returns (rows, n_fail, n_warn): rows = [(id, verdict, detail)]."""
    m, _ = fk_metrics(clip)
    rows = []

    def add(cid, ok, detail, warn=False):
        rows.append((cid, ("PASS" if ok else ("WARN" if warn else "FAIL")), detail))
        return 0 if ok else (0 if warn else 1)

    nf = 0
    nf += add("A1", m["elbp95"] <= 65.0, f"elbP p95 {m['elbp95']:.1f} deg <= 65 (healthy 20)")
    nf += add("A2", m["shoy"] <= 15.0, f"|shoY| mean {m['shoy']:.1f} deg <= 15 (twist comp cleared)")
    nf += add("A3", m["elby"] <= 20.0, f"|elbY| mean {m['elby']:.1f} deg <= 20")
    nf += add("A4", m["bend95"] <= 45.0, f"geom elbow bend p95 {m['bend95']:.1f} deg <= 45")
    anti_fail, anti_warn = ANTI_BOUNDS[cls]
    anti_ok = m["anti"] <= anti_fail
    nf += add("A5", anti_ok or m["anti"] <= anti_warn,
              f"shoulder L/R antiphase {m['anti']:+.2f} (<= -0.45; walk refs -0.63..-0.99)",
              warn=(not anti_ok) and m["anti"] <= anti_warn)
    nf += add("T1", m["lumsw"] <= LUMY_SWING[cls],
              f"lumY swing {m['lumsw']:.1f} deg <= {LUMY_SWING[cls]} [{cls}]")
    nf += add("G1", m["pen_min"] >= -0.003,
              f"sole penetration min {m['pen_min']*1000:.1f} mm >= -3")
    lrd = abs(m["pitch_med_0"] - m["pitch_med_1"])
    nf += add("G2", lrd <= LR_PITCH_DIFF[cls],
              f"stance pitch |L-R| {lrd:.1f} deg <= {LR_PITCH_DIFF[cls]} (L {m['pitch_med_0']:+.1f} R {m['pitch_med_1']:+.1f})")
    nf += add("G3", max(m["pitch_abs_0"], m["pitch_abs_1"]) <= STANCE_PITCH_ABS[cls],
              f"stance |pitch| med max {max(m['pitch_abs_0'], m['pitch_abs_1']):.1f} deg <= {STANCE_PITCH_ABS[cls]}")
    nf += add("C1", m["hipcorr"] >= +0.30,
              f"hip L/R corr {m['hipcorr']:+.2f} >= +0.3 (anti-aligned axes; symmetric gait)")
    al_fail, al_warn = ARMLEG_BOUNDS[cls]
    al_ok = m["armleg"] >= al_fail
    nf += add("C2", al_ok or m["armleg"] >= al_warn,
              f"arm-vs-opposite-leg coupling {m['armleg']:+.2f} (>= +0.45; refs 0.8-0.98)",
              warn=(not al_ok) and m["armleg"] >= al_warn)
    lo, hi = HIP_RATIO[cls]
    ratio_ok = lo <= m["hipratio"] <= hi
    nf += add("C3", ratio_ok or cls in HIP_RATIO_WARN_ONLY,
              f"hip swing ratio L/R {m['hipratio']:.2f} in [{lo}, {hi}] [{cls}]"
              f"{' (WARN-only class: source-inherent, mirrors compensate)' if cls in HIP_RATIO_WARN_ONLY else ''}",
              warn=(not ratio_ok) and cls in HIP_RATIO_WARN_ONLY)
    # FK fidelity
    nf += add("F1", m["kb_err_p95"] <= 0.015,
              f"FK vs stored key_body p95 {m['kb_err_p95']*1000:.1f} mm <= 15")
    # mirror correctness (independent recomputation)
    if mirror_clip is not None:
        src_kb = np.asarray(clip["key_body_pos"], float)
        mir_kb = np.asarray(mirror_clip["key_body_pos"], float)
        T = min(len(src_kb), len(mir_kb))
        want = src_kb[:T][:, LR_SWAP].copy()
        want[:, :, 1] *= -1
        e = float(np.percentile(np.linalg.norm(mir_kb[:T] - want, axis=2), 95))
        nf += add("M1", e <= 0.020, f"mirror key_body y-flip+swap p95 {e*1000:.1f} mm <= 20")
    else:
        rows.append(("M1", "FAIL", "mirror clip missing"))
        nf += 1
    return rows, nf, sum(1 for _, v, _ in rows if v == "WARN")


def weight_table_names(repo_root: Path):
    cfg = repo_root / "roboparty_train/robolab/robolab/tasks/manager_based/amp/x1_amp_env_cfg.py"
    if not cfg.exists():
        return None
    src = cfg.read_text()
    m = re.search(r"motion_data_weights\s*=\s*\{(.*?)\}", src, re.S)
    if not m:
        return None
    return {n: float(w) for n, w in re.findall(r'"([^"]+)":\s*([\d.]+)', m.group(1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "roboparty_train/robolab/data/motions/x1_lab_v31"))
    ap.add_argument("--repo-root", default=str(ROOT))
    ap.add_argument("--json", default=None)
    ap.add_argument("--strict", action="store_true", help="WARN -> FAIL")
    args = ap.parse_args()

    d = Path(args.dir)
    repo = Path(args.repo_root)
    try:
        setup(repo)
    except Exception as e:
        print(f"[SETUP-ERROR] {e}")
        sys.exit(2)

    files = sorted(p.stem for p in d.glob("*.pkl"))
    sources = [f for f in files if not f.endswith("_mirror")]
    mirrors = {f[:-7] for f in files if f.endswith("_mirror")}
    report = {"dir": str(d), "clips": {}, "summary": {}}
    total_fail = total_warn = 0

    print(f"[GATE] retarget gait-quality gate on {d.name}: {len(sources)} sources / {len(files)} files\n")

    # ---- S structure -----------------------------------------------------
    s_rows = []
    s_fail = 0
    unclassified = [s for s in sources if s not in {x for c in CLASSES.values() for x in c}]
    if unclassified:
        s_rows.append(("S1", "FAIL", f"unclassified clips (add to CLASSES or reject): {unclassified}"))
        s_fail += 1
    else:
        s_rows.append(("S1", "PASS", f"all {len(sources)} sources classified"))
    missing_mirror = [s for s in sources if s not in mirrors]
    orphan_mirror = [m for m in mirrors if m not in set(sources)]
    ok = not missing_mirror and not orphan_mirror
    s_fail += 0 if ok else 1
    s_rows.append(("S2", "PASS" if ok else "FAIL",
                   f"mirror pairs: {len(sources) - len(missing_mirror)}/{len(sources)}"
                   + (f" missing={missing_mirror}" if missing_mirror else "")
                   + (f" orphan={orphan_mirror}" if orphan_mirror else "")))
    wt = weight_table_names(repo)
    if wt is None:
        s_rows.append(("S3", "WARN", "weight table not parseable from env cfg"))
    else:
        fset = set(files)
        dangling = [n for n in wt if n not in fset]
        unweighted = [f for f in files if f not in wt]
        ok = not dangling and not unweighted
        s_fail += 0 if ok else 1
        s_rows.append(("S3", "PASS" if ok else "FAIL",
                       f"weights<->files 1:1 ({len(wt)} entries)"
                       + (f" dangling={dangling}" if dangling else "")
                       + (f" unweighted={unweighted}" if unweighted else "")))
    for cid, v, det in s_rows:
        mark = {"PASS": "  ", "WARN": " ~ ", "FAIL": " X "}[v]
        print(f"  {cid}{mark}{det}")
        if v == "FAIL":
            total_fail += 1
        elif v == "WARN":
            total_warn += 1

    # ---- per clip ---------------------------------------------------------
    for name in sources:
        cls = next(c for c, s in CLASSES.items() if name in s)
        clip = pickle.load(open(d / f"{name}.pkl", "rb"))
        mirror_clip = pickle.load(open(d / f"{name}_mirror.pkl", "rb")) \
            if (d / f"{name}_mirror.pkl").exists() else None

        # S4 schema
        shape_ok = (np.asarray(clip["dof_pos"]).shape[1] == 29
                    and np.asarray(clip["key_body_pos"]).shape[1:] == (6, 3)
                    and np.isfinite(clip["dof_pos"]).all()
                    and np.allclose(np.linalg.norm(clip["root_rot"], axis=1), 1.0, atol=1e-3))
        rows, nf, nw = check_clip(name, cls, clip, mirror_clip)
        if not shape_ok:
            rows.append(("S4", "FAIL", "schema/quat-norm/finite violation"))
            nf += 1
        total_fail += nf
        total_warn += nw
        if args.strict:
            total_fail += nw
        print(f"\n  {name} [{cls}]" + ("  <== FAIL" if nf else ""))
        for cid, v, det in rows:
            mark = {"PASS": "  ", "WARN": " ~ ", "FAIL": " X "}[v]
            print(f"    {cid}{mark}{det}")
        report["clips"][name] = {"class": cls, "fails": nf, "warns": nw,
                                 "rows": [{"id": c, "verdict": v, "detail": t} for c, v, t in rows]}

    verdict = "PASS" if total_fail == 0 else "FAIL"
    report["summary"] = {"verdict": verdict, "fails": total_fail, "warns": total_warn,
                         "strict": args.strict}
    print(f"\n[GATE] {'=' * 20} {verdict} ({total_fail} fail, {total_warn} warn) {'=' * 20}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"[GATE] report -> {args.json}")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
