#!/usr/bin/env python3
"""Fix the pathological arm/torso joint decomposition in retargeted X1 clips (v30).

ROOT CAUSE (v29 diagnosis): GMR auto-IK matches key-body GEOMETRY but the
joint-space solution is contorted:
  - elbow_pitch pinned at 105-113 deg (limit 114.6): "straight arm" decomposed
    as hyper-extended elbow + shoulder_yaw ~ -32 deg internal rotation
  - lumbar_yaw swings 46-56 deg (3x the human spine-chain 14-18 deg) which
    dominates world-frame upper-body motion -> policy "swings via the waist"
  - the AMP discriminator observes joint_pos, so it learns the contortion as
    STYLE (x1_gmr vs x1_lab identical -> lab conversion is lossless).

METHOD: per-frame damped Gauss-Newton re-solve of 11 joints
  [lumbar_yaw, (shoP shoR shoY elbP elbY) x L/R]
  - HARD: preserve the ELBOW HINGE (elbow_pitch_link origin = anatomical elbow
    GMR matched to SMPLX). NOT the key body elbow_yaw_link (its origin rides
    0.117 m down the forearm and preserving it RE-CREATES the contortion).
  - priors: elbow_pitch -> 20 deg, shoulder_yaw/elbow_yaw -> 0,
    lumbar_yaw -> clip-median + 35% of the original swing (56 -> ~20 deg).
  - wrist position / forearm direction deliberately NOT tracked: GMR reached
    the SMPLX wrist endpoint VIA the contortion (X1 arms are scaled 0.75 vs
    legs); any wrist matching drags the solution back to elbow_pitch ~ 106.
  - damped GN + backtracking line search (v30.1: full Newton steps of ~4 rad
    diverge; median-reset init throws elbows ~10 cm off).

Kinematic fallback ladder (some clips physically cannot compress the waist
because shoulder_roll saturates at its abduction limit):
  tier 1: lumbar swing compressed to 35% | tier 2: 65% | tier 3: original.
  Pick the first tier whose elbow-hinge drift p95 <= 15 mm (arm trajectory
  correctness dominates); tier >= 2 is reported as PARTIAL.

Gate (exit 1): any clip with elbow drift > 15 mm, elbP p95 > 65 deg,
|shoY| mean > 15 deg, or stored-FK inconsistency > 15 mm.

Usage:
  python fix_arm_decomposition.py [--src .../x1_lab] [--dst .../x1_lab_v30] [-j N]
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
sys.path.insert(0, str(ROOT / "robolab/scripts/tools"))
from sim2sim.mujoco_rollout import build_model, DEFAULT_Q, parse_yaml_list  # noqa: E402

XML = ROOT.parent / "gmr_x1_assets" / "x1.xml"
YAML = ROOT / "robolab/scripts/tools/retarget/config/x1.yaml"
# schema order of key_body_pos columns in the lab pkl (= yaml lab_key_body_names)
KEY_BODIES = ["left_knee_pitch_link", "right_knee_pitch_link",
              "left_ankle_roll_link", "right_ankle_roll_link",
              "left_elbow_yaw_link", "right_elbow_yaw_link"]

# anisotropic elbow-hinge weights in the ROOT frame: the fore/aft component
# IS the arm-swing signal (hard); relaxing vertical/lateral (visually
# invisible) buys waist compression against the arm-length constraint (lumY
# rotation moves the shoulder while the elbow is fixed -> |S'-E| must stay
# 0.242 m).
W_ELBOW_XYZ = np.array([250.0, 60.0, 60.0])
K_ELBOW_P = 5.0     # /rad pull elbow_pitch -> target
T_ELBOW_P = np.radians(20.0)
K_SHO_YAW = 2.5     # /rad pull shoulder_yaw -> 0
K_ELB_YAW = 1.5     # /rad pull elbow_yaw -> 0
K_LUM_YAW = 30.0    # /rad pull lumbar_yaw toward the compressed target
TIERS = [0.35, 0.65, 1.0]
GN_ITERS = 60
GN_DAMP = 1e-3
MAX_BACKTRACK = 8

VAR = (["lumbar_yaw_joint"]
       + [f"{s}_{j}" for s in ("left", "right")
          for j in ("shoulder_pitch_joint", "shoulder_roll_joint",
                    "shoulder_yaw_joint", "elbow_pitch_joint", "elbow_yaw_joint")])

_CTX = {}


def _setup():
    """Per-process lazy init of the MuJoCo model and index tables."""
    import mujoco
    if _CTX:
        return _CTX
    lab_names = parse_yaml_list(YAML, "lab_dof_names")
    idx = {n: i for i, n in enumerate(lab_names)}
    model, _ = build_model(XML)
    data = mujoco.MjData(model)
    jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in lab_names}
    _CTX.update(
        mujoco=mujoco, model=model, data=data, idx=idx,
        qadr=np.array([model.jnt_qposadr[jid[n]] for n in lab_names]),
        dofadr=np.array([model.jnt_dofadr[jid[n]] for n in lab_names]),
        jrange={n: model.jnt_range[jid[n]] for n in lab_names},
        bid={b: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in
             KEY_BODIES + ["left_elbow_pitch_link", "right_elbow_pitch_link",
                           "left_shoulder_pitch_link", "right_shoulder_pitch_link",
                           "left_wrist_roll_link", "right_wrist_roll_link"]},
        VI=np.array([idx[n] for n in VAR]),
        jacp=np.zeros((3, model.nv)), jacr=np.zeros((3, model.nv)),
    )
    return _CTX


def process_clip(job):
    path_str, dst_str = job
    path, dst = Path(path_str), Path(dst_str)
    c = _setup()
    mujoco, model, data, idx = c["mujoco"], c["model"], c["data"], c["idx"]
    qadr, dofadr, jrange, bid, VI = c["qadr"], c["dofadr"], c["jrange"], c["bid"], c["VI"]
    jacp, jacr = c["jacp"], c["jacr"]

    def set_pose(root_pos, quat_wxyz, dof):
        data.qpos[:] = 0
        data.qpos[0:3] = root_pos
        data.qpos[3:7] = quat_wxyz / np.linalg.norm(quat_wxyz)
        data.qpos[qadr] = dof
        mujoco.mj_forward(model, data)

    def pos(body):
        return data.xpos[bid[body]].copy()

    def frame_jac(body):
        mujoco.mj_jacBody(model, data, jacp, jacr, bid[body])
        return jacp[:, dofadr[VI]].copy()

    def clip_limits(x):
        return np.array([np.clip(x[k], *jrange[n]) for k, n in enumerate(VAR)])

    clip = pickle.load(open(path, "rb"))
    q_orig = np.asarray(clip["dof_pos"], dtype=np.float64)
    rp = np.asarray(clip["root_pos"], dtype=np.float64)
    rr = np.asarray(clip["root_rot"], dtype=np.float64)
    T = len(q_orig)

    # ---- targets from the original (contorted) pose ----------------------
    tgt_elb, kb_stored_err = [], []
    for t in range(T):
        set_pose(rp[t], rr[t], q_orig[t])
        tgt_elb.append(np.array([pos("left_elbow_pitch_link"), pos("right_elbow_pitch_link")]))
        kb_stored_err.append(max(
            np.linalg.norm(pos(b) - clip["key_body_pos"][t, k])
            for k, b in enumerate(KEY_BODIES)))
    tgt_elb = np.array(tgt_elb)
    kb_stored_err = float(np.percentile(kb_stored_err, 95))
    lum_med = float(np.median(q_orig[:, idx["lumbar_yaw_joint"]]))

    def solve_tier(compress):
        q = q_orig.copy()
        lum_tgt = lum_med + compress * (q[:, idx["lumbar_yaw_joint"]] - lum_med)
        q[:, idx["lumbar_yaw_joint"]] = lum_tgt
        for n, v in [("elbow_pitch_joint", T_ELBOW_P), ("shoulder_yaw_joint", 0.0),
                     ("elbow_yaw_joint", 0.0)]:
            for s in ("left", "right"):
                q[:, idx[f"{s}_{n}"]] = v

        def residuals(t, x):
            q[t, VI] = x
            set_pose(rp[t], rr[t], q[t])
            w, x_, y_, z_ = rr[t] / np.linalg.norm(rr[t])
            R = np.array([
                [1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - w * z_), 2 * (x_ * z_ + w * y_)],
                [2 * (x_ * y_ + w * z_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - w * x_)],
                [2 * (x_ * z_ - w * y_), 2 * (y_ * z_ + w * x_), 1 - 2 * (x_ * x_ + y_ * y_)]])
            WA = R.T @ np.diag(W_ELBOW_XYZ)
            r, J = [], []
            for s_i, eb in enumerate(["left_elbow_pitch_link", "right_elbow_pitch_link"]):
                r.append(WA @ (pos(eb) - tgt_elb[t, s_i]))
                J.append(WA @ frame_jac(eb))
            rp_, Jp_ = [], []

            def prior(name, val, k):
                rp_.append(float(val))
                Jrow = np.zeros(11)
                Jrow[VAR.index(name)] = k
                Jp_.append(Jrow)

            for s in ("left", "right"):
                prior(f"{s}_elbow_pitch_joint",
                      K_ELBOW_P * (x[VAR.index(f"{s}_elbow_pitch_joint")] - T_ELBOW_P), K_ELBOW_P)
                prior(f"{s}_shoulder_yaw_joint",
                      K_SHO_YAW * x[VAR.index(f"{s}_shoulder_yaw_joint")], K_SHO_YAW)
                prior(f"{s}_elbow_yaw_joint",
                      K_ELB_YAW * x[VAR.index(f"{s}_elbow_yaw_joint")], K_ELB_YAW)
            prior("lumbar_yaw_joint",
                  K_LUM_YAW * (x[VAR.index("lumbar_yaw_joint")] - lum_tgt[t]), K_LUM_YAW)
            return (np.concatenate([np.atleast_1d(np.asarray(e, dtype=float))
                                    for e in r + rp_]),
                    np.vstack(J + Jp_))

        for t in range(T):
            x = q[t, VI].copy()
            r, J = residuals(t, x)
            cost = 0.5 * float(r @ r)
            lam = GN_DAMP
            for _ in range(GN_ITERS):
                g = J.T @ r
                step = np.linalg.solve(J.T @ J + lam * np.eye(11), g)
                alpha, accepted = 1.0, False
                for _ in range(MAX_BACKTRACK):
                    x_new = clip_limits(x - alpha * step)
                    r_new, J_new = residuals(t, x_new)
                    cost_new = 0.5 * float(r_new @ r_new)
                    if cost_new <= cost - 1e-4 * alpha * float(g @ step):
                        accepted = True
                        break
                    alpha *= 0.5
                if not accepted:
                    lam *= 10.0
                    if lam > 1e6:
                        break
                    continue
                done = np.max(np.abs(x_new - x)) < 1e-7
                x, r, J, cost = x_new, r_new, J_new, cost_new
                lam = max(lam / 3.0, 1e-4)
                if done or cost < 1e-8:
                    break
            q[t, VI] = x

        # temporal smoothing on corrected channels (anti-jitter)
        for _ in range(2):
            qs = q.copy()
            q[1:-1, VI] = 0.25 * qs[:-2, VI] + 0.5 * qs[1:-1, VI] + 0.25 * qs[2:, VI]

        # metrics
        d_elb, d_elb_fwd, bends = [], [], []
        for t in range(T):
            set_pose(rp[t], rr[t], q[t])
            w, x_, y_, z_ = rr[t] / np.linalg.norm(rr[t])
            R = np.array([
                [1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - w * z_), 2 * (x_ * z_ + w * y_)],
                [2 * (x_ * y_ + w * z_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - w * x_)],
                [2 * (x_ * z_ - w * y_), 2 * (y_ * z_ + w * x_), 1 - 2 * (x_ * x_ + y_ * y_)]])
            for s_i, (sp, eb, wr) in enumerate(
                    [("left_shoulder_pitch_link", "left_elbow_pitch_link", "left_wrist_roll_link"),
                     ("right_shoulder_pitch_link", "right_elbow_pitch_link", "right_wrist_roll_link")]):
                d_elb.append(np.linalg.norm(pos(eb) - tgt_elb[t, s_i]))
                d_elb_fwd.append(abs((R.T @ (pos(eb) - tgt_elb[t, s_i]))[0]))
                u1, u2 = pos(eb) - pos(sp), pos(wr) - pos(eb)
                cc = np.dot(u1, u2) / np.linalg.norm(u1) / np.linalg.norm(u2)
                bends.append(np.degrees(np.arccos(np.clip(cc, -1, 1))))
        m = dict(
            d_elb=float(np.percentile(d_elb_fwd, 95)),
            bend=float(np.percentile(bends, 95)),
            elbp95=float(np.percentile(np.degrees(np.maximum(
                q[:, idx["left_elbow_pitch_joint"]], q[:, idx["right_elbow_pitch_joint"]])), 95)),
            shoy=float(np.abs(np.degrees(np.concatenate(
                [q[:, idx["left_shoulder_yaw_joint"]], q[:, idx["right_shoulder_yaw_joint"]]]))).mean()),
            lumsw=float(np.percentile(np.degrees(q[:, idx["lumbar_yaw_joint"]]), 95)
                       - np.percentile(np.degrees(q[:, idx["lumbar_yaw_joint"]]), 5)),
            shr_lim=100.0 * float(np.mean([
                np.mean((np.abs(q[:, idx[n]] - jrange[n][0]) < 1e-3) |
                        (np.abs(q[:, idx[n]] - jrange[n][1]) < 1e-3))
                for n in ["left_shoulder_roll_joint", "right_shoulder_roll_joint"]])),
        )
        return q, m

    chosen = None
    for tier_i, compress in enumerate(TIERS):
        q_sol, m = solve_tier(compress)
        chosen = (tier_i, compress, q_sol, m)
        if m["d_elb"] <= 0.015:
            break

    tier_i, compress, q, m = chosen
    tier_i += 1  # 1-based for reporting

    # ---- recompute key bodies ---------------------------------------------
    kb = np.zeros((T, 6, 3))
    for t in range(T):
        set_pose(rp[t], rr[t], q[t])
        for k, b in enumerate(KEY_BODIES):
            kb[t, k] = pos(b)

    out = dict(clip)
    out["dof_pos"] = q
    out["key_body_pos"] = kb
    with open(dst / path.name, "wb") as f:
        pickle.dump(out, f)

    hard_checks = {
        "elbow_drift<=15mm": m["d_elb"] <= 0.015,
        "elbP_p95<=65deg": m["elbp95"] <= 65.0,
        "shoY_mean<=15deg": m["shoy"] <= 15.0,
        "storedFK<=15mm": kb_stored_err <= 0.015,
    }
    posture_ok = m["lumsw"] <= 32.0
    ok = all(hard_checks.values())
    tag = "OK " if (ok and posture_ok and tier_i == 1) else ("PRTL" if ok else "FAIL")

    line = (f"[{tag}] {path.stem:26s} t{tier_i} c={compress:.2f} "
            f"elbDrift {m['d_elb']*1000:5.1f}mm bend95 {m['bend']:5.1f}deg | "
            f"elbP95 {m['elbp95']:5.1f} shoY {m['shoy']:4.1f} lumYsw {m['lumsw']:5.1f} | "
            f"shoR@lim {m['shr_lim']:4.1f}% srcFK {kb_stored_err*1000:4.1f}mm")
    return ok, line, " ".join(k for k, v in hard_checks.items() if not v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "robolab/data/motions/x1_lab"))
    ap.add_argument("--dst", default=str(ROOT / "robolab/data/motions/x1_lab_v30"))
    ap.add_argument("-j", "--jobs", type=int, default=min(8, mp.cpu_count()))
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    clips = sorted(p for p in src.glob("*.pkl") if not p.stem.endswith("_mirror"))
    print(f"[INFO] {len(clips)} clips {src} -> {dst} (jobs={args.jobs})")
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
    print(f"[{'PASS' if ok_all else 'FAIL'}] fixed clips -> {dst}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
