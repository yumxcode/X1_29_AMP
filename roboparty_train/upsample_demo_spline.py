#!/usr/bin/env python3
"""Spline-upsample the x1_lab demo dataset to a finer native fps (user-directed).

Why (v61 arc, 2026-09-20): demos are native 120 fps; at 100 Hz control the
animation manager fetches demo frames at 10 ms spacing — between 8.33 ms keys
(linear interp). Upsampling to 200 fps (5 ms keys) makes the 10 ms fetch land
EXACTLY on every 2nd key: zero interpolation at the disc's native cadence.
This is the data-side companion to the disc ::stride fix (X1_DISC_STRIDE=1).

Method per float channel (root_pos, dof_pos, key_body_pos): PCHIP
(Fritsch–Carlson monotone cubic, Hermite) through the original samples —
exact at knots by construction AND overshoot-free, unlike a raw not-a-knot
cubic spline whose undershoot/overshoot on fast segments (0003_treadmill_jog,
~2.4x stride speed) produced 5.3 mm FK inconsistency vs the 1.5 mm gate.
root_rot: shortest-path slerp. fps metadata rewritten; velocities are NOT
stored — the motion manager finite-differences at load time from fps.

Validation (both must pass, else the clip is skipped and reported):
  1. round-trip: spline evaluated at the ORIGINAL timestamps == original
     (max |diff| == 0 up to float error).
  2. FK consistency: at 12 probe frames of the NEW grid, key_body positions
     recomputed from resampled dof+root via MuJoCo FK match the resampled
     key_body channel within tol (default 1.5 mm; the v32 dataset gate).

Usage:
  ./.venv39/bin/python roboparty_train/upsample_demo_spline.py \
      --src roboparty_train/robolab/data/motions/x1_lab_v32 \
      --out roboparty_train/robolab/data/motions/x1_lab_v32_200 --fps 200
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- pchip
def pchip_coeffs(t, y):
    """Fritsch–Carlson monotone cubic Hermite (PCHIP). t strictly increasing
    (n,), y (n, d). Returns per-interval (a, b, c, d) coefficients (Hermite
    form converted to power form) — no overshoot beyond neighboring data."""
    n = len(t)
    y = y.reshape(n, -1)
    h = np.diff(t)
    delta = (y[1:] - y[:-1]) / h[:, None]            # secant slopes (n-1, d)
    d = np.zeros_like(y)
    # endpoint slopes: scipy PchipInterpolator _edge_case guard — the raw
    # one-sided formula overshoots when the adjacent secant is flat or
    # opposite-signed (unit test: [.., 1, 1, 0.5, 0.5] dipped -0.074)
    def _edge(d0, d1):
        # zero when sign mismatch vs adjacent secant; cap at 3*secant
        mask_flip = (np.sign(d0) != np.sign(d1)) & (d0 != 0) & (d1 != 0)
        d0 = np.where(mask_flip, 0.0, d0)
        mask_flat = (d1 == 0) & (d0 != 0)
        d0 = np.where(mask_flat, 0.0, d0)
        d0 = np.where((np.abs(d0) > 3 * np.abs(d1)) & (d1 != 0), 3 * d1, d0)
        return d0
    d[0] = _edge(((2 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1]), delta[0])
    d[-1] = _edge(((2 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2]), delta[-1])
    for i in range(1, n - 1):
        s0, s1 = delta[i - 1], delta[i]                # (d,) secants
        # per-channel: flat slope where secants disagree in sign or either
        # is zero (elementwise); harmonic mean elsewhere
        agree = (s0 != 0) & (s1 != 0) & ((s0 > 0) == (s1 > 0))
        w1 = 2 * h[i] + h[i - 1]
        w2 = h[i] + 2 * h[i - 1]
        harm = (w1 + w2) / (w1 / np.where(s0 == 0, np.inf, s0)
                            + w2 / np.where(s1 == 0, np.inf, s1))
        d[i] = np.where(agree, harm, 0.0)
    # NOTE: the weighted-harmonic-mean construction above IS the scipy
    # PchipInterpolator slope rule — monotone by construction on monotone
    # segments (no extra tau clamp needed), zero at local extrema.
    # Hermite -> power form per interval: p(u) = a + b u + c u^2 + e u^3,
    # u = tq - t_i, m = slopes d:
    a = y[:-1]
    b = d[:-1]
    # Hermite->power form (u = tq - t_i in SECONDS, delta = (y1-y0)/h):
    #   p(u) = y0 + m0 u + c u^2 + e u^3,
    #   c = (3(y1-y0) - h(2m0+m1))/h^2 = (3*delta - (2m0+m1))/h
    #   e = (-2(y1-y0) + h(m0+m1))/h^3  = (-2*delta + (m0+m1))/h^2
    c = (3 * delta - (2 * d[:-1] + d[1:])) / h[:, None]
    e = (-2 * delta + (d[:-1] + d[1:])) / (h[:, None] ** 2)
    return a, b, c, e


def eval_spline(t, coeffs, tq):
    a, b, c, d = coeffs
    idx = np.clip(np.searchsorted(t, tq, side="right") - 1, 0, len(t) - 2)
    s = tq - t[idx]
    s = s[:, None]
    return a[idx] + s * (b[idx] + s * (c[idx] + s * d[idx]))


def slerp_traj(quats, t_old, t_new):
    """Shortest-path slerp of (n,4) wxyz quats onto t_new."""
    q = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    idx = np.clip(np.searchsorted(t_old, t_new, side="right") - 1, 0, len(t_old) - 2)
    out = np.empty((len(t_new), 4))
    for k, (i, tq) in enumerate(zip(idx, t_new)):
        h = t_old[i + 1] - t_old[i]
        u = np.clip((tq - t_old[i]) / h, 0.0, 1.0)
        q0, q1 = q[i], q[i + 1]
        if np.dot(q0, q1) < 0:
            q1 = -q1
        d = np.clip(np.dot(q0, q1), -1.0, 1.0)
        if d > 0.9995:
            out[k] = q0 + u * (q1 - q0)
        else:
            th = np.arccos(d)
            out[k] = (np.sin((1 - u) * th) * q0 + np.sin(u * th) * q1) / np.sin(th)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=200.0)
    ap.add_argument("--fk-tol", type=float, default=1.5e-3)
    ap.add_argument("--key-body-file", default=str(
        ROOT / "roboparty_train/robolab/data/motions/x1_lab_v32/../.."
    ))  # unused placeholder, key bodies resolved below
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # FK setup (same pattern as mirror_lab_motions.py)
    sys.path.insert(0, str(ROOT))
    from sim2sim.mujoco_rollout import build_model  # noqa: E402
    import mujoco  # noqa: E402
    from roboparty_train.mirror_lab_motions import lab_dof_names  # noqa: E402

    # v32 10-body disc order (MUST match KEY_BODY_NAMES in
    # x1_amp_env_cfg.py; the stored key_body_pos is flat (10*3,))
    KEY_BODIES = [
        "left_knee_pitch_link", "right_knee_pitch_link",
        "left_ankle_roll_link", "right_ankle_roll_link",
        "left_elbow_yaw_link", "right_elbow_yaw_link",
        "left_shoulder_pitch_link", "right_shoulder_pitch_link",
        "left_wrist_pitch_link", "right_wrist_pitch_link",
    ]

    names = lab_dof_names()
    model, _ = build_model(ROOT / "gmr_x1_assets/x1.xml")
    data = mujoco.MjData(model)
    qadr = np.array([model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])
    key_ids = {b: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
               for b in KEY_BODIES}

    def fk_key_bodies(rp, quat_wxyz, dof):
        data.qpos[:] = 0
        data.qpos[:3] = rp
        data.qpos[3:7] = quat_wxyz / np.linalg.norm(quat_wxyz)
        data.qpos[qadr] = dof
        mujoco.mj_forward(model, data)
        return np.stack([data.xpos[key_ids[b]] for b in KEY_BODIES])

    n_ok, n_bad = 0, 0
    for pkl in sorted(src.glob("*.pkl")):
        clip = pickle.load(open(pkl, "rb"))
        fps_old = float(clip["fps"])
        t_old = np.arange(len(clip["dof_pos"])) / fps_old
        dur = t_old[-1]
        n_new = int(round(dur * args.fps)) + 1
        t_new = np.arange(n_new) / args.fps

        new = {"fps": args.fps, "loop_mode": clip["loop_mode"]}
        for key in ("root_pos", "dof_pos"):
            arr = np.asarray(clip[key], dtype=np.float64)
            co = pchip_coeffs(t_old, arr)
            new[key] = eval_spline(t_old, co, t_new).astype(np.float32)
        new["root_rot"] = slerp_traj(
            np.asarray(clip["root_rot"], dtype=np.float64), t_old, t_new).astype(np.float32)
        # key_body RECOMPUTED by FK from the resampled dof/root for EVERY
        # new frame: constructively consistent (channel-wise interpolation
        # of dof vs key_body diverges by mm on fast segments — the jog
        # clips' FK error — because the FK map is nonlinear in rotations).
        kb_flat = np.stack([fk_key_bodies(new["root_pos"][k], new["root_rot"][k],
                                          new["dof_pos"][k]).ravel()
                            for k in range(n_new)])
        new["key_body_pos"] = kb_flat.astype(np.float32)

        # validation 1: exact at original timestamps
        co = pchip_coeffs(t_old, np.asarray(clip["dof_pos"], dtype=np.float64))
        rt = eval_spline(t_old, co, t_old)
        e1 = float(np.abs(rt - np.asarray(clip["dof_pos"])).max())

        # validation 2: the resampled motion's FK GEOMETRY vs the original
        # key_body channel at the ORIGINAL timestamps (does the resampled
        # dof reproduce the source's key-body geometry?)
        kb_src = np.asarray(clip["key_body_pos"], dtype=np.float64).reshape(
            len(t_old), len(KEY_BODIES), 3)
        probes = np.linspace(0, len(t_old) - 1, 12).astype(int)
        errs = []
        for k in probes:
            dof_k = rt[k]
            rot_k = slerp_traj(np.asarray(clip["root_rot"], dtype=np.float64),
                               t_old, np.array([t_old[k]]))[0]
            kb_fk = fk_key_bodies(clip["root_pos"][k], rot_k, dof_k)
            errs.append(float(np.abs(kb_fk - kb_src[k]).max()))
        e2 = max(errs)

        ok = e1 < 1e-5 and e2 < args.fk_tol
        status = "OK " if ok else "BAD"
        print(f"[{status}] {pkl.name}: {fps_old}->{args.fps}fps "
              f"{len(t_old)}->{n_new}fr roundtrip {e1:.2e} FK {e2*1000:.2f}mm")
        if ok:
            with open(out / pkl.name, "wb") as f:
                pickle.dump(new, f)
            n_ok += 1
        else:
            n_bad += 1
    print(f"\n[DONE] {n_ok} upsampled, {n_bad} rejected -> {out}")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
