#!/usr/bin/env python3
"""Generate left-right MIRRORED copies of every x1_lab motion clip (v28).

Why: the retargeted AMASS references are heavily L/R asymmetric (measured hip
swing ratio 0.35-0.97, most < 0.85; acceptance/ref_gait_analysis.py), and the
v26/v27 AMP policies faithfully inherit that asymmetry (v27: 0.5 m/s hip ratio
0.832 < 0.85 gate). Mirroring every clip doubles the dataset into an exactly
symmetric one, so the discriminator's style prior becomes symmetric — the
principled fix, no reward hacking.

Mirror transform (about the sagittal x-z plane, y -> -y):
  root_pos  (x, y, z) -> (x, -y, z)
  root_rot  (w, x, y, z) -> (w, -x, y, -z)   [M R M, M = diag(1,-1,1)]
  dof_pos   pair swap L<->R with sign = -1 if the pair's defaults are
            anti-aligned (hip pitch/roll/yaw), +1 if aligned (knee, ankle);
            torso yaw/roll negate, pitch keeps; ambiguous (default 0) joints
            are disambiguated automatically by FK error below
  key_body_pos  y -> -y with L/R body names swapped

Self-verification: FK(mirrored clip) must equal mirror_y(FK(original clip))
within tolerance for the 6 key bodies. Writes {name}_mirror.pkl next to the
sources and prints a per-clip verdict; exits non-zero if any clip fails.

Usage:
  ./.venv_test/bin/python roboparty_train/mirror_lab_motions.py \
      [--src roboparty_train/robolab/data/motions/x1_lab] [--dry-run]
"""
import argparse
import functools
import pickle
import re
import sys
from pathlib import Path

import numpy as np

print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parent
YAML = ROOT / "robolab/scripts/tools/retarget/config/x1.yaml"
XML = ROOT.parent / "gmr_x1_assets" / "x1.xml"

KEY_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link",
              "left_knee_pitch_link", "right_knee_pitch_link",
              "left_elbow_yaw_link", "right_elbow_yaw_link"]


def lab_dof_names():
    names = []
    for line in YAML.read_text().splitlines():
        m = re.match(r"^\s*-\s*(\S+)\s*$", line)
        if m:
            names.append(m.group(1))
        if len(names) >= 29:
            break
    assert len(names) == 29
    return names


def other(name):
    return name.replace("left_", "right_") if name.startswith("left_") \
        else name.replace("right_", "left_")


def pair_sign(name, defaults):
    """Sign applied when swapping a L/R pair: -1 if defaults anti-aligned."""
    o = other(name)
    if o == name or o not in defaults:
        return None                       # torso / not a pair
    a, b = defaults.get(name, 0.0), defaults.get(o, 0.0)
    if abs(a) < 1e-6 and abs(b) < 1e-6:
        return None                       # ambiguous -> FK disambiguation
    return -1.0 if a * b < 0 else 1.0


def torso_sign(name):
    if name.startswith("lumbar_yaw") or name.startswith("lumbar_roll"):
        return -1.0
    if name.startswith("lumbar_pitch"):
        return 1.0
    return None


def main():
    import mujoco
    from sim2sim.mujoco_rollout import build_model, DEFAULT_Q

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "robolab/data/motions/x1_lab"))
    ap.add_argument("--tol", type=float, default=0.02, help="FK mirror tol (m)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    src = Path(args.src)
    names = lab_dof_names()
    idx = {n: i for i, n in enumerate(names)}

    model, _ = build_model(XML)
    data = mujoco.MjData(model)
    qadr = np.array([model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in names])
    key_ids = {b: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
               for b in KEY_BODIES}

    # ---- sign table ------------------------------------------------------
    sign = {}
    ambiguous = []
    for n in names:
        s = pair_sign(n, DEFAULT_Q)
        if s is None:
            s = torso_sign(n)
        if s is None and ("left_" in n or "right_" in n):
            ambiguous.append(n)
            s = 1.0                       # placeholder; FK disambiguates
        sign[n] = s if s is not None else 1.0
    # ankle_pitch/ankle_roll and all default-0 pairs: try FK with +, flip if bad

    def fk_key(rp, quat_wxyz, dof):
        data.qpos[:] = 0
        data.qpos[:3] = rp
        data.qpos[3:7] = quat_wxyz / np.linalg.norm(quat_wxyz)
        data.qpos[qadr] = dof
        mujoco.mj_forward(model, data)
        return {b: data.xpos[i].copy() for b, i in key_ids.items()}

    def mirror_err(clip, sgn):
        T = len(clip["dof_pos"])
        errs = []
        for t in np.linspace(0, T - 1, 8).astype(int):
            q0 = fk_key(clip["root_pos"][t], clip["root_rot"][t], clip["dof_pos"][t])
            dof_m = clip["dof_pos"][t].copy()
            for n in names:
                j = idx[n]
                if n in ambiguous:
                    dof_m[j] = sgn[n] * clip["dof_pos"][t][idx[other(n)]]
                elif pair_sign(n, DEFAULT_Q) is not None or torso_sign(n) is not None:
                    dof_m[j] = sign[n] * clip["dof_pos"][t][idx[other(n)]] \
                        if (pair_sign(n, DEFAULT_Q) is not None) else sign[n] * clip["dof_pos"][t][j]
            rp_m = clip["root_pos"][t] * np.array([1, -1, 1])
            q_m = clip["root_rot"][t] * np.array([1, -1, 1, -1])
            q1 = fk_key(rp_m, q_m, dof_m)
            for b in KEY_BODIES:
                want = q0[b] * np.array([1, -1, 1])
                errs.append(np.linalg.norm(q1[other(b)] - want))
        return float(np.mean(errs))

    clips = sorted(src.glob("*.pkl"))
    clips = [c for c in clips if not c.stem.endswith("_mirror")]
    print(f"[INFO] {len(clips)} source clips in {src}")
    ok_all = True
    for c in clips:
        clip = pickle.load(open(c, "rb"))
        # disambiguate ambiguous joints on this clip (greedy per joint)
        sgn = {n: 1.0 for n in ambiguous}
        base = mirror_err(clip, sgn)
        for n in ambiguous:
            sgn[n] = -1.0
            e2 = mirror_err(clip, sgn)
            if e2 < base:
                base = e2
            else:
                sgn[n] = 1.0
        # build the mirrored clip
        T = len(clip["dof_pos"])
        dof = clip["dof_pos"].copy()
        for n in names:
            j = idx[n]
            if n in ambiguous:
                dof[:, j] = sgn[n] * clip["dof_pos"][:, idx[other(n)]]
            elif pair_sign(n, DEFAULT_Q) is not None:
                dof[:, j] = sign[n] * clip["dof_pos"][:, idx[other(n)]]
            else:                          # torso
                dof[:, j] = sign[n] * clip["dof_pos"][:, j]
        kb = clip["key_body_pos"].copy()
        order = [KEY_BODIES.index(other(b)) for b in KEY_BODIES]
        kb = kb[:, order]
        kb[:, :, 1] *= -1
        out = dict(clip)
        out["dof_pos"] = dof
        out["root_pos"] = clip["root_pos"] * np.array([1, -1, 1])
        out["root_rot"] = clip["root_rot"] * np.array([1, -1, 1, -1])
        out["key_body_pos"] = kb
        e = mirror_err(clip, sgn)
        ok = e <= args.tol
        ok_all &= ok
        amb = {n: int(sgn[n]) for n in ambiguous} if not args.dry_run else {}
        print(f"{'[OK] ' if ok else '[FAIL]'} {c.stem:26s} FK-mirror err {e*1000:7.2f} mm "
              f"{' '.join(f'{n.split(chr(95))[0][:3]}{k[0]}{k[-1]}={v:+d}' for n,v in list(amb.items())[:0])}")
        if not args.dry_run:
            out_path = c.with_name(c.stem + "_mirror.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(out, f)
    print(f"[{'PASS' if ok_all else 'FAIL'}] all clips <= {args.tol*1000:.0f} mm")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
