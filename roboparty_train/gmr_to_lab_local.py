#!/usr/bin/env python3
"""Convert x1_gmr pkl -> x1_lab pkl LOCALLY (MuJoCo FK instead of Isaac Lab).

Reproduces scripts/tools/retarget/gmr_to_lab.py + dataset_retarget.py:
  - dof reorder: gmr_dof_names -> lab_dof_names (x1.yaml)
  - root_rot: xyzw -> wxyz, then quat_unique (canonical w >= 0, as
    math_utils.quat_unique does remotely) + normalize
  - key_body_pos: FK of the lab_key_body_names order (knee, ankle, elbow)
  - loop_mode 0 (CLAMP), fps kept
Gates: FK vs stored body_positions (gmr pkl truth) mean err <= 10 mm.

Usage: python gmr_to_lab_local.py --src .../x1_gmr --dst .../x1_lab_new \
    [--names 138_01 ...]
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "sim2sim"))
from sim2sim.mujoco_rollout import build_model, DEFAULT_Q, parse_yaml_list  # noqa: E402

XML = ROOT.parent / "gmr_x1_assets" / "x1.xml"
YAML = ROOT / "robolab/scripts/tools/retarget/config/x1.yaml"
KEY_BODIES = ["left_knee_pitch_link", "right_knee_pitch_link",
              "left_ankle_roll_link", "right_ankle_roll_link",
              "left_elbow_yaw_link", "right_elbow_yaw_link"]


def quat_unique(q):
    """Canonical quaternions with w >= 0 (mimics isaaclab quat_unique)."""
    sign = np.where(q[:, 0:1] < 0, -1.0, 1.0)
    return q * sign


def main():
    import mujoco
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "robolab/data/motions/x1_gmr"))
    ap.add_argument("--dst", required=True)
    ap.add_argument("--names", nargs="*", default=None)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    gmr_names = parse_yaml_list(YAML, "gmr_dof_names")
    lab_names = parse_yaml_list(YAML, "lab_dof_names")
    g2l = [gmr_names.index(n) for n in lab_names]

    model, _ = build_model(XML)
    data = mujoco.MjData(model)
    jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in lab_names}
    qadr = np.array([model.jnt_qposadr[jid[n]] for n in lab_names])
    kb_bid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in KEY_BODIES]

    clips = sorted(src.glob("*.pkl"))
    if args.names:
        clips = [p for p in clips if p.stem in set(args.names)]
    print(f"[INFO] converting {len(clips)} clips -> {dst}")
    for path in clips:
        c = pickle.load(open(path, "rb"), encoding="latin1")
        rp = np.asarray(c["root_pos"], float)
        rr_xyzw = np.asarray(c["root_rot"], float)
        q_gmr = np.asarray(c["dof_pos"], float)
        fps = float(c["fps"])
        T = len(rp)
        rr = quat_unique(rr_xyzw[:, [3, 0, 1, 2]])
        rr /= np.linalg.norm(rr, axis=1, keepdims=True)
        q_lab = q_gmr[:, g2l]

        kb = np.zeros((T, 6, 3))
        for t in range(T):
            data.qpos[:] = 0
            data.qpos[0:3] = rp[t]
            data.qpos[3:7] = rr[t]
            data.qpos[qadr] = q_lab[t]
            mujoco.mj_forward(model, data)
            for k, b in enumerate(kb_bid):
                kb[t, k] = data.xpos[b]

        # gate: compare FK key bodies vs gmr pkl's stored body_positions.
        # NOTE the GMR worker normalizes root xy AFTER computing
        # body_positions, so stored world positions live in the pre-shift
        # frame (constant offset up to meters on long walks). Compare
        # RELATIVE trajectories (frame-to-frame motion), which is invariant
        # to that shift but still catches dof mapping / quat convention bugs.
        if "body_positions" in c:
            bn = list(c["body_names"])
            src_ids = [bn.index(b) for b in KEY_BODIES]
            bp = np.asarray(c["body_positions"], float)
            a = kb - kb[0]
            b = bp[:, src_ids] - bp[0, src_ids]
            e = float(np.percentile(np.linalg.norm(a - b, axis=2), 95))
            tag = "OK" if e <= 0.010 else "MISMATCH"
            print(f"  {path.stem:20s} T={T:5d} FK-vs-stored relative p95 {e*1000:6.1f} mm [{tag}]")
            if e > 0.010:
                continue

        out = {"fps": fps, "root_pos": rp, "root_rot": rr,
               "dof_pos": q_lab, "loop_mode": 0, "key_body_pos": kb}
        with open(dst / path.name, "wb") as f:
            pickle.dump(out, f)
    print("[DONE]")


if __name__ == "__main__":
    main()
