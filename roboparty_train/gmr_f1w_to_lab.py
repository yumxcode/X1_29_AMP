#!/usr/bin/env python3
"""Convert f1w_gmr pkl -> f1w lab pkl (MuJoCo FK; weilai body names).

Adapted from gmr_to_lab_local.py (X1). Differences:
  - XML: gmr_f1w_assets/f1w.xml (weilai, mass-merged)
  - KEY_BODIES: weilai leg bodies lack the _link suffix; arms keep it
  - same gmr_dof_names -> lab_dof_names mapping (f1w GMR xml uses the
    same joint names as x1)
Gates: FK vs stored body_positions relative p95 <= 10 mm.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "sim2sim"))
from sim2sim.mujoco_rollout import build_model, parse_yaml_list  # noqa: E402

XML = ROOT.parent / "gmr_f1w_assets" / "f1w.xml"
YAML = ROOT / "robolab/scripts/tools/retarget/config/x1.yaml"
KEY_BODIES = ["left_knee_pitch", "right_knee_pitch",
              "left_ankle_roll", "right_ankle_roll",
              "left_elbow_yaw_link", "right_elbow_yaw_link"]


def quat_unique(q):
    sign = np.where(q[:, 0:1] < 0, -1.0, 1.0)
    return q * sign


def main():
    import mujoco
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "robolab/data/motions/f1w_gmr"))
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

        kb = np.zeros((T, len(KEY_BODIES), 3))
        for t in range(T):
            data.qpos[:] = 0
            data.qpos[0:3] = rp[t]
            data.qpos[3:7] = rr[t]
            data.qpos[qadr] = q_lab[t]
            mujoco.mj_forward(model, data)
            for k, b in enumerate(kb_bid):
                kb[t, k] = data.xpos[b]

        if "body_positions" in c:
            bn = list(c["body_names"])
            # GMR f1w xml body names may differ by suffix; match both forms
            def find_b(b):
                if b in bn:
                    return bn.index(b)
                alt = b + "_link" if not b.endswith("_link") else b[:-len("_link")]
                return bn.index(alt) if alt in bn else -1
            src_ids = [find_b(b) for b in KEY_BODIES]
            if all(i >= 0 for i in src_ids):
                bp = np.asarray(c["body_positions"], float)
                a = kb - kb[0]
                b2 = bp[:, src_ids] - bp[0, src_ids]
                e = float(np.percentile(np.linalg.norm(a - b2, axis=2), 95))
                tag = "OK" if e <= 0.010 else "MISMATCH"
                print(f"  {path.stem:20s} T={T:5d} FK-vs-stored relative p95 {e*1000:6.1f} mm [{tag}]")
                if e > 0.010:
                    continue

        # R4 velocity audit (stored fps finite diff)
        jv = np.abs(np.diff(q_lab, axis=0) * fps).max(axis=1)
        p99, mx = float(np.percentile(jv, 99)), float(jv.max())
        ok = p99 <= 15.0 and mx <= 25.0
        print(f"  {path.stem:20s} R4 jv p99={p99:5.1f} max={mx:5.1f} [{'ok' if ok else 'FAIL'}]")

        out = {"fps": fps, "root_pos": rp, "root_rot": rr.astype(np.float32),
               "dof_pos": q_lab, "loop_mode": 0, "key_body_pos": kb.astype(np.float32)}
        pickle.dump(out, open(dst / path.name, "wb"))
    print("[INFO] done")


if __name__ == "__main__":
    main()
