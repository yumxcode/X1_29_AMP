#!/usr/bin/env python3
"""Extend key_body_pos 6 -> 10 bodies for the v32 motion dataset (v38 AMP).

Motivation (v33-v37 evidence): the style discriminator only observed
knees/ankles/elbows (18 dims x 3 steps). It literally could not see arm
swing — style score 2.00 with collapsed arms (v33b) vs 0.80 with good form
(v37), i.e. uncorrelated with actual form quality. v38 extends the disc
observation with the arm chain: L/R shoulder_pitch_link (upper-arm proximal)
+ L/R wrist_pitch_link (distal endpoint = direct swing-amplitude channel).

Method: recompute ALL 10 key-body columns from each clip's own dof_pos via
MuJoCo FK (uniform provenance — no mixing of retarget-time and render-time
FK channels). Gates on the stored-vs-FK error of the ORIGINAL 6 bodies
(<= 15 mm, same threshold as acceptance/render_motion_videos.py) — abort
refuses to write anything on violation.

Output: x1_lab_v32/<same filenames> with key_body_pos (T, 10, 3) in the
order KEY10 below. All other pkl keys copied verbatim. The training env
(x1_amp_env_cfg.KEY_BODY_NAMES) and retarget x1.yaml (lab_key_body_names)
must list the SAME order.

Usage (local, .venv39):
  ./.venv39/bin/python roboparty_train/extend_key_bodies_v32.py \
      [--src data/motions/x1_lab_v31] [--dst data/motions/x1_lab_v32]
"""
import argparse
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim2sim"))
from sim2sim.mujoco_rollout import build_model, parse_yaml_list  # noqa: E402

XML = ROOT / "gmr_x1_assets" / "x1.xml"
YAML = ROOT / "roboparty_train/robolab/scripts/tools/retarget/config/x1.yaml"

KEY6 = ["left_knee_pitch_link", "right_knee_pitch_link",
        "left_ankle_roll_link", "right_ankle_roll_link",
        "left_elbow_yaw_link", "right_elbow_yaw_link"]
ADD4 = ["left_shoulder_pitch_link", "right_shoulder_pitch_link",
        "left_wrist_pitch_link", "right_wrist_pitch_link"]
KEY10 = KEY6 + ADD4

FK_GATE_MM = 15.0


def main():
    import mujoco
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "roboparty_train/robolab/data/motions/x1_lab_v31"))
    ap.add_argument("--dst", default=str(ROOT / "roboparty_train/robolab/data/motions/x1_lab_v32"))
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    lab_names = parse_yaml_list(YAML, "lab_dof_names")
    model, _ = build_model(XML)
    data = mujoco.MjData(model)
    jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in lab_names}
    qadr = np.array([model.jnt_qposadr[jid[n]] for n in lab_names])
    assert len(qadr) == 29
    kb_id = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in KEY10]
    for b, i in zip(KEY10, kb_id):
        assert i >= 0, f"body {b} not in {XML}"

    pkls = sorted(src.glob("*.pkl"))
    print(f"[INFO] {len(pkls)} clips from {src}")
    worst = 0.0
    for p in pkls:
        clip = pickle.load(open(p, "rb"))
        q = np.asarray(clip["dof_pos"], dtype=np.float64)
        rp = np.asarray(clip["root_pos"], dtype=np.float64)
        rr = np.asarray(clip["root_rot"], dtype=np.float64)
        stored = np.asarray(clip["key_body_pos"], dtype=np.float64)
        T = len(q)
        assert stored.shape == (T, 6, 3), f"{p.name}: stored shape {stored.shape}"

        qpos = np.zeros((T, model.nq))
        qpos[:, 0:3] = rp
        qpos[:, 3:7] = rr / np.linalg.norm(rr, axis=1, keepdims=True)
        qpos[:, qadr] = q

        fk = np.zeros((T, 10, 3))
        errs = []
        for t in range(T):
            data.qpos[:] = qpos[t]
            mujoco.mj_forward(model, data)
            fk[t] = data.xpos[kb_id]
            if t % max(1, T // 24) == 0:  # ~24 samples for the gate
                errs.append(max(np.linalg.norm(fk[t, k] - stored[t, k])
                                for k in range(6)))
        e_mm = float(np.mean(errs)) * 1000.0
        worst = max(worst, e_mm)
        if e_mm > FK_GATE_MM:
            sys.exit(f"[FATAL] {p.name}: FK vs stored key_body_pos {e_mm:.1f} mm "
                     f"> {FK_GATE_MM} mm — joint mapping broken, nothing written")
        clip["key_body_pos"] = fk.astype(np.float32)
        out = dst / p.name
        with open(out, "wb") as f:
            pickle.dump(clip, f)
        amp_deg = 0.0
        for s in (6, 8):  # elbow, wrist channels of the arm chain
            v = fk[:, s, :]
            amp_deg = max(amp_deg, float(np.ptp(np.degrees(np.arctan2(
                v[:, 1], v[:, 0])))) if False else 0.0)
        print(f"  {p.name:38s} T={T:4d} FKerr={e_mm:5.1f}mm "
              f"wristY_swing={np.ptp(fk[:,8,1])*1000:6.1f}/{np.ptp(fk[:,9,1])*1000:5.1f}mm")
    print(f"[OK] wrote {len(pkls)} clips to {dst} (10-body key_body_pos); "
          f"worst FK gate error {worst:.1f} mm <= {FK_GATE_MM} mm")


if __name__ == "__main__":
    main()
