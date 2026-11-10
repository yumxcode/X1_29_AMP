#!/usr/bin/env python3
"""RUN-tuned LOCAL GMR retarget worker (F1w run library, 2026-11-10).

Same pipeline as retarget_v72_local.py but with run-tuned IK knobs:
  - motion regularization DISABLED (v72 default acc=0.01/jerk=0.001 acts as
    a low-pass filter; at 3.5-4 m/s sprint it lags the feet 1-2 m/s —
    measured stance skate on 09_x family; walking unaffected at 1 m/s)
  - max_iter 10 -> 30 (tracking headroom at high speed)
Everything else identical (P6 footsole normalization, per-height retargeter,
x1_gmr schema output).

Usage: _gmr_local/venv/bin/python roboparty_train/retarget_run_local.py \
          [--clips run_09_01 ...] [--out x1_gmr_f1wrun2]
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

GMR_ROOT = Path(__file__).resolve().parent.parent / "_gmr_local" / "GMR"
sys.path.insert(0, str(GMR_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR  # noqa: E402
from general_motion_retargeting.utils.smpl import (  # noqa: E402
    load_smplx_file, get_smplx_data_offline_fast)
from general_motion_retargeting.kinematics_model import KinematicsModel  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SMPLX_FOLDER = GMR_ROOT / "assets" / "body_models"
SRC_DIR = REPO_ROOT / "data" / "amass" / "CMU"
FOOTSOLE_OFFSET = 0.04
MAX_ITER = int(__import__("os").environ.get("GMR_RUN_MAX_ITER", "30"))


def retarget_clip(npz_file, retargeter, kinematics_model, last_h):
    smplx_data, body_model, smplx_output, h = load_smplx_file(
        str(npz_file), str(SMPLX_FOLDER))
    src_fps = smplx_data["mocap_frame_rate"].item()
    frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=src_fps)
    if abs(last_h - h) > 0.01:
        print(f"  creating retargeter (height={h:.3f}m)", flush=True)
        retargeter = GMR(src_human="smplx", tgt_robot="x1", actual_human_height=h)
        retargeter.set_regularity(enabled=False)   # RUN TUNING
        retargeter.max_iter = MAX_ITER             # RUN TUNING
        kinematics_model = KinematicsModel(retargeter.xml_file, device="cpu")
        last_h = h
    assert kinematics_model is not None
    qpos_list = [retargeter.retarget(f).copy() for f in frames]
    qpos_list = np.array(qpos_list)
    root_pos = qpos_list[:, :3].copy()
    root_rot = qpos_list[:, 3:7].copy()
    root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]   # wxyz -> xyzw
    dof_pos = qpos_list[:, 7:].copy()
    body_pos, body_rot = kinematics_model.forward_kinematics(
        torch.tensor(root_pos, dtype=torch.float32),
        torch.tensor(root_rot, dtype=torch.float32),
        torch.tensor(dof_pos, dtype=torch.float32))
    lowest = torch.min(body_pos[..., 2]).item()
    root_pos[:, 2] -= (lowest - FOOTSOLE_OFFSET)
    root_pos[:, :2] -= root_pos[0, :2]
    motion_data = {
        "fps": aligned_fps,
        "dof_names": [retargeter.model.joint(i).name
                      for i in range(retargeter.model.njnt)],
        "dof_positions": dof_pos,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "body_positions": body_pos.numpy(),
        "body_rotations": body_rot.numpy(),
    }
    return motion_data, retargeter, kinematics_model, last_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--out", default="x1_gmr_f1wrun2")
    args = ap.parse_args()
    out_dir = REPO_ROOT / "roboparty_train" / "robolab" / "data" / "motions" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = args.clips
    if not clips:
        clips = sorted(p.stem[4:] for p in SRC_DIR.glob("run_*_stageii.npz"))
    retargeter, kinematics_model, last_h = None, None, -1.0
    ok, fail = 0, 0
    for i, name in enumerate(clips):
        npz = SRC_DIR / f"run_{name}_stageii.npz"
        print(f"[{i+1}/{len(clips)}] Retargeting run_{name}_stageii.npz", flush=True)
        try:
            data, retargeter, kinematics_model, last_h = retarget_clip(
                npz, retargeter, kinematics_model, last_h)
            with open(out_dir / f"run_{name}.pkl", "wb") as f:
                pickle.dump(data, f)
            print(f"  OK run_{name}.pkl: {len(data['dof_positions'])} frames "
                  f"{len(data['dof_positions'][0])} DOF", flush=True)
            ok += 1
        except Exception as e:
            print(f"  FAIL run_{name}: {e}", flush=True)
            fail += 1
    print(f"[DONE] {ok} ok, {fail} fail")


if __name__ == "__main__":
    main()
