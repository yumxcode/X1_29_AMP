#!/usr/bin/env python3
"""LOCAL GMR retarget worker for the v31 new-source clips (CPU).

Mirrors the remote batch worker in run_gmr_retarget.py exactly (same P5
per-height retargeter creation, same P6 footsole normalization) but:
  - device cpu (local Mac, no CUDA)
  - processes ONLY the clips listed in CLIPS (v31 CMU walking additions)
  - writes x1_gmr-format pkl into x1_gmr/ (same schema as existing files)
Usage: _gmr_local/venv/bin/python roboparty_train/retarget_new_clips_local.py
"""
import os
import pickle
import sys
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
OUTPUT_DIR = REPO_ROOT / "roboparty_train/robolab/data/motions/x1_gmr"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLIPS = ["CMU/103_07_stageii.npz", "CMU/138_18_stageii.npz"]

kinematics_model = None
retargeter = None
last_h = None
ok = fail = 0
for i, rel in enumerate(CLIPS):
    npz_file = REPO_ROOT / "AMASS_minimal" / rel
    out_name = npz_file.stem.replace("_stageii", "") + ".pkl"
    out_path = OUTPUT_DIR / out_name
    if out_path.exists():
        print(f"[{i+1}/{len(CLIPS)}] SKIP (exists): {out_name}", flush=True)
        ok += 1
        continue
    print(f"[{i+1}/{len(CLIPS)}] Retargeting {npz_file.name}", flush=True)
    try:
        smplx_data, body_model, smplx_output, h = load_smplx_file(str(npz_file), str(SMPLX_FOLDER))
        src_fps = smplx_data["mocap_frame_rate"].item()
        frames, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=src_fps)
        if retargeter is None or last_h is None or abs(last_h - h) > 0.01:
            print(f"  creating retargeter (height={h:.3f}m)", flush=True)
            retargeter = GMR(src_human="smplx", tgt_robot="x1", actual_human_height=h)
            if kinematics_model is None:
                kinematics_model = KinematicsModel(retargeter.xml_file, device="cpu")
            last_h = h
        qpos_list = [retargeter.retarget(f).copy() for f in frames]
        qpos_list = np.array(qpos_list)
        root_pos = qpos_list[:, :3].copy()
        root_rot = qpos_list[:, 3:7].copy()
        root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]   # wxyz -> xyzw (GMR save convention)
        dof_pos = qpos_list[:, 7:].copy()
        body_pos, body_rot = kinematics_model.forward_kinematics(
            torch.tensor(root_pos, dtype=torch.float32),
            torch.tensor(root_rot, dtype=torch.float32),
            torch.tensor(dof_pos, dtype=torch.float32))
        FOOTSOLE_OFFSET = 0.04
        lowest = torch.min(body_pos[..., 2]).item()
        root_pos[:, 2] -= (lowest - FOOTSOLE_OFFSET)
        root_pos[:, :2] -= root_pos[0, :2]
        motion_data = {
            "fps": aligned_fps, "root_pos": root_pos, "root_rot": root_rot,
            "dof_names": kinematics_model.dof_names, "body_names": kinematics_model.body_names,
            "dof_positions": dof_pos, "dof_pos": dof_pos,
            "body_positions": body_pos.cpu().numpy(),
            "body_rotations": body_rot.cpu().numpy(),
            "local_body_pos": body_pos.cpu().numpy(),
        }
        with open(out_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"  OK {out_name}: {len(qpos_list)} frames, {dof_pos.shape[1]} DOF", flush=True)
        ok += 1
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  FAIL: {e}", flush=True)
        fail += 1

print(f"[DONE] {ok} ok, {fail} fail")
sys.exit(0 if fail == 0 else 1)
