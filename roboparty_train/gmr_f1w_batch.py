#!/usr/bin/env python3
"""Batch-retarget the v1 source clips to F1W via GMR (after auto-IK).

Mirrors the X1 worker embedded in run_gmr_retarget.py (same output format:
fps/root_pos(xyzw-shifted)/root_rot/dof_names/body_names/dof_pos/
body_positions/... via KinematicsModel FK on cpu) but targets f1w.

Run with the GMR venv:
    _gmr_local/venv/bin/python roboparty_train/gmr_f1w_batch.py [names...]
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

GMR_ROOT = Path(__file__).parent.parent / "_gmr_local" / "GMR"
sys.path.insert(0, str(GMR_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR  # noqa: E402
from general_motion_retargeting.utils.smpl import (  # noqa: E402
    load_smplx_file, get_smplx_data_offline_fast)
from general_motion_retargeting.kinematics_model import KinematicsModel  # noqa: E402

REPO = Path(__file__).parent.parent
SMPLX_FOLDER = GMR_ROOT / "assets" / "body_models"
OUT = REPO / "roboparty_train" / "robolab" / "data" / "motions" / "f1w_gmr"
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = [
    "0005_normal_walk1", "07_01", "08_01", "08_03", "08_06", "08_08",
    "103_07", "35_02", "38_02", "run_02_03",
    "run_09_01", "run_09_04", "run_09_05", "run_09_06", "run_09_07",
    "run_09_11", "run_16_35", "run_35_17", "run_35_18", "run_35_19",
    "run_35_20", "run_accF_C5", "run_accM_C3", "run_accM_C4", "run_accM_C4b",
    "run_accM_C5",
]

only = set(sys.argv[1:]) if len(sys.argv) > 1 else None
retargeter = None
kinematics_model = None
last_h = None
ok = fail = skip = 0
for name in SOURCES:
    if only and name not in only:
        continue
    out = OUT / f"{name}.pkl"
    if out.exists():
        skip += 1
        continue
    npz = REPO / "data" / "amass" / "CMU" / f"{name}_stageii.npz"
    if not npz.exists():
        print(f"[MISS] {npz}", flush=True)
        continue
    try:
        smplx_data, body_model, smplx_output, h = load_smplx_file(str(npz), str(SMPLX_FOLDER))
        fps = smplx_data["mocap_frame_rate"].item()
        frames, aligned = get_smplx_data_offline_fast(
            smplx_data, body_model, smplx_output, tgt_fps=fps)
        if retargeter is None or last_h is None or abs(last_h - h) > 0.01:
            print(f"[GMR] new retargeter (height={h:.3f})", flush=True)
            retargeter = GMR(src_human="smplx", tgt_robot="f1w", actual_human_height=h)
            if kinematics_model is None:
                kinematics_model = KinematicsModel(retargeter.xml_file, device="cpu")
            last_h = h
        qpos_list = []
        for fr in frames:
            q = retargeter.retarget(fr)
            qpos_list.append(q.copy())
        qpos_list = np.array(qpos_list)

        root_pos = qpos_list[:, :3].copy()
        root_rot = qpos_list[:, 3:7].copy()
        root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]  # xyzw -> wxyz
        dof_pos = qpos_list[:, 7:].copy()

        body_pos, body_rot = kinematics_model.forward_kinematics(
            torch.tensor(root_pos, dtype=torch.float32),
            torch.tensor(root_rot, dtype=torch.float32),
            torch.tensor(dof_pos, dtype=torch.float32),
        )
        # foot-sole normalization (ankle_roll body origin ~3.5cm above the
        # sole on weilai; measured from the sole-probe geoms in f1w.xml)
        FOOTSOLE_OFFSET = 0.036
        lowest = torch.min(body_pos[..., 2]).item()
        root_pos[:, 2] -= (lowest - FOOTSOLE_OFFSET)
        root_pos[:, :2] -= root_pos[0, :2]

        motion_data = {
            "fps": aligned, "root_pos": root_pos, "root_rot": root_rot,
            "dof_names": kinematics_model.dof_names,
            "body_names": kinematics_model.body_names,
            "dof_positions": dof_pos, "dof_pos": dof_pos,
            "body_positions": body_pos.numpy(),
            "body_rotations": body_rot.numpy(),
            "local_body_pos": body_pos.numpy(),
        }
        with open(out, "wb") as f:
            pickle.dump(motion_data, f)
        ok += 1
        print(f"[OK] {name}: T={len(qpos_list)} dof={dof_pos.shape[1]}", flush=True)
    except Exception as e:
        fail += 1
        print(f"[ERR] {name}: {type(e).__name__}: {e}", flush=True)
print(f"=== batch: ok={ok} fail={fail} skip={skip} ===", flush=True)
