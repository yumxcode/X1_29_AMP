#!/usr/bin/env python3
"""LOCAL GMR retarget worker for the v72 CMU low-cadence clips (CPU).

Same per-height retargeter creation + P6 footsole normalization as the v31
worker (retarget_new_clips_local.py / run_gmr_retarget.py remote batch),
but reads the SMPL-H->stageii converted npz from data/amass/CMU/ and writes
x1_gmr-schema pkl into roboparty_train/robolab/data/motions/x1_gmr_v72/.

Usage: _gmr_local/venv/bin/python roboparty_train/retarget_v72_local.py \
           [--clips 103_07 35_06 ...] [--out x1_gmr_v72]
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


def retarget_clip(npz_file, retargeter, kinematics_model, last_h):
    smplx_data, body_model, smplx_output, h = load_smplx_file(
        str(npz_file), str(SMPLX_FOLDER))
    src_fps = smplx_data["mocap_frame_rate"].item()
    frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=src_fps)
    if abs(last_h - h) > 0.01:
        print(f"  creating retargeter (height={h:.3f}m)", flush=True)
        retargeter = GMR(src_human="smplx", tgt_robot="x1", actual_human_height=h)
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
        "fps": aligned_fps, "root_pos": root_pos, "root_rot": root_rot,
        "dof_names": kinematics_model.dof_names,
        "body_names": kinematics_model.body_names,
        "dof_positions": dof_pos, "dof_pos": dof_pos,
        "body_positions": body_pos.cpu().numpy(),
        "body_rotations": body_rot.cpu().numpy(),
        "local_body_pos": body_pos.cpu().numpy(),
    }
    return motion_data, h, retargeter, kinematics_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=None,
                    help="clip names (without _stageii.npz); default = all")
    ap.add_argument("--out", default="x1_gmr_v72")
    args = ap.parse_args()

    out_dir = REPO_ROOT / "roboparty_train" / "robolab" / "data" / "motions" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clips:
        files = [SRC_DIR / f"{c}_stageii.npz" for c in args.clips]
    else:
        files = sorted(SRC_DIR.glob("*_stageii.npz"))

    retargeter = None
    kinematics_model = None
    last_h = -1.0
    ok = fail = 0
    for i, npz_file in enumerate(files):
        out_path = out_dir / (npz_file.stem.replace("_stageii", "") + ".pkl")
        if out_path.exists():
            print(f"[{i+1}/{len(files)}] SKIP (exists): {out_path.name}", flush=True)
            ok += 1
            continue
        if not npz_file.exists():
            print(f"[{i+1}/{len(files)}] MISSING: {npz_file.name}", flush=True)
            fail += 1
            continue
        print(f"[{i+1}/{len(files)}] Retargeting {npz_file.name}", flush=True)
        t0 = time.time()
        try:
            if kinematics_model is None:
                # placeholder creation below (retargeter first needs h)
                pass
            md, last_h_new, retargeter_new, km_new = retarget_clip(
                npz_file, retargeter, kinematics_model, last_h)
            retargeter, last_h, kinematics_model = retargeter_new, last_h_new, km_new
            with open(out_path, "wb") as f:
                pickle.dump(md, f)
            dt = time.time() - t0
            print(f"  OK {out_path.name}: {len(md['dof_pos'])} frames "
                  f"{md['dof_pos'].shape[1]} DOF ({dt:.0f}s)", flush=True)
            ok += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAIL: {e}", flush=True)
            fail += 1
    print(f"[DONE] {ok} ok, {fail} fail")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
