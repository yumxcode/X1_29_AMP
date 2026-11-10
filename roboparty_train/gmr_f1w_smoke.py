#!/usr/bin/env python3
"""Smoke-retarget a few AMASS clips to F1W via the repo's GMR pipeline
(after register_f1w + auto-IK calibration). Validates smoothness (R4)
before any batch run.

Usage (GMR venv):
    python _gmr_local/venv/bin/python roboparty_train/gmr_f1w_smoke.py
Outputs: roboparty_train/robolab/data/motions/f1w_gmr_smoke/*.pkl
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np

GMR_ROOT = Path(__file__).parent.parent / "_gmr_local" / "GMR"
sys.path.insert(0, str(GMR_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting as GMR  # noqa: E402
from general_motion_retargeting.utils.smpl import (  # noqa: E402
    load_smplx_file, get_smplx_data_offline_fast)

REPO = Path(__file__).parent.parent
SMPLX_FOLDER = GMR_ROOT / "assets" / "body_models"
OUT = REPO / "roboparty_train" / "robolab" / "data" / "motions" / "f1w_gmr_smoke"
OUT.mkdir(parents=True, exist_ok=True)

CLIPS = [
    "data/amass/CMU/07_01_stageii.npz",     # walk anchor
    "data/amass/CMU/08_01_stageii.npz",     # jog anchor
    "data/amass/CMU/run_09_05_stageii.npz", # run
]

retargeter = None
last_h = None
for rel in CLIPS:
    npz = REPO / rel
    if not npz.exists():
        print(f"[SKIP] missing {rel}")
        continue
    smplx_data, body_model, smplx_output, h = load_smplx_file(str(npz), str(SMPLX_FOLDER))
    fps = smplx_data["mocap_frame_rate"].item()
    frames, aligned = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=fps)
    if retargeter is None or last_h is None or abs(last_h - h) > 0.01:
        retargeter = GMR(src_human="smplx", tgt_robot="f1w", actual_human_height=h)
        last_h = h
    qpos_list = []
    for fr in frames:
        q = retargeter.retarget(fr)
        qpos_list.append(np.asarray(q, dtype=np.float64))
    qpos = np.stack(qpos_list)
    out = OUT / (npz.stem.replace("_stageii", "") + ".pkl")
    pickle.dump({"qpos": qpos, "fps": float(aligned)}, open(out, "wb"))
    print(f"[OK] {out.name}: T={qpos.shape[0]} qpos_dim={qpos.shape[1]}")
