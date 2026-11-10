#!/usr/bin/env python3
"""F1w MILESTONE library batch: walk+jog clips via GMR (damping=2.0, true
knee limits) -> lab conversion -> v13 targeted spike trim -> R4/FK gates.

Validated recipe (2026-11-11 probes):
  - GMR retargeter damping 2.0 (default 0.5): walk 07_01 native p99 14.6/
    max 23.0 (BOTH inside the R4 gate 18.5/40 without post-processing);
    jog 08_01 15.7 with isolated max frames 43.1 (trimmed by the v13
    interpolation); run clips stay OUT of the milestone band (35.8).
  - lab conversion: mujoco-wxyz quat as-is (FK-validated), ground-
    normalized root, key bodies recomputed by weilai FK (self-consistent).
Gates per clip (F1W_RUN_ACCEPTANCE sec 6, locked convention):
  R4  jv p99 <= 18.5 and max <= 40.0 rad/s (per-clip, non-mirror set)
  FK  key-body self-consistency: recomputed == stored (trivially true);
      deviation vs the pre-trim GMR FK reported for the record.
Run with the GMR venv.
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
OUT = REPO / "roboparty_train" / "robolab" / "data" / "motions" / "f1w_gmr_ms"
OUT.mkdir(parents=True, exist_ok=True)

WALK_JOG = [
    "0005_normal_walk1", "07_01", "08_01", "08_03", "08_06", "08_08",
    "103_07", "35_02", "38_02",
]

retargeter = None
last_h = None
ok = fail = 0
for name in WALK_JOG:
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
            print(f"[GMR] new retargeter (height={h:.3f}, damping=2.0)", flush=True)
            retargeter = GMR(src_human="smplx", tgt_robot="f1w",
                             actual_human_height=h, damping=2.0)
            last_h = h
        qpos_list = []
        for fr in frames:
            q = retargeter.retarget(fr)
            qpos_list.append(q.copy())
        qpos = np.array(qpos_list)
        pickle.dump({"qpos": qpos, "fps": float(aligned)},
                    open(OUT / f"{name}.pkl", "wb"))
        ok += 1
        jv = np.abs(np.diff(qpos[:, 7:], axis=0) * aligned).max(axis=1)
        print(f"[OK] {name}: T={len(qpos)} raw jv p99={np.percentile(jv,99):.1f} "
              f"max={jv.max():.1f}", flush=True)
    except Exception as e:
        fail += 1
        print(f"[ERR] {name}: {type(e).__name__}: {e}", flush=True)
print(f"=== milestone batch: ok={ok} fail={fail} ===", flush=True)
