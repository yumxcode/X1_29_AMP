#!/usr/bin/env python3
"""
GMR retargeting pipeline for X1 humanoid on Gradmotion.

This script:
1. Reassembles SMPLX_NEUTRAL.pkl from git chunks
2. Clones and installs GMR in a separate venv (avoids Isaac Lab numpy conflict)
3. Registers X1 in GMR's params.py
4. Runs GMR batch retargeting on AMASS data → x1_gmr/*.pkl

Usage:
    python run_gmr_retarget.py --headless
"""

import argparse
import functools
import os
import shutil
import subprocess
import sys
import json
import pickle
import numpy as np
from pathlib import Path

print = functools.partial(print, flush=True)

# IMPORTANT: Do NOT import isaaclab or start AppLauncher.
# GMR has its own MuJoCo-based pipeline that conflicts with Isaac Lab's numpy.
# This script runs as a standalone Python script.

# ── Step 0: Locate workspace and repo root ──────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
# On Gradmotion: /workspace/isaaclab/X1_29_AMP/...
# Try to find repo root
REPO_ROOT = None
for candidate in [SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent.parent,
                  SCRIPT_DIR.parent.parent.parent.parent]:
    if (candidate / "AMASS_minimal").is_dir() and (candidate / "roboparty_train").is_dir():
        REPO_ROOT = candidate
        break
    if (candidate / "AMASS_minimal").is_dir() and (candidate / "X1_29_AMP" / "roboparty_train").is_dir():
        REPO_ROOT = candidate / "X1_29_AMP"
        break

if REPO_ROOT is None:
    # Search from isaaclab workspace
    for p in [Path("/workspace/isaaclab/X1_29_AMP")]:
        if p.is_dir():
            REPO_ROOT = p
            break

if REPO_ROOT is None:
    print("[FATAL] Cannot find repo root with AMASS_minimal and roboparty_train")
    sys.exit(1)

print(f"[INFO] REPO_ROOT = {REPO_ROOT}")


# ── Step 1: Reassemble SMPLX_NEUTRAL.pkl ───────────────────────────
def reassemble_smplx():
    chunks_dir = REPO_ROOT / "AMASS_minimal" / "smplx_parts"
    output_dir = REPO_ROOT / "AMASS_minimal" / "smplx"
    output_file = output_dir / "SMPLX_NEUTRAL.pkl"

    if output_file.exists():
        print(f"[INFO] SMPLX_NEUTRAL.pkl already exists ({output_file.stat().st_size} bytes)")
        return output_file

    print("[INFO] Reassembling SMPLX_NEUTRAL.pkl from chunks...")
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = sorted(chunks_dir.glob("SMPLX_NEUTRAL.pkl.part*"))
    if not chunks:
        print(f"[FATAL] No chunks found in {chunks_dir}")
        sys.exit(1)

    with open(output_file, "wb") as f:
        for chunk in chunks:
            f.write(chunk.read_bytes())

    size = output_file.stat().st_size
    print(f"[INFO] SMPLX_NEUTRAL.pkl reassembled: {size} bytes ({size / 1e6:.1f} MB)")
    if size < 100_000_000:
        print("[FATAL] File too small!")
        sys.exit(1)

    return output_file


# ── Step 2: Clone and install GMR in separate venv ────────────────
def setup_gmr():
    gmr_dir = REPO_ROOT / "GMR"

    if gmr_dir.exists() and (gmr_dir / "setup.py").exists():
        print(f"[INFO] GMR already exists at {gmr_dir}")
    else:
        print("[INFO] Cloning GMR...")
        subprocess.check_call([
            "git", "clone", "--depth", "1",
            "https://github.com/Roboparty/GMR.git", str(gmr_dir)
        ])

    # Create isolated venv to avoid Isaac Lab numpy conflict
    venv_dir = REPO_ROOT / "gmr_venv"
    if not venv_dir.exists():
        print("[INFO] Creating isolated venv for GMR...")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
        pip = str(venv_dir / "bin" / "pip")
        subprocess.check_call([pip, "install", "--upgrade", "pip", "-q"])
        # Install GMR + dependencies
        subprocess.check_call([pip, "install", "-e", str(gmr_dir), "-q"])
        # Also install mujoco and other deps that GMR needs
        subprocess.check_call([pip, "install", "mujoco", "mink", "qpsolvers", "scipy", "-q"])
        print("[INFO] GMR venv ready")

    # Set up SMPLX body models for GMR
    gmr_body_models = gmr_dir / "assets" / "body_models" / "smplx"
    gmr_body_models.mkdir(parents=True, exist_ok=True)
    smplx_pkl = REPO_ROOT / "AMASS_minimal" / "smplx" / "SMPLX_NEUTRAL.pkl"
    target = gmr_body_models / "SMPLX_NEUTRAL.pkl"
    if not target.exists() and smplx_pkl.exists():
        shutil.copy2(smplx_pkl, target)
        print(f"[INFO] Copied SMPLX_NEUTRAL.pkl to GMR body_models")

    return gmr_dir, venv_dir


# ── Step 3: Register X1 in GMR ─────────────────────────────────────
def register_x1_in_gmr(gmr_dir: Path):
    """Add X1 to GMR's params.py and copy assets."""
    # 3a. Copy X1 MJCF + meshes to GMR assets
    x1_assets_src = REPO_ROOT / "gmr_x1_assets"
    x1_assets_dst = gmr_dir / "assets" / "x1"
    if x1_assets_dst.exists():
        shutil.rmtree(x1_assets_dst)
    shutil.copytree(x1_assets_src, x1_assets_dst)
    print(f"[INFO] Copied X1 assets to {x1_assets_dst}")

    # 3b. Copy IK config
    ik_src = REPO_ROOT / "AMASS_minimal" / "smplx_to_x1.json"
    ik_dst = gmr_dir / "general_motion_retargeting" / "ik_configs" / "smplx_to_x1.json"
    shutil.copy2(ik_src, ik_dst)
    print(f"[INFO] Copied IK config to {ik_dst}")

    # 3c. Patch params.py
    params_file = gmr_dir / "general_motion_retargeting" / "params.py"
    content = params_file.read_text()

    if '"x1"' not in content:
        # Add to ROBOT_XML_DICT
        content = content.replace(
            '"rpo": ASSET_ROOT / "rpo" / "rpo.xml",',
            '"rpo": ASSET_ROOT / "rpo" / "rpo.xml",\n    "x1": ASSET_ROOT / "x1" / "x1.xml",'
        )
        # Add to IK_CONFIG_DICT["smplx"]
        content = content.replace(
            '"rpo": IK_CONFIG_ROOT / "smplx_to_rpo.json",',
            '"rpo": IK_CONFIG_ROOT / "smplx_to_rpo.json",\n        "x1": IK_CONFIG_ROOT / "smplx_to_x1.json",'
        )
        # Add to ROBOT_BASE_DICT
        content = content.replace(
            '"rpo": "base_link",',
            '"rpo": "base_link",\n    "x1": "base_link",'
        )
        # Add to VIEWER_CAM_DISTANCE_DICT
        content = content.replace(
            '"rpo": 2.0,\n}',
            '"rpo": 2.0,\n    "x1": 2.0,\n}'
        )
        params_file.write_text(content)
        print("[INFO] Patched params.py with X1 registration")
    else:
        print("[INFO] X1 already registered in params.py")


# ── Step 4: Run GMR batch retargeting via subprocess ──────────────
def run_gmr_retarget(gmr_dir: Path, venv_dir: Path):
    """Run GMR retargeting in isolated venv via subprocess."""
    output_dir = REPO_ROOT / "roboparty_train" / "robolab" / "data" / "motions" / "x1_gmr"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write the retarget worker script
    worker_script = gmr_dir / "run_x1_batch.py"
    worker_code = '''
import sys, os, pickle, numpy as np, torch
from pathlib import Path

# Ensure GMR is importable
gmr_root = Path(__file__).parent
sys.path.insert(0, str(gmr_root))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from general_motion_retargeting.kinematics_model import KinematicsModel

repo_root = Path(os.environ["X1_REPO_ROOT"])
smplx_folder = gmr_root / "assets" / "body_models" / "smplx"
output_dir = repo_root / "roboparty_train" / "robolab" / "data" / "motions" / "x1_gmr"

# Collect AMASS npz files
npz_files = []
for subdir in ["CMU", "BMLrub_stageii"]:
    d = repo_root / "AMASS_minimal" / subdir
    if d.is_dir():
        npz_files.extend(sorted(d.glob("**/*.npz")))

print(f"[GMR] Found {len(npz_files)} AMASS npz files", flush=True)

kinematics_model = None
retargeter = None

for i, npz_file in enumerate(npz_files):
    out_name = npz_file.stem.replace("_stageii", "") + ".pkl"
    out_path = output_dir / out_name
    if out_path.exists():
        print(f"[GMR] [{i+1}/{len(npz_files)}] SKIP (exists): {out_name}", flush=True)
        continue
    print(f"[GMR] [{i+1}/{len(npz_files)}] Retargeting: {npz_file.name}", flush=True)
    try:
        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(str(npz_file), str(smplx_folder))
        src_fps = smplx_data["mocap_frame_rate"].item()
        smplx_frame_data_list, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=src_fps)

        if retargeter is None:
            retargeter = GMR(src_human="smplx", tgt_robot="x1", actual_human_height=actual_human_height)
            kinematics_model = KinematicsModel(retargeter.xml_file, device="cuda:0")

        qpos_list = []
        for smplx_frame_data in smplx_frame_data_list:
            qpos = retargeter.retarget(smplx_frame_data)
            qpos_list.append(qpos.copy())
        qpos_list = np.array(qpos_list)

        root_pos = qpos_list[:, :3].copy()
        root_rot = qpos_list[:, 3:7].copy()
        root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]
        dof_pos = qpos_list[:, 7:].copy()

        # Height adjust
        body_pos, body_rot = kinematics_model.forward_kinematics(
            torch.tensor(root_pos, device="cuda:0", dtype=torch.float32),
            torch.tensor(root_rot, device="cuda:0", dtype=torch.float32),
            torch.tensor(dof_pos, device="cuda:0", dtype=torch.float32),
        )
        lowest = torch.min(body_pos[..., 2]).item()
        root_pos[:, 2] -= lowest
        root_pos[:, :2] -= root_pos[0, :2]

        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_names": kinematics_model.dof_names,
            "body_names": kinematics_model.body_names,
            "dof_positions": dof_pos,
            "dof_pos": dof_pos,
            "body_positions": body_pos.cpu().numpy(),
            "body_rotations": body_rot.cpu().numpy(),
            "local_body_pos": body_pos.cpu().numpy(),
        }
        with open(out_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"[GMR]   Saved: {out_path.name} ({len(qpos_list)} frames, {dof_pos.shape[1]} DOF)", flush=True)
    except Exception as e:
        print(f"[GMR]   FAILED: {e}", flush=True)

print("[GMR] Batch retargeting complete.", flush=True)
'''
    worker_script.write_text(worker_code)
    print(f"[INFO] Wrote worker script to {worker_script}")

    # Run in venv
    venv_python = str(venv_dir / "bin" / "python")
    env = {**os.environ, "X1_REPO_ROOT": str(REPO_ROOT)}

    print("[INFO] Running GMR batch retargeting in venv...")
    result = subprocess.run(
        [venv_python, str(worker_script)],
        env=env,
        capture_output=False,
        text=True
    )

    if result.returncode != 0:
        print(f"[ERROR] GMR retargeting failed with exit code {result.returncode}")
    else:
        print("[INFO] GMR retargeting completed successfully")

    pkl_count = len(list(output_dir.glob("*.pkl")))
    print(f"[INFO] Output: {output_dir} ({pkl_count} pkl files)")
    return output_dir


# ── Main ───────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("X1 GMR Retargeting Pipeline")
    print("=" * 60)

    # Step 1
    reassemble_smplx()

    # Step 2
    gmr_dir, venv_dir = setup_gmr()

    # Step 3
    register_x1_in_gmr(gmr_dir)

    # Step 4
    gmr_output = run_gmr_retarget(gmr_dir, venv_dir)

    print("\n" + "=" * 60)
    print(f"DONE! GMR output: {gmr_output}")
    print(f"Files: {len(list(gmr_output.glob('*.pkl')))}")
    print("=" * 60)


if __name__ == "__main__":
    main()
