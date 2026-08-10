#!/usr/bin/env python3
"""
GMR retargeting pipeline for X1 humanoid on Gradmotion.

Pipeline:
1. Reassemble SMPLX_NEUTRAL.pkl from git chunks
2. Clone and install GMR in a separate venv
3. Register X1 in GMR's params.py
4. Run GMR auto-IK generator → smplx_to_x1_auto.json (calibrated config)
5. Run GMR batch retargeting on AMASS data → x1_gmr/*.pkl

Usage:
    python run_gmr_retarget.py --headless
"""

import functools
import os
import shutil
import subprocess
import sys
import numpy as np
from pathlib import Path

print = functools.partial(print, flush=True)

# IMPORTANT: Do NOT import isaaclab or start AppLauncher.
# GMR has its own MuJoCo-based pipeline that conflicts with Isaac Lab's numpy.

# ── Step 0: Locate workspace and repo root ──────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
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
    for p in [Path("/workspace/isaaclab/X1_29_AMP")]:
        if p.is_dir():
            REPO_ROOT = p
            break

if REPO_ROOT is None:
    print("[FATAL] Cannot find repo root")
    sys.exit(1)

print(f"[INFO] REPO_ROOT = {REPO_ROOT}")


# ── Step 1: Reassemble SMPLX_NEUTRAL.pkl ───────────────────────────
def reassemble_smplx():
    chunks_dir = REPO_ROOT / "AMASS_minimal" / "smplx_parts"
    output_dir = REPO_ROOT / "AMASS_minimal" / "smplx"
    output_file = output_dir / "SMPLX_NEUTRAL.pkl"

    if output_file.exists():
        print(f"[INFO] SMPLX_NEUTRAL.pkl exists ({output_file.stat().st_size} bytes)")
        return output_file

    print("[INFO] Reassembling SMPLX_NEUTRAL.pkl...")
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = sorted(chunks_dir.glob("SMPLX_NEUTRAL.pkl.part*"))
    with open(output_file, "wb") as f:
        for chunk in chunks:
            f.write(chunk.read_bytes())

    size = output_file.stat().st_size
    print(f"[INFO] Reassembled: {size} bytes ({size / 1e6:.1f} MB)")
    return output_file


# ── Step 2: Clone and install GMR in separate venv ────────────────
def setup_gmr():
    gmr_dir = REPO_ROOT / "GMR"

    if not (gmr_dir / "setup.py").exists():
        print("[INFO] Cloning GMR...")
        subprocess.check_call([
            "git", "clone", "--depth", "1",
            "https://github.com/Roboparty/GMR.git", str(gmr_dir)
        ])

    # Create isolated venv
    venv_dir = REPO_ROOT / "gmr_venv"
    if not venv_dir.exists():
        print("[INFO] Creating isolated venv for GMR...")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
        pip = str(venv_dir / "bin" / "pip")
        subprocess.check_call([pip, "install", "--upgrade", "pip", "-q"])
        subprocess.check_call([pip, "install", "-e", str(gmr_dir), "-q"])
        print("[INFO] GMR venv ready")

    # Copy SMPLX body model
    # smplx library expects: {model_path}/smplx/SMPLX_{gender}.{ext}
    # So model_path should be the PARENT of smplx/
    gmr_body_models_root = gmr_dir / "assets" / "body_models"
    gmr_body_models = gmr_body_models_root / "smplx"
    gmr_body_models.mkdir(parents=True, exist_ok=True)
    smplx_pkl = REPO_ROOT / "AMASS_minimal" / "smplx" / "SMPLX_NEUTRAL.pkl"
    
    # Copy NEUTRAL to all 3 genders (GMR requires all 3)
    for gender in ["NEUTRAL", "MALE", "FEMALE"]:
        target = gmr_body_models / f"SMPLX_{gender}.pkl"
        if not target.exists() and smplx_pkl.exists():
            shutil.copy2(smplx_pkl, target)
            print(f"[INFO] Copied SMPLX_{gender}.pkl to GMR body_models")

    # Patch GMR's smpl.py to use ext='pkl' instead of default 'npz'
    # smplx library constructs filename as f'SMPLX_{gender}.{ext}'
    # Default ext='npz' → looks for .npz files we don't have
    gmr_smpl_py = gmr_dir / "general_motion_retargeting" / "utils" / "smpl.py"
    if gmr_smpl_py.exists():
        content = gmr_smpl_py.read_text()
        if "ext='pkl'" not in content and 'ext="pkl"' not in content:
            # Add ext='pkl' to all smplx.create() calls
            content = content.replace(
                'use_pca=False,',
                "use_pca=False,\n        ext='pkl',"
            )
            gmr_smpl_py.write_text(content)
            print("[INFO] Patched GMR smpl.py: added ext='pkl' to smplx.create()")

    return gmr_dir, venv_dir


# ── Step 3: Register X1 in GMR ─────────────────────────────────────
def register_x1_in_gmr(gmr_dir: Path):
    # 3a. Copy X1 MJCF + meshes to GMR assets
    x1_assets_src = REPO_ROOT / "gmr_x1_assets"
    x1_assets_dst = gmr_dir / "assets" / "x1"
    if x1_assets_dst.exists():
        shutil.rmtree(x1_assets_dst)
    shutil.copytree(x1_assets_src, x1_assets_dst)
    print(f"[INFO] Copied X1 assets to {x1_assets_dst}")

    # 3b. Copy rough IK config (as input for auto-IK)
    ik_src = REPO_ROOT / "AMASS_minimal" / "smplx_to_x1.json"
    ik_dst = gmr_dir / "general_motion_retargeting" / "ik_configs" / "smplx_to_x1.json"
    shutil.copy2(ik_src, ik_dst)
    print(f"[INFO] Copied IK config to {ik_dst}")

    # 3c. Copy T-pose JSON
    tpose_src = x1_assets_dst / "x1_tpose.json"
    tpose_dst = gmr_dir / "ik_config_manager" / "pose_inits" / "x1_tpose.json"
    shutil.copy2(tpose_src, tpose_dst)
    print(f"[INFO] Copied T-pose to {tpose_dst}")

    # 3d. Patch params.py
    params_file = gmr_dir / "general_motion_retargeting" / "params.py"
    content = params_file.read_text()
    if '"x1"' not in content:
        content = content.replace(
            '"rpo": ASSET_ROOT / "rpo" / "rpo.xml",',
            '"rpo": ASSET_ROOT / "rpo" / "rpo.xml",\n    "x1": ASSET_ROOT / "x1" / "x1.xml",'
        )
        content = content.replace(
            '"rpo": IK_CONFIG_ROOT / "smplx_to_rpo.json",',
            '"rpo": IK_CONFIG_ROOT / "smplx_to_rpo.json",\n        "x1": IK_CONFIG_ROOT / "smplx_to_x1.json",'
        )
        content = content.replace(
            '"rpo": "base_link",',
            '"rpo": "base_link",\n    "x1": "base_link",'
        )
        content = content.replace(
            '"rpo": 2.0,\n}',
            '"rpo": 2.0,\n    "x1": 2.0,\n}'
        )
        params_file.write_text(content)
        print("[INFO] Patched params.py with X1 registration")
    else:
        print("[INFO] X1 already registered in params.py")

    # 3e. Patch generate_keypoint_mapping_smplx.py to add x1 to choices
    gen_file = gmr_dir / "ik_config_manager" / "generate_keypoint_mapping_smplx.py"
    gen_content = gen_file.read_text()
    if '"x1"' not in gen_content:
        gen_content = gen_content.replace(
            '"openloong", "tienkung","joyin","joyin_add", "rpo"]',
            '"openloong", "tienkung","joyin","joyin_add", "rpo", "x1"]'
        )
        gen_file.write_text(gen_content)
        print("[INFO] Patched generate_keypoint_mapping_smplx.py with x1 choice")


# ── Step 4: Run GMR auto-IK generator ──────────────────────────────
def run_auto_ik(gmr_dir: Path, venv_dir: Path):
    """Run GMR's auto-IK config generator to calibrate scale/offset/quaternion."""
    output_config = gmr_dir / "general_motion_retargeting" / "ik_configs" / "smplx_to_x1_auto.json"

    if output_config.exists():
        print(f"[INFO] Auto-IK config already exists: {output_config}")
        # Use it as the active config
        active_config = gmr_dir / "general_motion_retargeting" / "ik_configs" / "smplx_to_x1.json"
        shutil.copy2(output_config, active_config)
        return output_config

    venv_python = str(venv_dir / "bin" / "python")
    gen_script = gmr_dir / "ik_config_manager" / "generate_keypoint_mapping_smplx.py"

    cmd = [
        venv_python, str(gen_script),
        "--smplx_file", str(gmr_dir / "ik_config_manager" / "SMPLX_TPOSE_UNIFIED_AMASS.npz"),
        "--robot", "x1",
        "--robot_qpos_init", str(gmr_dir / "ik_config_manager" / "pose_inits" / "x1_tpose.json"),
        "--ik_config_in", str(gmr_dir / "general_motion_retargeting" / "ik_configs" / "smplx_to_x1.json"),
        "--ik_config_out", str(output_config),
    ]

    print(f"[INFO] Running auto-IK generator...")
    print(f"[INFO] Command: {' '.join(cmd)}")

    env = {**os.environ, "X1_REPO_ROOT": str(REPO_ROOT)}

    result = subprocess.run(cmd, env=env, cwd=str(gmr_dir))

    # NOTE: The auto-IK generator may exit with non-zero due to GLFWError
    # (headless X11 display missing), but the config file is still generated.
    # Check for the output file regardless of exit code.
    if output_config.exists():
        print(f"[INFO] Auto-IK config generated: {output_config}")
        # Copy as active config
        active_config = gmr_dir / "general_motion_retargeting" / "ik_configs" / "smplx_to_x1.json"
        shutil.copy2(output_config, active_config)
        print(f"[INFO] Updated active IK config with auto-calibrated version")

        # Also save to repo for reference
        repo_copy = REPO_ROOT / "AMASS_minimal" / "smplx_to_x1_auto.json"
        shutil.copy2(output_config, repo_copy)

        return output_config
    else:
        if result.returncode != 0:
            print(f"[ERROR] Auto-IK failed with exit code {result.returncode} and no output file")
        print("[WARN] Falling back to manual config")
        return None


# ── Step 5: Run GMR batch retargeting via subprocess ──────────────
def run_gmr_retarget(gmr_dir: Path, venv_dir: Path):
    output_dir = REPO_ROOT / "roboparty_train" / "robolab" / "data" / "motions" / "x1_gmr"
    output_dir.mkdir(parents=True, exist_ok=True)

    worker_script = gmr_dir / "run_x1_batch.py"
    worker_code = '''
import sys, os, pickle, numpy as np, torch
from pathlib import Path

gmr_root = Path(__file__).parent
sys.path.insert(0, str(gmr_root))

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from general_motion_retargeting.kinematics_model import KinematicsModel

repo_root = Path(os.environ["X1_REPO_ROOT"])
smplx_folder = gmr_root / "assets" / "body_models"  # smplx lib appends /smplx/ internally
output_dir = repo_root / "roboparty_train" / "robolab" / "data" / "motions" / "x1_gmr"

npz_files = []
for subdir in ["CMU", "BMLrub_stageii"]:
    d = repo_root / "AMASS_minimal" / subdir
    if d.is_dir():
        npz_files.extend(sorted(d.glob("**/*.npz")))

print(f"[GMR] Found {len(npz_files)} AMASS npz files", flush=True)

kinematics_model = None
retargeter = None
successful = 0
failed = 0

for i, npz_file in enumerate(npz_files):
    out_name = npz_file.stem.replace("_stageii", "") + ".pkl"
    out_path = output_dir / out_name
    if out_path.exists():
        print(f"[GMR] [{i+1}/{len(npz_files)}] SKIP (exists): {out_name}", flush=True)
        successful += 1
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

        body_pos, body_rot = kinematics_model.forward_kinematics(
            torch.tensor(root_pos, device="cuda:0", dtype=torch.float32),
            torch.tensor(root_rot, device="cuda:0", dtype=torch.float32),
            torch.tensor(dof_pos, device="cuda:0", dtype=torch.float32),
        )
        lowest = torch.min(body_pos[..., 2]).item()
        root_pos[:, 2] -= lowest
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
        print(f"[GMR]   OK: {out_path.name} ({len(qpos_list)} frames, {dof_pos.shape[1]} DOF)", flush=True)
        successful += 1
    except Exception as e:
        print(f"[GMR]   FAIL: {e}", flush=True)
        failed += 1

print(f"[GMR] Done: {successful} ok, {failed} fail", flush=True)
'''
    worker_script.write_text(worker_code)

    venv_python = str(venv_dir / "bin" / "python")
    env = {**os.environ, "X1_REPO_ROOT": str(REPO_ROOT)}

    print("[INFO] Running GMR batch retargeting...")
    result = subprocess.run([venv_python, str(worker_script)], env=env)

    pkl_count = len(list(output_dir.glob("*.pkl")))
    print(f"[INFO] Output: {output_dir} ({pkl_count} pkl files)")
    return output_dir


# ── Main ───────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("X1 GMR Retargeting Pipeline (with Auto-IK)")
    print("=" * 60)

    reassemble_smplx()
    gmr_dir, venv_dir = setup_gmr()
    register_x1_in_gmr(gmr_dir)

    # Step 4: Auto-IK calibration
    print("\n--- Step 4: Auto-IK Config Generation ---")
    auto_config = run_auto_ik(gmr_dir, venv_dir)

    # Step 5: Batch retarget
    print("\n--- Step 5: Batch Retargeting ---")
    gmr_output = run_gmr_retarget(gmr_dir, venv_dir)

    print("\n" + "=" * 60)
    print(f"DONE! GMR output: {gmr_output}")
    print(f"Files: {len(list(gmr_output.glob('*.pkl')))}")
    if auto_config:
        print(f"Auto-IK config: {auto_config}")
    print("=" * 60)

    # Step 6: Package results as .pt for SDK auto-upload
    print("\n--- Step 6: Package Results ---")
    package_results(gmr_output, auto_config)


def package_results(gmr_output: Path, auto_config_path=None):
    """Package all pkl files into a single .pt file for Gradmotion SDK auto-upload.

    Gradmotion SDK auto-uploads .pt files to cloud storage. We package all
    retargeted motion pkls + auto-IK config into one .pt file, which the SDK
    will detect and upload. Then we can download via `gm task model list`.
    """
    import pickle
    import struct

    # Collect all pkl files
    pkl_files = sorted(gmr_output.glob("*.pkl"))
    if not pkl_files:
        print("[ERROR] No pkl files to package")
        return

    # Pack into a single dict and save as .pt (torch format)
    # Use torch.save since the SDK recognizes .pt extension
    import torch

    package = {}
    for f in pkl_files:
        with open(f, 'rb') as fh:
            package[f.name] = fh.read()  # raw bytes
        print(f"  Added: {f.name} ({f.stat().st_size // 1024}KB)")

    # Add auto-IK config if available
    auto_cfg_path = REPO_ROOT / "AMASS_minimal" / "smplx_to_x1_auto.json"
    if auto_cfg_path.exists():
        package["smplx_to_x1_auto.json"] = auto_cfg_path.read_bytes()
        print(f"  Added: smplx_to_x1_auto.json")

    # Save as .pt in a location the SDK will detect
    # SDK scans /workspace/isaaclab/X1_29_AMP/ recursively
    output_pt = REPO_ROOT / "x1_gmr_results.pt"
    torch.save(package, output_pt)
    size_mb = output_pt.stat().st_size / 1e6
    print(f"\n[INFO] Packaged {len(pkl_files)} files → {output_pt.name} ({size_mb:.1f}MB)")
    print("[INFO] Waiting 60s for Gradmotion SDK to upload...")
    import time
    time.sleep(60)
    print("[INFO] Wait complete. Check gm task model list for download URL.")


if __name__ == "__main__":
    main()
