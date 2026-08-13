#!/usr/bin/env python3
"""
Full X1 AMP training pipeline: retarget + train in one container.

1. Reassemble SMPLX
2. GMR retarget (isolated venv)
3. Isaac Lab dataset_retarget
4. AMP training

Usage:
    python run_x1_amp_train.py --headless
"""

import functools
import os
import shutil
import subprocess
import sys
import numpy as np
from pathlib import Path

print = functools.partial(print, flush=True)

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

# Import the GMR pipeline functions (reuse from run_gmr_retarget.py)
sys.path.insert(0, str(SCRIPT_DIR))
from run_gmr_retarget import (
    reassemble_smplx, setup_gmr, register_x1_in_gmr,
    run_auto_ik, run_gmr_retarget, run_dataset_retarget
)


def main():
    print("=" * 60)
    print("X1 Full AMP Training Pipeline")
    print("=" * 60)

    # === Phase 1: GMR Retarget ===
    print("\n=== Phase 1: GMR Retarget ===\n")
    reassemble_smplx()
    gmr_dir, venv_dir = setup_gmr()
    register_x1_in_gmr(gmr_dir)

    print("\n--- Auto-IK Calibration ---")
    auto_config = run_auto_ik(gmr_dir, venv_dir)

    print("\n--- Batch Retargeting ---")
    gmr_output = REPO_ROOT / "roboparty_train" / "robolab" / "data" / "motions" / "x1_gmr"
    if gmr_output.exists() and len(list(gmr_output.glob("*.pkl"))) >= 14:
        print(f"[INFO] x1_gmr already has {len(list(gmr_output.glob('*.pkl')))} files, skipping GMR retarget")
    else:
        run_gmr_retarget(gmr_dir, venv_dir)

    # === Phase 2: Isaac Lab Dataset Retarget ===
    print("\n=== Phase 2: Isaac Lab Dataset Retarget ===\n")
    lab_output = REPO_ROOT / "roboparty_train" / "robolab" / "data" / "motions" / "x1_lab"
    if lab_output.exists() and len(list(lab_output.glob("*.pkl"))) >= 14:
        print(f"[INFO] x1_lab already has {len(list(lab_output.glob('*.pkl')))} files, skipping dataset_retarget")
    else:
        run_dataset_retarget(gmr_output)

    # Verify lab files
    lab_files = list(lab_output.glob("*.pkl"))
    print(f"\n[INFO] x1_lab: {len(lab_files)} files")
    if len(lab_files) < 10:
        print("[ERROR] Not enough lab files for AMP training!")
        sys.exit(1)

    # Package retarget results for SDK upload
    # train.py saves to: logs/rsl_rl/{experiment_name}/{timestamp_run}/
    # SDK scans {log_dir} root = logs/rsl_rl/{experiment_name}/ for .pt files
    print("\n--- Packaging Retarget Results for Download ---")
    import pickle as _pkl
    sdk_scan_dir = REPO_ROOT / "logs" / "rsl_rl" / "x1_amp"
    sdk_scan_dir.mkdir(parents=True, exist_ok=True)

    # Use pickle (not torch) to avoid numpy/torch import conflicts
    retarget_pkg = {}
    for f in sorted(lab_files):
        retarget_pkg[f"x1_lab/{f.name}"] = f.read_bytes()
        print(f"  Added: x1_lab/{f.name} ({f.stat().st_size // 1024}KB)")
    for f in sorted(gmr_output.glob("*.pkl")):
        retarget_pkg[f"x1_gmr/{f.name}"] = f.read_bytes()
    auto_cfg = REPO_ROOT / "AMASS_minimal" / "smplx_to_x1_auto.json"
    if auto_cfg.exists():
        retarget_pkg["smplx_to_x1_auto.json"] = auto_cfg.read_bytes()

    retarget_pt = sdk_scan_dir / "x1_retarget_data.pt"
    with open(retarget_pt, "wb") as fh:
        _pkl.dump(retarget_pkg, fh)
    size_mb = retarget_pt.stat().st_size / 1e6
    print(f"[INFO] Packaged retarget data → {retarget_pt.name} ({size_mb:.1f}MB)")
    print(f"[INFO] Written to {retarget_pt}")
    print("[INFO] SDK will auto-upload from logs/rsl_rl/x1_amp/ when task completes")

    # === Phase 3: AMP Training ===
    print("\n=== Phase 3: AMP Training ===\n")

    robolab_src = REPO_ROOT / "roboparty_train" / "robolab"
    rsl_rl_src = REPO_ROOT / "roboparty_train" / "rsl_rl"

    # Ensure robolab/rsl_rl are installed (force reinstall rsl_rl to override Isaac Lab's version)
    print("[INFO] Installing robolab/rsl_rl (force rsl_rl to override Isaac Lab v3.1.2)...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "rsl-rl-lib", "-y", "-q"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(rsl_rl_src), "-q",
                    "--force-reinstall", "--no-deps"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(robolab_src), "-q",
                    "--no-deps"], check=True)

    # Add to sys.path for import
    for p in [str(robolab_src), str(rsl_rl_src)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    train_script = REPO_ROOT / "roboparty_train" / "robolab" / "scripts" / "rsl_rl" / "train.py"

    cmd = [
        sys.executable, str(train_script),
        "--task=X1-AMP",
        "--headless",
        "--logger=tensorboard",
        "--num_envs=4096",
    ]

    print(f"[INFO] Starting AMP training: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    # Phase 3b: Git push checkpoint (protect against balance exhaustion)
    print("\n--- Phase 3b: Saving Checkpoint to Git ---")
    import glob
    log_root = REPO_ROOT / "logs" / "rsl_rl" / "x1_amp"
    if log_root.exists():
        # Find latest run directory
        run_dirs = sorted([d for d in log_root.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime)
        if run_dirs:
            latest_run = run_dirs[-1]
            # Find all checkpoints
            checkpoints = sorted(latest_run.glob("model_*.pt"))
            if checkpoints:
                print(f"[INFO] Found {len(checkpoints)} checkpoints in {latest_run.name}")
                latest_ckpt = checkpoints[-1]
                print(f"[INFO] Latest: {latest_ckpt.name}")
                # Copy to a stable path for SDK upload
                stable_path = log_root / "latest_checkpoint.pt"
                import shutil
                shutil.copy2(latest_ckpt, stable_path)
                print(f"[INFO] Copied to {stable_path}")
            else:
                print("[WARN] No checkpoints found in run directory")
        else:
            print("[WARN] No run directories found")
    else:
        print(f"[WARN] Log root {log_root} does not exist")

    if result.returncode != 0:
        print(f"[ERROR] AMP training exited with code {result.returncode}")
    else:
        print("[INFO] AMP training completed!")


if __name__ == "__main__":
    main()
