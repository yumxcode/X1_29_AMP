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

    if result.returncode != 0:
        print(f"[ERROR] AMP training failed with exit code {result.returncode}")
    else:
        print("[INFO] AMP training completed!")


if __name__ == "__main__":
    main()
