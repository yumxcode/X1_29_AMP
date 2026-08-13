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
    # SDK scans: logs/{exp}/exported_data/{load_run}/model_*.pt
    # Must use model_ prefix and exported_data directory structure
    print("\n--- Packaging Retarget Results for Download ---")
    import pickle as _pkl
    from datetime import datetime as _dt

    # SDK upload path: logs/rsl_rl/x1_amp/exported_data/{run_name}/
    sdk_run_name = _dt.now().strftime("%Y-%m-%d_%H-%M-%S") + "x1_amp"
    sdk_export_dir = REPO_ROOT / "logs" / "rsl_rl" / "x1_amp" / "exported_data" / sdk_run_name
    sdk_export_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] SDK export dir: {sdk_export_dir}")

    # Delete old RPO .pt files from repo (they pollute SDK's .pt scan)
    old_pt_dirs = [
        REPO_ROOT / "roboparty_train" / "robolab" / "data" / "motions" / "rpo_lab",
        REPO_ROOT / "roboparty_train" / "robolab" / "data" / "motions" / "rpo_gmr",
        REPO_ROOT / "GMR" / "gvhmr_pt",
    ]
    for d in old_pt_dirs:
        if d.exists():
            for pt in d.glob("*.pt"):
                pt.unlink()
                print(f"  Deleted old: {pt.relative_to(REPO_ROOT)}")

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

    # Must use model_ prefix for SDK to detect and upload
    retarget_pt = sdk_export_dir / "model_retarget_data.pt"
    with open(retarget_pt, "wb") as fh:
        _pkl.dump(retarget_pkg, fh)
    size_mb = retarget_pt.stat().st_size / 1e6
    print(f"[INFO] Packaged retarget data → {retarget_pt.relative_to(REPO_ROOT)} ({size_mb:.1f}MB)")
    print("[INFO] SDK will scan exported_data/ for model_*.pt files")

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

    # Start a background thread to copy checkpoints to SDK scan directory
    # SDK scans: logs/{exp}/exported_data/{load_run}/model_*.pt
    # Checkpoints are saved to logs/rsl_rl/x1_amp/{timestamp}/model_*.pt (subdirectory)
    # We copy them to exported_data/{run_name}/ for SDK upload
    import threading, glob, shutil

    rsl_rl_log_root = REPO_ROOT / "logs" / "rsl_rl" / "x1_amp"
    rsl_rl_log_root.mkdir(parents=True, exist_ok=True)
    stop_monitor = threading.Event()

    def checkpoint_monitor():
        """Background thread: every 60s, copy new checkpoints to exported_data dir."""
        uploaded = set()
        while not stop_monitor.is_set():
            # Find latest run directory (Isaac Lab creates timestamped subdirs)
            run_dirs = sorted(rsl_rl_log_root.glob("*/"), key=lambda x: x.stat().st_mtime)
            for run_dir in run_dirs:
                # Skip exported_data itself
                if run_dir.name == "exported_data":
                    continue
                for ckpt in sorted(run_dir.glob("model_*.pt")):
                    if ckpt.name not in uploaded:
                        dst = sdk_export_dir / ckpt.name
                        shutil.copy2(ckpt, dst)
                        uploaded.add(ckpt.name)
                        print(f"[MONITOR] Copied {ckpt.name} → exported_data/{sdk_run_name}/")
            stop_monitor.wait(60)  # sleep 60s or until stopped

    monitor_thread = threading.Thread(target=checkpoint_monitor, daemon=True)
    monitor_thread.start()
    print(f"[INFO] Background checkpoint monitor started → exported_data/{sdk_run_name}/")

    print(f"[INFO] Starting AMP training: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    # Stop monitor
    stop_monitor.set()
    monitor_thread.join(timeout=5)

    # Final checkpoint copy: copy ALL checkpoints to exported_data
    run_dirs = sorted(rsl_rl_log_root.glob("*/"), key=lambda x: x.stat().st_mtime) if rsl_rl_log_root.exists() else []
    for run_dir in run_dirs:
        if run_dir.name == "exported_data":
            continue
        for ckpt in sorted(run_dir.glob("model_*.pt")):
            dst = sdk_export_dir / ckpt.name
            if not dst.exists():
                shutil.copy2(ckpt, dst)
                print(f"[FINAL] Copied {ckpt.name} → exported_data/{sdk_run_name}/")

    # List what we have for SDK upload
    exported_files = sorted(sdk_export_dir.glob("model_*.pt"))
    print(f"\n[INFO] Exported {len(exported_files)} files to SDK scan path:")
    for f in exported_files:
        print(f"  {f.name} ({f.stat().st_size // 1024}KB)")

    if result.returncode != 0:
        print(f"[ERROR] AMP training exited with code {result.returncode}")
    else:
        print("[INFO] AMP training completed!")

    # Wait for SDK to detect and upload files
    import time
    print("[INFO] Waiting 120s for SDK file upload...")
    time.sleep(120)
    print("[INFO] Done waiting.")


if __name__ == "__main__":
    main()
