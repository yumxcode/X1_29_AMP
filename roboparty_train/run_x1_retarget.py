#!/usr/bin/env python3
"""X1 retarget entry: install deps then run retarget.

This script is designed to be called via gm-run.
It installs robolab and rsl_rl packages, then runs the retarget.
"""
import subprocess
import sys
import os


def run(cmd, cwd=None):
    """Run a command and stream output."""
    print(f"[run_x1_retarget] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        print(f"[run_x1_retarget] Command failed with exit code {result.returncode}", flush=True)
        sys.exit(result.returncode)


def main():
    # Find workspace root (where the repo was cloned)
    # gm-run clones the repo into the workspace, so the repo files are at /workspace/
    workspace = os.environ.get("WORKSPACE", "/workspace")
    
    robolab_dir = os.path.join(workspace, "roboparty_train", "robolab")
    rsl_rl_dir = os.path.join(workspace, "roboparty_train", "rsl_rl")

    # Step 1: Install robolab
    print("=" * 60, flush=True)
    print("STEP 1: Installing robolab package", flush=True)
    print("=" * 60, flush=True)
    if os.path.isdir(robolab_dir):
        run([sys.executable, "-m", "pip", "install", "-e", robolab_dir, "-q"])
    else:
        print(f"WARNING: robolab dir not found at {robolab_dir}", flush=True)
        # Try alternative paths
        for alt in ["/workspace/roboparty_train/robolab", os.path.expanduser("~/roboparty_train/robolab")]:
            if os.path.isdir(alt):
                robolab_dir = alt
                print(f"Found robolab at {alt}", flush=True)
                run([sys.executable, "-m", "pip", "install", "-e", robolab_dir, "-q"])
                break

    # Step 2: Install rsl_rl
    print("=" * 60, flush=True)
    print("STEP 2: Installing rsl_rl package", flush=True)
    print("=" * 60, flush=True)
    if os.path.isdir(rsl_rl_dir):
        run([sys.executable, "-m", "pip", "install", "-e", rsl_rl_dir, "-q"])
    else:
        print(f"WARNING: rsl_rl dir not found at {rsl_rl_dir}", flush=True)

    # Step 3: Run retarget
    print("=" * 60, flush=True)
    print("STEP 3: Running X1 retarget", flush=True)
    print("=" * 60, flush=True)

    retarget_script = os.path.join(workspace, "roboparty_train", "robolab", "scripts", "tools", "retarget", "dataset_retarget.py")
    config_file = os.path.join(workspace, "roboparty_train", "robolab", "scripts", "tools", "retarget", "config", "x1.yaml")
    input_dir = os.path.join(workspace, "roboparty_train", "robolab", "data", "motions", "rpo_gmr")
    output_dir = os.path.join(workspace, "roboparty_train", "robolab", "data", "motions", "x1_lab")

    run([
        sys.executable, retarget_script,
        "--robot", "x1",
        "--input_dir", input_dir,
        "--output_dir", output_dir,
        "--config_file", config_file,
        "--loop", "clamp",
        "--headless",
    ])

    print("=" * 60, flush=True)
    print("RETARGET COMPLETE!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
