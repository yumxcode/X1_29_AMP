#!/usr/bin/env python3
"""
sim2sim task entry (runs INSIDE a Gradmotion container, not on Mac).

Mounts the Isaac-trained X1 AMP checkpoint (via task checkPointFilePath),
rolls the policy out in MuJoCo (x1.xml), renders walking videos, and mirrors
everything to SDK-scan locations.

Usage (startScript):
    gm-run X1_29_AMP/sim2sim/run_sim2sim_task.py --headless --ckpt <path-or-glob>
"""

import functools
import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

print = functools.partial(print, flush=True)

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
if not (REPO_ROOT / "AMASS_minimal").is_dir() and (REPO_ROOT.parent / "AMASS_minimal").is_dir():
    REPO_ROOT = REPO_ROOT.parent
print(f"[INFO] REPO_ROOT = {REPO_ROOT}")

UPLOAD_DIR = REPO_ROOT / "model_upload"
VIDEO_DIR = REPO_ROOT / "logs" / "x1_sim2sim"


def find_ckpt(hint: str) -> Path | None:
    """Locate the checkpoint: explicit path, mounted resume checkpoint, or
    newest model_*.pt anywhere under the repo."""
    cands = []
    if hint:
        cands += [Path(hint)] + [Path(p) for p in glob.glob(hint)]
    for pat in ["/workspace/**/*.pt", "/personal/**/*.pt",
                "/workspace/isaaclab/**/*.pt"]:
        try:
            cands += [Path(p) for p in glob.glob(pat, recursive=True)]
        except Exception:
            pass
    cands += sorted((REPO_ROOT / "model_upload").glob("model_*.pt")) if (REPO_ROOT / "model_upload").exists() else []
    best = None
    for c in cands:
        try:
            if c.is_file() and c.name.startswith("model_") and c.name.endswith(".pt"):
                stem = c.name[len("model_"):-3]
                if stem.isdigit() and (best is None or int(stem) > best[0]):
                    best = (int(stem), c)
        except Exception:
            continue
    return best[1] if best else None


def run(cmd, env=None, timeout=None):
    print(f"[RUN] {' '.join(map(str, cmd))}")
    return subprocess.run([str(c) for c in cmd], env=env, timeout=timeout,
                          cwd=str(REPO_ROOT))


def main():
    hint = None
    if "--ckpt" in sys.argv:
        hint = sys.argv[sys.argv.index("--ckpt") + 1]

    print("=" * 60)
    print("X1 sim2sim task (MuJoCo rollout of Isaac AMP policy)")
    print("=" * 60)

    ckpt = find_ckpt(hint)
    if ckpt is None:
        print("[FATAL] no model_*.pt checkpoint found (mount via "
              "checkPointFilePath or pass --ckpt)")
        sys.exit(1)
    print(f"[INFO] checkpoint: {ckpt}")

    # deps (Isaac python already has torch; add mujoco + video encoders)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "mujoco", "imageio", "imageio-ffmpeg"], check=False)

    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")  # headless GPU offscreen; osmesa fallback below

    rollouts = [
        ("walk_1.0", ["--cmd", "1.0", "0.0", "0.0", "--duration", "12"]),
        ("walk_1.5", ["--cmd", "1.5", "0.0", "0.0", "--duration", "10"]),
        ("walk_turn", ["--cmd", "1.0", "0.0", "0.8", "--duration", "10"]),
    ]

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    ok = 0
    for name, extra in rollouts:
        mp4 = VIDEO_DIR / f"x1_sim2sim_{name}.mp4"
        js = VIDEO_DIR / f"x1_sim2sim_{name}.json"
        cmd = [sys.executable, REPO_ROOT / "sim2sim" / "mujoco_rollout.py",
               "--ckpt", ckpt, "--repo-root", REPO_ROOT,
               "--video", mp4, "--json", js] + extra
        print(f"\n--- rollout: {name} ---")
        r = run(cmd, env=env, timeout=900)
        if r.returncode != 0 and env.get("MUJOCO_GL") == "egl":
            print("[WARN] EGL render failed, retrying with osmesa")
            env["MUJOCO_GL"] = "osmesa"
            r = run(cmd, env=env, timeout=900)
        if r.returncode == 0 and mp4.exists():
            ok += 1
            shutil.copy2(mp4, UPLOAD_DIR / mp4.name)
            print(f"[OK] {mp4.name} ({mp4.stat().st_size // 1024}KB)")
        else:
            print(f"[FAIL] rollout {name} rc={r.returncode}")

    print(f"\n[INFO] {ok}/{len(rollouts)} rollouts produced videos")
    print("[INFO] Waiting 180s for SDK upload...")
    time.sleep(180)
    print("[INFO] done")


if __name__ == "__main__":
    main()
