#!/usr/bin/env python3
"""
X1 AMP training pipeline v18 (acceptance-gated).

Phases:
  1. GMR retarget (SMPLX -> X1, isolated venv, installed OUTSIDE repo tree)
  2. Isaac Lab dataset_retarget (x1_gmr -> x1_lab)
  3. STRICT retarget acceptance gate (acceptance/check_retarget.py)
     -> FAIL aborts BEFORE burning GPU-hours on training
  4. AMP training (cwd=REPO_ROOT so logs land inside the SDK-scanned tree;
     stdout captured to file) with checkpoint upload monitor
  5. Play rollout + video (X1-AMP-Play, command 1.0 m/s forward)
  6. AMP training acceptance (acceptance/check_amp.py)
  7. Artifacts mirrored to repo-tree model_upload/ (SDK scans repo tree for
     *.pt during the run — this is the upload path that verifiably worked for
     gvhmr_pt in v16/v17) + exported_data/ paths.

Usage:
    python run_x1_amp_train.py --headless
"""

import functools
import os
import pickle as _pkl
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime as _dt
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

sys.path.insert(0, str(SCRIPT_DIR))
from run_gmr_retarget import (
    reassemble_smplx, setup_gmr, register_x1_in_gmr,
    run_auto_ik, run_gmr_retarget, run_dataset_retarget
)

# Repo-tree upload dir: SDK periodic scan registers *.pt here (proven pattern).
UPLOAD_DIR = REPO_ROOT / "model_upload"
MOTIONS_DIR = REPO_ROOT / "roboparty_train" / "robolab" / "data" / "motions"
TRAIN_LOG_FILE = REPO_ROOT / "train_stdout.log"
PLAY_LOG_FILE = REPO_ROOT / "play_stdout.log"


def wrap_json_for_upload(name: str, obj) -> Path:
    """SDK only uploads .pt files — wrap JSON payloads in a pickle .pt."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    p = UPLOAD_DIR / name
    with open(p, "wb") as f:
        _pkl.dump(obj, f)
    print(f"[UPLOAD] {p.relative_to(REPO_ROOT)} ({p.stat().st_size // 1024}KB)")
    return p


def wait_for_sdk(seconds: int, why: str):
    print(f"[INFO] Waiting {seconds}s for SDK upload queue ({why})...")
    time.sleep(seconds)


# ────────────────────────────────────────────────────────────────────
def phase_retarget():
    print("\n=== Phase 1-2: GMR Retarget + Isaac Lab dataset_retarget ===\n")
    reassemble_smplx()
    gmr_dir, venv_dir = setup_gmr()
    register_x1_in_gmr(gmr_dir)

    print("\n--- Auto-IK Calibration ---")
    run_auto_ik(gmr_dir, venv_dir)

    gmr_output = MOTIONS_DIR / "x1_gmr"
    if gmr_output.exists() and len(list(gmr_output.glob("*.pkl"))) >= 14:
        print(f"[INFO] x1_gmr already has {len(list(gmr_output.glob('*.pkl')))} files, skipping GMR retarget")
    else:
        run_gmr_retarget(gmr_dir, venv_dir)

    lab_output = MOTIONS_DIR / "x1_lab"
    if lab_output.exists() and len(list(lab_output.glob("*.pkl"))) >= 14:
        print(f"[INFO] x1_lab already has {len(list(lab_output.glob('*.pkl')))} files, skipping dataset_retarget")
    else:
        run_dataset_retarget(gmr_output)

    lab_files = list(lab_output.glob("*.pkl"))
    print(f"\n[INFO] x1_lab: {len(lab_files)} files")
    if len(lab_files) < 14:
        print("[ERROR] Expected 14 lab files for AMP training!")
        sys.exit(1)
    return gmr_output, lab_output, venv_dir


def phase_retarget_acceptance(venv_dir: Path) -> bool:
    """Strict gate: acceptance/check_retarget.py. Returns True on PASS."""
    print("\n=== Phase 3: STRICT Retarget Acceptance Gate ===\n")
    checker = REPO_ROOT / "acceptance" / "check_retarget.py"
    report_json = UPLOAD_DIR / "retarget_acceptance_report.json"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)  # checker json write needs it
    venv_python = str(venv_dir / "bin" / "python")
    cmd = [
        "python" if not Path(venv_python).exists() else venv_python,
        str(checker), "--repo-root", str(REPO_ROOT),
        "--json", str(report_json),
    ]
    print(f"[INFO] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))

    payload = {"phase": "retarget_acceptance",
               "passed": result.returncode == 0,
               "checker": "acceptance/check_retarget.py",
               "spec": "acceptance/RETARGET_ACCEPTANCE.md"}
    if report_json.exists():
        payload["report"] = report_json.read_text()
    wrap_json_for_upload("model_retarget_report.pt", payload)

    if result.returncode != 0:
        print("\n[FATAL] Retarget acceptance FAILED — blocking training start.")
        print("[INFO] Report wrapped for upload. Aborting (no GPU-hours spent).")
        wait_for_sdk(180, "upload failure report")
        sys.exit(1)
    print("\n[INFO] Retarget acceptance PASSED — proceeding to training.")
    return True


def phase_package_retarget(gmr_output: Path, lab_output: Path):
    print("\n=== Phase 3.5: Package retarget artifacts ===")
    retarget_pkg = {}
    for f in sorted(lab_output.glob("*.pkl")):
        retarget_pkg[f"x1_lab/{f.name}"] = f.read_bytes()
    for f in sorted(gmr_output.glob("*.pkl")):
        retarget_pkg[f"x1_gmr/{f.name}"] = f.read_bytes()
    auto_cfg = REPO_ROOT / "AMASS_minimal" / "smplx_to_x1_auto.json"
    if auto_cfg.exists():
        retarget_pkg["smplx_to_x1_auto.json"] = auto_cfg.read_bytes()
    p = UPLOAD_DIR / "model_retarget_data.pt"
    with open(p, "wb") as fh:
        _pkl.dump(retarget_pkg, fh)
    print(f"[UPLOAD] {p.relative_to(REPO_ROOT)} ({p.stat().st_size / 1e6:.1f}MB)")


def find_checkpoint_roots():
    """train.py writes logs relative to ITS cwd. We pass cwd=REPO_ROOT, but
    also sweep the process cwd root as belt-and-braces (v16/v17 bug: logs went
    to /workspace/isaaclab/logs while monitor watched REPO_ROOT/logs)."""
    roots = [REPO_ROOT / "logs" / "rsl_rl" / "x1_amp",
             Path.cwd() / "logs" / "rsl_rl" / "x1_amp"]
    return [r for r in roots if r.exists()]


def latest_run_dir():
    run_dirs = []
    for root in find_checkpoint_roots():
        for d in root.glob("*/"):
            if d.name == "exported_data":
                continue
            if list(d.glob("model_*.pt")):
                run_dirs.append(d)
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda d: d.stat().st_mtime)


def all_checkpoints():
    ckpts = {}
    for root in find_checkpoint_roots():
        for d in root.glob("*/"):
            if d.name == "exported_data":
                continue
            for c in d.glob("model_*.pt"):
                ckpts[c.name] = c  # later roots overwrite earlier
    return ckpts


def mirror_checkpoint(ckpt: Path, tag: str):
    """Mirror a checkpoint into every SDK-visible location."""
    copied = []
    # 1) repo tree upload dir (verifiably scanned during run)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dst = UPLOAD_DIR / ckpt.name
    if not dst.exists():
        shutil.copy2(ckpt, dst)
        copied.append(dst)
    # 2) exported_data pattern logs/{exp}/exported_data/{run}/model_*.pt
    for root in [REPO_ROOT / "logs", Path.cwd() / "logs"]:
        exp = root / "x1_amp" / "exported_data" / tag
        try:
            exp.mkdir(parents=True, exist_ok=True)
            dst2 = exp / ckpt.name
            if not dst2.exists():
                shutil.copy2(ckpt, dst2)
                copied.append(dst2)
        except OSError:
            pass
    for c in copied:
        try:
            print(f"[MONITOR] {ckpt.name} -> {c}")
        except Exception:
            pass


def phase_train() -> int:
    print("\n=== Phase 4: AMP Training ===\n")

    robolab_src = REPO_ROOT / "roboparty_train" / "robolab"
    rsl_rl_src = REPO_ROOT / "roboparty_train" / "rsl_rl"

    print("[INFO] Installing robolab/rsl_rl (force rsl_rl to override Isaac Lab v3.1.2)...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "rsl-rl-lib", "-y", "-q"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(rsl_rl_src), "-q",
                    "--force-reinstall", "--no-deps"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(robolab_src), "-q",
                    "--no-deps"], check=True)

    train_script = robolab_src / "scripts" / "rsl_rl" / "train.py"
    cmd = [sys.executable, str(train_script),
           "--task=X1-AMP", "--headless", "--logger=tensorboard", "--num_envs=4096"]

    tag = _dt.now().strftime("%Y-%m-%d_%H-%M-%S") + "x1_amp"
    stop_monitor = threading.Event()
    mirrored = set()

    def monitor():
        """Every 30s mirror milestone checkpoints (every 1000th) into the
        depth-2 repo-tree upload dir (proven SDK pattern) + exported_data."""
        while not stop_monitor.is_set():
            try:
                for name, c in sorted(all_checkpoints().items()):
                    if name in mirrored:
                        continue
                    stem = name[len("model_"):-len(".pt")] if name.startswith("model_") else ""
                    if not (stem.isdigit() and int(stem) % 1000 == 0):
                        continue  # milestones only; natural logs path covers the rest
                    mirror_checkpoint(c, tag)
                    mirrored.add(name)
            except Exception as e:
                print(f"[MONITOR] error: {e}")
            stop_monitor.wait(30)

    threading.Thread(target=monitor, daemon=True).start()

    print(f"[INFO] Starting AMP training: {' '.join(cmd)}")
    print(f"[INFO] cwd={REPO_ROOT}  stdout -> {TRAIN_LOG_FILE.name}")
    with open(TRAIN_LOG_FILE, "wb") as logf:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        for raw in iter(proc.stdout.readline, b""):
            logf.write(raw)
            logf.flush()
            try:
                print(raw.decode(errors="replace"), end="")
            except Exception:
                pass
        rc = proc.wait()
    stop_monitor.set()
    time.sleep(2)

    # Final sweep: mirror everything that exists (covers final + stragglers)
    print("\n[INFO] Final checkpoint sweep:")
    ckpts = all_checkpoints()
    if not ckpts:
        print("[ERROR] NO CHECKPOINTS FOUND anywhere under logs/ roots!")
    for name, c in sorted(ckpts.items()):
        if name not in mirrored:
            mirror_checkpoint(c, tag)
            mirrored.add(name)
    print(f"[INFO] Mirrored {len(mirrored)} checkpoints. Run dir: {latest_run_dir()}")
    return rc


def final_checkpoint() -> Path | None:
    ckpts = all_checkpoints()
    if not ckpts:
        return None
    def key(name):
        stem = name[len("model_"):-len(".pt")]
        return (0, int(stem)) if stem.isdigit() else (1, 0)
    best = max(ckpts, key=key)
    return ckpts[best]


def phase_play_video(ckpt: Path):
    """Record a fixed-command walk video with the final policy."""
    print("\n=== Phase 5: Play rollout + video ===\n")
    if ckpt is None:
        print("[WARN] No checkpoint — skipping play/video")
        return None
    play_script = REPO_ROOT / "roboparty_train" / "robolab" / "scripts" / "rsl_rl" / "play_amp.py"
    cmd = [sys.executable, str(play_script),
           "--task", "X1-AMP-Play",
           "--num_envs", "1",
           "--checkpoint", str(ckpt),
           "--video", "--video_length", "600",   # 600 steps = 12 s @ 50 Hz
           "--headless"]
    print(f"[INFO] {' '.join(cmd)}")
    try:
        with open(PLAY_LOG_FILE, "wb") as logf:
            rc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=logf,
                                stderr=subprocess.STDOUT, timeout=1500).returncode
    except subprocess.TimeoutExpired:
        rc = -1
        print("[WARN] play timed out after 1500s (killed) — continuing without fresh video")
    print(f"[INFO] play exit={rc} (log: {PLAY_LOG_FILE.name})")

    # locate produced mp4(s)
    videos = []
    for root in [REPO_ROOT / "logs", Path.cwd() / "logs"]:
        videos += list(root.rglob("*.mp4"))
    if not videos:
        print("[WARN] No mp4 found after play!")
        return None
    video = max(videos, key=lambda v: v.stat().st_mtime)
    # mirror to SDK video-scan locations: logs/{exp}/*.mp4 + upload dir
    dests = [REPO_ROOT / "logs" / "x1_amp" / f"x1_walk_{_dt.now():%H%M%S}.mp4",
             UPLOAD_DIR / f"x1_walk_{_dt.now():%H%M%S}.mp4"]
    for d in dests:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video, d)
        try:
            print(f"[VIDEO] -> {d}")
        except Exception:
            pass
    return video


def phase_amp_acceptance(video: Path | None):
    print("\n=== Phase 6: AMP Training Acceptance ===\n")
    checker = REPO_ROOT / "acceptance" / "check_amp.py"
    report_json = UPLOAD_DIR / "amp_acceptance_report.json"
    max_iter = 4000
    cmd = [sys.executable, str(checker), "--log", str(TRAIN_LOG_FILE),
           "--max-iters", str(max_iter), "--json", str(report_json)]
    if video is not None:
        cmd += ["--video", str(video)]
    if PLAY_LOG_FILE.exists():
        cmd += ["--play-log", str(PLAY_LOG_FILE)]
    rc = subprocess.run(cmd).returncode

    payload = {"phase": "amp_acceptance", "passed": rc == 0,
               "checker": "acceptance/check_amp.py",
               "spec": "acceptance/AMP_ACCEPTANCE.md", "max_iterations": max_iter}
    if report_json.exists():
        payload["report"] = report_json.read_text()
    wrap_json_for_upload("model_amp_report.pt", payload)
    return rc


def main():
    print("=" * 60)
    print("X1 AMP Pipeline v18 (acceptance-gated)")
    print("=" * 60)

    gmr_output, lab_output, venv_dir = phase_retarget()
    phase_retarget_acceptance(venv_dir)
    phase_package_retarget(gmr_output, lab_output)

    rc = phase_train()
    if rc != 0:
        print(f"[ERROR] AMP training exited with code {rc}")
    ckpt = final_checkpoint()
    print(f"[INFO] Final checkpoint: {ckpt}")

    video = phase_play_video(ckpt)
    amp_rc = phase_amp_acceptance(video)

    print("\n=== Phase 7: wrap-up ===")
    print(f"[INFO] Artifacts in {UPLOAD_DIR}:")
    for f in sorted(UPLOAD_DIR.glob("*")):
        print(f"  {f.name} ({f.stat().st_size // 1024}KB)")
    wait_for_sdk(300, "final checkpoint + video + reports")
    print("[INFO] Pipeline done.")


if __name__ == "__main__":
    main()
