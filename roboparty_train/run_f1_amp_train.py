#!/usr/bin/env python3
"""F1-AMP training pipeline v1 (acceptance-gated, pod-side).

F1 = X1 upper body + F1 lower body (orthogonal hip, perfect-mirror legs).
The retarget phase ran LOCALLY (retarget_f1_from_x1lab.py — leg IK re-solve
of x1_lab_v67 onto F1 kinematics; acceptance/check_f1_retarget.py PASS
0 fails/0 warns: knee corr 0.970, amplitude retention 1.003). This pod
pipeline therefore starts at training:

  1. Dataset gate: f1_lab_v67 present (20 clips) + F1 robot assets + env
     registration importable.
  2. AMP training: F1-AMP @ 50 Hz, 4096 envs, PYTHONPATH shadowing
     (rsl_rl first, then robolab — no pip; v27.3 lesson).
  3. Checkpoint monitor: mirrors every model_*.pt to the 5 SDK-visible
     locations (in-tree checkpoints/ PRIMARY per v29e3 evidence,
     model_upload/, repo-root flat, exported_data pattern, outside tree).
  4. Play rollout + video (F1-AMP-Play, 1.0 m/s forward, 12 s).

Env vars:
  F1_ITERS          fresh-run budget (default 4000, X1 standard)
  F1_RESUME_CKPT    path to .pt for fine-tune resume
  F1_FINE_TUNE_ITERS additional iterations on resume (default 1500)
  X1_* levers       inherit the X1 recipe knobs (control hz default 50)

Usage (startScript): gm-run X1_29_AMP/roboparty_train/run_f1_amp_train.py
"""

import functools
import os
import pickle
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

print = functools.partial(print, flush=True)

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = None
for candidate in [SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent.parent]:
    if (candidate / "roboparty_train").is_dir():
        REPO_ROOT = candidate
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

EXP = "f1_amp"
MOTIONS = REPO_ROOT / "roboparty_train" / "robolab" / "data" / "motions" / "f1_lab_v67"
ROBOT = REPO_ROOT / "roboparty_train" / "robolab" / "data" / "robots" / "f1"
UPLOAD_DIR = REPO_ROOT / "model_upload"
OUTSIDE_DIR = REPO_ROOT.parent / "f1_upload" if REPO_ROOT.parent.name == "isaaclab" else Path("/workspace/isaaclab/f1_upload")
TRAIN_LOG = REPO_ROOT / "train_stdout.log"
PLAY_LOG = REPO_ROOT / "play_stdout.log"
TAG = time.strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------- gates
def phase_dataset_gate() -> bool:
    print("\n=== Phase 1: dataset + assets gate ===")
    pkls = sorted(MOTIONS.glob("*.pkl"))
    ok = len(pkls) == 20
    print(f"[{'OK' if ok else 'FAIL'}] f1_lab_v67 clips: {len(pkls)}/20")
    for p in pkls[:3]:
        try:
            c = pickle.load(open(p, "rb"))
            shape_ok = (len(c["dof_pos"].shape) == 2 and c["dof_pos"].shape[1] == 29
                        and c["key_body_pos"].shape[1] == 10 and 30 <= c["fps"] <= 250)
            print(f"  {p.name}: fps={c['fps']} T={len(c['dof_pos'])} "
                  f"dof={c['dof_pos'].shape[1]} kb={c['key_body_pos'].shape[1]} "
                  f"[{'OK' if shape_ok else 'FAIL'}]")
            ok = ok and shape_ok
        except Exception as e:
            print(f"  {p.name}: LOAD FAIL {e}")
            ok = False
    urdf = ROBOT / "urdf" / "f1_29dof.urdf"
    print(f"[{'OK' if urdf.exists() else 'FAIL'}] F1 urdf: {urdf.exists()}")
    ok = ok and urdf.exists()
    n_mesh = len(list((ROBOT / "meshes").glob("*.STL"))) + len(list((ROBOT / "meshes").glob("*.stl")))
    print(f"[{'OK' if n_mesh >= 60 else 'FAIL'}] F1 meshes: {n_mesh}")
    ok = ok and n_mesh >= 60
    # env import check (cheap: file presence; real import needs isaac)
    for f in ["roboparty_train/robolab/robolab/assets/robots/f1.py",
              "roboparty_train/robolab/robolab/tasks/manager_based/amp/f1_amp_env_cfg.py",
              "roboparty_train/robolab/robolab/tasks/manager_based/amp/agents/f1_amp_agent_cfg.py"]:
        p = REPO_ROOT / f
        print(f"[{'OK' if p.exists() else 'FAIL'}] {Path(f).name}")
        ok = ok and p.exists()
    if not ok:
        print("[FATAL] dataset gate FAIL — aborting BEFORE burning GPU-hours")
    return ok


# ------------------------------------------------------------- checkpoints
def find_checkpoint_roots():
    roots = [REPO_ROOT / "logs" / "rsl_rl" / EXP,
             Path.cwd() / "logs" / "rsl_rl" / EXP]
    return [r for r in roots if r.exists()]


def all_checkpoints():
    ckpts = {}
    for root in find_checkpoint_roots():
        for d in root.glob("*/"):
            if d.name == "exported_data":
                continue
            for c in d.glob("model_*.pt"):
                ckpts[c.name] = c
    return ckpts


def wait_for_sdk(seconds: int, why: str):
    """Post-mirror wait: the SDK scans globally every ~16 min and the FINAL
    checkpoint appears right before task exit — v1d (TASK_20261010_133) lost
    model_4000..5498 to exactly this (detected 19:10:13, pod died before
    upload; the mid-run 091 uploads only succeeded because its run was long
    enough to span scan cycles)."""
    print(f"[WAIT] {why}: sleeping {seconds}s for the SDK upload cycle...")
    time.sleep(seconds)


def mirror_checkpoint(ckpt: Path, tag: str):
    """Five SDK-visible locations (v29e3 evidence: in-tree checkpoints/ is
    the PRIMARY; model_upload/ v23; repo-flat probe2; exported_data pattern
    skill-doc; outside-tree archive)."""
    copied = []
    for dst in [REPO_ROOT / "roboparty_train" / "checkpoints" / ckpt.name,
                UPLOAD_DIR / ckpt.name,
                REPO_ROOT / ckpt.name,
                REPO_ROOT / "logs" / "rsl_rl" / EXP / "exported_data" / tag / ckpt.name,
                OUTSIDE_DIR / tag / ckpt.name]:
        try:
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ckpt, dst)
                copied.append(dst)
        except OSError as e:
            print(f"[MONITOR] mirror failed {dst}: {e}")
    for c in copied:
        print(f"[MONITOR] {ckpt.name} -> {c}")


def final_checkpoint():
    cks = all_checkpoints()

    def num(p):
        s = p.stem[len("model_"):]
        return int(s) if s.isdigit() else -1
    if not cks:
        return None
    return max(cks.values(), key=lambda p: num(p))


# ----------------------------------------------------------------- train
def phase_train() -> int:
    print("\n=== Phase 2: AMP training (F1-AMP @ 50 Hz) ===")
    robolab_src = REPO_ROOT / "roboparty_train" / "robolab"
    rsl_rl_src = REPO_ROOT / "roboparty_train" / "rsl_rl"
    parts = [str(rsl_rl_src), str(robolab_src)]
    old = os.environ.get("PYTHONPATH", "")
    if old:
        parts.append(old)
    os.environ["PYTHONPATH"] = ":".join(parts)
    print(f"[INFO] PYTHONPATH shadowing (no pip): {os.environ['PYTHONPATH']}")

    train_script = robolab_src / "scripts" / "rsl_rl" / "train.py"
    cmd = [sys.executable, str(train_script),
           "--task=F1-AMP", "--headless", "--logger=tensorboard", "--num_envs=4096"]
    iters = os.environ.get("F1_ITERS", "")
    if iters and not os.environ.get("F1_RESUME_CKPT", "").strip():
        cmd += ["--max_iterations", iters]

    resume_ckpt = os.environ.get("F1_RESUME_CKPT", "").strip()
    if not resume_ckpt:
        def _num(p):
            s = p.stem[len("model_"):]
            return int(s) if s.isdigit() else -1
        cands = sorted((p for p in REPO_ROOT.glob("model_*.pt") if _num(p) >= 0), key=_num)
        if cands:
            resume_ckpt = str(cands[-1])
            print(f"[RESUME] auto-detected mounted checkpoint: {resume_ckpt}")
    if resume_ckpt:
        src = Path(resume_ckpt)
        if not src.is_file():
            raise FileNotFoundError(f"F1_RESUME_CKPT not found: {src}")
        resume_run = REPO_ROOT / "logs" / "rsl_rl" / EXP / "_resume_src"
        resume_run.mkdir(parents=True, exist_ok=True)
        dst = resume_run / src.name
        shutil.copy2(src, dst)
        # rsl_rl resume semantics (X1 launcher precedent, verified twice):
        # --resume is a bool; the checkpoint is located via --load_run +
        # --checkpoint; on resume max_iterations is ADDITIONAL.
        cmd += ["--resume", "--load_run", "_resume_src",
                "--checkpoint", src.name,
                "--max_iterations", os.environ.get("F1_FINE_TUNE_ITERS", "1500")]
        print(f"[RESUME] fine-tune from {src.name} +{os.environ.get('F1_FINE_TUNE_ITERS', '1500')}")

    stop_monitor = threading.Event()
    mirrored = set()

    def monitor():
        last = None
        while not stop_monitor.is_set():
            try:
                names = sorted(all_checkpoints())
                if names != last:
                    print(f"[MONITOR] checkpoints on disk: {names}")
                    last = names
                for nm, ck in sorted(all_checkpoints().items()):
                    if nm.startswith("model_") and nm not in mirrored:
                        print(f"[MONITOR] new checkpoint mirror: {nm}")
                        mirror_checkpoint(ck, TAG)
                        mirrored.add(nm)
            except Exception as e:
                print(f"[MONITOR] error: {e}")
            stop_monitor.wait(60)

    threading.Thread(target=monitor, daemon=True).start()

    print(f"[INFO] {' '.join(cmd)}")
    print(f"[INFO] cwd={REPO_ROOT}  stdout -> {TRAIN_LOG.name}")
    with open(TRAIN_LOG, "w", encoding="utf-8", errors="replace") as logf:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode(errors="replace").rstrip()
            print(f"[TRAIN] {line}", file=logf)
            logf.flush()
            # echo reward lines for live platform-log visibility
            if "mean_reward" in line or "iteration" in line.lower()[:20]:
                print(f"[TRAIN] {line[-160:]}")
        rc = proc.wait()
    stop_monitor.set()
    print(f"[INFO] training exit={rc} (full log: {TRAIN_LOG.name})")
    return rc


# ------------------------------------------------------------------ play
def phase_play_video(ckpt):
    print("\n=== Phase 3: Play rollout + video ===")
    if ckpt is None:
        print("[WARN] No checkpoint — skipping play/video")
        return None
    play_script = REPO_ROOT / "roboparty_train" / "robolab" / "scripts" / "rsl_rl" / "play_amp.py"
    cmd = [sys.executable, str(play_script),
           "--task", "F1-AMP-Play",
           "--num_envs", "1",
           "--checkpoint", str(ckpt),
           "--video", "--video_length", "600",   # 600 steps = 12 s @ 50 Hz
           "--headless"]
    print(f"[INFO] {' '.join(cmd)}")
    try:
        with open(PLAY_LOG, "wb") as logf:
            rc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=logf,
                                stderr=subprocess.STDOUT, timeout=1500).returncode
    except subprocess.TimeoutExpired:
        rc = -1
        print("[WARN] play timed out after 1500s — continuing")
    print(f"[INFO] play exit={rc} (log: {PLAY_LOG.name})")
    videos = []
    for root in [REPO_ROOT / "logs", Path.cwd() / "logs", REPO_ROOT,
                 ckpt.parent / "videos", OUTSIDE_DIR]:
        try:
            if root.exists():
                videos += [v for v in root.rglob("*.mp4") if "acceptance" not in v.parts]
        except OSError:
            pass
    print(f"[INFO] mp4 search: {len(videos)} found")
    video = max(videos, key=lambda v: v.stat().st_mtime) if videos else None
    if video:
        try:
            shutil.copy2(video, OUTSIDE_DIR / TAG / video.name)
        except OSError:
            pass
        print(f"[INFO] latest video: {video}")
    else:
        print("[WARN] No mp4 from Isaac play")
    return video


def main():
    t0 = time.time()
    # optional per-run env overrides committed as roboparty_train/f1_env.txt
    # (KEY=VALUE lines) — startScript cannot pass env vars to gm-run.
    env_file = SCRIPT_DIR / "f1_env.txt"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
        print(f"[ENV] f1_env.txt overrides: "
              f"{[l.split('=')[0] for l in env_file.read_text().splitlines() if '=' in l and not l.strip().startswith('#')]}")
    if not phase_dataset_gate():
        sys.exit(2)
    rc = phase_train()
    ck = final_checkpoint()
    print(f"[INFO] final checkpoint: {ck}")
    video = phase_play_video(ck)
    # final mirror pass (belt and braces)
    if ck is not None:
        mirror_checkpoint(ck, TAG)
    # v1d lesson: wait for the SDK upload cycle so the final model is
    # registered before the pod dies (16-min scan period; 1100s covers one
    # full cycle plus upload time)
    if ck is not None:
        wait_for_sdk(1100, "final checkpoint registration")
    print(f"\n=== DONE in {(time.time()-t0)/60:.1f} min ===")
    print(f"checkpoint={ck}")
    print(f"video={video}")
    print(f"train_exit={rc}")
    sys.exit(rc if rc != 0 else 0)


if __name__ == "__main__":
    main()
